import itertools
import operator
from dataclasses import dataclass
from functools import reduce
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from xformers.ops.fmha import memory_efficient_attention
from xformers.ops.fmha.attn_bias import (
    AttentionBias,
    BlockDiagonalCausalFromBottomRightMask,
    BlockDiagonalCausalMask
)


@dataclass
class AcousticModelConfig:
    n_semantic_types: int
    n_acoustic_types: int
    semantic_freq: float = 50.0  # Hz
    acoustic_freq: float = 40.0  # Hz
    local_advance: float = 0.05  # seconds
    dim: int = 768
    n_layers: int = 12
    head_dim: int = 64
    hidden_dim: int = 3072
    n_heads: int = 12
    n_kv_heads: int = 12
    dropout: float = 0.1
    norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    force_fp32_attention: bool = True


class AcousticModel(nn.Module):
    def __init__(self, cfg: AcousticModelConfig):
        super().__init__()
        self.cfg = cfg
        self._freqs_cis = None  # set lazily
        self.tok_embeddings = torch.nn.Embedding(self.input_vocab_size, cfg.dim)
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.output = torch.nn.Linear(cfg.dim, self.output_vocab_size, bias=False)

    @property
    def input_vocab_size(self) -> int:
        return (
            self.cfg.n_acoustic_types + 1 + 1 + 1 + self.cfg.n_semantic_types
        )  # EOU EOS BOS

    @property
    def output_vocab_size(self) -> int:
        return self.cfg.n_acoustic_types + 1 + 1  # EOU EOS

    @property
    def EOU(self) -> int:
        return self.cfg.n_acoustic_types

    @property
    def EOS(self) -> int:
        return self.cfg.n_acoustic_types + 1

    @property
    def BOS(self) -> int:
        return self.cfg.n_acoustic_types + 2

    @property
    def semantic_offset(self) -> int:
        return self.cfg.n_acoustic_types + 3

    def tokenize_infer(self, semantic_units: torch.Tensor):
        tokens = torch.cat(
            [
                semantic_units.long() + self.semantic_offset,
                torch.tensor(
                    [self.BOS], dtype=torch.long, device=semantic_units.device
                ),
            ]
        )
        return tokens

    def tokenize_train(
        self, segments: torch.Tensor, acoustic_units: torch.Tensor
    ) -> torch.Tensor:
        """
        segments is a Long Tensor of shape [n_segments, 3].
        Each item contains [segment_start_time_in_frames, segment_end_time_in_frames, segment_id].
        If we divide the start and end times (in frames) by self.cfg.semantic_freq, we get timestamps in seconds
        For example:
            [[0, 2, 1075],
            [2, 4, 7492],
            [4, 8, 7492],
            [8, 10, 4925]]

        acoustic_units_duped is a Long Tensor of shape [T,]
        It contains the framewise acoustic units at a frequence of self.cfg.acoustic_freq
        """
        dtype = segments.dtype
        device = segments.device
        assert dtype == torch.long

        EOU = torch.tensor(self.EOU, dtype=dtype, device=device)
        EOS = torch.tensor(self.EOS, dtype=dtype, device=device)
        BOS = torch.tensor(self.BOS, dtype=dtype, device=device)

        global_advance = torch.stack(
            [seg[2] + self.semantic_offset for seg in segments]
        )

        semantic_data = (
            (
                seg[0] / self.cfg.semantic_freq - self.cfg.local_advance,
                seg[2] + self.semantic_offset,
                1,
            )  # (timestamp, token, priority)
            for seg in segments
        )
        EOU_data = (
            (
                seg[1] / self.cfg.semantic_freq - self.cfg.local_advance,
                EOU,
                0,
            )  # (timestamp, token, priority)
            for seg in segments
        )
        acoustic_data = (
            (i / self.cfg.acoustic_freq, unit, 2)  # (timestamp, token, priority)
            for i, unit in enumerate(acoustic_units)
        )
        all_data = sorted(
            itertools.chain(semantic_data, EOU_data, acoustic_data),
            key=lambda x: (x[0], x[2]),  # timestamp then priority
        )

        interleaved = torch.stack([item[1] for item in all_data])

        tokens = torch.cat(
            [global_advance, BOS.unsqueeze(0), interleaved, EOS.unsqueeze(0)]
        )

        promptlen = len(global_advance)

        return tokens, promptlen

    def decode(self, tokens: torch.Tensor):
        acoustic_tokens = tokens[tokens < self.cfg.n_acoustic_types]
        return acoustic_tokens

    def pretty_print_encoding(self, tokens: torch.Tensor):
        str_values = []
        for token in tokens.tolist():
            if token < self.EOU:
                str_values.append(f"a{token}")
            elif token == self.EOU:
                str_values.append("[EOU]")
            elif token == self.EOS:
                str_values.append("[EOS]")
            elif token == self.BOS:
                str_values.append("[BOS]")
            elif token > self.BOS:
                str_values.append(f"s{token - self.semantic_offset}")
        return str_values

    @property
    def freqs_cis(self):
        # lazy init
        if self._freqs_cis is None:
            self._freqs_cis = precompute_freqs_cis(
                self.cfg.head_dim,
                8_000,
                theta=self.cfg.rope_theta,
                device=self.tok_embeddings.weight.device,
            )
        return self._freqs_cis

    def forward(
        self, input_ids: torch.Tensor, promptlens: List[int], seqlens: List[int]
    ) -> torch.Tensor:
        assert sum(seqlens) == input_ids.shape[0]

        h = self.tok_embeddings(input_ids)
        positions = positions_from_sizes(seqlens, self.freqs_cis.device)
        q_seqlen = [sl - pl for pl, sl in zip(promptlens, seqlens)]
        att_mask = BlockDiagonalCausalFromBottomRightMask.from_seqlens(
            q_seqlen, kv_seqlen=seqlens
        )
        #att_mask = BlockDiagonalCausalMask.from_seqlens(seqlens)

        freqs_cis = self.freqs_cis[positions].to(device=h.device)

        for layer in self.layers:
            h = layer(h, freqs_cis, att_mask)

        return self.output(self.norm(h)).float()

    @classmethod
    def from_pretrained_checkpoint(cls, checkpoint_path: str) -> "AcousticModel":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg = AcousticModelConfig(**checkpoint["cfg"])
        model = cls(cfg)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        print(f"AcousticModel loaded with {params:,} parameters")
        return model


class TransformerBlock(nn.Module):
    def __init__(self, args: AcousticModelConfig):
        super().__init__()

        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        att_mask: AttentionBias,
    ) -> torch.Tensor:
        r = self.attention(self.attention_norm(x), freqs_cis, att_mask)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        return h + r


class Attention(nn.Module):
    def __init__(self, args: AcousticModelConfig):
        super().__init__()
        self.args = args
        self.repeats = args.n_heads // args.n_kv_heads

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

        self.dropout = nn.Dropout(args.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: AttentionBias,
    ) -> torch.Tensor:
        seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(seqlen_sum, self.args.n_heads, self.args.head_dim)
        xk = xk.view(seqlen_sum, self.args.n_kv_heads, self.args.head_dim)
        xv = xv.view(seqlen_sum, self.args.n_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        key, val = xk, xv

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]

        if self.args.force_fp32_attention:
            with torch.amp.autocast(device_type='cuda', enabled=False):
                xq_fp32 = xq.float()
                key_fp32 = key.float()
                val_fp32 = val.float()
                output = memory_efficient_attention(xq_fp32, key_fp32, val_fp32, mask)
                output = output.to(x.dtype)  # Convert back to original dtype
        else:
            output = memory_efficient_attention(xq, key, val, mask)

        return self.dropout(self.wo(output.view(seqlen_sum, -1)))


class FeedForward(nn.Module):
    def __init__(self, args: AcousticModelConfig):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x) -> torch.Tensor:
        return self.dropout(self.w2(nn.functional.silu(self.w1(x)) * self.w3(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(
    dim: int, end: int, theta: float, device: Optional[torch.device] = None
) -> torch.Tensor:
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def positions_from_sizes(sizes: Iterable[int], device):
    return torch.tensor(
        reduce(operator.iadd, [list(range(s)) for s in sizes], []),
        dtype=torch.long,
        device=device,
    )
