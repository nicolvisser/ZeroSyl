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
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
)

from .utils.misc import is_contiguous


@dataclass
class AcousticModelConfig:
    n_semantic_types: int
    n_acoustic_types: int
    semantic_freq: float = 50.0  # Hz
    acoustic_freq: float = 40.0  # Hz
    acoustic_lag: float = 0.05  # seconds
    dim: int = 768
    n_layers: int = 12
    head_dim: int = 64
    hidden_dim: int = 3072
    n_heads: int = 12
    n_kv_heads: int = 12
    dropout: float = 0.1
    norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    force_fp32_attention: bool = False


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
            self.cfg.n_acoustic_types + 4 + self.cfg.n_semantic_types
        )  # LAG EOU EOS BOS

    @property
    def output_vocab_size(self) -> int:
        return self.cfg.n_acoustic_types + 3  # LAG EOU EOS

    @property
    def LAG(self) -> int:
        return self.cfg.n_acoustic_types

    @property
    def EOU(self) -> int:
        return self.cfg.n_acoustic_types + 1

    @property
    def EOS(self) -> int:
        return self.cfg.n_acoustic_types + 2

    @property
    def BOS(self) -> int:
        return self.cfg.n_acoustic_types + 3

    @property
    def semantic_offset(self) -> int:
        return self.cfg.n_acoustic_types + 4

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

        LAG = torch.tensor(self.LAG, dtype=dtype, device=device)
        EOU = torch.tensor(self.EOU, dtype=dtype, device=device)
        EOS = torch.tensor(self.EOS, dtype=dtype, device=device)
        BOS = torch.tensor(self.BOS, dtype=dtype, device=device)

        global_advance = torch.stack(
            [seg[2] + self.semantic_offset for seg in segments]
        )

        semantic_data = (
            (
                seg[0] / self.cfg.semantic_freq,
                seg[2] + self.semantic_offset,
                1,
            )  # (timestamp, token, priority)
            for seg in segments
        )

        if is_contiguous(segments):
            # use the end times to determine locations of EOU tokens
            EOU_data = (
                (
                    seg[1] / self.cfg.semantic_freq,
                    EOU,
                    0,
                )  # (timestamp, token, priority)
                for seg in segments
            )
        else:
            # the segmentation is gappy - like in Sylber https://arxiv.org/pdf/2410.07168
            # use the next-token start times to determine locations of EOU tokens
            override_end_times = torch.zeros(len(segments), dtype=torch.float32)
            override_end_times[:-1] = (
                segments[1:, 0] / self.cfg.semantic_freq
            )  # next-token start times
            override_end_times[-1] = (
                segments[-1, 1] / self.cfg.semantic_freq
            )  # last token end time
            EOU_data = ((t, EOU, 0) for t in override_end_times)

        # Add a lag to the acoustic data.
        # By allowing a lag from the input to output it gives the model a chance
        # to "peek" into the future before making predictions about the acoustics.
        # This is similar to the 'local advance' strategy of ELLA-V
        # but with an explicit [LAG] token that the model must predict.
        assert (
            self.cfg.acoustic_lag * self.cfg.acoustic_freq % 1 == 0
        ), "you should use a lag that is a multiple of 1/acoustic_freq"
        acoustic_units = torch.cat(
            [
                torch.repeat_interleave(
                    LAG, int(self.cfg.acoustic_lag * self.cfg.acoustic_freq)
                ),
                acoustic_units,
            ]
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

        return tokens

    def decode(self, tokens: torch.Tensor):
        acoustic_tokens = tokens[tokens < self.cfg.n_acoustic_types]
        return acoustic_tokens

    def pretty_print_encoding(self, tokens: torch.Tensor):
        str_values = []
        for token in tokens.tolist():
            if token < self.EOU:
                str_values.append(f"a{token}")
            elif token == self.LAG:
                str_values.append("[LAG]")
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

    def forward(self, input_ids: torch.Tensor, seqlens: List[int]) -> torch.Tensor:
        assert sum(seqlens) == input_ids.shape[0]

        h = self.tok_embeddings(input_ids)
        positions = positions_from_sizes(seqlens, self.freqs_cis.device)
        freqs_cis = self.freqs_cis[positions].to(device=h.device)
        att_mask = BlockDiagonalCausalMask.from_seqlens(seqlens)

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
            with torch.amp.autocast(device_type="cuda", enabled=False):
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


# =============== Cache ===============


@dataclass
class CacheInputMetadata:
    # rope absolute positions
    positions: torch.Tensor
    # where tokens should go in the cache
    cache_positions: torch.Tensor

    # if prefill, use block diagonal causal mask
    # else use causal with padded key mask
    prefill: bool
    mask: AttentionBias
    seqlens: List[int]


def interleave_list(
    l1: List[torch.Tensor], l2: List[torch.Tensor]
) -> List[torch.Tensor]:
    assert len(l1) == len(l2)
    return [v for pair in zip(l1, l2) for v in pair]


class CacheView:
    def __init__(
        self,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        metadata: CacheInputMetadata,
        kv_seqlens: torch.Tensor,
    ):
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.kv_seqlens = kv_seqlens
        self.metadata = metadata

    def update(self, xk: torch.Tensor, xv: torch.Tensor) -> None:
        """
        to_cache_mask masks the last [max_seq_len] tokens in each sequence
        """
        n_kv_heads, head_dim = self.cache_k.shape[-2:]
        flat_cache_k = self.cache_k.view(-1, n_kv_heads, head_dim)
        flat_cache_v = self.cache_v.view(-1, n_kv_heads, head_dim)

        flat_cache_k.index_copy_(0, self.metadata.cache_positions, xk)
        flat_cache_v.index_copy_(0, self.metadata.cache_positions, xv)

    def interleave_kv(
        self, xk: torch.Tensor, xv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is a naive implementation and not optimized for speed.
        """
        assert xk.ndim == xv.ndim == 3  # (B * T, H, D)
        assert xk.shape == xv.shape

        if all([s == 0 for s in self.metadata.seqlens]):
            # No cache to interleave
            return xk, xv

        # Make it a list of [(T, H, D)]
        xk: Tuple[torch.Tensor] = torch.split(xk, self.metadata.seqlens)  # type: ignore
        xv: Tuple[torch.Tensor] = torch.split(xv, self.metadata.seqlens)  # type: ignore
        assert len(xk) == len(
            self.kv_seqlens
        ), f"Batch size is {len(self.kv_seqlens)}, got {len(xk)}"

        # Retrieve cache
        cache_k = [
            cache_k[:seq_len] for cache_k, seq_len in zip(self.cache_k, self.kv_seqlens)
        ]
        cache_v = [
            cache_v[:seq_len] for cache_v, seq_len in zip(self.cache_v, self.kv_seqlens)
        ]

        interleaved_k = interleave_list(cache_k, list(xk))
        interleaved_v = interleave_list(cache_v, list(xv))

        return torch.cat(interleaved_k, dim=0), torch.cat(interleaved_v, dim=0)

    @property
    def max_seq_len(self) -> int:
        return self.cache_k.shape[1]

    @property
    def key(self) -> torch.Tensor:
        return self.cache_k[: len(self.kv_seqlens)]

    @property
    def value(self) -> torch.Tensor:
        return self.cache_v[: len(self.kv_seqlens)]

    @property
    def prefill(self) -> bool:
        return self.metadata.prefill

    @property
    def mask(self) -> AttentionBias:
        return self.metadata.mask


class BufferCache:
    """
    This is an example that implements a buffer cache, allowing for variable length sequences.
    Allocated cache is rectangular which is wasteful (see PagedAttention for better mechanisms)
    """

    def __init__(
        self,
        n_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.cache_k = torch.empty(
            (n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim)
        )
        self.cache_v = torch.empty(
            (n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim)
        )
        # holds the valid length for each batch element in the cache
        self.kv_seqlens: Optional[torch.Tensor] = None

    def get_view(self, layer_id: int, metadata: CacheInputMetadata) -> CacheView:
        assert self.kv_seqlens is not None
        return CacheView(
            self.cache_k[layer_id], self.cache_v[layer_id], metadata, self.kv_seqlens
        )

    def reset(self) -> None:
        self.kv_seqlens = None

    def init_kvseqlens(self, batch_size: int) -> None:
        self.kv_seqlens = torch.zeros(
            (batch_size,), device=self.device, dtype=torch.long
        )

    @property
    def device(self) -> torch.device:
        return self.cache_k.device

    def to(self, device: torch.device, dtype: torch.dtype) -> "BufferCache":
        self.cache_k = self.cache_k.to(device=device, dtype=dtype)
        self.cache_v = self.cache_v.to(device=device, dtype=dtype)

        return self

    def update_seqlens(self, seqlens: List[int]) -> None:
        assert self.kv_seqlens is not None
        self.kv_seqlens += torch.tensor(seqlens, device=self.device, dtype=torch.long)

    def get_input_metadata(self, seqlens: List[int]) -> CacheInputMetadata:
        """
        Get metadata about cache positions
        """
        if self.kv_seqlens is None:
            self.init_kvseqlens(len(seqlens))

        assert isinstance(self.kv_seqlens, torch.Tensor)
        assert len(seqlens) == len(
            self.kv_seqlens
        ), f"Batch size is {len(self.kv_seqlens)}, got {len(seqlens)}, did you forget to reset cache?"
        seqpos = self.kv_seqlens.tolist()

        assert len(seqlens) > 0, seqlens
        cached_elements = torch.tensor(seqlens, device=self.device, dtype=torch.long)

        positions = torch.cat(
            [torch.arange(pos, pos + seqlen) for pos, seqlen in zip(seqpos, seqlens)]
        ).to(device=self.device, dtype=torch.long)

        batch_idx = torch.tensor(
            sum([[i] * seqlen for i, seqlen in enumerate(seqlens)], []),
            device=self.device,
            dtype=torch.long,
        )
        cache_positions = positions + batch_idx * self.max_seq_len

        first_prefill = seqpos[0] == 0
        subsequent_prefill = any(seqlen > 1 for seqlen in seqlens)
        if first_prefill:
            assert all([pos == 0 for pos in seqpos]), seqpos
            mask = BlockDiagonalCausalMask.from_seqlens(seqlens).make_local_attention(
                self.max_seq_len
            )
        elif subsequent_prefill:
            mask = BlockDiagonalMask.from_seqlens(
                q_seqlen=seqlens,
                kv_seqlen=[
                    s + cached_s.clamp(max=self.max_seq_len).item()
                    for (s, cached_s) in zip(seqlens, self.kv_seqlens)
                ],
            ).make_local_attention_from_bottomright(self.max_seq_len)
        else:
            mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=seqlens,
                kv_padding=self.max_seq_len,
                kv_seqlen=(self.kv_seqlens + cached_elements)
                .clamp(max=self.max_seq_len)
                .tolist(),
            )

        return CacheInputMetadata(
            positions=positions,
            cache_positions=cache_positions,
            prefill=first_prefill or subsequent_prefill,
            mask=mask,
            seqlens=seqlens,
        )
