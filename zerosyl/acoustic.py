import itertools
import operator
from dataclasses import dataclass
from functools import reduce
from typing import Iterable

import torch
import torch.nn as nn
from xformers.ops.fmha import memory_efficient_attention
from xformers.ops.fmha.attn_bias import AttentionBias, BlockDiagonalCausalMask

from .cache import BufferCache, CacheView
from .utils.misc import is_contiguous


@dataclass
class AcousticModelConfig:
    n_semantic_types: int
    n_acoustic_types: int
    semantic_freq: float  # Hz
    acoustic_freq: float  # Hz
    acoustic_lag: float = 0.0  # seconds
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

    def generate(
        self,
        semantic_units: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 0.85,
        max_tokens_per_semantic_unit: int = 20,
        max_tokens: int = 2000,
    ):
        msg = "semntic_units must be a 1D torch.Tensor"
        assert isinstance(semantic_units, torch.Tensor), msg
        assert semantic_units.ndim == 1, msg

        semantic_tokens = semantic_units + self.semantic_offset
        encoded_prompt = torch.cat(
            [
                semantic_tokens,
                torch.tensor(
                    [self.BOS],
                    dtype=torch.long,
                    device="cuda",
                ),
            ]
        )

        cache_window = len(encoded_prompt) + max_tokens
        cache = BufferCache(
            n_layers=self.cfg.n_layers,
            max_batch_size=1,
            max_seq_len=cache_window,
            n_kv_heads=self.cfg.n_kv_heads,
            head_dim=self.cfg.head_dim,
        )
        cache.to(device="cuda", dtype=torch.float32)
        cache.reset()

        with torch.inference_mode():
            prelogits: torch.Tensor = self(
                encoded_prompt,
                seqlens=[len(encoded_prompt)],
                cache=cache,
            )

        logits = prelogits[-1:, :]

        logits
        del prelogits

        force_semantic_next = True
        cur_semantic_idx = 0
        max_semantic_idx = len(semantic_tokens) - 1
        num_tokens_since_semantic = 0
        generated_tokens = []

        for _ in range(max_tokens):

            sampled_token = sample(logits, temperature, top_p)

            if force_semantic_next:
                sampled_token = semantic_tokens[cur_semantic_idx].unsqueeze(0)
                cur_semantic_idx += 1
                num_tokens_since_semantic = -1

            generated_tokens.append(sampled_token.squeeze().item())
            num_tokens_since_semantic += 1

            if generated_tokens[-1] == self.EOS:
                break

            force_semantic_next = (
                (generated_tokens[-1] == self.EOU)
                or (num_tokens_since_semantic >= max_tokens_per_semantic_unit)
            ) and (cur_semantic_idx <= max_semantic_idx)

            with torch.inference_mode():
                logits = self(sampled_token, seqlens=[1], cache=cache)

        acoustic_tokens = [t for t in generated_tokens if t < self.cfg.n_acoustic_types]
        acoustic_tokens = torch.tensor(acoustic_tokens, dtype=torch.long, device="cuda")

        return acoustic_tokens

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

        if not is_contiguous(segments):
            # The segmentation is gappy.
            # We will override end times with the next-unit start times.
            segments[:-1, 1] = segments[1:, 0]
            # We need to do this because in Sylber https://arxiv.org/pdf/2410.07168
            # there are short gaps between syllables.
            # If we don't squeeze the acoustic tokens from these gaps within a semantic and EOU token,
            # then the ELLA-V model does not predict acoustic tokens for those gaps.
            # This means the starts and ends of syllable with not synthesize correctly.
            # For long silences, Sylber still gives explicit segments with a silence ID,
            # so the gaps will usually be very small and timing won't be affected much.

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
        # to "peek" into the future before making predictions for the acoustics.
        # This is similar to the 'local advance' strategy in the ELLA-V paper
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

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        acoustic_tokens = tokens[tokens < self.cfg.n_acoustic_types]
        return acoustic_tokens

    def pretty_print_encoding(self, tokens: torch.Tensor) -> list[str]:
        str_values = []
        for token in tokens.tolist():
            if token < self.LAG:
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

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: int,
        cache: BufferCache | None = None,
    ) -> torch.Tensor:

        if cache is not None:
            input_metadata = cache.get_input_metadata(seqlens)
            att_mask = None
            positions = input_metadata.positions
        else:
            assert sum(seqlens) == input_ids.shape[0]
            att_mask = BlockDiagonalCausalMask.from_seqlens(seqlens)
            positions = positions_from_sizes(seqlens, self.freqs_cis.device)

        h = self.tok_embeddings(input_ids)

        freqs_cis = self.freqs_cis[positions].to(device=h.device)

        for layer_id, layer in enumerate(self.layers):
            if cache is not None:
                cache_view = cache.get_view(layer_id, input_metadata)
            else:
                cache_view = None
            h = layer(h, freqs_cis, att_mask, cache_view)

        if cache is not None:
            cache.update_seqlens(seqlens)

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

    @classmethod
    def from_remote(
        cls,
        url: str = "https://storage.googleapis.com/zerospeech-checkpoints/ELLA-V-ZeroSylCollapsed-v040-k-9116-neucodec-LibriSpeech-train-clean-460.pt",
    ) -> "AcousticModel":
        checkpoint = torch.hub.load_state_dict_from_url(url)
        cfg = AcousticModelConfig(**checkpoint["cfg"])
        model = cls(cfg)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
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
        att_mask: AttentionBias | None = None,
        cache: CacheView | None = None,
    ) -> torch.Tensor:
        r = self.attention(self.attention_norm(x), freqs_cis, att_mask, cache)
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
        mask: AttentionBias | None = None,
        cache: CacheView | None = None,
    ) -> torch.Tensor:
        seqlen_sum, _ = x.shape
        assert (mask is not None) ^ (
            cache is not None
        ), "exactly one of mask and cache should be provided"

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(seqlen_sum, self.args.n_heads, self.args.head_dim)
        xk = xk.view(seqlen_sum, self.args.n_kv_heads, self.args.head_dim)
        xv = xv.view(seqlen_sum, self.args.n_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if cache is not None:
            if cache.prefill:
                key, val = cache.interleave_kv(xk, xv)
                cache.update(xk, xv)
            else:
                cache.update(xk, xv)
                key, val = cache.key, cache.value
                key = key.view(
                    seqlen_sum * cache.max_seq_len,
                    self.args.n_kv_heads,
                    self.args.head_dim,
                )
                val = val.view(
                    seqlen_sum * cache.max_seq_len,
                    self.args.n_kv_heads,
                    self.args.head_dim,
                )
        if mask is not None:
            key, val = xk, xv

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]

        # xformers requires (B=1, S, H, D)
        output = memory_efficient_attention(
            xq, key, val, mask if cache is None else cache.mask
        )

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

    def _norm(self, x) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(
    dim: int, end: int, theta: float, device: torch.device | None = None
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
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(
    keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def positions_from_sizes(sizes: Iterable[int], device) -> torch.Tensor:
    return torch.tensor(
        reduce(operator.iadd, [list(range(s)) for s in sizes], []),
        dtype=torch.long,
        device=device,
    )


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


def sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)
