import operator
from dataclasses import dataclass
from functools import reduce
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from xformers.ops.fmha import memory_efficient_attention
from xformers.ops.fmha.attn_bias import AttentionBias, BlockDiagonalCausalMask


@dataclass
class ULMConfig:
    dim: int = 1024
    n_layers: int = 12
    head_dim: int = 64
    hidden_dim: int = 4 * 1024
    n_heads: int = 16
    n_kv_heads: int = 16
    dropout: float = 0.2
    norm_eps: float = 1e-6
    vocab_size: int = 10_000
    rope_theta: float = 10_000.0


class ULM(nn.Module):
    def __init__(self, cfg: ULMConfig):
        super().__init__()
        self.cfg = cfg
        self._freqs_cis = None  # set lazily
        self.tok_embeddings = torch.nn.Embedding(cfg.vocab_size + 1, cfg.dim)
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.output = torch.nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    @property
    def BOS(self) -> int:
        return self.cfg.vocab_size

    def tokenize(self, units: torch.Tensor):
        tokens = torch.cat(
            [
                torch.tensor([self.BOS], dtype=torch.long, device=units.device),
                units.long(),
            ]
        )
        return tokens

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
        att_mask = BlockDiagonalCausalMask.from_seqlens(seqlens)

        freqs_cis = self.freqs_cis[positions].to(device=h.device)

        for layer in self.layers:
            h = layer(h, freqs_cis, att_mask)

        return self.output(self.norm(h)).float()

    @torch.inference_mode()
    def loglikelihood(self, ids: torch.Tensor) -> float:
        assert ids.ndim == 1, "ids must be a 1D tensor"
        device = ids.device
        input_ids = ids[:-1].clone().to(device)
        target_ids = ids[1:].clone().to(device)
        logits = self.forward(input_ids, [len(input_ids)])
        neg_obs_log_probs = torch.nn.functional.cross_entropy(
            logits,
            target_ids,
            reduction="none",
        )
        ll = -neg_obs_log_probs.sum().item()
        return ll

    def save_pretrained_checkpoint(
        self,
        checkpoint_path: str,
    ):
        state_dict = self.state_dict()
        torch.save(
            {
                "model_args": self.cfg.to_dict(),
                "model_state_dict": state_dict,
            },
            checkpoint_path,
        )

    @classmethod
    def from_pretrained_checkpoint(cls, checkpoint_path: str) -> "ULM":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_args = ULMConfig(**checkpoint["model_args"])
        model = cls(args=model_args)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        print(f"ULM loaded with {params:,} parameters")
        return model


class TransformerBlock(nn.Module):
    def __init__(self, args: ULMConfig):
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
    def __init__(self, args: ULMConfig):
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
        output = memory_efficient_attention(xq, key, val, mask)

        return self.dropout(self.wo(output.view(seqlen_sum, -1)))


class FeedForward(nn.Module):
    def __init__(self, args: ULMConfig):
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
