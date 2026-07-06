"""A small Llama-style decoder-only Transformer.

Architecture (modern nano-GPT):
  * RMSNorm (pre-norm)
  * Rotary position embeddings (RoPE)
  * Grouped-Query Attention (GQA) via F.scaled_dot_product_attention (Flash)
  * SwiGLU feed-forward
  * Weight-tied token embedding / LM head

Config-driven; the default `GPTConfig` is ~120M params. Instantiate and call
`model.num_params()` to see the exact count for any config.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 16384
    block_size: int = 1024
    n_layer: int = 16
    n_head: int = 12
    n_kv_head: int = 4          # GQA: n_head must be divisible by n_kv_head
    d_model: int = 768
    ffn_multiple_of: int = 256  # round SwiGLU hidden dim to this
    rope_theta: float = 10000.0
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head

    def __post_init__(self):
        assert self.n_head % self.n_kv_head == 0, "n_head must divide by n_kv_head"


# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf.type_as(x)) * self.weight


def _rope_tables(head_dim: int, seq_len: int, theta: float):
    """Return cos, sin of shape (1, 1, seq_len, head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)                 # (seq_len, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)          # (seq_len, head_dim)
    return emb.cos()[None, None], emb.sin()[None, None]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x, cos, sin):
    # x: (B, n_head, T, head_dim); cos/sin: (1, 1, T, head_dim)
    return x * cos + _rotate_half(x) * sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv = cfg.n_kv_head
        self.hd = cfg.head_dim
        self.dropout = cfg.dropout
        self.wq = nn.Linear(cfg.d_model, self.n_head * self.hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=False)
        self.wo = nn.Linear(self.n_head * self.hd, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, kv_cache=None):
        """cos/sin are pre-sliced to this chunk's absolute positions.

        kv_cache: optional (past_k, past_v) of shape (B, n_kv, T_past, hd) for
        incremental decoding. Returns (y, new_cache); new_cache is the updated
        (k, v) to pass back on the next step.
        """
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if kv_cache is not None:                     # prepend cached K/V
            pk, pv = kv_cache
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)
        new_cache = (k, v)

        kk, vv = k, v
        if self.n_kv != self.n_head:                 # expand KV heads for GQA
            rep = self.n_head // self.n_kv
            kk = k.repeat_interleave(rep, dim=1)
            vv = v.repeat_interleave(rep, dim=1)

        # prefill (q_len == k_len): causal. decode (q_len==1 < k_len): attend all.
        is_causal = q.shape[2] == kk.shape[2]
        y = F.scaled_dot_product_attention(
            q, kk, vv, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.hd)
        return self.wo(y), new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(8 * cfg.d_model / 3)            # 2/3 * 4d, Llama convention
        m = cfg.ffn_multiple_of
        hidden = m * ((hidden + m - 1) // m)
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)   # gate
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = GroupedQueryAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        a, new_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + a
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight       # weight tying

        cos, sin = _rope_tables(cfg.head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 trick)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()            # tied; counted once
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"seq len {T} > block {self.cfg.block_size}"
        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos[:, :, :T]
        sin = self.rope_sin[:, :, :T]
        for blk in self.blocks:
            x, _ = blk(x, cos, sin)
        x = self.norm(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1,
            )
            return logits, loss
        logits = self.lm_head(x[:, [-1], :])            # only last pos at infer
        return logits, None

    def configure_optimizers(self, weight_decay, lr, betas, device_type):
        # 2D+ params (matmuls, embeddings) decay; 1D params (norms) do not.
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

    def _sample(self, logits, temperature, top_k):
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Yield one new token (B,1) per step, KV-cached — for real-time streaming.

        Total length (prompt + generated) is capped at block_size. A long prompt
        is truncated to the last block_size tokens.
        """
        bs = self.cfg.block_size
        idx = idx[:, -bs:]                               # respect context window
        caches = [None] * len(self.blocks)
        pos = 0
        cur = idx
        for _ in range(max_new_tokens):
            T = cur.shape[1]
            if pos + T > bs:                             # window full; stop cleanly
                return
            x = self.drop(self.tok_emb(cur))
            cos = self.rope_cos[:, :, pos:pos + T]
            sin = self.rope_sin[:, :, pos:pos + T]
            for i, blk in enumerate(self.blocks):
                x, caches[i] = blk(x, cos, sin, caches[i])
            x = self.norm(x)
            logits = self.lm_head(x[:, [-1], :])
            pos += T
            nxt = self._sample(logits, temperature, top_k)
            yield nxt
            cur = nxt                                    # next step: only new token

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """KV-cached generation, O(T) per step. Returns prompt+generated ids."""
        idx = idx[:, -self.cfg.block_size:]
        for nxt in self.generate_stream(idx, max_new_tokens, temperature, top_k):
            idx = torch.cat((idx, nxt), dim=1)
        return idx

    @torch.no_grad()
    def generate_naive(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """No-cache fallback: O(T^2) but supports unbounded sliding-window generation."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            nxt = self._sample(logits, temperature, top_k)
            idx = torch.cat((idx, nxt), dim=1)
        return idx
