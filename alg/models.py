import copy
import functools
import math
from dataclasses import dataclass
from typing import Optional

import transformers
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput
import torch
import torch.nn as nn
import torch.nn.functional as F

def zero_grad_hook(grad):
    return torch.zeros_like(grad)


# ---------------------------------------------------------------------------
# Universal Transformer
# ---------------------------------------------------------------------------

def sinusoidal_encoding(seq_len: int, d_model: int, device) -> torch.Tensor:
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


def step_encoding(step: int, seq_len: int, d_model: int, device) -> torch.Tensor:
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(torch.tensor(step, dtype=torch.float, device=device) * div_term)
    pe[:, 1::2] = torch.cos(torch.tensor(step, dtype=torch.float, device=device) * div_term)
    return pe.unsqueeze(0)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff      = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class UniversalTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model:    int   = 256,
        n_heads:    int   = 4,
        d_ff:       int   = 512,
        max_steps:  int   = 8,
        n_blocks:   int   = 1,
        eps:        float = 0.01,
        tau:        float = 0.01,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.d_model   = d_model
        self.max_steps = max_steps
        self.eps       = eps
        self.tau       = tau

        self.embedding  = nn.Embedding(vocab_size, d_model)
        self.blocks     = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for i in range(n_blocks)])
        self.loop_gates = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(n_blocks)])
        for gate in self.loop_gates:
            nn.init.constant_(gate.bias, 1.0)
        self.head       = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, collect_hidden_states: bool = False,
                analysis_mode: bool = False, n_steps: int = None):
        B, T = x.shape
        device = x.device

        o = self.embedding(x)
        o = o + sinusoidal_encoding(T, self.d_model, device)  # position encoding once

        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)
        act_loss = torch.tensor(0.0, device=device)
        hidden_states = [] if collect_hidden_states else None
        gate_probs       = [] if analysis_mode else None  # list of (B, T) per step
        pct_halted_steps = [] if analysis_mode else None  # list of float per step

        # ACT mode: one shared block loops up to max_steps with halting
        halting_prob  = x.new_zeros(B, T, dtype=torch.float)
        remainders    = x.new_zeros(B, T, dtype=torch.float)
        n_updates     = x.new_zeros(B, T, dtype=torch.float)
        still_running = x.new_ones(B, T, dtype=torch.bool)
        final_state   = x.new_zeros(B, T, self.d_model, dtype=torch.float)

        if hidden_states is not None:
            hidden_states.append(o.detach().float().cpu())

        if self.max_steps > 1:
            steps = n_steps if n_steps is not None else self.max_steps

            for block, gate in zip(self.blocks, self.loop_gates):
                halting_prob  = x.new_zeros(B, T, dtype=torch.float)
                remainders    = x.new_zeros(B, T, dtype=torch.float)
                n_updates     = x.new_zeros(B, T, dtype=torch.float)
                still_running = x.new_ones(B, T, dtype=torch.bool)

                for step in range(steps):
                    o = o + step_encoding(step, T, self.d_model, device)
                    o = block(o, mask=causal_mask)

                    if hidden_states is not None:
                        hidden_states.append(o.detach().float().cpu())

                    p = torch.sigmoid(gate(o).squeeze(-1))

                    if gate_probs is not None:
                        gate_probs.append(p.detach().float().cpu())

                    new_halted = (
                        still_running
                        & ((halting_prob + p) >= 1.0 - self.eps)
                    )
                    still_running_next = (
                        still_running
                        & ((halting_prob + p) < 1.0 - self.eps)
                    )

                    remainders   += new_halted.float() * (1.0 - halting_prob).clamp(min=0)
                    halting_prob = (halting_prob + (
                        new_halted.float()           * (1.0 - halting_prob)
                        + still_running_next.float() * p
                    )).clamp(0, 1)

                    update_weights = (
                        new_halted.float()           * remainders
                        + still_running_next.float() * p
                    )

                    final_state += update_weights.unsqueeze(-1) * o

                    n_updates += still_running.float()

                    o = torch.where(
                        (~still_running_next).unsqueeze(-1),
                        o.detach(),
                        o,
                    )

                    still_running = still_running_next

                    if pct_halted_steps is not None:
                        pct_halted_steps.append((~still_running).float().mean().item())

                    if not analysis_mode and not still_running.any():
                        break

                ponder_time = n_updates + remainders
                act_loss += self.tau * ponder_time.mean()

        else:
            # standard stacked blocks mode (n_blocks > 1, no looping)
            blocks_to_run = self.blocks[:n_steps] if n_steps is not None else self.blocks
            for block in blocks_to_run:
                o = block(o, mask=causal_mask)
                if hidden_states is not None:
                    hidden_states.append(o.detach().float().cpu())
            final_state = o

        # finally compute logits
        logits = self.head(final_state)

        avg_loops = n_updates.mean()
        avg_remainder = remainders.mean()
        return logits, {"n_updates": n_updates, "remainders": remainders, "act_loss": act_loss, "avg_loops": avg_loops, "avg_remainder": avg_remainder, "hidden_states": hidden_states, "gate_probs": gate_probs, "pct_halted_steps": pct_halted_steps, "last_hidden": o}


# ---------------------------------------------------------------------------
# Ouro Transformer (ByteDance-style looped transformer with adaptive exit)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def precompute_rope_freqs(d_head: int, max_len: int, base: float = 10000.0, device=None) -> torch.Tensor:
    """
    Precompute RoPE sin/cos tables.
    Returns (max_len, d_head) float tensor: interleaved [cos0, sin0, cos1, sin1, ...].
    Stored as real-valued to stay torch.compile-friendly (avoids complex ops).
    """
    theta = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    pos   = torch.arange(max_len, device=device).float()
    ang   = torch.outer(pos, theta)                     # (T, d//2)
    # interleave cos/sin so we can apply in pairs
    cos  = ang.cos()                                    # (T, d//2)
    sin  = ang.sin()                                    # (T, d//2)
    return torch.stack([cos, sin], dim=-1).reshape(max_len, d_head)  # (T, d_head)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to x of shape (B, n_heads, T, d_head).
    freqs: (T, d_head) with interleaved [cos, sin] pairs.
    """
    T, D = freqs.shape
    cos = freqs[:, 0::2]   # (T, d//2)
    sin = freqs[:, 1::2]   # (T, d//2)
    # Rotate pairs (x0, x1) → (x0·cos − x1·sin, x0·sin + x1·cos)
    x0 = x[..., 0::2]     # (B, H, T, d//2)
    x1 = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, T, d//2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * cos - x1 * sin
    out[..., 1::2] = x0 * sin + x1 * cos
    return out


class OuroAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads    = n_heads
        self.d_head     = d_model // n_heads
        self.q_proj     = nn.Linear(d_model, d_model, bias=False)
        self.k_proj     = nn.Linear(d_model, d_model, bias=False)
        self.v_proj     = nn.Linear(d_model, d_model, bias=False)
        self.o_proj     = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop  = dropout

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask=None) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k = apply_rope(q, freqs_cis), apply_rope(k, freqs_cis)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, T, D))


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class OuroBlock(nn.Module):
    """Single Ouro block: pre-RMSNorm sandwich, RoPE MHA, SwiGLU FFN."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm_attn = RMSNorm(d_model)
        self.attn      = OuroAttention(d_model, n_heads, dropout)
        self.norm_ffn  = RMSNorm(d_model)
        self.ffn       = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask=None) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), freqs_cis, mask)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class OuroTransformer(nn.Module):
    """
    Ouro Transformer: a single shared block looped up to max_steps times.

    Training objective (ELBO with entropy regularization, eq. 4 in the paper):
        L = Σ_t p_φ(t|x) · L^(t)  −  β · H(p_φ(·|x))

    Exit distribution (survival-based, eq. 3):
        λ_t = σ(Linear(h^(t)))          per-token exit probability at step t
        S_t  = ∏_{j=1}^{t} (1 − λ_j),   S_0 = 1
        p_φ(t|x) = λ_t · S_{t−1}        for t < T_max
        p_φ(T_max|x) = S_{T_max−1}      remaining mass

    avg_loops = E[t] = Σ_t t · mean(p_φ(t|x))
    """
    def __init__(
        self,
        vocab_size: int,
        d_model:    int   = 256,
        n_heads:    int   = 4,
        d_ff:       int   = 512,
        max_steps:  int   = 8,
        beta:       float = 0.01,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.d_model   = d_model
        self.max_steps = max_steps
        self.beta      = beta
        self.d_head    = d_model // n_heads

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.block     = OuroBlock(d_model, n_heads, d_ff, dropout)
        self.gate      = nn.Linear(d_model, 1)   # per-token exit gate
        self.norm_out  = RMSNorm(d_model)
        self.head      = nn.Linear(d_model, vocab_size, bias=False)

        # Neutral gate init: λ_t ≈ 0.5 initially → explores all depths
        nn.init.zeros_(self.gate.bias)

        # Register a small buffer so the freqs device follows the model, but we
        # recompute for the actual T each forward to avoid any shape mismatch.
        self.register_buffer("_rope_dummy", torch.zeros(1), persistent=False)

    def forward(
        self,
        x:                    torch.Tensor,
        labels:               torch.Tensor = None,
        collect_hidden_states: bool = False,
        analysis_mode:        bool = False,
        n_steps:              int  = None,
    ):
        B, T = x.shape
        device = x.device

        h     = self.embedding(x)   # (B, T, d_model) — RoPE handles position
        freqs = precompute_rope_freqs(self.d_head, max_len=T, device=device)
        causal = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)

        steps    = n_steps if n_steps is not None else self.max_steps
        survival = torch.ones(B, T, device=device)   # S_0 = 1

        step_probs:  list = []   # p_φ(t|x), each (B, T)
        step_logits: list = []   # logits at step t, each (B, T, V)
        hidden_states = [h.detach().float().cpu()] if collect_hidden_states else None

        for t in range(1, steps + 1):
            h = self.block(h, freqs, causal)

            if collect_hidden_states:
                hidden_states.append(h.detach().float().cpu())

            lam = torch.sigmoid(self.gate(h).squeeze(-1))   # (B, T)

            if t < steps:
                p_t      = lam * survival          # λ_t · S_{t−1}
                survival = survival * (1.0 - lam)  # S_t
            else:
                p_t = survival                     # remaining mass

            step_probs.append(p_t)
            step_logits.append(self.head(self.norm_out(h)))   # (B, T, V)

        probs_stack  = torch.stack(step_probs,  dim=0)   # (steps, B, T)
        logits_stack = torch.stack(step_logits, dim=0)   # (steps, B, T, V)

        # Expected logits: Σ_t p_t · logits_t  (for inference / perplexity monitoring)
        expected_logits = (probs_stack.unsqueeze(-1) * logits_stack).sum(dim=0)  # (B, T, V)

        # Average exit step: E[t] = Σ_t t · p_t, averaged over B and T
        t_idx     = torch.arange(1, steps + 1, device=device, dtype=torch.float)
        avg_loops = (probs_stack * t_idx.view(-1, 1, 1)).sum(dim=0).mean()

        # Entropy of exit distribution H(p) = −Σ_t p_t log p_t
        entropy = -(probs_stack * probs_stack.clamp(min=1e-8).log()).sum(dim=0).mean()

        # ELBO loss (computed when labels are provided)
        ouro_loss = None
        if labels is not None:
            V              = logits_stack.shape[-1]
            labels_shifted = labels[:, 1:].contiguous()   # (B, T-1)
            expected_ce    = torch.tensor(0.0, device=device)
            for t_i in range(steps):
                logits_t = logits_stack[t_i][:, :-1, :].contiguous()  # (B, T-1, V)
                p_t      = probs_stack[t_i][:, :-1].contiguous()       # (B, T-1)
                ce_t     = F.cross_entropy(
                    logits_t.view(-1, V), labels_shifted.view(-1), reduction='none'
                ).view(B, T - 1)                                        # (B, T-1)
                expected_ce = expected_ce + (p_t * ce_t).sum() / (B * (T - 1))
            ouro_loss = expected_ce - self.beta * entropy

        return expected_logits, {
            "ouro_loss":     ouro_loss,
            "avg_loops":     avg_loops,
            "entropy":       entropy,
            "hidden_states": hidden_states,
            "step_probs":    [p.detach().float().cpu() for p in step_probs]
                             if (collect_hidden_states or analysis_mode) else None,
            "last_hidden":   h,
        }


@dataclass
class OuroCausalLMOutput(ModelOutput):
    logits:        Optional[torch.Tensor] = None
    ouro_loss:     Optional[torch.Tensor] = None
    avg_loops:     Optional[torch.Tensor] = None
    entropy:       Optional[torch.Tensor] = None
    hidden_states: Optional[list]         = None
    step_probs:    Optional[list]         = None


class OuroConfig(PretrainedConfig):
    model_type = "ouro_transformer"

    def __init__(
        self,
        vocab_size: int   = 50257,
        d_model:    int   = 256,
        n_heads:    int   = 4,
        d_ff:       int   = 512,
        max_steps:  int   = 8,
        beta:       float = 0.01,
        dropout:    float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.d_ff       = d_ff
        self.max_steps  = max_steps
        self.beta       = beta
        self.dropout    = dropout


class OuroTransformerForCausalLM(PreTrainedModel):
    config_class = OuroConfig
    is_ouro      = True   # detected by the training objective

    def __init__(self, config: OuroConfig):
        super().__init__(config)
        self.ouro = OuroTransformer(
            vocab_size = config.vocab_size,
            d_model    = config.d_model,
            n_heads    = config.n_heads,
            d_ff       = config.d_ff,
            max_steps  = config.max_steps,
            beta       = config.beta,
            dropout    = config.dropout,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.ouro.embedding

    def set_input_embeddings(self, value):
        self.ouro.embedding = value

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        pass

    def forward(self, input_ids, labels=None, collect_hidden_states=False,
                analysis_mode=False, n_steps=None, **kwargs):
        expected_logits, aux = self.ouro(
            input_ids,
            labels               = labels,
            collect_hidden_states = collect_hidden_states,
            analysis_mode        = analysis_mode,
            n_steps              = n_steps,
        )
        return OuroCausalLMOutput(
            logits        = expected_logits,
            ouro_loss     = aux["ouro_loss"],
            avg_loops     = aux["avg_loops"],
            entropy       = aux["entropy"],
            hidden_states = aux["hidden_states"],
            step_probs    = aux["step_probs"],
        )


# ---------------------------------------------------------------------------

@dataclass
class UTCausalLMOutput(ModelOutput):
    logits: Optional[torch.Tensor] = None
    act_loss: Optional[torch.Tensor] = None
    avg_loops: Optional[torch.Tensor] = None
    avg_remainder: Optional[torch.Tensor] = None
    hidden_states: Optional[list] = None


class UniversalTransformerConfig(PretrainedConfig):
    model_type = "universal_transformer"

    def __init__(
        self,
        vocab_size: int   = 50257,
        d_model:    int   = 256,
        n_heads:    int   = 4,
        d_ff:       int   = 512,
        max_steps:  int   = 8,
        n_blocks:   int   = 1,
        eps:        float = 0.01,
        tau:        float = 0.01,
        dropout:    float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.d_ff       = d_ff
        self.max_steps  = max_steps
        self.n_blocks   = n_blocks
        self.eps        = eps
        self.tau        = tau
        self.dropout    = dropout


class UniversalTransformerForCausalLM(PreTrainedModel):
    config_class = UniversalTransformerConfig

    def __init__(self, config: UniversalTransformerConfig):
        super().__init__(config)
        self.ut = UniversalTransformer(
            vocab_size = config.vocab_size,
            d_model    = config.d_model,
            n_heads    = config.n_heads,
            d_ff       = config.d_ff,
            max_steps  = config.max_steps,
            n_blocks   = config.n_blocks,
            eps        = config.eps,
            tau        = config.tau,
            dropout    = config.dropout,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.ut.embedding

    def set_input_embeddings(self, value):
        self.ut.embedding = value

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        pass  # UT is small; gradient checkpointing not needed

    def forward(self, input_ids, collect_hidden_states=False,
                analysis_mode=False, n_steps=None, **kwargs):
        logits, aux = self.ut(input_ids, collect_hidden_states=collect_hidden_states,
                              analysis_mode=analysis_mode, n_steps=n_steps)
        return UTCausalLMOutput(logits=logits, act_loss=aux["act_loss"], avg_loops=aux["avg_loops"], avg_remainder=aux["avg_remainder"], hidden_states=aux["hidden_states"])


# ---------------------------------------------------------------------------

def get_model_tokenizer(model_args, **model_kwargs):
    if getattr(model_args, "model_type", "hf") == "ouro":
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.ut_tokenizer)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        config = OuroConfig(
            vocab_size = model_args.ut_vocab_size or tokenizer.vocab_size,
            d_model    = model_args.ut_d_model,
            n_heads    = model_args.ut_n_heads,
            d_ff       = model_args.ut_d_ff,
            max_steps  = model_args.ut_max_steps,
            beta       = model_args.ut_beta,
            dropout    = model_args.ut_dropout,
        )
        model = OuroTransformerForCausalLM(config)
        return model, tokenizer

    if getattr(model_args, "model_type", "hf") == "universal_transformer":
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.ut_tokenizer)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        config = UniversalTransformerConfig(
            vocab_size = model_args.ut_vocab_size or tokenizer.vocab_size,
            d_model    = model_args.ut_d_model,
            n_heads    = model_args.ut_n_heads,
            d_ff       = model_args.ut_d_ff,
            max_steps  = model_args.ut_max_steps,
            n_blocks   = model_args.ut_n_blocks,
            eps        = model_args.ut_eps,
            tau        = model_args.ut_tau,
            dropout    = model_args.ut_dropout,
        )
        model = UniversalTransformerForCausalLM(config)
        return model, tokenizer

    if model_args.use_lk:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        # automodel_cls = AutoLigerKernelForCausalLM
        # TODO: remove hack below, use above comment once https://github.com/linkedin/Liger-Kernel/issues/242 is fixed
        class PatchedAutoLiger(AutoLigerKernelForCausalLM):
            @staticmethod
            def from_config(config, *args, **kwargs):
                AutoLigerKernelForCausalLM.from_pretrained(config._name_or_path)
                return AutoLigerKernelForCausalLM.from_config(config, *args, **kwargs)
        automodel_cls = PatchedAutoLiger

    else:
        automodel_cls = transformers.AutoModelForCausalLM
    model = automodel_cls.from_pretrained(
        model_args.model_name,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        **model_kwargs
    )
    
    model = model.type(torch.bfloat16)
    # model.model called here 
    replace_linear_layers(model.model)
    # weight copy upcasts to float32, so we downcast
    model = model.type(torch.bfloat16)
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name)
    except:
        raise ValueError
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

class QuantizeLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.weight 
        # straight through estimator for grads
        x = x + (quantize(x) - x).detach()
        output = torch.nn.functional.linear(input, x, bias=self.bias)
        return output

def replace_linear_layers(module):
    """
    Recursively replace all nn.Linear layers in 'module' (and its submodules)
    with LpLqLinear layers, preserving weight and bias.
    """
    for name, child in module.named_children():
        # If the child is a Linear layer, replace it with LpLqLinear
        if isinstance(child, nn.Linear):
            # Create new LpLqLinear with matching hyperparameters and dimensions
            new_layer = QuantizeLinear(child.in_features, child.out_features, bias=False)
            # Copy old weights
            new_layer.weight.data.copy_(child.weight.data) 
            # Assign the new layer back to the parent
            setattr(module, name, new_layer)
        else:
            # Recursively descend into child modules
            replace_linear_layers(child)

def quantize(input, max_scale=0.7):
    # TWN (Ternary Weight Networks) per row Quantizer
    out = input.clone().detach()
    out = out.reshape(-1, input.shape[-1])
    # Per Channel/Group Quantization
    n = out[0].nelement()
    m = out.data.norm(p=1, dim=1).div(n)
    thres = (max_scale * m).view(-1, 1).expand_as(out)
    pos = (out > thres).float()
    neg = (out < -thres).float()
    mask = (out.abs() > thres).float()
    alpha = ((mask * out).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
    result = alpha * pos - alpha * neg

    result = result.reshape(input.shape) 

    return result