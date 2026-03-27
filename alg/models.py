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
        self.blocks      = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for i in range(n_blocks)])
        self.loop_gate  = nn.Linear(d_model, 1)
        nn.init.constant_(self.loop_gate.bias, 1.0)
        self.head       = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        B, T = x.shape
        device = x.device

        o = self.embedding(x)
        o = o + sinusoidal_encoding(T, self.d_model, device)  # position encoding once

        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)
        act_loss = torch.tensor(0.0, device=device)

        # ACT mode: one shared block loops up to max_steps with halting
        halting_prob  = x.new_zeros(B, T, dtype=torch.float)
        remainders    = x.new_zeros(B, T, dtype=torch.float)
        n_updates     = x.new_zeros(B, T, dtype=torch.float)
        still_running = x.new_ones(B, T, dtype=torch.bool)
        final_state   = x.new_zeros(B, T, self.d_model, dtype=torch.float)
        
        if self.max_steps > 1:

            for block in self.blocks:
                for step in range(self.max_steps):
                    o = o + step_encoding(step, T, self.d_model, device)
                    o = block(o, mask=causal_mask)

                    p = torch.sigmoid(self.loop_gate(o).squeeze(-1))

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

                    if not still_running.any():
                        break

                ponder_time = n_updates + remainders
                act_loss += self.tau * ponder_time.mean()

        else:
            # standard stacked blocks mode (n_blocks > 1, no looping)
            for block in self.blocks:
                o = block(o, mask=causal_mask)
            final_state = o

        # finally compute logits
        logits = self.head(final_state)

        return logits, {"n_updates": n_updates, "remainders": remainders, "act_loss": act_loss}


@dataclass
class UTCausalLMOutput(ModelOutput):
    logits: Optional[torch.Tensor] = None
    act_loss: Optional[torch.Tensor] = None


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

    def forward(self, input_ids, **kwargs):
        logits, aux = self.ut(input_ids)
        return UTCausalLMOutput(logits=logits, act_loss=aux["act_loss"])


# ---------------------------------------------------------------------------

def get_model_tokenizer(model_args, **model_kwargs):
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