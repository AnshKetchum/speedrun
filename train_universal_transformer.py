import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math


# ---------------------------------------------------------------------------
# Position & step encodings
# ---------------------------------------------------------------------------

def sinusoidal_encoding(seq_len: int, d_model: int, device) -> torch.Tensor:
    """Standard sinusoidal positional encoding. Returns (1, T, E)."""
    position = torch.arange(seq_len, device=device).unsqueeze(1)        # (T, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model)
    )                                                                    # (E/2,)
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)                                               # (1, T, E)


def step_encoding(step: int, seq_len: int, d_model: int, device) -> torch.Tensor:
    """Sinusoidal encoding for the recurrence step index. Returns (1, T, E)."""
    # Same formula as position encoding but broadcasting a scalar step value
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(torch.tensor(step, dtype=torch.float, device=device) * div_term)
    pe[:, 1::2] = torch.cos(torch.tensor(step, dtype=torch.float, device=device) * div_term)
    return pe.unsqueeze(0)                                               # (1, T, E)


# ---------------------------------------------------------------------------
# Single shared transformer block  (attention + FFN + residuals + layernorm)
# ---------------------------------------------------------------------------

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
        # x: (B, T, E)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


# ---------------------------------------------------------------------------
# Universal Transformer with Adaptive Computation Time (ACT)
# ---------------------------------------------------------------------------

class UniversalTransformer(nn.Module):
    """
    Universal Transformer (Dehghani et al., 2018) with dynamic ACT halting.

    Args:
        vocab_size: input vocabulary size
        d_model:    embedding / hidden dimension  (E)
        n_heads:    number of attention heads
        d_ff:       feedforward inner dimension
        max_steps:  maximum number of recurrent steps (caps the loop)
        eps:        halting threshold  (token halts when cumulative_p >= 1 - eps)
        tau:        ACT loss coefficient
        dropout:    dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int   = 256,
        n_heads:    int   = 4,
        d_ff:       int   = 512,
        max_steps:  int   = 8,
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

        # Single shared block — weights are reused at every recurrent step
        self.block      = TransformerBlock(d_model, n_heads, d_ff, dropout)

        # Halting unit: linear → sigmoid  (bias init to 1 per Graves 2016)
        self.loop_gate  = nn.Linear(d_model, 1)
        nn.init.constant_(self.loop_gate.bias, 1.0)

        # Output head for language modeling
        self.head       = nn.Linear(d_model, vocab_size)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T)  token indices

        Returns:
            final_state: (B, T, E)
            aux: dict with keys
                'n_updates'  — (B, T) ponder time per token
                'remainders' — (B, T) remainder per token
                'act_loss'   — scalar auxiliary loss
        """
        B, T = x.shape
        device = x.device

        o = self.embedding(x)                           # (B, T, E)

        # ── ACT state ────────────────────────────────────────────────
        halting_prob  = x.new_zeros(B, T,    dtype=torch.float)  # cumulative p
        remainders    = x.new_zeros(B, T,    dtype=torch.float)  # remainder (regret)
        n_updates     = x.new_zeros(B, T,    dtype=torch.float)  # ponder time
        final_state   = x.new_zeros(B, T, self.d_model, dtype=torch.float)
        still_running = x.new_ones(B, T,    dtype=torch.bool)    # not-yet-halted

        for step in range(self.max_steps):

            # Inject position + step encodings at every recurrent pass
            pos_enc  = sinusoidal_encoding(T, self.d_model, device)   # (1, T, E)
            step_enc = step_encoding(step, T, self.d_model, device)   # (1, T, E)
            o = o + pos_enc + step_enc

            # Shared block (self-attention + FFN)
            o = self.block(o)                                          # (B, T, E)

            # Per-token halting probability for this step
            p = torch.sigmoid(self.loop_gate(o).squeeze(-1))          # (B, T)

            # Tokens newly crossing the halting threshold this step
            new_halted = (
                still_running
                & ((halting_prob + p) >= 1.0 - self.eps)
            )                                                          # (B, T) bool

            # Tokens still running and not yet at threshold
            still_running_next = (
                still_running
                & ((halting_prob + p) < 1.0 - self.eps)
            )                                                          # (B, T) bool

            # Accumulate remainders for newly halted tokens
            #    remainder = 1 - sum_of_previous_p  (not p itself)
            remainders   += new_halted.float() * (1.0 - halting_prob)
            halting_prob += (
                new_halted.float()       * (1.0 - halting_prob)   # clamp to 1
                + still_running_next.float() * p                  # normal accum
            )

            # Per-token update weights for the weighted hidden-state sum:
            #    halted  → remainder  |  running → p
            update_weights = (
                new_halted.float()       * remainders
                + still_running_next.float() * p
            )                                                          # (B, T)

            # Probability-weighted accumulation of the hidden state
            #    (NOT a binary mask — gradients flow through the weights)
            final_state += update_weights.unsqueeze(-1) * o           # (B, T, E)

            # Increment ponder time only for tokens still being processed
            n_updates += still_running.float()

            # Freeze halted token representations in o so they still
            #    participate correctly in cross-token attention next step
            o = torch.where(
                (~still_running_next).unsqueeze(-1),                   # halted
                o.detach(),
                o,
            )

            # Advance the running mask
            still_running = still_running_next

            # Early exit once every token in every batch has halted
            if not still_running.any():
                break

        # ── Auxiliary (ponder) loss ───────────────────────────────────
        ponder_time = n_updates + remainders                            # (B, T)
        act_loss    = self.tau * ponder_time.mean()

        logits = self.head(final_state)  # (B, T, vocab_size)

        return logits, {
            "n_updates":  n_updates,   # (B, T)
            "remainders": remainders,  # (B, T)
            "act_loss":   act_loss,    # scalar
        }


# ---------------------------------------------------------------------------
# Dummy dataset for next-token prediction
# ---------------------------------------------------------------------------

class DummyDataset:
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = torch.roll(x, -1, dims=0)  # shift for next token prediction
        y[-1] = 0  # dummy token for last position
        return x, y

def train_step(inputs: torch.Tensor, targets: torch.Tensor, model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    # forward
    logits, metadata = model(inputs)

    loss_ntp = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss_aux = metadata["act_loss"]
    loss = loss_ntp + loss_aux

    return logits, loss

def train(dataloader: DataLoader, model: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer, train_steps: int = 10):
    model = model.to(device)
    model.train() 

    for i, (inputs, targets) in enumerate(dataloader):

        if i >= train_steps:
            print(f"training complete on step {i}, stopping")
            break

        optimizer.zero_grad()

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        preds, loss = train_step(
           inputs, 
           targets,
           model, 
        )
        
        loss.backward()

        optimizer.step()

        if i % 10 == 0:
            print(f"Iter {i} - Loss {loss.item():.2f}")
    

if __name__ == "__main__":
    torch.manual_seed(42)

    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    # Hyperparameters
    B, T         = args.batch_size, args.sequence_length
    vocab_size   = args.vocab_size
    train_steps  = args.train_steps

    # Model
    model = UniversalTransformer(
        vocab_size = vocab_size,
        d_model    = args.d_model,
        n_heads    = args.n_heads,
        d_ff       = args.d_ff,
        max_steps  = args.max_steps,
        eps        = args.eps,
        tau        = args.tau,
    )

    # Dummy dataset
    dataset = DummyDataset(vocab_size, T, args.num_samples)
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pretrain
    print("Starting pretraining...")
    train(dataloader, model, device, optimizer, train_steps)
    print("Pretraining complete!")