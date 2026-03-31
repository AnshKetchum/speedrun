#!/usr/bin/env python3
"""
Analyze a trained UniversalTransformer or OuroTransformer checkpoint.

For UT:  loops over n_steps in 1..last_n_blocks (skip last N ACT steps).
For Ouro: same structure — run only first K steps, truncating the exit distribution.

Plots produced:
  - angular distance heatmap (depth × sequence position)
  - logit MSE vs full pass (per sequence position)
  - prob-dist MSE vs full pass (per sequence position)
  - % change in probability vs full pass
  - loop-gate / exit-probability vs depth (Ouro: also per-step exit distribution)
  - cumulative % halted vs depth (UT) / expected exit step histogram (Ouro)
  - NMI line + heatmap

Usage (UT):
    python alg/analyze_block_skipping.py \
        --checkpoint-path checkpoints/ut_model_1_block_6_loop/checkpoint-470 \
        --last-n-blocks 5 \
        --output-folder layer-experiments

Usage (Ouro):
    python alg/analyze_block_skipping.py \
        --checkpoint-path checkpoints/ouro_model_6_steps/checkpoint-470 \
        --last-n-blocks 5 \
        --output-folder layer-experiments-ouro
"""

import argparse
import json
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score

import transformers
from models import UniversalTransformerForCausalLM, OuroTransformerForCausalLM


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_fixed_batch(batch_idx: int, batch_size: int = 4, seq_len: int = 512):
    dataset = load_dataset(
        "/home/ac/CodingWorkspace/research/ideas/speedrun/data/test_dataset",
        split="train",
    )
    dataset = dataset.select_columns(["token_ids"])
    start, end = batch_idx * batch_size, batch_idx * batch_size + batch_size
    sequences  = [row["token_ids"] for row in dataset.select(range(start, end))]
    processed  = []
    for seq in sequences:
        if len(seq) >= seq_len:
            processed.append(seq[:seq_len])
        else:
            processed.append(seq + [0] * (seq_len - len(seq)))
    return torch.tensor(processed, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model-type detection
# ---------------------------------------------------------------------------

def detect_model_type(checkpoint_path: str) -> str:
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg.get("model_type", "universal_transformer")


def load_model(checkpoint_path: str):
    mtype = detect_model_type(checkpoint_path)
    if mtype == "ouro_transformer":
        model = OuroTransformerForCausalLM.from_pretrained(checkpoint_path)
    else:
        model = UniversalTransformerForCausalLM.from_pretrained(checkpoint_path)
    return model, mtype


# ---------------------------------------------------------------------------
# Forward passes — UT
# ---------------------------------------------------------------------------

@torch.no_grad()
def ut_forward_full(model, input_ids):
    ut = model.ut
    _, aux = ut(input_ids, collect_hidden_states=True, analysis_mode=True)
    logits = ut.head(aux["last_hidden"]).float().cpu()
    return logits, aux["hidden_states"], aux["gate_probs"], aux["pct_halted_steps"]


@torch.no_grad()
def ut_forward_skip(model, input_ids, skip_last_n: int):
    ut = model.ut
    total = ut.max_steps if ut.max_steps > 1 else len(ut.blocks)
    _, aux = ut(input_ids, collect_hidden_states=True, analysis_mode=True,
                n_steps=total - skip_last_n)
    logits = ut.head(aux["last_hidden"]).float().cpu()
    return logits, aux["hidden_states"]


# ---------------------------------------------------------------------------
# Forward passes — Ouro
# ---------------------------------------------------------------------------

@torch.no_grad()
def ouro_forward_full(model, input_ids):
    """Run all steps; collect hidden states and per-step exit probs."""
    expected_logits, aux = model.ouro(
        input_ids, collect_hidden_states=True, analysis_mode=True
    )
    return expected_logits.float().cpu(), aux["hidden_states"], aux["step_probs"], aux["avg_loops"]


@torch.no_grad()
def ouro_forward_skip(model, input_ids, skip_last_n: int):
    """Run only (max_steps − skip_last_n) steps; remaining mass goes to last step."""
    steps = model.ouro.max_steps - skip_last_n
    expected_logits, aux = model.ouro(
        input_ids, collect_hidden_states=True, analysis_mode=True, n_steps=steps
    )
    return expected_logits.float().cpu(), aux["hidden_states"]


# ---------------------------------------------------------------------------
# Unified dispatchers
# ---------------------------------------------------------------------------

def forward_full(model, input_ids, mtype):
    if mtype == "ouro_transformer":
        logits, hs, step_probs, avg_loops = ouro_forward_full(model, input_ids)
        return logits, hs, step_probs, None   # gate_probs=step_probs, pct_halted=None
    else:
        return ut_forward_full(model, input_ids)


def forward_skip(model, input_ids, skip_last_n, mtype):
    if mtype == "ouro_transformer":
        return ouro_forward_skip(model, input_ids, skip_last_n)
    else:
        return ut_forward_skip(model, input_ids, skip_last_n)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_angular_heatmap(hidden_states, max_depth: int):
    if len(hidden_states) < max_depth:
        hidden_states = hidden_states + [hidden_states[-1]] * (max_depth - len(hidden_states))
    states = torch.stack(hidden_states).mean(dim=1)         # (D, T, E)
    norms  = states.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    states = states / norms
    cos_sim = (states[:-1] * states[1:]).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos_sim).numpy().T                    # (T, D-1)


def mse_per_position(a, b):
    return (a - b).pow(2).mean(dim=(0, 2)).cpu().numpy()


def pct_change_per_position(probs_skip, probs_full):
    pct = (probs_skip - probs_full) / probs_full.clamp(min=1e-9) * 100.0
    pct = pct.mean(dim=0)
    return pct.mean(dim=-1).cpu().numpy(), pct.min(dim=-1).values.cpu().numpy(), pct.max(dim=-1).values.cpu().numpy()


# ---------------------------------------------------------------------------
# Token prediction printing
# ---------------------------------------------------------------------------

def print_token_predictions(label, logits, tokenizer, full_probs=None):
    probs    = torch.softmax(logits, dim=-1)
    last_p   = probs[:, -1, :]
    top_ids  = last_p.argmax(dim=-1)
    top_prob = last_p[range(len(top_ids)), top_ids]
    print(f"\n  [{label}]")
    for b in range(len(top_ids)):
        tok   = tokenizer.decode([top_ids[b].item()])
        extra = ""
        if full_probs is not None:
            extra = f"  (full prob: {full_probs[b, -1, top_ids[b]].item():.4f})"
        print(f"    seq {b}: {repr(tok):20s}  p={top_prob[b].item():.4f}{extra}")


# ---------------------------------------------------------------------------
# NMI / clustering
# ---------------------------------------------------------------------------

def fit_kmeans_layer1(hidden_states, n_clusters):
    hs   = hidden_states[1].numpy()
    B, T, D = hs.shape
    km   = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=5, max_iter=300)
    km.fit(hs.reshape(B * T, D))
    return km


def assign_labels(km, hidden_states):
    out = []
    for hs in hidden_states:
        arr = hs.numpy()
        B, T, D = arr.shape
        out.append(km.predict(arr.reshape(B * T, D)))
    return out


def compute_nmi_vs_ref(all_labels, ref_idx=1):
    ref = all_labels[min(ref_idx, len(all_labels) - 1)]
    return np.array([normalized_mutual_info_score(ref, lbl, average_method="arithmetic")
                     for lbl in all_labels])


def compute_nmi_matrix(all_labels):
    n   = len(all_labels)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = normalized_mutual_info_score(all_labels[i], all_labels[j],
                                             average_method="arithmetic")
            mat[i, j] = mat[j, i] = v
    return mat


# ---------------------------------------------------------------------------
# Plotting — shared
# ---------------------------------------------------------------------------

def save_figure(skip_n, n_steps, heatmap, mse_logits, mse_probs, output_folder,
                label="", pct_change=None):
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    title = label or f"Skip last {skip_n}  —  {n_steps - skip_n}/{n_steps} steps active"
    fig.suptitle(title, fontsize=13)

    ax = axes[0]
    im = ax.imshow(heatmap, aspect="auto", cmap="viridis", vmin=0, vmax=np.pi)
    ax.set_xlabel("Depth transition"); ax.set_ylabel("Sequence position")
    ax.set_title("Angular distances (rad)"); plt.colorbar(im, ax=ax, label="rad")

    axes[1].plot(mse_logits, color="steelblue")
    axes[1].set_xlabel("Sequence position"); axes[1].set_ylabel("MSE")
    axes[1].set_title("Logit MSE  (partial vs full)"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(mse_probs, color="tomato")
    axes[2].set_xlabel("Sequence position"); axes[2].set_ylabel("MSE")
    axes[2].set_title("Prob-dist MSE  (partial vs full)"); axes[2].grid(True, alpha=0.3)

    ax = axes[3]
    if pct_change is not None:
        mean, lo, hi = pct_change
        xs = np.arange(len(mean))
        ax.plot(xs, mean, color="darkorchid", label="mean")
        ax.fill_between(xs, lo, hi, alpha=0.2, color="darkorchid", label="min/max")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.legend(fontsize=8)
    ax.set_xlabel("Sequence position"); ax.set_ylabel("% change")
    ax.set_title("Prob % change vs full  (mean ± min/max)"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    slug  = label.replace(" ", "_") if label else f"skip_{skip_n:02d}"
    fname = os.path.join(output_folder, f"{slug}.png")
    fig.canvas.draw()
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


def save_summary_figure(results, n_steps, output_folder):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Step-skipping effect across all n", fontsize=13)
    cmap   = plt.get_cmap("plasma")
    colors = [cmap(i / max(len(results) - 1, 1)) for i in range(len(results))]
    for (skip_n, mse_l, mse_p), color in zip(results, colors):
        ax1.plot(mse_l, color=color, label=f"skip {skip_n}", alpha=0.8)
        ax2.plot(mse_p, color=color, label=f"skip {skip_n}", alpha=0.8)
    for ax, title in [(ax1, "Logit MSE"), (ax2, "Prob-dist MSE")]:
        ax.set_title(f"{title}  (partial vs full)"); ax.set_xlabel("Sequence position")
        ax.set_ylabel("MSE"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(output_folder, "summary_mse.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved summary: {fname}")


def save_angular_depth_plot(depth_entries, output_folder):
    fig, (ax, ax_diff) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    cmap   = plt.get_cmap("plasma")
    colors = [cmap(i / max(len(depth_entries) - 1, 1)) for i in range(len(depth_entries))]
    bw     = 0.8 / max(len(depth_entries), 1)
    for idx, ((label, heatmap), color) in enumerate(zip(depth_entries, colors)):
        avg = heatmap.mean(axis=0); lo = heatmap.min(axis=0); hi = heatmap.max(axis=0)
        xs  = np.arange(len(avg)); offset = (idx - len(depth_entries) / 2 + 0.5) * bw
        ax.bar(xs + offset, hi - lo, bottom=lo, width=bw * 0.8, color=color, alpha=0.25)
        ax.plot(xs, avg, marker="o", label=label, color=color)
        if len(avg) > 1:
            ax_diff.plot(xs[1:], np.diff(avg), marker="o", label=label, color=color)
    ax.set_ylabel("Angular distance (rad)")
    ax.set_title("Average angular distance per depth transition (bars = min/max)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax_diff.set_xlabel("Depth transition"); ax_diff.set_ylabel("Δ angular distance (rad)")
    ax_diff.set_title("First derivative of angular distance across depth")
    ax_diff.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_diff.legend(fontsize=8); ax_diff.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(output_folder, "angular_dist_vs_depth.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plotting — UT-specific
# ---------------------------------------------------------------------------

def save_cumulative_halted_plot(pct_halted_steps, output_folder):
    xs  = np.arange(len(pct_halted_steps))
    pct = np.array(pct_halted_steps) * 100.0
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, pct, marker="o", color="darkorange")
    for x, v in zip(xs, pct):
        ax.annotate(f"{v:.1f}%", (x, v), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8)
    ax.set_xlabel("Depth step"); ax.set_ylabel("Tokens halted (%)")
    ax.set_title("Cumulative % of tokens halted vs depth  (full pass)")
    ax.set_ylim(0, 105); ax.grid(True, alpha=0.3); plt.tight_layout()
    fname = os.path.join(output_folder, "cumulative_halted_vs_depth.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


def save_ut_gate_prob_plot(gate_probs, output_folder, eps=0.01):
    arrs = [p.numpy() for p in gate_probs]
    avg  = np.array([a.mean() for a in arrs])
    std  = np.array([a.std()  for a in arrs])
    xs   = np.arange(len(avg))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, avg, marker="o", color="steelblue", label="mean")
    ax.fill_between(xs, (avg - std).clip(0), (avg + std).clip(0, 1),
                    alpha=0.2, color="steelblue", label="±1 std")
    for x, v in zip(xs, avg):
        ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8)
    ax.axhline(1.0 - eps, color="crimson", linewidth=1.2, linestyle="--",
               label=f"1 − ε  ({1-eps:.3f})")
    ax.set_xlabel("Depth step"); ax.set_ylabel("Halting probability")
    ax.set_title("Average loop-gate probability vs depth  (full pass)")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(output_folder, "loop_gate_prob_vs_depth.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plotting — Ouro-specific
# ---------------------------------------------------------------------------

def save_ouro_exit_distribution_plot(step_probs, output_folder):
    """
    Plot the learned exit distribution p_φ(t|x) across loop steps.

    step_probs: list of (B, T) cpu float tensors, one per step.
    Shows mean ± std of p_t across all (batch, token) positions.
    """
    arrs  = [p.numpy() for p in step_probs]   # list of (B, T)
    avg   = np.array([a.mean() for a in arrs])
    std   = np.array([a.std()  for a in arrs])
    steps = np.arange(1, len(avg) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mean exit prob per step
    ax = axes[0]
    ax.bar(steps, avg, color="steelblue", alpha=0.7, label="mean p_φ(t|x)")
    ax.errorbar(steps, avg, yerr=std, fmt="none", color="black", capsize=4)
    ax.set_xlabel("Loop step t"); ax.set_ylabel("p_φ(t|x)")
    ax.set_title("Exit distribution: mean ± std per step  (full pass)")
    ax.set_xticks(steps); ax.grid(True, alpha=0.3, axis="y"); ax.legend(fontsize=9)

    # Right: cumulative exit mass
    ax = axes[1]
    cumulative = np.cumsum(avg)
    ax.plot(steps, cumulative, marker="o", color="darkorange")
    for s, v in zip(steps, cumulative):
        ax.annotate(f"{v:.2f}", (s, v), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8)
    ax.axhline(1.0, color="crimson", linewidth=1, linestyle="--", label="total mass = 1")
    ax.set_xlabel("Loop step t"); ax.set_ylabel("Cumulative exit mass")
    ax.set_title("Cumulative exit probability vs step  (full pass)")
    ax.set_ylim(0, 1.05); ax.set_xticks(steps); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_folder, "ouro_exit_distribution.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


def save_ouro_exit_heatmap(step_probs, output_folder):
    """
    Heatmap of per-token exit probability p_φ(t|x): (n_steps, T) averaged over batch.
    """
    arrs  = [p.numpy() for p in step_probs]   # list of (B, T)
    mat   = np.stack([a.mean(axis=0) for a in arrs], axis=0)  # (n_steps, T)

    fig, ax = plt.subplots(figsize=(max(6, mat.shape[1] // 20 + 2), max(4, mat.shape[0] * 0.6 + 1)))
    im = ax.imshow(mat, aspect="auto", cmap="plasma", vmin=0, vmax=mat.max())
    ax.set_xlabel("Sequence position"); ax.set_ylabel("Loop step t")
    ax.set_title("Exit probability p_φ(t|x) per token  (avg over batch)")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels([str(t + 1) for t in range(mat.shape[0])])
    plt.colorbar(im, ax=ax, label="p_φ(t|x)")
    plt.tight_layout()
    fname = os.path.join(output_folder, "ouro_exit_heatmap.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


def save_ouro_step_logit_comparison(step_logits_per_step, probs_full, output_folder, tokenizer):
    """
    For each step t, compare logits_t against the full expected logits.
    Shows MSE in logit space and prob space vs sequence position.

    step_logits_per_step: list of (B, T, V) tensors (one per step)
    probs_full: (B, T, V)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Per-step logit/prob divergence from expected output  (Ouro)", fontsize=13)
    cmap   = plt.get_cmap("plasma")
    n      = len(step_logits_per_step)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    for t_i, (logits_t, color) in enumerate(zip(step_logits_per_step, colors)):
        probs_t    = torch.softmax(logits_t.float(), dim=-1)
        mse_l      = mse_per_position(logits_t.float(), torch.log(probs_full.clamp(min=1e-9)))
        mse_p      = mse_per_position(probs_t, probs_full)
        label      = f"step {t_i + 1}"
        axes[0].plot(mse_l, color=color, label=label, alpha=0.8)
        axes[1].plot(mse_p, color=color, label=label, alpha=0.8)

    for ax, title, ylabel in [
        (axes[0], "Logit MSE vs expected logits", "MSE"),
        (axes[1], "Prob-dist MSE vs expected probs", "MSE"),
    ]:
        ax.set_title(title); ax.set_xlabel("Sequence position"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_folder, "ouro_step_divergence.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# NMI plots
# ---------------------------------------------------------------------------

def save_nmi_line_plot(nmi_curves, output_folder):
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap   = plt.get_cmap("plasma")
    colors = [cmap(i / max(len(nmi_curves) - 1, 1)) for i in range(len(nmi_curves))]
    for (label, nmi_arr), color in zip(nmi_curves, colors):
        ax.plot(np.arange(len(nmi_arr)), nmi_arr, marker="o", label=label, color=color)
    ax.set_xlabel("Layer index"); ax.set_ylabel("NMI vs layer 1")
    ax.set_title("NMI vs layer depth  (reference: layer-1 k-means, full pass)")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(output_folder, "nmi_vs_layer.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


def save_nmi_heatmap(label, nmi_matrix, output_folder):
    n   = nmi_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(max(5, n * 0.55 + 1), max(4, n * 0.5 + 1)))
    im  = ax.imshow(nmi_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("Layer index"); ax.set_ylabel("Layer index")
    ax.set_title(f"NMI between layers — {label}")
    plt.colorbar(im, ax=ax, label="NMI")
    ticks = np.arange(n)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([str(i) for i in ticks], fontsize=7)
    ax.set_yticklabels([str(i) for i in ticks], fontsize=7)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{nmi_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if nmi_matrix[i, j] < 0.6 else "black")
    plt.tight_layout()
    fname = os.path.join(output_folder, f"nmi_heatmap_{label.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step-skipping analysis for UT and Ouro")
    parser.add_argument("--last-n-blocks",   type=int, required=True,
                        help="Maximum number of final steps to skip (1..N)")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--batch",           type=int, default=0)
    parser.add_argument("--batch-size",      type=int, default=4)
    parser.add_argument("--seq-len",         type=int, default=512)
    parser.add_argument("--output-folder",   type=str, default="layer-experiments")
    parser.add_argument("--n-clusters",      type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint_path}")
    model, mtype = load_model(args.checkpoint_path)
    tokenizer    = transformers.AutoTokenizer.from_pretrained(args.checkpoint_path)
    model.eval().to(device)

    if mtype == "ouro_transformer":
        total_steps = model.ouro.max_steps
        print(f"  Model: Ouro  |  max_steps={total_steps}")
    else:
        total_steps = model.ut.max_steps if model.ut.max_steps > 1 else len(model.ut.blocks)
        print(f"  Model: UT  |  n_blocks={len(model.ut.blocks)}, max_steps={model.ut.max_steps}, total={total_steps}")

    if args.last_n_blocks >= total_steps:
        args.last_n_blocks = total_steps - 1
        print(f"WARNING: clamped --last-n-blocks to {args.last_n_blocks}")

    print(f"Loading batch {args.batch} (size={args.batch_size}, seq_len={args.seq_len})")
    input_ids = get_fixed_batch(args.batch, args.batch_size, args.seq_len).to(device)
    print(f"  input_ids: {input_ids.shape}")

    # ---- Full forward pass ----
    print("Running full forward pass ...")
    logits_full, hidden_full, extra_full, _ = forward_full(model, input_ids, mtype)
    logits_full = logits_full.float().cpu()
    probs_full  = torch.softmax(logits_full, dim=-1)
    max_depth   = len(hidden_full)
    print(f"  {max_depth} depth states, logits {logits_full.shape}")
    print_token_predictions("full pass", logits_full, tokenizer)

    heatmap_full = compute_angular_heatmap(hidden_full, max_depth)
    T = logits_full.shape[1]
    save_figure(0, total_steps, heatmap_full, np.zeros(T), np.zeros(T), args.output_folder,
                label="full pass", pct_change=(np.zeros(T), np.zeros(T), np.zeros(T)))

    # ---- Model-specific full-pass plots ----
    if mtype == "ouro_transformer" and extra_full is not None:
        step_probs_full = extra_full   # list of (B, T) cpu tensors
        save_ouro_exit_distribution_plot(step_probs_full, args.output_folder)
        save_ouro_exit_heatmap(step_probs_full, args.output_folder)

        # Per-step logit divergence from expected output
        # Re-run to collect step logits (forward_full doesn't return them separately;
        # use a quick per-step forward that collects logits)
        with torch.no_grad():
            from models import precompute_rope_freqs
            ouro     = model.ouro
            h        = ouro.embedding(input_ids)
            freqs    = precompute_rope_freqs(ouro.d_head, max_len=T, device=device)
            causal   = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)
            survival = torch.ones(input_ids.shape[0], T, device=device)
            all_step_logits = []
            for t in range(1, total_steps + 1):
                h    = ouro.block(h, freqs, causal)
                lam  = torch.sigmoid(ouro.gate(h).squeeze(-1))
                if t < total_steps:
                    survival = survival * (1.0 - lam)
                all_step_logits.append(ouro.head(ouro.norm_out(h)).float().cpu())
        save_ouro_step_logit_comparison(all_step_logits, probs_full, args.output_folder, tokenizer)

    elif mtype != "ouro_transformer" and extra_full is not None:
        gate_probs_full, pct_halted_full = extra_full, _
        # Unpack: forward_full for UT returns (logits, hs, gate_probs, pct_halted)
        # We stored gate_probs as extra_full; pct_halted was returned as 4th element (None here)
        # Re-unpack properly:
        logits_full2, hidden_full2, gate_probs_full2, pct_halted2 = ut_forward_full(model, input_ids)
        if gate_probs_full2:
            save_ut_gate_prob_plot(gate_probs_full2, args.output_folder, eps=model.ut.eps)
        if pct_halted2:
            save_cumulative_halted_plot(pct_halted2, args.output_folder)

    # ---- Per-skip analysis ----
    summary_results = []
    depth_entries   = [("full pass", heatmap_full)]

    for skip_n in range(1, args.last_n_blocks + 1):
        print(f"\nSkipping last {skip_n} step(s) ...")
        logits_skip, hidden_skip = forward_skip(model, input_ids, skip_n, mtype)
        logits_skip = logits_skip.float().cpu()
        probs_skip  = torch.softmax(logits_skip, dim=-1)

        heatmap    = compute_angular_heatmap(hidden_skip, max_depth)
        mse_logits = mse_per_position(logits_skip, logits_full)
        mse_probs  = mse_per_position(probs_skip,  probs_full)
        pct_change = pct_change_per_position(probs_skip, probs_full)

        print_token_predictions(f"skip {skip_n}", logits_skip, tokenizer, full_probs=probs_full)
        save_figure(skip_n, total_steps, heatmap, mse_logits, mse_probs, args.output_folder,
                    pct_change=pct_change)
        summary_results.append((skip_n, mse_logits, mse_probs))
        depth_entries.append((f"skip {skip_n}", heatmap))

    # ---- Summary plots ----
    print("\nGenerating summary figures ...")
    save_summary_figure(summary_results, total_steps, args.output_folder)
    save_angular_depth_plot(depth_entries, args.output_folder)

    # ---- NMI analysis ----
    print(f"\nRunning NMI analysis (k={args.n_clusters}) ...")
    km = fit_kmeans_layer1(hidden_full, n_clusters=args.n_clusters)

    pass_hidden = {"full pass": hidden_full}
    for sn in range(1, args.last_n_blocks + 1):
        _, hs = forward_skip(model, input_ids, sn, mtype)
        pass_hidden[f"skip {sn}"] = hs

    nmi_curves = []
    for pass_label, hs in pass_hidden.items():
        labels_km = assign_labels(km, hs)
        nmi_arr   = compute_nmi_vs_ref(labels_km, ref_idx=1)
        nmi_curves.append((pass_label, nmi_arr))
        mat = compute_nmi_matrix(labels_km)
        save_nmi_heatmap(pass_label, mat, args.output_folder)

    save_nmi_line_plot(nmi_curves, args.output_folder)
    print("Done.")


if __name__ == "__main__":
    main()
