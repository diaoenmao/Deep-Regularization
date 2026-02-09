#!/usr/bin/env python
"""
Deep diagnostic for xor (m=128) failure case.

Produces 3 outputs:
  A. "Death Valley" — Wanda Score histogram (signal vs noise features)
  B. "Evolution"    — Loss & Sparsity vs Epoch (dual-axis)
  C. "Top-K"        — Table of top-10 feature scores with ground truth labels

Also runs a sanity check: 2-layer MLP (Hidden=32, ReLU) to test if
non-linearity is the bottleneck.
"""
import sys, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(ROOT, "Feature-Selection-Benchmark")
sys.path.insert(0, BENCH)
sys.path.insert(0, ROOT)

from src.data import generate_dataset
from src.admm_lasso_wrapper import (
    FeatureSelectionMLP, _Scaler, _warm_start_from_lasso,
    compute_mlp_wanda_scores, _adaptive_rho_update, _extract_feature_importance,
)
from optimizer.ADMM_global import ADMM_Adam_global
from src.utils import TrainingSet

# ── Config ────────────────────────────────────────────────────────────
DATASET   = "xor"
M         = 128
K         = 2        # xor has 2 informative features
N_SAMPLES = 500
EPOCHS    = 100
LR        = 0.005
C_SPARSE  = 0.08
BATCH     = 64
SEED      = 42
OUT_DIR   = os.path.join(ROOT, "results", "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Generate data ─────────────────────────────────────────────────────
np.random.seed(SEED)
torch.manual_seed(SEED)

X, X_tilde, y = generate_dataset(DATASET, N_SAMPLES, M)
X = 2.0 * X - 1.0

# Permute features (as the benchmark does)
perm = np.arange(M)
np.random.shuffle(perm)
X = X[:, perm]
gt_indices = set(np.where(perm < K)[0].tolist())  # ground truth positions
print(f"Ground truth feature indices (after permutation): {sorted(gt_indices)}")

X_train, y_train = X[:400], y[:400]

scaler = _Scaler()
X_train_s = scaler.fit_transform(X_train)


# ── Helper: instrumented training loop ────────────────────────────────
def train_instrumented(model, X_tr, y_tr, epochs, lr, C, tag=""):
    """Train and record per-epoch loss, sparsity, rho."""
    N = len(X_tr)
    dataset = TrainingSet(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0)

    params = list(model.parameters())
    zeros = [torch.zeros_like(p) for p in params]
    score_bufs = [torch.ones_like(p) for p in params]

    opt = ADMM_Adam_global(
        params, lr=lr, N=N, C=C,
        vk=[z.clone() for z in zeros],
        wk=[z.clone() for z in zeros],
        yk=[p.clone().detach() for p in params],
        zk=[p.clone().detach() for p in params],
        score=score_bufs,
    )

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    history = {"loss": [], "sparsity": [], "rho": []}

    model.train()
    for epoch in range(epochs):
        # Refresh WANDA scores
        if epoch % 10 == 0:
            model.eval()
            sample = torch.FloatTensor(X_tr[:min(256, N)])
            new_scores = compute_mlp_wanda_scores(model, sample)
            for sb, ns in zip(score_bufs, new_scores):
                sb.copy_(ns)
            model.train()

        # Adaptive rho
        if epoch > 0 and epoch % 5 == 0:
            _adaptive_rho_update(opt)

        epoch_losses = []
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb).reshape(len(xb))
            loss = criterion(out, yb.float())
            if torch.isnan(loss):
                epoch_losses.append(float("nan"))
                break
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        # Sparsity of first layer weights
        W1 = model.first_linear.weight.data
        sp = (W1.abs() < 1e-8).float().mean().item()

        history["loss"].append(np.mean(epoch_losses))
        history["sparsity"].append(sp)
        history["rho"].append(opt.rho)

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  {tag} Epoch {epoch:3d}: loss={history['loss'][-1]:.4f}  "
                  f"sparsity={sp:.3f}  rho={opt.rho:.1f}")

    model.eval()
    return history


# ══════════════════════════════════════════════════════════════════════
# PART 1: Main diagnostic — 5-layer MLP (current architecture)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 1: ADMM_global on xor (m=128) — 5-layer MLP (Mish)")
print("="*70)

model_main = FeatureSelectionMLP(
    input_size=M, n_classes=2, latent_size=58,
    n_hidden_layers=5, dropout=0.0, activation="mish",
)
_warm_start_from_lasso(model_main, X_train_s, y_train, 2)

hist_main = train_instrumented(model_main, X_train_s, y_train, EPOCHS, LR, C_SPARSE,
                               tag="[5L-Mish]")

# Extract final scores
scores_main = _extract_feature_importance(model_main, X_train_s)

# ── Plot A: "Death Valley" — Wanda Score histogram ────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
signal_mask = np.array([i in gt_indices for i in range(M)])
noise_mask  = ~signal_mask

ax.hist(scores_main[noise_mask], bins=30, alpha=0.6, color="gray",
        label=f"Noise features (n={noise_mask.sum()})", edgecolor="black")
if signal_mask.sum() > 0:
    for idx in np.where(signal_mask)[0]:
        ax.axvline(scores_main[idx], color="red", linewidth=2.5, linestyle="--",
                   label=f"Signal feat {idx} (score={scores_main[idx]:.2f})")
ax.set_xlabel("Wanda Score", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(f"A. Death Valley — Wanda Score Distribution\nxor m={M}, ADMM_global, 5L-Mish", fontsize=13)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "A_death_valley_xor128.png"), dpi=150)
print(f"\n[Saved] A_death_valley_xor128.png")

# ── Plot B: "Evolution" — Loss & Sparsity vs Epoch ────────────────────
fig, ax1 = plt.subplots(figsize=(10, 5))
epochs_x = np.arange(EPOCHS)

color_loss = "tab:blue"
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12, color=color_loss)
ax1.plot(epochs_x, hist_main["loss"], color=color_loss, linewidth=1.5, label="Loss")
ax1.tick_params(axis="y", labelcolor=color_loss)

ax2 = ax1.twinx()
color_sp = "tab:red"
ax2.set_ylabel("First-Layer Sparsity", fontsize=12, color=color_sp)
ax2.plot(epochs_x, hist_main["sparsity"], color=color_sp, linewidth=1.5,
         linestyle="--", label="Sparsity")
ax2.tick_params(axis="y", labelcolor=color_sp)

# Also show rho on a secondary annotation
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))
color_rho = "tab:green"
ax3.set_ylabel("ρ (rho)", fontsize=12, color=color_rho)
ax3.plot(epochs_x, hist_main["rho"], color=color_rho, linewidth=1.0,
         linestyle=":", alpha=0.7, label="ρ")
ax3.tick_params(axis="y", labelcolor=color_rho)

ax1.set_title(f"B. Evolution — Loss, Sparsity & ρ vs Epoch\nxor m={M}, ADMM_global, 5L-Mish", fontsize=13)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc="center right")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "B_evolution_xor128.png"), dpi=150)
print(f"[Saved] B_evolution_xor128.png")

# ── Table C: "Top-K Inspection" ───────────────────────────────────────
ranked = np.argsort(scores_main)[::-1]
print(f"\nC. Top-K Inspection — xor m={M}, ADMM_global")
print(f"   Ground truth indices: {sorted(gt_indices)}")
print(f"   {'Rank':<6} {'Feature':<10} {'Score':<12} {'Is Signal?':<12}")
print("   " + "-"*40)
for rank, idx in enumerate(ranked[:10]):
    is_gt = "✓ SIGNAL" if idx in gt_indices else ""
    print(f"   {rank+1:<6} {idx:<10} {scores_main[idx]:<12.4f} {is_gt}")

# Save Table C: Top-10 as CSV and as a small PNG table for easy viewing
rows = []
for rank, idx in enumerate(ranked[:10]):
    rows.append({
        "rank": rank + 1,
        "feature": int(idx),
        "score": float(scores_main[idx]),
        "is_signal": bool(idx in gt_indices),
    })

# CSV
csv_path = os.path.join(OUT_DIR, "C_top10_xor128.csv")
with open(csv_path, "w", newline="") as f:
    import csv
    writer = csv.DictWriter(f, fieldnames=["rank", "feature", "score", "is_signal"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
print(f"[Saved] C_top10_xor128.csv")

# PNG table
fig_tab, ax_tab = plt.subplots(figsize=(6, 2.5))
ax_tab.axis("off")
cell_text = [[r['rank'], r['feature'], f"{r['score']:.4f}", "SIGNAL" if r['is_signal'] else ""] for r in rows]
col_labels = ["Rank", "Feature", "Score", "Signal?"]
tbl = ax_tab.table(cellText=cell_text, colLabels=col_labels, loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.5)
plt.tight_layout()
img_path = os.path.join(OUT_DIR, "C_top10_xor128.png")
fig_tab.savefig(img_path, dpi=150, bbox_inches='tight')
plt.close(fig_tab)
print(f"[Saved] C_top10_xor128.png")

# Where do the true features actually rank?
for gt_idx in sorted(gt_indices):
    rank_pos = np.where(ranked == gt_idx)[0][0] + 1
    print(f"\n   → Signal feature {gt_idx}: rank {rank_pos}/{M}, score={scores_main[gt_idx]:.4f}")
    print(f"     Score percentile: {(1 - rank_pos/M)*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════
# PART 2: Sanity Check — 2-layer MLP (Hidden=32, ReLU)
# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "="*70)
print("PART 2: Sanity Check — 2-layer MLP (Hidden=32, ReLU)")
print("="*70)

model_shallow = FeatureSelectionMLP(
    input_size=M, n_classes=2, latent_size=32,
    n_hidden_layers=2, dropout=0.0, activation="relu",
)
_warm_start_from_lasso(model_shallow, X_train_s, y_train, 2)

hist_shallow = train_instrumented(model_shallow, X_train_s, y_train, EPOCHS, LR, C_SPARSE,
                                  tag="[2L-ReLU]")

scores_shallow = _extract_feature_importance(model_shallow, X_train_s)

ranked_shallow = np.argsort(scores_shallow)[::-1]
print(f"\nTop-10 features (2L-ReLU):")
print(f"   Ground truth indices: {sorted(gt_indices)}")
print(f"   {'Rank':<6} {'Feature':<10} {'Score':<12} {'Is Signal?':<12}")
print("   " + "-"*40)
for rank, idx in enumerate(ranked_shallow[:10]):
    is_gt = "✓ SIGNAL" if idx in gt_indices else ""
    print(f"   {rank+1:<6} {idx:<10} {scores_shallow[idx]:<12.4f} {is_gt}")

for gt_idx in sorted(gt_indices):
    rank_pos = np.where(ranked_shallow == gt_idx)[0][0] + 1
    print(f"\n   → Signal feature {gt_idx}: rank {rank_pos}/{M}, score={scores_shallow[gt_idx]:.4f}")

# Best-k for both
top_k_main = set(ranked[:K].tolist())
top_k_shallow = set(ranked_shallow[:K].tolist())
bestk_main = sum(1 for i in top_k_main if i in gt_indices) / K
bestk_shallow = sum(1 for i in top_k_shallow if i in gt_indices) / K

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"  5-layer Mish  best-k = {bestk_main:.0%}  (top-2: {sorted(top_k_main)})")
print(f"  2-layer ReLU  best-k = {bestk_shallow:.0%}  (top-2: {sorted(top_k_shallow)})")
print(f"  Ground truth:          {sorted(gt_indices)}")


# ══════════════════════════════════════════════════════════════════════
# PART 3: Extra — No ADMM baseline (pure LogisticRegression L1)
# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "="*70)
print("PART 3: Baseline — sklearn L1-Logistic (no neural net)")
print("="*70)

from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(penalty="l1", solver="saga", C=1.0,
                              max_iter=500, tol=1e-4, random_state=42)
lr_model.fit(X_train_s, y_train)
coef = np.abs(lr_model.coef_).ravel()
ranked_lr = np.argsort(coef)[::-1]

print(f"\nTop-10 features (sklearn L1-Logistic):")
print(f"   Ground truth indices: {sorted(gt_indices)}")
print(f"   {'Rank':<6} {'Feature':<10} {'|Coef|':<12} {'Is Signal?':<12}")
print("   " + "-"*40)
for rank, idx in enumerate(ranked_lr[:10]):
    is_gt = "✓ SIGNAL" if idx in gt_indices else ""
    print(f"   {rank+1:<6} {idx:<10} {coef[idx]:<12.4f} {is_gt}")

for gt_idx in sorted(gt_indices):
    rank_pos = np.where(ranked_lr == gt_idx)[0][0] + 1
    print(f"\n   → Signal feature {gt_idx}: rank {rank_pos}/{M}, |coef|={coef[gt_idx]:.4f}")

top_k_lr = set(ranked_lr[:K].tolist())
bestk_lr = sum(1 for i in top_k_lr if i in gt_indices) / K
print(f"\n  L1-Logistic   best-k = {bestk_lr:.0%}  (top-2: {sorted(top_k_lr)})")

# ── Combined histogram: all three methods ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (title, sc) in zip(axes, [
    ("5L-Mish ADMM", scores_main),
    ("2L-ReLU ADMM", scores_shallow),
    ("L1-Logistic", coef),
]):
    ax.hist(sc[noise_mask], bins=30, alpha=0.6, color="gray",
            label="Noise", edgecolor="black")
    for idx in np.where(signal_mask)[0]:
        ax.axvline(sc[idx], color="red", linewidth=2.5, linestyle="--",
                   label=f"Signal {idx} ({sc[idx]:.3f})")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Score")
    ax.legend(fontsize=8)

plt.suptitle(f"Death Valley Comparison — xor m={M}", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "D_death_valley_comparison_xor128.png"), dpi=150,
            bbox_inches="tight")
print(f"\n[Saved] D_death_valley_comparison_xor128.png")

print(f"\n\nAll diagnostics saved to: {OUT_DIR}")
