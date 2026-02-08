"""Plot Accuracy vs Sparsity from bugfix experiment results."""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_FILE = "results/metrics/admm_vs_lasso_bugfix_20260207_075122.json"
OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

with open(RESULTS_FILE) as f:
    data = json.load(f)

# Organize by score type
by_score = {}
for exp in data:
    score = exp["score_name"]
    by_score.setdefault(score, []).append(exp)

# Color/marker scheme per method family
STYLE = {
    "ADMM_Adam_Global":  {"color": "#e63946", "marker": "o",  "ls": "-"},
    "ADMM_Adam_Layer":   {"color": "#f4a261", "marker": "s",  "ls": "-"},
    "ADMM_Adam_Neuron":  {"color": "#e76f51", "marker": "^",  "ls": "-"},
    "Lasso_Adam_Global": {"color": "#457b9d", "marker": "o",  "ls": "--"},
    "Lasso_Adam_Layer":  {"color": "#2a9d8f", "marker": "s",  "ls": "--"},
    "Lasso_Adam_Neuron": {"color": "#264653", "marker": "^",  "ls": "--"},
}

LABEL_MAP = {
    "ADMM_Adam_Global":  "ADMM Global",
    "ADMM_Adam_Layer":   "ADMM Layer",
    "ADMM_Adam_Neuron":  "ADMM Neuron",
    "Lasso_Adam_Global": "Lasso Global",
    "Lasso_Adam_Layer":  "Lasso Layer",
    "Lasso_Adam_Neuron": "Lasso Neuron",
}

# ---------- Plot 1: One subplot per score type ----------
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
fig.suptitle("Accuracy vs Sparsity — ADMM vs Lasso (Post η-fix)", fontsize=15, fontweight="bold")

for ax, (score_name, exps) in zip(axes, sorted(by_score.items())):
    for exp in exps:
        name = exp["class_name"]
        st = STYLE[name]
        spar = [(1 - r) * 100 for r in exp["remaining_weights"]]
        acc = exp["accuracy"]
        ax.plot(spar, acc, color=st["color"], marker=st["marker"], ls=st["ls"],
                linewidth=2, markersize=7, label=LABEL_MAP[name])
    ax.set_title(f"Score: {score_name}", fontsize=13)
    ax.set_xlabel("Sparsity (%)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/acc_vs_sparsity_by_score.png", dpi=200, bbox_inches="tight")
print(f"Saved {OUT_DIR}/acc_vs_sparsity_by_score.png")

# ---------- Plot 2: Zoomed-in high accuracy region ----------
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
fig2.suptitle("Accuracy vs Sparsity — Zoomed (Acc ≥ 90%)", fontsize=15, fontweight="bold")

for ax, (score_name, exps) in zip(axes2, sorted(by_score.items())):
    for exp in exps:
        name = exp["class_name"]
        st = STYLE[name]
        spar = [(1 - r) * 100 for r in exp["remaining_weights"]]
        acc = exp["accuracy"]
        # Only plot points with acc >= 85
        mask = [a >= 85 for a in acc]
        spar_f = [s for s, m in zip(spar, mask) if m]
        acc_f = [a for a, m in zip(acc, mask) if m]
        if spar_f:
            ax.plot(spar_f, acc_f, color=st["color"], marker=st["marker"], ls=st["ls"],
                    linewidth=2, markersize=7, label=LABEL_MAP[name])
    ax.set_title(f"Score: {score_name}", fontsize=13)
    ax.set_xlabel("Sparsity (%)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(90, 99)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/acc_vs_sparsity_zoomed.png", dpi=200, bbox_inches="tight")
print(f"Saved {OUT_DIR}/acc_vs_sparsity_zoomed.png")

# ---------- Plot 3: Split by method family (ADMM vs Lasso side-by-side) ----------
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
fig3.suptitle("ADMM vs Lasso — By Method Family and Score Type", fontsize=15, fontweight="bold")

families = {"ADMM": [], "Lasso": []}
for exp in data:
    if "ADMM" in exp["class_name"]:
        families["ADMM"].append(exp)
    else:
        families["Lasso"].append(exp)

for col, (family, exps) in enumerate(families.items()):
    fam_by_score = {}
    for exp in exps:
        fam_by_score.setdefault(exp["score_name"], []).append(exp)

    for row, (score_name, score_exps) in enumerate(sorted(fam_by_score.items())):
        ax = axes3[row, col]
        for exp in score_exps:
            name = exp["class_name"]
            st = STYLE[name]
            spar = [(1 - r) * 100 for r in exp["remaining_weights"]]
            acc = exp["accuracy"]
            # Annotate C values
            ax.plot(spar, acc, color=st["color"], marker=st["marker"], ls=st["ls"],
                    linewidth=2, markersize=6, label=LABEL_MAP[name])
            # Label first and last C value
            ax.annotate(f"C={exp['C'][0]}", (spar[0], acc[0]), fontsize=7,
                        textcoords="offset points", xytext=(5, 5), color=st["color"])
            ax.annotate(f"C={exp['C'][-1]}", (spar[-1], acc[-1]), fontsize=7,
                        textcoords="offset points", xytext=(5, -10), color=st["color"])
        ax.set_title(f"{family} — {score_name}", fontsize=12)
        ax.set_xlabel("Sparsity (%)", fontsize=11)
        ax.set_ylabel("Test Accuracy (%)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 105)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/acc_vs_sparsity_by_family.png", dpi=200, bbox_inches="tight")
print(f"Saved {OUT_DIR}/acc_vs_sparsity_by_family.png")

# ---------- Plot 4: Pareto front comparison ----------
fig4, ax4 = plt.subplots(figsize=(10, 7))
ax4.set_title("Pareto Front: Accuracy vs Sparsity (All Methods)", fontsize=14, fontweight="bold")

for exp in data:
    name = exp["class_name"]
    score = exp["score_name"]
    st = STYLE[name]
    spar = [(1 - r) * 100 for r in exp["remaining_weights"]]
    acc = exp["accuracy"]
    linestyle = st["ls"] if score == "Magnitude" else ":"
    alpha = 1.0 if score == "Magnitude" else 0.6
    label = f"{LABEL_MAP[name]} ({score[:3]})"
    ax4.plot(spar, acc, color=st["color"], marker=st["marker"], ls=linestyle,
             linewidth=1.5, markersize=5, alpha=alpha, label=label)

ax4.set_xlabel("Sparsity (%)", fontsize=12)
ax4.set_ylabel("Test Accuracy (%)", fontsize=12)
ax4.legend(fontsize=8, ncol=2, loc="lower left")
ax4.grid(True, alpha=0.3)
ax4.set_xlim(30, 105)
ax4.set_ylim(50, 99)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/pareto_all_methods.png", dpi=200, bbox_inches="tight")
print(f"Saved {OUT_DIR}/pareto_all_methods.png")

print("\nAll plots saved to results/figures/")
