"""Post-bugfix ADMM vs Lasso comparison experiment.

Bug fixes applied before this run:
1. ADMM z-threshold: heuristic (lr*C*0.01/score) → derived C/(N*||y||_2)
2. ADMM_neuron η: removed double-score in ||dk|| (dk already contains score)
3. Lasso global/layer: unified threshold to lr*C/N (matching lasso_neuron)

Runs all 6 ADMM+Lasso variants × 4 score types × 10 C values = 240 experiments.
Uses the SAME C values as the previous full experiment for direct comparison.

Usage:
    python run_admm_vs_lasso.py --epochs 10 --device cuda
    python run_admm_vs_lasso.py --epochs 10 --device cuda --quick  # 2 scores only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from network.cnn3 import CNN
from optimizer.ADMM_global import ADMM_Adam_global
from optimizer.ADMM_layer import ADMM_Adam_layer
from optimizer.ADMM_neuron import ADMM_Adam_neuron
from optimizer.lasso_global import Lasso_global
from optimizer.lasso_layer import Lasso_layer
from optimizer.lasso_neuron import Lasso_neuron
from score.wanda_score import WANDA_ScoreCalculator
from score.get_grad import GradientCollector
from score.score_choos import choose_score, normalize_scores


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="data/MNIST", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="data/MNIST", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    return train_loader, test_loader, len(train_ds)


def align_scores(model, score_dict):
    return [score_dict[name] for name, _ in model.named_parameters()]


def compute_sparsity(model):
    total = sum(p.numel() for p in model.parameters())
    zeros = sum((p == 0).sum().item() for p in model.parameters())
    return zeros / total if total > 0 else 0.0


def compute_pq_index(model):
    w = torch.cat([p.data.abs().flatten() for p in model.parameters()])
    if len(w) == 0:
        return 0.0
    l1, l2 = w.sum(), torch.sqrt((w ** 2).sum())
    return (l1 / l2 / (len(w) ** 0.5)).item() if l2 > 0 else 0.0


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return 100.0 * correct / total


EARLY_STOP_PATIENCE = 3
EARLY_STOP_MIN_ACC = 15.0
EARLY_STOP_MIN_DELTA = 0.1


def train_loop(model, optimizer, train_loader, test_loader, device, epochs, score_name,
               wanda_calc, grad_collector, score_buffers):
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    patience = 0
    needs_fisher = "second order" in score_name.lower()

    for epoch in range(epochs):
        model.train()
        if needs_fisher:
            grad_collector.reset_fisher()

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            if needs_fisher:
                grad_collector.accumulate_fisher()

            sd = normalize_scores(choose_score(wanda_calc, grad_collector, score_name))
            for buf, new in zip(score_buffers, align_scores(model, sd)):
                buf.copy_(torch.clamp(new, min=1e-8))
            optimizer.step()

        acc = evaluate(model, test_loader, device)
        if acc < EARLY_STOP_MIN_ACC:
            break
        if acc > best_acc + EARLY_STOP_MIN_DELTA:
            best_acc = acc
            patience = 0
        else:
            patience += 1
        if patience >= EARLY_STOP_PATIENCE:
            break

    final_acc = evaluate(model, test_loader, device)
    return max(best_acc, final_acc), compute_sparsity(model), compute_pq_index(model)


# ---------------------------------------------------------------------------
# Factory: create model + optimizer
# ---------------------------------------------------------------------------

def make_admm(scope, device, lr, c_val, N_total):
    model = CNN().to(device)
    params = list(model.parameters())
    zeros = [torch.zeros_like(p) for p in params]
    score_bufs = [torch.ones_like(p) for p in params]
    cls = {"global": ADMM_Adam_global, "layer": ADMM_Adam_layer, "neuron": ADMM_Adam_neuron}[scope]
    opt = cls(params, lr=lr, N=N_total, C=c_val,
              vk=[z.clone() for z in zeros], wk=[z.clone() for z in zeros],
              yk=[p.clone().detach() for p in params],
              zk=[p.clone().detach() for p in params],
              score=score_bufs)
    return model, opt, score_bufs


def make_lasso(scope, device, lr, c_val, N_total):
    model = CNN().to(device)
    params = list(model.parameters())
    zeros = [torch.zeros_like(p) for p in params]
    score_bufs = [torch.ones_like(p) for p in params]
    cls = {"global": Lasso_global, "layer": Lasso_layer, "neuron": Lasso_neuron}[scope]
    opt = cls(params, lr=lr, N=N_total, C=c_val,
              vk=zeros, zk=[p.clone() for p in params], score=score_bufs)
    return model, opt, score_bufs


# ---------------------------------------------------------------------------
# C values — SAME as previous full experiment
# ---------------------------------------------------------------------------
C_ADMM_GLOBAL  = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06]
C_ADMM_LAYER   = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2]
C_ADMM_NEURON  = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2]
C_LASSO_GLOBAL = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5]
C_LASSO_LAYER  = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5]
C_LASSO_NEURON = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

CLASS_MAP = {
    "ADMM_global": "ADMM_Adam_Global", "ADMM_layer": "ADMM_Adam_Layer",
    "ADMM_neuron": "ADMM_Adam_Neuron",
    "Lasso_global": "Lasso_Adam_Global", "Lasso_layer": "Lasso_Adam_Layer",
    "Lasso_neuron": "Lasso_Adam_Neuron",
}
SCORE_MAP = {
    "magnitude": "Magnitude", "first order": "First-Order",
    "second order": "Second-Order", "first order + second order": "First+Second-Order",
}

METHODS = [
    # (key, family, scope, c_values)
    ("ADMM_global",  "admm",  "global",  C_ADMM_GLOBAL),
    ("ADMM_layer",   "admm",  "layer",   C_ADMM_LAYER),
    ("ADMM_neuron",  "admm",  "neuron",  C_ADMM_NEURON),
    ("Lasso_global", "lasso", "global",  C_LASSO_GLOBAL),
    ("Lasso_layer",  "lasso", "layer",   C_LASSO_LAYER),
    ("Lasso_neuron", "lasso", "neuron",  C_LASSO_NEURON),
]


def run(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    set_seed(args.seed)
    train_loader, test_loader, N_total = get_dataloaders(args.batch_size)

    score_names = (["magnitude", "first order + second order"]
                   if args.quick
                   else ["magnitude", "first order", "second order", "first order + second order"])

    total = sum(len(m[3]) for m in METHODS) * len(score_names)

    # Log to file for reliable monitoring
    os.makedirs("results", exist_ok=True)
    logfile = open("results/experiment_progress.log", "w")
    def log(msg):
        print(msg, flush=True)
        logfile.write(msg + "\n")
        logfile.flush()

    log(f"Device: {device} | Epochs: {args.epochs} | Experiments: {total}")
    log(f"Score types: {score_names}")
    log(f"Bug-fixed run — ADMM threshold=C/(N*||y||), Lasso threshold=lr*C/N")
    log("=" * 70)

    results = []
    exp_i = 0
    t0 = time.time()

    for key, family, scope, c_vals in METHODS:
        for sn in score_names:
            entry = {"class_name": CLASS_MAP[key], "score_name": SCORE_MAP[sn],
                     "C": [], "accuracy": [], "remaining_weights": [], "pq_index": []}

            for c in c_vals:
                exp_i += 1
                set_seed(args.seed)  # reproducible per run
                maker = make_admm if family == "admm" else make_lasso
                model, opt, score_bufs = maker(scope, device, args.lr, c, N_total)
                wanda = WANDA_ScoreCalculator(model)
                gc = GradientCollector(model)

                acc, sp, pq = train_loop(model, opt, train_loader, test_loader,
                                         device, args.epochs, sn, wanda, gc, score_bufs)
                wanda.remove_hooks()

                entry["C"].append(c)
                entry["accuracy"].append(acc)
                entry["remaining_weights"].append(1.0 - sp)
                entry["pq_index"].append(pq)

                elapsed = time.time() - t0
                eta = elapsed / exp_i * (total - exp_i)
                log(f"[{exp_i}/{total}] {key:16s} | {sn:28s} | C={c:<8g} | "
                    f"Acc={acc:6.2f}% Spar={sp*100:5.1f}% | ETA {eta/60:.0f}m")

            # Sort by remaining weights
            idx = sorted(range(len(entry["remaining_weights"])),
                         key=lambda i: entry["remaining_weights"][i], reverse=True)
            entry["save_wei_sorted"]    = [entry["remaining_weights"][i] for i in idx]
            entry["save_accwei_sorted"] = [entry["accuracy"][i] for i in idx]
            entry["save_pq_sorted"]     = [entry["pq_index"][i] for i in idx]
            entry["save_accpq_sorted"]  = [entry["accuracy"][i] for i in idx]
            results.append(entry)

            # Save incremental results after each method+score combo
            os.makedirs("results/metrics", exist_ok=True)
            inc_path = "results/metrics/admm_vs_lasso_bugfix_latest.json"
            with open(inc_path, "w") as f:
                json.dump(results, f, indent=2)

    # Save final
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/metrics/admm_vs_lasso_bugfix_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {out_path}")
    log(f"Total time: {(time.time()-t0)/3600:.2f} hours")

    # Print summary table
    log("\n" + "=" * 90)
    log(f"{'Method':<20s} {'Score':<22s} {'Best Acc%':>9s} {'Sparsity%':>10s} {'Best C':>8s}")
    log("-" * 90)
    for e in results:
        best_i = max(range(len(e["accuracy"])),
                     key=lambda i: e["accuracy"][i] if e["remaining_weights"][i] < 0.5 else -1,
                     default=0)
        # Fallback: just pick highest accuracy
        if e["accuracy"][best_i] <= 0:
            best_i = max(range(len(e["accuracy"])), key=lambda i: e["accuracy"][i])
        sp = (1 - e["remaining_weights"][best_i]) * 100
        log(f"{e['class_name']:<20s} {e['score_name']:<22s} "
            f"{e['accuracy'][best_i]:>8.2f}% {sp:>9.1f}% {e['C'][best_i]:>8g}")
    log("=" * 90)
    logfile.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="Only 2 score types (faster)")
    run(p.parse_args())
