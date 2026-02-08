"""Quick head-to-head: ADMM vs Lasso (global/layer/neuron) on identical C values.

Trains each method for 200 steps on a 1024-sample MNIST subset, evaluates on
1000 test samples. Uses magnitude score and lr=0.002 for a fair comparison.
"""
from __future__ import annotations
import os, time, torch, random
from torch import nn
from torch.utils.data import DataLoader, Subset
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
from score.score_choos import choose_score

SEED = 42
LR = 0.002
TRAIN_STEPS = 100
EPOCHS = 2
SCORE_NAME = "magnitude"
C_VALUES = [0.01, 0.02, 0.03, 0.05]
VAL_SIZE = 1000
MAX_TIME_PER_RUN = 60  # seconds – skip if exceeded

def set_seed(s=SEED):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def align_scores(model, score_dict):
    return [score_dict[n] for n, _ in model.named_parameters()]

def compute_sparsity(model):
    total = sum(p.numel() for p in model.parameters())
    zeros = sum((p == 0).sum().item() for p in model.parameters())
    return zeros / total if total else 0.0

def make_data():
    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST("data/MNIST", train=True, download=True, transform=tf)
    test_ds  = datasets.MNIST("data/MNIST", train=False, download=True, transform=tf)
    train_loader = DataLoader(Subset(train_ds, list(range(1024))), batch_size=64, shuffle=True)
    val_loader   = DataLoader(Subset(test_ds,  list(range(VAL_SIZE))), batch_size=128)
    return train_loader, val_loader, len(train_ds)

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total

def train_one(method_name, device, lr, c_val, train_loader, val_loader, N, score_name):
    set_seed()
    model = CNN().to(device)
    wanda = WANDA_ScoreCalculator(model)
    grad_col = GradientCollector(model)
    criterion = nn.CrossEntropyLoss()
    params = list(model.parameters())
    zeros = [torch.zeros_like(p) for p in params]
    score_bufs = [torch.ones_like(p) for p in params]

    if method_name.startswith("ADMM"):
        cls = {"ADMM_global": ADMM_Adam_global, "ADMM_layer": ADMM_Adam_layer,
               "ADMM_neuron": ADMM_Adam_neuron}[method_name]
        opt = cls(params, lr=lr, N=N, C=c_val,
                  vk=[z.clone() for z in zeros], wk=[z.clone() for z in zeros],
                  yk=[p.clone().detach() for p in params],
                  zk=[p.clone().detach() for p in params],
                  score=score_bufs)
    else:
        cls = {"Lasso_global": Lasso_global, "Lasso_layer": Lasso_layer,
               "Lasso_neuron": Lasso_neuron}[method_name]
        opt = cls(params, lr=lr, N=N, C=c_val,
                  vk=[z.clone() for z in zeros],
                  zk=[z.clone() for z in zeros],
                  score=score_bufs)

    model.train()
    steps = 0
    t_start = time.time()
    for _ in range(EPOCHS):
        for imgs, tgts in train_loader:
            if time.time() - t_start > MAX_TIME_PER_RUN:
                break
            imgs, tgts = imgs.to(device), tgts.to(device)
            opt.zero_grad()
            loss = criterion(model(imgs), tgts)
            loss.backward()
            sd = choose_score(wanda, grad_col, score_name)
            for buf, new in zip(score_bufs, align_scores(model, sd)):
                buf.copy_(torch.clamp(new, min=1e-3))
            opt.step()
            steps += 1
            if steps >= TRAIN_STEPS:
                break
        if steps >= TRAIN_STEPS or time.time() - t_start > MAX_TIME_PER_RUN:
            break

    acc = evaluate(model, val_loader, device)
    sp = compute_sparsity(model)
    wanda.remove_hooks()
    return acc, sp

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, N = make_data()
    methods = ["ADMM_global", "ADMM_layer", "ADMM_neuron",
               "Lasso_global", "Lasso_layer", "Lasso_neuron"]

    print(f"{'Method':<16} {'C':>6} {'Accuracy':>10} {'Sparsity':>10} {'Time':>7}")
    print("-" * 55)

    for c in C_VALUES:
        for m in methods:
            t0 = time.time()
            acc, sp = train_one(m, device, LR, c, train_loader, val_loader, N, SCORE_NAME)
            elapsed = time.time() - t0
            print(f"{m:<16} {c:>6.3f} {acc*100:>9.2f}% {sp*100:>9.1f}% {elapsed:>6.1f}s")
        print()

if __name__ == "__main__":
    main()
