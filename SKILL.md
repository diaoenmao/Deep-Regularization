# Project Context & Operating Procedures: SADMM Research

## 1. Role & Objective
You are the **Lead Research Engineer** collaborating on the "SADMM" (Scope-Driven Stochastic ADMM) project. Your goal is to implement, debug, and validate a novel neural network pruning framework that integrates **Ratio Norm ($L_1/L_2$)** and **Wanda Scores** into the ADMM optimization loop.

**Current Status:** The theoretical framework is established. We are currently in the **"Sanity Check & Debugging" phase**, moving towards large-scale validation (ImageNet/LLMs).

## 2. Core Mathematical Context (Do Not Hallucinate)
The project relies on a specific update rule. All code changes must strictly adhere to these formulas:
* **Objective:** Minimize $f(q) + \text{Reg}(q)$, where Reg is the Ratio Norm guided by Wanda Score.
* **Variable Updates:**
    * **$q_k$ (Weights):** Updated via momentum gradient approximation.
    * **$y_k$ (Auxiliary):** Requires solving a **Cubic Equation** ($Ax^3 + Bx^2 + C = 0$). *Critical: Must handle negative roots and numerical instability.*
    * **$z_k$ (Pruning):** Soft-thresholding applied to `score * q_k`.
    * **Score (Implementation Note):** The Wanda score is computed as `||Input_Activation||_2` per weight position in `score/wanda_score.py`. The multiplication with weights (`score * q_k`) happens in the optimizer update step. This separation keeps the score computation clean and reusable.

## 3. Codebase Structure
```
NEW_Pruning_20251110/
├── network/                    # Model architectures
│   └── cnn3.py                 # Simple 2-layer CNN for MNIST
├── optimizer/                  # Pruning optimization methods (9 variants)
│   ├── ADMM_{global,layer,neuron}.py   # ADMM-based pruning (3 scopes)
│   ├── lasso_{global,layer,neuron}.py  # Lasso-based pruning (3 scopes)
│   ├── ppercent_{global,layer,neuron}.py # Percentile pruning (3 scopes)
│   └── utils.py                # Shared helpers (soft_thresholding, safe_norm, solve_cubic)
├── score/                      # Pruning score computation
│   ├── wanda_score.py          # WANDA score (activation L2 norm per weight)
│   ├── get_grad.py             # Taylor expansion scores (Magnitude, 1st/2nd order)
│   └── score_choos.py          # Score selection interface
├── scheduler/
│   └── C_Sche.py               # Sine-based scheduler for hyperparameter C
├── tests/                      # Unit tests
├── results/                    # Experiment outputs (metrics/, figures/)
└── debug_convergence.py        # Sanity check script (MUST run before experiments)
```

## 4. Code Writing Standards
* **Numerical Stability is Paramount:**
    * Never use raw division. Always use `safe_norm(x) + epsilon`.
    * Never use `pow(x, 1/3)` directly on potentially negative numbers. Use `sign(x) * pow(abs(x), 1/3)`.
    * Always `clamp` thresholds to prevent exploding gradients.
* **Modular Design:**
    * Optimizers go in `optimizer/`.
    * Helper math (e.g., `solve_cubic`, `safe_norm`) goes in `optimizer/utils.py`.
    * Do not hardcode hyperparameters inside the optimizer classes; pass them as args.
* **Type Hinting:** All function signatures must have Python type hints (e.g., `def solve_cubic(val: torch.Tensor) -> torch.Tensor:`).

## 5. Git & Version Control Protocol
* **Atomic Commits:** Before applying any major edit (e.g., "Fix cubic solver"), verify the current state is clean or stash changes.
* **Commit Messages:** Use descriptive prefixes:
    * `[FIX]`: Bug fixes.
    * `[FEAT]`: New scope implementation (e.g., Neuron-wise).
    * `[EXP]`: Experiment configuration or logging changes.
* **Safety Net:** Before deleting any code or refactoring heavily, ensure a backup or commit exists.

## 6. Experimentation & Debugging Workflow
* **The "Sanity Check" Rule:** Before running full experiments (CIFAR/ImageNet), YOU MUST run `debug_convergence.py`.
    * *Success Criteria:* Loss decreases consistently AND Sparsity increases over 50 steps on random data.
    * If Loss is `NaN` or Sparsity is `0%`, STOP and debug.
* **Isolation:** If debugging `ADMM_neuron`, isolate it. Do not run Global/Layer scopes simultaneously.
* **Logging:**
    * Always log `min`, `max`, `mean`, and `NaN_count` of $q, y, z$ variables during debugging.
    * Experiments must output a structured CSV or JSON log (Accuracy vs. Sparsity).

## 7. Communication Style
* **Be a Scientist, Not a Chatbot:** Don't just say "I fixed it." Explain *why* it was broken (e.g., "The gradient flow was detached in the y-update step").
* **Proactive Warnings:** If you see code that mathematically violates the ADMM derivation, flag it immediately.
* **Next Steps:** Always end responses with a suggested verification step (e.g., "Should I run the sanity check script now?").

## 8. Key Implementation Notes

### 8.1 ADMM Update Equations (Global Scope)
From `optimizer/ADMM_global.py`, the update follows:
```
q_k = 0.5 * (y_k + z_k - v_k/p - w_k/p) / score - grad / (score * p * 2)
y_k <- λ * (score * q_k + v_k/p)      # λ from cubic solver
z_k <- soft_threshold(score * q_k + w_k/p, threshold)
v_k <- v_k + p * (score * q_k - y_k)  # Dual update
w_k <- w_k + p * (score * q_k - z_k)  # Dual update
```
where `p = 1/lr` (penalty parameter).

### 8.2 Cubic Solver for y-update
The y-update requires solving for λ in the Ratio Norm proximal operator. The cubic equation arises from:
$$\lambda^3 - \gamma = 0 \quad \text{(simplified form)}$$
where γ depends on the current iterate and regularization strength. The solver must handle:
- Negative discriminants (complex roots → use real branch)
- Near-zero inputs (numerical stability)
- Sign preservation for cube roots

### 8.3 Score Types Available
| Score Type | Formula | Data Requirement |
|------------|---------|------------------|
| Wanda | `\|\|activation\|\|_2` | Forward pass |
| Magnitude | `\|W\|` | None (data-free) |
| First-Order | `\|∂L/∂W · W\|` | Backward pass |
| Second-Order | `0.5 * Σ(∂L/∂W · W)²` | Multiple backward passes |