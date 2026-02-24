# ADMM Code vs Paper Analysis

## Date: 2025-02-24
## Status: ACTIVE — informs paper rewrite decisions

---

## Executive Summary

The paper describes a **3-variable ADMM** (g, y, z with duals v, w).
The code implements a **2-variable ADMM** (g, z with dual u).

However, the code's z-step is NOT the true proximal operator of L1/L2.
It is a **sequential decomposition**: soft-threshold (L1) then cubic rescale (L2).
This is mathematically closer to **one round of the paper's y-step + z-step collapsed
into a single pass**, but with important differences in the cubic equation and inputs.

The true proximal operator of ||z||_1/||z||_2 requires sorting + bisection over
sparsity levels + coupled nonlinear system solve (see Tao & Lou, SIAM 2021;
Wang et al., arXiv 2108.01269). Neither the paper nor the code computes this.

---

## Detailed Comparison

### Variables and Constraints

| Aspect | Paper (3-variable) | Code (2-variable) |
|--------|--------------------|--------------------|
| Variables | g, y, z | g, z |
| Constraints | g=y, g=z | g=z |
| Duals | v (for g=y), w (for g=z) | u (for g=z) |
| y's role | Handle L2 denominator | Does not exist |
| z's role | Handle L1 numerator | Combined: soft-threshold + cubic |

### The g-step

| Aspect | Paper | Code |
|--------|-------|------|
| Frequency | Per mini-batch | Per mini-batch |
| Method | Adam on augmented Lagrangian | Adam on augmented Lagrangian |
| Penalty terms | (ρ/2)||g-y+v/ρ||² + (ρ/2)||g-z+w/ρ||² | (ρ/2)||g-z+u||² |
| # penalty terms | TWO (one per constraint) | ONE |

**Mismatch severity: MEDIUM.** Paper has two quadratic penalties pulling g toward
both y and z. Code has one pulling g toward z only. Since y≈z at convergence this
may not matter much in practice, but the gradient is different during training.

### The y-step (paper) vs cubic rescale (code)

**Paper y-step:**
- Subproblem: min_y  c/(N||y||_2) + (ρ/2)||y - d||²
- Where d = g + v/ρ  (input is g plus FIRST dual)
- Where c = Σ λ_j |z_j|  (uses PREVIOUS z values)
- Solution: y = τ·d where τ solves τ³ - τ² - D = 0
- D = c / (N·ρ·||d||_2³)
- Cubic: **τ³ - τ² - D = 0** (NOT depressed)

**Code cubic rescale:**
- Input: v_shrunk (AFTER soft-thresholding of g+u)
- D_k = (C · ||v_shrunk||_1 · score²) / (N · ρ · ||v_shrunk||_2³)
- τ solves: **τ³ - τ - D = 0** (depressed cubic)
- Output: z = τ · v_shrunk

**Key differences:**
1. **Different cubic:** τ³-τ²-D=0 (paper) vs τ³-τ-D=0 (code)
2. **Different input:** d = g+v/ρ (paper) vs v_shrunk = S(g+u, λ/ρ) (code, post-threshold)
3. **Different D formula:** Paper uses c=Σλ|z_old|, code uses C·||v_shrunk||_1
4. **Order:** Paper does cubic THEN threshold. Code does threshold THEN cubic.
5. **Per-element vs scalar:** Code D_k is per-element (via score²), paper D is scalar

### The z-step (paper) vs soft-threshold (code)

**Paper z-step:**
- Subproblem: min_z  (1/(N||y||_2)) Σ λ_j|z_j| + (ρ/2)||g-z+w/ρ||²
- Input: g + w/ρ  (uses SECOND dual)
- Threshold: λ_j / (N·ρ·||y||_2)
- Note: threshold depends on ||y||_2 from the y-step

**Code soft-threshold:**
- Input: v = g + u  (single dual)
- Threshold: λ_j / ρ  (where λ_j = C/score_j)
- Note: NO dependence on any y or L2 norm in the threshold

**Key difference:** Paper's z-step threshold is scaled by 1/||y||_2, coupling it to
the y-step. Code's threshold is just λ/ρ with no such coupling.

### Dual updates

| Paper | Code |
|-------|------|
| v ← v + ρ(g - y) | (no v) |
| w ← w + ρ(g - z) | u ← u + g - z |

### Order of operations within each epoch

**Paper:** g-step → y-step (cubic) → z-step (threshold) → dual (v,w)
**Code:** g-step → threshold → cubic → z = τ·v_shrunk → dual (u)

The code reverses the order: threshold first, then cubic.
The paper does cubic first (on d=g+v/ρ), then threshold (on g+w/ρ with ||y||_2 scaling).

---

## Which Cubic is Correct?

### Paper's cubic derivation (τ³ - τ² - D = 0):
From min_y c/(N||y||_2) + (ρ/2)||y-d||²:
- Set y = τd, substitute:  c/(Nτ||d||_2) + (ρ/2)(τ-1)²||d||²
- Differentiate w.r.t. τ:  -c/(Nτ²||d||_2) + ρ(τ-1)||d||² = 0
- Multiply by τ²/(ρ||d||²):  τ³ - τ² - c/(Nρ||d||³) = 0
- So D = c/(Nρ||d||³)  →  **τ³ - τ² - D = 0** ✓ (paper is correct for its subproblem)

### Code's cubic (τ³ - τ - D = 0):
The code solves a DIFFERENT optimization problem. Let me derive what it actually solves.

If we write z = τ·v_shrunk and want to minimize:
  ||z||_1/||z||_2 + (ρ/2)||z - v_shrunk||²

Since z = τ·v_shrunk:
  |τ|·||v_shrunk||_1 / (|τ|·||v_shrunk||_2) + (ρ/2)(τ-1)²||v_shrunk||²
= ||v_shrunk||_1/||v_shrunk||_2 + (ρ/2)(τ-1)²||v_shrunk||²

The ratio norm is SCALE INVARIANT, so minimizing over τ just gives τ=1.
That can't be right — the code clearly gets τ≠1.

**Alternative derivation:** The code may be solving a different problem where the
L1 and L2 norms are weighted differently (by score). Need to check more carefully.

Looking at the code's D_k formula:
  D_k = (C · ||v_shrunk||_1 · score²) / (N · ρ · ||v_shrunk||_2³)

This is per-element (score² is a vector), so each feature gets a different τ.
This is NOT the standard Ratio Norm proximal at all — it's a custom heuristic
that uses the Ratio Norm's structure but applies it element-wise via importance scores.

---

## What the Code Actually Does (Honest Description)

The code implements a 2-variable ADMM where the z-step uses a **heuristic
decomposition** of the Ratio Norm proximal:

1. Soft-threshold v = S(g+u, λ/ρ) — sparsifies (L1-like)
2. Cubic rescale z = τ·v where τ_j solves τ³-τ-D_j=0 — adjusts scale per-feature

This is NOT:
- The true proximal of L1/L2 (which requires sorting + bisection)
- The paper's 3-variable ADMM (different cubic, different inputs, different order)
- A standard algorithm from the literature

It IS:
- An effective heuristic that works well empirically
- Inspired by the Ratio Norm structure
- A valid ADMM framework (the g-step and dual update are standard)

---

## Recommendations for Paper Rewrite

### Option A: Describe what the code does honestly (RECOMMENDED)
- Present as 2-variable ADMM: min L(θ,g) + λR(z) s.t. g=z
- Describe z-step as "approximate Ratio Norm proximal via sequential decomposition"
- Derive the cubic from the actual code's optimization (need to figure out what
  optimization problem the code's cubic corresponds to)
- Be upfront: "Computing the exact proximal of L1/L2 requires O(n log n) sorting
  and iterative bisection (Tao & Lou 2021). We use an efficient approximation..."

### Option B: Change the code to match the paper's 3-variable formulation
- Add y variable and second dual v
- Use paper's cubic τ³-τ²-D=0 for y-step
- Use paper's z-step with ||y||_2 scaling in threshold
- Risk: may change experimental results

### Option C: Keep 3-variable description but fix the cubic
- Keep the paper's structure but acknowledge the code uses a simplified variant
- Fix the cubic to match whichever is actually correct
- Risk: reviewers may ask "why not just use 2-variable?"

**Recommendation: Option A.** Describe what the code does. The results are good.
The method works. Just be honest about the algorithm.

---

## Key Literature

- Tao & Lou (2021). "Minimization of L1/L2 for Sparse Signal Recovery."
  SIAM J. Sci. Comput. — True proximal of L1/L2 requires sorting + bisection.
- Wang et al. (2021). "Unified Analysis on L1/L2 Minimization." arXiv:2108.01269
  — ADMM for L1/L2 with proximal operator algorithm.
- Boyd et al. (2011). "Distributed Optimization and Statistical Learning via ADMM."
  — Standard 2-variable ADMM reference.

---

## Cubic Derivation Analysis

### Paper's cubic is correct for its subproblem

The y-subproblem: min_y  c/(N||y||_2) + (ρ/2)||y - d||²

With y = τd:
  f(τ) = c/(Nτ||d||_2) + (ρ||d||²/2)(τ-1)²
  f'(τ) = -c/(Nτ²||d||_2) + ρ(τ-1)||d||² = 0

Multiply by τ²/(ρ||d||²):
  τ³ - τ² - c/(Nρ||d||³) = 0
  → **τ³ - τ² - D = 0** where D = c/(Nρ||d||³)  ✓

### Code's cubic does NOT correspond to any obvious proximal problem

The depressed cubic τ³ - τ - D = 0 is the first-order condition of:
  F(τ) = τ⁴/4 - τ²/2 - Dτ

I checked multiple candidate optimization problems:
- min c/(N||y||_2) + (ρ/2)||y||²  → gives τ³ = D (not τ³-τ-D)
- min c/(N||y||_2) + (ρ/2)(||y||²-||d||²)  → gives τ³ = D
- Various other penalty forms (||z||_1/||z||_2², etc.)  → all give τ³-τ²-D=0

The substitution τ = t + 1/3 transforms τ³-τ²-D=0 into:
  t³ - t/3 - (D + 2/27) = 0

This is NOT the same as τ³ - τ - D = 0 (which would need coefficient -1 on τ, not -1/3).

**Conclusion:** The code's cubic (τ³-τ-D=0) does not correspond to the paper's
y-subproblem, nor to any standard proximal problem I can identify. Most likely:
1. A derivation error (incorrect depressed-cubic transformation), OR
2. An intentional simplification that works well empirically

### Per-element τ is non-standard

The code computes D_k as a VECTOR (via score²), giving each feature a different τ_j.
The paper's formulation gives a SCALAR τ applied uniformly to all features.
Per-element scaling is fundamentally different from the paper's y = τd.

### The Ratio Norm is scale-invariant — implications

||τz||_1 / ||τz||_2 = ||z||_1 / ||z||_2 for any τ > 0.

This means you CANNOT optimize the Ratio Norm by scalar scaling alone.
The code's cubic rescale does not change the Ratio Norm value — it only
affects the quadratic penalty term. This is consistent with the paper's
y-step (which also just rescales d), but the coupling to the z-step
(via ||y||_2 in the threshold) is what makes the 3-variable approach work.

The code LACKS this coupling: the soft-threshold uses λ/ρ (no ||y||_2 factor).

---

## Summary of All Mismatches

| # | Aspect | Paper | Code | Severity |
|---|--------|-------|------|----------|
| 1 | Variables | g, y, z (3-var) | g, z (2-var) | HIGH |
| 2 | Duals | v, w (two) | u (one) | HIGH |
| 3 | Cubic | τ³-τ²-D=0 | τ³-τ-D=0 | HIGH |
| 4 | Order | cubic → threshold | threshold → cubic | MEDIUM |
| 5 | Threshold scaling | λ/(Nρ||y||_2) | λ/ρ | MEDIUM |
| 6 | τ scope | scalar (all features) | per-element (via score²) | MEDIUM |
| 7 | g-step penalties | 2 quadratic terms | 1 quadratic term | MEDIUM |
| 8 | D formula | c/(Nρ||d||³) | C·||v||₁·s²/(Nρ||v||₂³) | MEDIUM |

---

## Recommendations for Paper Rewrite (Updated)

### Option A: Describe what the code does honestly (RECOMMENDED)
- Present as 2-variable ADMM: min L(θ,g) + λR(z) s.t. g=z
- z-step: "approximate Ratio Norm proximal via sequential soft-threshold + cubic rescale"
- Derive the cubic honestly: present it as a heuristic scaling step that prevents
  the L2 norm from collapsing after soft-thresholding
- Cite the true proximal literature (Tao & Lou 2021) and explain why we use
  an approximation (O(1) per feature vs O(n log n) for exact proximal)
- The per-element score-weighted D_k is a novel contribution — present it as such

### Option B: Fix the code to match the paper's 3-variable formulation
- Add y variable and second dual v
- Use paper's cubic τ³-τ²-D=0 for y-step
- Use paper's z-step with ||y||_2 scaling in threshold
- Risk: changes experimental results, need to re-run everything

### Option C: Hybrid — keep 3-variable math, acknowledge code simplification
- Present the 3-variable formulation as the theoretical framework
- Note that the implementation uses a "collapsed" variant
- Fix the cubic in the paper to match the code (τ³-τ-D=0)
- Risk: reviewers will ask why not implement the full version

**Strong recommendation: Option A.** The results speak for themselves.
Describe the actual algorithm honestly. The 2-variable + heuristic proximal
is simpler, faster, and works. Don't pretend it's something it's not.

---

## Key Literature

- Tao & Lou (2021). "Minimization of L1/L2 for Sparse Signal Recovery."
  SIAM J. Sci. Comput. — True proximal of L1/L2 requires sorting + bisection.
- Wang et al. (2021). "Unified Analysis on L1/L2 Minimization." arXiv:2108.01269
  — ADMM for L1/L2 with proximal operator algorithm.
- Boyd et al. (2011). "Distributed Optimization and Statistical Learning via ADMM."
  — Standard 2-variable ADMM reference.
