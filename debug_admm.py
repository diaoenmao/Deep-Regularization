"""Debug script to investigate ADMM implementation issues."""

import torch
import numpy as np

# Import the cubic solver
import sys
sys.path.insert(0, '.')
from optimizer.utils import solve_cubic_ratio_norm, solve_cubic_depressed, safe_cbrt

def test_cubic_solver():
    """Test the cubic solver for correctness."""
    print("=" * 60)
    print("Testing Cubic Solver: tau^3 - tau - D_k = 0")
    print("=" * 60)

    # Test values
    test_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 0.01, 0.001]

    print(f"\n{'D_k':>10} | {'tau (solver)':>12} | {'tau^3-tau-D_k':>15} | {'Valid?':>8}")
    print("-" * 55)

    for D_k in test_values:
        D_k_tensor = torch.tensor([D_k])
        tau = solve_cubic_ratio_norm(D_k_tensor).item()
        residual = tau**3 - tau - D_k
        valid = abs(residual) < 1e-6
        print(f"{D_k:>10.4f} | {tau:>12.6f} | {residual:>15.10f} | {'OK' if valid else 'FAIL':>8}")

    # Test with negative D_k (shouldn't happen but let's check)
    print("\nTesting negative D_k values:")
    for D_k in [-0.1, -1.0, -5.0]:
        D_k_tensor = torch.tensor([D_k])
        tau = solve_cubic_ratio_norm(D_k_tensor).item()
        residual = tau**3 - tau - D_k
        print(f"D_k={D_k:>6.2f}: tau={tau:.6f}, residual={residual:.10f}")

    # Test the constraint tau >= 1
    print("\nChecking tau >= 1 constraint:")
    D_k_small = torch.tensor([0.001, 0.01, 0.1])
    tau_small = solve_cubic_ratio_norm(D_k_small)
    print(f"D_k = {D_k_small.tolist()}")
    print(f"tau = {tau_small.tolist()}")
    print(f"All tau >= 1? {(tau_small >= 1.0).all().item()}")


def analyze_threshold_differences():
    """Compare threshold calculations between ADMM and Lasso."""
    print("\n" + "=" * 60)
    print("Analyzing Threshold Differences: ADMM vs Lasso")
    print("=" * 60)

    # Typical values
    lr = 0.002
    C = 0.02
    weight_norm = 10.0  # typical weight norm
    score = torch.tensor([1.0, 0.5, 0.1, 0.01])  # varying importance

    # Lasso threshold: lr * C * 0.01 / score
    lasso_base = lr * C * 0.01
    lasso_thresh = lasso_base / score
    lasso_thresh = torch.clamp(lasso_thresh, min=1e-6, max=0.1)

    # ADMM threshold: C * 0.01 / weight_norm (NOT scaled by lr!)
    admm_base = C * 0.01
    admm_thresh = admm_base / weight_norm
    admm_thresh = torch.clamp(torch.tensor([admm_thresh]), min=1e-6, max=0.5)

    print(f"\nParameters: lr={lr}, C={C}, weight_norm={weight_norm}")
    print(f"\nLasso base threshold: lr * C * 0.01 = {lasso_base:.6f}")
    print(f"ADMM base threshold:  C * 0.01 = {admm_base:.6f}")
    print(f"\nRatio (ADMM/Lasso base): {admm_base/lasso_base:.1f}x")

    print(f"\nLasso thresholds (per score): {lasso_thresh.tolist()}")
    print(f"ADMM threshold (global):      {admm_thresh.item():.6f}")

    print("\n*** KEY FINDING ***")
    print(f"ADMM threshold is {admm_base/lasso_base:.0f}x larger than Lasso base!")
    print("This means ADMM prunes MORE aggressively at the same C value.")


def analyze_qk_update():
    """Analyze the q_k update equation in ADMM."""
    print("\n" + "=" * 60)
    print("Analyzing q_k Update Equation")
    print("=" * 60)

    # The ADMM q_k update:
    # qk = 0.5 * (yk + zk - vk/p - wk/p) / score - grad / (score * p * 2)

    # At initialization: yk=0, zk=weights, vk=0, wk=0
    # So: qk = 0.5 * zk / score - grad / (score * p * 2)
    #       = (zk - grad/p) / (2 * score)

    # Compare to standard gradient descent:
    # w_new = w - lr * grad = w - grad/p

    print("\nAt initialization (yk=0, vk=0, wk=0, zk=weights):")
    print("  qk = 0.5 * zk / score - grad / (score * p * 2)")
    print("     = (zk - grad/p) / (2 * score)")
    print("     = (weights - lr*grad) / (2 * score)")
    print("\nThis divides by score, which REDUCES the update for high-importance weights!")
    print("But we want to PRESERVE high-importance weights, not reduce their updates.")

    print("\n*** POTENTIAL BUG ***")
    print("The division by score in q_k may be inverting the intended behavior.")
    print("High score (important) -> smaller q_k -> smaller update")
    print("Low score (unimportant) -> larger q_k -> larger update")
    print("This seems backwards for pruning!")


def analyze_yk_update():
    """Analyze the y_k update and its effect."""
    print("\n" + "=" * 60)
    print("Analyzing y_k Update (Ratio Norm Proximal)")
    print("=" * 60)

    # y_k = tau * dk where dk = score * qk + vk/p
    # tau solves tau^3 - tau - D_k = 0
    # D_k = (mu * score^2) / (p * eta^3)
    # mu = C * ck / N
    # eta = ||score * dk||_2

    print("\ny_k update: y_k = tau * dk")
    print("where dk = score * qk + vk/p")
    print("and tau solves tau^3 - tau - D_k = 0")
    print("\nD_k = (mu * score^2) / (p * eta^3)")
    print("mu = C * ck / N")
    print("eta = ||score * dk||_2")

    # Simulate typical values
    C = 0.02
    N = 60000
    lr = 0.002
    p_scale = 1.0 / lr  # = 500

    # Typical score and dk values
    score = torch.tensor([1.0])
    dk = torch.tensor([0.1])
    zk = torch.tensor([0.05])

    ck = torch.norm(score * zk, p=1).item()
    yita = torch.norm(score * dk, p=2).item() + 1e-8
    miu = C * ck / N
    D_k = (miu * score.item()**2) / (p_scale * yita**3)

    print(f"\nTypical values (C={C}, N={N}, lr={lr}):")
    print(f"  p_scale = 1/lr = {p_scale}")
    print(f"  ck = ||score * zk||_1 = {ck:.6f}")
    print(f"  eta = ||score * dk||_2 = {yita:.6f}")
    print(f"  mu = C * ck / N = {miu:.10f}")
    print(f"  D_k = {D_k:.10f}")

    tau = solve_cubic_ratio_norm(torch.tensor([D_k])).item()
    print(f"  tau = {tau:.6f}")

    print(f"\nSince D_k is very small ({D_k:.2e}), tau is clamped to 1.0")
    print("This means y_k = dk, so the Ratio Norm has minimal effect!")

    print("\n*** KEY INSIGHT ***")
    print("With typical values, D_k is extremely small (< 1e-8)")
    print("This makes tau = 1 (clamped), so y_k = dk")
    print("The Ratio Norm regularization has almost no effect!")


def analyze_dual_variable_accumulation():
    """Analyze how dual variables accumulate over iterations."""
    print("\n" + "=" * 60)
    print("Analyzing Dual Variable Accumulation")
    print("=" * 60)

    # Dual updates:
    # vk += p * (score * qk - yk)
    # wk += p * (score * qk - zk)

    # If score * qk ~ yk and score * qk ~ zk, dual variables stay small
    # But if there's a mismatch, they accumulate

    print("\nDual variable updates:")
    print("  vk += p * (score * qk - yk)")
    print("  wk += p * (score * qk - zk)")
    print("\nWith p = 1/lr = 500 (for lr=0.002), even small mismatches")
    print("can cause large dual variable accumulation.")

    # Simulate accumulation
    lr = 0.002
    p_scale = 1.0 / lr

    # Small mismatch
    mismatch = 0.001
    vk_after_1 = p_scale * mismatch
    vk_after_10 = 10 * p_scale * mismatch
    vk_after_100 = 100 * p_scale * mismatch

    print(f"\nWith mismatch = {mismatch} per iteration:")
    print(f"  After 1 iter:   vk = {vk_after_1:.2f}")
    print(f"  After 10 iter:  vk = {vk_after_10:.2f}")
    print(f"  After 100 iter: vk = {vk_after_100:.2f}")

    print("\n*** POTENTIAL ISSUE ***")
    print("Dual variables can grow very large, destabilizing the optimization.")


def compare_update_magnitudes():
    """Compare the magnitude of updates in ADMM vs Lasso."""
    print("\n" + "=" * 60)
    print("Comparing Update Magnitudes: ADMM vs Lasso")
    print("=" * 60)

    # Simulate one step
    lr = 0.002
    C = 0.02

    # Initial weights and gradient
    w = torch.tensor([0.5, 0.3, 0.1, 0.05, 0.01])
    grad = torch.tensor([0.1, 0.08, 0.05, 0.02, 0.01])
    score = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.1])

    # Lasso update
    v_lasso = w - lr * grad
    lasso_thresh = lr * C * 0.01 / (score + 1e-8)
    lasso_thresh = torch.clamp(lasso_thresh, min=1e-6, max=0.1)
    w_lasso = torch.sign(v_lasso) * torch.clamp(torch.abs(v_lasso) - lasso_thresh, min=0)

    # ADMM update (simplified, first iteration)
    p_scale = 1.0 / lr
    score_safe = score + 1e-8

    # At first iteration: yk=0, vk=0, wk=0, zk=w
    qk = 0.5 * w / score_safe - grad / (score_safe * p_scale * 2.0)

    # z_k update
    weight_scale = torch.norm(w)
    admm_thresh = torch.clamp(torch.tensor([C * 0.01 / weight_scale]), min=1e-6, max=0.5)
    update_val = score_safe * qk  # wk/p = 0 at first iter
    w_admm = torch.sign(update_val) * torch.clamp(torch.abs(update_val) - admm_thresh, min=0)

    print(f"Initial weights: {w.tolist()}")
    print(f"Gradients:       {grad.tolist()}")
    print(f"Scores:          {score.tolist()}")
    print(f"\nLasso threshold: {lasso_thresh.tolist()}")
    print(f"ADMM threshold:  {admm_thresh.item():.6f}")
    print(f"\nAfter Lasso:     {w_lasso.tolist()}")
    print(f"After ADMM:      {w_admm.tolist()}")

    print(f"\nLasso sparsity:  {(w_lasso == 0).sum().item()}/{len(w)}")
    print(f"ADMM sparsity:   {(w_admm == 0).sum().item()}/{len(w)}")


def main():
    print("ADMM Implementation Debug Analysis")
    print("=" * 60)

    test_cubic_solver()
    analyze_threshold_differences()
    analyze_qk_update()
    analyze_yk_update()
    analyze_dual_variable_accumulation()
    compare_update_magnitudes()

    print("\n" + "=" * 60)
    print("SUMMARY OF POTENTIAL ISSUES")
    print("=" * 60)
    print("""
1. THRESHOLD SCALING: ADMM threshold is NOT scaled by lr, making it
   ~500x larger than Lasso at the same C value. This causes more
   aggressive pruning in ADMM.

2. SCORE DIVISION IN q_k: The q_k update divides by score, which may
   invert the intended importance weighting. High-importance weights
   get smaller updates, low-importance get larger updates.

3. RATIO NORM INEFFECTIVE: With typical values, D_k is extremely small
   (< 1e-8), making tau = 1 (clamped). The Ratio Norm regularization
   has almost no effect on the optimization.

4. DUAL VARIABLE ACCUMULATION: With p = 1/lr = 500, dual variables
   can accumulate rapidly, potentially destabilizing optimization.

RECOMMENDATIONS:
- Scale ADMM threshold by lr to match Lasso behavior
- Review the q_k update equation for correct score weighting
- Consider increasing C or adjusting D_k computation for Ratio Norm effect
- Add dual variable damping or reset mechanism
""")


if __name__ == "__main__":
    main()
