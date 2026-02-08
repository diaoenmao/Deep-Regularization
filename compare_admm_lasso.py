"""Extract ADMM vs Lasso comparison from existing full experiment results."""
import json

with open("results/metrics/full_experiment_20260205_081150.json") as f:
    data = json.load(f)

# Build lookup: (class, score) -> entry
lookup = {}
for entry in data:
    lookup[(entry["class_name"], entry["score_name"])] = entry

# Compare ADMM vs Lasso at same C values
# ADMM uses C values: 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06
# Lasso uses C values: need to check

print("=" * 90)
print("FULL EXPERIMENT RESULTS: ADMM vs Lasso (3 epochs, full training)")
print("=" * 90)

for scope in ["Global", "Layer", "Neuron"]:
    admm_key = f"ADMM_Adam_{scope}"
    lasso_key = f"Lasso_Adam_{scope}"
    
    print(f"\n{'='*90}")
    print(f"  SCOPE: {scope}")
    print(f"{'='*90}")
    
    for score in ["Magnitude", "First_Order", "Second_Order", "First_Order+Second_Order"]:
        admm = lookup.get((admm_key, score))
        lasso = lookup.get((lasso_key, score))
        
        if not admm or not lasso:
            continue
        
        print(f"\n  Score: {score}")
        print(f"  {'Method':<22} {'C':>8} {'Accuracy':>10} {'Sparsity':>10} {'Remaining%':>12}")
        print(f"  {'-'*65}")
        
        # Show ADMM results
        for i, c in enumerate(admm["C"]):
            acc = admm["accuracy"][i]
            rem = admm["remaining_weights"][i]
            sp = (1 - rem) * 100
            print(f"  {'ADMM_'+scope:<22} {c:>8.3f} {acc:>9.2f}% {sp:>9.1f}% {rem*100:>11.2f}%")
        
        print()
        # Show Lasso results
        for i, c in enumerate(lasso["C"]):
            acc = lasso["accuracy"][i]
            rem = lasso["remaining_weights"][i]
            sp = (1 - rem) * 100
            print(f"  {'Lasso_'+scope:<22} {c:>8.3f} {acc:>9.2f}% {sp:>9.1f}% {rem*100:>11.2f}%")

# Summary: best results at similar sparsity levels
print(f"\n{'='*90}")
print("DIRECT COMPARISON: Best accuracy at high sparsity (>90%)")
print(f"{'='*90}")
print(f"{'Method':<28} {'Score':<25} {'C':>6} {'Accuracy':>10} {'Sparsity':>10}")
print("-" * 85)

results = []
for entry in data:
    for i, c in enumerate(entry["C"]):
        acc = entry["accuracy"][i]
        rem = entry["remaining_weights"][i]
        sp = (1 - rem) * 100
        if sp > 90 and acc > 50:
            results.append((entry["class_name"], entry["score_name"], c, acc, sp))

results.sort(key=lambda x: -x[3])  # sort by accuracy desc
for name, score, c, acc, sp in results[:20]:
    print(f"{name:<28} {score:<25} {c:>6.3f} {acc:>9.2f}% {sp:>9.1f}%")

print(f"\n{'='*90}")
print("DIRECT COMPARISON: Best accuracy at moderate sparsity (50-90%)")
print(f"{'='*90}")
print(f"{'Method':<28} {'Score':<25} {'C':>6} {'Accuracy':>10} {'Sparsity':>10}")
print("-" * 85)

results2 = []
for entry in data:
    for i, c in enumerate(entry["C"]):
        acc = entry["accuracy"][i]
        rem = entry["remaining_weights"][i]
        sp = (1 - rem) * 100
        if 50 <= sp <= 90 and acc > 50:
            results2.append((entry["class_name"], entry["score_name"], c, acc, sp))

results2.sort(key=lambda x: -x[3])
for name, score, c, acc, sp in results2[:20]:
    print(f"{name:<28} {score:<25} {c:>6.3f} {acc:>9.2f}% {sp:>9.1f}%")
