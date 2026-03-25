# File Cleanup Summary

**Date**: 2026-03-23
**Reason**: Remove temporary test/debug scripts after adaptive architecture implementation

---

## Files Deleted

### Test Scripts (8 files)
1. `test_bug_fixes.py` - One-time bug fix verification for cubic solver
2. `test_clean_integration.py` - Redundant integration test
3. `test_consistency.py` - One-time consistency check between old/new results
4. `test_fixed_implementation.py` - Redundant with test_small.py
5. `test_integration.py` - Integration test, served its purpose
6. `test_simple_integration.py` - Redundant with test_integration.py
7. `test_small.py` - Small test for admm_input_group verification
8. `test_fixed_implementation.py` - Redundant test

### Debug Scripts (1 file)
9. `debug_admm_convergence.py` - One-time ADMM convergence debugging

### Verification Scripts (3 files)
10. `quick_verify.py` - Quick 5-seed verification
11. `quick_verify_fixed.py` - Redundant with quick_verify.py
12. `replicate_admm_results.py` - One-time result replication script

### Comparison Scripts (3 files)
13. `compare_old_vs_new.py` - Old vs new architecture comparison
14. `compare_all_results.py` - Comprehensive result comparison
15. `merge_results.py` - Result merging utility

**Total**: 15 files deleted

---

## Files Kept

### Experiment Runners
- `run_ablations.py` - Ablation study runner
- `run_adaptive_comparison.py` - **NEW** Adaptive architecture comparison
- `run_admm_input_group.py` - ADMM input group benchmark
- `run_admm_input_group_benchmark.py` - ADMM benchmark
- `run_architecture_comparison.py` - Original architecture comparison
- `run_consistent_synthetic.py` - Consistent synthetic experiments
- `run_dropout_ablation_m128.py` - Dropout ablation
- `run_dropout_xor512.py` - Dropout ablation for XOR
- `run_efficient_synthetic.py` - Efficient synthetic experiments
- `run_epoch_sweep.py` - Epoch sweep experiments
- `run_fixed_benchmark.py` - Fixed benchmark
- `run_focused_synthetic.py` - Focused synthetic experiments
- `run_full_synthetic.py` - Full synthetic experiments
- `run_modern_admm_stg.py` - Modern ADMM/STG comparison
- `run_multiseed_dag.py` - Multi-seed DAG experiments
- `run_parallel_benchmark.py` - Parallel benchmark
- `run_penalty_ablation.py` - Penalty ablation
- `run_phase1_experiments.py` - Phase 1 experiments
- `run_realworld_admm.py` - Real-world ADMM experiments
- `run_single.py` - Single dataset runner
- `run_single_dataset.py` - Single dataset runner
- `run_single_dimension.py` - Single dimension runner
- `run_synthetic.py` - Synthetic experiments
- `run_xor_2v3.py` - XOR 2-var vs 3-var comparison

### Tuning Scripts
- `tune_extended.py` - Extended hyperparameter tuning
- `tune_feat_drop.py` - Feature dropout tuning

### Utility Scripts
- `parse_results.py` - Result parsing utility

---

## Rationale

### Why Delete These Files?

1. **Temporary Purpose**: Test scripts like `test_bug_fixes.py` and `debug_admm_convergence.py` served one-time verification purposes.

2. **Redundancy**: Multiple test scripts (`test_clean_integration.py`, `test_simple_integration.py`, etc.) performed similar functions.

3. **Results Preserved**: Comparison scripts (`compare_old_vs_new.py`, `merge_results.py`) have already saved their results to JSON files.

4. **Cleaner Repository**: Removing temporary files makes it easier to find active experiment scripts.

### Why Keep Experiment Runners?

All `run_*.py` files are kept because:
- They may be needed to reproduce results
- They serve as templates for future experiments
- They contain documented experimental protocols

---

## Impact

- **No loss of experimental data**: All results are saved in `results/` directory
- **No loss of functionality**: Active experiment scripts preserved
- **Improved clarity**: Easier to find relevant scripts
- **Reduced clutter**: 15 fewer temporary files

---

## Future Recommendations

1. **Use a `tests/` directory**: Move any new test scripts to a dedicated `tests/` folder
2. **Use a `scripts/` directory**: Move utility scripts like `parse_results.py` to `scripts/`
3. **Document experiment status**: Add comments to `run_*.py` files indicating which are actively used
