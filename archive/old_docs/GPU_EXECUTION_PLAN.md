# GPU Execution Plan - RTX 3060 Ti (8GB VRAM)

**Generated**: 2026-04-05
**Target**: Complete all TODO ablation experiments

---

## Hardware Notes

- **GPU**: RTX 3060 Ti (8GB VRAM)
- **Constraint**: Limited VRAM for large batch sizes
- **Strategy**: Use smaller batch sizes (32-64), gradient accumulation if needed

---

## Experiments Overview

| # | Experiment | Script | Est. Time | Priority |
|---|------------|--------|-----------|----------|
| 1 | Polynomial Ablation | `run_polynomial_ablation.py` | ~15 min | HIGH |
| 2 | Iterative Ablation | `run_iterative_ablation.py` | ~20 min | MEDIUM |
| 3 | Transformer Pretrain | `run_transformer_pretrain_ablation.py` | ~30 min | LOW |

**Total estimated time**: ~65 minutes

---

## Phase 1: Polynomial Ablation (HIGH PRIORITY)

**Why first**: Already shows promising results (expanded-space achieves 1.0 best-k)

**Configuration**:
```bash
cd E:/Projects/NEW_Pruning_20251110/custom_admm
python run_polynomial_ablation.py --device cuda --full
```

**Expected output**:
- Compare: `group` vs `expanded` vs `hierarchical`
- Datasets: xor, ring, ring+xor (m=128-256)
- Degrees: 1, 2
- Seeds: 5

**Memory usage**: ~2-4 GB (polynomial expansion increases feature count)

---

## Phase 2: Iterative Ablation (MEDIUM PRIORITY)

**Why**: Test if iterative pruning improves over single-pass

**Configuration**:
```bash
python run_iterative_ablation.py --device cuda --full
```

**Expected output**:
- Compare: `single_pass` vs `iterative_hard` vs `gradual_admm`
- Datasets: xor, ring, ring+xor
- Seeds: 5

**Memory usage**: ~1-2 GB (MLP-based, moderate)

---

## Phase 3: Transformer Pretrain (LOW PRIORITY)

**Why**: Previous results show transformer backbone fails; this confirms

**Configuration**:
```bash
python run_transformer_pretrain_ablation.py --device cuda --full
```

**Expected output**:
- Compare: `mlp_baseline` vs `transformer_no_pretrain` vs `transformer_pretrain`
- Datasets: xor, ring
- Seeds: 5

**Memory usage**: ~2-3 GB (transformer attention)

---

## Execution Commands

### Option A: Run All Sequentially

```bash
# From E:/Projects/NEW_Pruning_20251110/custom_admm

# Phase 1
python run_polynomial_ablation.py --device cuda --full

# Phase 2
python run_iterative_ablation.py --device cuda --full

# Phase 3
python run_transformer_pretrain_ablation.py --device cuda --full
```

### Option B: Run in Background (Recommended)

```bash
# Create run script
cat > run_all_ablations.bat << 'EOF'
@echo off
cd /d E:\Projects\NEW_Pruning_20251110\custom_admm

echo ========================================
echo Phase 1: Polynomial Ablation
echo ========================================
python run_polynomial_ablation.py --device cuda --full

echo ========================================
echo Phase 2: Iterative Ablation
echo ========================================
python run_iterative_ablation.py --device cuda --full

echo ========================================
echo Phase 3: Transformer Pretrain Ablation
echo ========================================
python run_transformer_pretrain_ablation.py --device cuda --full

echo All experiments complete!
EOF

# Run
run_all_ablations.bat
```

### Option C: Quick Mode (15 min total)

```bash
# Quick test on all three
python run_polynomial_ablation.py --device cuda --quick
python run_iterative_ablation.py --device cuda --quick
python run_transformer_pretrain_ablation.py --device cuda --quick
```

---

## Expected Results Files

After completion, check results in:
```
custom_admm/results/
├── polynomial_ablation_YYYYMMDD_HHMMSS.json
├── iterative_ablation_YYYYMMDD_HHMMSS.json
└── transformer_pretrain_ablation_YYYYMMDD_HHMMSS.json
```

---

## Monitoring

```bash
# Check GPU usage
nvidia-smi -l 5

# Check running processes
nvidia-smi pmon -c 10
```

---

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size in scripts
batch_size = 32  # instead of 64
```

### Slow Training

```python
# Reduce epochs for quick test
epochs = 200  # instead of 400
```

### Import Errors

```bash
# Run from custom_admm directory
cd E:/Projects/NEW_Pruning_20251110/custom_admm
python run_xxx.py --device cuda
```

---

## Success Criteria

| Experiment | Success Metric |
|------------|----------------|
| Polynomial | expanded >= group on at least one dataset |
| Iterative | Any method > single_pass |
| Transformer | pretrain > no_pretrain |

---

## Next Steps After Results

1. **If polynomial expanded wins**: Promote expanded-space selection to main method
2. **If iterative wins**: Add iterative refinement to default pipeline
3. **If transformer pretrain fails**: Mark as confirmed negative result in paper