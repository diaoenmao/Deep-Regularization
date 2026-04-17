# SADMM-FS: Sparse ADMM Feature Selection

[![Status](https://img.shields.io/badge/Status-UAI%2026%20Submitted-blue)]()

A feature selection method combining global scalar gating with Linearized ADMM and Ratio Norm penalty.

## Quick Links

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Project guide for Claude Code |
| [EXPERIMENTS_SUMMARY.md](docs/EXPERIMENTS_SUMMARY.md) | Complete experiment results |
| [EXPERIMENT_PLAN_ITERATIVE_EXPANSION.md](docs/EXPERIMENT_PLAN_ITERATIVE_EXPANSION.md) | Iterative/Expansion experiment plan |
| [PAPER_RESULTS_TABLES.xlsx](results/tables/PAPER_RESULTS_TABLES.xlsx) | All results in paper-ready format |

## Directory Structure

```
SADMM-FS/
├── CLAUDE.md               # Project entry
├── README.md               # This file
├── src/                    # Core source code (24 files)
├── experiments/            # Experiment scripts
│   ├── main/               # Main benchmark (5 scripts)
│   ├── ablation/           # Ablation studies (5 scripts)
│   └── utils/              # Utilities (6 scripts)
├── results/                # Results
│   ├── tables/             # PAPER_RESULTS_TABLES.xlsx
│   ├── ablation/           # Ablation results (26 json)
│   └── archive/            # Archived results
├── docs/                   # Documentation
│   ├── EXPERIMENTS_SUMMARY.md
│   ├── analysis/           # Analysis reports (15 files)
│   └── legacy/             # Legacy docs (4 files)
├── data/                   # Datasets
├── vendor_pkgs/            # Baseline packages (STG, CAE, TabNet)
├── deep-reg-paper/         # Paper drafts
└── archive/                # All old files
```

## Key Results

| Method | Mean best-k | Notes |
|--------|-------------|-------|
| **SADMM-FS** | **0.6278** | Best overall |
| STG | 0.625 | Backbone only match |
| TabNet | 0.2851 | Standard impl |

**Real-World (NIPS 2003):**
| Dataset | SADMM-FS | STG |
|---------|----------|-----|
| madelon | 0.965 | 0.847 |
| gisette | 0.985 | 0.963 |
| arcene | 0.887 | 0.808 |
| dexter | 0.889 | 0.825 |

## Key Ablation Findings

- Linear gate > Sigmoid gate (+3%)
- MLP > Transformer backbone (0.63 vs 0.25)
- Gradual ADMM helps Ring (+33%)
- **NEW**: Soft pruning + no re-weighting achieves **perfect 1.00** on Ring!

## Quick Start

```bash
# Run main benchmark
python experiments/main/run_admm_input_group_benchmark.py

# Run ablation
python experiments/ablation/run_iterative_ablation.py

# Generate results table
python experiments/utils/create_paper_xlsx.py
```

## Citation

```bibtex
@article{sadmmfs2026,
  title={SADMM-FS: Sparse ADMM Feature Selection},
  author={...},
  journal={UAI 2026},
  year={2026}
}
```