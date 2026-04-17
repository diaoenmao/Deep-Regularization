# 项目整理方案

## 当前问题

1. **custom_admm/** 混乱：
   - 41个.py脚本分散
   - 11个.md文档分散
   - 大量临时/测试文件

2. **analysis/** 分析报告分散

3. **根目录** TODO文档分散

## 新目录结构

```
NEW_Pruning_20251110/
├── CLAUDE.md                  # 项目入口文档
├── README.md                  # 项目README
├── docs/                      # 所有文档集中
│   ├── EXPERIMENTS_SUMMARY.md # 主实验总结
│   ├── EXPERIMENT_PLAN_*.md   # 实验计划
│   ├── analysis/              # 分析报告（从analysis/移入）
│   └── legacy/                # 过时文档
├── src/                       # 核心源代码（从custom_admm/src/移入）
│   ├── admm_input_group_wrapper.py
│   ├── gradual_admm_with_pruning.py
│   ├── data.py
│   └── ...
├── experiments/               # 实验脚本
│   ├── main/                  # 主实验脚本
│   ├── ablation/              # 消融实验脚本
│   ├── realworld/             # 真实数据实验
│   └── utils/                 # 工具脚本
├── results/                   # 结果数据
│   ├── tables/                # 汇总表格（xlsx）
│   ├── main/                  # 主实验结果
│   ├── ablation/              # 消融结果
│   └── archive/               # 旧结果备份
├── data/                      # 数据目录（保留）
├── vendor_pkgs/               # 第三方包（保留）
└── archive/                   # 整体归档（旧文件）
```

## 分类规则

### 文档分类
- **保留**: EXPERIMENTS_SUMMARY.md, EXPERIMENT_PLAN_*.md, CLAUDE.md
- **analysis/** → **docs/analysis/**
- **过时文档** → **docs/legacy/**

### 脚本分类
- **主实验**: run_admm_input_group_benchmark.py, run_synthetic.py
- **消融**: run_*_ablation.py, run_iterative_ablation.py
- **真实数据**: run_realworld*.py
- **工具**: create_paper_xlsx.py, parse_results.py, tune_*.py
- **临时/测试** → **archive/**

### 结果分类
- **汇总表格**: PAPER_RESULTS_TABLES.xlsx, ALL_METHODS_RESULTS_SUMMARY.xlsx
- **主结果**: merged_results.json, benchmark_with_tracking.json
- **消融结果**: custom_admm/results/*_ablation_*.json → results/ablation/
- **旧结果**: results_backup/, custom_admm/results/旧文件 → results/archive/

## 执行步骤

1. 创建新目录结构
2. 移动核心源代码 (src/)
3. 分类移动实验脚本 (experiments/)
4. 整理文档 (docs/)
5. 整理结果 (results/)
6. 归档旧文件 (archive/)
7. 更新CLAUDE.md索引