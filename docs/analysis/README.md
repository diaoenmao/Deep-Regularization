# Analysis 索引

**最后更新**: 2026-04-10

---

## 核心文档

| 文档 | 内容 | 状态 |
|------|------|------|
| [EXPERIMENTS_SUMMARY.md](../EXPERIMENTS_SUMMARY.md) | **主文档**: 所有实验总结 | ✅ 当前 |
| [current_results_report_20260403.md](current_results_report_20260403.md) | 详细结果报告 | ✅ 完整 |
| [fairness_protocol_20260403.md](fairness_protocol_20260403.md) | 公平性比较协议 | ✅ 定义 |

---

## 实验状态

| 文档 | 内容 |
|------|------|
| [experiment_status_20260403.md](experiment_status_20260403.md) | 完成进度追踪 |
| [todo_experiments_summary_20260408.md](todo_experiments_summary_20260408.md) | 消融实验总结 |

---

## 消融分析

| 文档 | 内容 | 结论 |
|------|------|------|
| [transformer_pretrain_negative_result.md](transformer_pretrain_negative_result.md) | Transformer 预训练 | ❌ Negative result |
| [gating_fix_rerun_20260330.md](gating_fix_rerun_20260330.md) | Gating 修复 | Linear > Sigmoid |
| [training_order_full_summary_20260330.md](training_order_full_summary_20260330.md) | 训练顺序 | Select-then-MLP 最优 |

---

## 设计文档

| 文档 | 内容 |
|------|------|
| [todo_implementation_designs_20260403.md](todo_implementation_designs_20260403.md) | 实现设计 |
| [mentor_experiments_integrated_20260330.md](mentor_experiments_integrated_20260330.md) | 实验集成 |
| [mentor_method_pseudocode_audit_20260330.md](mentor_method_pseudocode_audit_20260330.md) | 方法审计 |

---

## 其他

| 文档 | 内容 |
|------|------|
| [venue_strategy_20260326.md](venue_strategy_20260326.md) | 投稿策略 |
| [todo_experiments_issues_20260407.md](todo_experiments_issues_20260407.md) | 待解决问题 |

---

## 快速参考

### 关键结论

1. **SADMM-FS 是最优方法**: best-k 0.6278 > 所有 baseline
2. **Linear gate > Sigmoid gate**: +3-8% best-k
3. **Gradual ADMM 改善 Ring**: +33%
4. **Transformer 是 negative result**: best-k 0.33 vs 1.00

### 推荐配置

```python
{
    "backbone": "MLP",
    "gate": "linear_unbounded",
    "feat_drop": 0.6,
    "optimizer": "adagrad",
    "lr": 0.00176,
}
```

---

*索引生成: 2026-04-10*