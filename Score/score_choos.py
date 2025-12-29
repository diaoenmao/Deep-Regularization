"""Score selection helpers used in pruning experiments.

基于论文的泰勒展开框架实现4种Score：
1. Magnitude (L2): |W| - 权重绝对值，Data-free基准方法
2. First-Order: |∂L/∂W · W| - 梯度×权重，一阶敏感度
3. Second-Order: (1/2)Σ(∂L/∂W · W)² - Fisher近似的二阶项
4. First+Second: |一阶项 - 二阶项| - 完整泰勒展开

另外保留Wanda方法作为对比（基于激活值的剪枝）
"""

import torch


def _normalize_name(score_name: str) -> str:
    return score_name.strip().lower()


def choose_score(wanda_calculator, gradient_collector, score_name: str):
    """Return a per-parameter score tensor dict.

    Accepted names (case-insensitive):
    - "magnitude": |W| - 权重绝对值 (Data-free baseline)
    - "first order": |∂L/∂W · W| - 梯度×权重 (一阶泰勒展开)
    - "second order": (1/2)Σ(∂L/∂W · W)² - Fisher近似 (二阶泰勒展开)
    - "first order + second order": |一阶 - 二阶| - 完整泰勒展开
    - "wanda": Wanda激活分数 (用于对比)
    """
    normalized = _normalize_name(score_name)

    # Magnitude: 只用权重大小，不需要梯度
    if normalized == "magnitude":
        return gradient_collector.compute_magnitude()

    # First-Order: 梯度 × 权重
    if normalized == "first order":
        return gradient_collector.compute_first_order()

    # Second-Order: Fisher近似的二阶项
    if normalized == "second order":
        return gradient_collector.compute_second_order()

    # First + Second Order: 完整泰勒展开
    if normalized == "first order + second order":
        return gradient_collector.compute_first_plus_second_order()

    # Wanda: 基于激活值的剪枝方法（保留用于对比）
    if normalized == "wanda":
        return wanda_calculator.compute_wanda_scores()

    raise ValueError(
        f"Unknown score name: {score_name}. "
        f"Expected one of: 'magnitude', 'first order', 'second order', "
        f"'first order + second order', 'wanda'."
    )


def normalize_scores(score_dict: dict) -> dict:
    """Normalize scores to [0, 1] range for comparability across score types.
    
    This ensures different score types (magnitude, first-order, second-order)
    have similar scales, so the same C/p parameters work across all types.
    """
    normalized = {}
    for name, score in score_dict.items():
        score_flat = score.flatten()
        min_val = score_flat.min()
        max_val = score_flat.max()
        
        if max_val - min_val > 1e-10:
            # Min-max normalization
            normalized[name] = (score - min_val) / (max_val - min_val)
        else:
            # All values are the same, use uniform score
            normalized[name] = torch.ones_like(score)
    return normalized


def choose_score_normalized(wanda_calculator, gradient_collector, score_name: str):
    """Return normalized per-parameter score tensor dict.
    
    Same as choose_score but with normalization applied.
    """
    raw_scores = choose_score(wanda_calculator, gradient_collector, score_name)
    return normalize_scores(raw_scores)