import torch


class GradientCollector:
    """
    该类用于计算各种剪枝Score，基于论文的泰勒展开框架：
    - Magnitude: |W| (权重绝对值)
    - First-Order: |∂L/∂W · W| (梯度×权重)
    - Second-Order: (1/2) Σ (∂L/∂W · W)² (Fisher近似的二阶项)
    """

    def __init__(self, model):
        self.model = model
        # 用于累积Fisher信息（二阶项）
        self.fisher_accumulator = {}
        self.fisher_count = 0

    def reset_fisher(self):
        """重置Fisher累积器，用于新的calibration数据集"""
        self.fisher_accumulator = {}
        self.fisher_count = 0

    def compute_gradients(self):
        """返回原始梯度"""
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.detach().clone()
            else:
                gradients[name] = torch.zeros_like(param)
        return gradients

    def compute_ones(self):
        """返回全1张量（用于baseline）"""
        ones_dict = {}
        for name, param in self.model.named_parameters():
            ones_dict[name] = torch.ones_like(param)
        return ones_dict

    def compute_magnitude(self):
        """
        Magnitude Score: |W|
        基于权重大小，Data-free方法
        """
        magnitude = {}
        for name, param in self.model.named_parameters():
            magnitude[name] = torch.abs(param.data.detach())
        return magnitude

    def compute_first_order(self):
        """
        First-Order Score: |∂L/∂W · W|
        基于泰勒展开的一阶项，衡量Loss对权重的一阶敏感度
        公式: I_1st = |∂L(D)/∂W · W|
        """
        first_order = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # 梯度 × 权重 的绝对值
                first_order[name] = torch.abs(param.grad.detach() * param.data.detach())
            else:
                # 如果没有梯度，返回0（不应该剪掉）
                first_order[name] = torch.zeros_like(param)
        return first_order

    def accumulate_fisher(self):
        """
        累积Fisher信息，用于计算Second-Order Score
        应该在每个mini-batch的backward之后调用
        """
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_weight = param.grad.detach() * param.data.detach()
                squared = grad_weight ** 2
                
                if name not in self.fisher_accumulator:
                    self.fisher_accumulator[name] = squared.clone()
                else:
                    self.fisher_accumulator[name] += squared
        self.fisher_count += 1

    def compute_second_order(self):
        """
        Second-Order Score: (1/2) * (1/N) * Σ (∂L/∂W · W)²
        基于Fisher信息矩阵对角线近似Hessian
        公式: I_2nd ≈ (1/2) Σ_j (∂L(D_j)/∂W · W)²
        """
        second_order = {}
        for name, param in self.model.named_parameters():
            if name in self.fisher_accumulator and self.fisher_count > 0:
                # 平均Fisher信息，乘以1/2
                second_order[name] = 0.5 * self.fisher_accumulator[name] / self.fisher_count
            else:
                # 如果没有累积的Fisher信息，用当前batch的近似
                if param.grad is not None:
                    grad_weight = param.grad.detach() * param.data.detach()
                    second_order[name] = 0.5 * (grad_weight ** 2)
                else:
                    second_order[name] = torch.zeros_like(param)
        return second_order

    def compute_first_plus_second_order(self):
        """
        Complete Taylor Expansion Score: |I_1st - I_2nd|
        公式: I ≈ |∂L/∂W · W - (1/2) Σ (∂L/∂W · W)²|
        """
        first_order = self.compute_first_order()
        second_order = self.compute_second_order()
        
        combined = {}
        for name in first_order:
            # 一阶项 - 二阶项，取绝对值
            combined[name] = torch.abs(first_order[name] - second_order[name])
        return combined