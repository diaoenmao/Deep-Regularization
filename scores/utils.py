import torch

class GradientCollector:
    """
    该类用于计算各个参数对于自己的gradient，用于做lora prune
    """

    def __init__(self, model):
        self.model = model

    def compute_gradients(self):
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.detach()
            else:
                # 如果没有梯度，则返回一个与参数相同形状的全 1 张量
                gradients[name] = torch.ones_like(param)
        return gradients

    def compute_ones(self):
        ones_dict = {}
        for name, param in self.model.named_parameters():
            ones_dict[name] = torch.ones_like(param)
        return ones_dict