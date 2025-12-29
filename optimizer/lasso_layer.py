from torch.optim import Optimizer
import torch
def soft_thresholding(b, u):
    return torch.sign(b) * torch.max(torch.zeros_like(b), torch.abs(b) - u)


class Lasso_layer(Optimizer):

    def __init__(self, params, lr, N, C, vk, zk, score):
        self.lr = lr
        self.N = N  # NUMBER OF SAMPLE
        self.C = C  # CONSTANT
        self.vk = vk
        self.zk = zk
        self.score = score
        super(Lasso_layer, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        for group in self.param_groups:
            for w, vk_temp, zk_temp, wanda_temp in zip(group['params'], self.vk, self.zk, self.score):
                grad = w.grad
                if grad is None:
                    continue
                lr = self.lr
                score_safe = wanda_temp + 1e-8
                
                # Gradient descent step
                v = w - lr * grad
                
                # Soft thresholding with score-adjusted threshold
                # Higher score = more important = lower threshold (keep)
                base_thresh = lr * self.C * 0.01  # C controls sparsity
                thresh = base_thresh / score_safe
                thresh = torch.clamp(thresh, min=1e-6, max=0.1)
                
                new_w = soft_thresholding(v, thresh)
                w.copy_(new_w)
                '''
                以下内容是sparse group lasso的代码

                # v = w - lr * grad
                v = w - self.lr * grad

                # 【步骤2】元素级软阈值（L1 部分），按元素门槛为 lr * C1 * |wanda_temp|
                tilde_w = soft_thresholding(v, self.lr * self.C/self.N * torch.abs(wanda_temp))

                # 【步骤3】组级软阈值（Group Lasso 部分）：
                # 将整个参数张量看作一个组
                norm_group = torch.norm(tilde_w * wanda_temp, p=2)
                threshold = self.lr * self.C/self.N
                if norm_group <= threshold:
                    new_w = torch.zeros_like(tilde_w)
                else:
                    new_w = (1 - threshold / norm_group) * tilde_w

                # 更新辅助变量 zk_temp（保存更新结果），再将结果写回参数 w
                zk_temp.copy_(new_w)
                w.copy_(zk_temp)
                '''
        return loss


    __all__ = ["Lasso_layer", "soft_thresholding"]