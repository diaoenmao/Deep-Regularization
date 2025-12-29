import torch
import torch.nn as nn
import torch.nn.functional as F

class WANDA_ScoreCalculator:
    def __init__(self, model):

        self.model = model
        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                hook = module.register_forward_hook(self._get_activation(name))
                self.hooks.append(hook)

    def _get_activation(self, name):


        def hook(module, input, output):
            self.activations[name] = input[0].detach()

        return hook

    def remove_hooks(self):

        for hook in self.hooks:
            hook.remove()

    def compute_wanda_scores(self):
        """
        计算所有目标层的 WANDA Scores。

        返回:
            一个字典，字典内是按顺序摆放的每一层参数的wandascore。
        """
        wanda_scores = {} #初始化字典
        for name, activation in self.activations.items():#找出各个层的activation以及其对于层名
            layer = dict(self.model.named_modules())[name]#定位该层内容
            if isinstance(layer, nn.Conv2d):
                layer_scores = self._compute_conv_wanda_score(layer, activation)
            elif isinstance(layer, nn.Linear):
                layer_scores = self._compute_linear_wanda_score(layer, activation)
            elif isinstance(layer, nn.BatchNorm2d):
                layer_scores = self._compute_batchnorm_wanda_score(layer, activation)
            else:
                continue
            # 将每个参数的 WANDA Score 存入字典，键为完整的参数名称
            for param_name, score in layer_scores.items():
                full_param_name = name + '.' + param_name#参数名字
                wanda_scores[full_param_name] = score#参数对应的wanda score
        return wanda_scores



    def _compute_conv_wanda_score(self, layer, activation):
        weights = layer.weight.data  # [out_channels, in_channels, kH, kW]
        bias = layer.bias.data  # [out_channels]

        out_channels, in_channels, kH, kW = weights.shape
        stride = layer.stride
        padding = layer.padding
        dilation = layer.dilation
        kernel_size = layer.kernel_size

        batch_size, in_channels_act, H, W = activation.shape
        assert in_channels_act == in_channels, "输入通道数与卷积层不匹配"

        # 使用 unfold 提取输入激活窗口
        unfolded = F.unfold(activation, kernel_size=kernel_size, dilation=dilation, padding=padding,
                            stride=stride)
        # unfolded shape: [batch_size, in_channels * kH * kW, L]
        # L 是滑动窗口的数量，即输出特征图的空间位置数量

        # 重新形状为 [batch_size, in_channels, kH, kW, L]
        L = unfolded.shape[-1]
        unfolded = unfolded.view(batch_size, in_channels, kH, kW, L)
        total_element = batch_size * L

        # 计算每个权重对应的输入激活的 L2 范数
        # l2_norm shape: [in_channels, kH, kW]
        l2_norm = torch.norm(unfolded, p=2, dim=(0,4)) / total_element # 对 batch 求 L2 范数


        # 计算权重的 WANDA Scores矩阵，即activation矩阵
        weight_scores = torch.ones_like(weights) * l2_norm.unsqueeze(0)  # Broadcasting over out_channels, GET ALL WINDOWS PARAMETERS

        # 计算偏置的 WANDA Scores矩阵，即activation矩阵
        activation_number = batch_size * H * W * in_channels
        activation_norm = torch.norm(activation, p = 2, dim=(0,1, 2, 3)) / activation_number # [in_channels]
        bias_scores = torch.ones_like(bias) * activation_norm

        # 合并到一起

        scores = {
            'weight': weight_scores,
            'bias': bias_scores
        }
        return scores

    def _compute_linear_wanda_score(self, layer, activation):
        """
        同上内容
        """
        weights = layer.weight.data  # [out_features, in_features]
        bias = layer.bias.data  # [out_features]

        out_features, in_features = weights.shape
        batch_size, in_features_act = activation.shape
        assert in_features_act == in_features, "输入特征数与全连接层不匹配"

        # 计算每个输入特征的 L2 范数 across the batch
        l2_norm = torch.norm(activation, p=2, dim=0) / batch_size # Shape: [in_features]
        a = l2_norm.unsqueeze(0)
        # 计算权重的 WANDA Scores
        weight_scores = torch.ones_like(weights) * l2_norm.unsqueeze(0)  # Broadcasting multiplication

        # 计算偏置的 WANDA Scores
        l2_norm = torch.norm(activation, p=2, dim=(0,1)) / (batch_size * in_features_act)

        bias_scores = torch.ones_like(bias) * l2_norm  # 标量

        # 保存为字典

        scores = {
            'weight': weight_scores,
            'bias': bias_scores
        }
        return scores

    def _compute_batchnorm_wanda_score(self, layer, activation):
        """
        同上，该层用于计算batchnorm的score
        """
        # 获取权重和偏置
        weight = layer.weight.data  # [num_features]
        bias = layer.bias.data      # [num_features]
        batch_size, in_channels_act, H, W = activation.shape
        # 计算每个通道的激活的 L2 范数
        l2_norm = torch.norm(activation, p=2, dim=(0, 2, 3)) / (batch_size * H * W) # [num_features]

        # 计算权重的 WANDA Scores
        weight_scores = torch.ones_like(weight) * l2_norm

        bias_scores = weight_scores
        scores = {
            'weight': weight_scores,
            'bias': bias_scores
        }

        return scores