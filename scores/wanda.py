import torch
import torch.nn as nn
import torch.nn.functional as F

class WandaScoreCalculator:
    """
    Calculates WANDA scores for neural network parameters based on activation values.
    """
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture layer activations."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                hook = module.register_forward_hook(self._get_activation(name))
                self.hooks.append(hook)

    def _get_activation(self, name):
        """Hook function to store input activations."""
        def hook(module, input, output):
            self.activations[name] = input[0].detach()
        return hook

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()

    def compute_wanda_scores(self):
        """
        Compute WANDA scores for all target layers.
        
        Returns:
            dict: Dictionary mapping parameter names to their WANDA scores
        """
        wanda_scores = {}
        for name, activation in self.activations.items():
            layer = dict(self.model.named_modules())[name]
            
            if isinstance(layer, nn.Conv2d):
                layer_scores = self._compute_conv_scores(layer, activation)
            elif isinstance(layer, nn.Linear):
                layer_scores = self._compute_linear_scores(layer, activation)
            elif isinstance(layer, nn.BatchNorm2d):
                layer_scores = self._compute_batchnorm_scores(layer, activation)
            else:
                continue

            for param_name, score in layer_scores.items():
                full_param_name = f"{name}.{param_name}"
                wanda_scores[full_param_name] = score
                
        return wanda_scores

    def _compute_conv_scores(self, layer, activation):
        """Compute WANDA scores for convolutional layer."""
        weights = layer.weight.data
        bias = layer.bias.data if layer.bias is not None else None
        
        out_channels, in_channels, kH, kW = weights.shape
        batch_size, in_channels_act, H, W = activation.shape
        
        # Extract input windows using unfold
        unfolded = F.unfold(activation, 
                        kernel_size=layer.kernel_size,
                        dilation=layer.dilation,
                        padding=layer.padding,
                        stride=layer.stride)
        
        L = unfolded.shape[-1]
        unfolded = unfolded.view(batch_size, in_channels, kH, kW, L)
        total_elements = batch_size * L

        # Compute L2 norm of activations
        l2_norm = torch.norm(unfolded, p=2, dim=(0,4)) / total_elements

        # Compute weight scores
        weight_scores = torch.ones_like(weights) * l2_norm.unsqueeze(0)

        scores = {'weight': weight_scores}
        
        if bias is not None:
            activation_number = batch_size * H * W * in_channels
            activation_norm = torch.norm(activation, p=2, dim=(0,1,2,3)) / activation_number
            bias_scores = torch.ones_like(bias) * activation_norm
            scores['bias'] = bias_scores

        return scores

    def _compute_linear_scores(self, layer, activation):
        """Compute WANDA scores for linear layer."""
        weights = layer.weight.data
        bias = layer.bias.data if layer.bias is not None else None
        
        batch_size, in_features = activation.shape

        # Compute L2 norm across batch
        l2_norm = torch.norm(activation, p=2, dim=0) / batch_size
        
        # Compute weight scores
        weight_scores = torch.ones_like(weights) * l2_norm.unsqueeze(0)
        
        scores = {'weight': weight_scores}
        
        if bias is not None:
            l2_norm_bias = torch.norm(activation, p=2, dim=(0,1)) / (batch_size * in_features)
            bias_scores = torch.ones_like(bias) * l2_norm_bias
            scores['bias'] = bias_scores
            
        return scores

    def _compute_batchnorm_scores(self, layer, activation):
        """Compute WANDA scores for batch normalization layer."""
        weight = layer.weight.data
        bias = layer.bias.data if layer.bias is not None else None
        
        batch_size, in_channels, H, W = activation.shape
        
        # Compute L2 norm per channel
        l2_norm = torch.norm(activation, p=2, dim=(0,2,3)) / (batch_size * H * W)
        
        # Compute scores
        weight_scores = torch.ones_like(weight) * l2_norm
        scores = {'weight': weight_scores}
        
        if bias is not None:
            scores['bias'] = weight_scores.clone()
            
        return scores