from torch.optim import Optimizer
import torch
import torch.nn as nn

class P_Percent_Neuron(Optimizer):
    def __init__(self, params, model, lr, p_percent, score):
        self.model = model
        self.lr = lr
        self.p_percent = p_percent  # percentage of neurons to prune (0-100)
        self.score = score  # Score per neuron for each layer
        super(P_Percent_Neuron, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self._handle_conv_layer(module, name)
            elif isinstance(module, nn.Linear):
                self._handle_linear_layer(module, name)

        return loss

    def _handle_conv_layer(self, layer, name):
        """Handle convolutional layer neuron-wise pruning"""
        if layer.weight.grad is None:
            return

        # Shape: [out_channels, in_channels, kernel_h, kernel_w]
        w = layer.weight.data
        grad = layer.weight.grad
        score = self.score[name] if name in self.score else torch.ones(w.size(0))

        # Standard gradient step
        w.data = w.data - self.lr * grad

        # Calculate L2 norm for each output channel (neuron)
        neuron_norms = torch.norm(w.view(w.size(0), -1), p=2, dim=1)
        
        # Apply importance scores
        neuron_norms = neuron_norms * score

        # Calculate number of neurons to prune
        k = int(w.size(0) * (self.p_percent / 100.0))

        if k > 0:
            # Find threshold
            threshold = torch.kthvalue(neuron_norms, k).values

            # Create pruning mask
            mask = (neuron_norms > threshold).float()

            # Reshape mask to match weights and apply
            mask = mask.view(-1, 1, 1, 1).expand_as(w)
            w.data = w.data * mask

    def _handle_linear_layer(self, layer, name):
        """Handle linear layer neuron-wise pruning"""
        if layer.weight.grad is None:
            return

        # Shape: [out_features, in_features]
        w = layer.weight.data
        grad = layer.weight.grad
        score = self.score[name] if name in self.score else torch.ones(w.size(0))

        # Standard gradient step
        w.data = w.data - self.lr * grad

        # Calculate L2 norm for each output neuron
        neuron_norms = torch.norm(w, p=2, dim=1)
        
        # Apply importance scores
        neuron_norms = neuron_norms * score

        # Calculate number of neurons to prune
        k = int(w.size(0) * (self.p_percent / 100.0))

        if k > 0:
            # Find threshold
            threshold = torch.kthvalue(neuron_norms, k).values

            # Create pruning mask
            mask = (neuron_norms > threshold).float()

            # Reshape mask to match weights and apply
            mask = mask.view(-1, 1).expand_as(w)
            w.data = w.data * mask

    def update_base_learning_rate(self, new_lr):
        self.lr = new_lr

    def update_p_percent(self, new_p_percent):
        """Update the pruning percentage"""
        self.p_percent = new_p_percent
