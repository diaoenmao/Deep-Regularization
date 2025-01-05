from torch.optim import Optimizer
import torch
import torch.nn as nn

class LASSO_Neuron(Optimizer):
    def __init__(self, params, model, lr, N, C, score):
        self.model = model
        self.lr = lr
        self.N = N  # NUMBER OF SAMPLES
        self.C = C  # REGULARIZATION CONSTANT
        self.score = score  # Score per neuron
        super(LASSO_Neuron, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for name, module in self.model.named_modules():
            # Handle different layer types
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
        score = self.score[name] if name in self.score else torch.ones_like(w)

        # Treat each output channel as a neuron
        for i in range(w.size(0)):  # iterate over output channels
            # Get all weights connected to this neuron
            neuron_weights = w[i]  # [in_channels, kernel_h, kernel_w]
            neuron_grad = grad[i]
            neuron_score = score[i] if len(score.shape) > 1 else score

            # Compute L2 norm of the neuron's weights
            norm = torch.norm(neuron_weights)
            if norm > 0:
                # Apply gradient step
                neuron_weights = neuron_weights * neuron_score - self.lr * neuron_grad

                # Apply soft thresholding on the entire neuron
                threshold = (self.C / self.N) * self.lr / (neuron_score + 1e-8)
                scale = max(0, 1 - threshold / (norm + 1e-8))
                neuron_weights *= scale

                # Update weights
                w[i] = neuron_weights

    def _handle_linear_layer(self, layer, name):
        """Handle linear layer neuron-wise pruning"""
        if layer.weight.grad is None:
            return

        # Shape: [out_features, in_features]
        w = layer.weight.data
        grad = layer.weight.grad
        score = self.score[name] if name in self.score else torch.ones_like(w)

        # Treat each output feature as a neuron
        for i in range(w.size(0)):  # iterate over output features
            # Get all weights connected to this neuron
            neuron_weights = w[i]  # [in_features]
            neuron_grad = grad[i]
            neuron_score = score[i] if len(score.shape) > 1 else score

            # Compute L2 norm of the neuron's weights
            norm = torch.norm(neuron_weights)
            if norm > 0:
                # Apply gradient step
                neuron_weights = neuron_weights * neuron_score - self.lr * neuron_grad

                # Apply soft thresholding on the entire neuron
                threshold = (self.C / self.N) * self.lr / (neuron_score + 1e-8)
                scale = max(0, 1 - threshold / (norm + 1e-8))
                neuron_weights *= scale

                # Update weights
                w[i] = neuron_weights

    def update_base_learning_rate(self, new_lr):
        self.lr = new_lr
