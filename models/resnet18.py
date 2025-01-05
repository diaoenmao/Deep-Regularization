from torchvision import models
from torch import nn

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(num_classes=num_classes)

    def forward(self, x):
        return self.resnet18(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_name(self):
        return 'resnet18'

__all__ = ['ResNet18']