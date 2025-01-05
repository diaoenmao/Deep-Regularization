from torch import nn

class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.fc = nn.Linear(784, num_classes)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_name(self):
        return 'mlp'
    
__all__ = ['MLP']