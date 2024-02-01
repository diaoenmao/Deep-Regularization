import torch 
import torch.nn as nn 
import os 
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(device) 

root = os.path.join('~/Research/Deep-Regularization/data')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

training_data = datasets.CIFAR10(
    root=root,            
    train=True,            
    download=True,          
    transform=transform    
)
test_data = datasets.CIFAR10(
    root=root,
    train=False,
    download=True,
    transform=transform 
)

train_dataloader = DataLoader(training_data,    # our dataset
                              batch_size=64,    # batch size
                              shuffle=True      # shuffling the data
                            )
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32*32*3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)

def soft_threshold(param:torch.Tensor, lmbda:torch.Tensor) -> torch.Tensor: 
    """
    Helper function that takes in parameter tensor and does the soft_threshold 
    function on each element. Outputs the tensor.  
    """
    x = torch.sign(param) * torch.max(torch.abs(param) - lmbda, torch.zeros_like(param))
    return x

for name, param in model.named_parameters(): 
    if "bias" not in name: 
        print(param[0][0].detach().cpu().item())

for name, param in model.named_parameters(): 
    if "bias" not in name: 
        with torch.no_grad(): 
            param.copy_(soft_threshold(param, torch.tensor(0.01)))

for name, param in model.named_parameters(): 
    if "bias" not in name: 
        print(param[0][0].detach().cpu().item())


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),     # which parameters to optimize
    lr=1e-3,                 # learning rate 
    momentum=0.9
)

