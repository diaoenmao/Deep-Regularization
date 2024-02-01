from re import A
import torch
import torch.nn as nn 
import os 
from torchvision import datasets 
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader

device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


root = os.path.join('./data')

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
