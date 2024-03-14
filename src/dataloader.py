import os 
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
from src.regularizers import * 

def load_dataloaders(name:str, batch_size:int=64): 
  root = os.path.join('./data')

  if name == "mnist": 
    transform = Compose([
      ToTensor(),
      Normalize((0.5), (0.5))
    ])

    training_data = datasets.MNIST(
      root=root,            
      train=True,            
      download=True,          
      transform=transform    
    )
    test_data = datasets.MNIST(
      root=root,
      train=False,
      download=True,
      transform=transform 
    )
    input_size, output_size = 28*28*1, 10

  elif name == "fashion-mnist": 
    transform = Compose([
      ToTensor(),
      Normalize((0.5), (0.5))
    ])

    training_data = datasets.FashionMNIST(
      root=root,            
      train=True,            
      download=True,          
      transform=transform    
    )
    test_data = datasets.FashionMNIST(
      root=root,
      train=False,
      download=True,
      transform=transform 
    )
    
    input_size, output_size = 28*28*1, 10

  elif name == "cifar10": 
    transform = Compose([
      ToTensor(),
      Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

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

    input_size, output_size = 32*32*3, 10

  elif name == "cifar100": 
    transform = Compose([
      ToTensor(),
      Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    training_data = datasets.CIFAR100(
      root=root,            
      train=True,            
      download=True,          
      transform=transform    
    )
    test_data = datasets.CIFAR100(
      root=root,
      train=False,
      download=True,
      transform=transform 
    )

    input_size, output_size = 32*32*3, 100

  else: 
    raise Exception("Not a valid dataset")

  train_dataloader = DataLoader(
    training_data,      # type: ignore
    batch_size=batch_size,
    shuffle=True    
  )

  test_dataloader = DataLoader(
    test_data,          # type: ignore
    batch_size=batch_size, 
    shuffle=True
  )

  return train_dataloader, test_dataloader, input_size, output_size


