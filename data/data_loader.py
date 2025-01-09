'''
Prepare data for training and testing
'''

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

def get_dataset(dataset_name):
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Use a single directory for MNIST
        data_dir = 'data/MNIST'
        os.makedirs(data_dir, exist_ok=True)

        try:
            # Try to load from the same directory
            train_dataset = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
            test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
        except RuntimeError:
            print("MNIST dataset not found. Downloading...")
            train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        
        return train_dataset, test_dataset
    
    elif dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        os.makedirs('data/cifar10', exist_ok=True)
        os.makedirs('data/cifar10/train', exist_ok=True)
        os.makedirs('data/cifar10/test', exist_ok=True)
        train_dataset = datasets.CIFAR10(root='data/cifar10/train', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='data/cifar10/test', train=False, download=True, transform=transform_test)
        return train_dataset, test_dataset
    
    elif dataset_name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2673, 0.2564, 0.2762)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2673, 0.2564, 0.2762)),
        ])
        os.makedirs('data/cifar100', exist_ok=True)
        os.makedirs('data/cifar100/train', exist_ok=True)
        os.makedirs('data/cifar100/test', exist_ok=True)
        train_dataset = datasets.CIFAR100(root='data/cifar100/train', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root='data/cifar100/test', train=False, download=True, transform=transform_test)
        return train_dataset, test_dataset
    
    elif dataset_name == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2861,), (0.3530,))
        ])
        os.makedirs('data/fashion_mnist', exist_ok=True)
        os.makedirs('data/fashion_mnist/train', exist_ok=True)
        os.makedirs('data/fashion_mnist/test', exist_ok=True)
        train_dataset = datasets.FashionMNIST(root='data/fashion_mnist/train', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='data/fashion_mnist/test', train=False, transform=transform)
        return train_dataset, test_dataset
    
    elif dataset_name == 'tiny_imagenet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        os.makedirs('data/tiny_imagenet', exist_ok=True)
        os.makedirs('data/tiny_imagenet/train', exist_ok=True)
        os.makedirs('data/tiny_imagenet/val', exist_ok=True)
        train_dataset = datasets.ImageFolder(root='data/tiny_imagenet/train', transform=transform)
        test_dataset = datasets.ImageFolder(root='data/tiny_imagenet/val', transform=transform)
        return train_dataset, test_dataset
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
