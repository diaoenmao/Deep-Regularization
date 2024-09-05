import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import trange
import os
import json
import matplotlib.pyplot as plt
from config import METRICS_DIR, PLOTS_DIR

from models import get_model
from utils import calculate_pq_index, calculate_remaining_weights, save_metrics_and_plots
from schedulers import CScheduler
from optimizers import ADMM_Adam_Layer, ADMM_Adam_neuron, ADMM_EMA, ADMM_SGD, ADMM_LASSO
import config

def get_Variable(x):
    x = torch.autograd.Variable(x)
    return x.cuda() if torch.cuda.is_available() else x

def get_dataset(dataset_name):
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='data/', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='data/', train=False, transform=transform)
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
        train_dataset = datasets.CIFAR10(root='data/', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='data/', train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_dataset, test_dataset

def get_optimizer(optimizer_type, model_params):
    if optimizer_type == 'ADMM_Adam_Layer':
        return ADMM_Adam_Layer(model_params, lr=config.LEARNING_RATE, N=config.N, C=config.C_SCHEDULER['start_value'], 
                            vk=[], wk=[], yk=[], zk=[], beta=config.BETA1, beta2=config.BETA2, v0=[], v1=[], k=0)
    elif optimizer_type == 'ADMM_ADAM_neuron':
        return ADMM_Adam_neuron(model_params, lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2), N=config.N, C=config.C_SCHEDULER['start_value'])
    elif optimizer_type == 'ADMM_EMA':
        return ADMM_EMA(model_params, lr=config.LEARNING_RATE, N=config.N, C=config.C_SCHEDULER['start_value'])
    elif optimizer_type == 'ADMM_SGD':
        return ADMM_SGD(model_params, lr=config.LEARNING_RATE, N=config.N, C=config.C_SCHEDULER['start_value'])
    elif optimizer_type == 'ADMM_LASSO':
        return ADMM_LASSO(model_params, lr=config.LEARNING_RATE, N=config.N, C=config.C_SCHEDULER['start_value'])
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

def train():
    train_dataset, test_dataset = get_dataset(config.DATASET)
    
    dataloader_train = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    dataloader_test = DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = get_model(config.MODEL_TYPE, config.DATASET)
    model = model.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    c_scheduler = CScheduler(**config.C_SCHEDULER)

    optimizer = get_optimizer(config.OPTIMIZER_TYPE, model.parameters())

    metrics = {'remaining_weights': [], 'accuracy': [], 'pq_index': []}

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in trange(config.EPOCHS):
        model.train()
        running_loss = 0.0
        running_correct = 0

        for data, target in dataloader_train:
            data, target = get_Variable(data), get_Variable(target)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            running_correct += (predicted == target).sum().item()

        model.eval()
        test_correct = 0
        with torch.no_grad():
            for data, target in dataloader_test:
                data, target = get_Variable(data), get_Variable(target)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                test_correct += (predicted == target).sum().item()

        accuracy = 100 * test_correct / len(test_dataset)
        remaining_weights = calculate_remaining_weights(model)
        pq_index = calculate_pq_index(model)

        metrics['remaining_weights'].append(remaining_weights)
        metrics['accuracy'].append(accuracy)
        metrics['pq_index'].append(pq_index)

        train_losses.append(running_loss / len(train_dataset))
        train_accuracies.append(100 * running_correct / len(train_dataset))
        val_losses.append(0)  # Placeholder for validation loss
        val_accuracies.append(accuracy)

        print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss: {running_loss/len(train_dataset):.4f}, "
            f"Train Accuracy: {100*running_correct/len(train_dataset):.2f}%, "
            f"Test Accuracy: {accuracy:.2f}%, "
            f"Remaining Weights: {remaining_weights:.2f}%, "
            f"PQ Index: {pq_index:.4f}")

        new_C = c_scheduler.get_c(epoch + 1)
        if hasattr(optimizer, 'update_hyperparameters'):
            optimizer.update_hyperparameters(C=new_C)

    # Save metrics
    save_metrics_and_plots(train_losses, train_accuracies, val_losses, val_accuracies, metrics)

if __name__ == "__main__":
    print("This script should be run through main.py")