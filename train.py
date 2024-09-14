import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import trange
import os
import json
import matplotlib.pyplot as plt
from config import METRICS_DIR, PLOTS_DIR
from torch.nn.utils import parameters_to_vector
import math

from models import get_model
from utils import calculate_pq_index, calculate_remaining_weights, save_metrics_and_plots, save_experiment_results, plot_experiment_results
from schedulers import CScheduler
from optimizers import ADMM_Adam_Layer, ADMM_Adam_neuron, ADMM_EMA, ADMM_SGD, ADMM_LASSO, ADMM_Adam_global
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

def get_optimizer(optimizer_type, model_params, model, C=None):
    if optimizer_type == 'ADMM_Adam_layer':
        vk = []
        wk = []
        yk = []
        zk = []
        v0 = []
        v1 = []
        for _, param in model.named_parameters():
            vk.append(torch.zeros_like(param))
            wk.append(torch.zeros_like(param))
            yk.append(param.clone())
            zk.append(param.clone())
            v0.append(torch.zeros_like(param))
            v1.append(torch.zeros_like(param))
        return ADMM_Adam_Layer(model_params, lr=config.LEARNING_RATE, N=config.N, C=C, 
                            vk=vk, wk=wk, yk=yk, zk=zk, beta=config.BETA1, beta2=config.BETA2, 
                            v0=v0, v1=v1, k=0)
    elif optimizer_type == 'ADMM_Adam_neuron':
        vk = []
        wk = []
        yk = []
        zk = []
        for _, param in model.named_parameters():
            vk.append(torch.zeros_like(param))
            wk.append(torch.zeros_like(param))
            yk.append(param.clone())
            zk.append(param.clone())
        return ADMM_Adam_neuron(model_params, lr=config.LEARNING_RATE, N=config.N, C=C,
                                vk=vk, wk=wk, yk=yk, zk=zk)
    elif optimizer_type == 'ADMM_Adam_global':
        w = parameters_to_vector(model.parameters())
        vk = torch.zeros_like(w)
        wk = torch.zeros_like(w)
        v0 = torch.zeros_like(w)
        v1 = torch.zeros_like(w)
        yk = w
        zk = w
        return ADMM_Adam_global(model_params, model=model, lr=config.LEARNING_RATE, N=config.N, C=C, 
                                vk=vk, wk=wk, yk=yk, zk=zk, beta=config.BETA1, beta2=config.BETA2, 
                                v0=v0, v1=v1, k=0)
    elif optimizer_type == 'ADMM_EMA':
        return ADMM_EMA(model_params, lr=config.LEARNING_RATE, N=config.N, C=C)
    elif optimizer_type == 'ADMM_SGD':
        return ADMM_SGD(model_params, lr=config.LEARNING_RATE, N=config.N, C=C)
    elif optimizer_type == 'ADMM_LASSO':
        return ADMM_LASSO(model_params, lr=config.LEARNING_RATE, N=config.N, C=C)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

def train(C=None):
    train_dataset, test_dataset = get_dataset(config.DATASET)
    
    dataloader_train = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    dataloader_test = DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = get_model(config.MODEL_TYPE, config.DATASET)
    model = model.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()

    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    metrics = {'remaining_weights': [], 'accuracy': [], 'pq_index': []}

    for epoch in trange(config.EPOCHS):
        # Cosine annealing learning rate scheduler
        lr = config.LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * epoch / config.EPOCHS))
        # lr = config.LEARNING_RATE
        optimizer = get_optimizer(config.OPTIMIZER_TYPE, model.parameters(), model=model, C=C)
        optimizer.update_base_learning_rate(lr)

        model.train()
        running_loss = 0.0
        running_correct = 0

        for data, target in dataloader_train:

            data, target = get_Variable(data), get_Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            running_correct += (predicted == target).sum().item()

        train_loss = running_loss / len(dataloader_train)
        train_accuracy = 100 * running_correct / len(train_dataset)

        model.eval()
        test_loss = 0.0
        test_correct = 0

        with torch.no_grad():
            for data, target in dataloader_test:
                data, target = get_Variable(data), get_Variable(target)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                test_correct += (predicted == target).sum().item()

        test_loss /= len(dataloader_test)
        test_accuracy = 100 * test_correct / len(test_dataset)
        
        remaining_weights = calculate_remaining_weights(model)
        pq_index = calculate_pq_index(model)

        metrics['remaining_weights'].append(remaining_weights)
        metrics['accuracy'].append(test_accuracy)
        metrics['pq_index'].append(pq_index)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(test_loss)
        val_accuracies.append(test_accuracy)

        print(f"Epoch [{epoch+1}/{config.EPOCHS}], "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
            f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, "
            f"Remaining Weights: {remaining_weights:.2f}%, "
            f"PQ Index: {pq_index:.4f}")

    return model, train_losses, train_accuracies, val_losses, val_accuracies, metrics

def run_experiments(num_experiments=10):
    results = []
    c_scheduler = CScheduler(**config.C_SCHEDULER)

    for i in range(num_experiments):
        C = c_scheduler.get_c(i)
        print(f"Running experiment {i+1}/{num_experiments} with C={C}")
        accuracy, remaining_weights, pq_index = train(fixed_C=C)
        results.append({
            'C': C,
            'accuracy': accuracy,
            'remaining_weights': remaining_weights,
            'pq_index': pq_index
        })

    # Save and plot results
    save_experiment_results(results)
    plot_experiment_results(results)

if __name__ == "__main__":
    print("This script should be run through main.py")