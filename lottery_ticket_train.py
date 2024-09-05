import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange
import copy
import matplotlib.pyplot as plt
import numpy as np
from config import METRICS_DIR, PLOTS_DIR

from models import get_model
from utils import calculate_pq_index, calculate_remaining_weights, save_metrics_and_plots
from schedulers import CScheduler
from lottery_ticket.lottery_ticket import LotteryTicket
from optimizers import ADMM_Adam_Layer, ADMM_Adam_neuron, ADMM_EMA, ADMM_SGD, ADMM_LASSO
import config
import json
import os

# Import necessary functions from train.py
from train import get_Variable, get_dataset, get_optimizer

def lottery_ticket_train():
    train_dataset, test_dataset = get_dataset(config.DATASET)
    
    dataloader_train = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    dataloader_test = DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = get_model(config.MODEL_TYPE, config.DATASET)
    model = model.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    c_scheduler = CScheduler(**config.C_SCHEDULER)

    initial_state_dict = copy.deepcopy(model.state_dict())
    best_accuracy = 0
    best_remaining_weights = 100
    all_metrics = []  # List to store metrics for each pruning iteration

    lottery_ticket = LotteryTicket(
        model,
        prune_percent=config.LOTTERY_TICKET['prune_percent'],
        prune_iterations=config.LOTTERY_TICKET['prune_iterations']
    )

    for pruning_iteration in range(config.LOTTERY_TICKET['prune_iterations']):
        print(f"Pruning Iteration {pruning_iteration + 1}/{config.LOTTERY_TICKET['prune_iterations']}")
        model.load_state_dict(initial_state_dict)
        model = lottery_ticket.apply_mask(model, lottery_ticket.mask)  # Apply the current mask
        
        remaining_weights_after_mask = calculate_remaining_weights(model)
        print(f"Remaining weights after applying mask: {remaining_weights_after_mask:.2f}%")

        optimizer = get_optimizer(config.OPTIMIZER_TYPE, model.parameters())

        iteration_metrics = {'remaining_weights': [], 'accuracy': [], 'pq_index': [], 'loss': []}
        
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

            # Update best accuracy if current accuracy is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_remaining_weights = remaining_weights

            iteration_metrics['remaining_weights'].append(remaining_weights)
            iteration_metrics['accuracy'].append(accuracy)
            iteration_metrics['pq_index'].append(calculate_pq_index(model))
            iteration_metrics['loss'].append(running_loss / len(train_dataset))

            print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss: {running_loss/len(train_dataset):.4f}, "
                f"Train Accuracy: {100*running_correct/len(train_dataset):.2f}%, "
                f"Test Accuracy: {accuracy:.2f}%, "
                f"Remaining Weights: {remaining_weights:.2f}%, "
                f"PQ Index: {calculate_pq_index(model):.4f}")

        all_metrics.append({
            'pruning_iteration': pruning_iteration,
            'metrics': iteration_metrics
        })

        # Prune the model after each training cycle
        model = lottery_ticket.prune(model)

    # Final evaluation
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for data, target in dataloader_test:
            data, target = get_Variable(data), get_Variable(target)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            test_correct += (predicted == target).sum().item()
    
    final_accuracy = 100 * test_correct / len(test_dataset)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")

    print(f"Best accuracy: {best_accuracy:.2f}% at {best_remaining_weights:.2f}% remaining weights")

    # Save metrics
    save_metrics(all_metrics)

    # Plot metrics for all pruning iterations
    plot_all_metrics(all_metrics)

def save_metrics(all_metrics):
    # Convert numpy arrays to lists for JSON serialization
    for pruning_data in all_metrics:
        for key, value in pruning_data['metrics'].items():
            pruning_data['metrics'][key] = [float(v) for v in value]

    metrics_file = os.path.join(METRICS_DIR, 'lottery_ticket_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

def plot_all_metrics(all_metrics):
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    metrics_to_plot = ['accuracy', 'loss', 'remaining_weights', 'pq_index']
    colors = plt.cm.rainbow(np.linspace(0, 1, len(all_metrics)))

    for i, metric in enumerate(metrics_to_plot):
        ax = axs[i // 2, i % 2]
        for j, pruning_data in enumerate(all_metrics):
            pruning_iteration = pruning_data['pruning_iteration']
            metrics = pruning_data['metrics']
            ax.plot(metrics[metric], color=colors[j], label=f'Pruning {pruning_iteration + 1}')
        ax.set_title(metric.capitalize())
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()

    plt.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, 'lottery_ticket_metrics.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Saved plot to {plot_file}")

    # Print the number of pruning iterations plotted
    print(f"Plotted metrics for {len(all_metrics)} pruning iterations")

if __name__ == "__main__":
    lottery_ticket_train()
