import os
import json
import matplotlib.pyplot as plt
from utils import plot_metrics, plot_accuracy_vs_pruning
from config import METRICS_DIR, PLOTS_DIR, MODEL_TYPE, OPTIMIZER_TYPE, SAVE_DIR, EPOCHS

def save_metrics_and_plots(train_losses, train_accuracies, val_losses, val_accuracies, metrics):
    # Save metrics
    metrics_data = {
        'train_loss': train_losses,
        'train_acc': train_accuracies,
        'val_loss': val_losses,
        'val_acc': val_accuracies,
        'remaining_weights': metrics['remaining_weights'],
        'accuracy': metrics['accuracy'],
        'pq_index': metrics['pq_index']
    }
    
    metrics_file = os.path.join(METRICS_DIR, f'{MODEL_TYPE}_{OPTIMIZER_TYPE}_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f)

    # Create and save plots
    epochs = range(1, EPOCHS + 1)

    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, f'{MODEL_TYPE}_{OPTIMIZER_TYPE}_loss.png'))
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, f'{MODEL_TYPE}_{OPTIMIZER_TYPE}_accuracy.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.scatter(metrics['remaining_weights'], metrics['accuracy'])
    plt.title('Accuracy vs Remaining Weights')
    plt.xlabel('Remaining Weights (%)')
    plt.ylabel('Accuracy (%)')
    plt.savefig(os.path.join(PLOTS_DIR, f'{MODEL_TYPE}_{OPTIMIZER_TYPE}_accuracy_vs_weights.png'))
    plt.close()

    # Remaining weights vs Epoch plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics['remaining_weights'])
    plt.title('Remaining Weights vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Remaining Weights (%)')
    plt.savefig(os.path.join(PLOTS_DIR, f'{MODEL_TYPE}_{OPTIMIZER_TYPE}_weights_vs_epoch.png'))
    plt.close()

    plot_metrics(metrics, save_dir=SAVE_DIR)
    plot_accuracy_vs_pruning(metrics['remaining_weights'], metrics['accuracy'], save_dir=SAVE_DIR)

def save_experiment_results(results):
    results_file = os.path.join(METRICS_DIR, f'{MODEL_TYPE}_{OPTIMIZER_TYPE}_experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved experiment results to {results_file}")

def plot_experiment_results(results):
    C_values = [result['C'] for result in results]
    accuracies = [result['accuracy'] for result in results]
    remaining_weights = [result['remaining_weights'] for result in results]
    pq_indices = [result['pq_index'] for result in results]

    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.plot(C_values, accuracies, 'o-')
    plt.xlabel('C value')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs C value')

    plt.subplot(132)
    plt.plot(C_values, remaining_weights, 'o-')
    plt.xlabel('C value')
    plt.ylabel('Remaining Weights (%)')
    plt.title('Remaining Weights vs C value')

    plt.subplot(133)
    plt.plot(C_values, pq_indices, 'o-')
    plt.xlabel('C value')
    plt.ylabel('PQ Index')
    plt.title('PQ Index vs C value')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{MODEL_TYPE}_{OPTIMIZER_TYPE}_experiment_results.png'))
    plt.close()