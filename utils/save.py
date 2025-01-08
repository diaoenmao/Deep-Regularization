import os
import json
import matplotlib.pyplot as plt

def save_metrics(results, save_dir='results', model_type='cnn3', optimizer_type='ADMM_layer_magnitude'):
    """
    Save experiment metrics to JSON file
    """
    metrics_dir = os.path.join(save_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    results_file = os.path.join(metrics_dir, f'{model_type}_{optimizer_type}_experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved experiment results to {results_file}")

def plot_metrics(results, save_dir='results', model_type='cnn3', optimizer_type='ADMM_layer_magnitude'):
    """
    Plot training metrics and save figures
    """
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(results['accuracy'])
    ax1.set_title('Test Accuracy')
    ax1.set_xlabel('Experiment')
    ax1.set_ylabel('Accuracy (%)')
    
    # Plot remaining weights
    ax2.plot(results['remaining_weights'])
    ax2.set_title('Remaining Weights')
    ax2.set_xlabel('Experiment')
    ax2.set_ylabel('Ratio')
    
    # Plot PQ index
    ax3.plot(results['pq_index'])
    ax3.set_title('PQ Index')
    ax3.set_xlabel('Experiment')
    ax3.set_ylabel('Index Value')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(plots_dir, f'{model_type}_{optimizer_type}_metrics.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Saved metrics plot to {plot_file}")

def plot_accuracy_vs_pruning(results, save_dir='results', model_type='cnn3', optimizer_type='ADMM_layer_magnitude'):
    """
    Plot accuracy vs pruning ratio
    """
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(results['remaining_weights'], results['accuracy'])
    plt.xlabel('Remaining Weights Ratio')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Pruning Ratio')
    
    plot_file = os.path.join(plots_dir, f'{model_type}_{optimizer_type}_acc_vs_pruning.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Saved accuracy vs pruning plot to {plot_file}")