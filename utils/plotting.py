import os
import matplotlib.pyplot as plt

def generate_plots(epochs, pruning_percentages, train_accuracies, test_accuracies, remaining_weights, pq_indices):
    os.makedirs('results/plots', exist_ok=True)

    # Pruning vs Accuracy
    plt.figure()
    plt.plot(pruning_percentages, test_accuracies)
    plt.xlabel('Pruning Percentage')
    plt.ylabel('Test Accuracy')
    plt.title('Pruning vs Accuracy')
    plt.savefig('results/plots/pruning_vs_accuracy.png')
    plt.close()

    # Accuracy vs Epoch
    plt.figure()
    plt.plot(epochs, train_accuracies, label='Train')
    plt.plot(epochs, test_accuracies, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    plt.savefig('results/plots/accuracy_vs_epoch.png')
    plt.close()

    # Remaining Weights vs Epoch
    plt.figure()
    plt.plot(epochs, remaining_weights)
    plt.xlabel('Epoch')
    plt.ylabel('Remaining Weights (%)')
    plt.title('Remaining Weights vs Epoch')
    plt.savefig('results/plots/remaining_weights_vs_epoch.png')
    plt.close()

    # PQ Index vs Epoch
    plt.figure()
    plt.plot(epochs, pq_indices)
    plt.xlabel('Epoch')
    plt.ylabel('PQ Index')
    plt.title('PQ Index vs Epoch')
    plt.savefig('results/plots/pq_index_vs_epoch.png')
    plt.close()

def plot_metrics(metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(metrics['remaining_weights']) + 1)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(epochs, metrics['remaining_weights'])
    plt.xlabel('Epoch')
    plt.ylabel('Remaining Weights (%)')
    plt.title('Remaining Weights vs Epoch')
    
    plt.subplot(132)
    plt.plot(epochs, metrics['accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epoch')
    
    plt.subplot(133)
    plt.plot(epochs, metrics['pq_index'])
    plt.xlabel('Epoch')
    plt.ylabel('PQ Index')
    plt.title('PQ Index vs Epoch')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics.png'))
    plt.close()

def plot_accuracy_vs_pruning(remaining_weights, accuracies, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure()
    plt.plot(remaining_weights, accuracies)
    plt.xlabel('Remaining Weights (%)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Pruning')
    plt.savefig(os.path.join(save_dir, 'accuracy_vs_pruning.png'))
    plt.close()