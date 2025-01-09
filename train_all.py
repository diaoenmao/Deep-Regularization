import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector
from data.data_loader import get_dataset
from scores.score_loader import choose_score
from models.cnn import CNN
from scores.wanda import WandaScoreCalculator
from scores.lora import LoraScore
from scores.magnitude import MagnitudeScore
from schedulers.lr_scheduler import CosineScheduler

# Import optimizers
from optimizers.lasso_global import LASSO_Global
from optimizers.lasso_layer import LASSO_Layer
from optimizers.lasso_neuron import LASSO_Neuron
from optimizers.ppercent_global import P_Percent_Global
from optimizers.ppercent_layer import P_Percent_Layer
from optimizers.ppercent_neuron import P_Percent_Neuron
from optimizers.admm_global import ADMM_Global
from optimizers.admm_layer import ADMM_Layer
from optimizers.admm_neuron import ADMM_Neuron

def load_mnist():
    train_dataset, test_dataset = get_dataset('mnist')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader

def train_epoch(model, train_loader, optimizer, criterion, device, score_type, scheduler, update_interval=100):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Update scores periodically
        if batch_idx % update_interval == 0:
            if score_type == 'wanda':
                scores_dict = choose_score(WandaScoreCalculator, score_type, model)
            elif score_type == 'lora':
                scores_dict = choose_score(LoraScore, score_type, model)
            else:  # magnitude
                scores_dict = choose_score(MagnitudeScore, score_type, model)
            
            scores_list = []
            for name, param in model.named_parameters():
                score_new = scores_dict.get(name, torch.zeros_like(param))
                scores_list.append(score_new)
            
            if hasattr(optimizer, 'score'):
                optimizer.score = scores_list
        
        optimizer.step()
        
        # Update learning rate after each batch
        scheduler.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f'Batch [{batch_idx}/{len(train_loader)}], LR: {current_lr:.6f}')
    
    return train_loss / len(train_loader), 100. * correct / total

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    return test_loss / len(test_loader), 100. * correct / total

def calculate_remaining_weights(model):
    total = 0
    nonzero = 0
    for param in model.parameters():
        total += param.numel()
        nonzero += torch.count_nonzero(param).item()
    return 100.0 * nonzero / total

def calculate_pq_index(model):
    total_pq = 0
    total_layers = 0
    for param in model.parameters():
        if len(param.shape) > 1:  # Skip biases and 1D params
            total_layers += 1
            nonzero = torch.count_nonzero(param, dim=1).float()
            total = param.shape[1]
            layer_pq = torch.mean((nonzero / total) ** 2).item()
            total_pq += layer_pq
    return total_pq / total_layers if total_layers > 0 else 0

def plot_results(results, score_types):
    plt.figure(figsize=(15, 15))
    markers = {'admm': 'o', 'ppercent': 's', 'lasso': '^'}
    colors = {
        'admm': '#1f77b4',  # blue
        'ppercent': '#d62728',  # red
        'lasso': '#2ca02c'  # green
    }
    
    granularities = ['global', 'layer', 'neuron']
    
    for i, granularity in enumerate(granularities):
        for j, score_type in enumerate(score_types):
            plt.subplot(3, 3, i * 3 + j + 1)
            
            for opt_type in ['admm', 'ppercent', 'lasso']:
                weights_list = []
                accs_list = []
                
                for experiment in range(5):
                    key = f"{opt_type}_{granularity}_{score_type}_exp{experiment}"
                    if key in results:
                        final_weights = results[key]['final_weights']
                        final_acc = results[key]['final_acc']
                        weights_list.append(final_weights)
                        accs_list.append(final_acc)
                
                if weights_list:
                    plt.scatter(weights_list, accs_list, 
                              marker=markers[opt_type],
                              c=colors[opt_type],
                              label=f"{opt_type}",
                              s=100)  # Increased marker size
            
            plt.xlabel('Remaining Weights (%)')
            plt.ylabel('Accuracy (%)')
            plt.title(f'{granularity} - {score_type}')
            plt.legend()
            plt.grid(True)
            plt.ylim(80, 100)  # Assuming MNIST accuracy range
    
    plt.tight_layout()
    plt.savefig('results/comparison_all.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_mnist()
    
    optimizers = {
        'lasso_global': LASSO_Global,
        'lasso_layer': LASSO_Layer,
        'lasso_neuron': LASSO_Neuron,
        'ppercent_global': P_Percent_Global,
        'ppercent_layer': P_Percent_Layer,
        'ppercent_neuron': P_Percent_Neuron,
        'admm_global': ADMM_Global,
        'admm_layer': ADMM_Layer,
        'admm_neuron': ADMM_Neuron
    }
    
    score_types = ['magnitude', 'wanda', 'lora']
    results = {}
    
    # Different pruning settings for each optimizer type
    pruning_settings = {
        'admm': {'C_values': [0.001, 0.005, 0.01, 0.05, 0.1]},
        'ppercent': {'p_percent_values': [10, 30, 50, 70, 90]},
        'lasso': {'C_values': [0.001, 0.005, 0.01, 0.05, 0.1]}
    }
    
    for score_type in score_types:
        for opt_name, optimizer_class in optimizers.items():
            # Determine optimizer type (admm, ppercent, or lasso)
            opt_type = next(k for k in pruning_settings.keys() if k in opt_name)
            
            # Run multiple experiments with different pruning settings
            for exp_idx, pruning_value in enumerate(
                pruning_settings[opt_type]['C_values' if opt_type != 'ppercent' else 'p_percent_values']
            ):
                print(f"\nTraining with {opt_name} optimizer and {score_type} scores - Experiment {exp_idx}")
                
                model = CNN().to(device)
                criterion = nn.CrossEntropyLoss()
                
                # Initialize scores
                scores_dict = choose_score(
                    WandaScoreCalculator if score_type == 'wanda' else
                    LoraScore if score_type == 'lora' else
                    MagnitudeScore,
                    score_type, model
                )
                
                scores_list = []
                for name, param in model.named_parameters():
                    scores_list.append(scores_dict.get(name, torch.zeros_like(param)))

                # Initialize auxiliary variables
                vk = [p.clone() for p in model.parameters()]
                wk = [p.clone() for p in model.parameters()]
                yk = [p.clone() for p in model.parameters()]
                zk = [p.clone() for p in model.parameters()]

                if 'global' in opt_name:
                    vk = parameters_to_vector(vk)
                    wk = parameters_to_vector(wk)
                    yk = parameters_to_vector(yk)
                    zk = parameters_to_vector(zk)
                    scores_list = parameters_to_vector(scores_list)

                # Initialize optimizer with appropriate parameters
                optimizer_params = {
                    'model': model,
                    'lr': 0.001,
                    'beta': 0.9,
                    'beta2': 0.999,
                    'v0': torch.zeros(1).to(device),
                    'v1': torch.zeros(1).to(device),
                    'k': 0,
                    'score': scores_list,
                }

                if opt_type in ['admm', 'lasso']:
                    optimizer_params.update({
                        'N': 60000,
                        'C': pruning_value,
                        'vk': vk,
                        'wk': wk,
                        'yk': yk,
                        'zk': zk,
                        'adam': True
                    })
                else:  # ppercent
                    optimizer_params.update({
                        'p_percent': pruning_value,
                    })

                optimizer = optimizer_class(model.parameters(), **optimizer_params)
                
                # Initialize scheduler
                scheduler = CosineScheduler(
                    optimizer,
                    warmup_epochs=5,
                    max_epochs=100,
                    min_lr=1e-6,
                    warmup_start_lr=1e-6,
                    base_lr=0.001
                )

                # Training loop
                best_acc = 0
                for epoch in range(100):
                    print(f"\nEpoch [{epoch}/100], Current LR: {scheduler.get_last_lr()[0]:.6f}")
                    
                    train_loss, train_acc = train_epoch(
                        model, train_loader, optimizer, criterion, device,
                        score_type, scheduler, update_interval=100
                    )
                    test_loss, test_acc = test(model, test_loader, criterion, device)
                    
                    if epoch % 10 == 0:
                        print(f'Epoch {epoch}: Test Acc: {test_acc:.2f}%, '
                              f'Train Acc: {train_acc:.2f}%, '
                              f'Train Loss: {train_loss:.4f}')
                    
                    best_acc = max(best_acc, test_acc)

                # Store final results
                final_weights = calculate_remaining_weights(model)
                results[f"{opt_name}_{score_type}_exp{exp_idx}"] = {
                    'final_weights': final_weights,
                    'final_acc': best_acc
                }

        # Plot results for current score type
        plot_results(results, score_types)

if __name__ == "__main__":
    main() 