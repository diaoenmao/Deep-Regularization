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
from tqdm import tqdm
import sys
import os
import json
import glob
import re

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

def train_epoch(model, train_loader, optimizer, criterion, device, score_type, scheduler, epoch, total_epochs, pbar=None):
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
        # scheduler.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if pbar is not None:
            pbar.set_postfix({
                'loss': f'{train_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
            pbar.update(1)
    
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

def save_checkpoint(state, score_type, opt_name, exp_idx):
    """
    Save checkpoint with informative filename
    """
    os.makedirs('checkpoints', exist_ok=True)
    filename = f'checkpoint_{score_type}_{opt_name}_exp{exp_idx}.pth'
    torch.save(state, os.path.join('checkpoints', filename))
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(checkpoint_dir='checkpoints'):
    """
    Load the most recent checkpoint and extract experiment information from filename.
    Filename format: checkpoint_<score_type>_<opt_type>_<granularity>_exp<exp_idx>.pth
    """
    try:
        # Get all checkpoint files
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.pth'))
        if not checkpoint_files:
            return None
        
        # Sort by modification time to get the most recent
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        
        # Parse filename to get experiment details
        filename = os.path.basename(latest_checkpoint)
        pattern = r'checkpoint_(\w+)_(\w+)_(\w+)_exp(\d+).pth'
        match = re.match(pattern, filename)
        
        if not match:
            print(f"Warning: Checkpoint filename {filename} doesn't match expected pattern")
            return None
            
        score_type, opt_type, granularity, exp_idx = match.groups()
        exp_idx = int(exp_idx)
        
        # Load checkpoint data
        checkpoint = torch.load(latest_checkpoint)
        
        # Add parsed information to checkpoint
        checkpoint.update({
            'score_type': score_type,
            'opt_type': opt_type,
            'granularity': granularity,
            'exp_idx': exp_idx,
            'checkpoint_file': latest_checkpoint
        })
        
        print(f"\nLoaded checkpoint from {latest_checkpoint}")
        print(f"Score type: {score_type}")
        print(f"Optimizer: {opt_type}_{granularity}")
        print(f"Experiment: {exp_idx}")
        print(f"Epoch: {checkpoint['epoch']}")
        print(f"Best accuracy: {checkpoint['best_acc']:.2f}%")
        
        return checkpoint
        
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    train_loader, test_loader = load_mnist()
    print("Dataset loaded: MNIST")
    
    if os.path.exists('results/results_all.json'):
        with open('results/results_all.json', 'r') as f:
            results = json.load(f)
    else:
        results = {}
    
    optimizers = {
        # 'admm_global': ADMM_Global,
        # 'admm_layer': ADMM_Layer,
        # 'admm_neuron': ADMM_Neuron,
        # 'lasso_global': LASSO_Global,
        # 'lasso_layer': LASSO_Layer,
        # 'lasso_neuron': LASSO_Neuron,
        'ppercent_global': P_Percent_Global,
        'ppercent_layer': P_Percent_Layer,
        'ppercent_neuron': P_Percent_Neuron
    }
    
    score_types = ['magnitude', 'wanda', 'lora']
    
    # Different pruning settings for each optimizer type
    pruning_settings = {
        # 'admm': {'C_values': [0.001, 0.005, 0.01, 0.05, 0.1]},
        # 'lasso': {'C_values': [0.001, 0.005, 0.01, 0.05, 0.1]},
        'ppercent': {'p_percent_values': [10, 30, 50, 70, 90]}
    }
    
    total_experiments = len(score_types) * len(optimizers) * 5  # 5 experiments per combination
    experiment_count = 0
    
    # Load previous progress if exists
    checkpoint = load_checkpoint()
    if checkpoint:
        results = checkpoint['results']
        start_score_idx = score_types.index(checkpoint['score_type'])
        start_opt_idx = list(optimizers.keys()).index(f"{checkpoint['opt_type']}_{checkpoint['granularity']}")
        start_exp_idx = checkpoint['exp_idx']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        results = {}
        start_score_idx = 0
        start_opt_idx = 0
        start_exp_idx = 0
        start_epoch = 0
    
    for score_idx, score_type in enumerate(score_types[start_score_idx:], start_score_idx):
        for opt_idx, (opt_name, optimizer_class) in enumerate(list(optimizers.items())[start_opt_idx:], start_opt_idx):
            opt_type = next(k for k in pruning_settings.keys() if k in opt_name)
            
            for exp_idx, pruning_value in enumerate(
                pruning_settings[opt_type]['C_values' if opt_type != 'ppercent' else 'p_percent_values'][start_exp_idx:],
                start_exp_idx
            ):
                # Skip if experiment already exists in results
                exp_key = f"{opt_name}_{score_type}_exp{exp_idx}"
                if exp_key in results:
                    print(f"Skipping existing experiment: {exp_key}")
                    continue
                
                experiment_count += 1
                print(f"\nExperiment {exp_idx + 1}/5 - Progress: [{experiment_count}/{total_experiments}]")
                print(f"Pruning percent: {pruning_value}")
                
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

                v0 = torch.zeros(1).to(device)
                v1 = torch.zeros(1).to(device)
                k = 0
                beta = 0.9
                beta2 = 0.999
                lr = 0.001

                # Initialize auxiliary variables based on optimizer type
                if opt_type == 'admm':
                    vk = [p.clone() for p in model.parameters()]
                    wk = [p.clone() for p in model.parameters()]
                    yk = [p.clone() for p in model.parameters()]
                    zk = [p.clone() for p in model.parameters()]
                elif opt_type == 'lasso':
                    vk = [p.clone() for p in model.parameters()]
                    wk = [p.clone() for p in model.parameters()]
                    zk = [p.clone() for p in model.parameters()]
                elif opt_type == 'ppercent':
                    # ppercent doesn't need auxiliary variables
                    vk = None
                    wk = None
                    zk = None


                if 'global' in opt_name:
                    if 'ppercent' not in opt_name:
                        vk = parameters_to_vector(vk)
                        wk = parameters_to_vector(wk)
                        zk = parameters_to_vector(zk)
                        if 'admm' in opt_name:
                            yk = parameters_to_vector(yk)
                    scores_list = parameters_to_vector(scores_list)

                # Initialize optimizer with appropriate parameters
                optimizer_params = {
                    'model': model,
                    'lr': lr,
                    'score': scores_list,
                }

                if opt_type == 'admm':
                    optimizer_params.update({
                        'N': 60000,
                        'C': pruning_value,
                        'vk': vk,
                        'wk': wk,
                        'yk': yk,
                        'zk': zk,
                        'beta': beta,
                        'beta2': beta2,
                        'v0': v0,
                        'v1': v1,
                        'k': k,
                        'adam': True
                    })
                elif opt_type == 'lasso':
                    optimizer_params.update({
                        'N': 60000,
                        'C': pruning_value,
                        'vk': vk,
                        'wk': wk,
                        'zk': zk,
                        'beta': beta,
                        'beta2': beta2,
                        'v0': v0,
                        'v1': v1,
                        'k': k,
                        'adam': True
                    })
                else:  # ppercent
                    optimizer_params.update({
                        'p_percent': pruning_value,
                        'v0': v0,
                        'v1': v1,
                        'k': k,
                        'beta': beta,
                        'beta2': beta2,
                        'adam': True
                    })

                optimizer = optimizer_class(model.parameters(), **optimizer_params)
                
                # Initialize scheduler
                scheduler = CosineScheduler(
                    optimizer,
                    warmup_epochs=5,
                    max_epochs=100,
                    min_lr=1e-6,
                    verbose=False
                )

                # Training loop with progress bar
                best_acc = 0
                total_batches = len(train_loader) * 100  # 100 epochs
                
                with tqdm(total=total_batches, desc='Training', 
                        file=sys.stdout, dynamic_ncols=True) as pbar:
                    for epoch in range(100):
                        train_loss, train_acc = train_epoch(
                            model, train_loader, optimizer, criterion, device,
                            score_type, scheduler, epoch, 100, pbar
                        )
                        test_loss, test_acc = test(model, test_loader, criterion, device)

                        scheduler.step()
                        updated_lr = scheduler.get_lr()[0]
                        optimizer.update_base_learning_rate(updated_lr)
                        
                        if epoch % 10 == 0:
                            remaining = calculate_remaining_weights(model)
                            pbar.write(
                                f'Epoch {epoch:3d} | '
                                f'Test Acc: {test_acc:6.2f}% | '
                                f'Train Acc: {train_acc:6.2f}% | '
                                f'Remaining: {remaining:6.2f}%'
                            )
                        
                        best_acc = max(best_acc, test_acc)
                        
                        # Save checkpoint every 10 epochs
                        if epoch % 10 == 0:
                            checkpoint = {
                                'results': results,
                                'score_idx': score_idx,
                                'opt_idx': opt_idx,
                                'exp_idx': exp_idx,
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_acc': best_acc
                            }
                            save_checkpoint(checkpoint, score_type, opt_name, exp_idx)
                    
                    # After experiment completes, remove its checkpoint
                    checkpoint_file = f'checkpoints/checkpoint_{score_type}_{opt_name}_exp{exp_idx}.pth'
                    if os.path.exists(checkpoint_file):
                        os.remove(checkpoint_file)
                
                # Store results
                final_weights = calculate_remaining_weights(model)
                results[f"{opt_name}_{score_type}_exp{exp_idx}"] = {
                    'final_weights': final_weights,
                    'final_acc': best_acc
                }

                # save the results to a json file
                with open(f'results/results_all.json', 'w') as f:
                    json.dump(results, f)
            
            # Reset exp_idx when moving to next optimizer
            start_exp_idx = 0
        
        # Reset opt_idx when moving to next score type
        start_opt_idx = 0

    print("\nAll experiments completed! Generating plots...")
    plot_results(results, score_types)
    print("Plots saved in results/comparison_all.png")


def test_lasso():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    train_loader, test_loader = load_mnist()
    print("Dataset loaded: MNIST")
    
    # Single experiment setup with magnitude score
    C = 1  # LASSO hyperparameter
    lr = 0.1
    
    print(f"\nRunning LASSO Layer with:")
    print(f"Score type: Magnitude")
    print(f"C value: {C}")
    
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize Magnitude scores
    scores_dict = choose_score(MagnitudeScore, 'magnitude', model)
    scores_list = []
    for name, param in model.named_parameters():
        scores_list.append(scores_dict.get(name, torch.zeros_like(param)))

    # Initialize auxiliary variables
    vk = []
    wk = []
    zk = []
    for name, parameters in model.named_parameters():
        vk.append(torch.zeros_like(parameters))
    for name, parameters in model.named_parameters():
        wk.append(torch.zeros_like(parameters))
    for name, parameters in model.named_parameters():
        zk.append(parameters.clone())

    v0 = torch.zeros(1).to(device)
    v1 = torch.zeros(1).to(device)
    k = 0
    beta = 0.9
    beta2 = 0.999
    lr = 0.001

    # Initialize optimizer
    optimizer = LASSO_Layer(
        model.parameters(),
        model=model,
        lr=lr,
        N=600,
        C=C,
        vk=vk,
        wk=wk,
        zk=zk,
        beta=beta,
        beta2=beta2,
        v0=v0,
        v1=v1,
        k=k,
        score=scores_list,
        adam=False
    )

        # Initialize scheduler
    scheduler = CosineScheduler(
        optimizer,
        warmup_epochs=5,
        max_epochs=100,
        min_lr=1e-6,
        verbose=False
    )

    # Training loop
    best_acc = 0
    # total_batches = len(train_loader) * 100  # 100 epochs
    
    # with tqdm(total=total_batches, desc='Training', 
    #         file=sys.stdout, dynamic_ncols=True) as pbar:
    
    for epoch in range(100):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            'magnitude', scheduler, epoch, 100, pbar=None
        )
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        if epoch % 10 == 0:
            remaining = calculate_remaining_weights(model)
            print(f"Epoch {epoch}, Remaining weights: {remaining:.2f}%")
            print(f"Train Loss: {train_loss:.2f}, 
                  Train Acc: {train_acc:.2f}%, 
                  Test Loss: {test_loss:.2f}, 
                  Test Acc: {test_acc:.2f}%")
    
        best_acc = max(best_acc, test_acc)

        # Print final results
    final_weights = calculate_remaining_weights(model)
    print("\nTraining completed!")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Remaining weights: {final_weights:.2f}%")


if __name__ == "__main__":
    test_lasso() 