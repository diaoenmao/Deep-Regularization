import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector
from data.data_loader import get_dataset
from scores.score_loader import choose_score
from models.cnn import CNN
from scores.wanda import WandaScoreCalculator
from scores.lora import LoraScore
from scores.magnitude import MagnitudeScore
from utils import (
    calculate_pq_index,
    calculate_remaining_weights,
    plot_metrics,
    plot_accuracy_vs_pruning,
    generate_plots,
    save_metrics
)

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

def train_epoch(model, train_loader, optimizer, criterion, device, score_type, update_interval=100):
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
            # Get scores dictionary
            if score_type == 'wanda':
                scores_dict = choose_score(WandaScoreCalculator, score_type, model)
            elif score_type == 'lora':
                scores_dict = choose_score(LoraScore, score_type, model)
            else:  # magnitude
                scores_dict = choose_score(MagnitudeScore, score_type, model)
            
            # Convert scores dictionary to list
            params_with_names = list(model.named_parameters())
            scores_list = []
            for name, param in params_with_names:
                score_new = scores_dict.get(name, torch.zeros_like(param))
                scores_list.append(score_new)
            
            # Update optimizer's scores
            if hasattr(optimizer, 'score'):
                optimizer.score = scores_list
        
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
    return train_loss / len(train_loader), 100. * correct / total

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / len(test_loader.dataset)

def process_experiment_metrics(model, list1, list2, list3, get_final):
    """Process metrics from experiment results"""
    exp_acc = np.array(list2[-get_final:]).mean()
    exp_wei = np.array(list1[-get_final:]).mean()
    exp_pq = np.array(list3[-get_final:]).mean()
    
    # Sort results for plotting
    combined1 = sorted(zip(list1, list2), reverse=True)
    combined2 = sorted(zip(list3, list2))
    
    save_wei_sorted, save_accwei_sorted = zip(*combined1)
    save_pq_sorted, save_accpq_sorted = zip(*combined2)
    
    return {
        'accuracy': exp_acc,
        'remaining_weights': exp_wei,
        'pq_index': exp_pq,
        'save_wei_sorted': list(save_wei_sorted),
        'save_accwei_sorted': list(save_accwei_sorted),
        'save_pq_sorted': list(save_pq_sorted),
        'save_accpq_sorted': list(save_accpq_sorted)
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_mnist()
    
    optimizers = {
        'ppercent_layer': P_Percent_Layer,
        'ppercent_neuron': P_Percent_Neuron,
        'ppercent_global': P_Percent_Global,
        'lasso_neuron': LASSO_Neuron,
        'lasso_layer': LASSO_Layer,
        'lasso_global': LASSO_Global,
        'admm_global': ADMM_Global,
        'admm_neuron': ADMM_Neuron,
        'admm_layer': ADMM_Layer,
    }
    
    score_types = ['lora', 'magnitude', 'wanda']
    results = {}
    
    for opt_name, optimizer_class in optimizers.items():
        for score_type in score_types:
            print(f"\nTraining with {opt_name} optimizer and {score_type} scores")
            
            # Initialize model and lists for tracking metrics
            model = CNN().to(device)
            criterion = nn.CrossEntropyLoss()
            
            # Initialize tracking lists
            list1 = []  # remaining weights
            list2 = []  # accuracy
            list3 = []  # pq index
            get_final = 10

            model_type = 'cnn3'
            
            # Initial scores computation
            if score_type == 'wanda':
                scores_dict = choose_score(WandaScoreCalculator, score_type, model)
            elif score_type == 'lora':
                scores_dict = choose_score(LoraScore, score_type, model)
            else:  # magnitude
                scores_dict = choose_score(MagnitudeScore, score_type, model)
            
            # Convert scores dictionary to list
            params_with_names = list(model.named_parameters())
            scores_list = []
            for name, param in params_with_names:
                score_new = scores_dict.get(name, torch.zeros_like(param))
                scores_list.append(score_new)

            vk = []
            wk = []
            yk = []
            zk = []

            # Clone parameters for initialization
            for name, parameters in model.named_parameters():
                para_1 = parameters.clone()
                vk.append(para_1)
            
            for name, parameters in model.named_parameters():
                para_1 = parameters.clone()
                wk.append(para_1)
            
            for name, parameters in model.named_parameters():
                para_1 = parameters.clone()
                yk.append(para_1)
            
            for name, parameters in model.named_parameters():
                para_1 = parameters.clone()
                zk.append(para_1)

            if 'global' in opt_name:
                vk = parameters_to_vector(vk)
                wk = parameters_to_vector(wk)
                yk = parameters_to_vector(yk)
                zk = parameters_to_vector(zk)
                scores_list = parameters_to_vector(scores_list)
            
            # Initialize optimizer with appropriate parameters
            if 'admm' in opt_name:

                # print types of variables
                optimizer = optimizer_class(
                    model.parameters(),
                    model=model,
                    lr=0.0005,
                    N=60000,
                    C=0.01,
                    vk=vk,
                    wk=wk,
                    yk=yk,
                    zk=zk,
                    beta=0.9,
                    beta2=0.999,
                    v0=torch.zeros(1).to(device),
                    v1=torch.zeros(1).to(device),
                    k=0,
                    score=scores_list
                )

            elif 'ppercent' in opt_name:
                optimizer = optimizer_class(
                    model.parameters(),
                    model=model,
                    lr=0.0005,
                    p_percent=10,
                    beta=0.9,
                    beta2=0.999,
                    v0 = torch.zeros(1).to(device),
                    v1 = torch.zeros(1).to(device),
                    k = 0,
                    score=scores_list
                )
            elif 'lasso' in opt_name:
                
                optimizer = optimizer_class(
                    model.parameters(),
                    model=model,
                    lr=0.0005,
                    N=60000,
                    C=0.01,
                    vk=vk,
                    wk=wk,
                    zk=zk,
                    beta=0.9,
                    beta2=0.999,
                    v0=torch.zeros(1).to(device),
                    v1=torch.zeros(1).to(device),
                    k=0,
                    score=scores_list
                )
            else:
                raise ValueError(f"Invalid optimizer: {opt_name}")
            
            # Training loop
            best_acc = 0
            for epoch in range(10):
                train_loss, train_acc = train_epoch(
                    model, train_loader, optimizer, criterion, device,
                    score_type, update_interval=100
                )
                test_loss, test_acc = test(model, test_loader, criterion, device)
                
                # Calculate and store metrics
                remaining_weights = calculate_remaining_weights(model)
                pq_index = calculate_pq_index(model)
                
                list1.append(remaining_weights)
                list2.append(test_acc)
                list3.append(pq_index)
                
                print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Train Loss: {train_loss:.2f}, '
                        f'Test Acc: {test_acc:.2f}%, Test Loss: {test_loss:.2f}')
                print(f'Remaining weights: {remaining_weights:.2f}, PQ index: {pq_index:.2f}')
                best_acc = max(best_acc, test_acc)
            
            results[f"{opt_name}_{score_type}"] = best_acc
            
            # After training, process and save results
            metrics = process_experiment_metrics(model, list1, list2, list3, get_final)
            
            # Save metrics
            save_metrics(metrics, save_dir='results', model_type=model_type, optimizer_type=opt_name)
            
            # Generate plots
            plot_metrics(metrics)
            plot_accuracy_vs_pruning(metrics)
            generate_plots(metrics)
    
    # Print final results
    print("\nFinal Results:")
    for key, value in results.items():
        print(f"{key}: {value:.2f}%")

if __name__ == "__main__":
    main() 