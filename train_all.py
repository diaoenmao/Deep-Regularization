import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.data_loader import get_dataset
from scores.score_loader import choose_score
from models.cnn import CNN
from scores.wanda import WandaScoreCalculator
from scores.lora import LoraScore
from scores.magnitude import MagnitudeScore

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
    
    score_types = ['wanda', 'lora', 'magnitude']
    results = {}
    
    for opt_name, optimizer_class in optimizers.items():
        for score_type in score_types:
            print(f"\nTraining with {opt_name} optimizer and {score_type} scores")
            
            # Initialize model
            model = CNN().to(device)
            criterion = nn.CrossEntropyLoss()
            
            # Initial scores computation
            if score_type == 'wanda':
                # Run one batch through the model to get activations
                for data, _ in train_loader:
                    data = data.to(device)
                    model(data)
                    break
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
            
            # Initialize optimizer with appropriate parameters
            if 'admm' in opt_name:
                # Initialize ADMM variables
                vk = []
                wk = []
                yk = []
                zk = []
                
                # Initialize vk and wk with zeros
                for name, parameters in model.named_parameters():
                    vk.append(torch.zeros_like(parameters).to(device))
                    wk.append(torch.zeros_like(parameters).to(device))
                
                # Initialize yk and zk with parameter clones
                for name, parameters in model.named_parameters():
                    yk.append(parameters.clone().to(device))
                    zk.append(parameters.clone().to(device))
                
                optimizer = optimizer_class(
                    model.parameters(),
                    model=model,
                    lr=0.001,
                    N=60000,
                    C=0.0001,
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
                    lr=0.001,
                    p_percent=30,
                    score=scores_list
                )
            elif 'lasso' in opt_name:
                optimizer = optimizer_class(
                    model.parameters(),
                    model=model,
                    lr=0.001,
                    N=60000,
                    C=0.0001,
                    score=scores_list
                )
            
            # Training loop
            best_acc = 0
            for epoch in range(10):
                train_loss, train_acc = train_epoch(
                    model, 
                    train_loader, 
                    optimizer, 
                    criterion, 
                    device,
                    score_type,
                    update_interval=100  # Update scores every 100 batches
                )
                test_loss, test_acc = test(model, test_loader, criterion, device)
                
                print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
                best_acc = max(best_acc, test_acc)
            
            results[f"{opt_name}_{score_type}"] = best_acc
    
    # Print final results
    print("\nFinal Results:")
    for key, value in results.items():
        print(f"{key}: {value:.2f}%")

if __name__ == "__main__":
    main() 