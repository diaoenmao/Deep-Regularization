import torch
from torch.linalg import vector_norm
from torch.optim import LBFGS
from src.regularizers import * 

def helperFunc(params1, params2): 
    
    return torch.norm(torch.stack([torch.norm(params1[i] - params2[i]) for i in range(len(params1))]))

def proximalPQI(model, device, p, q, lmbda, tau): 

    d = 0
    for param in model.parameters(): 
        d += math.prod(param.size())
        
    u = [W.data.detach().clone() for W in model.parameters()]
        
    def objective_function(model): 
        p_norm = torch.norm(torch.stack([torch.norm(param, p) for param in model.parameters()]), p)
        q_norm = torch.norm(torch.stack([torch.norm(param, q) for param in model.parameters()]), q)
        output = lmbda * (d ** ((1/q) - (1/p)) * (p_norm/q_norm)) + 1/(2*tau) * helperFunc([W.data for W in model.parameters()], u) ** 2
        return  output.to(device)

    optimizer = LBFGS(model.parameters(), lr=0.001)

    def closure():
        optimizer.zero_grad()
        loss = objective_function(model)
        loss.backward()
        return loss

    old_value = objective_function(model).item()
    for i in range(500): 
        optimizer.step(closure)
        new_value = objective_function(model).item()
        print(new_value)
        if new_value > old_value: 
            break

    # return initial_guess

def train(dataloader, model, loss_fn, optimizer, device, l1:float=0.0, l2:float=0.0, pqi:float=0.0):
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0.0
    
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        
        
        if l1 != 0.0: 
            L1_regularizer(model, device, optimizer, l1)
        if l2 != 0.0: 
            L2_regularizer(model, device, optimizer, l2)
        if pqi != 0.0: 
            # PQI_regularizer(model, device, optimizer, pqi, 1, 2)
            proximalPQI(model, device, 1, 2, pqi, 1e-3)

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    return {"loss" : train_loss / num_batches, 
            "accuracy" : 0.0}
         
            
def test(dataloader, model, loss_fn, device, l1:float=0.0, l2:float=0.0, pqi:float=0.0):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() 
            
            if l1 != 0.0: 
                test_loss += l1 * p_norm(model, device, 1)
            if l2 != 0.0: 
                test_loss += l2 * p_norm(model, device, 2) 
            if pqi != 0.0: 
                test_loss += pqi * PQI(model, device, 1, 2)
                
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return {"loss" : test_loss, 
            "accuracy" : correct}