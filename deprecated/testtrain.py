import torch
from torch.linalg import vector_norm
from torch.optim import LBFGS
from src.traintest.regularizers import * 

def L2diff(params1, params2): 
    
    return torch.norm(torch.stack([torch.norm(params1[i] - params2[i]) for i in range(len(params1))]))

def proximalPQI(model, device, p, q, lmbda, tau): 

    d = 0
    for param in model.parameters(): 
        d += math.prod(param.size())
        
    u = [W.data.detach().clone() for W in model.parameters()]
        
    def objective_function(model): 
        p_norm = torch.norm(torch.stack([torch.norm(param, p) for param in model.parameters()]), p)
        q_norm = torch.norm(torch.stack([torch.norm(param, q) for param in model.parameters()]), q)
        l2_diff = torch.norm(torch.stack([torch.norm([W for W in model.parameters()][i] - u[i]) for i in range(len(u))]))
        output = lmbda * (d ** ((1/q) - (1/p)) * (p_norm/q_norm)) + 1/(2*tau) * l2_diff ** 2
        return  output.to(device)

    # optimizer = LBFGS(model.parameters(), lr=5 * lmbda*tau)
    optimizer = torch.optim.SGD(model.parameters(), lr=lmbda*tau * 0.1, momentum=0.0)

    def closure():
        optimizer.zero_grad()
        loss = objective_function(model)
        loss.backward()
        return loss

    old_value = objective_function(model).item()
    for i in range(500): 
        positives1 = []
        negatives1 = []
        for W in model.parameters(): 
            mask1 = torch.where(W.data >= 0, 1.0, 0.0)
            mask2 = torch.where(W.data <= 0, 1.0, 0.0)
            positives1.append(mask1) 
            negatives1.append(mask2)
            
        optimizer.step(closure)
        new_value = objective_function(model).item()
        
        positives2 = [] 
        negatives2 = [] 
        for W in model.parameters(): 
            mask1 = torch.where(W.data >= 0, 1.0, 0.0)
            mask2 = torch.where(W.data <= 0, 1.0, 0.0)
            positives2.append(mask1) 
            negatives2.append(mask2)
            
        clip_mask = [] 
        for i in range(len(positives1)): 
            clip = positives1[i] * positives2[i] + negatives1[i] * negatives2[i]
            clip_mask.append(clip)
        
        with torch.no_grad(): 
            i = 0
            for name, param in model.named_parameters(): 
                param.copy_(clip_mask[i] * param)
                i += 1 
        
        if i % 1 == 0: 
            # print("value", new_value)
            pass 
        # if new_value >= old_value: 
        #     new_value = old_value
        #     print(i)
        #     break
        
    with torch.no_grad(): 
        for name, param in model.named_parameters(): 
            clipped_param = param.clone()
            clipped_param[torch.logical_and(-1e-0 * lmbda*tau < clipped_param, clipped_param < 1e-0 * lmbda*tau)] = 0.0
            param.copy_(clipped_param)
        
def proximalL1(model, device, lmbda, tau): 
    u = [W.data.detach().clone() for W in model.parameters()]
        
    def objective_function(model): 
        one_norm = torch.norm(torch.stack([torch.norm(param, 1) for param in model.parameters()]), 1)
        # print("one_norm", one_norm.item())
        l2_diff = torch.norm(torch.stack([torch.norm([W for W in model.parameters()][i] - u[i]) for i in range(len(u))]))
        # print("l2_diff", l2_diff.item() ** 2)
        output = 1/(2*tau) * l2_diff ** 2 + lmbda * one_norm 
        return output.to(device)

    
    # with torch.no_grad(): 
    #     for name, param in model.named_parameters(): 
    #         param.copy_(0.01 * torch.randn_like(param))
    
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=lmbda*tau * 0.1, tolerance_change=1e-6)
    optimizer = torch.optim.SGD(model.parameters(), lr=lmbda*tau * 0.1, momentum=0.0)

    def closure():
        optimizer.zero_grad()
        loss = objective_function(model)
        loss.backward()
        return loss

    old_value = objective_function(model).item()
    for i in range(500): 
        positives1 = []
        negatives1 = []
        for W in model.parameters(): 
            mask1 = torch.where(W.data >= 0, 1.0, 0.0)
            mask2 = torch.where(W.data <= 0, 1.0, 0.0)
            positives1.append(mask1) 
            negatives1.append(mask2)
            
        optimizer.step(closure)
        new_value = objective_function(model).item()
        
        positives2 = [] 
        negatives2 = [] 
        for W in model.parameters(): 
            mask1 = torch.where(W.data >= 0, 1.0, 0.0)
            mask2 = torch.where(W.data <= 0, 1.0, 0.0)
            positives2.append(mask1) 
            negatives2.append(mask2)
            
        clip_mask = [] 
        for i in range(len(positives1)): 
            clip = positives1[i] * positives2[i] + negatives1[i] * negatives2[i]
            clip_mask.append(clip)
        
        with torch.no_grad(): 
            i = 0
            for name, param in model.named_parameters(): 
                param.copy_(clip_mask[i] * param)
                i += 1 
        
        if i % 1 == 0: 
            # print("value", new_value)
            pass 
        # if new_value >= old_value: 
        #     new_value = old_value
        #     print(i)
        #     break
        
    with torch.no_grad(): 
        for name, param in model.named_parameters(): 
            clipped_param = param.clone()
            clipped_param[torch.logical_and(clipped_param >-1e-0 * lmbda*tau, clipped_param < 1e-0 * lmbda*tau)] = 0.0
            param.copy_(clipped_param)
            
def soft_threshold(param:torch.tensor, lmbda:torch.tensor): 
    x = torch.sign(param) * torch.max(torch.abs(param) - lmbda, torch.zeros_like(param))
    return x


def helperFunc(param1, param2, param3): 

    return 

def admmL1(model, device, lmbda, rho): 
    
    u = [W.data.detach().clone() for W in model.parameters()]
    
    z = [W.data.detach().clone() for W in model.parameters()]
        
    def objective_function(model): 
        
        # update theta by minimizing ridge regression problem
        
        squaredLoss = 0.0
        ridge = 0.0 
        output = 0.5 * squaredLoss + (rho/2) * ridge
        return  output.to(device)

    tau = 0.0
    optimizer = LBFGS(model.parameters(), lr=lmbda*tau)

    def closure():
        optimizer.zero_grad()
        loss = objective_function(model)
        loss.backward()
        return loss

    old_value = objective_function(model).item()
    for i in range(500): 
        optimizer.step(closure)
        new_value = objective_function(model).item()
        if i % 1 == 0: 
            print(new_value)
            pass 
        if new_value >= old_value: 
            new_value = old_value
            break

def train(dataloader, model, loss_fn, optimizer, device, l1:float=0.0, l2:float=0.0, pqi:float=0.0, t:int=0, soft_thresh = True):
    
    lr = optimizer.state_dict()["param_groups"][0]["lr"]
    
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
        
        if t != 0: 
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if l1 != 0.0:
                # L1_regularizer(model, device, optimizer, l1)
                if soft_thresh == True: 
                    with torch.no_grad(): 
                        for name, param in model.named_parameters(): 
                            param.copy_(soft_threshold(param, lr * torch.tensor(l1)))
                else: 
                    proximalL1(model, device, lmbda=l1, tau=lr)
                    
            if l2 != 0.: 
                L2_regularizer(model, device, optimizer, l2)
            if pqi != 0.: 
                proximalPQI(model, device, 1, 2, pqi, tau=lr)

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