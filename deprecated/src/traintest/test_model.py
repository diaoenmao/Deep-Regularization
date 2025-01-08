import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from src.traintest.regularizers import * 
from src.metrics.metrics import * 

def test(dataloader:DataLoader, model:Module, loss_fn, device:str, regularizer = "none", t:int = 0):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y)
            
            if regularizer != "none": 
                test_loss += regularizer.penalty
            
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"L0 Sparsity : {100 * L0_sparsity(model)}%")
    print(f"PQ Sparsity : {PQI(model, device, 1, 2)}")
    
    print(f"--------------------------------------")
    
    return {"loss" : test_loss.item(), 
            "accuracy" : correct}

