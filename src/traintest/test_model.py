import torch
from torch.optim import LBFGS
from src.traintest.regularizers import * 

def test(dataloader, model, loss_fn, device, regularizer:Regularizer = None, t:int = 0):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y)
            
            if regularizer is not None: 
                test_loss += regularizer.penalty
            
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return {"loss" : test_loss.item(), 
            "accuracy" : correct}