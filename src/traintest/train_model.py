from src.traintest.regularizers import * 
import time 

def train(dataloader, model, loss_fn, optimizer, device, regularizer:Regularizer = None, t:int = 0):
    start = time.time() 
    
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
            
            if regularizer is not None: 
                regularizer.step()
            
            
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    print(f"Epoch Time : {round(time.time() - start, 2)}")
            
    return {"loss" : train_loss / num_batches, 
            "accuracy" : 0.0}
         
        