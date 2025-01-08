from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from src.regularizers import * 
from src import metrics
import time 

def train(dataloader:DataLoader, model:Module, loss_fn, optimizer:Optimizer, 
          device:str, regularizer:Regularizer, t:int = 0):
  print(f"Epoch {t}\n------------------------------------")
  
  start = time.time() 
  
  size = len(dataloader.dataset)    # type: ignore
  num_batches = len(dataloader)
  
  print_checkpoint = int(num_batches / 4) 
  if print_checkpoint == 0: 
    print_checkpoint += 1

  train_loss = 0.0
  
  model.train()
  for batch_num, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    # Compute prediction error
    pred = model(X)
    loss = loss_fn(pred, y)
    train_loss += loss.item()
    
    if t != 0: 
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      regularizer.step()


    if batch_num % print_checkpoint == 0: 
      loss, current = loss.item(), (batch_num + 1) * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
  
  print(f"Epoch Time : {round(time.time() - start, 2)}s")

  return {"loss" : train_loss / num_batches, "accuracy" : 0.0}

def test(dataloader:DataLoader, model:Module, loss_fn, device:str, regularizer:Regularizer):
  size = len(dataloader.dataset)        # type: ignore 
  num_batches = len(dataloader)
  model.eval()
  test_loss, correct = Tensor([0.]).to(device), 0
  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y) + regularizer.penalty       # type: ignore
      
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss.item():>8f} \n")
  print(f"L0 Sparsity : {100 * metrics.L0_sparsity(model)}%")
  print(f"PQ Sparsity : {metrics.PQI(model, device, 1, 2)}")
  
  print(f"--------------------------------------")
  
  return {"loss" : test_loss, "accuracy" : correct}


