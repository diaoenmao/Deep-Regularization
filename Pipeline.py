import torch 
import torch.nn as nn 
import os 

from src.device import select_device 
from src.testtrain import train, test
from src.data import fetch_dataset, dataloader
from src.models.mlp import MLP 

torch.manual_seed(0)

device = select_device()

training_data, test_data = fetch_dataset("MNIST")
train_dataloader, test_dataloader = dataloader(training_data, test_data)

model = MLP(
    data_shape=(28, 28), 
    hidden_size=512, 
    scale_factor=1, 
    num_layers=3, 
    activation="relu", 
    target_size=10
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),     # which parameters to optimize
    lr=1e-3,                 # learning rate 
    momentum=0.9, 
    # weight_decay=0.1
)

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, device, lmbda=0.0)
    test(test_dataloader, model, loss_fn, device, lmbda=0.0)