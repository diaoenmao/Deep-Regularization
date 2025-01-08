import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.models.linear import Linear 
from src.models.mlp import MLP 
from src.models.cnn import CNN 
from src.config import cfg

def make_model(model_name:str): 
    if model_name == "linear": 
        return Linear(
            data_shape = cfg["data_shape"], 
            target_size= cfg["target_shape"]
        ).to(cfg["device"])
    elif model_name == "mlp": 
        return MLP(
            data_shape = cfg["data_shape"], 
            hidden_size = cfg["hyperparameters"]["hidden_size"], 
            scale_factor = cfg["hyperparameters"]["scale_factor"], 
            num_layers = cfg["hyperparameters"]["num_layers"], 
            activation = cfg["hyperparameters"]["activation"], 
            target_size = cfg["target_shape"]
        ).to(cfg["device"])
    elif model_name == "cnn": 
        return CNN(
            data_shape = cfg["data_shape"], 
            hidden_size = cfg["hyperparameters"]["hidden_size"],
            target_size = cfg["target_shape"]
        ).to(cfg["device"])
    else: 
        raise Exception("Not a valid model name. ")