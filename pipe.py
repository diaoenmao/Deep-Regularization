import torch 
import torch.nn as nn 
from pprint import pprint
from src.traintest.device import select_device 
from src.data import fetch_dataset, dataloader
from src.models.linear import Linear
from src.traintest.regularizers import * 
from src.metrics.metrics import * 
from src.traintest.train_model import train
from src.traintest.test_model import test
from src.metrics.loggers import Logger, plot_loggers
from src.hyper import set_hyperparameters
import matplotlib.pyplot as plt 
import copy
import argparse
from src.config import cfg, process_args

# Initialize the arguments with the ones in config.yml
# cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
args = vars(parser.parse_args())
process_args(args) 

def main(): 
    set_hyperparameters() 
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(len(seeds)): 
        cfg["model_tag"] = f"{i}_{cfg['data_name']}_{cfg['model_name']}"
        # pprint(cfg)
        print(f"Experiment : {cfg['model_tag']}")
        runExperiment()
    
def runExperiment(): 
    # Set seed 
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    
    # set device
    device = cfg["device"]
    
    # get t
    training_data, test_data = fetch_dataset(cfg["data_name"], verbose=bool(cfg["verbose"]))
    
    train_dataloader, test_dataloader = dataloader(
        training_data, 
        test_data, 
        train_batch_size=cfg["hyperparameters"]["batch_size"]["train"], 
        test_batch_size=cfg["hyperparameters"]["batch_size"]["test"]
    )
    
    logger = Logger()
    
    if cfg["model_name"] == "linear": 
        model = Linear(data_shape = cfg["data_shape"], target_size=10).to(cfg["device"])
    
    if cfg["hyperparameters"]["optimizer_name"] == "SGD": 
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr = cfg["hyperparameters"]["lr"], 
            momentum = cfg["hyperparameters"]["momentum"]
        )
        
    loss = nn.CrossEntropyLoss()
    regularizer = None 
    
    for t in range(cfg["hyperparameters"]["num_epochs"]): 
        train_dict = train(train_dataloader, model, loss, optimizer, device, regularizer, t)
        test_dict = test(test_dataloader, model, loss, device, regularizer, t)

        logger.push(
            train_loss = train_dict["loss"], 
            test_loss = test_dict["loss"], 
            test_accuracy = test_dict["accuracy"], 
            PQI_sparsity = PQI(model, device, 1, 2).item(), 
            L0_sparsity = L0_sparsity(model)
        )
        pprint(f"L0 Sparsity : {100 * L0_sparsity(model)}%")
        pprint(f"PQ Sparsity : {PQI(model, device, 1, 2).item()}")
    


if __name__ == "__main__":
    main() 
