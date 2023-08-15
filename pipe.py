import torch 
import torch.nn as nn 
from src.data import fetch_dataset, dataloader
from src.traintest.regularizers import * 
from src.metrics.metrics import * 
from src.traintest.train_model import train
from src.traintest.test_model import test
from src.metrics.loggers import Logger, plot_loggers
from src.hyper import set_hyperparameters
from src.models.model import make_model
from src.traintest.optim import make_optimizer
from src.post.io import * 
import matplotlib.pyplot as plt 
from pprint import pprint
import argparse, os 
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
    
    pprint(cfg)
    
    # set device
    device = cfg["device"]
    
    training_data, test_data = fetch_dataset(cfg["data_name"], verbose=bool(cfg["verbose"]))
    
    train_dataloader, test_dataloader = dataloader(
        training_data, 
        test_data, 
        train_batch_size=cfg["hyperparameters"]["batch_size"]["train"], 
        test_batch_size=cfg["hyperparameters"]["batch_size"]["test"]
    )
    
    logger = Logger(cfg)
    
    model = make_model(cfg["model_name"])
    optimizer = make_optimizer(cfg["hyperparameters"]["optimizer_name"], model)
    loss = nn.CrossEntropyLoss()
    
    regularizer = make_regularizer(
        reg = cfg["regularizer_name"], 
        model = model, 
        device = cfg["device"], 
        lmbda = cfg["regularization_parameters"]["lambda"], 
        tau = cfg["regularization_parameters"]["tau"], 
        optimizer = optimizer, 
        reg_optimizer = cfg["regularization_parameters"]["reg_optimizer"], 
        reg_initialization = cfg["regularization_parameters"]["reg_initialization"], 
        clipping_scale = cfg["regularization_parameters"]["clipping_scale"], 
        line_crossing = cfg["regularization_parameters"]["line_crossing"], 
        p = cfg["regularization_parameters"]["p"], 
        q = cfg["regularization_parameters"]["q"]
    )
    
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
        print(f"L0 Sparsity : {100 * L0_sparsity(model)}%")
        print(f"PQ Sparsity : {PQI(model, device, 1, 2).item()}")
        
    
    model_tag_path = os.path.join('output', cfg['model_tag'])
    save(logger, model_tag_path)
    

if __name__ == "__main__":
    main() 
    