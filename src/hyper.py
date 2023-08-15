from src.config import cfg


def set_hyperparameters():
    data_shape = {'MNIST': [1, 28, 28], 'FashionMNIST': [1, 28, 28], 'SVHN': [3, 32, 32], 'CIFAR10': [3, 32, 32],
                  'CIFAR100': [3, 32, 32]}
    
    cfg['data_shape'] = data_shape[cfg['data_name']]
    
    if cfg["model_name"] == "mlp": 
        cfg["hyperparameters"] = {'hidden_size': 128, 'scale_factor': 2, 'num_layers': 2, 'activation': 'relu'}
    elif cfg["model_name"] == "cnn": 
        cfg["hyperparameters"] = {'hidden_size': [64, 128, 256, 512]}
    elif cfg["model_name"] == "linear": 
        cfg["hyperparameters"] = {}
        
    cfg["hyperparameters"]['shuffle'] = {'train': True, 'test': False}
    cfg["hyperparameters"]['optimizer_name'] = 'SGD'
    cfg["hyperparameters"]['lr'] = 1e-1
    cfg["hyperparameters"]['momentum'] = 0.9
    cfg["hyperparameters"]['weight_decay'] = 0.0
    # cfg["hyperparameters"]['nesterov'] = True
    cfg["hyperparameters"]['scheduler_name'] = 'CosineAnnealingLR'
    cfg["hyperparameters"]['num_epochs'] = 400
    cfg["hyperparameters"]['batch_size'] = {'train': 250, 'test': 250}
    return
