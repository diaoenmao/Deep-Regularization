from src.config import cfg


def set_hyperparameters():
    data_shape = {
        'MNIST': {"input" : [1, 28, 28], "target": 10}, 
        'FashionMNIST': {"input" : [1, 28, 28], "target": 10}, 
        'SVHN': {"input" : [3, 32, 32], "target": 10}, 
        'CIFAR10': {"input" : [3, 32, 32], "target": 10}, 
        'CIFAR100': {"input" : [3, 32, 32], "target": 100}
    }
    cfg["data_name"] = cfg["control"]["data_name"]
    
    cfg['data_shape'] = data_shape[cfg['data_name']]["input"]
    cfg["target_shape"] = data_shape[cfg['data_name']]["target"]
    
    cfg["model_name"] = cfg["control"]["model_name"]
    
    if cfg["model_name"] == "mlp": 
        cfg["hyperparameters"] = {
            'hidden_size': 128, 
            'scale_factor': 2, 
            'num_layers': 2, 
            'activation': 'relu'
        }
    elif cfg["model_name"] == "cnn": 
        cfg["hyperparameters"] = {
            'hidden_size': [64, 128, 256, 512]
        }
    elif cfg["model_name"] == "linear": 
        cfg["hyperparameters"] = {}
        
    cfg["hyperparameters"]['shuffle'] = {'train': True, 'test': False}
    cfg["hyperparameters"]['optimizer_name'] = 'SGD'
    cfg["hyperparameters"]['lr'] = 1e-3
    cfg["hyperparameters"]['momentum'] = 0.9
    cfg["hyperparameters"]['weight_decay'] = 0.0
    # cfg["hyperparameters"]['nesterov'] = True
    cfg["hyperparameters"]['scheduler_name'] = 'CosineAnnealingLR'
    cfg["hyperparameters"]['num_epochs'] = 100
    cfg["hyperparameters"]['batch_size'] = {'train': -1, 'test': -1}
        
    cfg["regularizer_name"] = cfg["control"]["regularizer_name"]    
    
    if cfg["regularizer_name"] in ["none", "l1softthreshold", "l1proximal", "l1admm", "l1sgd_naive", "l2", "pqiproximal", "pqiadmm"]: 
        cfg["regularization_parameters"] = {
            "lambda" : float(cfg["control"]["lambda"]), 
            "tau" : cfg["hyperparameters"]['lr'], 
            "p" : float(cfg["control"]["p"]), 
            "q" : float(cfg["control"]["q"]), 
            "reg_optimizer" : cfg["control"]["reg_optimizer"], 
            "reg_initialization" : cfg["control"]["reg_initialization"], 
            "clipping_scale" : float(cfg["control"]["clipping_scale"]), 
            "line_crossing" : eval(cfg["control"]["line_crossing"])
        }
    else: 
        raise Exception("Not a viable regularization name. ")
    
    
