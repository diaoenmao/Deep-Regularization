from .mnist import cnn3 as mnist_cnn3
from .cifar10 import cnn3 as cifar10_cnn3

def get_model(model_name, dataset):
    model_mapping = {
        'mnist': {
            'cnn3': mnist_cnn3,
        },
        'cifar10': {
            'cnn3': cifar10_cnn3,
        },
    }

    try:
        return model_mapping[dataset][model_name]()
    except KeyError:
        raise ValueError(f"Unknown model: {model_name} for dataset: {dataset}")