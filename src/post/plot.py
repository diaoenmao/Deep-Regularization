import matplotlib.pyplot as plt 
from src.post.io import * 
import os 

def plot(x_axis = "epochs", y_axis = "test accuracies", filepaths = []): 
    
    plt.figure(figsize=(10, 6))
    for filepath in filepaths: 
        logger = load(filepath) 
        X = range(len(logger.test_accuracies))
        
        if y_axis == "test accuracies": 
            Y = logger.test_accuracies
        elif y_axis == "train losses": 
            Y = logger.train_losses 
        elif y_axis == "test losses": 
            Y = logger.test_losses 
        elif y_axis == "PQI sparsities": 
            Y = logger.PQI_sparsities 
        elif y_axis == "L0_sparsities": 
            Y = logger.L0_sparsities            
            
        plt.plot(X, Y, marker="+", label=filepath.split("/")[1], linewidth=1)
        
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend() 
    plt.show() 
    plt.savefig("figs/test.png")
