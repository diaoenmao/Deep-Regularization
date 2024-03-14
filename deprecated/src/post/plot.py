import matplotlib.pyplot as plt 
from src.post.io import * 
import os 

def plot(x_axis = "epochs", y_axis = "L0 sparsities", filepaths = [], filename="test", title = ""): 
    
    plt.figure(figsize=(10, 6))
    for filepath in filepaths: 
        logger = load(os.path.join("output", filepath)) 
        X = range(len(logger.test_accuracies))
        
        if y_axis == "test accuracies": 
            Y = logger.test_accuracies
        elif y_axis == "train losses": 
            Y = logger.train_losses 
        elif y_axis == "test losses": 
            Y = logger.test_losses 
        elif y_axis == "PQI sparsities": 
            Y = [x.item() for x in logger.PQI_sparsities]
        elif y_axis == "L0 sparsities": 
            Y = logger.L0_sparsities            
            
        plt.plot(X, Y, marker="+", label=filepath, linewidth=1)
        
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    if title == "": 
        plt.title(filename)
    else: 
        plt.title(title)
    plt.legend() 
    plt.show() 
    plt.savefig(f"figs/{filename}.png")
