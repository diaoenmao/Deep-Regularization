import matplotlib.pyplot as plt 

class Logger: 
    
    def __init__(self, path:str): 
        self.path = path
        self.train_losses = [] 
        self.test_losses = [] 
        self.test_accuracies = [] 
        self.PQI_sparsities = [] 
        self.L0_sparsities = [] 
        
    def push(self, train_loss, test_loss,test_accuracy, PQI_sparsity, L0_sparsity): 
        self.train_losses.append(train_loss) 
        self.test_losses.append(test_loss) 
        self.test_accuracies.append(test_accuracy)
        self.PQI_sparsities.append(PQI_sparsity) 
        self.L0_sparsities.append(L0_sparsity)
        
        
def plot_loggers(loggers:list[Logger]): 
    epochs = len(loggers[0].train_losses)
    
    plt.plot(range(1, epochs+1), loggers[0].test_accuracies, marker="+", label="0 reg.", linewidth=1)
    plt.plot(range(1, epochs+1), loggers[1].test_accuracies, marker="+", label="L1=0.01 softthresh", linewidth=1)
    plt.plot(range(1, epochs+1), loggers[2].test_accuracies, marker="+", label="L1=0.01 prox", linewidth=1)
    # plt.plot(range(1, epochs+1), loggers[3].test_accuracies, marker="+", label="PQI=0.1 prox", linewidth=1)
    # plt.plot(range(1, epochs+1), loggers[4].test_accuracies, marker="+", label="PQI=1.0 prox", linewidth=1)
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy on MNIST w/ Linear Model")
    plt.legend()
    plt.show() 