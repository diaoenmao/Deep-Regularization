from src.post.plot import plot 
import os 
from pprint import pprint 

# 0  seed
# 1  data 
# 2  model 
# 3  reg_name 
# 4  lambda 
# 5  reg_optim_name 
# 6  reg_initialization 
# 7  clipping_scale
# 8  line_crossing
# 9  p 
# 10 q


if __name__ == "__main__":

    filepaths = [
        filepath for filepath in os.listdir("output") 
                 if filepath.split("_")[1] == "CIFAR10" and 
                 filepath.split("_")[2] == "linear" and 
                 (filepath.split("_")[3] == "pqiproximal" or filepath.split("_")[3] == "none" or filepath.split("_")[3] == "l1proximal") and 
                #  filepath.split("_")[4] == "0.01" and 
                 
                 filepath.split("_")[6] == "inplace" and 
                 filepath.split("_")[7] == "1.0" and 
                 filepath.split("_")[8] == "False" 
    ]

    
    plot(
        x_axis = "epochs", 
        y_axis = "test accuracies", 
        filepaths = filepaths, 
        filename="MNIST linear PQI Proximal", 
        title = "MNIST linear PQI Proximal"
    ) 