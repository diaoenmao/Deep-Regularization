import matplotlib.pyplot as plt 
from src.post.io import load




filepaths = [
    "output/0_MNIST_linear_none_0.0_SGD_inplace_1.0_False_1.0_2.0", 
    "output/0_MNIST_linear_l1softthreshold_1.0_SGD_inplace_1.0_False_1.0_2.0", 
    "output/0_MNIST_linear_l1softthreshold_0.1_SGD_inplace_1.0_False_1.0_2.0", 
    "output/0_MNIST_linear_l1softthreshold_0.01_SGD_inplace_1.0_False_1.0_2.0"
]

plot(x_axis = "epochs", y_axis = "test accuracies", filepaths = filepaths) 