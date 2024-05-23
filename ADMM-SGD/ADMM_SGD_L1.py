import numpy as np
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms,utils
from re import A
import torch
import torch.nn as nn
import os
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.optimizer import required
import torch
import torch.nn as nn
from torch.nn import Sequential
from matplotlib import pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
epochs = 6
batch_size = 2

transform = transforms.Compose([
    transforms.ToTensor(),  #
    transforms.Normalize(
        mean=[0.5, ],
        std=[0.5, ]
    )
])


data_train = datasets.MNIST(root='data/', transform=transform, train=True, download=True)
n = len(data_train)

data_test = datasets.MNIST(root='data/', transform=transform, train=False)


dataloader_train = DataLoader(dataset=data_train, batch_size=64, shuffle=True)

dataloader_test = DataLoader(dataset=data_test, batch_size=64, shuffle=True)


class SGD_L1_clipping(Optimizer):

    def __init__(self, params, lr, N, C):
        self.lr = lr
        self.N = N #NUMBER OF SAMPLE
        self.C = C #CONSTANT
        super(SGD_L1_clipping, self).__init__(params, {})

    def step(self, closure=None):

        loss = None
        for group in self.param_groups:
            for w in group['params']:
                w.data = w.data - self.lr * w.grad

                w_copy = w.data
                mask1 = torch.where(w.data > 0, 1, 0.0)
                mask2 = torch.where(w.data < 0, 1, 0.0)
                w.data = mask1 * (w.data - (self.C / self.N) * self.lr) + mask2 * (w.data + (self.C / self.N) * self.lr)
                w.data = (torch.where(abs(w.data - w_copy) < abs(w_copy), 1, 0)) * w.data

        return loss



class CNN(nn.Module):
    def __init__(self):

        super(CNN, self).__init__()


        self.conv1 = Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        self.conv2 = Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        self.dense = Sequential(
            nn.Linear(7 * 7 * 128, 1024),
            nn.ReLU(),

            nn.Linear(1024, 10)
        )

    def forward(self, x):  # 正向传播
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = x2.view(-1, 7 * 7 * 128)
        x = self.dense(x)
        return x



####################################################train#########################################################334



def get_Variable(x):
    x = torch.autograd.Variable(x)  # Pytorch 的自动求导


    return x.cuda() if torch.cuda.is_available() else x


def calculate_pq_index(model):
    """
    Calculate the PQ Index based on the network weights and the correct formula:
    PQ Index = 1 - d^(1/q - 1/p) * ||w||_p / ||w||_q
    Where:
    - p = 0.5, q = 1
    - d is the total number of weights and biases in the model
    - ||w||_p is the p-norm (fractional norm) of the weights
    - ||w||_q is the q-norm (sum of absolute values) of the weights
    """
    p, q = 0.5, 1
    all_weights = torch.cat([param.view(-1) for param in model.parameters()])
    d = all_weights.numel()

    # Calculate ||w||_p for p = 0.5
    norm_p = (torch.sum(torch.abs(all_weights) ** p)) ** (1 / p)

    # Calculate ||w||_q for q = 1
    norm_q = torch.sum(torch.abs(all_weights))

    # Calculate PQ Index
    pq_index = 1 - (d ** (1 / q - 1 / p)) * (norm_p / norm_q)
    return pq_index.item()


cnn = CNN()


if torch.cuda.is_available():
    cnn = cnn.cuda()


loss_F = nn.CrossEntropyLoss()


#optimizer = SGD_L1_clipping(cnn.parameters(), lr = 0.01, N = 60000, C = 10)
#optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)
#optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01)

# 训练
list1 = []#l0 norm
list2 = []#acc
list3 = []#pqindex
for epoch in range(epochs):
    running_loss = 0.0
    running_correct = 0.0
    print("Epoch [{}/{}]".format(epoch, epochs))
    #define L1 norm ADMM-SGD
    optimizer = SGD_L1_clipping(cnn.parameters(), lr=0.1, N=60000, C=10)
    for data in dataloader_train:

        X_train, y_train = data
        X_train, y_train = get_Variable(X_train), get_Variable(y_train)
        outputs = cnn(X_train)
        _, pred = torch.max(outputs.data, 1)

        optimizer.zero_grad()

        loss = loss_F(outputs, y_train)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)

    testing_correct = 0.0

    for data in dataloader_test:
        X_test, y_test = data
        X_test, y_test = get_Variable(X_test), get_Variable(y_test)
        outputs = cnn(X_test)
        _, pred = torch.max(outputs, 1)
        testing_correct += torch.sum(pred == y_test.data)
        # print(testing_correct)
    print("Loss: {:.4f}  Train Accuracy: {:.4f}%  Test Accuracy: {:.4f}%".format(
        running_loss / len(data_train), 100 * running_correct / len(data_train),
        100 * testing_correct / len(data_test)))
    non = 0
    total = sum([param.nelement() for param in cnn.parameters()])
    for para in cnn.parameters():
        non_zero = torch.count_nonzero(para).item()
        non = non + non_zero

    list1.append((total-non)/total)
    acc = 100 * testing_correct / len(data_test)
    acc = acc.tolist()
    list2.append(acc)

    list3.append(calculate_pq_index(cnn))
list1 = (1-np.array(list1)).tolist()
plt.plot(list1)
plt.show()
plt.plot(list2)
plt.show()
plt.plot(list3)
plt.show()
'''
plt.plot(list1)
plt.show()
plt.plot(list2)
plt.show()
plt.plot(list3)
plt.show()
def split_iteration_max(list1, n):
    l = len(list1)  
    step = int(l / n)  
    b = [list1[i:i + step] for i in range(0, l, step)]
    list_new = []
    for i in b:
        a = max(i)
        list_new.append(a)
    return list_new

def split_iteration_min(list1, n):
    l = len(list1)  
    step = int(l / n)  
    b = [list1[i:i + step] for i in range(0, l, step)]
    list_new = []
    for i in b:
        a = min(i)
        list_new.append(a)
    return list_new

list1 = split_iteration_min(list1, 30)
list2 = split_iteration_max(list2, 30)
list3 = split_iteration_max(list3, 30)

# Plotting
pdf_pages = PdfPages("MNIST_CNN_Results1.pdf")
fig, axs = plt.subplots(1, 3, figsize=(15, 4))


axs[0].plot(list2, label='Test Accuracy')
axs[0].set_title('Accuracy')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Accuracy')
axs[0].legend()

axs[1].plot(list1)
axs[1].set_title('Percent of Remaining Weights')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Percent of Remaining Weights')

axs[2].plot(list3)
axs[2].set_title('PQ Index')
axs[2].set_xlabel('Iteration')
axs[2].set_ylabel('PQ Index')
axs[2].set_ylim(0, 1)  


plt.tight_layout()
pdf_pages.savefig(fig)
pdf_pages.close()
'''