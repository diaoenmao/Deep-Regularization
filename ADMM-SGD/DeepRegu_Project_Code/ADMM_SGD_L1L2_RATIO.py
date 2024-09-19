import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
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
import torch
import pandas as pd

###########import our model##########################33
from OPT.ADMM_Ratio_Optimizer import SGD_L1_clipping
from archs.mnist.cnn3 import cnn3


a = torch.cuda.is_available()
print(a)
epochs = 4

transform = transforms.Compose([
    transforms.ToTensor(),  # to tensor
    transforms.Normalize(  # standarize
        mean=[0.5, ],  # exp
        std=[0.5, ]  # std
    )
])

# loading traning set
data_train = datasets.MNIST(root='data/', transform=transform, train=True, download=True)
n = len(data_train)
# loading testing set
data_test = datasets.MNIST(root='data/', transform=transform, train=False)


dataloader_train = DataLoader(dataset=data_train, batch_size=100, shuffle=True)

dataloader_test = DataLoader(dataset=data_test, batch_size=100, shuffle=True)


def calculate_pq_index(model):

    p, q = 1, 2
    all_weights = torch.cat([param.view(-1) for param in model.parameters()])
    d = all_weights.numel()

    # Calculate ||w||_p for p = 1
    norm_p = torch.norm(all_weights, p = 1)

    # Calculate ||w||_q for q = 2
    norm_q = torch.norm(all_weights, p = 2)

    # Calculate PQ Index
    pq_index = 1 - (d ** (1 / q - 1 / p)) * (norm_p / norm_q)
    return pq_index.item()

def get_Variable(x):
    x = torch.autograd.Variable(x)

    #
    return x.cuda() if torch.cuda.is_available() else x

def different_C(method, epoch, C = 1):
    if method == "fixed":
        C = C
    if method == "step":
        if epoch <= 500:
            C = 0.01
        elif epoch > 500 and epoch <= 600:
            C = 0.05
        elif epoch > 600 and epoch <= 700:
            C = 0.1
        elif epoch > 700 and epoch <= 750:
            C = 0.5
        elif epoch > 750 and epoch <= 800:
            C = 1
        elif epoch > 800:
            C = 2
    return C


####################################################train#########################################################334
# training and opt


# define a network
cnn = cnn3()

# using GPU
if torch.cuda.is_available():
    cnn = cnn.cuda()

# loss is CrossEntropyLoss
loss_F = nn.CrossEntropyLoss()


#optimizer = SGD_L1_clipping(cnn.parameters(), lr = 0.01, N = 60000, C = 10)
#optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)
#optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01)

# training
k = 1
lr0 = 0.1 #LEARNING RATE
vk = [] #ADMM PARAMETER
wk = []#ADMM PARAMETER
yk = []#ADMM PARAMETER
zk = []#ADMM PARAMETER
list1 = [] # save remaining weight
list2 = [] # save accuracy
list3 = [] # save PQINDEX


for name, parameters in cnn.named_parameters():
    para_1 = torch.zeros_like(parameters)
    vk.append(para_1)
for name, parameters in cnn.named_parameters():
    para_1 = torch.zeros_like(parameters)
    wk.append(para_1)
for name, parameters in cnn.named_parameters():
    para_1 = parameters
    yk.append(para_1)
for name, parameters in cnn.named_parameters():
    para_1 = parameters
    zk.append(para_1)


####################################################################################################

for epoch in range(epochs):
    running_loss = 0.0  # one epoch loss
    running_correct = 0.0  # acc
    print("Epoch [{}/{}]".format(epoch, epochs))
    for data in dataloader_train:
        lr = lr0 / (1 + k/ 60000)

        X_train, y_train = data
        X_train, y_train = get_Variable(X_train), get_Variable(y_train)
        outputs = cnn(X_train)
        _, pred = torch.max(outputs.data, 1)

        #using our ADMM-SGD
        #using different C

        C = different_C(method = "step", epoch = epoch)



        optimizer = SGD_L1_clipping(cnn.parameters(), lr = lr, N=600, C = C, vk = vk, wk = wk, yk = yk, zk = zk)
        optimizer.zero_grad()

        loss = loss_F(outputs, y_train)
        # GET LOSS
        loss.backward()
        # BACKPRO
        optimizer.step()  #optimize
        running_loss += loss.item()  #
        running_correct += torch.sum(pred == y_train.data)
        k = k + 1

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
    total = sum([param.nelement() for param in cnn.parameters()])#总共多少个参数
    for para in cnn.parameters():#从中逐一提取参数张量
        non_zero = torch.count_nonzero(para).item() #计算该层中非0参数个数
        non = non + non_zero #累加，for循环结束后，non表示总共网络中非零的个数
    list1.append((total-non)/total)#全部-非0=零的个数，除以total，为0参数的占比
    acc = 100 * testing_correct / len(data_test)
    acc = acc.tolist()
    list2.append(acc)

    list3.append(calculate_pq_index(cnn))
list1 = ((1-np.array(list1)) * 100).tolist()#1-0参数的占比=remaining weight，乘以100转化为比率

plt.plot(list1)
plt.xlabel("epoch",size=20)
plt.title("Percent of Remaining Weights")
plt.savefig(f"{os.getcwd()}/plots/ADMM/Remaining_Weights_vs_epoch.png", dpi=1200)
plt.close()

plt.plot(list2)
plt.xlabel("epoch",size=20)
plt.title('Test Accuracy')
plt.savefig(f"{os.getcwd()}/plots/ADMM/Test Accuracy_vs_epoch.png", dpi=1200)
plt.close()

plt.plot(list3)
plt.xlabel("epoch",size=20)
plt.title("PQINDEX")
plt.savefig(f"{os.getcwd()}/plots/ADMM/PQINDEX_vs_epoch.png", dpi=1200)
plt.close()

list_new = (np.round(list1,3)).tolist()
plt.plot(list2)
plt.xticks(range(len(list_new)), list_new, rotation ="vertical")
plt.title("Accuracy VS Pruning")
plt.xlabel("remaning weight",size=20)
plt.ylabel("test accuracy",size=20)
plt.savefig(f"{os.getcwd()}/plots/ADMM/Accuracy_VS_Pruning.png", dpi=1200)
plt.close()

'''
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
axs[2].set_ylim(0, 1)  # 设置y轴的范围为0到1
'''