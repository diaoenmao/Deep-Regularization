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
a = torch.cuda.is_available()
print(a)
epochs = 20

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

def Soft_Thresholding(b, u):
    z = torch.sign(b) * torch.max(torch.zeros_like(b), torch.abs(b)-u)
    return z

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

def update_method(w, vk_temp, yk_temp, zk_temp, wk_temp, lr, N, C,w_grad):
    grad = w_grad
    p = 1 / lr
    #######################################################################################################################

    qk = (1 / 2) * (yk_temp + zk_temp - vk_temp / p - wk_temp / p - grad / p)  # have some issue?

    ###########################################################################33##########################################3
    ck = torch.norm(zk_temp, p=1)
    dk = qk + vk_temp / p
    yita = torch.norm(dk, p=2)
    D_k = ((C / N) * ck) / (p * (yita ** 3))
    C_K = ((27 * D_k + 2 + ((27 * D_k + 2) ** 2 - 4) ** (1 / 2)) / 2) ** (1 / 3)
    tao_k = 1 / 3 + (1 / 3) * (C_K + 1 / C_K)
    judge = torch.zeros_like(dk)
    if torch.equal(dk, judge):
        fangsuo = (ck / p) ** (1 / 3)
        random_tensor = torch.randn_like(yk_temp)
        yk_temp = random_tensor * (fangsuo / torch.norm(random_tensor, p=2))
    else:
        yk_temp = tao_k * dk
    ######################################3#################################################3
    b = qk + wk_temp / p
    u = (C / N) * 1 / (p * torch.norm(yk_temp, p=2))

    zk_temp = Soft_Thresholding(b=qk + wk_temp / p,
                                     u=(C / N) * 1 / (p * torch.norm(yk_temp, p=2)))  ############issue
    ###########################################################################################
    vk_temp = vk_temp + p * (qk - yk_temp)
    wk_temp = wk_temp + p * (qk - zk_temp)
    w = zk_temp
    return w, vk_temp, yk_temp, zk_temp, wk_temp


def CNN_neuronwise_pruning(w, vk, yk, zk, wk, lr, N, C):#问题在于这里的输入，不会因为update_method而改变
    inchannel = w.shape[1]
    width = w.shape[2]
    lenth = w.shape[3]
    w_grad = w.grad
    a = w[:,0,0,1]#   weight  (64,1,3,3)  filter--1
    for i in range(inchannel):
        for j in range(width):
            for k in range(lenth):
                with torch.no_grad():
                    w[:, i, j, k], vk[:, i, j, k], yk[:, i, j, k], zk[:, i, j, k], wk[:, i, j, k] = \
                        update_method(w[:, i, j, k], vk[:, i, j, k], yk[:, i, j, k], zk[:, i, j, k], wk[:, i, j, k], lr,N, C, w_grad[:, i, j, k])

    return w, vk, yk, zk, wk

def batchnorm_and_bias_pruning(w, vk, yk, zk, wk, lr, N, C):
    w_grad = w.grad
    with torch.no_grad():
        w, vk, yk, zk, wk = update_method(w, vk, yk, zk, wk, lr, N, C, w.grad)
    return w, vk, yk, zk, wk

def fullycont(w, vk, yk, zk, wk, lr, N, C):
    w_grad = w.grad
    outchannel = w.shape[0]
    for i in range(outchannel):
        with torch.no_grad():
            w[i, :], vk[i, :], yk[i, :], zk[i, :], wk[i, :] = update_method(w[i, :], vk[i, :], yk[i, :], zk[i, :], wk[i, :], lr, N, C, w_grad[i, :])
    return w, vk, yk, zk, wk


class SGD_L1_clipping(Optimizer):

    def __init__(self, params, lr, N, C, vk, wk, yk, zk):
        self.lr = lr
        self.N = N #NUMBER OF SAMPLE
        self.C = C #CONSTANT
        self.vk = vk
        self.wk = wk
        self.yk = yk
        self.zk = zk
        super(SGD_L1_clipping, self).__init__(params, {})

    def step(self, closure=None):

        for group in self.param_groups:
            for w, vk_temp, yk_temp, zk_temp, wk_temp in zip(group['params'], self.vk, self.yk, self.zk, self.wk):
                w_len = len(w.shape)
                if w_len == 4:
                    with torch.no_grad():
                        w, vk_temp, yk_temp, zk_temp, wk_temp = CNN_neuronwise_pruning(w, vk_temp, yk_temp, zk_temp, wk_temp, lr=self.lr, N=self.N, C=self.C)
                elif w_len == 2:
                    with torch.no_grad():
                        w, vk_temp, yk_temp, zk_temp, wk_temp = fullycont(w, vk_temp, yk_temp, zk_temp, wk_temp, lr=self.lr, N=self.N, C=self.C)
                elif w_len == 1:
                        w.data, vk_temp.data, yk_temp.data, zk_temp.data, wk_temp.data = batchnorm_and_bias_pruning(w, vk_temp, yk_temp, zk_temp, wk_temp, lr=self.lr, N=self.N, C=self.C)

        return None


# 构建卷积神经网络
class CNN(nn.Module):
    def __init__(self):  #
        # super()
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
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = x2.view(-1, 7 * 7 * 128)
        x = self.dense(x)
        return x



####################################################train#########################################################334
# training and opt


def get_Variable(x):
    x = torch.autograd.Variable(x)

    #
    return x.cuda() if torch.cuda.is_available() else x


# define a network
cnn = CNN()

# using GPU
if torch.cuda.is_available():
    cnn = cnn.cuda()

# loss is CrossEntropyLoss
loss_F = nn.CrossEntropyLoss()


#optimizer = SGD_L1_clipping(cnn.parameters(), lr = 0.01, N = 60000, C = 10)
#optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)
#optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01)

# training
uk = 0
k = 1
lr0 = 0.1
#########################################bulid q0 ##################################################
vk = []
list1 = []
list2 = []
list3 = []
wk = []
yk = []
zk = []

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


        optimizer = SGD_L1_clipping(cnn.parameters(), lr = lr, N=600, C=0.5, vk = vk, wk = wk, yk = yk, zk = zk)
        optimizer.zero_grad()

        loss = loss_F(outputs, y_train)
        # GET LOSS
        loss.backward()
        # BACKPRO
        optimizer.step()
        #


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
plt.xlabel("epoch",size=20)
plt.title("Percent of Remaining Weights")
plt.show()
plt.plot(list2)
plt.xlabel("epoch",size=20)
plt.title('Test Accuracy')
plt.show()
plt.plot(list3)
plt.xlabel("epoch",size=20)
plt.title("PQINDEX")
plt.show()
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