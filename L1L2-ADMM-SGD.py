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
epochs = 100

transform = transforms.Compose([
    transforms.ToTensor(),  # 把数据转换为张量（Tensor）
    transforms.Normalize(  # 标准化，即使数据服从期望值为 0，标准差为 1 的正态分布
        mean=[0.5, ],  # 期望
        std=[0.5, ]  # 标准差
    )
])

# 训练集导入
data_train = datasets.MNIST(root='data/', transform=transform, train=True, download=True)
n = len(data_train)
# 数据集导入
data_test = datasets.MNIST(root='data/', transform=transform, train=False)

# 数据装载
# 训练集装载
dataloader_train = DataLoader(dataset=data_train, batch_size=100, shuffle=True)
# 数据集装载
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
                grad = w.grad
                p = 1/ self.lr
                #######################################################################################################################

                qk = (1/2) * (yk_temp + zk_temp - vk_temp/ p - wk_temp/ p - grad/ p)  #have some issue?

                ###########################################################################33##########################################3
                ck =  torch.norm(zk_temp, p = 1)
                dk = qk + vk_temp/ p
                yita = torch.norm(dk, p = 2)
                D_k = ((self.C/ self.N) *  ck)/(p * (yita ** 3))
                C_K = (( 27 * D_k + 2 +( (27 * D_k + 2) ** 2 - 4)**(1/2))/ 2) ** (1/ 3)
                tao_k = 1/ 3 +(1/ 3) * (C_K + 1/ C_K)
                judge = torch.zeros_like(dk)
                if torch.equal(dk, judge):
                    fangsuo = (ck/ p) ** (1/ 3)
                    random_tensor = torch.randn_like(yk_temp)
                    yk_temp.data = random_tensor * (fangsuo / torch.norm(random_tensor, p = 2))
                else :
                    yk_temp.data = tao_k * dk
                ######################################3#################################################3
                b = qk + wk_temp/ p
                u = (self.C / self.N) * 1 / (p * torch.norm(yk_temp, p=2))


                zk_temp.data = Soft_Thresholding(b = qk + wk_temp/ p, u =(self.C/ self.N) * 1/(p * torch.norm(yk_temp,p = 2)))############issue
                ###########################################################################################
                vk_temp.data = vk_temp + p * (qk - yk_temp)
                wk_temp.data = wk_temp + p * (qk - zk_temp)
                w.data = zk_temp

        return None


# 构建卷积神经网络
class CNN(nn.Module):  # 从父类 nn.Module 继承
    def __init__(self):  # 相当于 C++ 的构造函数
        # super() 函数是用于调用父类(超类)的一个方法，是用来解决多重继承问题的
        super(CNN, self).__init__()

        # 第一层卷积层。Sequential(意为序列) 括号内表示要进行的操作
        self.conv1 = Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第二卷积层
        self.conv2 = Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全连接层（Dense，密集连接层）
        self.dense = Sequential(
            nn.Linear(7 * 7 * 128, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, x):  # 正向传播
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = x2.view(-1, 7 * 7 * 128)
        x = self.dense(x)
        return x



####################################################train#########################################################334
# 训练和参数优化

# 定义求导函数
def get_Variable(x):
    x = torch.autograd.Variable(x)  # Pytorch 的自动求导

    # 判断是否有可用的 GPU
    return x.cuda() if torch.cuda.is_available() else x


# 定义网络
cnn = CNN()

# 判断是否有可用的 GPU 以加速训练
if torch.cuda.is_available():
    cnn = cnn.cuda()

# 设置损失函数为 CrossEntropyLoss（交叉熵损失函数）
loss_F = nn.CrossEntropyLoss()

# 设置优化器为 Adam 优化器
#optimizer = SGD_L1_clipping(cnn.parameters(), lr = 0.01, N = 60000, C = 10)
#optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)
#optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01)

# 训练
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
    running_loss = 0.0  # 一个 epoch 的损失
    running_correct = 0.0  # 准确率
    print("Epoch [{}/{}]".format(epoch, epochs))
    for data in dataloader_train:
        lr = lr0 / (1 + k/ 60000)
        # DataLoader 返回值是一个 batch 内的图像和对应的 label
        X_train, y_train = data
        X_train, y_train = get_Variable(X_train), get_Variable(y_train)
        outputs = cnn(X_train)
        _, pred = torch.max(outputs.data, 1)
        # 后面的参数代表降低 outputs.data 的维度 1 个维度再输出
        # 第一个返回值是张量中最大值，第二个返回值是最大值索引
        # -------------------下面内容与随机梯度下降类似-----------------------------
        optimizer = SGD_L1_clipping(cnn.parameters(), lr = lr, N=60000, C=10, vk = vk, wk = wk, yk = yk, zk = zk)
        optimizer.zero_grad()
        # 梯度置零
        loss = loss_F(outputs, y_train)
        # 求损失
        loss.backward()
        # 反向传播
        optimizer.step()
        # 更新参数同时得到新的qk

        # --------------------上面内容与随机梯度下降类似----------------------------
        running_loss += loss.item()  # 此处 item() 表示返回每次的 loss 值
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
    list1.append((total-non)/total) #0所占的比例
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