import torch
from torch import nn
from torch.nn import Sequential

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

        # 修改全连接层输入特征数为 7*7*128
        self.dense = Sequential(
            nn.Linear(7 * 7 * 128, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        # 展平时修改为 7*7*128
        x = x2.view(-1, 7 * 7 * 128)
        x = self.dense(x)
        return x