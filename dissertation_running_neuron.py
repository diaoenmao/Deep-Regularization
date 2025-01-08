import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms, utils
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
import os
import json
from torch.nn.utils import parameters_to_vector, vector_to_parameters





def Soft_Thresholding(b, u):
    z = torch.sign(b) * torch.max(torch.zeros_like(b), torch.abs(b) - u)
    return z


def calculate_pq_index(model):
    p, q = 1, 2
    all_weights = torch.cat([param.view(-1) for param in model.parameters()])
    d = all_weights.numel()


    norm_p = torch.norm(all_weights, p=1)


    norm_q = torch.norm(all_weights, p=2)


    pq_index = 1 - (d ** (1 / q - 1 / p)) * (norm_p / norm_q)
    return pq_index.item()


import torch
from torch.optim import Optimizer

def soft_thresholding(b, u):
    return torch.sign(b) * torch.max(torch.zeros_like(b), torch.abs(b) - u)

class ADMM_Adam_neuron(Optimizer):
    def __init__(self, params, lr, N, C, vk, wk, yk, zk, score):
        self.lr = lr
        self.N = N
        self.C = C
        self.vk = vk
        self.wk = wk
        self.yk = yk
        self.zk = zk
        self.score = score

        super(ADMM_Adam_neuron, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group['params']

            for w, vk_temp, yk_temp, zk_temp, wk_temp, wanda_score_1 in zip(group['params'], self.vk, self.yk, self.zk, self.wk,self.score):

                w_len = len(w.shape)

                if w_len == 4: #判断该prune哪一层，该层是CNN层，如果加了别的模型，如果多了一些别的层，这个地方可能需要重新调整，不够robust
                     #进入该层后使用cnn_neuronwise_pruning函数执行ADMM更新
                    w.data, vk_temp.data, yk_temp.data, zk_temp.data, wk_temp.data = self.cnn_neuronwise_pruning(
                        w, vk_temp, yk_temp, zk_temp, wk_temp, self.lr, self.N, self.C, wanda_score_1
                    )
                elif w_len == 2: #同上，此是全连接层
                    w.data, vk_temp.data, yk_temp.data, zk_temp.data, wk_temp.data = self.fullycont(
                        w, vk_temp, yk_temp, zk_temp, wk_temp, self.lr, self.N, self.C, wanda_score_1
                    )
                elif w_len == 1:#同上，此是所有bias参数以及batchnorm的参数
                    w.data, vk_temp.data, yk_temp.data, zk_temp.data, wk_temp.data = self.batchnorm_and_bias_pruning(
                        w, vk_temp, yk_temp, zk_temp, wk_temp, self.lr, self.N, self.C, wanda_score_1
                    )

        return loss

    def cnn_neuronwise_pruning(self, w, vk_temp, yk_temp, zk_temp, wk_temp, lr, N, C,wanda_score_1):
        shape0, _, _, _ = w.shape
        p = 1 / lr
        grad = w.grad

        qk = 0.5 * (yk_temp + zk_temp - vk_temp / p - wk_temp / p - grad / p)

        ck = torch.norm((wanda_score_1 * zk_temp).view(shape0, -1), p=1, dim=1).view(shape0, 1, 1, 1).expand_as(w) #此处有修改，在于将zk_temp修改为了wanda_score_1 * zk_temp

        dk = qk + vk_temp / p
        yita = torch.norm((wanda_score_1 * dk).view(shape0, -1), p=2, dim=1).view(shape0, 1, 1, 1).expand_as(w) + 1e-8 #同上改动
        miu = self.C * ck / self.N
        D_k = (miu * torch.mul(wanda_score_1, wanda_score_1)) / (p * (yita) ** 3) #在scorebased更新中，D_K的表述有所修改
        C_K = ((27 * D_k + 2 + ((27 * D_k + 2) ** 2 - 4) ** (1 / 2)) / 2) ** (1 / 3)
        tao_k = 1 / 3 + (1 / 3) * (C_K + 1 / C_K)

        if torch.all(dk == 0):
            fangsuo = (ck / p) ** (1 / 3)
            random_tensor = torch.randn_like(yk_temp)
            yk_temp = random_tensor * (fangsuo / torch.norm(random_tensor.view(shape0, -1), p=2, dim=1).view(shape0, 1, 1, 1).expand_as(w))
        else:
            yk_temp = torch.mul(tao_k, dk)

        b = qk + wk_temp / p
        yk_temp_norm = torch.norm(yk_temp.view(shape0, -1), p=2, dim=1).view(shape0, 1, 1, 1).expand_as(w) + 1e-8
        u = (C / N) / (p * yk_temp_norm)

        zk_temp = Soft_Thresholding(b=b,
                                    u=u)
        vk_temp = vk_temp + p * (qk - yk_temp)
        wk_temp = wk_temp + p * (qk - zk_temp)
        w = zk_temp

        return w, vk_temp, yk_temp, zk_temp, wk_temp

    def fullycont(self, w, vk_temp, yk_temp, zk_temp, wk_temp, lr, N, C, wanda_score_1): #此处同cnn_neuronwise_pruning的改动
        shape0, shape1 = w.shape
        p = 1 / lr
        grad = w.grad

        qk = 0.5 * (yk_temp + zk_temp - vk_temp / p - wk_temp / p - grad / p)

        ck = torch.norm(wanda_score_1 * zk_temp, p=1, dim=1).unsqueeze(1).expand_as(w)
        dk = qk + vk_temp / p
        yita = torch.norm(wanda_score_1 * dk, p=2, dim=1).unsqueeze(1).expand_as(w)
        miu = self.C * ck / self.N
        D_k = (miu * torch.mul(wanda_score_1, wanda_score_1)) / (p * (yita) ** 3)
        C_K = ((27 * D_k + 2 + ((27 * D_k + 2) ** 2 - 4) ** 0.5) / 2) ** (1 / 3)
        tao_k = 1 / 3 + (1 / 3) * (C_K + 1 / C_K)

        if torch.all(dk == 0):
            fangsuo = (ck / p) ** (1 / 3)
            random_tensor = torch.randn_like(yk_temp)
            yk_temp = random_tensor * (fangsuo / torch.norm(random_tensor, p=2, dim=1).unsqueeze(1).expand_as(w))
        else:
            yk_temp = torch.mul(tao_k, dk)

        b = qk + wk_temp / p
        yk_temp_norm = torch.norm(yk_temp, p=2, dim=1).unsqueeze(1).expand_as(w)
        u = (C / N) / (p * yk_temp_norm)
        zk_temp = soft_thresholding(b, u)

        vk_temp = vk_temp + p * (qk - yk_temp)
        wk_temp = wk_temp + p * (qk - zk_temp)
        w = zk_temp

        return w, vk_temp, yk_temp, zk_temp, wk_temp

    def batchnorm_and_bias_pruning(self, w, vk_temp, yk_temp, zk_temp, wk_temp, lr, N, C, wanda_score_1): #此处同cnn_neuronwise_pruning的改动
        p = 1 / lr
        grad = w.grad

        qk = 0.5 * (yk_temp + zk_temp - vk_temp / p - wk_temp / p - grad / p)

        ck = torch.norm(wanda_score_1 * zk_temp, p=1)
        dk = qk + vk_temp / p
        yita = torch.norm(wanda_score_1 * dk, p=2)
        miu = self.C * ck / self.N
        D_k = (miu * torch.mul(wanda_score_1, wanda_score_1)) / (p * (yita) ** 3)
        C_K = ((27 * D_k + 2 + ((27 * D_k + 2) ** 2 - 4) ** 0.5) / 2) ** (1 / 3)
        tao_k = 1 / 3 + (1 / 3) * (C_K + 1 / C_K)

        if torch.all(dk == 0):
            fangsuo = (ck / p) ** (1 / 3)
            random_tensor = torch.randn_like(yk_temp)
            yk_temp = random_tensor * (fangsuo / torch.norm(random_tensor, p=2))
        else:
            yk_temp = tao_k * dk

        b = qk + wk_temp / p
        u = (C / N) / (p * torch.norm(yk_temp, p=2))
        zk_temp = soft_thresholding(b, u)

        vk_temp = vk_temp + p * (qk - yk_temp)
        wk_temp = wk_temp + p * (qk - zk_temp)
        w = zk_temp

        return w, vk_temp, yk_temp, zk_temp, wk_temp

    def update_base_learning_rate(self, new_lr):
        self.lr = new_lr





# 构建卷积神经网络
class CNN(nn.Module): #
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
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = x2.view(-1, 7 * 7 * 128)
        x = self.dense(x)
        return x


class WANDA_ScoreCalculator:
    def __init__(self, model):

        self.model = model
        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                hook = module.register_forward_hook(self._get_activation(name))
                self.hooks.append(hook)

    def _get_activation(self, name):


        def hook(module, input, output):
            self.activations[name] = input[0].detach()

        return hook

    def remove_hooks(self):

        for hook in self.hooks:
            hook.remove()

    def compute_wanda_scores(self):
        """
        计算所有目标层的 WANDA Scores。

        返回:
            一个字典，字典内是按顺序摆放的每一层参数的wandascore。
        """
        wanda_scores = {}#初始化字典
        for name, activation in self.activations.items():#找出各个层的activation以及其对于层名
            layer = dict(self.model.named_modules())[name] #定位该层内容
            if isinstance(layer, nn.Conv2d):
                layer_scores = self._compute_conv_wanda_score(layer, activation)
            elif isinstance(layer, nn.Linear):
                layer_scores = self._compute_linear_wanda_score(layer, activation)
            elif isinstance(layer, nn.BatchNorm2d):
                layer_scores = self._compute_batchnorm_wanda_score(layer, activation)
            else:
                continue
            # 将每个参数的 WANDA Score 存入字典，键为完整的参数名称
            for param_name, score in layer_scores.items():
                full_param_name = name + '.' + param_name #参数名字
                wanda_scores[full_param_name] = score#参数对于的wanda score
        return wanda_scores



    def _compute_conv_wanda_score(self, layer, activation):
        weights = layer.weight.data
        bias = layer.bias.data

        out_channels, in_channels, kH, kW = weights.shape
        stride = layer.stride
        padding = layer.padding
        dilation = layer.dilation
        kernel_size = layer.kernel_size

        batch_size, in_channels_act, H, W = activation.shape
        assert in_channels_act == in_channels, "输入通道数与卷积层不匹配"

        # 使用 unfold 提取输入激活窗口
        unfolded = F.unfold(activation, kernel_size=kernel_size, dilation=dilation, padding=padding,
                            stride=stride)
        # unfolded shape: [batch_size, in_channels * kH * kW, L]
        # L 是滑动窗口的数量，即输出特征图的空间位置数量

        # 重新形状为 [batch_size, in_channels, kH, kW, L]
        L = unfolded.shape[-1]
        unfolded = unfolded.view(batch_size, in_channels, kH, kW, L)
        total_element = batch_size * L

        # 计算每个权重对应的输入激活的 L2 范数
        # l2_norm shape: [in_channels, kH, kW]
        l2_norm = torch.norm(unfolded, p=2, dim=(0,4)) / total_element # 对 batch 求 L2 范数


        # 计算权重的 WANDA Scores
        weight_scores = torch.ones_like(weights) * l2_norm.unsqueeze(0)  # Broadcasting over out_channels, GET ALL WINDOWS PARAMETERS

        # 计算偏置的 WANDA Scores
        activation_number = batch_size * H * W * in_channels
        activation_norm = torch.norm(activation, p = 2, dim=(0,1, 2, 3)) / activation_number # [in_channels]
        bias_scores = torch.ones_like(bias) * activation_norm

        # 合并到一起

        scores = {
            'weight': weight_scores,
            'bias': bias_scores
        }
        return scores

    def _compute_linear_wanda_score(self, layer, activation):
        """
        同上内容
        """
        weights = layer.weight.data  # [out_features, in_features]
        bias = layer.bias.data  # [out_features]

        out_features, in_features = weights.shape
        batch_size, in_features_act = activation.shape
        assert in_features_act == in_features, "输入特征数与全连接层不匹配"

        # 计算每个输入特征的 L2 范数 across the batch
        l2_norm = torch.norm(activation, p=2, dim=0) / batch_size # Shape: [in_features]
        a = l2_norm.unsqueeze(0)
        # 计算权重的 WANDA Scores
        weight_scores = torch.ones_like(weights) * l2_norm.unsqueeze(0)  # Broadcasting multiplication

        # 计算偏置的 WANDA Scores
        l2_norm = torch.norm(activation, p=2, dim=(0,1)) / (batch_size * in_features_act)

        bias_scores = torch.ones_like(bias) * l2_norm  # 标量

        # 保存为字典

        scores = {
            'weight': weight_scores,
            'bias': bias_scores
        }
        return scores

    def _compute_batchnorm_wanda_score(self, layer, activation):
        """
        同上，该层用于计算batchnorm的score
        """
        # 获取权重和偏置
        weight = layer.weight.data  # [num_features]
        bias = layer.bias.data      # [num_features]
        batch_size, in_channels_act, H, W = activation.shape
        # 计算每个通道的激活的 L2 范数
        l2_norm = torch.norm(activation, p=2, dim=(0, 2, 3)) / (batch_size * H * W) # [num_features]

        # 计算权重的 WANDA Scores
        weight_scores = torch.ones_like(weight) * l2_norm

        bias_scores = weight_scores
        scores = {
            'weight': weight_scores,
            'bias': bias_scores
        }

        return scores

######################################################################################################################

class GradientCollector:
    """
    该类用于计算各个参数对于自己的gradient，用于做lora prune
    """

    def __init__(self, model):
        self.model = model

    def compute_gradients(self):
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.detach()
            else:
                # 如果没有梯度，则返回一个与参数相同形状的全 1 张量
                gradients[name] = torch.ones_like(param)
        return gradients

    def compute_ones(self):
        ones_dict = {}
        for name, param in self.model.named_parameters():
            ones_dict[name] = torch.ones_like(param)
        return ones_dict
####################################################################################################################
def Score_Choosing_function(wanda_calculator,GradientCollector, Score_name):
    if Score_name == 'Wanda':
        output = wanda_calculator.compute_wanda_scores() #这个是wanda score的激活输出，并不是W*X的wanda score，我需要这个激活矩阵在优化器中做优化
    elif Score_name == 'Gradient + Activation':
        activation_norm = wanda_calculator.compute_wanda_scores()
        output = torch.abs(activation_norm * GradientCollector.compute_gradients(activation_norm)) #乘法的第二项为各个参数的gradient
    elif Score_name == 'Lora':
        print('Lora prune not finished')
        output = GradientCollector.compute_ones()
    elif Score_name == 'Normal':
        output = GradientCollector.compute_ones()
    return output

####################################################train#########################################################334


def get_Variable(x):
    x = torch.autograd.Variable(x)

    #
    return x.cuda() if torch.cuda.is_available() else x

class CScheduler:
    def __init__(self, expriment_num):
        self.expriment_num = expriment_num

    def get_c(self, experiment_number):
        return self._sine(experiment_number)

    def _sine(self, experiment_number):
        start_value = 0.01
        end_value = 10.01
        x_values = np.linspace(-np.pi / 2, np.pi / 2, self.expriment_num)
        # Generate sine values, shift and scale to range [min_value, max_value]
        sin_values = (np.sin(x_values) + 1) * (end_value - start_value) / 2 + start_value
        return sin_values[experiment_number]


def save_experiment_results(results, METRICS_DIR, MODEL_TYPE, OPTIMIZER_TYPE):
    os.makedirs(METRICS_DIR, exist_ok=True)
    results_file = os.path.join(METRICS_DIR, f'{MODEL_TYPE}_{OPTIMIZER_TYPE}_experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved experiment results to {results_file}")


def plot_experiment_results(results,METRICS_DIR, MODEL_TYPE, OPTIMIZER_TYPE):
    C_values = [result['C'] for result in results]
    accuracies = [result['accuracy'] for result in results]
    remaining_weights = [result['remaining_weights'] for result in results]
    pq_indices = [result['pq_index'] for result in results]
    save_wei_sorted = [result['save_wei_sorted'] for result in results]
    save_accwei_sorted = [result['save_accwei_sorted'] for result in results]
    save_pq_sorted = [result['save_pq_sorted'] for result in results]
    save_accpq_sorted = [result['save_accpq_sorted'] for result in results]

    C_values = C_values[0]
    accuracies = accuracies[0]
    remaining_weights = remaining_weights[0]
    pq_indices = pq_indices[0]
    save_wei_sorted = save_wei_sorted[0]
    save_accwei_sorted = save_accwei_sorted[0]
    save_pq_sorted = save_pq_sorted[0]
    save_accpq_sorted = save_accpq_sorted[0]

    plt.figure(figsize=(30, 15))

    plt.subplot(331)
    plt.plot(C_values, accuracies, linestyle='-', marker='o')
    plt.xlabel('C value')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs C value')

    plt.subplot(332)
    plt.plot(C_values, remaining_weights, linestyle='-', marker='o')
    plt.xlabel('C value')
    plt.ylabel('Remaining Weights (%)')
    plt.title('Remaining Weights vs C value')

    plt.subplot(333)
    plt.plot(C_values, pq_indices, linestyle='-', marker='o')
    plt.xlabel('C value')
    plt.ylabel('PQ Index')
    plt.title('PQ Index vs C value')

    plt.subplot(334)
    plt.plot(save_wei_sorted, save_accwei_sorted, linestyle='-', marker='o')
    plt.xlabel('Remaining Weights (%)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Remaining Weights')
    plt.gca().invert_xaxis()

    plt.subplot(335)
    plt.plot(save_pq_sorted, save_accpq_sorted, linestyle='-', marker='o')
    plt.xlabel('PQ Index')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs PQ Index')

    plt.subplot(336)
    plt.plot(accuracies, linestyle='-', marker='o')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy (%) ')

    plt.subplot(337)
    plt.plot(remaining_weights, linestyle='-', marker='o')
    plt.ylabel('Remaining Weights (%)')
    plt.title('Remaining Weights (%)')

    plt.subplot(338)
    plt.plot(pq_indices, linestyle='-', marker='o')
    plt.ylabel('PQ Index')
    plt.title('PQ Index')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{MODEL_TYPE}_{OPTIMIZER_TYPE}_experiment_results.png'))
    plt.close()
a = torch.cuda.is_available()
print(a)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, ],
        std=[0.5, ]
    )
])

# loading traning set
data_train = datasets.MNIST(root='data/', transform=transform, train=True, download=True)
n = len(data_train)
# loading testing set
data_test = datasets.MNIST(root='data/', transform=transform, train=False)

dataloader_train = DataLoader(dataset=data_train, batch_size=100, shuffle=True)

dataloader_test = DataLoader(dataset=data_test, batch_size=100, shuffle=True)

get_final = 1#取最后几个？ 4
expriment_num = 10 #30
epochs = 1 #10
save_acc = []
save_wei = []
save_pq = []
save_C = []
results = []
for exp in range(expriment_num):
    C = CScheduler(expriment_num = expriment_num).get_c(exp)
    print('C equal to:', C)
    cnn = CNN()
    if torch.cuda.is_available():
        cnn = cnn.cuda()
    loss_F = nn.CrossEntropyLoss()
    # training
    uk = 0
    k = 1
    lr0 = 0.1
    # 初始化优化器参数
    vk = []
    wk = []
    yk = []
    zk = []
    ######
    list1 = []
    list2 = []
    list3 = []

    for name, parameters in cnn.named_parameters():
        para_1 = torch.zeros_like(parameters)
        vk.append(para_1)
    for name, parameters in cnn.named_parameters():
        para_1 = torch.zeros_like(parameters)
        wk.append(para_1)
    for name, parameters in cnn.named_parameters():
        para_1 = parameters.clone()
        yk.append(para_1)
    for name, parameters in cnn.named_parameters():
        para_1 = parameters.clone()
        zk.append(para_1)
    wanda_calculator = WANDA_ScoreCalculator(cnn)

    for epoch in range(epochs):
        running_loss = 0.0  # one epoch loss
        running_correct = 0.0  # acc
        print("Epoch [{}/{}]".format(epoch + 1, epochs))
        for data in dataloader_train:  # [data1,data2,.,.,data100]  [ [a1,(b1,c1)], [a2,(b2,c2)]  for _,(_,label) in enu
            torch.cuda.empty_cache()
            lr = lr0 / (1 + k / 60000)

            X_train, y_train = data
            X_train, y_train = get_Variable(X_train), get_Variable(y_train)
            outputs = cnn(X_train)

            # 计算  Scores
            gscore = GradientCollector(cnn)  # 初始化梯度收集器

            score = Score_Choosing_function(wanda_calculator=wanda_calculator, GradientCollector=gscore,
                                            Score_name='Normal')

            params_with_names = list(cnn.named_parameters())
            scores_list = []
            for name, param in params_with_names:  # 创造一个可迭代的list以便后续与优化器一起迭代score
                score_new = score.get(name, torch.zeros_like(param))
                scores_list.append(score_new)

            # 提取参数列表
            params = [param for name, param in params_with_names]

            _, pred = torch.max(outputs.data, 1)

            # 使用自定义的优化器 SGD_L1_clipping  params, lr, N, C, vk, wk, yk, zk, score
            optimizer = ADMM_Adam_neuron(params, lr=lr, N=600, C=C, vk=vk, wk=wk, yk=yk, zk=zk, score=scores_list)
            optimizer.zero_grad()

            loss = loss_F(outputs, y_train)
            # 反向传播
            loss.backward()
            # 更新参数

            optimizer.step()

            running_loss += loss.item()
            running_correct += torch.sum(pred == y_train.data)
            k = k + 1

        testing_correct = 0.0
        with torch.no_grad():
            for data in dataloader_test:
                X_test, y_test = data
                X_test, y_test = get_Variable(X_test), get_Variable(y_test)
                outputs = cnn(X_test)

                _, pred = torch.max(outputs, 1)
                testing_correct += torch.sum(pred == y_test.data)
            print("Loss: {:.4f}  Train Accuracy: {:.4f}%  Test Accuracy: {:.4f}%".format(
                running_loss / len(data_train), 100 * running_correct / len(data_train),
                100 * testing_correct / len(data_test)))
            non = 0
            total = sum([param.nelement() for param in cnn.parameters()])
            for para in cnn.parameters():
                non_zero = torch.count_nonzero(para).item()
                non = non + non_zero
            list1.append((non) / total) #remaining weight
            print('this epoch the remaining weight is:',(non) / total)
            acc = 100 * testing_correct / len(data_test)
            acc = acc.tolist()
            list2.append(acc)
            list3.append(calculate_pq_index(cnn))




        # 在每个 epoch 结束时，移除 hooks，并重新注册，以防止内存泄漏，但我这只写了activation的防泄露，没写gradient的
        wanda_calculator.remove_hooks()
        wanda_calculator = WANDA_ScoreCalculator(cnn)
    exp_acc = np.array(list2[-get_final:]).mean()
    exp_wei = np.array(list1[-get_final:]).mean()
    exp_pq = np.array(list3[-get_final:]).mean()
    save_acc.append(exp_acc)
    save_wei.append(exp_wei)
    save_pq.append(exp_pq)
    save_C.append(C)


# 将两个列表打包成元组列表，并按 save_wei 的值从大到小排序
combined1 = sorted(zip(save_wei, save_acc), reverse=True)
combined2 = sorted(zip(save_pq, save_acc))

# 解包回到两个排序后的列表
save_wei_sorted, save_accwei_sorted = zip(*combined1)
save_pq_sorted, save_accpq_sorted = zip(*combined2)

# 转换为列表类型（因为 zip 返回的是元组）
save_wei_sorted = list(save_wei_sorted)
save_accwei_sorted= list(save_accwei_sorted)
save_pq_sorted = list(save_pq_sorted)
save_accpq_sorted= list(save_accpq_sorted)

print(save_wei_sorted )

results.append({
         'C': save_C,
         'accuracy': save_acc,
         'remaining_weights': save_wei,
         'pq_index': save_pq,
         'save_wei_sorted': save_wei_sorted,
         'save_accwei_sorted': save_accwei_sorted,
         'save_pq_sorted': save_pq_sorted,
         'save_accpq_sorted': save_accpq_sorted

        })

SAVE_DIR = 'results'
OPTIMIZER_TYPE = 'ADMM_neuron_magnitude'
MODEL_TYPE = 'cnn3'
METRICS_DIR = os.path.join(SAVE_DIR, 'metrics')
PLOTS_DIR = os.path.join(SAVE_DIR, 'plots')

save_experiment_results(results=results, METRICS_DIR=METRICS_DIR, MODEL_TYPE = MODEL_TYPE, OPTIMIZER_TYPE = OPTIMIZER_TYPE )
plot_experiment_results(results = results, METRICS_DIR=METRICS_DIR, MODEL_TYPE = MODEL_TYPE, OPTIMIZER_TYPE = OPTIMIZER_TYPE)