from torch.optim import Optimizer
import torch
import torch.nn as nn



'''   中文

这是我们最新的优化器，其可以很好的控制住accuracy震荡的幅度，保证了模型不会崩溃，如果要与其他的baseline做对比，请使用该优化器


To Yifan

！！！！！！！！！！！！！！！！！！！！！！！！
Yifan 请注意，该代码中，我并未将neuron-scope整合进去，该代码是在layer-wise上面运行的，在与别的baseline对比时请先把该代码与我写的neuron-wise的代码相结合
！！！！！！！！！！！！！！！！！！！！！！！！

'''

'''  English
This is our latest optimizer, which can well control the amplitude of accuracy oscillation and ensure that the model will not collapse.
If you want to compare with other baselines, please use this optimizer.

'''

def Soft_Thresholding(b, u):
    z = torch.sign(b) * torch.max(torch.zeros_like(b), torch.abs(b)-u)
    return z

class ADMM_Adam_Layer(Optimizer):

    def __init__(self, params, lr, N, C, vk, wk, yk, zk, beta, beta2 ,v0, v1, k):
        self.lr = lr
        self.N = N #NUMBER OF SAMPLE
        self.C = C #CONSTANT
        self.vk = vk
        self.wk = wk
        self.yk = yk
        self.zk = zk
        self.beta = beta
        self.beta2 = beta2
        self.v0 = v0
        self.v1 = v1
        self.k = k
        super(ADMM_Adam_Layer, self).__init__(params, {})

    def step(self, closure=None):
        epi = 1e-8

        for group in self.param_groups:
            for w, vk_temp, yk_temp, zk_temp, wk_temp,v0_temp,v1_temp in zip(group['params'], self.vk, self.yk, self.zk, self.wk, self.v0,self.v1):
                grad = w.grad
                ###################################33
                v0_temp.data = self.beta * v0_temp + (1-self.beta) * grad
                v1_temp.data = self.beta2 * v1_temp + (1-self.beta) * torch.mul(grad, grad)
                v1_new = v1_temp/(1 - self.beta2 ** self.k)
                lr = self.lr/(torch.sqrt(v1_new)+epi)/(1-self.beta** self.k)

                ########################################################


                grad = v0_temp

                p = 1/ lr
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