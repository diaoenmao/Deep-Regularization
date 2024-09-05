from torch.optim import Optimizer
import torch
import torch.nn as nn

#####！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！


########这是传统的ADMM与SGD相结合的算法优化器，其结果是accuracy震荡非常大，而且很容易出现训练突然崩溃的问题随着 C （即pruning力度的逐渐增大，C的作用参考我推的公式，其控制的Pruning力度）
###也即是我们最原初版本的优化器

#####！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

'''   中文

这是传统的ADMM与SGD相结合的算法优化器，其结果是accuracy震荡非常大，而且很容易出现训练突然崩溃的问题随着 C （即pruning力度的逐渐增大，C的作用参考我推的公式，其控制的Pruning力度）
###也即是我们最原初版本的优化器

'''

'''  English
This is an algorithm optimizer that combines traditional ADMM (Alternating Direction Method of Multipliers) with SGD (Stochastic Gradient Descent). 
The result is significant accuracy oscillation, with the potential for sudden training collapse as the pruning intensity, 
controlled by parameter C (as per the formula I proposed), increases. This was our most original version of the optimizer.

'''

def Soft_Thresholding(b, u):
    z = torch.sign(b) * torch.max(torch.zeros_like(b), torch.abs(b)-u)
    return z

class ADMM_SGD(Optimizer):

    def __init__(self, params, lr, N, C, vk, wk, yk, zk):
        self.lr = lr
        self.N = N #NUMBER OF SAMPLE
        self.C = C #CONSTANT
        self.vk = vk
        self.wk = wk
        self.yk = yk
        self.zk = zk

        super(ADMM_SGD, self).__init__(params, {})

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