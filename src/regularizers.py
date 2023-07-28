import torch 

def l1_reg(model, device): 
    out = torch.tensor(0.0).to(device)
    # model.parameters() penalizes the bias terms, 
    # call model.named_parameters()
    for name, param in model.named_parameters(): 
        out += torch.norm(param, p=1)
    return out.to(device)

def l2_reg(model, device): 
    l2_Reg = None
    for W in model.parameters(): 
        if l2_Reg == None: 
            l2_Reg = W.norm(2) ** 2
        else: 
            l2_Reg += W.norm(2) ** 2
    return 0.5 * l2_Reg.to(device)

def PQI_reg(model, device): 
    pass 