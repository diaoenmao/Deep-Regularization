import torch

def calculate_pq_index(model):#该算法用来计算PQ INDEX，这是一个模型评定指标，在画图中会用到

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

def calculate_remaining_weights(model):
    non_zero = 0
    total = 0
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only consider weight parameters
            non_zero += torch.count_nonzero(param).item()
            total += param.numel()
    return 100 * non_zero / total if total > 0 else 100.0