import torch

def calculate_pq_index(model):

    p, q = 1, 2
    all_weights = torch.cat([param.view(-1) for param in model.parameters()])

    d = all_weights.numel()

    # Calculate ||w||_p for p = 1
    norm_p = torch.norm(all_weights, p = 1)

    # Calculate ||w||_q for q = 2
    norm_q = torch.norm(all_weights, p=2) + 1e-8

    # Calculate PQ Index
    pq_index = 1 - (d ** (1 / q - 1 / p)) * (norm_p / norm_q)
    
    return pq_index.item()

def calculate_remaining_weights(model):
    """
    Calculate the percentage of remaining (non-zero) weights in the model.
    
    Args:
        model: PyTorch model
        threshold: Values below this threshold are considered zero (default: 1e-6)
    
    Returns:
        float: Percentage of non-zero weights (0-100)
    """
    total_weights = model.parameters()
    non_zero_weights = torch.sum(torch.abs(total_weights)==0)
    percentage_non_zero = (non_zero_weights / total_weights.numel()) * 100
    return percentage_non_zero.item()