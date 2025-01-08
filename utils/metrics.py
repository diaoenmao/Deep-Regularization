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

def calculate_remaining_weights(model, threshold=1e-6):
    """
    Calculate the percentage of remaining (non-zero) weights in the model.
    
    Args:
        model: PyTorch model
        threshold: Values below this threshold are considered zero (default: 1e-6)
    
    Returns:
        float: Percentage of non-zero weights (0-100)
    """
    try:
        non_zero = 0
        total = 0
        
        for name, param in model.named_parameters():
            # Get statistics for this layer
            abs_weights = torch.abs(param.data)            
            # Count non-zero weights
            layer_nonzero = torch.sum(abs_weights >= threshold).item()
            layer_total = param.numel()

            non_zero += layer_nonzero
            total += layer_total
            
        if total == 0:
            return 0.0
            
        final_percentage = 100.0 * non_zero / total

        return final_percentage
        
    except Exception as e:
        print(f"Error calculating remaining weights: {str(e)}")
        return 0.0