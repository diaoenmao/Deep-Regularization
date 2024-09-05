import torch
import copy
from .mask import create_mask, prune_by_percentile, apply_mask
from utils import calculate_remaining_weights

class LotteryTicket:
    def __init__(self, model, prune_percent, prune_iterations):
        self.initial_state_dict = copy.deepcopy(model.state_dict())
        self.mask = create_mask(model)
        self.prune_percent = prune_percent
        self.prune_iterations = prune_iterations
        self.current_iteration = 0
        self.best_accuracy = 0
        self.best_remaining_weights = 100  # Start at 100%
        self.expected_remaining_weights = 100  # Track expected remaining weights

    def prune(self, model):
        if self.current_iteration < self.prune_iterations:
            old_remaining = calculate_remaining_weights(model)
            self.mask = prune_by_percentile(model, self.mask, self.prune_percent)
            model = self.apply_mask(model, self.mask)
            new_remaining = calculate_remaining_weights(model)
            self.expected_remaining_weights *= (1 - self.prune_percent / 100)
            print(f"Pruning iteration {self.current_iteration + 1}: "
                f"Remaining weights before: {old_remaining:.2f}%, "
                f"after: {new_remaining:.2f}%, "
                f"expected: {self.expected_remaining_weights:.2f}%")
            self.current_iteration += 1
        return model

    def apply_mask(self, model, mask):
        for name, param in model.named_parameters():
            if 'weight' in name:
                param.data.mul_(mask[name])
        return model

    def reset_weights(self, model):
        model.load_state_dict(self.initial_state_dict)
        model = apply_mask(model, self.mask)
        return model

    def get_prune_iterations(self):
        return self.prune_iterations

    def get_current_iteration(self):
        return self.current_iteration

    def update(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_remaining_weights = calculate_remaining_weights(model)