import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs=5, max_epochs=100, 
                min_lr=1e-6, last_epoch=-1, verbose=False):
        """
        Args:
            optimizer: optimizer
            warmup_epochs: number of epochs for learning rate warmup
            max_epochs: total number of training epochs
            min_lr: minimum learning rate
            last_epoch: the index of last epoch
            verbose: print learning rate updates
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.verbose = verbose
        
        # Store initial learning rate
        self.initial_lr = optimizer.defaults['lr']
        
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Warmup phase
        if self.last_epoch < self.warmup_epochs:
            lr_scale = self.last_epoch / self.warmup_epochs
            return [self.initial_lr * lr_scale]
        
        # After warmup: cosine decay
        progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        lr = max(self.min_lr, self.initial_lr * cosine_decay)
        
        return [lr]

    def step(self):
        super().step()
        if self.verbose:
            print(f'Epoch {self.last_epoch}: lr = {self.get_lr()[0]:.6f}')
