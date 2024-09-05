import numpy as np

class CScheduler:
    def __init__(self, strategy, **kwargs):
        self.strategy = strategy
        self.kwargs = kwargs

    def get_c(self, epoch):
        if self.strategy == 'constant':
            return self._constant()
        elif self.strategy == 'linear':
            return self._linear(epoch)
        elif self.strategy == 'cosine':
            return self._cosine(epoch)
        elif self.strategy == 'sine':
            return self._sine(epoch)
        else:
            raise ValueError(f"Unknown C scheduling strategy: {self.strategy}")

    def _constant(self):
        return self.kwargs.get('c_value', 1.0)

    def _linear(self, epoch):
        start_value = self.kwargs.get('start_value', 0.01)
        end_value = self.kwargs.get('end_value', 2.01)
        total_epochs = self.kwargs.get('total_epochs', 100)
        return start_value + (end_value - start_value) * (epoch / total_epochs)

    def _cosine(self, epoch):
        start_value = self.kwargs.get('start_value', 0.01)
        end_value = self.kwargs.get('end_value', 2.01)
        total_epochs = self.kwargs.get('total_epochs', 100)
        return end_value + 0.5 * (start_value - end_value) * (1 + np.cos(np.pi * epoch / total_epochs))

    def _sine(self, epoch):
        start_value = self.kwargs.get('start_value', 0.01)
        end_value = self.kwargs.get('end_value', 2.01)
        total_epochs = self.kwargs.get('total_epochs', 100)
        return start_value + (end_value - start_value) * np.sin(np.pi * epoch / (2 * total_epochs))