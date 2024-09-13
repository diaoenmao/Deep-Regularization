import numpy as np

class CScheduler:
    def __init__(self, strategy, **kwargs):
        self.strategy = strategy
        self.kwargs = kwargs
        self.total_experiments = kwargs.get('total_experiments', 10)

    def get_c(self, experiment_number):
        if self.strategy == 'constant':
            return self._constant()
        elif self.strategy == 'linear':
            return self._linear(experiment_number)
        elif self.strategy == 'cosine':
            return self._cosine(experiment_number)
        elif self.strategy == 'sine':
            return self._sine(experiment_number)
        else:
            raise ValueError(f"Unknown C scheduling strategy: {self.strategy}")

    def _constant(self):
        return self.kwargs.get('c_value', 1.0)

    def _linear(self, experiment_number):
        start_value = self.kwargs.get('start_value', 0.01)
        end_value = self.kwargs.get('end_value', 2.01)
        return start_value + (end_value - start_value) * (experiment_number / self.total_experiments)

    def _cosine(self, experiment_number):
        start_value = self.kwargs.get('start_value', 0.01)
        end_value = self.kwargs.get('end_value', 2.01)
        return end_value + 0.5 * (start_value - end_value) * (1 + np.cos(np.pi * experiment_number / self.total_experiments))

    def _sine(self, experiment_number):
        start_value = self.kwargs.get('start_value', 0.01)
        end_value = self.kwargs.get('end_value', 2.01)
        return start_value + (end_value - start_value) * np.sin(np.pi * experiment_number / (2 * self.total_experiments))