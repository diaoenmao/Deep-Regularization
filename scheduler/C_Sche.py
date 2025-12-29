import numpy as np

class CScheduler:
    def __init__(self, expriment_num):
        self.expriment_num = expriment_num

    def get_c(self, experiment_number):
        return self._sine(experiment_number)

    def _sine(self, experiment_number):
        start_value = 0.01
        end_value = 10.01
        x_values = np.linspace(-np.pi / 2, np.pi / 2, self.expriment_num)
        # Generate sine values, shift and scale to range [min_value, max_value]
        sin_values = (np.sin(x_values) + 1) * (end_value - start_value) / 2 + start_value
        return sin_values[experiment_number]