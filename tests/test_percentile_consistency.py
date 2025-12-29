import unittest
import torch

from optimizer.ppercent_neuron import Ppercent_Adam_Neuron


class TestPercentileConsistency(unittest.TestCase):
    def _run_and_measure(self, shape, p):
        w = torch.arange(1, torch.prod(torch.tensor(shape)) + 1, dtype=torch.float32).view(*shape)
        w.requires_grad = True
        w.grad = torch.ones_like(w)
        scores = torch.abs(w).clone()
        opt = Ppercent_Adam_Neuron([w], lr=0.0, p=p, score=[scores])
        opt.step()
        kept = (w != 0).sum().item()
        return kept / w.numel()

    def test_conv_prunes_correct_ratio(self):
        # discrete masks: tolerate one-element rounding error
        keep_ratio = self._run_and_measure((2, 2, 2, 2), p=20)  # prune 20%
        self.assertLessEqual(abs(keep_ratio - 0.8), 1 / 8)

    def test_linear_prunes_correct_ratio(self):
        keep_ratio = self._run_and_measure((3, 10), p=50)
        self.assertLessEqual(abs(keep_ratio - 0.5), 1 / 10)

    def test_bias_prunes_correct_ratio(self):
        keep_ratio = self._run_and_measure((10,), p=30)
        self.assertLessEqual(abs(keep_ratio - 0.7), 1 / 10)


if __name__ == "__main__":
    unittest.main()
