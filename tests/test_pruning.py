import os
import sys
import unittest

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from Score.get_grad import GradientCollector  # noqa: E402
from Score.score_choos import choose_score  # noqa: E402
from optimizer.ADMM_global import ADMM_Adam_global  # noqa: E402


class TestScoreSelection(unittest.TestCase):
    def test_magnitude_matches_abs_weights(self):
        """Test that 'magnitude' score returns absolute weight values."""
        model = nn.Sequential(nn.Linear(4, 2, bias=False))
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([[1.0, -2.0, 0.5, -0.5], [0.1, -0.2, 0.3, -0.4]]))

        collector = GradientCollector(model)
        wanda_dummy = type("Dummy", (), {"compute_wanda_scores": lambda self=None: {}})()
        scores = choose_score(wanda_dummy, collector, "magnitude")

        expected = torch.abs(model[0].weight.data)
        self.assertIn("0.weight", scores)
        self.assertTrue(torch.allclose(scores["0.weight"], expected))

    def test_first_order_uses_gradient_times_weight(self):
        """Test that 'first order' score computes |grad × weight|."""
        model = nn.Sequential(nn.Linear(4, 2, bias=False))
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([[1.0, -2.0, 0.5, -0.5], [0.1, -0.2, 0.3, -0.4]]))
        
        # Do a forward-backward pass to get gradients
        x = torch.randn(2, 4)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        collector = GradientCollector(model)
        wanda_dummy = type("Dummy", (), {"compute_wanda_scores": lambda self=None: {}})()
        scores = choose_score(wanda_dummy, collector, "first order")

        # First-order should be |grad × weight|
        expected = torch.abs(model[0].weight.grad * model[0].weight.data)
        self.assertIn("0.weight", scores)
        self.assertTrue(torch.allclose(scores["0.weight"], expected))


class TestADMMGlobal(unittest.TestCase):
    def test_admm_global_step_runs(self):
        model = nn.Linear(3, 1, bias=False)
        x = torch.randn(2, 3)
        y = torch.randn(2, 1)
        criterion = nn.MSELoss()

        params = list(model.parameters())
        zeros_like = [torch.zeros_like(p) for p in params]
        scores = [torch.ones_like(p) for p in params]
        opt = ADMM_Adam_global(params, lr=1e-2, N=2, C=1.0, vk=zeros_like, wk=zeros_like, yk=zeros_like, zk=zeros_like, score=scores)

        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        # simple wanda placeholder: use ones to avoid zero division
        for s in scores:
            s.fill_(1.0)

        before = model.weight.detach().clone()
        opt.step()
        after = model.weight.detach().clone()

        self.assertFalse(torch.isnan(after).any())
        self.assertFalse(torch.allclose(before, after))


if __name__ == "__main__":
    unittest.main()
