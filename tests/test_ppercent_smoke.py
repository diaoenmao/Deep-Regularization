import unittest
import torch
from torch import nn

from optimizer.ppercent_neuron import Ppercent_neuron


class TinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, bias=True)
        self.head = nn.Linear(4 * 26 * 26, 2)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


def run_step(model, opt, score_list):
    x = torch.randn(2, 1, 28, 28)
    y = torch.tensor([0, 1])
    out = model(x)
    loss = nn.CrossEntropyLoss()(out, y)
    loss.backward()
    for buf, score in zip(score_list, score_list):
        buf.copy_(score)
    opt.step()
    return loss.item(), model


class TestPpercentSmoke(unittest.TestCase):
    def test_forward_backward_and_prune(self):
        model = TinyConvNet()
        params = list(model.parameters())
        scores = [torch.ones_like(p) for p in params]
        opt = Ppercent_neuron(params, lr=1e-3, p=50, score=scores)

        loss, model = run_step(model, opt, scores)
        # check no nan and some pruning happened (rough check: expect zeros)
        zeros = sum((p == 0).sum().item() for p in model.parameters())
        self.assertFalse(torch.isnan(torch.tensor(loss)))
        self.assertGreater(zeros, 0)


if __name__ == "__main__":
    unittest.main()
