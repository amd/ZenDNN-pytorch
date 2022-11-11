from torch import nn
import torch

class LinearModel(nn.Module):
    """
    AotAutogradStrategy.compile_fn ignore graph with at most 1 call nodes.
    Make sure this model calls 2 linear layers to avoid being skipped.
    """
    def __init__(self, nlayer=2):
        super().__init__()
        layers = []
        for _ in range(nlayer):
            layers.append(nn.Linear(10, 10))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def get_example_inputs(self):
        return (torch.randn(2, 10),)
