import torch
import torch._dynamo as dynamo

class MockModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        for _ in range(10):
            x = torch.sin(x)
        x = torch.nn.functional.relu(x)
        for _ in range(10):
            x = torch.cos(x)
        return x


mod = MockModule()

opt_mod = dynamo.optimize("inductor")(mod)

x = torch.randn(4, device="cuda")
opt_mod(x)
