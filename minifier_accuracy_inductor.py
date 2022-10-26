import torch
import torch._dynamo as dynamo

class MockModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        for _ in range(10):
            x = torch.sin(x)
        x = torch.sigmoid(x)
        for _ in range(10):
            x = torch.cos(x)
        return x


mod = MockModule()

x = torch.randn(4, device="cuda")
ref = mod(x)

opt_mod = dynamo.optimize("inductor")(mod)
res = opt_mod(x)

torch.testing.assert_allclose(ref, res)
assert torch.allclose(ref, res)
