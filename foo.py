import torch
import torch._dynamo

def fn(x):
    torch._dynamo.graph_break()
    return x.sin().sum()

@torch.compile(backend='eager')
def wrapper_fn(x):
    grad_f = torch.func.grad(fn)
    result = grad_f(x)
    return result

x = torch.randn(3)
result = wrapper_fn(x)
assert torch.allclose(result, x.cos())
