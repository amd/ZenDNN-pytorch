import torch
from torch.utils._python_dispatch import TorchDispatchMode
import functorch
from functorch.compile import aot_function, aot_module, draw_graph, print_compile
import torch.utils.checkpoint


def print_rng_seed_and_offset():
    state = torch.cuda.get_rng_state()
    seed = state[800:808].view(dtype=torch.int64)
    offset = state[808:].view(dtype=torch.int64)
    print(f"seed={seed}, offset={offset}", flush=True)




def test_rst_state_in_between():
    def fn(x):
        state = torch.cuda.get_rng_state()
        x = torch.sin(x)
        x = x + torch.rand(4, device="cuda")
        torch.cuda.set_rng_state(state)
        x = x + torch.rand(4, device="cuda")
        return x

    x = torch.randn(4, device="cuda")

    aot_mod = aot_function(fn, print_compile)
    aot_mod(x)


def test_negative_testing():
    torch.manual_seed(10)
    bad_state = torch.cuda.get_rng_state()
    def fn(x):
        torch.manual_seed(20)
        x = torch.sin(x)
        x = x + torch.rand(4, device="cuda")
        torch.cuda.set_rng_state(bad_state)
        x = x + torch.rand(4, device="cuda")
        return x

    x = torch.randn(4, device="cuda")

    aot_mod = aot_function(fn, print_compile)
    try:
        aot_mod(x)
        assert False
    except NotImplementedError:
        pass


def test_checkpointing():

    @torch._dynamo.allow_in_graph
    class MockModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward_impl_(self, x):
            a = torch.rand(4, device="cuda") + torch.sin(x)
            a = torch.rand(4, 4, device="cuda").sum(axis=0) + torch.sin(a)
            a = torch.rand(4, device="cuda") + torch.sin(a)
            a = torch.nn.functional.dropout(a)
            return a

        def forward(self, x):
            return torch.utils.checkpoint.checkpoint(self.forward_impl_, x, use_reentrant=False, preserve_rng_state=True)

    mod = MockModule()


    def fn(x):
        a = torch.sigmoid(x)
        a = mod(x)
        return a

    x = torch.randn(4, device="cuda", requires_grad=True)

    aot_mod = aot_function(fn, print_compile)
    print_rng_seed_and_offset()
    aot_mod(x).sum().backward()

    for _ in range(10):
        print_rng_seed_and_offset()
        aot_mod(x).sum().backward()
    # opt_mod = torch.compile(fn, backend="aot_eager_decomp_partition")
    # opt_mod(x).sum().backward()

if __name__ == "__main__":
    # test_rst_state_in_between()
    # test_negative_testing()
    test_checkpointing()