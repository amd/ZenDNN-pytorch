# Owner(s): ["oncall: jit"]

from torch.testing._internal.jit_utils import JitTestCase
import torch
import torch._C
from torch.testing import FileCheck
from typing import Callable

class FunctionalLinear(torch.nn.Module):
    def __init__(self, weight, bias):
        super(FunctionalLinear, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        res = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            res.add_(self.bias)
        return res

class TestGraphRewritePasses(JitTestCase):
    def check_rewrite_functional_linear(
        self, 
        old_kind: str, 
        new_kind: str, 
        check_not: list[str], 
        jit_pass: Callable[[str], None]
    ):
        x1 = torch.rand(3)
        w1 = torch.rand(5, 3)
        b1 = torch.rand(5)
        for has_bias in [True, False]:
            bias = b1 if has_bias else None
            model = torch.jit.trace(FunctionalLinear(w1, bias), [x1])
            for node in model.graph.nodes():
                if node.kind() == old_kind:
                    source_range_1 = node.sourceRange()
            jit_pass(model.graph)
            for node in model.graph.nodes():
                if node.kind() == new_kind:
                    source_range_2 = node.sourceRange()
            FileCheck().check(new_kind).run(model.graph)
            for cn in check_not:
                FileCheck().check_not(cn).run(model.graph)
            self.assertTrue(source_range_1 == source_range_2)
            model(x1)  # make sure it runs

    def test_fuse_linear(self):
        check_not = ["aten::matmul", "aten::addmm", "aten::add_", "aten::t("]
        self.check_rewrite_functional_linear("aten::matmul", "aten::linear", check_not, torch._C._jit_pass_fuse_linear)

        # check matmuls are not fused
        class Matmul(torch.nn.Module):
            def __init__(self, weight):
                super(Matmul, self).__init__()
                self.weight = weight

            def forward(self, x):
                return torch.matmul(x, self.weight)

        x = torch.rand(5, 6, 5)
        w = torch.rand(5, 5, 100)
        model = torch.jit.trace(Matmul(w), [x])
        torch._C._jit_pass_fuse_linear(model.graph)
        # check 3d matmul is not fused
        FileCheck().check("aten::matmul").run(model.graph)
        FileCheck().check_not("aten::linear").run(model.graph)
        # make sure it runs
        model(x)

    def test_vulkan_insert_pre_packed_ops(self):
        check_not = ["aten::matmul", "aten::add_", "aten::t"]
        self.check_rewrite_functional_linear("aten::matmul", "vulkan_prepack::linear_run", check_not, torch._C._jit_pass_vulkan_insert_prepacked_ops)
