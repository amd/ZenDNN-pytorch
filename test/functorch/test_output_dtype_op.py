# Owner(s): ["module: functorch"]
import torch
from functorch.experimental.output_dtype import output_dtype
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._dynamo.eval_frame import export


class TestOutputDtypeOp(TestCase):
    def test_output_dtype_op(self):
        from torch.testing import FileCheck

        class M(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                return output_dtype(
                    torch.ops.aten.mm.default, torch.int32, x, self.weight
                )

        weight = torch.randn(5, 5).char()
        m = M(weight)
        x = torch.randn(3, 5).char()

        gm, _ = export(
            m,
            x,
            aten_graph=True,
        )
        FileCheck().check("torch.ops.output_dtype").check("aten.mm.default").run(gm.code)
        self.assertTrue(torch.allclose(m(x), gm(x)))
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is output_dtype:
                # Result of this node should be int32
                self.assertTrue(node.meta["val"].dtype, torch.int32)
                # Argument of this node should be int8
                self.assertTrue(node.args[2].meta["val"].dtype, torch.int8)

        gm, _ = export(
            m,
            x,
            aten_graph=True,
            functionalize=True,
        )
        FileCheck().check("torch.ops.output_dtype").check("aten.mm.default").run(gm.code)
        self.assertTrue(torch.allclose(m(x), gm(x)))
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is output_dtype:
                # Result of this node should be int32
                self.assertTrue(node.meta["val"].dtype, torch.int32)
                # Argument of this node should be int8
                self.assertTrue(node.args[2].meta["val"].dtype, torch.int8)

if __name__ == '__main__':
    run_tests()
