# Owner(s): ["module: sparse"]
import itertools
import random
import unittest

import torch
from torch import nn

from torch.sparse.semi_structured import (
    _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG,
    SparseSemiStructuredTensor,
)

from torch.testing import make_tensor

from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)

from torch.testing._internal.common_dtype import all_types_and_complex

from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    subtest,
    TestCase,
    TEST_WITH_ROCM
)

from torch._inductor.utils import has_triton


SEMI_STRUCTURED_SUPPORTED_DTYPES = _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG.keys()

_IS_SM8X = False
if torch.cuda.is_available():
    _IS_SM8X = torch.cuda.get_device_capability(0)[0] == 8

def rand_sparse_semi_structured_mask(
    r, c, dtype=torch.float16, device="cuda", choice=None
):
    """
    This function returns a 1:2 sparse matrix of size (r, c).
    Note that this means this matrix will also be 2:4 and 4:8 sparse as well.
    """

    choices = [[0, 1], [1, 0]]
    mask_entries = [choice or random.choice(choices) for i in range(r * c // 2)]

    return (
        torch.tensor(mask_entries, dtype=dtype, device=device)
        .reshape(r, c)
        .contiguous()
    )

def rand_dense_2by4(r, c, dtype, device, choice=None):
    choices = [
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1]
    ]
    mask_entries = [choice or random.choice(choices) for i in range(r * c // 4)]
    mask = torch.tensor(mask_entries, dtype=torch.bool).view(r, c).to(device)
    dense = make_tensor(r, c, dtype=dtype, device=device)
    dense[dense == 0] = 1  # To prevent zeros except where mask applied.
    dense = dense.masked_fill(~mask, 0)
    return (dense, mask)


class TestSparseSemiStructured(TestCase):

    @unittest.skipIf(not _IS_SM8X, "semi-structured sparsity not supported on this library version")
    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
    def test_mm_sparse_first_NT(self, dtype, device):
        """
        Ensure torch.mm(A_sparse, B) is correct for float16 and will throw error for int8
        Ensure torch.mm(A_sparse, B.t()) is correct
        """
        A = rand_sparse_semi_structured_mask(128, 128, dtype=dtype)
        A_sparse = SparseSemiStructuredTensor(A)

        B = torch.rand((128, 128), device=A_sparse.device).to(dtype)

        # Currently we don't support int matmul on GPU, so evaluate on CPU and copy over
        if dtype is torch.int8:
            # This should fail
            with self.assertRaisesRegex(RuntimeError, "two_four_sgemm_cutlass_dispatch_layouts"):
                sparse_result = torch.mm(A_sparse, B)

            # test transpose
            # NOTE: CUTLASS and cuSPARSELt have slightly different int8 behavior.
            # CUTLASS will output to an int32 tensor while cuSPARSELt will output to a int8 tensor
            dense_result = torch.mm(A.cpu(), B.t().cpu()).to(device, dtype=torch.int32)
            sparse_result = torch.mm(A_sparse, B.t())
            assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)
        else:
            dense_result = torch.mm(A, B)
            sparse_result = torch.mm(A_sparse, B)
            assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)
            # test transpose
            dense_result = torch.mm(A, B.t())
            sparse_result = torch.mm(A_sparse, B.t())
            assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @unittest.skipIf(not _IS_SM8X, "semi-structured sparsity not supported on this library version")
    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
    def test_mm_sparse_first_T(self, dtype, device):
        """
        Ensure torch.mm(A_sparse.t(), B) throws error
        """
        A = rand_sparse_semi_structured_mask(128, 128, dtype=dtype)
        A_sparse = SparseSemiStructuredTensor(A)

        B = torch.rand((128, 128), device=A_sparse.device).to(dtype)

        with self.assertRaisesRegex(
            NotImplementedError,
            r"arg0: SparseSemiStructuredTensor\(.*transposed=True",
        ):
            torch.mm(A_sparse.t(), B)

    @unittest.skipIf(not _IS_SM8X, "semi-structured sparsity not supported on this library version")
    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
    def test_mm_sparse_second_T(self, dtype, device):
        """
        Ensure torch.mm(A, B_sparse.t()) is correct
        """
        B = rand_sparse_semi_structured_mask(128, 128, dtype=dtype)
        B_sparse = SparseSemiStructuredTensor(B)

        A = torch.rand((128, 128), device=B_sparse.device).to(dtype)

        # Currently we don't support int matmul on GPU, so evaluate on CPU and copy over
        if dtype is torch.int8:
            dense_result = torch.mm(A.cpu(), B.t().cpu()).to(device, dtype=torch.int32)
            sparse_result = torch.mm(A, B_sparse.t())
        else:
            dense_result = torch.mm(A, B.t())
            sparse_result = torch.mm(A, B_sparse.t())

        assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @unittest.skipIf(not _IS_SM8X, "semi-structured sparsity not supported on this library version")
    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
    def test_mm_sparse_second_NT(self, dtype, device):
        """
        Ensure torch.mm(A, B_sparse) throws error
        """
        B = rand_sparse_semi_structured_mask(128, 128, dtype=dtype)
        B_sparse = SparseSemiStructuredTensor(B)

        A = torch.rand((128, 128), device=B_sparse.device).to(dtype)

        with self.assertRaisesRegex(
            NotImplementedError,
            r"arg1: SparseSemiStructuredTensor\(.*transposed=False",
        ):
            sparse_result = torch.mm(A, B_sparse)

    @unittest.skipIf(not _IS_SM8X, "semi-structured sparsity not supported on this library version")
    @parametrize("inference_mode", [subtest(False), subtest(True)])
    def test_linear(self, inference_mode, device):
        """
        Test nn.Linear has the same numerics
        """
        input = torch.rand(128, 128, device=device).half()
        model = nn.Linear(128, 128).to(device).half()
        m, n = model.weight.shape
        mask = rand_sparse_semi_structured_mask(m, n, device=device, dtype=torch.bool)
        # set masked weight
        model.weight = nn.Parameter(model.weight * mask)

        dense_result = model(input)

        model.weight = nn.Parameter(SparseSemiStructuredTensor(model.weight))

        if inference_mode:
            with torch.inference_mode():
                sparse_result = model(input)
        else:
            sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-5, atol=1e-5)

    @unittest.skipIf(not _IS_SM8X, "semi-structured sparsity not supported on this library version")
    def test_values(self):
        A = rand_sparse_semi_structured_mask(128, 128)
        A_sparse = SparseSemiStructuredTensor(A)
        assert A_sparse.values().shape == (128, 64)
        assert (A_sparse.values() == 1).all()

    @unittest.skipIf(not _IS_SM8X, "semi-structured sparsity not supported on this library version")
    def test_indices(self):
        A = rand_sparse_semi_structured_mask(128, 128)
        A_sparse = SparseSemiStructuredTensor(A)
        assert A_sparse.indices().shape == (128, 8)

    @unittest.skipIf(not _IS_SM8X, "semi-structured sparsity not supported on this library version")
    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
    def test_unsupported_shape(self, dtype, device):
        A = rand_sparse_semi_structured_mask(4, 4, dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "Error original_tensor.shape"):
            A_sparse = SparseSemiStructuredTensor(A)

    @unittest.skipIf(not _IS_SM8X, "semi-structured sparsity not supported on this library version")
    @dtypes(*all_types_and_complex())
    def test_unsupported_dtype(self, dtype, device):
        A = rand_sparse_semi_structured_mask(128, 128, dtype=dtype, device=device)

        if dtype not in SEMI_STRUCTURED_SUPPORTED_DTYPES:
            with self.assertRaisesRegex(RuntimeError, "Error original_tensor.dtype"):
                A_sparse = SparseSemiStructuredTensor(A)
        else:
            A_sparse = SparseSemiStructuredTensor(A)

    @unittest.skipIf(not _IS_SM8X, "semi-structured sparsity not supported on this library version")
    def test_unsupported_dim(self, device):
        A = torch.rand(128, 128, 128, device=device, dtype=torch.float16)

        with self.assertRaisesRegex(RuntimeError, "Error original_tensor.dim"):
            A_sparse = SparseSemiStructuredTensor(A)

    @unittest.skipIf(not _IS_SM8X, "semi-structured sparsity not supported on this library version")
    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support CUTLASS")
    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
    def test_linear_cutlass(self, device, dtype):
        def run_test(batch_shape, m, n, k, device, dtype, dtype_out, add_bias, activation, rtol, atol):
            weight, mask = rand_dense_2by4(m, k, dtype, device)
            input = make_tensor((*batch_shape, n, k), dtype=dtype, device=device)
            bias = make_tensor((m,), dtype=dtype_out, device=device) if add_bias else None

            dtype_dense = torch.float
            input_dense = input.to(dtype_dense)
            weight_dense = weight.to(dtype_dense)
            bias_dense = bias.to(dtype_dense) if add_bias else None
            output0 = torch.nn.functional.linear(input_dense, weight_dense, bias=bias_dense)
            if activation == "relu":
                relu = torch.nn.ReLU()
                output0 = relu(output0)
            elif activation == "silu":
                silu = torch.nn.SiLU()
                output0 = silu(output0)

            weight_sparse = weight.masked_select(weight != 0).view(m, k // 2)

            output1, meta = torch._structured_sparse_linear(input, weight_sparse, mask, bias=bias, activation=activation)
            torch.testing.assert_close(output1.to(dtype_dense), output0, rtol=rtol, atol=atol)

            output1, _ = torch._structured_sparse_linear(input, weight_sparse, meta, bias=bias, activation=activation)
            torch.testing.assert_close(output1.to(dtype_dense), output0, rtol=rtol, atol=atol)

        batch_shapes = [[], [3], [3, 1]]
        dtype_out = {torch.int8: torch.int32, torch.half: torch.half, torch.bfloat16: torch.bfloat16}
        activations = [None, "relu", "silu"]
        rtol, atol = 1e-3, 1e-3
        if dtype == torch.bfloat16:
            rtol, atol = 5e-3, 5e-3
        for batch_shape, m, n, k, add_bias, activation in \
                itertools.product(batch_shapes, range(3), range(3), range(3), (False, True), activations):
            if activation == "silu" and dtype == torch.int8:
                continue  # SiLU not supported for integer inputs

            m = 2 ** m * 32
            n = 2 ** n * 32
            k = 2 ** k * 128
            run_test(batch_shape, m, n, k, device, dtype, dtype_out[dtype], add_bias, activation, rtol, atol)

    @unittest.skipIf(not _IS_SM8X, "semi-structured sparsity not supported on this library version")
    @unittest.skipIf(not has_triton(), "Test needs triton and recent GPU arch")
    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
    def test_conversions(self, device, dtype):
        def run_test(r, c, device, dtype):
            dense_ref, _ = rand_dense_2by4(r, c, dtype, device)

            compressed = SparseSemiStructuredTensor(dense_ref)

            _, meta_ref = torch.ops.aten._to_sparse_semi_structured(dense_ref)
            meta = compressed.indices()
            torch.testing.assert_close(meta, meta_ref, rtol=0, atol=0)

            dense = compressed.to_dense()
            torch.testing.assert_close(dense, dense_ref, rtol=0, atol=0)

        shapes = [[32, 128], [32, 256], [64, 128], [64, 256]]
        for r, c in shapes:
            run_test(r, c, device, dtype)


instantiate_device_type_tests(TestSparseSemiStructured, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
