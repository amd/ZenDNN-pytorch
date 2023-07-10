/******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
* 3. Neither the name of the copyright holder nor the names of its contributors
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*
******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/zendnn/Matmul.h>
#if !AT_ZENDNN_ENABLED()

namespace at {
namespace native {

void zendnn_matmul(
    const Tensor &mat1,
    const Tensor &mat2,
    const Tensor &result,
    float beta,
    float alpha) {
  TORCH_CHECK(false, "zendnn_matmul: ATen not compiled with zendnn support");
}

bool use_zendnn_bf16_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result_opt){
  return false;
}

bool zendnn_bf16_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const c10::BFloat16 *a, int64_t lda,
    const c10::BFloat16 *b, int64_t ldb,
    float beta,
    c10::BFloat16 *c, int64_t ldc) {
  return false;
}

} // namespace native
} // namespace at

#else // AT_ZENDNN_ENABLED

#include <ATen/native/zendnn/ZENDNNCommon.h>
#include <ATen/native/zendnn/Utils.h>

namespace at {
namespace native {

static bool use_zendnn_bf16_matmul() {
  return (
      at::globalContext().userEnabledZendnn() &&
      zendnn_bf16_device_check());
}

bool zendnn_bf16_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const c10::BFloat16 *a_data, int64_t lda,
    const c10::BFloat16 *b_data, int64_t ldb,
    float beta,
    c10::BFloat16 *c_data, int64_t ldc) {
  if (!use_zendnn_bf16_matmul() ||
      (m * n * k <= 16 * 16 * 16) ||
      (alpha == 0.0f)) {
    return false;
  }

  adeep::attr_t op_attr;
  // Use zendnn post ops to perform the add.
  if (beta != 0.0f) {
    op_attr = adeep::attr_t::fuse_sum();
  }
  // NOTE: View as c-contiguous to avoid extra reordering in zendnn
  // Use identity: C = AB <=> C^T = B^T A^T
  adeep::tensor::dims a_strides{{lda, 1}}, b_strides{{ldb, 1}}, c_strides{{ldc, 1}};
  if (transa != TransposeType::NoTranspose) {
    std::swap(a_strides[0], a_strides[1]);
  }
  if (transb != TransposeType::NoTranspose) {
    std::swap(b_strides[0], b_strides[1]);
  }

  adeep::tensor a({
      /*sizes=*/{k, m},
      adeep::tensor::data_type::bf16,
      /*strides=*/a_strides},
    const_cast<c10::BFloat16*>(a_data));
  adeep::tensor b({
      /*sizes=*/{n, k},
      adeep::tensor::data_type::bf16,
      /*strides=*/b_strides},
    const_cast<c10::BFloat16*>(b_data));
  adeep::tensor c({
      /*sizes=*/{n, m},
      adeep::tensor::data_type::bf16,
      /*strides=*/c_strides},
    c_data);

  adeep::matmul_forward::compute(
      b, a, c, alpha, beta,
      adeep::scale_t(), adeep::scale_t(), adeep::scale_t(), op_attr);

  if (c.get_data_handle() != c_data){
    // adeep will query onednn expect format of output
    // if given output format is not expected, adeep will re-init an output buffer
    // under this case, we need copy the re-inited buffer back to given buffer
    adeep::tensor real_output({
        /*sizes=*/{n, m},
        adeep::tensor::data_type::bf16,
        /*strides=*/c_strides},
      c_data);
    c.reorder_to(real_output);
  }

  return true;
}

void zendnn_matmul(
    const Tensor &mat1,
    const Tensor &mat2,
    const Tensor &result,
    float beta,
    float alpha) {

  TORCH_CHECK((mat1.dim() == 2 && mat2.dim() == 2) || // aten::addmm
              (mat1.dim() == 3 && mat2.dim() == 3) || // aten::bmm, aten::baddbmm
              (mat1.dim() == 2 && mat2.dim() == 1) || // aten::mv
              (mat1.dim() == 1 && mat2.dim() == 1),  // aten::dot
              "zendnn_matmul:  unsupported dims for mat and mat2");

  TORCH_CHECK(zendnn_bf16_device_check(),
    "zendnn_matmul: zendnn_matmul bf16 path needs the cpu support avx512bw, avx512vl and avx512dq, or AWS Graviton3");

#if defined(__aarch64__)
  if (zendnn_bf16_device_check_arm()) {
     //onednn fastmath mode can leverage bf16 HW even for the fp32 input, e.g. Arm Neoverse V1
     //so, don't restrict the zendnn_matmul only for bf16 inputs, allow it for float as well
     TORCH_CHECK((mat1.scalar_type() == mat2.scalar_type()) && (mat1.scalar_type() == result.scalar_type()) &&
                 ((mat1.scalar_type() == at::kFloat) || (mat1.scalar_type() == at::kBFloat16)),
                 "zendnn_matmul:  only enabled for fp32 and bf16 path");
  } else
#endif
  {
     TORCH_CHECK(mat1.scalar_type() == at::kBFloat16 &&
                 mat2.scalar_type() == at::kBFloat16 &&
                 result.scalar_type() == at::kBFloat16, "zendnn_matmul:  only enabled for bf16 path");
  }

  auto mat1_unsqueezed = mat1.dim() == 1 ? mat1.unsqueeze(0) : mat1;
  auto mat2_unsqueezed = mat2.dim() == 1 ? mat2.unsqueeze(1) : mat2;
  auto result_unsqueezed = result.dim() == 1 ? result.unsqueeze(1) : result;

  adeep::attr_t op_attr;
  // "addmm", "addbmm" "baddbmm" in pytorch allow bias to be 2-D or 3-D tensor
  // but zendnn matmul primitive only support bias be 1-D tensors
  // to address their differences, we use zendnn post ops to perform a fused "add" after matrix multiplication is over
  if (beta != 0.0f) op_attr = adeep::attr_t::fuse_sum();
  // If alpha = 0, dose not need actually do gemm computation
  if (alpha == 0)
    return;

  auto is_zendnn_optimized_format = [&](const Tensor& t) {
    if (t.is_contiguous()) return true;
    const auto sizes = t.sizes();
    const auto strides = t.strides();
    if (t.dim() == 2){
      return strides[0] == 1 && strides[1] == sizes[0];
    } else {
      // dim = 3
      return strides[0] == sizes[1] * sizes[2] && strides[1] == 1 && strides[2] == sizes[1];
    }
  };

  // zendnn only optimized for contiguous or transposed (transpose last 2 dim if 3-D tensor) format now
  // Will remove this "contiguous" after zendnn have fully supported
  Tensor mat1_ = is_zendnn_optimized_format(mat1_unsqueezed) ? mat1_unsqueezed : mat1_unsqueezed.contiguous();
  Tensor mat2_ = is_zendnn_optimized_format(mat2_unsqueezed) ? mat2_unsqueezed : mat2_unsqueezed.contiguous();

  // zendnn_matmul only proceed CPU tensor
  const adeep::tensor x = itensor_view_from_dense(mat1_);
  const adeep::tensor w = itensor_view_from_dense(mat2_);
  adeep::tensor y = itensor_view_from_dense(result_unsqueezed);
  adeep::matmul_forward::compute(x, w, y, alpha, beta,
      adeep::scale_t(), adeep::scale_t(), adeep::scale_t(), op_attr);
  if (y.get_data_handle() != result.data_ptr()){
    // adeep will query onednn expect format of output
    // if given output format is not expected, adeep will re-init an output buffer
    // under this case, we need copy the re-inited buffer back to given buffer
    adeep::tensor public_y = itensor_view_from_dense(result);
    y.reorder_to(public_y);
  }

  if (mat1.dim() == 1 && mat2.dim() == 1){
    // aten::dot
    result.squeeze_();
  }

}

inline bool checksize(const Tensor& mat1, const Tensor& mat2){
  // if dim = 2, mat1's size = (m * n), mat2's size = (n * k)
  // else if dim = 3, mat1's size = (b * m * n), mat2's size = (b * n * k)
  // else called from aten::mv, mat1.size = (m * n), mat2.size = (n)
  // only m * n * b * k(if exist) are large enough we can get benefit from zendnn optimized gemm kernel
  static const int64_t zendnn_gemm_min_size = 16 * 16 * 16;
  if (mat1.dim() == 1 && mat2.dim() == 1) {
    // aten::dot
    return mat1.size(0) > zendnn_gemm_min_size;
  } else if (mat1.dim() == 2 && mat2.dim() == 1) {
    // aten::mv
    return mat1.size(0) * mat1.size(1) > zendnn_gemm_min_size;
  } else if (mat2.dim() == 2 && mat2.dim() == 2) {
    // aten::addmm
    return mat1.size(0) * mat1.size(1) * mat2.size(1) > zendnn_gemm_min_size;
  } else {
    // aten::bmm, aten::baddbmm
    return mat1.size(0) * mat1.size(1) * mat1.size(2) * mat2.size(2) > zendnn_gemm_min_size;
  }
}

bool use_zendnn_bf16_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result) {
#if defined(__aarch64__)
  if (zendnn_bf16_device_check_arm()) {
     //onednn fastmath mode can leverage bf16 HW even for the fp32 input, e.g. Arm Neoverse V1
     //so, don't restrict the zendnn_matmul only for bf16 inputs, allow it for float as well
     return (
        use_zendnn_bf16_matmul() &&
        (mat1.scalar_type() == mat2.scalar_type()) && (!result.defined() || (mat1.scalar_type() == result.scalar_type())) &&
        ((mat1.scalar_type() == kFloat) || (mat1.scalar_type() == kBFloat16)) &&
        mat1.numel() != 0 &&
        mat2.numel() != 0 &&
        checksize(mat1, mat2));
  } else
#endif
  {
     return (
        use_zendnn_bf16_matmul() &&
        mat1.scalar_type() == kBFloat16 &&
        mat2.scalar_type() == kBFloat16 &&
        (!result.defined() || result.scalar_type() == kBFloat16) &&
        mat1.numel() != 0 &&
        mat2.numel() != 0 &&
        checksize(mat1, mat2));
  }
}

} // namespace native
} // namespace at

#endif // AT_ZENDNN_ENABLED
