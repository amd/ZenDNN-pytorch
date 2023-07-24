/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_ZENDNN_ENABLED()

namespace at { namespace native {

Tensor& _baddbmm_zendnn_(Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  AT_ERROR("bmm: ATen not compiled with ZENDNN support");
}
}}

#else // AT_ZENDNN_ENABLED

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/Utils.h>
#include <ATen/NativeFunctions.h>

#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>

#include <zendnn.h>

extern "C" {
void zenBatchMatMul(
        bool Layout,
        bool TransA,
        bool TransB,
        int *M_Array,
        int *N_Array,
        int *K_Array,
        const float *alpha_Array,
        const float **A_Array,
        int *lda_Array,
        const float **B_Array,
        int *ldb_Array,
        const float *beta_Array,
        float **C_Array,
        int *ldc_Array,
        int group_count,
        int *group_size
);

//Matmul kernel
void zenMatMul_gemm_wrapper(
        const bool Layout,
        const bool transpose_input,
        const bool transpose_filter,
        const int m,
        const int k,
        const int n,
        const float alpha,
        const float *input,
        const int lda,
        const float *filter,
        const int ldb,
        const float *bias,
        const bool relu,
        const int gelu,
        const float beta,
        float *output,
        const int ldc
);

}

namespace at { namespace native {

static inline void gemm_batch(
  int batch_size, bool trans_A, bool trans_B, int M, int N, int K,
  const float alpha, const float** A, int lda, const float** B,
  int ldb, const float beta, float** C, int ldc){
  bool Layout=true; //CblasRowMajor
  zenBatchMatMul(Layout, trans_A, trans_B, &M, &N, &K, &alpha,
          A, &lda, B, &ldb, &beta, C, &ldc, 1, &batch_size);
}

static inline void gemm(bool trans_A, bool trans_B,
  const int  M, const int N, const int K, const float alpha, const float* A,
  const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc) {

  const bool Layout=true;
  zenMatMul_gemm_wrapper(Layout, trans_A, trans_B, M, K, N, alpha,
                         (const float*)A, lda, (const float*)B, ldb, NULL,
                         0, 0, beta, (float *)C, ldc);
}

template <typename scalar_t>
static inline void baddbmm_zendnn_template(const Tensor& res, const Tensor& mat1, const Tensor& mat2, Scalar beta_, Scalar alpha_) {
  const auto mat1_strides = mat1.strides();
  const auto mat2_strides = mat2.strides();
  const auto mat1_sizes = mat1.sizes();
  const auto mat2_sizes = mat2.sizes();

  auto is_transposed = [](const c10::IntArrayRef& strides, const c10::IntArrayRef& sizes) {
    return strides[1] == 1 && strides[2] >= sizes[1];
  };

  bool trans_A = is_transposed(mat1_strides, mat1_sizes) ? 1 : 0;
  bool trans_B = is_transposed(mat2_strides, mat2_sizes) ? 1 : 0;

  // mat1: batch_size * M * K
  const int batch_size = mat1_sizes[0];
  const int M = mat1_sizes[1];
  // mat2: batch_size * K * N
  const int N = mat2_sizes[2];
  const int K = mat1_sizes[2];

  scalar_t alpha = alpha_.to<scalar_t>();
  scalar_t beta = beta_.to<scalar_t>();

  const int lda = trans_A == 1 ? mat1_strides[2] : mat1_strides[1];
  const int ldb = trans_B == 1 ? mat2_strides[2] : mat2_strides[1];
  const int ldc = res.strides()[1];

  // avoid using tensor accessor in the case of mat1/mat2 not being transposed
  // or only transposed in the last two axes
  const bool canAvoidTensorAccessor = mat1_strides[0] == mat1_sizes[1] * mat1_sizes[2] &&
    mat2_strides[0] == mat2_sizes[1] * mat2_sizes[2];

  scalar_t* const res_data = res.data_ptr<scalar_t>();

  if (batch_size == 1) {
    const scalar_t* A;
    const scalar_t* B;
    if (canAvoidTensorAccessor) {
      scalar_t* mat1_data = mat1.data_ptr<scalar_t>();
      scalar_t* mat2_data = mat2.data_ptr<scalar_t>();
      A = mat1_data;
      B = mat2_data;
    } else {
      auto mat1_acc = mat1.accessor<scalar_t, 3>();
      auto mat2_acc = mat2.accessor<scalar_t, 3>();
      A = mat1_acc[0].data();
      B = mat2_acc[0].data();
    }
    gemm(trans_A, trans_B, M, N, K, alpha, (const float *)A, lda,
	(const float *) B, ldb, beta, (float *) res_data, ldc);
    return;
  }

  std::vector<const scalar_t*> A;
  A.reserve(batch_size);
  std::vector<const scalar_t*> B;
  B.reserve(batch_size);
  std::vector<scalar_t*> C;
  C.reserve(batch_size);

  // avoid using tensor accessor in the case of mat1/mat2 not being transposed
  // or only transposed in the last two axis
  const auto res_sizes = res.sizes();
  if (canAvoidTensorAccessor) {
    scalar_t* mat1_data = mat1.data_ptr<scalar_t>();
    scalar_t* mat2_data = mat2.data_ptr<scalar_t>();
    for (int64_t batch = 0; batch < batch_size; batch++) {
      A.emplace_back(mat1_data + batch * mat1_sizes[1] * mat1_sizes[2]);
      B.emplace_back(mat2_data + batch * mat2_sizes[1] * mat2_sizes[2]);
      C.emplace_back(res_data + batch * res_sizes[1] * res_sizes[2]);
    }
  } else {
    auto mat1_acc = mat1.accessor<scalar_t, 3>();
    auto mat2_acc = mat2.accessor<scalar_t, 3>();
    for (int64_t batch = 0; batch < batch_size; batch++) {
      A.emplace_back(mat1_acc[batch].data());
      B.emplace_back(mat2_acc[batch].data());
      C.emplace_back(res_data + batch * res_sizes[1] * res_sizes[2]);
    }
  }

  gemm_batch(batch_size, trans_A, trans_B, M, N, K, alpha,
	(const float **) A.data(), lda, (const float **) B.data(), ldb, beta,
	(float **) C.data(), ldc);
}

Tensor& _baddbmm_zendnn_(Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  // checks are done in native/LinearAlgebra.cpp
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "baddbmm__zendnn", [&] {
      baddbmm_zendnn_template<scalar_t>(self, batch1, batch2, beta, alpha);
    });
  return self;
}

}} // namespace at::native

#endif
