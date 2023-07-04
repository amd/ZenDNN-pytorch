#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

namespace at::native {

namespace {

inline void _store(
    float* dst,
    vec::Vectorized<float> src) {
  src.store(dst);
}

inline void _store(
    at::BFloat16* dst,
    vec::Vectorized<float> src) {
  auto res = vec::convert_float_bfloat16(src, src);
  res.store(dst, vec::Vectorized<float>::size());
}

inline void _mkl_gemm(
    const bool& need_trans_a,
    const bool& need_trans_b,
    const int& m,
    const int& n,
    const int& k,
    const float& alpha,
    float* a,
    const int& lda,
    float* b,
    const int& ldb,
    const float& beta,
    float* c,
    const int& ldc) {
  cpublas::gemm(
      need_trans_a ? TransposeType::Transpose : TransposeType::NoTranspose,
      need_trans_b ? TransposeType::Transpose : TransposeType::NoTranspose,
      m,
      n,
      k,
      alpha,
      a,
      lda,
      b,
      ldb,
      beta,
      c,
      ldc);
}

inline void _mkl_gemm(
    const bool& need_trans_a,
    const bool& need_trans_b,
    const int& m,
    const int& n,
    const int& k,
    const float& alpha,
    at::BFloat16* a,
    const int& lda,
    at::BFloat16* b,
    const int& ldb,
    const float& beta,
    float* c,
    const int& ldc) {
  cblas_gemm_bf16bf16f32(
      CblasColMajor,
      need_trans_a ? CblasTrans : CblasNoTrans,
      need_trans_b ? CblasTrans : CblasNoTrans,
      m,
      n,
      k,
      alpha,
      (const MKL_BF16*)(a),
      lda,
      (const MKL_BF16*)(b),
      ldb,
      beta,
      c,
      ldc);
}

template <typename scalar_t>
inline void fill_stub(scalar_t* data, scalar_t val, int64_t size) {
  using Vec = Vectorized<scalar_t>;
  Vec data_vec = Vec(val);
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    data_vec.store(data + d);
  }
  #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  for (; d < size; d++) {
    data[d] = val;
  }
}

inline void _exp_reduce_sum_fusion_kernel(
    float* a,
    const int& size,
    float* out,
    float& val) {
  using fVec = vec::Vectorized<float>;
  vec::map<float>(
    [val](fVec x) { return (x - fVec(val)).exp(); }, out, a, size);
  val = vec::reduce_all<float>(
    [](fVec& x, fVec& y) { return x + y; }, out, size);
}

template <typename scalar_t>
inline void _normalization_kernel(
    const float* a,
    const float& sum,
    const int& size,
    scalar_t* out) {
  using fVec = vec::Vectorized<float>;
  auto vec_sum = fVec(sum);
  int64_t i = 0;
  for (i = 0; i < fVec::size() * (size / fVec::size()); i += fVec::size()) {
    auto tmp0 = fVec::loadu(a + i);
    auto tmp1 = tmp0 / vec_sum;
    _store(out + i, tmp1);
  }
  #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  for (; i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 / sum;
    out[i] = tmp1;
  }
}

inline void _reduce_max_fusion_kernel(
    const float* a,
    const int& size,
    float* out,
    float& max) {
  using fVec = vec::Vectorized<float>;
  max = vec::reduce_all<float>(
    [](fVec& x, fVec& y) { return vec::maximum(x, y); }, out, size);
}

/**
 * This kernel is used to reorder the MHA output
 * with strides.
 * src: MKL GEMM output buffer
 * dst: Final MHA output
 */
template <typename scalar_t>
inline void _reorder_mha_output_kernel(
    float* src,
    scalar_t* dst,
    const int& rows,
    const int& cols,
    const int& dst_stride) {
  using fVec = vec::Vectorized<float>;
  for (int64_t i = 0; i < rows; ++i) {
    int64_t j = 0;
    for (j = 0; j < fVec::size() * (cols / fVec::size()); j += fVec::size()) {
      auto tmp0 = fVec::loadu(src + i * cols + j);
      _store(dst + i * dst_stride + j, tmp0);
    }
    #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (; j < cols; j++) {
      dst[i * dst_stride + j] = src[i * cols + j];
    }
  }
}

/**
 * This kernel is used to update the MHA output with the latest MAX
 * and SUM values block by block.
 * exp_val: exp(max_old - max_new)
 * In the i th block, the softmax(qk - i th) * v - i th was calculated
 * with the old MAX and SUM values, max_old and sum_old. When moving to
 * the i + 1 th block, since softmax(qk - i + 1 th) will be calculated
 * with the new MAX and SUM values, max_new and sum_new, thus the MHA
 * buffer which stores the summation of blocked softmax(qk) * v should
 * be also updated using max_new and sum_new:
 * a = a * sum_old / sum_new
 * a = a * exp(max_old) / exp(max_new) = a * exp_val
 */
inline void _mha_update_sum_max_kernel(
    const float* a,
    const float& sum_old,
    const float& sum_new,
    const float& exp_val,
    const int& size,
    float* out) {
  using fVec = vec::Vectorized<float>;
  float sum_cor = sum_old / sum_new;
  vec::map<float>(
    [sum_cor, exp_val](fVec x)
      { return x * fVec(sum_cor) * fVec(exp_val); },
      out, a, size);
}

template <typename scalar_t>
void _mha_softmax_kernel(
    float* a,
    scalar_t* b,
    float* dst,
    float* max,
    float* sum,
    const int& qsize,
    const int& kvsize,
    const int& headsize,
    const int& idx) {
  using accum_t = at::opmath_type<float>;
  accum_t tmp_max = 0.f, tmp_sum = 0.f, sum_old = 0.f, exp_tmp = 0.f;

  for (int i = 0; i < qsize; ++i) {
    sum_old = sum[i];

    _reduce_max_fusion_kernel(
        a + i * kvsize, kvsize, a + i * kvsize, tmp_max);

    tmp_max = max[i] > tmp_max ? max[i] : tmp_max;

    tmp_sum = tmp_max;
    _exp_reduce_sum_fusion_kernel(
        a + i * kvsize, kvsize, a + i * kvsize, tmp_sum);
    exp_tmp = std::exp(max[i] - tmp_max);
    sum[i] = tmp_sum + exp_tmp * sum[i];
    max[i] = tmp_max;

    _normalization_kernel<scalar_t>(
        a + i * kvsize, sum[i], kvsize, b + i * kvsize);

    if (idx) {
      _mha_update_sum_max_kernel(
        dst + i * headsize,
        sum_old,
        sum[i],
        exp_tmp,
        headsize,
        dst + i * headsize);
    }
  }
}

// Return the first output with other outputs arbitrary because they are used
// for backward.
// TODO: Calculate other outputs when adding cpu backward.
template <typename scalar_t>
void cpu_flash_attention(
    const Tensor& output,
    const Tensor& logsumexp,
    const Tensor& cum_seq_q,
    const Tensor& cum_seq_k,
    int64_t& max_q,
    int64_t& max_k,
    const Tensor& philox_seed,
    const Tensor& philox_offset,
    const Tensor& debug_attn_mask,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    c10::optional<double> scale,
    const std::vector<int64_t>& qsplit_range,
    const std::vector<int64_t>& qsplit_size,
    const std::vector<int64_t>& kvsplit_range,
    const std::vector<int64_t>& kvsplit_size) {

  // Query (Batch x Num_heads x Q_seq_len  x Dim_per_head)
  // Key   (Batch x Num_heads x KV_seq_len x Dim_per_head)
  // Value (Batch x Num_heads x KV_seq_len x Dim_per_head)
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(2);
  int64_t kvSize = value.size(2);
  int64_t num_head = query.size(1);
  int64_t headSize = query.size(3);
  int64_t hiddenSize = num_head * headSize;

  float scaling_factor =
      sdp::calculate_scale(query, scale).as_float_unchecked();

  // Query (Batch x Q_seq_len  x Num_heads x Dim_per_head)
  // Key   (Batch x KV_seq_len x Num_heads x Dim_per_head)
  // Value (Batch x KV_seq_len x Num_heads x Dim_per_head)
  Tensor q = query.transpose(1, 2);
  Tensor k = key.transpose(1, 2);
  Tensor v = value.transpose(1, 2);

  int64_t qStrideB = q.stride(0);
  int64_t qStrideM = q.stride(1);
  int64_t qStrideH = q.stride(2);
  int64_t kStrideB = k.stride(0);
  int64_t kStrideN = k.stride(1);
  int64_t kStrideH = k.stride(2);
  int64_t vStrideB = v.stride(0);
  int64_t vStrideN = v.stride(1);
  int64_t vStrideH = v.stride(2);
  int64_t oStrideB = output.stride(0);
  int64_t oStrideM = output.stride(1);
  int64_t oStrideH = output.stride(2);

  int64_t qSplitSize = qSize;
  for (size_t i = 0; i < qsplit_range.size(); ++i) {
    if (qSize > qsplit_range[i]) {
      qSplitSize = qsplit_size[i];
      break;
    }
  }
  int64_t kvSplitSize = kvSize;
  for (size_t i = 0; i < kvsplit_range.size(); ++i) {
    if (kvSize > kvsplit_range[i]) {
      kvSplitSize = kvsplit_size[i];
      break;
    }
  }

  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  int64_t qTail = (qSize - 1) % qSplitSize + 1;

  int64_t num_thread = at::get_num_threads();

  at::Tensor qk = at::empty({num_thread, qSplitSize, kvSplitSize}, at::kFloat);
  at::Tensor qk_norm = at::empty({num_thread, qSplitSize, kvSplitSize}, q.options());
  at::Tensor qk_max = at::empty({num_thread, qSplitSize}, at::kFloat);
  at::Tensor qk_sum = at::empty({num_thread, qSplitSize}, at::kFloat);
  at::Tensor dst = at::empty({num_thread, qSplitSize, headSize}, at::kFloat);

  scalar_t* q_data = q.data_ptr<scalar_t>();
  scalar_t* k_data = k.data_ptr<scalar_t>();
  scalar_t* v_data = v.data_ptr<scalar_t>();
  scalar_t* out_data = output.data_ptr<scalar_t>();
  float* qk_data = qk.data_ptr<float>();
  scalar_t* qk_norm_data = qk_norm.data_ptr<scalar_t>();
  float* qk_max_data = qk_max.data_ptr<float>();
  float* qk_sum_data = qk_sum.data_ptr<float>();
  float* dst_data = dst.data_ptr<float>();

  at::parallel_for(0, batchSize * num_head * qSlice, 0, [&](int64_t begin, int64_t end) {
    int64_t i = 0, j = 0, k = 0;
    data_index_init(begin, i, batchSize, j, num_head, k, qSlice);
    int ompIdx = at::get_thread_num();
    for (const auto x : c10::irange(begin, end)) {
      (void)x; // Suppress unused variable
      int64_t m = k * qSplitSize;
      int64_t qBlockSize = (k == qSlice - 1) ? qTail : qSplitSize;
      // Initialize max and sum
      fill_stub(qk_max_data + ompIdx * qSplitSize, -std::numeric_limits<float>::infinity(), qBlockSize);
      fill_stub(qk_sum_data + ompIdx * qSplitSize, 0.f, qBlockSize);
      int64_t num_keys = is_causal ? std::min(m + qSplitSize, kvSize) : kvSize;
      for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
        int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
        // Calculate Q @ K.T
        _mkl_gemm(
            true,
            false,
            kvBlockSize,
            qBlockSize,
            headSize,
            scaling_factor,
            k_data + i * kStrideB + j * kStrideH +
                n * kStrideN,
            kStrideN,
            q_data + i * qStrideB + j * qStrideH +
                m * qStrideM,
            qStrideM,
            0.f,
            qk_data + ompIdx * qSplitSize * kvSplitSize,
            kvBlockSize);
        // Apply casual mask
        if (is_causal && num_keys - n <= kvSplitSize) {
          for (const auto row : c10::irange(qBlockSize)) {
            int64_t last_col = m + row - n;
            float* row_ptr = qk_data + ompIdx * qSplitSize * kvSplitSize + row * kvSplitSize;
            fill_stub(row_ptr + last_col + 1, -std::numeric_limits<float>::infinity(), kvBlockSize - last_col - 1);
          }
        }
        // Update coefficients with Softmax
        _mha_softmax_kernel<scalar_t>(
            qk_data + ompIdx * qSplitSize * kvSplitSize,
            qk_norm_data + ompIdx * qSplitSize * kvSplitSize,
            dst_data + ompIdx * qSplitSize * headSize,
            qk_max_data + ompIdx * qSplitSize,
            qk_sum_data + ompIdx * qSplitSize,
            qBlockSize,
            kvBlockSize,
            headSize,
            n);
        // Calculate Softmax(Q @ K.T) @ V
        _mkl_gemm(
            false,
            false,
            headSize,
            qBlockSize,
            kvBlockSize,
            1.f,
            v_data + i * vStrideB + j * vStrideH +
                n * vStrideN,
            vStrideN,
            qk_norm_data + ompIdx * qSplitSize * kvSplitSize,
            kvBlockSize,
            n == 0 ? 0.f : 1.f,
            dst_data + ompIdx * qSplitSize * headSize,
            headSize);
      }
      _reorder_mha_output_kernel<scalar_t>(
          dst_data + ompIdx * qSplitSize * headSize,
          out_data + i * oStrideB + j * oStrideH +
              m * oStrideM,
          qBlockSize,
          headSize,
          oStrideM);
      data_index_step(i, batchSize, j, num_head, k, qSlice);
    }
  });

}

void flash_attention_kernel_impl(
    const Tensor& output,
    const Tensor& logsumexp,
    const Tensor& cum_seq_q,
    const Tensor& cum_seq_k,
    int64_t& max_q,
    int64_t& max_k,
    const Tensor& philox_seed,
    const Tensor& philox_offset,
    const Tensor& debug_attn_mask,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    c10::optional<double> scale) {
  const std::vector<int64_t> qsplit_range{767, 191, 31};
  const std::vector<int64_t> qsplit_size{256, 64, 32};
  const std::vector<int64_t> kvsplit_range{512};
  const std::vector<int64_t> kvsplit_size{512};
  AT_DISPATCH_SWITCH(query.scalar_type(), "flash_attention",
    AT_DISPATCH_CASE(ScalarType::Float, [&] {
      cpu_flash_attention<scalar_t>(
          output, logsumexp, cum_seq_q, cum_seq_k,
          max_q, max_k, philox_seed, philox_offset, debug_attn_mask,
          query, key, value, dropout_p, is_causal, return_debug_mask, scale,
          qsplit_range, qsplit_size, kvsplit_range, kvsplit_size);
    });
    AT_DISPATCH_CASE(ScalarType::BFloat16, [&] {
      cpu_flash_attention<scalar_t>(
          output, logsumexp, cum_seq_q, cum_seq_k,
          max_q, max_k, philox_seed, philox_offset, debug_attn_mask,
          query, key, value, dropout_p, is_causal, return_debug_mask, scale,
          qsplit_range, qsplit_size, kvsplit_range, kvsplit_size);
    });
  );
}

} // anonymous namespace

REGISTER_DISPATCH(flash_attention_kernel, &flash_attention_kernel_impl);

} // at::native
