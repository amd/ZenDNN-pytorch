#pragma once

#include <ATen/Parallel.h>
#include <ATen/NumericUtils.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/ReductionType.h>
#include <c10/util/irange.h>
#include <ATen/OpMathType.h>

namespace at::native {
inline namespace CPU_CAPABILITY {

using namespace vec;

#define AT_DISPATCH_REDUCTION_TYPES(op, ...)                                   \
  [&] {                                                                        \
    switch (op) {                                                              \
      case ReductionType::SUM: {                                               \
        static constexpr auto reduce = ReductionType::SUM;                     \
        return __VA_ARGS__();                                                  \
      }                                                                        \
      case ReductionType::MEAN: {                                              \
        static constexpr auto reduce = ReductionType::MEAN;                    \
        return __VA_ARGS__();                                                  \
      }                                                                        \
      case ReductionType::MIN: {                                               \
        static constexpr auto reduce = ReductionType::MIN;                     \
        return __VA_ARGS__();                                                  \
      }                                                                        \
      case ReductionType::MAX: {                                               \
        static constexpr auto reduce = ReductionType::MAX;                     \
        return __VA_ARGS__();                                                  \
      }                                                                        \
      case ReductionType::PROD: {                                              \
        static constexpr auto reduce = ReductionType::PROD;                    \
        return __VA_ARGS__();                                                  \
      }                                                                        \
    }                                                                          \
  }()

template <typename scalar_t, ReductionType reduce>
inline vec_scalar_t<scalar_t> init_value() {
  using acc_t = vec_scalar_t<scalar_t>;
  acc_t val;
  if (reduce == ReductionType::SUM ||
      reduce == ReductionType::MEAN) {
    val = static_cast<acc_t>(0);
  } else if (reduce == ReductionType::PROD) {
    val = static_cast<acc_t>(1);
  } else if (reduce == ReductionType::MAX) {
    val = -std::numeric_limits<acc_t>::infinity();
  } else {
    TORCH_INTERNAL_ASSERT(reduce == ReductionType::MIN);
    val = std::numeric_limits<acc_t>::infinity();
  }
  return val;
}

template <typename scalar_t, ReductionType reduce>
inline vec_scalar_t<scalar_t> init_value(const c10::optional<Scalar>& initial) {
  using acc_t = vec_scalar_t<scalar_t>;
  if (initial.has_value()) {
    return initial.value().to<acc_t>();
  } else {
    return init_value<scalar_t, reduce>();
  }
}

template <typename scalar_t>
inline void init(scalar_t* out, int64_t size, const vec_scalar_t<scalar_t>& val) {
  using Vec = Vectorized<vec_scalar_t<scalar_t>>;
  map<scalar_t>(
      [val](Vec x) { return Vec(val); },
      out,
      out,
      size);
}

template <typename scalar_t, ReductionType reduce>
inline void init(scalar_t* out, int64_t size, const c10::optional<Scalar>& initial) {
  using acc_t = vec_scalar_t<scalar_t>;
  acc_t val = init_value<scalar_t, reduce>(initial);
  init(out, size, val);
}

// overload with `include_self`, used by scatter_reduce
template <typename scalar_t, ReductionType reduce>
inline void init(scalar_t* out, int64_t size, bool include_self = false) {
  using acc_t = vec_scalar_t<scalar_t>;
  if (!include_self) {
    acc_t val = init_value<scalar_t, reduce>();
    init(out, size, val);
  }
}

template <typename scalar_t, ReductionType reduce>
inline void _init(scalar_t* self_ptr, at::opmath_type<scalar_t>* buffer_ptr, int64_t size, bool include_self) {
  if (!include_self) {
    init<at::opmath_type<scalar_t>, reduce>(buffer_ptr, size, include_self);
  } else {
    vec::convert(self_ptr, buffer_ptr, size);
  }
}

template <typename scalar_t>
inline scalar_t _max(const scalar_t& x, const scalar_t& y) {
  return at::_isnan(y) ? y : std::max(x, y);
}

template <typename scalar_t>
inline Vectorized<scalar_t> _max(const Vectorized<scalar_t>& x, const Vectorized<scalar_t>& y) {
  // vec::maximum propagates NaN
  return vec::maximum(x, y);
}

template <typename scalar_t>
inline scalar_t _min(const scalar_t& x, const scalar_t& y) {
  return at::_isnan(y) ? y : std::min(x, y);
}

template <typename scalar_t>
inline Vectorized<scalar_t> _min(const Vectorized<scalar_t>& x, const Vectorized<scalar_t>& y) {
  // vec::minimum propagates NaN
  return vec::minimum(x, y);
}

// for Max and Min, propagate NaN:
template <typename T, ReductionType reduce>
inline T update(const T& x, const T& y) {
  if (reduce == ReductionType::SUM ||
      reduce == ReductionType::MEAN) {
    return x + y;
  } else if (reduce == ReductionType::PROD) {
    return x * y;
  } else if (reduce == ReductionType::MAX) {
    return _max(x, y);
  } else {
    TORCH_INTERNAL_ASSERT(reduce == ReductionType::MIN);
    return _min(x, y);
  }
}

template <typename scalar_t, ReductionType reduce>
inline void update(scalar_t* out, scalar_t* data, int64_t K) {
  using Vec = vec::Vectorized<vec_scalar_t<scalar_t>>;
  map2<scalar_t>(
      [](Vec x, Vec y) { return update<Vec, reduce>(x, y); },
      out,
      out,
      data,
      K);
}

template <typename scalar_t, ReductionType reduce,
          typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t>, int> = 0>
inline void _update(scalar_t* src_data, at::opmath_type<scalar_t>* buffer_ptr, int64_t col, int64_t K) {
  update<scalar_t, reduce>(buffer_ptr, src_data + col * K, K);
}

template <typename scalar_t, ReductionType reduce,
          typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline void _update(scalar_t* src_data, at::opmath_type<scalar_t>* buffer_ptr, int64_t col, int64_t K) {
  using opmath_t = at::opmath_type<scalar_t>;
  using Vec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<opmath_t>;
  int64_t k = 0;
  constexpr int64_t kVecSize = Vec::size();
  constexpr int64_t kfVecSize = fVec::size();
  for (k = 0; k < K - (K % kVecSize); k += kVecSize) {
    Vec src_vec = Vec::loadu(src_data + col * K + k);
    fVec src_vec0, src_vec1;
    std::tie(src_vec0, src_vec1) = convert_to_float<scalar_t>(src_vec);
    fVec buf_vec0, buf_vec1;
    buf_vec0 = update<fVec, reduce>(fVec::loadu(buffer_ptr + k), src_vec0);
    buf_vec1 = update<fVec, reduce>(fVec::loadu(buffer_ptr + k + kfVecSize), src_vec1);
    buf_vec0.store(buffer_ptr + k);
    buf_vec1.store(buffer_ptr + k + fVec::size());
  }
  for (; k < K; k++) {
    opmath_t src_val = opmath_t(src_data[col * K + k]);
    buffer_ptr[k] = update<opmath_t, reduce>(buffer_ptr[k], src_val);
  }
}

template <typename scalar_t, ReductionType reduce>
inline void write(scalar_t* out, int64_t count, int64_t K) {
  using Vec = vec::Vectorized<vec_scalar_t<scalar_t>>;
  if (reduce == ReductionType::MEAN) {
    if (count > 0) {
      vec::map<scalar_t>(
          [count](Vec x) { return x / Vec(count); },
          out,
          out,
          K);
    }
  }
}

} // namespace CPU_CAPABILITY
} // namespace at::native
