#include <ATen/Tensor.h>

namespace torch {
namespace inductor {

at::Tensor _alloc_from_pool(
    const at::Tensor& self,
    int64_t offset_bytes,
    at::ScalarType dtype,
    at::IntArrayRef size,
    at::IntArrayRef stride);

at::Tensor _reinterpret_tensor(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    int64_t offset_increment = 0);

inline int64_t _align(int64_t nbytes, int64_t ALIGN_BYTES) {
  // Round up to the nearest multiple of ALIGN_BYTES
  // ALIGN_BYTES must be a power of 2
  return (nbytes + ALIGN_BYTES - 1) & -ALIGN_BYTES;
}

} // namespace inductor
} // namespace torch
