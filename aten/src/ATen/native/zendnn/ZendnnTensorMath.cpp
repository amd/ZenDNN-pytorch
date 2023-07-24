/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>

#if !AT_ZENDNN_ENABLED()

namespace at {
namespace native {

Tensor& zendnn_zero_(Tensor& self) {
  TORCH_CHECK("zendnn_zero_: ATen not compiled with ZENDNN support");
  return self;
}

} // namespace native
} // namespace at

#else // AT_ZENDNN_ENABLED

#include <ATen/native/zendnn/ZENDNNCommon.h>

namespace at {
namespace native {

Tensor& zendnn_zero_(Tensor& self) {
  using Vec = vec::Vectorized<float>;

  adeep::tensor& x = itensor_from_zendnn(self);

  auto n = x.get_nelems();
  auto* x_ = static_cast<float*>(x.get_data_handle());
  parallel_for(0, n, 2048, [x_](int64_t begin, int64_t end) {
    vec::map(
        [](Vec /* unused */) { return 0.0; },
        x_ + begin,
        x_ + begin,
        end - begin);
  });

  return self;
}

} // namespace native
} // namespace at

#endif // AT_ZENDNN_ENABLED
