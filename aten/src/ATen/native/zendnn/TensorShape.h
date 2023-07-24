/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#pragma once

#include <ATen/ATen.h>

namespace at {
namespace native {

Tensor zendnn_view(const Tensor& self, IntArrayRef size);

Tensor zendnn_clone(const Tensor& self);

} // namespace native
} // namespace at
