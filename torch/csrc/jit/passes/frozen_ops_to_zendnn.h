/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Converts operators & their parameters to mkldnn if it is profitable
// Currently encompassing Conv2d and Conv3d, and Linear
// Op must be in float32 and mkldnn must be built
// This pass only works on frozen graph
TORCH_API void ConvertFrozenOpsToZENDNN(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
