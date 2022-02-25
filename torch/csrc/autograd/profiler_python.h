#pragma once

#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace profiler {
namespace python_tracer {

void init(pybind11::module_ m);

} // namespace python_tracer
} // namespace profiler
} // namespace torch
