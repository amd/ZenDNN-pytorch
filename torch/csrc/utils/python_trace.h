#pragma once

#include <string>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;


namespace torch {
namespace profiler {
namespace impl {

void register_trace(py::module& py_module);

}}}  // torch::profiler::impl
