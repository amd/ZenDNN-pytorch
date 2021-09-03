#pragma once

#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <c10/util/variant.h>

namespace py = pybind11;


// TODO: For some reason this doesn't work in `torch/csrc/utils/pybind.h`.
namespace pybind11 { namespace detail {
    
template <typename... Ts>
struct type_caster<c10::variant<Ts...>> : variant_caster<c10::variant<Ts...>> {};

}} // namespace pybind11::detail


namespace torch {
namespace profiler {
namespace impl {

void register_trace(py::module& py_module);

}}}  // torch::profiler::impl
