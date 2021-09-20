/* C++ template for Timer methods.

This template will replace:
    `GLOBAL_SETUP_TEMPLATE_LOCATION`,
    `SETUP_TEMPLATE_LOCATION`
    `STMT_TEMPLATE_LOCATION`
      and
    `TEARDOWN_TEMPLATE_LOCATION`
sections with user provided statements.
*/

#include <chrono>

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <c10/util/irange.h>
#include "callgrind.h"

// Global setup. (e.g. #includes)
// GLOBAL_SETUP_TEMPLATE_LOCATION

void call(int n_iter) {
    pybind11::gil_scoped_release no_gil;
    // SETUP_TEMPLATE_LOCATION
    for(const auto _ : c10::irange(n_iter)) {  // NOLINT(clang-diagnostic-unused-variable)
        // STMT_TEMPLATE_LOCATION
    }
    // TEARDOWN_TEMPLATE_LOCATION
}

float measure_wall_time(int n_iter, int n_warmup_iter, bool cuda_sync) {
    pybind11::gil_scoped_release no_gil;
    // SETUP_TEMPLATE_LOCATION

    for(const auto _ : c10::irange(n_warmup_iter)) {  // NOLINT(clang-diagnostic-unused-variable)
        // STMT_TEMPLATE_LOCATION
    }

    if (cuda_sync) {
        torch::cuda::synchronize();
    }
    auto start_time = std::chrono::high_resolution_clock::now();

    for(const auto _ : c10::irange(n_iter)) {  // NOLINT(clang-diagnostic-unused-variable)
        // STMT_TEMPLATE_LOCATION
    }

    if (cuda_sync) {
        torch::cuda::synchronize();
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto result = std::chrono::duration<double>(end_time - start_time).count();

    // TEARDOWN_TEMPLATE_LOCATION

    return result;
}

void collect_callgrind(int n_iter, int n_warmup_iter) {
    // NB:
    //  We need `pybind11::gil_scoped_release` not because we want to measure
    //  in parallel, but rather because Autograd requires that the GIL is not
    //  held. (Since in any other context holding the GIL during backward is a
    //  pessimization.)
    pybind11::gil_scoped_release no_gil;

    // SETUP_TEMPLATE_LOCATION

    for(const auto _ : c10::irange(n_warmup_iter)) {  // NOLINT(clang-diagnostic-unused-variable)
        // STMT_TEMPLATE_LOCATION
    }

    // torch._C._valgrind_toggle()
    CALLGRIND_TOGGLE_COLLECT;  // NOLINT

    for(const auto _ : c10::irange(n_iter)) {  // NOLINT(clang-diagnostic-unused-variable)
        // STMT_TEMPLATE_LOCATION
    }

    // torch._C._valgrind_toggle_and_dump_stats()
    CALLGRIND_TOGGLE_COLLECT;  // NOLINT
    CALLGRIND_DUMP_STATS;  // NOLINT

    // TEARDOWN_TEMPLATE_LOCATION
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "call",
        &call,
        py::arg("n_iter"));
    m.def(
        "measure_wall_time",
        &measure_wall_time,
        py::arg("n_iter"),
        py::arg("n_warmup_iter"),
        py::arg("cuda_sync"));
    m.def(
        "collect_callgrind",
        &collect_callgrind,
        py::arg("n_iter"),
        py::arg("n_warmup_iter"));
}
