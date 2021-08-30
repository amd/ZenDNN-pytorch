#include <torch/csrc/utils/python_trace.h>

#include <atomic>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <Python.h>
#include <frameobject.h>
#include <pybind11/stl.h>

#include <ATen/record_function.h>


namespace torch {
namespace profiler {
namespace impl {


// Simplified metadata to record inputs and outputs. This is a simplified
// version of IValue, and is here because we don't (yet) want to deal with
// the full complexity of IValue.
enum Tag {
    kTensor,
    kInt,
    kOther,
};


struct Element {
    Element(const c10::IValue& i, bool is_input) : is_input_(is_input) {
        if (i.isTensor()) {
            tag_ = Tag::kTensor;
            id_ = i.hash().toInt();
        } else if (i.isInt()) {
            tag_ = Tag::kInt;
            id_ = i.toInt();
        } else {
            tag_ = Tag::kOther;
            id_ = (other_id_counter_++);
        }
    }

    bool is_input_;
    Tag tag_;
    size_t id_;

    // Anything that is not a known type we simply keep an id.
    static std::atomic<size_t> other_id_counter_;
};

std::atomic<size_t> Element::other_id_counter_{0};


struct ObserverEvent {
    std::string name;
    std::vector<Element> elements;
};


struct PyEvent {
    int event;
    PyCodeObject* f_code; 
};


c10::optional<at::CallbackHandle> handle_;
std::vector<ObserverEvent> observer_events_;
std::vector<PyEvent> py_events_;


static std::unique_ptr<at::ObserverContext> onFunctionEnter(const at::RecordFunction& fn) {
    return nullptr;
}


static void onFunctionExit(const at::RecordFunction& fn, at::ObserverContext* ctx_ptr) {
    ObserverEvent e;
    e.name = fn.name().str();

    for (auto& i : fn.inputs()) {
        e.elements.emplace_back(i, /*is_input=*/true);
    }
    for (auto& i : fn.outputs()) {
        e.elements.emplace_back(i, /*is_input=*/false);
    }

    observer_events_.push_back(e);
}

int py_profile_fn(PyObject *obj, PyFrameObject *frame, int event, PyObject *arg) {
    PyEvent e {
        event,
        frame->f_code
    };
    py_events_.push_back(e);
    return 0;
}

void enable_py_tracing() {
    TORCH_CHECK(!handle_.has_value(), "enable_py_tracing() was called, but tracing is already enabled.");
    auto callback = at::RecordFunctionCallback(onFunctionEnter, onFunctionExit)
        .needsInputs(true)
        .needsOutputs(true);
    handle_ = at::addGlobalCallback(callback);
    PyEval_SetProfile(py_profile_fn, NULL);
}

std::tuple<std::vector<ObserverEvent>, std::vector<PyEvent>> disable_py_tracing() {
    TORCH_CHECK(handle_.has_value(), "disable_py_tracing() was called, but tracing is not active.");
    at::removeCallback(handle_.value());
    handle_.reset();
    PyEval_SetProfile(NULL, NULL);
    std::vector<ObserverEvent> observer_out(observer_events_);
    observer_events_.clear();

    std::vector<PyEvent> py_out(py_events_);
    py_events_.clear();

    return {observer_out, py_out};
}

void register_trace(py::module& py_module) {
    py::module trace_module = py_module.def_submodule("_trace");

    py::enum_<Tag>(trace_module, "Tag")
        .value("kTensor", Tag::kTensor)
        .value("kInt", Tag::kInt)
        .value("kOther", Tag::kOther);

    py::class_<Element>(trace_module, "Element")
        .def_property_readonly("is_input", [](const Element& self) { return self.is_input_; })
        .def_property_readonly("tag", [](const Element& self) { return self.tag_; })
        .def_property_readonly("id", [](const Element& self) { return self.id_; });

    py::class_<ObserverEvent>(trace_module, "ObserverEvent")
        .def_property_readonly("name", [](const ObserverEvent& self) { return self.name; })
        .def_property_readonly("elements", [](const ObserverEvent& self) { return self.elements; });

    py::class_<PyEvent>(trace_module, "PyEvent")
        .def_property_readonly("event", [](const PyEvent& self) { return self.event; })
        .def_property_readonly("f_code_id", [](const PyEvent& self) { return (size_t)(self.f_code); });

    trace_module.def(
        "_enter_module_trace",
        &enable_py_tracing
    );

    trace_module.def(
        "_exit_module_trace",
        &disable_py_tracing
    );
}

}}}  // torch::profiler::impl
