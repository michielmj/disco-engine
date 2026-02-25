#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "EventQueue.h"

namespace py = pybind11;

namespace {

// Convert a Python dict {str: str|bool|int|float} -> disco::Headers.
// bool must be checked before int: in Python, bool is a subclass of int,
// so py::isinstance<py::bool_> must come first to avoid storing True/False as long long.
disco::Headers headers_from_py(const py::object& headers_obj) {
    disco::Headers headers;
    if (headers_obj.is_none()) return headers;

    for (auto& item : headers_obj.cast<py::dict>()) {
        std::string key = item.first.cast<std::string>();
        py::object val = py::reinterpret_borrow<py::object>(item.second);

        if (py::isinstance<py::bool_>(val)) {
            headers[key] = val.cast<bool>();
        } else if (py::isinstance<py::int_>(val)) {
            headers[key] = val.cast<long long>();
        } else if (py::isinstance<py::float_>(val)) {
            headers[key] = val.cast<double>();
        } else if (py::isinstance<py::str>(val)) {
            headers[key] = val.cast<std::string>();
        } else {
            throw std::runtime_error(
                "Header values must be str, bool, int, or float; got: " +
                val.attr("__class__").attr("__name__").cast<std::string>()
            );
        }
    }
    return headers;
}

// Convert disco::Headers -> Python dict, preserving the concrete Python type
// for each variant alternative.
py::dict headers_to_py(const disco::Headers& headers) {
    py::dict out;
    for (const auto& kvp : headers) {
        py::object val = std::visit([](const auto& v) -> py::object {
            return py::cast(v);
        }, kvp.second);
        out[py::str(kvp.first)] = std::move(val);
    }
    return out;
}

py::list events_to_py_list(std::vector<disco::Event>&& events) {
    py::list out;
    for (auto& ev : events) {
        // Transfer ownership of the owned PyObject* reference from C++ to Python.
        py::object data = py::reinterpret_steal<py::object>(ev.release_data());
        py::dict headers = headers_to_py(ev.headers);

        // Flat 5-tuple: (sender_node, sender_simproc, epoch, data, headers)
        out.append(py::make_tuple(
            std::move(ev.sender_node),
            std::move(ev.sender_simproc),
            ev.epoch,
            std::move(data),
            std::move(headers)
        ));
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Disco event queue (pybind11 binding)";

    py::class_<disco::EventQueue>(m, "EventQueue")
        .def(py::init<>())

        .def_property_readonly(
            "epoch",
            [](disco::EventQueue& q) -> double {
                // NOTE: We intentionally do NOT release the GIL here,
                // because the core implementation may INCREF/DECREF under internal locks.
                return q.getEpoch();
            })

        .def_property_readonly(
            "next_epoch",
            [](disco::EventQueue& q) -> py::object {
                const double v = q.getNextEpoch();
                if (std::isnan(v)) {
                    return py::none();
                }
                return py::float_(v);
            })

        .def_property_readonly(
            "waiting_for",
            [](disco::EventQueue& q) -> std::string {
                return q.getWaitingFor();
            })

        .def(
            "push",
            [](disco::EventQueue& q,
               const std::string& sender_node,
               const std::string& sender_simproc,
               double epoch,
               py::object data,
               py::object headers_obj) -> bool {
                disco::Headers headers = headers_from_py(headers_obj);

                // Core queue will INCREF the borrowed PyObject* internally (RAII).
                return q.push(sender_node, sender_simproc, epoch, data.ptr(), headers);
            },
            py::arg("sender_node"),
            py::arg("sender_simproc"),
            py::arg("epoch"),
            py::arg("data"),
            py::arg("headers") = py::none())

        .def(
            "pop",
            [](disco::EventQueue& q) -> py::list {
                return events_to_py_list(q.pop());
            })

        .def(
            "promise",
            [](disco::EventQueue& q,
               const std::string& sender_node,
               const std::string& sender_simproc,
               unsigned long seqnr,
               double epoch,
               unsigned long num_events) -> bool {
                return q.promise(sender_node, sender_simproc, seqnr, epoch, num_events);
            },
            py::arg("sender_node"),
            py::arg("sender_simproc"),
            py::arg("seqnr"),
            py::arg("epoch"),
            py::arg("num_events"))

        .def(
            "try_next_epoch",
            [](disco::EventQueue& q) -> bool {
                return q.tryNextEpoch();
            })

        .def_property_readonly(
            "has_predecessors",
            [](disco::EventQueue& q) -> bool {
                return q.hasPredecessors();
            })

        .def(
            "register_predecessor",
            [](disco::EventQueue& q,
               const std::string& sender_node,
               const std::string& sender_simproc) {
                q.registerPredecessor(sender_node, sender_simproc);
            },
            py::arg("sender_node"),
            py::arg("sender_simproc"))

        .def_property_readonly(
            "empty",
            [](disco::EventQueue& q) -> bool {
                return q.empty();
            });
}
