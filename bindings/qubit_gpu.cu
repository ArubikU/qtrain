/*
 * qubit_gpu.cu — pybind11 module (qubit_gpu_native) exposing qubit's CUDA
 * adjoint executors to Python. This is the implementation/adapter layer;
 * the kernels and executor classes are the library (qubit/adjoint_gpu.cuh).
 *
 * Build: nvcc -arch=sm_XX --shared (bindings/build_gpu.{bat,sh}).
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "qubit/adjoint_gpu.cuh"

namespace py = pybind11;
using qubit::Ham;

PYBIND11_MODULE(qubit_gpu_native, m) {
	m.doc() = "CUDA adjoint training executors for the qubit simulator (complex64 / int16).";
	py::class_<GPUCircuit>(m, "GPUCircuit")
		.def(py::init<int>(), py::arg("num_qubits"))
		.def_property_readonly("num_qubits", &GPUCircuit::num_qubits)
		.def_property_readonly("num_params", &GPUCircuit::num_params)
		.def("rot", &GPUCircuit::rot,
		     py::arg("gen"), py::arg("q"), py::arg("theta"), py::arg("trainable"), py::arg("slot"))
		.def("fixed", &GPUCircuit::fixed,
		     py::arg("q"), py::arg("m00"), py::arg("m01"), py::arg("m10"), py::arg("m11"))
		.def("cfixed", &GPUCircuit::cfixed,
		     py::arg("ctrl"), py::arg("q"), py::arg("m00"), py::arg("m01"), py::arg("m10"), py::arg("m11"))
		.def("value_and_grad",
		     [](GPUCircuit& c,
		        const std::vector<std::pair<double, std::vector<std::pair<int, int>>>>& terms) {
			     Ham H; for (auto& t : terms) H.push_back({t.first, t.second});
			     auto r = c.run(H, 0);
			     return std::make_tuple(std::get<0>(r), std::get<1>(r));
		     }, py::arg("hamiltonian"))
		.def("value_and_grad_q",
		     [](GPUCircuit& c,
		        const std::vector<std::pair<double, std::vector<std::pair<int, int>>>>& terms, int levels) {
			     Ham H; for (auto& t : terms) H.push_back({t.first, t.second});
			     return c.run(H, levels);
		     }, py::arg("hamiltonian"), py::arg("levels"));

	py::class_<GPUCircuitQ>(m, "GPUCircuitQ")
		.def(py::init<int>(), py::arg("num_qubits"))
		.def_property_readonly("num_qubits", &GPUCircuitQ::num_qubits)
		.def_property_readonly("num_params", &GPUCircuitQ::num_params)
		.def("rot", &GPUCircuitQ::rot,
		     py::arg("gen"), py::arg("q"), py::arg("theta"), py::arg("trainable"), py::arg("slot"))
		.def("fixed", &GPUCircuitQ::fixed,
		     py::arg("q"), py::arg("m00"), py::arg("m01"), py::arg("m10"), py::arg("m11"))
		.def("cfixed", &GPUCircuitQ::cfixed,
		     py::arg("ctrl"), py::arg("q"), py::arg("m00"), py::arg("m01"), py::arg("m10"), py::arg("m11"))
		.def("value_and_grad",
		     [](GPUCircuitQ& c,
		        const std::vector<std::pair<double, std::vector<std::pair<int, int>>>>& terms) {
			     Ham H; for (auto& t : terms) H.push_back({t.first, t.second});
			     return c.run(H);   /* (value, grad, D) — int16 storage */
		     }, py::arg("hamiltonian"),
		     "Adjoint with int16 resident storage (4 B/amp/traj, half of "
		     "complex64). Returns (value, grad, D).");
}
