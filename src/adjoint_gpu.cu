/*
 * adjoint_gpu.cu — CUDA port of the adjoint training kernels (adjoint.h).
 *
 * Same algorithm, on the GPU, in complex64 (float) so a 6 GB card holds a
 * larger state. The four state ops become CUDA kernels; the backward pass
 * keeps only the two trajectories (phi, lambda) resident, so the dense
 * ceiling is 2 * 8 * 2^n bytes. Phase 3's compressed variant
 * (value_and_grad_q) round-trips both through the int16 transform on device.
 *
 * Reuses the host structs (AGate, Ham, Gen) from adjoint.h; only execution
 * is GPU. Bound as the `qubit_gpu_native` module (GPUCircuit).
 *
 * Build: nvcc -arch=sm_86 --shared (see bindings/build_gpu.bat).
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cmath>

#include "adjoint.h"   // qtrain::AGate, Gen, Term, Ham (host structs)

namespace py = pybind11;
using qtrain::AGate; using qtrain::Ham; using qtrain::Term;
using qtrain::GEN_NONE; using qtrain::GEN_X; using qtrain::GEN_Y; using qtrain::GEN_Z;

using cf = thrust::complex<float>;

#define CUDA_OK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) \
	throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_)); } while (0)

static const int TPB = 256;
static inline int blocks(uint64_t N) { return int((N + TPB - 1) / TPB); }

/* ---- kernels ---- */
__global__ void k_apply2x2(cf* s, uint64_t N, uint64_t bit, uint64_t cmask,
                           cf m0, cf m1, cf m2, cf m3) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N || (i & bit)) return;
	if ((i & cmask) != cmask) return;
	uint64_t j = i | bit;
	cf a = s[i], b = s[j];
	s[i] = m0 * a + m1 * b;
	s[j] = m2 * a + m3 * b;
}

__global__ void k_generator(cf* s, uint64_t N, uint64_t bit, int gen) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N || (i & bit)) return;
	uint64_t j = i | bit;
	cf a = s[i], b = s[j];
	if (gen == GEN_X) { s[i] = b; s[j] = a; }
	else if (gen == GEN_Y) { s[i] = cf(0, -1) * b; s[j] = cf(0, 1) * a; }
	else { s[j] = -b; }
}

/* out += c * in */
__global__ void k_axpy(cf* out, const cf* in, float c, uint64_t N) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) out[i] += c * in[i];
}
__global__ void k_copy(cf* out, const cf* in, uint64_t N) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) out[i] = in[i];
}
__global__ void k_setzero(cf* s, uint64_t N) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) s[i] = cf(0, 0);
}

/* Re<a|b> accumulation into a double */
__global__ void k_redot(const cf* a, const cf* b, uint64_t N, double* acc) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	cf v = thrust::conj(a[i]) * b[i];
	atomicAdd(acc, (double)v.real());
}

/* Im<lambda|G|phi> over disjoint pairs -> grad term */
__global__ void k_gradterm(const cf* L, const cf* P, uint64_t N, uint64_t bit,
                           int gen, double* acc) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N || (i & bit)) return;
	uint64_t j = i | bit;
	cf t;
	if (gen == GEN_X)      t = thrust::conj(L[i]) * P[j] + thrust::conj(L[j]) * P[i];
	else if (gen == GEN_Y) t = thrust::conj(L[i]) * (cf(0,-1) * P[j]) + thrust::conj(L[j]) * (cf(0,1) * P[i]);
	else                   t = thrust::conj(L[i]) * P[i] + thrust::conj(L[j]) * (-P[j]);
	atomicAdd(acc, (double)t.imag());
}

/* quantization: pass 1 max|.|, pass 2 round + accumulate err^2 */
__global__ void k_absmax(const cf* s, uint64_t N, unsigned* umax) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	float m = fmaxf(fabsf(s[i].real()), fabsf(s[i].imag()));
	atomicMax(umax, __float_as_uint(m));   /* nonneg floats: uint order == float order */
}
__global__ void k_quant(cf* s, uint64_t N, float scale, double* err2) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	float re = rintf(s[i].real() / scale) * scale;
	float im = rintf(s[i].imag() / scale) * scale;
	float dr = s[i].real() - re, di = s[i].imag() - im;
	atomicAdd(err2, (double)(dr * dr + di * di));
	s[i] = cf(re, im);
}

/* ---- host helpers ---- */
static void rot_cf(int gen, double th, cf m[4]) {
	float c = (float)std::cos(th / 2), s = (float)std::sin(th / 2);
	if (gen == GEN_Z)      { m[0] = cf(c, -s); m[1] = 0; m[2] = 0; m[3] = cf(c, s); }
	else if (gen == GEN_Y) { m[0] = cf(c, 0); m[1] = cf(-s, 0); m[2] = cf(s, 0); m[3] = cf(c, 0); }
	else                   { m[0] = cf(c, 0); m[1] = cf(0, -s); m[2] = cf(0, -s); m[3] = cf(c, 0); }
}
static void mat_cf(const AGate& g, cf m[4]) {
	for (int i = 0; i < 4; i++) m[i] = cf((float)g.m[i].real(), (float)g.m[i].imag());
}
static void dagger_cf(const cf in[4], cf out[4]) {
	out[0] = thrust::conj(in[0]); out[1] = thrust::conj(in[2]);
	out[2] = thrust::conj(in[1]); out[3] = thrust::conj(in[3]);
}

struct DevAccum {
	double* d = nullptr;
	DevAccum() { CUDA_OK(cudaMalloc(&d, sizeof(double))); }
	~DevAccum() { cudaFree(d); }
	void zero() { CUDA_OK(cudaMemset(d, 0, sizeof(double))); }
	double get() { double h; CUDA_OK(cudaMemcpy(&h, d, sizeof(double), cudaMemcpyDeviceToHost)); return h; }
};

class GPUCircuit {
public:
	explicit GPUCircuit(int n) : n_(n) {
		if (n < 1 || n > 40) throw std::runtime_error("n out of range");
	}
	int num_qubits() const { return n_; }
	int num_params() const { return nparams_; }

	void rot(int gen, int q, double theta, bool trainable, int slot) {
		AGate g; g.gen = gen; g.q = q; g.theta = theta;
		g.param = trainable; g.pidx = trainable ? slot : -1;
		if (trainable && slot + 1 > nparams_) nparams_ = slot + 1;
		gates_.push_back(g);
	}
	void fixed(int q, std::complex<double> a, std::complex<double> b,
	           std::complex<double> c, std::complex<double> d) {
		AGate g; g.q = q; g.m[0]=a; g.m[1]=b; g.m[2]=c; g.m[3]=d; gates_.push_back(g);
	}
	void cfixed(std::vector<int> ctrl, int q, std::complex<double> a, std::complex<double> b,
	            std::complex<double> c, std::complex<double> d) {
		AGate g; g.q = q; g.ctrl = std::move(ctrl);
		g.m[0]=a; g.m[1]=b; g.m[2]=c; g.m[3]=d; gates_.push_back(g);
	}

	/* levels<=0 => exact; else compress phi/lambda each boundary.
	   returns (value, grad, D). */
	std::tuple<double, std::vector<double>, double>
	run(const Ham& H, int levels) {
		const uint64_t N = uint64_t(1) << n_;
		const int B = blocks(N);
		cf *phi, *lambda;
		CUDA_OK(cudaMalloc(&phi, N * sizeof(cf)));
		CUDA_OK(cudaMalloc(&lambda, N * sizeof(cf)));

		/* forward: phi = U|0> */
		k_setzero<<<B, TPB>>>(phi, N);
		{ cf one(1, 0); CUDA_OK(cudaMemcpy(phi, &one, sizeof(cf), cudaMemcpyHostToDevice)); }
		for (auto& g : gates_) apply_gate(phi, g, N);

		DevAccum acc;
		/* lambda = H|phi>, built WITHOUT a third buffer: apply each term's
		   Pauli string to phi in place, axpy into lambda, then re-apply to
		   restore phi (Paulis are involutions). Peak stays at 2 states, so
		   the dense ceiling is 2*8*2^n bytes. */
		k_setzero<<<B, TPB>>>(lambda, N);
		for (auto& t : H) {
			for (auto& op : t.ops) k_generator<<<B, TPB>>>(phi, N, uint64_t(1) << op.first, op.second);
			k_axpy<<<B, TPB>>>(lambda, phi, (float)t.coeff, N);
			for (auto& op : t.ops) k_generator<<<B, TPB>>>(phi, N, uint64_t(1) << op.first, op.second);
		}
		/* value = <phi|H|phi> = Re<phi|lambda> */
		acc.zero();
		k_redot<<<B, TPB>>>(phi, lambda, N, acc.d);
		double value = acc.get();

		std::vector<double> grad(nparams_, 0.0);
		double D = 0;
		for (int k = int(gates_.size()) - 1; k >= 0; k--) {
			const AGate& g = gates_[k];
			if (g.param && g.pidx >= 0) {
				acc.zero();
				k_gradterm<<<B, TPB>>>(lambda, phi, N, uint64_t(1) << g.q, g.gen, acc.d);
				grad[g.pidx] = acc.get();
			}
			apply_gate_inv(phi, g, N);
			apply_gate_inv(lambda, g, N);
			if (levels > 0) { D += quantize(phi, N, levels); D += quantize(lambda, N, levels); }
		}
		CUDA_OK(cudaFree(phi));
		CUDA_OK(cudaFree(lambda));
		return {value, grad, D};
	}

private:
	void apply_gate(cf* s, const AGate& g, uint64_t N) {
		cf m[4];
		if (g.gen != GEN_NONE) rot_cf(g.gen, g.theta, m); else mat_cf(g, m);
		uint64_t cmask = 0; for (int c : g.ctrl) cmask |= uint64_t(1) << c;
		k_apply2x2<<<blocks(N), TPB>>>(s, N, uint64_t(1) << g.q, cmask, m[0], m[1], m[2], m[3]);
	}
	void apply_gate_inv(cf* s, const AGate& g, uint64_t N) {
		cf m[4];
		if (g.gen != GEN_NONE) rot_cf(g.gen, -g.theta, m);
		else { cf tmp[4]; mat_cf(g, tmp); dagger_cf(tmp, m); }
		uint64_t cmask = 0; for (int c : g.ctrl) cmask |= uint64_t(1) << c;
		k_apply2x2<<<blocks(N), TPB>>>(s, N, uint64_t(1) << g.q, cmask, m[0], m[1], m[2], m[3]);
	}
	double quantize(cf* s, uint64_t N, int levels) {
		unsigned* umax; CUDA_OK(cudaMalloc(&umax, sizeof(unsigned)));
		CUDA_OK(cudaMemset(umax, 0, sizeof(unsigned)));
		k_absmax<<<blocks(N), TPB>>>(s, N, umax);
		unsigned uh; CUDA_OK(cudaMemcpy(&uh, umax, sizeof(unsigned), cudaMemcpyDeviceToHost));
		CUDA_OK(cudaFree(umax));
		float mx = __uint_as_float_host(uh);
		if (mx == 0) return 0;
		float scale = mx / levels;
		DevAccum err; err.zero();
		k_quant<<<blocks(N), TPB>>>(s, N, scale, err.d);
		return std::sqrt(err.get());
	}
	/* host reinterpret of the uint bits produced by __float_as_uint */
	static float __uint_as_float_host(unsigned u) { float f; std::memcpy(&f, &u, 4); return f; }

	int n_;
	int nparams_ = 0;
	std::vector<AGate> gates_;
};

PYBIND11_MODULE(qubit_gpu_native, m) {
	m.doc() = "CUDA adjoint training kernels for the qubit simulator (complex64).";
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
}
