/*
 * adjoint.h — adjoint (reverse-mode) differentiation for the variational
 * training loop. All parameter gradients of <psi|H|psi> in one forward +
 * one backward pass (Jones & Gacon 2020), vs 2P forward passes for
 * parameter-shift.
 *
 * Phase 2: dense complex128 state, self-contained (promotes the validated
 * spike kernels, spike/adjoint_spike.cpp). Deliberately structured around a
 * small set of state operations — apply / apply_inv / generator / dot — so
 * Phase 3 can swap the dense std::vector for qubit's tiered-compressed
 * representation and store compressed checkpoints on the backward path.
 *
 * Optimized: OpenMP over the 2^n hot loops (disjoint index pairs, safe to
 * parallelize), and the per-parameter gradient is computed as a single fused
 * <lambda|G|phi> pass with no state copy. Pragmas are no-ops without /openmp.
 *
 * Header-only, namespace qtrain; no dependency on qubit.h. Driven from
 * Python via the qubit_native bindings (ACircuit + value_and_grad).
 */
#pragma once
#include <vector>
#include <complex>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <utility>
#include <tuple>

namespace qtrain {

using cd = std::complex<double>;
using Vec = std::vector<cd>;

/* generator of a parametric single-qubit rotation exp(-i theta G / 2) */
enum Gen { GEN_NONE = 0, GEN_X = 1, GEN_Y = 2, GEN_Z = 3 };

struct AGate {
	int q = 0;                 /* target qubit */
	std::vector<int> ctrl;     /* control qubits (fixed gates only) */
	cd m[4] = {};              /* fixed-gate 2x2 (row-major) when gen==NONE */
	int gen = GEN_NONE;        /* parametric rotation generator */
	double theta = 0.0;        /* parametric angle */
	bool param = false;        /* trainable rotation contributing a gradient */
	int pidx = -1;             /* gradient slot */
};

/* 2x2 matrix of exp(-i theta G / 2) */
inline void rot_matrix(int gen, double th, cd m[4]) {
	double c = std::cos(th / 2), s = std::sin(th / 2);
	if (gen == GEN_Z)      { m[0] = cd(c, -s); m[1] = 0; m[2] = 0; m[3] = cd(c, s); }
	else if (gen == GEN_Y) { m[0] = c; m[1] = -s; m[2] = s; m[3] = c; }
	else                   { m[0] = c; m[1] = cd(0, -s); m[2] = cd(0, -s); m[3] = c; } /* X */
}

inline void dagger(const cd in[4], cd out[4]) {
	out[0] = std::conj(in[0]); out[1] = std::conj(in[2]);
	out[2] = std::conj(in[1]); out[3] = std::conj(in[3]);
}

/* apply a 2x2 on target q, conditioned on all control bits set */
inline void apply_2x2(Vec& s, const std::vector<int>& ctrl, int q, const cd m[4]) {
	const uint64_t bit = 1ull << q;
	uint64_t cmask = 0;
	for (int cq : ctrl) cmask |= 1ull << cq;
	const cd m0 = m[0], m1 = m[1], m2 = m[2], m3 = m[3];
	const long long N = (long long)s.size();
	cd* p = s.data();
	#pragma omp parallel for schedule(static)
	for (long long i = 0; i < N; i++) {
		if (i & bit) continue;
		if ((i & cmask) != cmask) continue;
		uint64_t j = i | bit;
		cd a = p[i], b = p[j];
		p[i] = m0 * a + m1 * b;
		p[j] = m2 * a + m3 * b;
	}
}

inline void apply_gate(Vec& s, const AGate& g) {
	if (g.gen != GEN_NONE) { cd m[4]; rot_matrix(g.gen, g.theta, m); apply_2x2(s, g.ctrl, g.q, m); }
	else apply_2x2(s, g.ctrl, g.q, g.m);
}

inline void apply_gate_inv(Vec& s, const AGate& g) {
	if (g.gen != GEN_NONE) { cd m[4]; rot_matrix(g.gen, -g.theta, m); apply_2x2(s, g.ctrl, g.q, m); }
	else { cd md[4]; dagger(g.m, md); apply_2x2(s, g.ctrl, g.q, md); }
}

/* Pauli generator G (no phase, no controls) — the dU/dtheta factor */
inline void apply_generator(Vec& s, int q, int gen) {
	const uint64_t bit = 1ull << q;
	const long long N = (long long)s.size();
	cd* p = s.data();
	#pragma omp parallel for schedule(static)
	for (long long i = 0; i < N; i++) {
		if (i & bit) continue;
		uint64_t j = i | bit; cd a = p[i], b = p[j];
		if (gen == GEN_X) { p[i] = b; p[j] = a; }
		else if (gen == GEN_Y) { p[i] = cd(0, -1) * b; p[j] = cd(0, 1) * a; }
		else { p[j] = -b; }   /* Z */
	}
}

/* fused gradient inner product: Re( (-i/2) <lambda| G |phi> ) * 2, no copy.
   iterate disjoint pairs (i, j=i|bit); accumulate conj(lambda)*(G phi). */
inline double grad_term(const Vec& lambda, const Vec& phi, int q, int gen) {
	const uint64_t bit = 1ull << q;
	const long long N = (long long)phi.size();
	const cd* L = lambda.data();
	const cd* P = phi.data();
	double re = 0, im = 0;   /* accumulate <lambda|G|phi> */
	#pragma omp parallel for schedule(static) reduction(+:re,im)
	for (long long i = 0; i < N; i++) {
		if (i & bit) continue;
		uint64_t j = i | bit;
		cd acc;
		if (gen == GEN_X)      acc = std::conj(L[i]) * P[j] + std::conj(L[j]) * P[i];
		else if (gen == GEN_Y) acc = std::conj(L[i]) * (cd(0,-1) * P[j]) + std::conj(L[j]) * (cd(0,1) * P[i]);
		else                   acc = std::conj(L[i]) * P[i] + std::conj(L[j]) * (-P[j]);
		re += acc.real(); im += acc.imag();
	}
	/* g = 2 Re( (-i/2) * <lambda|G|phi> ) = Im(<lambda|G|phi>) */
	(void)re;
	return im;
}

/* int16-tier block-scaled quantization round-trip (the COMPRESSED tier's
   exact transform). `levels` sets coarseness: fine ~32767 == int16, coarse
   == small. Models storing the trajectory compressed and reading it back;
   returns the injected L2 norm (the per-boundary budget contribution). */
inline double quantize_roundtrip(Vec& s, int levels) {
	double mx = 0;
	for (auto& z : s) { mx = std::max(mx, std::fabs(z.real())); mx = std::max(mx, std::fabs(z.imag())); }
	if (mx == 0) return 0;
	const double scale = mx / levels;
	const long long N = (long long)s.size();
	cd* p = s.data();
	double err2 = 0;
	#pragma omp parallel for schedule(static) reduction(+:err2)
	for (long long i = 0; i < N; i++) {
		double re = std::round(p[i].real() / scale) * scale;
		double im = std::round(p[i].imag() / scale) * scale;
		double dr = p[i].real() - re, di = p[i].imag() - im;
		err2 += dr * dr + di * di;
		p[i] = cd(re, im);
	}
	return std::sqrt(err2);
}

/* ---- Hamiltonian: weighted sum of Pauli strings ---- */
/* pauli code per (wire): GEN_X/Y/Z */
struct Term { double coeff; std::vector<std::pair<int, int>> ops; };
using Ham = std::vector<Term>;

inline void add_term_into(Vec& out, const Vec& s, const Term& t) {
	Vec r = s;
	for (auto& op : t.ops) apply_generator(r, op.first, op.second);
	const long long N = (long long)out.size();
	const double c = t.coeff;
	cd* o = out.data(); const cd* rp = r.data();
	#pragma omp parallel for schedule(static)
	for (long long i = 0; i < N; i++) o[i] += c * rp[i];
}
inline double energy(const Vec& psi, const Ham& H) {
	double e = 0;
	for (auto& t : H) {
		Vec ts = psi;
		for (auto& op : t.ops) apply_generator(ts, op.first, op.second);
		const long long N = (long long)psi.size();
		const cd* pp = psi.data(); const cd* tp = ts.data();
		double re = 0;
		#pragma omp parallel for schedule(static) reduction(+:re)
		for (long long i = 0; i < N; i++) re += (std::conj(pp[i]) * tp[i]).real();
		e += t.coeff * re;
	}
	return e;
}
inline Vec apply_ham(const Vec& psi, const Ham& H) {
	Vec r(psi.size(), cd(0, 0));
	for (auto& t : H) add_term_into(r, psi, t);
	return r;
}

/* ---- circuit builder driven from Python ---- */
class ACircuit {
public:
	explicit ACircuit(int n) : n_(n) {}
	int num_qubits() const { return n_; }
	int num_params() const { return nparams_; }

	/* parametric single-qubit rotation; slot<0 means fixed (non-trainable) */
	void rot(int gen, int q, double theta, bool trainable, int slot) {
		AGate g; g.gen = gen; g.q = q; g.theta = theta;
		g.param = trainable; g.pidx = trainable ? slot : -1;
		if (trainable && slot + 1 > nparams_) nparams_ = slot + 1;
		gates_.push_back(g);
	}
	void fixed(int q, cd m00, cd m01, cd m10, cd m11) {
		AGate g; g.q = q; g.m[0] = m00; g.m[1] = m01; g.m[2] = m10; g.m[3] = m11;
		gates_.push_back(g);
	}
	void cfixed(std::vector<int> ctrl, int q, cd m00, cd m01, cd m10, cd m11) {
		AGate g; g.q = q; g.ctrl = std::move(ctrl);
		g.m[0] = m00; g.m[1] = m01; g.m[2] = m10; g.m[3] = m11;
		gates_.push_back(g);
	}

	const std::vector<AGate>& gates() const { return gates_; }

	Vec forward() const {
		Vec s(1ull << n_, cd(0, 0)); s[0] = 1;
		for (auto& g : gates_) apply_gate(s, g);
		return s;
	}

	/* value = <psi|H|psi>; grad[p] = d value / d theta_p (adjoint) */
	std::pair<double, std::vector<double>> value_and_grad(const Ham& H) const {
		Vec psi = forward();
		double value = energy(psi, H);
		Vec lambda = apply_ham(psi, H);
		Vec phi = std::move(psi);
		std::vector<double> grad(nparams_, 0.0);
		for (int k = int(gates_.size()) - 1; k >= 0; k--) {
			const AGate& g = gates_[k];
			if (g.param && g.pidx >= 0)
				grad[g.pidx] = grad_term(lambda, phi, g.q, g.gen);
			apply_gate_inv(phi, g);
			apply_gate_inv(lambda, g);
		}
		return {value, grad};
	}

	/* Phase 3 [CORE]: adjoint where the carried trajectories (phi, lambda)
	   are round-tripped through int16 compression at each gate boundary.
	   Returns (value, grad, D) with D the total injected L2 norm — the
	   budget the paper-2 bound relates to the gradient error. levels<=0
	   reduces to the exact value_and_grad. */
	std::tuple<double, std::vector<double>, double>
	value_and_grad_q(const Ham& H, int levels) const {
		Vec psi = forward();
		double value = energy(psi, H);
		Vec lambda = apply_ham(psi, H);
		Vec phi = std::move(psi);
		double D = 0;
		std::vector<double> grad(nparams_, 0.0);
		for (int k = int(gates_.size()) - 1; k >= 0; k--) {
			const AGate& g = gates_[k];
			if (g.param && g.pidx >= 0)
				grad[g.pidx] = grad_term(lambda, phi, g.q, g.gen);
			apply_gate_inv(phi, g);
			apply_gate_inv(lambda, g);
			if (levels > 0) {
				D += quantize_roundtrip(phi, levels);
				D += quantize_roundtrip(lambda, levels);
			}
		}
		return {value, grad, D};
	}

private:
	int n_;
	int nparams_ = 0;
	std::vector<AGate> gates_;
};

} // namespace qtrain
