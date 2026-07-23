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
	uint64_t bit = 1ull << q, cmask = 0;
	for (int cq : ctrl) cmask |= 1ull << cq;
	for (uint64_t i = 0; i < s.size(); i++) {
		if (i & bit) continue;
		if ((i & cmask) != cmask) continue;
		uint64_t j = i | bit;
		cd a = s[i], b = s[j];
		s[i] = m[0] * a + m[1] * b;
		s[j] = m[2] * a + m[3] * b;
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
	uint64_t bit = 1ull << q;
	for (uint64_t i = 0; i < s.size(); i++) {
		if (i & bit) continue;
		uint64_t j = i | bit; cd a = s[i], b = s[j];
		if (gen == GEN_X) { s[i] = b; s[j] = a; }
		else if (gen == GEN_Y) { s[i] = cd(0, -1) * b; s[j] = cd(0, 1) * a; }
		else { s[j] = -b; }   /* Z */
	}
}

/* ---- Hamiltonian: weighted sum of Pauli strings ---- */
/* pauli code per (wire): GEN_X/Y/Z */
struct Term { double coeff; std::vector<std::pair<int, int>> ops; };
using Ham = std::vector<Term>;

inline Vec apply_term(const Vec& s, const Term& t) {
	Vec r = s;
	for (auto& op : t.ops) apply_generator(r, op.first, op.second);
	return r;
}
inline double energy(const Vec& psi, const Ham& H) {
	double e = 0;
	for (auto& t : H) {
		Vec ts = apply_term(psi, t); cd acc = 0;
		for (uint64_t i = 0; i < psi.size(); i++) acc += std::conj(psi[i]) * ts[i];
		e += t.coeff * acc.real();
	}
	return e;
}
inline Vec apply_ham(const Vec& psi, const Ham& H) {
	Vec r(psi.size(), cd(0, 0));
	for (auto& t : H) { Vec ts = apply_term(psi, t); for (uint64_t i = 0; i < r.size(); i++) r[i] += t.coeff * ts[i]; }
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
		Vec phi = psi;
		std::vector<double> grad(nparams_, 0.0);
		for (int k = int(gates_.size()) - 1; k >= 0; k--) {
			const AGate& g = gates_[k];
			if (g.param && g.pidx >= 0) {
				Vec mu = phi; apply_generator(mu, g.q, g.gen);
				cd acc = 0;
				for (uint64_t i = 0; i < mu.size(); i++) acc += std::conj(lambda[i]) * mu[i];
				acc *= cd(0, -0.5);
				grad[g.pidx] = 2.0 * acc.real();
			}
			apply_gate_inv(phi, g);
			apply_gate_inv(lambda, g);
		}
		return {value, grad};
	}

private:
	int n_;
	int nparams_ = 0;
	std::vector<AGate> gates_;
};

} // namespace qtrain
