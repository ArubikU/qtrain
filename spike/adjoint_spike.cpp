/*
 * adjoint_spike.cpp — Phase 0, step 1.
 *
 * Validate adjoint differentiation against parameter-shift on a raw
 * statevector (self-contained; no engine dependency yet). If the two
 * gradients agree to ~1e-9, the adjoint method is implemented correctly
 * and we can move to step 2 (adjoint through int16 compression).
 *
 * Cost C(theta) = <psi|O|psi>, |psi> = U(theta)|0>, O = Z_0 (Pauli Z on
 * qubit 0). Circuit: `layers` of {RY,RZ per qubit} + CNOT ring.
 *
 * Adjoint (Jones & Gacon 2020): with |lambda> = O|psi| and |phi| swept
 * backward, for a gate exp(-i theta G / 2),
 *     dC/dtheta = 2 Re( <lambda| (-i/2) G |phi_after_gate> ).
 *
 * Build: cl /EHsc /std:c++17 /O2 adjoint_spike.cpp /Fe:adjoint_spike.exe
 */
#include <vector>
#include <complex>
#include <random>
#include <cstdio>
#include <cmath>
#include <algorithm>

using cd = std::complex<double>;
using Vec = std::vector<cd>;
static const double PI = 3.14159265358979323846;

enum Pauli { X, Y, Z };
struct Gate { bool param; int q, q2; Pauli gen; double theta; bool cnot; };

/* apply a single-qubit rotation exp(-i theta G / 2) on qubit q, in place */
static void rot(Vec& s, int n, int q, Pauli g, double th) {
	double c = std::cos(th / 2), sn = std::sin(th / 2);
	/* 2x2 matrix of exp(-i th G/2) */
	cd m00, m01, m10, m11;
	if (g == Z) { m00 = cd(c, -sn); m01 = 0; m10 = 0; m11 = cd(c, sn); }
	else if (g == Y) { m00 = c; m01 = -sn; m10 = sn; m11 = c; }
	else { m00 = c; m01 = cd(0, -sn); m10 = cd(0, -sn); m11 = c; }  /* X */
	uint64_t bit = 1ull << q;
	for (uint64_t i = 0; i < s.size(); i++) {
		if (i & bit) continue;
		uint64_t j = i | bit;
		cd a = s[i], b = s[j];
		s[i] = m00 * a + m01 * b;
		s[j] = m10 * a + m11 * b;
	}
}

/* apply Pauli generator G (no phase) on qubit q, in place */
static void pauli(Vec& s, int q, Pauli g) {
	uint64_t bit = 1ull << q;
	for (uint64_t i = 0; i < s.size(); i++) {
		if (i & bit) continue;
		uint64_t j = i | bit;
		cd a = s[i], b = s[j];
		if (g == X) { s[i] = b; s[j] = a; }
		else if (g == Y) { s[i] = cd(0, -1) * b; s[j] = cd(0, 1) * a; }
		else { s[j] = -b; }   /* Z */
	}
}

static void cnot(Vec& s, int c, int t) {
	uint64_t cb = 1ull << c, tb = 1ull << t;
	for (uint64_t i = 0; i < s.size(); i++)
		if ((i & cb) && !(i & tb)) std::swap(s[i], s[i | tb]);
}

static void apply(Vec& s, int n, const Gate& g) {
	if (g.cnot) cnot(s, g.q, g.q2);
	else rot(s, n, g.q, g.gen, g.theta);
}

/* forward: |psi> = U|0> */
static Vec forward(int n, const std::vector<Gate>& gs) {
	Vec s(1ull << n, cd(0, 0)); s[0] = 1;
	for (auto& g : gs) apply(s, n, g);
	return s;
}

/* cost C = <psi| Z_0 |psi> */
static double cost(const Vec& psi) {
	double e = 0;
	for (uint64_t i = 0; i < psi.size(); i++)
		e += ((i & 1) ? -1.0 : 1.0) * std::norm(psi[i]);
	return e;
}

/* --- parameter-shift reference gradient --- */
static std::vector<double> grad_pshift(int n, std::vector<Gate> gs) {
	std::vector<double> g;
	for (auto& gate : gs) {
		if (!gate.param) continue;
		double save = gate.theta;
		gate.theta = save + PI / 2; double cp = cost(forward(n, gs));
		gate.theta = save - PI / 2; double cm = cost(forward(n, gs));
		gate.theta = save;
		g.push_back(0.5 * (cp - cm));
	}
	return g;
}

/* --- adjoint gradient (one forward + one backward) --- */
static std::vector<double> grad_adjoint(int n, const std::vector<Gate>& gs) {
	Vec psi = forward(n, gs);
	Vec lambda = psi; pauli(lambda, 0, Z);   /* |lambda> = O|psi>, O = Z_0 */
	Vec phi = psi;
	std::vector<double> grad;                 /* filled reverse, fix order after */
	for (int k = int(gs.size()) - 1; k >= 0; k--) {
		const Gate& g = gs[k];
		if (g.param) {
			/* phi is phi_k (state after gate k). mu = (-i/2) G |phi_k> */
			Vec mu = phi; pauli(mu, g.q, g.gen);
			cd acc = 0;                        /* <lambda | (-i/2 G) | phi_k> */
			for (uint64_t i = 0; i < mu.size(); i++) acc += std::conj(lambda[i]) * mu[i];
			acc *= cd(0, -0.5);
			grad.push_back(2.0 * acc.real());
		}
		/* undo gate k on both states */
		if (g.cnot) { cnot(phi, g.q, g.q2); cnot(lambda, g.q, g.q2); }
		else { rot(phi, n, g.q, g.gen, -g.theta); rot(lambda, n, g.q, g.gen, -g.theta); }
	}
	std::reverse(grad.begin(), grad.end());   /* was collected last-to-first */
	return grad;
}

int main() {
	std::mt19937_64 rng(12345);
	std::uniform_real_distribution<double> u(-PI, PI);

	double worst = 0;
	int trials = 20;
	for (int t = 0; t < trials; t++) {
		int n = 3 + int(rng() % 4);            /* 3..6 qubits */
		int layers = 2 + int(rng() % 3);
		std::vector<Gate> gs;
		for (int L = 0; L < layers; L++) {
			for (int q = 0; q < n; q++) {
				gs.push_back({true, q, 0, Y, u(rng), false});
				gs.push_back({true, q, 0, Z, u(rng), false});
			}
			for (int q = 0; q < n; q++)
				gs.push_back({false, q, (q + 1) % n, Z, 0, true});
		}
		auto ga = grad_adjoint(n, gs);
		auto gp = grad_pshift(n, gs);
		double d = 0;
		for (size_t i = 0; i < ga.size(); i++) d = std::max(d, std::fabs(ga[i] - gp[i]));
		worst = std::max(worst, d);
		if (t < 3) printf("trial %d: n=%d params=%zu  max|adj-pshift|=%.2e\n", t, n, ga.size(), d);
	}
	printf("\nworst max|adjoint - parameter-shift| over %d trials: %.3e\n", trials, worst);
	printf("%s\n", worst < 1e-9 ? "PASS: adjoint gradients are correct." : "FAIL: mismatch, debug adjoint.");
	return 0;
}
