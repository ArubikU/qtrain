/*
 * compress_spike.cpp — Phase 0, steps 2 & 3.
 *
 * Step 2: run adjoint differentiation while the carried states are
 * round-tripped through int16 block-scaled quantization (the
 * COMPRESSED tier's exact transform) at every gate boundary. Sweep the
 * quantization coarseness, measure the total injected L2 norm D and the
 * resulting max gradient error. Theory (theory/gradient-bound.md)
 * predicts the error is ~linear in D.
 *
 * Step 3: train a small VQE (transverse-field Ising ground state) with
 * exact vs compressed gradients and compare the converged energy — does
 * the deterministic quantization bias break training?
 *
 * Build: cl /EHsc /std:c++17 /O2 compress_spike.cpp /Fe:compress_spike.exe
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

static void rot(Vec& s, int q, Pauli g, double th) {
	double c = std::cos(th/2), sn = std::sin(th/2);
	cd m00, m01, m10, m11;
	if (g == Z) { m00 = cd(c,-sn); m01 = 0; m10 = 0; m11 = cd(c,sn); }
	else if (g == Y) { m00 = c; m01 = -sn; m10 = sn; m11 = c; }
	else { m00 = c; m01 = cd(0,-sn); m10 = cd(0,-sn); m11 = c; }
	uint64_t bit = 1ull << q;
	for (uint64_t i = 0; i < s.size(); i++) if (!(i & bit)) {
		uint64_t j = i | bit; cd a = s[i], b = s[j];
		s[i] = m00*a + m01*b; s[j] = m10*a + m11*b;
	}
}
static void pauli(Vec& s, int q, Pauli g) {
	uint64_t bit = 1ull << q;
	for (uint64_t i = 0; i < s.size(); i++) if (!(i & bit)) {
		uint64_t j = i | bit; cd a = s[i], b = s[j];
		if (g == X) { s[i] = b; s[j] = a; }
		else if (g == Y) { s[i] = cd(0,-1)*b; s[j] = cd(0,1)*a; }
		else { s[j] = -b; }
	}
}
static void cnot(Vec& s, int c, int t) {
	uint64_t cb = 1ull<<c, tb = 1ull<<t;
	for (uint64_t i = 0; i < s.size(); i++) if ((i&cb) && !(i&tb)) std::swap(s[i], s[i|tb]);
}
static void apply(Vec& s, const Gate& g) { g.cnot ? cnot(s,g.q,g.q2) : rot(s,g.q,g.gen,g.theta); }
static Vec forward(int n, const std::vector<Gate>& gs) {
	Vec s(1ull<<n, cd(0,0)); s[0]=1; for (auto& g : gs) apply(s, g); return s;
}

/* int16-style block-scaled quantization round-trip; `levels` sets the
 * coarseness (fine=32767 ~ int16, coarse=small). Returns injected norm. */
static double quantize_roundtrip(Vec& s, int levels) {
	double mx = 0;
	for (auto& z : s) mx = std::max({mx, std::fabs(z.real()), std::fabs(z.imag())});
	if (mx == 0) return 0;
	double scale = mx / levels;
	double err2 = 0;
	for (auto& z : s) {
		double re = std::round(z.real()/scale)*scale;
		double im = std::round(z.imag()/scale)*scale;
		err2 += (z.real()-re)*(z.real()-re) + (z.imag()-im)*(z.imag()-im);
		z = cd(re, im);
	}
	return std::sqrt(err2);
}

/* ---------- Hamiltonian (sum of weighted Pauli strings) ---------- */
struct Term { double coeff; std::vector<std::pair<int,Pauli>> ops; };
using Ham = std::vector<Term>;

static Ham tfim(int n, double J, double h) {   /* -J ZZ - h X */
	Ham H;
	for (int i = 0; i + 1 < n; i++) H.push_back({-J, {{i,Z},{i+1,Z}}});
	for (int i = 0; i < n; i++) H.push_back({-h, {{i,X}}});
	return H;
}
static Vec apply_term(const Vec& s, const Term& t) {
	Vec r = s; for (auto& op : t.ops) pauli(r, op.first, op.second); return r;
}
static double energy(const Vec& psi, const Ham& H) {
	double e = 0;
	for (auto& t : H) {
		Vec ts = apply_term(psi, t); cd acc = 0;
		for (uint64_t i = 0; i < psi.size(); i++) acc += std::conj(psi[i]) * ts[i];
		e += t.coeff * acc.real();
	}
	return e;
}
static Vec apply_ham(const Vec& psi, const Ham& H) {
	Vec r(psi.size(), cd(0,0));
	for (auto& t : H) { Vec ts = apply_term(psi, t); for (uint64_t i=0;i<r.size();i++) r[i] += t.coeff*ts[i]; }
	return r;
}

/* adjoint gradient of <psi|H|psi>; if levels>0, round-trip carried
 * states through quantization each gate boundary; report injected D. */
static std::vector<double> grad_adjoint(int n, const std::vector<Gate>& gs,
					const Ham& H, int levels, double* Dtot) {
	Vec psi = forward(n, gs);
	Vec lambda = apply_ham(psi, H);
	Vec phi = psi;
	double D = 0;
	std::vector<double> grad;
	for (int k = int(gs.size())-1; k >= 0; k--) {
		const Gate& g = gs[k];
		if (g.param) {
			Vec mu = phi; pauli(mu, g.q, g.gen);
			cd acc = 0; for (uint64_t i=0;i<mu.size();i++) acc += std::conj(lambda[i])*mu[i];
			acc *= cd(0,-0.5);
			grad.push_back(2.0*acc.real());
		}
		if (g.cnot) { cnot(phi,g.q,g.q2); cnot(lambda,g.q,g.q2); }
		else { rot(phi,g.q,g.gen,-g.theta); rot(lambda,g.q,g.gen,-g.theta); }
		if (levels > 0) { D += quantize_roundtrip(phi, levels); D += quantize_roundtrip(lambda, levels); }
	}
	std::reverse(grad.begin(), grad.end());
	if (Dtot) *Dtot = D;
	return grad;
}

static std::vector<Gate> ansatz(int n, int layers, std::mt19937_64& g) {
	std::uniform_real_distribution<double> u(-PI, PI);
	std::vector<Gate> gs;
	for (int L = 0; L < layers; L++) {
		for (int q = 0; q < n; q++) { gs.push_back({true,q,0,Y,u(g),false}); gs.push_back({true,q,0,Z,u(g),false}); }
		for (int q = 0; q < n; q++) gs.push_back({false,q,(q+1)%n,Z,0,true});
	}
	return gs;
}

int main() {
	std::mt19937_64 rng(777);

	/* ---- Step 2: gradient error vs injected budget D ---- */
	printf("=== Step 2: adjoint gradient error vs compression budget ===\n");
	int n = 6, layers = 3;
	auto gs = ansatz(n, layers, rng);
	Ham H = tfim(n, 1.0, 1.0);
	double dummy;
	auto gexact = grad_adjoint(n, gs, H, 0, &dummy);
	printf("levels     injected D     max|grad err|   err/D\n");
	for (int levels : {4, 8, 16, 32, 64, 256, 1024}) {
		double D;
		auto gc = grad_adjoint(n, gs, H, levels, &D);
		double e = 0; for (size_t i=0;i<gc.size();i++) e = std::max(e, std::fabs(gc[i]-gexact[i]));
		printf("%5d     %.3e     %.3e     %.3f\n", levels, D, e, D>0? e/D : 0);
	}
	printf("(linear in D => err/D roughly constant across rows)\n");

	/* ---- Step 3: train VQE with exact vs compressed gradients ---- */
	printf("\n=== Step 3: VQE training, exact vs compressed gradients ===\n");
	int nv = 4;
	Ham Hv = tfim(nv, 1.0, 1.0);
	auto base = ansatz(nv, 3, rng);

	auto train = [&](int levels) {
		auto gs2 = base;                    /* same init for fair comparison */
		double lr = 0.1; int steps = 120;
		for (int s = 0; s < steps; s++) {
			double D;
			auto grad = grad_adjoint(nv, gs2, Hv, levels, &D);
			int p = 0;
			for (auto& g : gs2) if (g.param) g.theta -= lr * grad[p++];
		}
		return energy(forward(nv, gs2), Hv);
	};
	double e_exact = train(0);
	double e_c256  = train(256);
	double e_c32   = train(32);
	double e_c8    = train(8);
	printf("final energy  exact-grad : %.5f\n", e_exact);
	printf("final energy  levels=256 : %.5f  (gap %.1e)\n", e_c256, std::fabs(e_c256-e_exact));
	printf("final energy  levels=32  : %.5f  (gap %.1e)\n", e_c32,  std::fabs(e_c32 -e_exact));
	printf("final energy  levels=8   : %.5f  (gap %.1e)\n", e_c8,   std::fabs(e_c8  -e_exact));
	printf("\nif compressed-gradient training reaches ~the same energy,\n"
	       "quantization bias does not break training (Step 3 PASS).\n");
	return 0;
}
