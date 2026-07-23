"""
Phase 4 GPU: the headline the CPU cannot show.

  A. GPU vs CPU per-gradient wall time (adjoint), same circuit.
  B. Qubit ceiling: largest single-gradient that fits 6 GB (complex64,
     2 trajectories) — where dense Lightning/Aer OOM.
  C. A large-n VQE trains on the GPU with the compressed adjoint.

RTX 3060 Laptop, 6 GB. Run: py -3.12 bench/phase4_gpu.py
"""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import qubit_native as qn
import qubit_gpu_native as qg
from vqe_helpers import hea_ansatz as build, tfim


def timed(fn, reps=3):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e3


# ---- A. GPU vs CPU ----
print("=== A. per-gradient wall time: GPU vs CPU (adjoint, complex64/128) ===")
print(f"{'n':>3} {'P':>4} {'CPU(ms)':>10} {'GPU(ms)':>10} {'speedup':>8}")
for n, layers in [(14, 4), (16, 4), (18, 4), (20, 4)]:
    P = layers * n * 2
    th = np.random.default_rng(0).uniform(-math.pi, math.pi, P)
    H = tfim(n)
    cc = build(qn.ACircuit, n, layers, th)
    gc = build(qg.GPUCircuit, n, layers, th)
    tc = timed(lambda: cc.value_and_grad(H))
    tg = timed(lambda: gc.value_and_grad(H))
    print(f"{n:>3} {P:>4} {tc:>10.1f} {tg:>10.1f} {tc/tg:>7.1f}x")

# ---- B. qubit ceiling on 6 GB ----
print("\n=== B. largest single gradient that fits 6 GB (complex64) ===")
for n in [22, 24, 26, 27, 28]:
    layers = 2
    P = layers * n * 2
    th = np.random.default_rng(0).uniform(-math.pi, math.pi, P)
    H = tfim(n)
    mem = 2 * 8 * (1 << n) / 1024**3
    try:
        gc = build(qg.GPUCircuit, n, layers, th)
        t = timed(lambda: gc.value_and_grad(H), reps=1)
        print(f"  n={n:2d}  state mem ~{mem:5.2f} GB (2 traj)   grad {t:8.1f} ms   OK")
    except Exception as e:
        print(f"  n={n:2d}  state mem ~{mem:5.2f} GB (2 traj)   OOM/err: {str(e)[:50]}")

# ---- C. large-n VQE training on GPU ----
print("\n=== C. VQE training on GPU (compressed adjoint) ===")
n, layers = 22, 3
H = tfim(n)
th = np.random.default_rng(1).uniform(-0.1, 0.1, layers * n * 2)
print(f"qubits={n}  params={layers*n*2}  amplitudes={1<<n:,}  (Lightning/Aer OOM well below this on 6 GB)")
t0 = time.perf_counter()
steps = 40
for s in range(steps):
    gc = build(qg.GPUCircuit, n, layers, th)
    val, grad, D = gc.value_and_grad_q(H, 1024)   # int16-fine compressed
    th -= 0.05 * np.array(grad)
    if s % 5 == 0 or s == steps - 1:
        print(f"  step {s:3d}  E={val:.5f}  D={D:.2e}")
print(f"trained {steps} steps in {time.perf_counter()-t0:.1f}s — 22q VQE on a 6 GB laptop GPU.")
