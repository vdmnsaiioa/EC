"""
E3-QP4's cost (notes/e3qp4-preregistration.md section 3d): one training step of E_0 (the loss and its gradient, energies
100 and forces 1, as E3-QP3 trains) on a batch of 32 clusters of 3-5 waters, at width 16 and at width 32.  The ratio
scales E3-QP3's times on the GPU to job A's and B's.

    python scratch/e3qp4_width_timing.py          # ~1 min, CPU
"""
import os, sys, time
import jax

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from sseft import water as W, model as M, structure as S, train as T
import e3_water as E3

structs = W.cluster_dataset((3, 4, 5), 11, seed=0, mc_steps=2)[:32]
data = S.pad_batch(structs)
times = {}
for F in (16, 32):
    a = type("A", (), dict(lA=4.0, rcut=6.0, F=F, l1=0.75, rs_over_l1=0.0, response_edge=0.0))()
    rung = E3.configure(E3.RUNGS["M_inf"], a)
    params = M.init_params(jax.random.PRNGKey(0), rung)
    loss = T.make_loss(rung, 1e-3, w_energy=100.0, w_force=1.0)
    step = jax.jit(jax.value_and_grad(lambda p: loss(p, data)[0]))
    t0 = time.time(); jax.block_until_ready(step(params)); compile_s = time.time() - t0
    t0 = time.time()
    for _ in range(10):
        jax.block_until_ready(step(params))
    times[F] = (time.time() - t0) / 10
    print(f"width {F}: compile {compile_s:.1f} s, step {times[F] * 1000:.0f} ms", flush=True)
print(f"width 32 / width 16: {times[32] / times[16]:.2f}")
