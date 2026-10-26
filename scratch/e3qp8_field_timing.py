"""
E3-QP8's cost (notes/e3qp8-preregistration.md section 3): one training step (the loss and its gradient, energies 100 and
forces 1, as E3-QP4 to E3-QP7 train) on a batch of 32 clusters of 3-5 waters at width 32, for each split rung (Rung.
e0_field_self_on / _off, Rung.e0_field_cross_species) against its base, M_16C2pm_w and E3-QP7's cross rung.  The ratios
scale E3-QP7's times on the A100 to E3-QP8's (E3-QP7 found the A100's ratios about 0.15 above the CPU's).  The site
model's charges are a stand-in of TTM3-F's form (the cost is the geometry's, not the charges').

    python scratch/e3qp8_field_timing.py          # ~5 min, CPU
"""
import os, sys, time
import numpy as np
import jax

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from sseft import water as W, model as M, structure as S, train as T
import e3_water as E3


class SiteModel:
    def site_model(self, pos_ang):
        n = len(pos_ang) // 3
        q = np.zeros(3 * n); w = np.zeros((n, 3 * n))
        for m in range(n):
            q[3 * m + 1:3 * m + 3] = 0.6; w[m, 3 * m:3 * m + 3] = [0.54, 0.23, 0.23]
        return q, w, np.full(n, -1.2), np.arange(n)


structs = W.cluster_dataset((3, 4, 5), 11, seed=0, mc_steps=2)[:32]
W.pin_site_model(structs, SiteModel()); E3.pin_dispersion(structs)
a = type("A", (), dict(lA=4.0, rcut=6.0, F=32, l1=0.75, rs_over_l1=0.0, response_edge=0.75))()
times = {}
for name in ("M_16C2pm", "M_16C2pm_w", "M_16C2pm_wx", "M_16C2pm_ws", "M_16C2pm_us"):
    rung = E3.configure(E3.RUNGS[name], a)
    data = S.pad_batch(structs, n_freq=rung.K)
    params = M.init_params(jax.random.PRNGKey(0), rung)
    loss = T.make_loss(rung, 1e-3, w_energy=100.0, w_force=1.0)
    step = jax.jit(jax.value_and_grad(lambda p: loss(p, data)[0]))
    t0 = time.time(); jax.block_until_ready(step(params)); compile_s = time.time() - t0
    t0 = time.time()
    for _ in range(5):
        jax.block_until_ready(step(params))
    times[name] = (time.time() - t0) / 5
    print(f"{name:11s} compile {compile_s:5.1f} s, step {times[name] * 1000:6.0f} ms", flush=True)
for base, new in (("M_16C2pm", "_w"), ("M_16C2pm", "_wx"), ("M_16C2pm", "_ws"), ("M_16C2pm", "_us")):
    print(f"{base}{new} / {base}: {times[base + new] / times[base]:.2f}")
