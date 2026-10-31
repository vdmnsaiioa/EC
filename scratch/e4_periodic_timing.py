"""
The periodic path's cost on E4's cells (v0.0.20): E3's frozen model M_16C2pm_us in a cell -- E_0 at width 32, the site
model, the band field, the self-consistent response, the field term -- on one frame of each size, with every channel
switched on (the parameters are not trained: the cost does not depend on them).

    python scratch/e4_periodic_timing.py [--dir data/e4] [--sizes 64 128 256 512 1024] [--solvers direct gmres]

For every size and solver: the k-vectors, the time of the energy, of energy and forces, and of one training step (the
energy-and-force loss's gradient in the parameters, on one frame), each after its compilation, and the device's peak
memory.  A size that does not fit is reported as such and the next one runs.  The direct solve builds the dipole tensor
(O(N_k N^2)) and is skipped above --direct_max molecules.  It also checks, on the smallest size, that the two solves
agree.
"""
import argparse
import os
import sys
import time
import traceback
from dataclasses import replace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import jax                      # noqa: E402
import jax.numpy as jnp         # noqa: E402
import sseft                    # noqa: E402
from sseft import model as M, structure as S, periodic as PB, water as W  # noqa: E402
from sseft.units import ang_to_bohr as A2B  # noqa: E402

FILES = {64: "train_n64.npz", 128: "train_n128.npz", 256: "train_n256.npz", 512: "test_n512.npz", 1024: "test_n1024.npz"}


def frozen(F=32, r_cut=6.0, b_max=4.0):
    """M_16C2pm_us at E3-QP8's settings (configs/water_qaqua_pol8.yaml), in a cell."""
    e0 = {**M.RUNGS["M_1"].e0, "r_cut": A2B(r_cut), "F": F, "n_rbf_readout": 8}
    return M.with_rung(M.RUNGS["M_1"], name="M_16C2pm_us", pin_q=True, msites=True, dispersion=True, pin_alpha=True,
                       band_fields=2, bf_species=True, bf_mutual=-1, e0_field=True, e0_field_whole=True,
                       e0_field_on=A2B(r_cut - 1.0), e0_field_off=A2B(r_cut), e0_field_self_on=A2B(r_cut - 2.0),
                       e0_field_self_off=A2B(r_cut - 1.0), e0_field_cross_species=True, l1=A2B(0.75), bf_l=A2B(0.75),
                       bf_rs=0.0, e0=e0, periodic=True, ewald_b_max=b_max)


def switched_on(rung):
    p = M.init_params(jax.random.PRNGKey(0), rung)
    k = jax.random.split(jax.random.PRNGKey(1), 2)
    last = p["readout"]["theta"][-1]
    p["readout"]["theta"][-1] = {"w": 0.01 * jax.random.normal(k[0], last["w"].shape), "b": last["b"]}
    il = p["readout"]["iso"][-1]
    p["readout"]["iso"][-1] = {"w": jnp.zeros_like(il["w"]), "b": jnp.full_like(il["b"], 0.5)}
    fl = p["field"]["mlp"][-1]
    p["field"]["mlp"][-1] = {"w": 0.05 * jax.random.normal(k[1], fl["w"].shape), "b": fl["b"]}
    p["field_x"] = {"a": jnp.full_like(p["field_x"]["a"], 0.1), "c": jnp.full_like(p["field_x"]["c"], 0.05)}
    return p


def peak_gb():
    try:
        st = jax.devices()[0].memory_stats() or {}
        return st.get("peak_bytes_in_use", 0) / 1e9
    except Exception:
        return float("nan")


def timed(f, *args, repeat=2):
    t0 = time.time()
    out = f(*args); jax.block_until_ready(out)
    t1 = time.time()
    for _ in range(repeat):
        out = f(*args); jax.block_until_ready(out)
    return out, t1 - t0, (time.time() - t1) / repeat


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dir", default=os.path.join(REPO, "data", "e4"))
    ap.add_argument("--sizes", type=int, nargs="*", default=[64, 128, 256, 512, 1024])
    ap.add_argument("--solvers", nargs="*", default=["direct", "gmres"])
    ap.add_argument("--direct_max", type=int, default=512)
    ap.add_argument("--b_max", type=float, default=4.0)
    a = ap.parse_args()
    print(f"JAX {jax.__version__}, sseft {sseft.__version__}, devices {jax.devices()}")
    base = frozen(b_max=a.b_max)
    params = switched_on(base)
    rows, check = [], {}
    for n in a.sizes:
        path = os.path.join(a.dir, FILES[n])
        if not os.path.exists(path):
            print(f"{n} molecules: no {path}, skipped"); continue
        st = W.e4_structures(path, pins=True)[0]
        b = {k: v[0] for k, v in S.pad_batch([st], n_freq=8).items()}
        n_max = PB.n_max_for(np.asarray(b["cell"]), base.l1, a.b_max)
        n_k = (n_max + 1) * (2 * n_max + 1) ** 2
        for solver in a.solvers:
            if solver == "direct" and n > a.direct_max:
                continue
            rung = replace(base, ewald_n_max=n_max, bf_solver=solver)
            label = f"{n:5d} molecules, {solver:6s}"
            try:
                fE = jax.jit(lambda p, bb: M.energy_single(p, rung, bb)[0])
                E, cE, tE = timed(fE, params, b)
                fF = jax.jit(lambda p, bb: M.energy_forces_single(p, rung, bb))
                (E2, F), cF, tF = timed(fF, params, b)

                def loss(p, bb):
                    e, f = M.energy_forces_single(p, rung, bb)
                    return (e - bb["energy"]) ** 2 + jnp.sum((f - bb["forces"]) ** 2)
                fL = jax.jit(jax.grad(loss))
                _, cL, tL = timed(fL, params, b, repeat=1)
                mem = peak_gb()
                check[(n, solver)] = (float(E2), np.asarray(F))
                print(f"{label}: n_max {n_max} ({n_k} k on the grid), E {float(E):.8f}; energy {tE:.3f} s, energy+forces "
                      f"{tF:.3f} s, training step {tL:.3f} s (compiles {cE:.0f} / {cF:.0f} / {cL:.0f} s); peak {mem:.2f} GB",
                      flush=True)
                rows.append((n, solver, tE, tF, tL, mem))
            except Exception as e:                                          # out of memory and the like: report, go on
                msg = str(e).splitlines()[0] if str(e) else type(e).__name__
                print(f"{label}: failed -- {type(e).__name__}: {msg[:300]}", flush=True)
                if "RESOURCE_EXHAUSTED" not in str(e):
                    traceback.print_exc()
    for n in sorted({k[0] for k in check}):
        if (n, "direct") in check and (n, "gmres") in check:
            (Ed, Fd), (Eg, Fg) = check[(n, "direct")], check[(n, "gmres")]
            print(f"{n} molecules: gmres - direct: energy {Eg - Ed:+.2e} hartree, forces {np.abs(Fg - Fd).max():.2e} "
                  f"hartree/bohr (largest)")


if __name__ == "__main__":
    main()
