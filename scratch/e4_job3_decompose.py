"""
Where a read-out's finite-k susceptibility comes from (notes/e4-job3-preregistration.md section 3b): on one liquid cell,
per harmonic, the cells' chi(k) as the read-out's energy gives it (model.mode_response: chi_mode = -d2E/da2) against the
dipoles' share, sum_i u_i . dmu_i/da (the atoms' induced dipoles in the mode's field, all that MB-pol's point dipoles
have), for the read-out and for its order-0 sector alone (e4_water.order_zero), against MB-pol's.  No training.

    python scratch/e4_job3_decompose.py --config configs/e4_job3.yaml --arm C [--split heldout] [--frame 0]
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import jax                                        # noqa: E402
from dataclasses import replace                   # noqa: E402
from sseft import model as M, water as W, train as T, structure as S, modes as MD   # noqa: E402
import e4_water as E4                             # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", required=True, help="job 3's config (kread.arms: the arms' configs and caches)")
    ap.add_argument("--arm", default="C")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--frame", type=int, default=0)
    a = ap.parse_args()
    cfg = E4.load_config(a.config)
    kc, dcfg = cfg["kread"], cfg["data"]
    K, om = int(cfg["model"]["K"]), float(cfg["model"]["omega0"])
    train = W.e4_structures(os.path.join(dcfg["dir"], f"train_n{dcfg['train_size']}.npz"), pins=True, K=K, omega0=om)
    scale = T.energy_scale(train)
    probes = E4.load_probes(dcfg["probes"], K, om)
    response = E4.load_response(dcfg["response"], cfg)
    rk = E4.load_response_k(dcfg["response_k"], cfg, response)
    spec = kc["arms"][a.arm]
    cfg_arm = E4.load_config(spec["config"])
    rung_cell, rung_open = E4.rungs_for(kc["rung"], cfg_arm, train[0].cell)
    lq = cfg_arm["training"]["readout"].get("liquid")
    liquid = E4.liquid_block(rung_cell, response, lq) if lq else None
    readout, _ = E4.matched(kc["rung"], rung_open, probes, scale, cfg_arm, spec["out_dir"], liquid=liquid)
    st = rk[a.split]["structures"][a.frame]
    r = E4.in_cell(replace(rung_cell, bf_solver="direct"), st.cell)
    b = {k: v[0] for k, v in S.pad_batch([st], n_freq=r.K).items()}
    pos, cell = np.asarray(b["positions"]), np.asarray(st.cell)
    V = abs(np.linalg.det(cell))
    f = jax.jit(lambda p, bb, u, g, h: M.mode_response(p, r, bb, u, g, h, per_atom=True))
    n0 = 2 * int(rung_open.e0.get("n_rbf_readout", rung_open.e0["n_rbf"]))
    axes = sorted({m[0] for m in rk["modes"]})
    for label, ro in ((a.arm, readout), (f"{a.arm}0", E4.order_zero(readout, n0))):
        params = E4.with_readout(M.init_params(jax.random.PRNGKey(0), rung_open, energy_scale=scale), ro)
        for n in sorted({m[1] for m in rk["modes"]}):
            chi_e = chi_d = chi_t = 0.0
            for j, (axis, nn, ph) in enumerate(rk["modes"]):
                if nn != n:
                    continue
                u, g, h = MD.mode_field(pos, cell, axis, n, ph)
                dmu, c = f(params, b, u, g, h)
                chi_e += float(c) / (len(axes) * V)
                chi_d += float(np.sum(u * np.asarray(dmu))) / (len(axes) * V)
                chi_t += float(rk[a.split]["chi"][a.frame][j]) / (len(axes) * V)
            print(f"{label} harmonic {n}: chi {chi_e:.5f} ({100 * (chi_e / chi_t - 1):+.2f} %), the dipoles' share "
                  f"{chi_d:.5f} ({100 * (chi_d / chi_t - 1):+.2f} %), MB-pol's {chi_t:.5f}", flush=True)


if __name__ == "__main__":
    main()
