"""
The k-dependence of the cells' susceptibility, read-out by read-out and edge by edge (notes/e4-job3-results.md
section 4, a reading beside job 3's bands): chi(k) / chi(0+) - 1 at harmonics of the cell on held-out liquid frames,
for matched read-outs of the frozen form at chosen mutual edges, with and without the orders >= 1, where
chi(0+) = (1 - 1/eps_inf) / 4 pi is the read-out's own screened limit (its eps_inf on the same frames).  With --mbx,
MB-pol's at the same harmonics on the same frames (MBX, as e4_water.liquid_response_k computes it).  No training.

    python scratch/e4_job3_kdep.py --config CFG --variant LABEL:RUNG:EDGE:PKL [...] [--harmonics 1,2,3,4]
                                   [--frames 1] [--mbx PATH]

A variant's PKL is a matched read-out as e4_water.matched() caches it ({"rung", "scale", "readout", ...}); its rung is
rebuilt from RUNG and the mutual edge EDGE (A) with CFG's model section and checked against the cached one.  RUNG0
reads RUNG's read-out with its orders >= 1 off (e4_water.order_zero).
"""
import argparse
import copy
import math
import os
import pickle
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import jax                                        # noqa: E402
from dataclasses import replace                   # noqa: E402
from sseft import model as M, water as W, structure as S, modes as MD   # noqa: E402
from sseft.units import bohr_to_ang                # noqa: E402
import e4_water as E4                             # noqa: E402


def fit(ks, qs, orders):
    """least squares q = sum_p c_p k^(2p) over p in orders; exact when len(ks) == len(orders)."""
    A = np.array([[k ** (2 * p) for p in orders] for k in ks])
    c, *_ = np.linalg.lstsq(A, np.array(qs), rcond=None)
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", action="append", default=[], help="LABEL:RUNG:EDGE:PKL (RUNG0: orders >= 1 off)")
    ap.add_argument("--harmonics", default="1,2,3,4")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--mbx", default=None, help="MBX's root: MB-pol's chi(k) at the same harmonics and frames")
    a = ap.parse_args()
    cfg = E4.load_config(a.config)
    dcfg = cfg["data"]
    response = E4.load_response(dcfg["response"], cfg)
    h = response[a.split]
    nf = min(a.frames, len(h["structures"]))
    harmonics = [int(x) for x in a.harmonics.split(",")]
    modes = [(ax, n, ph) for n in harmonics for ax in (0, 1, 2) for ph in (0, 1)]
    st0 = h["structures"][0]
    ks = {n: float(np.linalg.norm(2 * np.pi * n * np.linalg.inv(np.asarray(st0.cell)).T[0])) / bohr_to_ang(1.0)
          for n in harmonics}                                                        # k along axis 0, 1/A
    print(f"{a.split} frames {nf} of {len(h['structures'])}; harmonics {harmonics}: k "
          + ", ".join(f"{ks[n]:.4f}" for n in harmonics) + " /A", flush=True)
    rows = []
    if a.mbx:
        truth = W.MBPolTruth(a.mbx)
        idx = h["index"][:nf]
        fpath = response["meta"]["splits"][a.split]["file"]
        d = np.load(fpath)
        chis, eps = {n: [] for n in harmonics}, []
        t0 = time.time()
        for j, i in enumerate(idx):
            pos = d["positions"][i]
            cell = d["cell"] if d["cell"].ndim == 2 else d["cell"][i]
            V = abs(np.linalg.det(cell))
            _, chi = E4.liquid_response_k(truth.mb, pos, cell, modes, float(cfg["liquid_k"]["delta"]))
            for n in harmonics:
                chis[n].append(sum(c for c, md in zip(chi, modes) if md[1] == n) / (3 * V))
            print(f"  MB-pol frame {i}  [{time.time() - t0:.0f} s]", flush=True)
        rows.append(("MB-pol", None, {n: float(np.mean(chis[n])) for n in harmonics}))
    eps_truth = None
    for spec in a.variant:
        label, rung_name, edge, pkl = spec.split(":", 3)
        zero = rung_name.endswith("0") and rung_name not in E4.E3.RUNGS
        name = rung_name[:-1] if zero else rung_name
        with open(pkl, "rb") as fh:
            cached = pickle.load(fh)
        cfg_v = copy.deepcopy(cfg)
        cfg_v["model"]["mutual_edge_angstrom"] = float(edge) if float(edge) > 0 else None
        rung_cell, rung_open = E4.rungs_for(name, cfg_v, st0.cell)
        same = repr(rung_open) == cached["rung"]
        readout = cached["readout"]
        if zero:
            readout = E4.order_zero(readout, 2 * int(rung_open.e0.get("n_rbf_readout", rung_open.e0["n_rbf"])))
        params = E4.with_readout(M.init_params(jax.random.PRNGKey(0), rung_open, energy_scale=cached["scale"]), readout)
        t0 = time.time()
        sub = {**response, a.split: {k: (v[:nf] if k in ("structures", "mu", "R", "index") else v) for k, v in h.items()}}
        v = E4.read_liquid(params, rung_cell, sub, a.split)
        eps_truth = v["eps_inf_truth"]
        r = E4.in_cell(replace(rung_cell, bf_solver="direct"), st0.cell)
        f = jax.jit(lambda p, b, u, g, hh: M.mode_response(p, r, b, u, g, hh))
        chis = {n: [] for n in harmonics}
        for st in h["structures"][:nf]:
            b = {k: vv[0] for k, vv in S.pad_batch([st], n_freq=r.K).items()}
            pos, cell = np.asarray(b["positions"]), np.asarray(st.cell)
            mask = np.asarray(b["mask"], float)[:, None]
            V = abs(np.linalg.det(cell))
            acc = {n: 0.0 for n in harmonics}
            for axis, n, phase in modes:
                u, g, hh = MD.mode_field(pos, cell, axis, n, phase)
                _, c = f(params, b, u * mask, g * mask[..., None], hh * mask[..., None, None])
                acc[n] += float(c) / (3 * V)
            for n in harmonics:
                chis[n].append(acc[n])
        rows.append((label, v["eps_inf_model"], {n: float(np.mean(chis[n])) for n in harmonics}))
        print(f"  {label}: rung {name} at the mutual edge {edge} A{' (orders >= 1 off)' if zero else ''}, the cached "
              f"rung {'the same' if same else 'DIFFERENT'}; eps_inf {v['eps_inf_model']:.4f}  [{time.time() - t0:.0f} s]",
              flush=True)
    print()
    for label, eps, chi in rows:
        if eps is None:
            eps = eps_truth
        if eps is None:
            print(f"{label}: no eps_inf to read chi(0+) against"); continue
        c0 = (1 - 1 / eps) / (4 * math.pi)
        q = {n: chi[n] / c0 - 1 for n in harmonics}
        kk = [ks[n] for n in harmonics]
        line = "  ".join(f"k{n} {100 * q[n]:+7.2f} %" for n in harmonics)
        c2 = fit([ks[1], ks[2]], [q[1], q[2]], (1, 2)) if 1 in q and 2 in q else None
        extra = ""
        if c2 is not None:
            extra = f"; from k1, k2: c2 {c2[0]:.4f} A^2, c4 {c2[1]:+.4f} A^4"
            for n in harmonics:
                if n > 2:
                    extra += f"; k{n} predicted {100 * (c2[0] * ks[n] ** 2 + c2[1] * ks[n] ** 4):+.1f} %"
        if len(harmonics) >= 3:
            c3 = fit(kk, [q[n] for n in harmonics], (1, 2, 3))
            extra += f"; fit c2 c4 c6 over all: {c3[0]:.4f}, {c3[1]:+.4f}, {c3[2]:+.5f}"
        print(f"{label:24s} eps_inf {eps:.4f}  chi(k)/chi(0+) - 1: {line}{extra}")


if __name__ == "__main__":
    main()
