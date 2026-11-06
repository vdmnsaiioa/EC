"""
What a liquid calibration changes at contact, by read-out order, without training (notes/e4-job2-results.md sections
4b and 4f): on liquid frames, the model's energy per molecule and rigid forces with the far-field read-out (arm A),
with a calibrated one, and with only its order-0 columns (and node term) or only its orders >= 1 taken over. E_0 and
the field term at their initialisation are the same in every case and cancel in the differences; the read-out
switched off gives its own energy and forces.

    python scratch/e4_job2_calibration_orders.py --config local.yaml --far matched_..._72b7...pkl \\
        --calibrated matched_..._72b7..._liq49cad17a.pkl [--name M_16C2pm_us] [--edge 2.0] [--frames test]
"""
import argparse
import os
import pickle
import sys
from dataclasses import replace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import jax                                        # noqa: E402
import e4_water as E4                             # noqa: E402
from sseft import model as M, water as W, train as T, structure as S   # noqa: E402
from sseft.units import bohr_to_ang               # noqa: E402

N_RBF = 8                                         # the read-out's radial columns per invariant (E3's n_rbf_readout)


def taken(base, other, order0):
    """base with other's order-0 columns and node term (order0=True), or with other's orders >= 1 columns."""
    n0 = 2 * N_RBF
    w, b = np.array(base["theta"][-1]["w"]), np.array(base["theta"][-1]["b"])
    wo, bo = np.array(other["theta"][-1]["w"]), np.array(other["theta"][-1]["b"])
    sl = slice(0, n0) if order0 else slice(n0, None)
    w[:, sl], b[sl] = wo[:, sl], bo[sl]
    out = {**base, "theta": list(base["theta"])}
    out["theta"][-1] = {"w": w, "b": b}
    if order0:
        out["iso"] = list(other["iso"])
    return out


def switched_off(ro):
    out = {**ro, "theta": list(ro["theta"]), "iso": list(ro["iso"])}
    out["theta"][-1] = {k: np.zeros_like(np.array(v)) for k, v in ro["theta"][-1].items()}
    out["iso"][-1] = {k: np.zeros_like(np.array(v)) for k, v in ro["iso"][-1].items()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", required=True)
    ap.add_argument("--far", required=True, help="the far-field matched read-out (E4.matched's pickle)")
    ap.add_argument("--calibrated", required=True, help="a calibrated read-out (E4.matched's pickle, _liq<tag>)")
    ap.add_argument("--name", default="M_16C2pm_us")
    ap.add_argument("--edge", type=float, default=2.0)
    ap.add_argument("--frames", default="test", help="the data set of the training size read (its every frame)")
    a = ap.parse_args()
    cfg = E4.merge(E4.load_config(a.config), {"model": {"mutual_edge_angstrom": a.edge}})
    d, L0 = cfg["data"]["dir"], cfg["data"]["train_size"]
    train = W.e4_structures(os.path.join(d, f"train_n{L0}.npz"), pins=True)
    rc, ro = E4.rungs_for(a.name, cfg, train[0].cell)
    far, cal = (pickle.load(open(p, "rb"))["readout"] for p in (a.far, a.calibrated))
    sets = {"off": switched_off(far), "A": far, "calibrated": cal, "its order 0 alone": taken(far, cal, True),
            "its orders >= 1 alone": taken(far, cal, False)}
    p0 = M.init_params(jax.random.PRNGKey(0), ro, energy_scale=T.energy_scale(train))
    r = E4.in_cell(replace(rc, bf_solver="direct"), train[0].cell)
    f = jax.jit(lambda p, b: M.energy_forces_single(p, r, b))
    frames = W.e4_structures(os.path.join(d, f"{a.frames}_n{L0}.npz"), pins=True)
    n_mol = frames[0].n_atoms // 3
    res = {k: [] for k in sets}
    for st in frames:
        b = {k: v[0] for k, v in S.pad_batch([st], n_freq=r.K).items()}
        for k, rk in sets.items():
            E, Fo = f(E4.with_readout(p0, rk), b)
            res[k].append((float(E), np.asarray(Fo)[:st.n_atoms]))

    def change(k, ref):
        dE, ss, cc = [], 0.0, 0.0
        for (E0, F0), (Ek, Fk), st in zip(res[ref], res[k], frames):
            dE.append((Ek - E0) / n_mol * E4.KCAL)
            _, _, s_, c_ = E4.rigid_errors((Fk - F0) * E4.FORCE_UNIT, bohr_to_ang(np.asarray(st.positions)),
                                           bohr_to_ang(np.asarray(st.cell)), n_mol)
            ss += s_; cc += c_
        return float(np.mean(dE)), float(np.sqrt(ss / cc))
    print(f"{a.name} at {a.edge:g} A, {len(frames)} {a.frames} frames of {n_mol} molecules", flush=True)
    for k in ("A", "calibrated"):
        e, fr = change(k, "off")
        print(f"  the read-out {k}: its own energy {e:+.3f} kcal/mol per molecule, rigid forces {fr:.3f} kcal/mol/A rms")
    for k in ("calibrated", "its order 0 alone", "its orders >= 1 alone"):
        e, fr = change(k, "A")
        print(f"  {k} against A: energy {e:+.3f} kcal/mol per molecule, rigid forces changed by {fr:.3f} kcal/mol/A rms")


if __name__ == "__main__":
    main()
