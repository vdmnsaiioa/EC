"""
e4_far.py -- MB-pol's far-field test structures (the probe file of scripts/e4_water.py) for any model, NumPy only, so
that it runs in the baselines' environment too (baselines/mace, without JAX).

    python scripts/e4_far.py extxyz data/e4/probes_mbpol.npz             # -> data/e4/probes_mbpol_far.extxyz
    python scripts/e4_far.py read data/e4/probes_mbpol.npz energies.npy   # a model's energies (eV), in the extxyz's order

The extxyz holds, open (pbc F F F), in this order:
  - every test probe: the configuration (kind=probe), its core (kind=core) and its dimers, core molecule i with the probe
    (kind=pair);
  - every far trimer (kind=trimer) and its three dimers (kind=trimer_pair), when the file has them.
REF_energy (eV, interaction energy, the isolated monomer at zero) is given where the file has the truth's: the probe,
its core, the trimer and its dimers.  `read` turns a model's energies of these frames into E3's probe reading (the
probe's interaction E - E_core and its non-additive part, less the probe's dimers, by distance) and T5 (the trimers'
three-body energy against the truth's: rmse, rms and slope), the same numbers scripts/e4_water.py reads for the
harness's rungs.  A model whose sources do not answer a far field -- MACE, MACE-LES (Theorem 2), the dagger -- gives
T5 exactly zero.

    python scripts/e4_far.py cells data/e4/test_n128.npz mace_test_n128.npz [--json out.json]

The bulk readings of any model on an E4 data set, as the driver reads the harness's rungs: energy per molecule (rmse,
mean, per frame) and the forces' rigid-body part (each molecule's net force and torque, sseft.train.rigid_force_sums),
with the atomic forces beside them.  The model file holds energy (F,) in eV -- the interaction energy, REF_energy's
convention -- and forces (F, 3n, 3) in eV/A, in the data set's frame order; from MACE's evaluated extxyz, e.g.
    fr = ase.io.read("test_n128_mace.extxyz", ":")
    np.savez("mace_test_n128.npz", energy=[a.info["MACE_energy"] for a in fr], forces=[a.arrays["MACE_forces"] for a in fr])
"""
import argparse
import json
import os
import sys

import numpy as np

EV_PER_HARTREE = 27.211386245988
KCAL_PER_HARTREE = 627.5094740631
BOHR_PER_ANGSTROM = 1.8897259886
PAIRS = ((0, 1), (0, 2), (1, 2))


def frames(path):
    """[(kind, positions (A), REF_energy (hartree) or None, tags)] in the extxyz's order."""
    d = np.load(path, allow_pickle=False)
    out = []
    off = d["test_offsets"]
    for k in range(len(off) - 1):
        pos = d["test_positions"][off[k]:off[k + 1]]
        n = len(pos) // 3
        tags = {"k": k, "d": float(d["test_d"][k])}
        out.append(("probe", pos, float(d["test_energy"][k]), tags))
        out.append(("core", pos[:-3], float(d["test_E_core"][k]), tags))
        for i in range(n - 1):
            out.append(("pair", np.concatenate([pos[3 * i:3 * i + 3], pos[-3:]]), None, {**tags, "i": i}))
    if "trimer_positions" in d.files:
        for t, pos in enumerate(d["trimer_positions"]):
            pairs = d["trimer_pairs"][t]
            out.append(("trimer", pos, float(pairs.sum() + d["trimer_three_body"][t]), {"t": t}))
            for (i, j), e in zip(PAIRS, pairs):
                out.append(("trimer_pair", np.concatenate([pos[3 * i:3 * i + 3], pos[3 * j:3 * j + 3]]), float(e),
                            {"t": t, "ij": f"{i}{j}"}))
    return out


def write_extxyz(path, out_path=None):
    out_path = out_path or os.path.splitext(path)[0] + "_far.extxyz"
    with open(out_path, "w") as fh:
        for kind, pos, E, tags in frames(path):
            sym = np.tile(["O", "H", "H"], len(pos) // 3)
            extra = "".join(f" {k}={v}" for k, v in tags.items())
            ref = f" REF_energy={E * EV_PER_HARTREE:.12f}" if E is not None else ""
            fh.write(f"{len(pos)}\nProperties=species:S:1:pos:R:3 pbc=\"F F F\" config_type={kind}{ref}{extra}\n")
            for s, (x, y, z) in zip(sym, pos):
                fh.write(f"{s} {x:.10f} {y:.10f} {z:.10f}\n")
    return out_path


def readings(path, energies_ev):
    """E3's probe reading and T5 from a model's energies (eV) of frames(path), in their order (hartree inside)."""
    d = np.load(path, allow_pickle=False)
    fr = frames(path)
    E = np.asarray(energies_ev, float) / EV_PER_HARTREE
    if len(E) != len(fr):
        raise ValueError(f"{len(E)} energies for {len(fr)} frames")
    n_probe = len(d["test_offsets"]) - 1
    inter, pairs = np.zeros(n_probe), np.zeros(n_probe)
    core = np.zeros(n_probe)
    tri, tri_pairs = {}, {}
    for (kind, _, _, tags), e in zip(fr, E):
        if kind == "probe":
            inter[tags["k"]] += e
        elif kind == "core":
            core[tags["k"]] += e
        elif kind == "pair":
            pairs[tags["k"]] += e
        elif kind == "trimer":
            tri[tags["t"]] = e
        else:
            tri_pairs[tags["t"]] = tri_pairs.get(tags["t"], 0.0) + e
    inter = inter - core
    ref_inter = d["test_energy"] - d["test_E_core"]
    ref_nadd = ref_inter - d["test_pair_energy"]
    nadd = inter - pairs
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))

    def block(m):
        return {"count": int(m.sum()), "interaction_rmse": rms(inter[m] - ref_inter[m]), "interaction_rms": rms(ref_inter[m]),
                "nonadditive_rmse": rms(nadd[m] - ref_nadd[m]), "nonadditive_rms": rms(ref_nadd[m])}
    dist = d["test_d"].astype(float)
    out = {"probes": {**block(np.ones(n_probe, bool)),
                      "by_distance_angstrom": {f"{x:g}": block(dist == x) for x in np.unique(dist)}}}
    if tri:
        t = sorted(tri)
        model = np.array([tri[i] - tri_pairs[i] for i in t])
        ref = d["trimer_three_body"][t]
        out["far_trimers"] = {"count": len(t), "three_body_rmse": rms(model - ref), "three_body_rms": rms(ref),
                              "slope": float(np.dot(model, ref) / np.dot(ref, ref)), "model": model.tolist()}
    return out


def rigid_errors(err, pos, cell):
    """one frame of whole water molecules (atoms O H H in order): each molecule's net force error, its torque error about
    the centroid, and the rigid-body projection's sum of squares and degrees of freedom -- sseft.train.rigid_force_sums
    and scripts/e4_water.py's rigid_errors, in NumPy.  err (3n, 3) kcal/mol/A, pos (3n, 3) A, cell (3, 3) A."""
    n = len(pos) // 3
    d = pos.reshape(n, 3, 3) - pos.reshape(n, 3, 3)[:, :1]
    d = d - np.round(d @ np.linalg.inv(cell)) @ cell
    d = d - d.mean(axis=1, keepdims=True)
    e = err.reshape(n, 3, 3)
    net, tau = e.sum(axis=1), np.cross(d, e).sum(axis=1)
    inertia = np.sum(d * d, axis=(1, 2))[:, None, None] * np.eye(3) - np.einsum("mai,maj->mij", d, d)
    lam, vec = np.linalg.eigh(inertia)
    on = lam > 1e-8 * np.maximum(lam.max(axis=1, keepdims=True), 1e-300)
    proj = np.einsum("mjk,mj->mk", vec, tau)
    rot = np.sum(np.where(on, proj ** 2 / np.where(on, lam, 1.0), 0.0), axis=1)
    return net, tau, float(np.sum(np.sum(net ** 2, axis=1) / 3.0 + rot)), float(np.sum(3 + on.sum(axis=1)))


def cell_readings(data_path, energies_ev, forces_ev_ang):
    """a model's bulk readings on an E4 data set (scripts/e4_data.py's frames, in their order), the numbers
    scripts/e4_water.py reads for the harness's rungs: energy per molecule rmse, mean and per-frame errors (kcal/mol),
    rigid-body force rmse, net force and torque rmse, atomic force rmse (kcal/mol/A).  The model's energies are the
    interaction energy (the isolated monomer at zero, as REF_energy in the data set's extxyz) in eV, its forces in eV/A."""
    d = np.load(data_path, allow_pickle=False)
    pos, E_ref, F_ref = d["positions"], d["energy"], d["forces"]
    cells = d["cell"] if d["cell"].ndim == 3 else np.broadcast_to(d["cell"], (len(pos), 3, 3))
    E = np.asarray(energies_ev, float).reshape(-1)
    F = np.asarray(forces_ev_ang, float).reshape(pos.shape)
    if len(E) != len(pos):
        raise ValueError(f"{len(E)} energies for {len(pos)} frames")
    n_mol = pos.shape[1] // 3
    dE = (E / EV_PER_HARTREE - E_ref) * KCAL_PER_HARTREE / n_mol
    net, tau, ss, cc, fa = [], [], 0.0, 0.0, []
    for f in range(len(pos)):
        err = F[f] * KCAL_PER_HARTREE / EV_PER_HARTREE - F_ref[f] * KCAL_PER_HARTREE * BOHR_PER_ANGSTROM
        fa.append(np.mean(err ** 2))
        n_, t_, s_, c_ = rigid_errors(err, pos[f], np.asarray(cells[f], float))
        net.append(n_); tau.append(t_); ss += s_; cc += c_
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    return {"frames": int(len(pos)), "n_mol": int(n_mol), "energy_per_molecule_rmse": rms(dE),
            "energy_per_molecule_mean_error": float(np.mean(dE)), "energy_per_molecule_errors": dE.tolist(),
            "force_rigid_rmse": float(np.sqrt(ss / cc)), "net_force_rmse": rms(np.concatenate(net)),
            "torque_rmse": rms(np.concatenate(tau)), "force_atomic_rmse": float(np.sqrt(np.mean(fa)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("extxyz", help="the far-field test structures as extxyz")
    x.add_argument("probes"); x.add_argument("--out", default=None)
    r = sub.add_parser("read", help="a model's probe reading and T5 from its energies of the extxyz's frames")
    r.add_argument("probes"); r.add_argument("energies", help=".npy (eV), one per frame in the extxyz's order")
    r.add_argument("--json", default=None)
    c = sub.add_parser("cells", help="a model's bulk readings on an E4 data set (energy per molecule, rigid forces)")
    c.add_argument("data", help="an E4 data set, e.g. data/e4/test_n128.npz")
    c.add_argument("model", help=".npz with energy (F,) in eV and forces (F, 3n, 3) in eV/A, in the data set's order")
    c.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cmd == "extxyz":
        print(write_extxyz(a.probes, a.out))
        return
    if a.cmd == "cells":
        m = np.load(a.model, allow_pickle=False)
        v = cell_readings(a.data, m["energy"], m["forces"])
        print(f"{a.data} ({v['frames']} frames of {v['n_mol']} molecules): energy per molecule rmse "
              f"{v['energy_per_molecule_rmse']:.4f} kcal/mol (mean {v['energy_per_molecule_mean_error']:+.4f}), rigid "
              f"forces {v['force_rigid_rmse']:.3f} kcal/mol/A (net {v['net_force_rmse']:.3f}, torque {v['torque_rmse']:.3f} "
              f"kcal/mol; atomic {v['force_atomic_rmse']:.3f})")
        if a.json:
            with open(a.json, "w") as fh:
                json.dump(v, fh, indent=1)
        return
    out = readings(a.probes, np.load(a.energies))
    pct = lambda b, k: 100 * b[f"{k}_rmse"] / b[f"{k}_rms"]
    print("far-field probes, non-additive: " + ", ".join(
        f"{dd} A {pct(b, 'nonadditive'):.1f} %" for dd, b in out["probes"]["by_distance_angstrom"].items()))
    if "far_trimers" in out:
        v = out["far_trimers"]
        print(f"far trimers (T5): three-body rmse {v['three_body_rmse'] * KCAL_PER_HARTREE:.6f} of rms "
              f"{v['three_body_rms'] * KCAL_PER_HARTREE:.6f} kcal/mol, slope {v['slope']:.4f}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
