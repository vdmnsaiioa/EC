"""
The numbers of E4's X3 pre-registration (notes/e4-x3-preregistration.md), computed before any periodic training: the
truth's (MB-pol through MBX) and every prediction that the matching alone fixes.  In this model a uniform field reaches
only the read-out (E_0 and the field term never see it) and the read-out is matched on the open probes and then held,
so eps_inf, the far field and the Form I tail of contents whose images lie beyond r_c are set before training: they
are computed here, on E_0 at its initialisation, and training cannot move them.

    python scratch/e4_x3_preregistration.py --mbx external/mbx/build/MBX --data data/e4 \
        [--parts contents,form1,eps,probes,sweep] [--out scratch/e4_x3_preregistration.json]

contents  the Form I contents: a polar hexamer (E3's recipe on MB-pol: 300 Metropolis steps at 300 K), a non-polar
          tetramer (a relaxed dimer and its inversion image: total dipole zero by symmetry, permanent and induced), one
          molecule; their permanent and total (with MB-pol's induced) dipoles and largest atom-atom distance.
form1     the truth's Delta(L) = E(L) - E(open) in cubic cells of side L and in (L, L, 2L) cells (MBX; the contents' atoms
          all lie within 6.5 A of each other, so MBX's open and periodic dispersion agree inside them), and the same for
          the dagger M_16pm and the C rungs (the model's own open and periodic energies; no image within r_c).
eps       the clamped-nuclei eps_inf of the 64-molecule frames: MB-pol's (MBX's induced dipoles in a field) and the C
          rungs' (model.dielectric_inf), with the C rung's response truncated at first order beside it.
probes    the far-field test probes read on the dagger and the C rungs (E3's 10-15 A reading).
sweep     job 2's band-edge sweep, its matched part: l_1 = 0.75 and 1.0 A (band and read-out edges together) at read-out
          orders 0 (M_16C0pm_us) and 2 (M_16C2pm_us) -- eps_inf, the far field and the hexamer's Form I level, and
          their regulator dependence at each order.
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import jax                                        # noqa: E402
import sseft                                      # noqa: E402,F401
from sseft import model as M, water as W, structure as S, train as T   # noqa: E402
from sseft.units import ang_to_bohr, bohr_to_ang, KCALMOL_PER_HARTREE as KCAL  # noqa: E402
import e4_water as E4                             # noqa: E402

CUBIC = (15.0, 20.0, 25.0, 30.0, 40.0)
TETRAGONAL = (15.0, 20.0, 25.0)
DEBYE = 4.80320471257                             # e A -> D


def one(st, K=8):
    return {k: v[0] for k, v in S.pad_batch([st], n_freq=K).items()}


def dipole_perm(pos, site_q, gamma):
    n = len(pos) // 3; mu = np.zeros(3)
    for m in range(n):
        O, H1, H2 = pos[3 * m:3 * m + 3]
        Msite = O + 0.5 * gamma * (H1 + H2 - 2 * O)
        mu += site_q[0] * O + site_q[1] * H1 + site_q[2] * H2 + site_q[3] * Msite
    return mu


def contents(truth, seed=600):
    rng = np.random.default_rng(seed)
    geom = dict(T_K=300.0, radius_ang=None, min_OO=2.7, min_any=1.7)
    hexamer = W.random_cluster(6, rng, mc_steps=300, energy=truth.energy, **geom)
    dimer = W.random_cluster(2, rng, mc_steps=300, energy=truth.energy, **geom)
    dimer = dimer - dimer.mean(axis=0)
    side = np.repeat([0, 1], 6); across = side[:, None] != side[None, :]          # the dimer's atoms against its image's
    E_dimer = truth.energy(dimer, 2)
    found = []
    for trial in range(400):                    # the dimer and its inversion image, as close as the contacts allow
        D = dimer @ W.random_rotation(rng).T
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        for s in np.arange(0.5, 6.0, 0.05):
            A = D + s * u; pos = np.concatenate([A, -A])
            d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
            if d[0::3, 0::3][across[0::3, 0::3]].min() >= 2.7 and d[across].min() >= 1.7:
                found.append((d.max(), pos))
                break
    found.sort(key=lambda t: t[0])
    # the most compact of them whose two dimers attract each other (a bound tetramer, not a squeezed one)
    tetramer = next(pos for _, pos in found if truth.energy(pos, 4) < 2 * E_dimer)
    one_mol = W.monomer_positions() - W.monomer_positions().mean(axis=0)
    out = {}
    for name, pos in (("hexamer", hexamer), ("tetramer_ci", tetramer), ("monomer", one_mol)):
        pos = pos - pos.mean(axis=0)
        n = len(pos) // 3
        E = truth.energy(pos, n)
        mu_ind = truth.mb.induced_dipole(pos) if n > 1 else np.zeros(3)
        d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        out[name] = {"positions": pos, "n_mol": n, "energy_open": E,
                     "mu_perm_debye": (dipole_perm(pos, truth.site_charges, truth.gamma_m) * DEBYE).tolist(),
                     "mu_ind_debye": (np.asarray(mu_ind) * DEBYE).tolist(), "largest_distance": float(d.max())}
    return out


def cells():
    return [("cubic", L, np.diag([L, L, L])) for L in CUBIC] + [("tetragonal", L, np.diag([L, L, 2 * L])) for L in TETRAGONAL]


def form1_truth(truth, cont):
    out = {}
    for name, c in cont.items():
        rows = []
        for shape, L, cell in cells():
            pos = c["positions"] + 0.5 * np.diag(cell)
            t0 = time.time()
            E = truth.energy(pos, c["n_mol"], cell=cell)
            rows.append({"shape": shape, "L": L, "delta": E - c["energy_open"]})
            print(f"  truth {name} {shape} L {L:g}: Delta {(E - c['energy_open']) * KCAL:+.6f} kcal/mol  "
                  f"[{time.time() - t0:.0f} s]", flush=True)
        out[name] = rows
    return out


def model_form1(params, rung_cell, rung_open, cont, site_q, gamma):
    out = {}
    for name, c in cont.items():
        st = W.pin_rigid_site_model([W.water_structure(c["positions"], pinned_q=False)], site_q, gamma)[0]
        E_open = float(M.energy_single(params, rung_open, one(st))[0])
        rows = []
        for shape, L, cell in cells():
            sc = W.pin_rigid_site_model([W.water_structure(c["positions"] + 0.5 * np.diag(cell), pinned_q=False)],
                                        site_q, gamma)[0]
            sc.cell = ang_to_bohr(cell); sc.pbc = True
            r = E4.in_cell(rung_cell, sc.cell)
            gap = L - c["largest_distance"]                  # the nearest image of any atom is at least this far
            E = float(jax.jit(lambda p, b: M.energy_single(p, r, b)[0])(params, one(sc)))
            rows.append({"shape": shape, "L": L, "delta": E - E_open, "image_gap_angstrom": gap})
        out[name] = rows
    return out


def eps_frames(paths, K=8):
    out = []
    for p in paths:
        if os.path.exists(p):
            out += W.e4_structures(p, pins=True, K=K)
    return out


def eps_truth(truth, frames):
    vals = []
    for st in frames:
        pos, cell = bohr_to_ang(np.asarray(st.positions)), bohr_to_ang(np.asarray(st.cell))
        R = truth.mb.induced_response(pos, cell)
        vals.append(np.eye(3) + 4 * math.pi / abs(np.linalg.det(cell)) * R)
    return np.array(vals)


def eps_model(params, rung_cell, frames):
    r = E4.in_cell(rung_cell, frames[0].cell)
    f = jax.jit(lambda p, b: M.dielectric_inf(p, r, b))
    return np.array([np.asarray(f(params, one(st, r.K))) for st in frames])


def iso(e):
    return [float(np.trace(x) / 3) for x in e]


def model_params(name, cfg, train_cell, probes, scale, cache):
    rung_cell, rung_open = E4.rungs_for(name, cfg, train_cell)
    params = M.init_params(jax.random.PRNGKey(0), rung_open, energy_scale=scale)
    if rung_cell.band_fields >= 0:
        readout, match = E4.matched(name, rung_open, probes, scale, cfg, cache)
        params = {**params, "readout": readout}
    else:
        match = None
    return params, rung_cell, rung_open, match


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--mbx", required=True)
    ap.add_argument("--data", default=os.path.join(REPO, "data", "e4"))
    ap.add_argument("--probes", default=None)
    ap.add_argument("--cache", default=None, help="where the matched read-outs are kept (default: the data directory)")
    ap.add_argument("--parts", default="contents,form1,eps,probes,trimers,sweep")
    ap.add_argument("--eps_frames", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(REPO, "scratch", "e4_x3_preregistration.json"))
    a = ap.parse_args()
    parts = a.parts.split(",")
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    truth = W.MBPolTruth(a.mbx)
    cfg = E4.load_config(None)
    probes = E4.load_probes(a.probes or os.path.join(a.data, "probes_mbpol.npz"))
    train = W.e4_structures(os.path.join(a.data, "train_n64.npz"), pins=True)
    scale = T.energy_scale(train)
    cache = a.cache or a.data
    out.update(probes_hash=probes["hash"], energy_scale=scale, sseft=sseft.__version__, truth=truth.describe())
    print(f"probes hash {probes['hash']}; energy scale {scale * KCAL:.4f} kcal/mol per atom", flush=True)
    sq, gamma = probes["site_q"], probes["site_gamma"]
    t_all = time.time()

    cont = contents(truth)
    out["contents"] = {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
                       for k, v in cont.items()}
    for k, v in cont.items():
        print(f"{k}: {v['n_mol']} molecules, |mu_perm| {np.linalg.norm(v['mu_perm_debye']):.3f} D, |mu_ind| "
              f"{np.linalg.norm(v['mu_ind_debye']):.3f} D, largest distance {v['largest_distance']:.2f} A, E_open "
              f"{v['energy_open'] * KCAL:.4f} kcal/mol", flush=True)

    models = {"M_16pm": None, "M_16C2pm_us": None}
    for name in models:
        models[name] = model_params(name, cfg, train[0].cell, probes, scale, cache)

    if "form1" in parts:
        print("Form I, the truth:", flush=True)
        out["form1_truth"] = form1_truth(truth, cont)
        out["form1_model"] = {}
        for name, (params, rc, ro, _) in models.items():
            t0 = time.time()
            out["form1_model"][name] = model_form1(params, rc, ro, cont, sq, gamma)
            print(f"Form I, {name}: done [{time.time() - t0:.0f} s]", flush=True)
            for k, rows in out["form1_model"][name].items():
                tr = {(r["shape"], r["L"]): r["delta"] for r in out["form1_truth"][k]}
                print("  " + k + ": " + ", ".join(f"{r['shape'][0]}{r['L']:g} {r['delta'] * KCAL:+.5f} "
                                                   f"({r['delta'] / tr[(r['shape'], r['L'])]:.3f})" for r in rows), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)

    frames = eps_frames([os.path.join(a.data, f"{s}_n64.npz") for s in ("test", "valid", "train")])[:a.eps_frames]
    if "eps" in parts and frames:
        t0 = time.time()
        if out.get("eps_truth", {}).get("frames") == len(frames):
            et = np.array([np.eye(3) * v for v in out["eps_truth"]["isotropic"]])       # read before: the isotropic parts
        else:
            et = eps_truth(truth, frames)
            out["eps_truth"] = {"frames": len(frames), "isotropic": iso(et), "mean_tensor": et.mean(axis=0).tolist()}
        print(f"eps_inf, MB-pol, {len(frames)} frames of 64: {np.mean(iso(et)):.4f} +- {np.std(iso(et)):.4f}  "
              f"[{time.time() - t0:.0f} s]", flush=True)
        params, rc, ro, _ = models["M_16C2pm_us"]
        # the isolated monomer's polarisability, the model's (the matched read-out's response to a uniform field, open)
        # against MB-pol's (MBX: 1.4075 A^3 isotropic)
        mono = W.pin_rigid_site_model([W.water_structure(W.monomer_positions(), pinned_q=False)], sq, gamma)[0]
        Hm = np.asarray(jax.hessian(lambda ext: M.energy_single(params, ro, {**one(mono), "E_ext": ext})[0])(np.zeros(3)))
        a_mod = -Hm * bohr_to_ang(1.0) ** 3
        a_tru = truth.mb.induced_response(W.monomer_positions())
        out["alpha_monomer"] = {"model_angstrom3": a_mod.tolist(), "truth_angstrom3": a_tru.tolist(),
                                "model_eigenvalues": np.linalg.eigvalsh(0.5 * (a_mod + a_mod.T)).tolist(),
                                "truth_eigenvalues": np.linalg.eigvalsh(0.5 * (a_tru + a_tru.T)).tolist()}
        print(f"monomer polarisability: model {np.trace(a_mod) / 3:.4f} A^3 (eigenvalues "
              f"{np.round(np.linalg.eigvalsh(0.5 * (a_mod + a_mod.T)), 4)}), MB-pol {np.trace(a_tru) / 3:.4f} "
              f"({np.round(np.linalg.eigvalsh(0.5 * (a_tru + a_tru.T)), 4)})", flush=True)
        out["eps_model"] = {}
        for label, rung in (("M_16C2pm_us", rc), ("M_16C2pm_us_first_order", replace(rc, bf_mutual=1, bf_solver="direct"))):
            t0 = time.time()
            em = eps_model(params, rung, frames)
            out["eps_model"][label] = {"isotropic": iso(em), "mean_tensor": em.mean(axis=0).tolist(),
                                       "ratio_to_truth": [float(x / y) for x, y in zip(iso(em), iso(et))]}
            print(f"eps_inf, {label}: {np.mean(iso(em)):.4f} +- {np.std(iso(em)):.4f}; (eps - 1) / (eps* - 1) "
                  f"{np.mean([(x - 1) / (y - 1) for x, y in zip(iso(em), iso(et))]):.4f}  [{time.time() - t0:.0f} s]",
                  flush=True)
        json.dump(out, open(a.out, "w"), indent=1)

    if "probes" in parts:
        out["probes_model"] = {}
        for name, (params, rc, ro, match) in models.items():
            m = E4.read_probes(params, ro, probes)
            out["probes_model"][name] = m
            print(f"far-field probes, {name}: {E4.E3.probe_summary(m)}", flush=True)
        json.dump(out, open(a.out, "w"), indent=1)

    if "trimers" in parts and "trimers" in probes:
        out["trimers_truth"] = {"count": len(probes["trimers"]["three_body"]),
                                "three_body": probes["trimers"]["three_body"].tolist()}
        out["trimers_model"] = {}
        for name, (params, rc, ro, match) in models.items():
            v = E4.read_trimers(params, ro, probes)
            out["trimers_model"][name] = v
            print(f"far trimers (T5), {name}: three-body rmse {v['three_body_rmse'] * KCAL:.6f} of rms "
                  f"{v['three_body_rms'] * KCAL:.6f} kcal/mol, slope {v['slope']:.4f}", flush=True)
        json.dump(out, open(a.out, "w"), indent=1)

    if "sweep" in parts:
        out["sweep"] = {}
        hexa = {"hexamer": cont["hexamer"]}
        for edge in (0.75, 1.0):
            cfg_e = E4.merge(cfg, {"model": {"band_edge_angstrom": edge, "response_edge_angstrom": edge}})
            for name in ("M_16C0pm_us", "M_16C2pm_us"):
                key = f"{name}_l{edge:g}"
                t0 = time.time()
                rung_cell, rung_open = E4.rungs_for(name, cfg_e, train[0].cell)
                params = M.init_params(jax.random.PRNGKey(0), rung_open, energy_scale=scale)
                readout, match = E4.matched(f"{name}_l{edge:g}", rung_open, probes, scale, cfg_e, cache)
                params = {**params, "readout": readout}
                rec = {"matching": match}
                if frames:
                    em = eps_model(params, rung_cell, frames)
                    rec["eps"] = {"isotropic": iso(em)}
                rec["probes"] = E4.read_probes(params, rung_open, probes)
                if "trimers" in probes:
                    rec["trimers"] = E4.read_trimers(params, rung_open, probes)
                rec["form1_hexamer"] = model_form1(params, rung_cell, rung_open, hexa, sq, gamma)["hexamer"]
                out["sweep"][key] = rec
                print(f"sweep {key}: eps_inf {np.mean(rec['eps']['isotropic']) if frames else float('nan'):.4f}; "
                      f"probes {E4.E3.probe_summary(rec['probes'])}  [{time.time() - t0:.0f} s]", flush=True)
                json.dump(out, open(a.out, "w"), indent=1)
    print(f"done in {time.time() - t_all:.0f} s: {a.out}", flush=True)


if __name__ == "__main__":
    main()
