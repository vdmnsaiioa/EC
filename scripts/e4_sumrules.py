"""
E4's sum-rule test S0 (notes/e4-s0-preregistration.md; the project's path-integral-ir-sector.md section 4.1): the
longitudinal and transverse polarisation fluctuations of MB-pol's liquid on E4's own trajectories, and how a candidate
polarisation regresses on MB-pol's, read against the registered bands. No training and no model MD. Three steps:

    python scripts/e4_sumrules.py mbpol --mbx external/mbx/build/MBX [--sizes 64 128 256 512 1024] [--part i/n]
    python scripts/e4_sumrules.py model --config configs/e4_job2c.yaml --readout <matched pickle> [--label C]
    python scripts/e4_sumrules.py read [--json results/e4_s0/s0.json]

mbpol: MB-pol's electrostatic sites at zero external field on every frame of data/e4's sets of a size (train, valid and
  test together: 240, 120, 60, 60 and 52 frames), with the truth's settings (sseft.mbpol.DEFAULTS): the site positions,
  charges and converged induced dipoles, O, H, H, M per molecule, into <out>/mbpol_n<n>[_p<i>of<k>].npz. The size's
  first frame is solved again with the dynamics' settings (scripts/e4_data.py DYNAMICS) and the induced dipoles'
  relative rms difference recorded. MBX only; no GPU.
model: a frozen form's own polarisation on the 64-256 sets: the pinned site charges and, at every atom, the induced
  dipole -dE/dE_ext,i at zero external field -- the order-0 sector's mu_ind with the orders >= 1's field-gradient terms,
  as model.mode_response's dipoles -- and the order-0 mu_ind alone, into <out>/model_<label>_n<n>.npz. Both from one
  pass, with the mutual response solved as the config's model is (GMRES; --solver direct for the dense solve, whose
  cost grows as N^3). The read-out must have been matched for the rung with this config's model; the reciprocal grid
  (ewald_n_max) is the cell's, so it is left out of that check.
read: on the (100), (110) and (111) shells of every cell,
  L(k) = (4 pi beta / V) <|P_L,k|^2>, P_L from the charge density of the site charges and point dipoles,
  T(k) = (4 pi beta / V) <|e.P_T,k|^2> per transverse direction, P from the molecules' dipoles at their centres of mass,
  the regression slopes of MB-pol's P on a candidate Y, b_L = Re<P_L Y_L*> / <|Y_L|^2> and b_T the same over both
  transverse directions, and r = b_T / b_L,
  for the candidates "perm" (MB-pol's site charges without their induced dipoles: the dagger's sources) and every model
  file; errors by jackknife over blocks of consecutive frames within a trajectory. Then S0.1-S0.4 and the readings
  beside them. NumPy only.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from sseft.units import KCALMOL_PER_HARTREE, BOHR_PER_ANGSTROM  # noqa: E402

T_K = 300.0                                     # the dynamics' thermostat (scripts/e4_data.py)
KB_KCAL = 0.0019872043                          # kcal/mol/K
COULOMB = KCALMOL_PER_HARTREE / BOHR_PER_ANGSTROM   # e^2 / (4 pi eps0 A) in kcal/mol: 332.06
BETA_E2 = COULOMB / (KB_KCAL * T_K)             # e^2 / (A k_B T) at 300 K: 557.0
EPS_INF, C2 = 1.6885, 0.274                     # MB-pol's eps_inf (job 2) and its chi(k) = chi(0+)(1 + c2 k^2), A^2 (job 3)
MASSES = np.array([15.999, 1.008, 1.008])       # O, H, H
SPLITS = ("train", "valid", "test")
SIZES = (64, 128, 256, 512, 1024)
SHELLS = {"100": [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
          "110": [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)],
          "111": [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]}     # one of each +-k pair: rho_-k = rho_k*
BLOCK = 4                                       # frames a block: 2 ps at 0.5 ps a frame, against tau_L ~ 0.5 ps
K_FIT = 0.46                                    # S0.1: the shells with k <= 0.46 /A of the 128-1024 cells
BANDS = {"S0.1": (0.48, 0.68), "S0.2": (20.0, np.inf), "S0.3": (1.45, 1.95), "S0.4": (0.90, 1.10)}
S04_MODEL = "C"                                 # S0.4 reads arm C's read-out (model_C_n<n>.npz)


# ---------------------------------------------------------------- the frames ----------------------------------------
def load_frames(data_dir, n):
    """the frames of every set of n molecules in data_dir, train, valid and test in that order: positions (F, 3n, 3) A,
    cell (3, 3) A, source (F,) the trajectory's seed, time_ps (F,), split (F,), site_q (4,) and site_gamma; None if no
    set of that size exists."""
    parts = []
    for split in SPLITS:
        p = os.path.join(data_dir, f"{split}_n{n}.npz")
        if os.path.exists(p):
            parts.append((split, dict(np.load(p, allow_pickle=False))))
    if not parts:
        return None
    d0 = parts[0][1]
    cell = d0["cell"] if d0["cell"].ndim == 2 else d0["cell"][0]
    for split, d in parts:
        c = d["cell"] if d["cell"].ndim == 2 else d["cell"][0]
        if not np.allclose(c, cell) or not np.allclose(d["site_q"], d0["site_q"]):
            raise ValueError(f"{split}_n{n}: another cell or site model than {parts[0][0]}_n{n}")
    cat = lambda k, fill: np.concatenate([d[k] if k in d else fill(len(d["positions"])) for _, d in parts])
    return {"positions": cat("positions", None), "cell": np.asarray(cell, float),
            "source": cat("source", lambda m: np.zeros(m, int)).astype(int),
            "time_ps": cat("time_ps", lambda m: np.zeros(m)).astype(float),
            "split": np.concatenate([[s] * len(d["positions"]) for s, d in parts]),
            "site_q": np.asarray(d0["site_q"], float), "site_gamma": float(d0["site_gamma"]), "n": n}


def site_positions(pos, gamma):
    """the sites O, H, H, M of every molecule, (F, 4n, 3): the M-site at r_O + gamma/2 (r_H1 + r_H2 - 2 r_O)."""
    F, N, _ = pos.shape
    P = pos.reshape(F, N // 3, 3, 3)
    m = P[:, :, 0] + 0.5 * gamma * (P[:, :, 1] + P[:, :, 2] - 2 * P[:, :, 0])
    return np.concatenate([P, m[:, :, None]], axis=2).reshape(F, 4 * (N // 3), 3)


def centres(pos):
    """the molecules' centres of mass, (F, n, 3) (molecules whole, as the sets store them)."""
    F, N, _ = pos.shape
    P = pos.reshape(F, N // 3, 3, 3)
    return np.einsum("fmax,a->fmx", P, MASSES) / MASSES.sum()


def molecular_dipoles(xyz, q, mu, com):
    """every molecule's dipole about its centre of mass, (F, n, 3) e A: its sites' charges and point dipoles."""
    F, S, _ = xyz.shape
    n = S // 4
    X = xyz.reshape(F, n, 4, 3) - com[:, :, None, :]
    qq = np.broadcast_to(np.asarray(q, float).reshape(-1, S), (F, S)).reshape(F, n, 4)
    return np.einsum("fms,fmsx->fmx", qq, X) + mu.reshape(F, n, 4, 3).sum(axis=2)


def kvectors(cell, shell):
    """the shell's wavevectors, (K, 3) 1/A: k = 2 pi n . inv(cell)^T, one of each +-k pair."""
    G = 2 * np.pi * np.linalg.inv(np.asarray(cell, float)).T
    return np.array(SHELLS[shell], float) @ G


def charge_density(xyz, q, mu, kv):
    """rho_k = sum_a q_a e^{-ik.r_a} - i sum_s (k.mu_s) e^{-ik.r_s}, (F, K): point charges and point dipoles."""
    ph = np.exp(-1j * np.einsum("fsx,kx->fsk", xyz, kv))
    qq = np.broadcast_to(np.asarray(q, float).reshape(-1, xyz.shape[1]), xyz.shape[:2])
    return np.einsum("fs,fsk->fk", qq, ph) - 1j * np.einsum("fsx,kx,fsk->fk", mu, kv, ph)


def polarisation_k(d, com, kv):
    """P_k = sum_m d_m e^{-ik.r_m}, (F, K, 3), the molecules' dipoles at their centres of mass."""
    ph = np.exp(-1j * np.einsum("fmx,kx->fmk", com, kv))
    return np.einsum("fmx,fmk->fkx", d, ph)


def channels(xyz, q, mu, com, kv):
    """the longitudinal P_L = i rho_k / |k| (F, K) and the transverse P_T = P - k^(k^.P) (F, K, 3) of one polarisation."""
    kn = np.linalg.norm(kv, axis=1)
    PL = 1j * charge_density(xyz, q, mu, kv) / kn
    P = polarisation_k(molecular_dipoles(xyz, q, mu, com), com, kv)
    kh = kv / kn[:, None]
    return PL, P - np.einsum("fk,kx->fkx", np.einsum("fkx,kx->fk", P, kh), kh)


# ---------------------------------------------------------------- the statistics ------------------------------------
def blocks(source, time_ps, block=BLOCK):
    """a block id per frame: consecutive frames of one trajectory (source), `block` at a time in time order."""
    ids = np.empty(len(source), int)
    nxt = 0
    for s in np.unique(source):
        idx = np.flatnonzero(source == s)
        idx = idx[np.argsort(time_ps[idx], kind="stable")]
        for j in range(0, len(idx), block):
            ids[idx[j:j + block]] = nxt
            nxt += 1
    return ids


def jackknife(sums, ids, stat):
    """a statistic of frame-wise sums and its jackknife error over blocks: sums {name: (F,) array}, stat(dict of
    totals) -> float. Returns (value, error)."""
    names = list(sums)
    B = np.unique(ids)
    per = {k: np.array([sums[k][ids == b].sum() for b in B]) for k in names}
    tot = {k: per[k].sum() for k in names}
    val = stat(tot)
    if len(B) < 2:
        return float(val), float("nan")
    loo = np.array([stat({k: tot[k] - per[k][i] for k in names}) for i in range(len(B))])
    return float(val), float(np.sqrt((len(B) - 1) / len(B) * np.sum((loo - loo.mean()) ** 2)))


def inv_eps_inf(k):
    """1 / eps_inf(k) of MB-pol from job 3's fit: chi(k) = chi(0+)(1 + c2 k^2), chi(0+) = (1 - 1/eps_inf) / 4 pi."""
    return 1.0 - (1.0 - 1.0 / EPS_INF) * (1.0 + C2 * np.asarray(k) ** 2)


# ---------------------------------------------------------------- mbpol ---------------------------------------------
def run_mbpol(a):
    from sseft import water as W, mbpol as MB
    import e4_data
    os.makedirs(a.out, exist_ok=True)
    truth = W.MBPolTruth(a.mbx)
    for n in a.sizes:
        fr = load_frames(a.data, n)
        if fr is None:
            print(f"n {n}: no data set", flush=True)
            continue
        F = len(fr["positions"])
        i, k = (int(x) for x in a.part.split("/")) if a.part else (0, 1)
        sel = np.array_split(np.arange(F), k)[i]
        xyz, q, mu = [], [], []
        t0 = time.time()
        for j, f in enumerate(sel):
            truth.mb.set_external_field(fr["positions"][f], None, fr["cell"])
            s = truth.mb.sites(fr["positions"][f], fr["cell"])
            xyz.append(s["xyz"]); q.append(s["charges"]); mu.append(s["induced"])
            if j == 0 or (j + 1) % 10 == 0 or j + 1 == len(sel):
                print(f"  n {n}: frame {f} ({j + 1} of {len(sel)}): rms induced dipole "
                      f"{np.sqrt(np.mean(np.sum(s['induced'] ** 2, axis=1))):.4f} e A  [{time.time() - t0:.0f} s]",
                      flush=True)
        meta = {"n": n, "frames": sel.tolist(), "of": F, "truth": truth.describe(), "time": time.time() - t0}
        if i == 0:                                  # the dynamics' settings on the first frame
            dyn = MB.MBPol(a.mbx, settings=e4_data.DYNAMICS)
            dyn.set_external_field(fr["positions"][0], None, fr["cell"])
            m2 = dyn.sites(fr["positions"][0], fr["cell"])["induced"]
            dyn.finalize()
            m1 = mu[0]
            meta["dynamics_settings_rel_rms"] = float(np.sqrt(np.sum((m2 - m1) ** 2) / np.sum(m1 ** 2)))
            print(f"  n {n}: the dynamics' settings move the first frame's induced dipoles by "
                  f"{100 * meta['dynamics_settings_rel_rms']:.3f} % (rms)", flush=True)
        name = f"mbpol_n{n}" + (f"_p{i}of{k}" if k > 1 else "") + ".npz"
        np.savez(os.path.join(a.out, name), index=sel, xyz=np.array(xyz), q=np.array(q), mu=np.array(mu),
                 meta=np.array(json.dumps(meta)))
        print(f"n {n}: {len(sel)} frames into {os.path.join(a.out, name)}  [{time.time() - t0:.0f} s]", flush=True)


# ---------------------------------------------------------------- model ---------------------------------------------
def run_model(a):
    import pickle
    from dataclasses import replace
    import jax
    import jax.numpy as jnp
    import e4_water as E4
    from sseft import model as M, structure as S, water as W
    from sseft.units import bohr_to_ang
    os.makedirs(a.out, exist_ok=True)
    cfg = E4.load_config(a.config)
    with open(a.readout, "rb") as fh:
        cached = pickle.load(fh)
    K, om = int(cfg["model"]["K"]), float(cfg["model"]["omega0"])
    grid = lambda s: re.sub(r"ewald_n_max=\d+", "ewald_n_max=*", s)    # the reciprocal grid follows the cell
    for n in a.sizes:
        fr = load_frames(a.data, n)
        if fr is None:
            print(f"n {n}: no data set", flush=True)
            continue
        structs = []
        for split in SPLITS:
            p = os.path.join(a.data, f"{split}_n{n}.npz")
            if os.path.exists(p):
                structs += W.e4_structures(p, pins=True, K=K, omega0=om)
        rung_cell, rung_open = E4.rungs_for(a.rung, cfg, structs[0].cell)
        if not grid(str(cached["rung"])).startswith(grid(repr(rung_open))):
            raise ValueError(f"{a.readout} was matched for another rung than {a.rung} with {a.config}'s model")
        params = E4.with_readout(M.init_params(jax.random.PRNGKey(0), rung_open, energy_scale=cached["scale"]),
                                 cached["readout"])
        solver = a.solver or rung_cell.bf_solver
        r = E4.in_cell(replace(rung_cell, bf_solver=solver), structs[0].cell)

        def dipoles(p, b):        # (-dE/dE_ext,i, the order-0 mu_ind) at every atom, the field and its derivatives zero
            N = b["positions"].shape[0]
            z = (jnp.zeros((N, 3)), jnp.zeros((N, 3, 3)), jnp.zeros((N, 3, 3, 3)))

            def e(f):
                en, aux = M.energy_single(p, r, {**b, "ext_field": f, "ext_grad": z[1], "ext_grad2": z[2]})
                return en, aux["mu_ind"]
            g, mu0 = jax.grad(e, has_aux=True)(z[0])
            return -g, mu0
        f_dip = jax.jit(dipoles)
        mu_full, mu_ind = [], []
        t0 = time.time()
        for j, st in enumerate(structs):
            b = {k: v[0] for k, v in S.pad_batch([st], n_freq=r.K).items()}
            m = st.n_atoms
            for out, dd in zip((mu_full, mu_ind), f_dip(params, b)):
                d = bohr_to_ang(np.asarray(dd)[:m])                            # e bohr -> e A
                out.append(np.concatenate([d.reshape(m // 3, 3, 3), np.zeros((m // 3, 1, 3))], axis=1).reshape(-1, 3))
            if j == 0 or (j + 1) % 20 == 0 or j + 1 == len(structs):
                print(f"  n {n}: frame {j + 1} of {len(structs)}: rms induced dipole "
                      f"{np.sqrt(np.mean(np.sum(mu_full[-1] ** 2, axis=1)) * 4 / 3):.4f} e A per atom  "
                      f"[{time.time() - t0:.0f} s]", flush=True)
        q4 = np.tile(fr["site_q"], n)
        meta = {"n": n, "label": a.label, "config": a.config, "readout": a.readout, "rung": a.rung, "solver": solver,
                "time": time.time() - t0}
        path = os.path.join(a.out, f"model_{a.label}_n{n}.npz")
        np.savez(path, q=q4, mu_full=np.array(mu_full), mu_ind=np.array(mu_ind), meta=np.array(json.dumps(meta)))
        print(f"n {n}: {len(structs)} frames into {path}  [{time.time() - t0:.0f} s]", flush=True)


# ---------------------------------------------------------------- read ----------------------------------------------
def load_mbpol(out, n, F):
    """MB-pol's sites of the n-molecule frames from the part files: xyz (F, 4n, 3), q (F, 4n), mu (F, 4n, 3), meta."""
    files = sorted(glob.glob(os.path.join(out, f"mbpol_n{n}.npz")) + glob.glob(os.path.join(out, f"mbpol_n{n}_p*of*.npz")))
    if not files:
        return None
    xyz, q, mu = np.full((F, 4 * n, 3), np.nan), np.full((F, 4 * n), np.nan), np.full((F, 4 * n, 3), np.nan)
    metas = []
    for p in files:
        d = np.load(p, allow_pickle=False)
        idx = d["index"]
        xyz[idx], q[idx], mu[idx] = d["xyz"], d["q"], d["mu"]
        metas.append(json.loads(str(d["meta"])))
    missing = int(np.isnan(mu[:, 0, 0]).sum())
    return {"xyz": xyz, "q": q, "mu": mu, "meta": metas, "missing": missing}


def read_size(fr, mb, models, block=BLOCK):
    """every shell's readings for one cell size: L, T, T/L, 1/eps_s(k), and b_L, b_T, r of every candidate."""
    ok = ~np.isnan(mb["mu"][:, 0, 0])
    V = abs(np.linalg.det(fr["cell"]))
    C = 4 * np.pi * BETA_E2 / V
    pos = fr["positions"][ok]
    com = centres(pos)
    ids = blocks(fr["source"][ok], fr["time_ps"][ok], block)
    xyz, q, mu = mb["xyz"][ok], mb["q"][ok], mb["mu"][ok]
    cands = {"perm": (xyz, q, np.zeros_like(mu))}
    sites = site_positions(pos, fr["site_gamma"])
    for label, md in models.items():
        cands[label] = (sites, md["q"], md["mu_full"][ok])
        cands[label + "_o0"] = (sites, md["q"], md["mu_ind"][ok])         # its order-0 sector's dipoles alone
    out = {"n": fr["n"], "frames": int(ok.sum()), "blocks": int(len(np.unique(ids))), "volume": float(V), "shells": {}}
    for shell in SHELLS:
        kv = kvectors(fr["cell"], shell)
        k = float(np.linalg.norm(kv[0]))
        PL, PT = channels(xyz, q, mu, com, kv)
        sums = {"LL": np.sum(np.abs(PL) ** 2, axis=1), "TT": 0.5 * np.sum(np.abs(PT) ** 2, axis=(1, 2)),
                "nk": np.full(len(PL), float(len(kv)))}
        for c, (cx, cq, cm) in cands.items():
            YL, YT = channels(cx, cq, cm, com, kv)
            sums[f"{c}_PY_L"] = np.real(np.sum(PL * np.conj(YL), axis=1))
            sums[f"{c}_YY_L"] = np.sum(np.abs(YL) ** 2, axis=1)
            sums[f"{c}_PY_T"] = np.real(np.sum(PT * np.conj(YT), axis=(1, 2)))
            sums[f"{c}_YY_T"] = np.sum(np.abs(YT) ** 2, axis=(1, 2))
        L = jackknife(sums, ids, lambda t: C * t["LL"] / t["nk"])
        T = jackknife(sums, ids, lambda t: C * t["TT"] / t["nk"])
        TL = jackknife(sums, ids, lambda t: t["TT"] / t["LL"])
        row = {"k": k, "L": L, "T": T, "T_over_L": TL, "inv_eps_inf": float(inv_eps_inf(k)),
               "inv_eps_s": (float(inv_eps_inf(k)) - L[0], L[1]), "eps_s_from_T": (T[0] + EPS_INF, T[1]),
               "candidates": {}}
        for c in cands:
            bL = jackknife(sums, ids, lambda t, c=c: t[f"{c}_PY_L"] / t[f"{c}_YY_L"])
            bT = jackknife(sums, ids, lambda t, c=c: t[f"{c}_PY_T"] / t[f"{c}_YY_T"])
            rr = jackknife(sums, ids, lambda t, c=c: (t[f"{c}_PY_T"] / t[f"{c}_YY_T"]) / (t[f"{c}_PY_L"] / t[f"{c}_YY_L"]))
            row["candidates"][c] = {"b_L": bL, "b_T": bT, "r": rr}
        out["shells"][shell] = row
    # the static constant by the tin-foil formula, eps_s = eps_inf + 4 pi beta <M^2> / 3V, M the cell's total dipole
    M = molecular_dipoles(xyz, q, mu, com).sum(axis=1)
    mom = {"M2": np.sum(M ** 2, axis=1), "Mx": M[:, 0], "My": M[:, 1], "Mz": M[:, 2], "one": np.ones(len(M))}
    out["eps_s_tinfoil"] = jackknife(mom, ids, lambda t: EPS_INF + C / 3 * (
        t["M2"] / t["one"] - (t["Mx"] ** 2 + t["My"] ** 2 + t["Mz"] ** 2) / t["one"] ** 2))
    return out


def fit_intercept(points):
    """L = A + B k^2 by weighted least squares over (k, L, sigma): (A, sigma_A, B, chi2 per degree of freedom)."""
    k, L, s = (np.array(x, float) for x in zip(*points))
    X = np.stack([np.ones_like(k), k ** 2], axis=1)
    W = 1.0 / s ** 2
    cov = np.linalg.inv(X.T @ (W[:, None] * X))
    coef = cov @ X.T @ (W * L)
    chi2 = float(np.sum(W * (L - X @ coef) ** 2) / max(len(L) - 2, 1))
    return float(coef[0]), float(np.sqrt(cov[0, 0])), float(coef[1]), chi2


def evaluate(sizes):
    """S0.1-S0.4 from read_size's readings by size ({n: reading}): each with its value, error, band and verdict."""
    out = {}
    pts = [(row["k"], row["L"][0], row["L"][1]) for n, rd in sizes.items() if n >= 128
           for row in rd["shells"].values() if row["k"] <= K_FIT]
    if len(pts) >= 3:
        A, sA, B, chi2 = fit_intercept(pts)
        lo, hi = BANDS["S0.1"]
        out["S0.1"] = {"value": A, "error": sA, "slope_B": B, "chi2_dof": chi2, "points": len(pts),
                       "band": [lo, hi], "holds": bool(lo <= A <= hi)}
    tl = [(n, sh, row["T_over_L"]) for n, rd in sizes.items() if n >= 128 for sh, row in rd["shells"].items()
          if row["k"] <= K_FIT]
    if tl:
        out["S0.2"] = {"values": {f"n{n} ({sh})": v for n, sh, v in tl}, "band": [BANDS["S0.2"][0], None],
                       "holds": bool(all(v[0] >= BANDS["S0.2"][0] for _, _, v in tl))}
    for key, lab, ns in (("S0.3", "perm", (512, 1024)), ("S0.4", S04_MODEL, (128, 256))):
        rows = {}
        for n in ns:
            if n in sizes and lab in sizes[n]["shells"]["100"]["candidates"]:
                rows[f"n{n} (100), {lab}"] = sizes[n]["shells"]["100"]["candidates"][lab]["r"]
        if rows:
            lo, hi = BANDS[key]
            out[key] = {"values": rows, "band": [lo, hi], "holds": bool(all(lo <= v[0] <= hi for v in rows.values()))}
    return out


def run_read(a):
    sizes, notes = {}, {}
    for n in SIZES:
        fr = load_frames(a.data, n)
        if fr is None:
            continue
        mb = load_mbpol(a.out, n, len(fr["positions"]))
        if mb is None:
            print(f"n {n}: no MB-pol sites in {a.out}: run the mbpol step", flush=True)
            continue
        if mb["missing"]:
            print(f"n {n}: {mb['missing']} of {len(fr['positions'])} frames have no MB-pol sites yet (a part missing)")
        models = {}
        for p in sorted(glob.glob(os.path.join(a.out, f"model_*_n{n}.npz"))):
            label = os.path.basename(p)[len("model_"):-len(f"_n{n}.npz")]
            d = np.load(p, allow_pickle=False)
            models[label] = {"q": d["q"], "mu_full": d["mu_full"], "mu_ind": d["mu_ind"]}
        sizes[n] = read_size(fr, mb, models, a.block)
        notes[n] = [m.get("dynamics_settings_rel_rms") for m in mb["meta"] if "dynamics_settings_rel_rms" in m]
    if not sizes:
        raise SystemExit("nothing to read")
    bands = evaluate(sizes)
    pm = lambda v, d=3: f"{v[0]:.{d}f} +- {v[1]:.{d}f}"
    print(f"E4, S0: the polarisation sum rules on MB-pol's trajectories (notes/e4-s0-preregistration.md), {T_K:g} K, "
          f"blocks of {a.block} frames")
    for name, b in bands.items():
        lo, hi = b["band"]
        band = f"[{lo:g}, {hi:g}]" if hi is not None else f">= {lo:g}"
        print(f"  {name}: {'holds' if b['holds'] else 'FAILS'} (band {band})")
        if "value" in b:
            print(f"      L(k -> 0) = A = {b['value']:.3f} +- {b['error']:.3f} (B {b['slope_B']:+.2f} A^2; "
                  f"{b['points']} shells, chi2/dof {b['chi2_dof']:.2f})")
        for k, v in b.get("values", {}).items():
            print(f"      {k:24s} {pm(v, 3 if name != 'S0.2' else 1)}")
    print("\n  Beside them, not registered: every shell")
    for n, rd in sizes.items():
        print(f"  n {n} ({rd['frames']} frames, {rd['blocks']} blocks; the dynamics' settings move the induced "
              f"dipoles by {', '.join(f'{100 * x:.3f} %' for x in notes[n]) or 'not checked'}): eps_s by the tin-foil "
              f"formula {pm(rd['eps_s_tinfoil'], 1)}")
        for sh, row in rd["shells"].items():
            c = row["candidates"]
            print(f"    ({sh}) k {row['k']:.4f}: L {pm(row['L'])}, T {pm(row['T'], 2)}, T/L {pm(row['T_over_L'], 1)}; "
                  f"1/eps_inf(k) {row['inv_eps_inf']:.3f}, 1/eps_s(k) {pm(row['inv_eps_s'])}, eps_s from T "
                  f"{pm(row['eps_s_from_T'], 1)}")
            print("        r = b_T / b_L: " + "; ".join(f"{lab} {pm(v['r'], 2)} (b_L {v['b_L'][0]:.3f}, "
                                                        f"b_T {v['b_T'][0]:.3f})" for lab, v in c.items()))
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump({"bands": bands, "sizes": {str(n): v for n, v in sizes.items()},
                       "settings_check": {str(n): v for n, v in notes.items()},
                       "constants": {"T_K": T_K, "beta_e2_per_A": BETA_E2, "eps_inf": EPS_INF, "c2": C2,
                                     "block": a.block}}, fh, indent=1)
        print(f"\nwritten {a.json}")
    return bands, sizes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("mbpol", "model", "read"):
        s = sub.add_parser(name)
        s.add_argument("--data", default="data/e4")
        s.add_argument("--out", default="results/e4_s0")
    m = sub.choices["mbpol"]
    m.add_argument("--mbx", required=True, help="MBX's build, e.g. external/mbx/build/MBX")
    m.add_argument("--sizes", type=int, nargs="*", default=list(SIZES))
    m.add_argument("--part", default=None, help="i/k: the i-th of k contiguous shares of each size's frames")
    m = sub.choices["model"]
    m.add_argument("--config", required=True)
    m.add_argument("--readout", required=True, help="a matched read-out (scripts/e4_water.py matched's pickle)")
    m.add_argument("--rung", default="M_16C2pm_us")
    m.add_argument("--label", default="C")
    m.add_argument("--sizes", type=int, nargs="*", default=[64, 128, 256])
    m.add_argument("--solver", default=None, choices=["gmres", "direct"],
                   help="the mutual response's solve (default: the config's model's)")
    m = sub.choices["read"]
    m.add_argument("--json", default="results/e4_s0/s0.json")
    m.add_argument("--block", type=int, default=BLOCK)
    a = ap.parse_args()
    {"mbpol": run_mbpol, "model": run_model, "read": run_read}[a.cmd](a)


if __name__ == "__main__":
    main()
