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
  S0.5 (the pre-registration's section 6, from the nuclear EFT note): the transverse regression's residual
  v_eta = (4 pi beta / V) <|P_T - b_T Y_T|^2> per transverse direction, and rho = (r - eps_inf(k)) / v_eta, which the
  EFT's Coulomb term makes 1 for Y = P_perm. Beside it, the proper stiffnesses 1/A_T = 4 pi / T and
  1/A_L = 4 pi x / (1 - eps_inf(k) x), x = 1/eps_inf(k) - L, with their a + b k^2 fits over S0.1's shells and the
  overscreening zero; 4 pi / (eps_s - eps_inf) from each cell's own dipole; and the structure factors
  S_nn(k) = <|n_k|^2> / N and S_nZ(k) = Re<n_k rho_k*> / N of the molecules' centres of mass and the charge density.
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
BANDS = {"S0.1": (0.48, 0.68), "S0.2": (20.0, np.inf), "S0.3": (1.45, 1.95), "S0.4": (0.90, 1.10),
         "S0.5": (0.7, 1.4)}                    # S0.5: registered 12 Nov, 14:48 CET (the pre-registration's section 6)
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
def dipole_fn(r):
    """for one cell's rung r, a function of (params, b) giving at every atom, with the external field and its
    derivatives held at zero: -dE/dE_ext,i (the frozen form's polarisation: the order-0 sector's mu_ind with the
    orders >= 1's field-gradient terms), the order-0 mu_ind alone, and the isotropic part tr C_i / 3 of the order-0
    local polarisabilities (bohr^3; zero for a rung without them), all in atomic units. Used by the model step here and
    by scripts/e4_md.py at every saved frame."""
    import jax
    import jax.numpy as jnp
    from sseft import model as M

    def dipoles(p, b):
        N = b["positions"].shape[0]
        z = (jnp.zeros((N, 3)), jnp.zeros((N, 3, 3)), jnp.zeros((N, 3, 3, 3)))

        def e(f):
            en, aux = M.energy_single(p, r, {**b, "ext_field": f, "ext_grad": z[1], "ext_grad2": z[2]})
            return en, (aux.get("mu_ind", jnp.zeros((N, 3))), aux.get("C_pol", jnp.zeros((N, 3, 3))))
        g, (mu0, C) = jax.grad(e, has_aux=True)(z[0])
        return -g, mu0, jnp.trace(C, axis1=-2, axis2=-1) / 3.0
    return dipoles


def with_m_sites(d):
    """atom dipoles (3n, 3) as site dipoles (4n, 3), O, H, H, M per molecule, with nothing at the M-sites."""
    m = len(d)
    return np.concatenate([d.reshape(m // 3, 3, 3), np.zeros((m // 3, 1, 3))], axis=1).reshape(-1, 3)


def run_model(a):
    import pickle
    from dataclasses import replace
    import jax
    import e4_water as E4
    from sseft import model as M, structure as S, water as W
    from sseft import kernels as kn
    from sseft.units import bohr_to_ang
    os.makedirs(a.out, exist_ok=True)
    cfg = E4.load_config(a.config)
    with open(a.readout, "rb") as fh:
        cached = pickle.load(fh)
    K, om = int(cfg["model"]["K"]), float(cfg["model"]["omega0"])
    grid = lambda s: re.sub(r"ewald_n_max=\d+", "ewald_n_max=*", s)    # the reciprocal grid follows the cell
    # the tie (R10; the v2 proposal): MB-pol's long-range c6 per site and the dispersion band's pinned alpha(0)
    w_cp = np.asarray(kn.casimir_polder_grid(K, om)[0])
    pins = np.asarray(W.mbpol_dispersion_pins(np.array([8, 1]), K=K, omega0=om))
    tie = {"c6_site_kcal_A6_sqrt": {"O": float(W.MBPOL_C6_LR[8]), "H": float(W.MBPOL_C6_LR[1])},
           "pins_alpha0_bohr3": {"O": float(pins[0, 0] * (1 + (w_cp[0] / 0.5) ** 2)),
                                 "H": float(pins[1, 0] * (1 + (w_cp[0] / 0.5) ** 2))}}
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
        f_dip = jax.jit(dipole_fn(r))
        mu_full, mu_ind, c_iso = [], [], []
        t0 = time.time()
        for j, st in enumerate(structs):
            b = {k: v[0] for k, v in S.pad_batch([st], n_freq=r.K).items()}
            m = st.n_atoms
            full, ind, ci = f_dip(params, b)
            mu_full.append(with_m_sites(bohr_to_ang(np.asarray(full)[:m])))          # e bohr -> e A
            mu_ind.append(with_m_sites(bohr_to_ang(np.asarray(ind)[:m])))
            c_iso.append(np.asarray(ci)[:m])
            if j == 0 or (j + 1) % 20 == 0 or j + 1 == len(structs):
                print(f"  n {n}: frame {j + 1} of {len(structs)}: rms induced dipole "
                      f"{np.sqrt(np.mean(np.sum(mu_full[-1] ** 2, axis=1)) * 4 / 3):.4f} e A per atom  "
                      f"[{time.time() - t0:.0f} s]", flush=True)
        q4 = np.tile(fr["site_q"], n)
        meta = {"n": n, "label": a.label, "config": a.config, "readout": a.readout, "rung": a.rung, "solver": solver,
                "time": time.time() - t0, "tie": tie}
        path = os.path.join(a.out, f"model_{a.label}_n{n}.npz")
        np.savez(path, q=q4, mu_full=np.array(mu_full), mu_ind=np.array(mu_ind), c_iso=np.array(c_iso),
                 meta=np.array(json.dumps(meta)))
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
    """every shell's readings for one cell size: L, T, T/L, 1/eps_s(k), the proper stiffnesses 1/A_T and 1/A_L, S_nn
    and S_nZ / k^2, and b_L, b_T, r, v_eta and rho = (r - eps_inf(k)) / v_eta of every candidate; and the cell's
    eps_s by the tin-foil formula with a = 4 pi / (eps_s - eps_inf)."""
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
        n_k = np.exp(-1j * np.einsum("fmx,kx->fmk", com, kv)).sum(axis=1)          # the centres' density, (F, K)
        sums = {"LL": np.sum(np.abs(PL) ** 2, axis=1), "TT": 0.5 * np.sum(np.abs(PT) ** 2, axis=(1, 2)),
                "nk": np.full(len(PL), float(len(kv))), "NN": np.sum(np.abs(n_k) ** 2, axis=1),
                "NZ": np.real(np.sum(n_k * np.conj(charge_density(xyz, q, mu, kv)), axis=1))}
        for c, (cx, cq, cm) in cands.items():
            YL, YT = channels(cx, cq, cm, com, kv)
            sums[f"{c}_PY_L"] = np.real(np.sum(PL * np.conj(YL), axis=1))
            sums[f"{c}_YY_L"] = np.sum(np.abs(YL) ** 2, axis=1)
            sums[f"{c}_PY_T"] = np.real(np.sum(PT * np.conj(YT), axis=(1, 2)))
            sums[f"{c}_YY_T"] = np.sum(np.abs(YT) ** 2, axis=(1, 2))
        ie = float(inv_eps_inf(k))
        L = jackknife(sums, ids, lambda t: C * t["LL"] / t["nk"])
        T = jackknife(sums, ids, lambda t: C * t["TT"] / t["nk"])
        TL = jackknife(sums, ids, lambda t: t["TT"] / t["LL"])

        def inv_A_L(t):              # 4 pi / (eps_s,L - eps_inf(k)) with 1/eps_s,L = x, smooth through x = 0
            x = ie - C * t["LL"] / t["nk"]
            return 4 * np.pi * x / (1.0 - x / ie)
        row = {"k": k, "L": L, "T": T, "T_over_L": TL, "inv_eps_inf": ie,
               "inv_eps_s": (ie - L[0], L[1]), "eps_s_from_T": (T[0] + EPS_INF, T[1]),
               "inv_A_T": jackknife(sums, ids, lambda t: 4 * np.pi * t["nk"] / (C * t["TT"])),
               "inv_A_L": jackknife(sums, ids, inv_A_L),
               "S_nn": jackknife(sums, ids, lambda t: t["NN"] / (t["nk"] * fr["n"])),
               "S_nZ_over_k2": jackknife(sums, ids, lambda t: t["NZ"] / (t["nk"] * fr["n"] * k ** 2)),
               "candidates": {}}
        for c in cands:
            b_T = lambda t, c=c: t[f"{c}_PY_T"] / t[f"{c}_YY_T"]
            r_ = lambda t, c=c: b_T(t) / (t[f"{c}_PY_L"] / t[f"{c}_YY_L"])
            v_eta = lambda t, c=c: C * (t["TT"] - 0.5 * t[f"{c}_PY_T"] ** 2 / t[f"{c}_YY_T"]) / t["nk"]   # per direction
            row["candidates"][c] = {"b_L": jackknife(sums, ids, lambda t, c=c: t[f"{c}_PY_L"] / t[f"{c}_YY_L"]),
                                    "b_T": jackknife(sums, ids, b_T), "r": jackknife(sums, ids, r_),
                                    "v_eta": jackknife(sums, ids, v_eta),
                                    "rho": jackknife(sums, ids, lambda t: (r_(t) - 1.0 / ie) / v_eta(t)
                                                     if v_eta(t) > 1e-9 * C * t["TT"] / t["nk"]
                                                     else float("nan"))}         # nan: Y is P itself
        out["shells"][shell] = row
    # the static constant by the tin-foil formula, eps_s = eps_inf + 4 pi beta <M^2> / 3V, M the cell's total dipole
    M = molecular_dipoles(xyz, q, mu, com).sum(axis=1)
    mom = {"M2": np.sum(M ** 2, axis=1), "Mx": M[:, 0], "My": M[:, 1], "Mz": M[:, 2], "one": np.ones(len(M))}
    var_M = lambda t: t["M2"] / t["one"] - (t["Mx"] ** 2 + t["My"] ** 2 + t["Mz"] ** 2) / t["one"] ** 2
    out["eps_s_tinfoil"] = jackknife(mom, ids, lambda t: EPS_INF + C / 3 * var_M(t))
    out["a_cell"] = jackknife(mom, ids, lambda t: 4 * np.pi / (C / 3 * var_M(t)))       # 4 pi / (eps_s - eps_inf)
    return out


def fit_line(points):
    """y = a + b k^2 by weighted least squares over (k, y, sigma): (a, sigma_a, b, sigma_b, chi2 per degree of freedom)."""
    k, L, s = (np.array(x, float) for x in zip(*points))
    X = np.stack([np.ones_like(k), k ** 2], axis=1)
    W = 1.0 / s ** 2
    cov = np.linalg.inv(X.T @ (W[:, None] * X))
    coef = cov @ X.T @ (W * L)
    chi2 = float(np.sum(W * (L - X @ coef) ** 2) / max(len(L) - 2, 1))
    return float(coef[0]), float(np.sqrt(cov[0, 0])), float(coef[1]), float(np.sqrt(cov[1, 1])), chi2


def fit_intercept(points):
    """L = A + B k^2 by weighted least squares over (k, L, sigma): (A, sigma_A, B, chi2 per degree of freedom)."""
    a, sa, b, _, chi2 = fit_line(points)
    return a, sa, b, chi2


def eft_fits(sizes):
    """the nuclear EFT note's (7) on S0.1's shells (128-1024 molecules, k <= K_FIT): 1/A_T and 1/A_L each fitted as
    a + b k^2, the overscreening zero k_os = (-a_L / b_L)^(1/2) where a_L and b_L differ in sign, and a = 4 pi /
    (eps_s - eps_inf) from every cell's own dipole beside them. Not registered."""
    out = {}
    for ch in ("T", "L"):
        pts = [(row["k"], row[f"inv_A_{ch}"][0], row[f"inv_A_{ch}"][1]) for n, rd in sizes.items() if n >= 128
               for row in rd["shells"].values() if row["k"] <= K_FIT]
        pts = [p for p in pts if np.isfinite(p[1]) and np.isfinite(p[2]) and p[2] > 0]
        if len(pts) >= 3:
            a, sa, b, sb, chi2 = fit_line(pts)
            out[ch] = {"a": a, "a_error": sa, "b": b, "b_error": sb, "chi2_dof": chi2, "points": len(pts)}
    if "L" in out and out["L"]["a"] * out["L"]["b"] < 0:
        out["k_os"] = float(np.sqrt(-out["L"]["a"] / out["L"]["b"]))
    out["a_cell"] = {str(n): rd["a_cell"] for n, rd in sizes.items()}
    return out


def evaluate(sizes):
    """S0.1-S0.5 from read_size's readings by size ({n: reading}): each with its value, error, band and verdict."""
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
    for key, lab, ns, what in (("S0.3", "perm", (512, 1024), "r"), ("S0.4", S04_MODEL, (128, 256), "r"),
                               ("S0.5", "perm", (512, 1024), "rho")):
        rows, parts = {}, {}
        for n in ns:
            if n in sizes and lab in sizes[n]["shells"]["100"]["candidates"]:
                row = sizes[n]["shells"]["100"]
                cand = row["candidates"][lab]
                rows[f"n{n} (100), {lab}"] = cand[what]
                if key == "S0.5":       # its two sides: r - eps_inf(k) (r's error) and v_eta
                    parts[f"n{n} (100), {lab}"] = {"excess": (cand["r"][0] - 1.0 / row["inv_eps_inf"], cand["r"][1]),
                                                   "v_eta": cand["v_eta"]}
        if rows:
            lo, hi = BANDS[key]
            out[key] = {"values": rows, "band": [lo, hi], "holds": bool(all(lo <= v[0] <= hi for v in rows.values()))}
            if parts:
                out[key]["parts"] = parts
    return out


def tie_reading(c_iso, tie):
    """R10's tie read on a frozen form's order-0 polarisabilities (notes/v2-one-field-proposal.md; not registered).
    c_iso: {n: (F, 3n) tr C_i / 3 in bohr^3}; tie: the model file's meta["tie"] (MB-pol's long-range c6 per site, the
    dispersion pins' alpha(0)). Per species and per molecule: alpha(0)'s mean and spread over the liquid frames; the
    frequency one oscillator per site (and per molecule) needs to give MB-pol's c6 from it, omega = 4 C6 / 3 alpha^2;
    the cross C6_OH that the one-oscillator tie then implies, against MB-pol's; and the implied C6's relative spread at
    fixed omega, which a tied model would carry where MB-pol's c6 has none."""
    C = np.concatenate([np.asarray(c, float) for c in c_iso.values()])
    aO = C[:, 0::3].ravel()
    aH = np.concatenate([C[:, 1::3].ravel(), C[:, 2::3].ravel()])
    amol = (C[:, 0::3] + C[:, 1::3] + C[:, 2::3]).ravel()
    u = BOHR_PER_ANGSTROM ** 6 / KCALMOL_PER_HARTREE                 # kcal/mol A^6 -> hartree bohr^6
    cO, cH = tie["c6_site_kcal_A6_sqrt"]["O"], tie["c6_site_kcal_A6_sqrt"]["H"]
    C6 = {"OO": cO * cO * u, "HH": cH * cH * u, "OH": cO * cH * u, "mol": (cO + 2 * cH) ** 2 * u}
    w = {"O": 4 * C6["OO"] / (3 * aO.mean() ** 2), "H": 4 * C6["HH"] / (3 * aH.mean() ** 2),
         "mol": 4 * C6["mol"] / (3 * amol.mean() ** 2)}
    cross = 1.5 * aO.mean() * aH.mean() * w["O"] * w["H"] / (w["O"] + w["H"])
    rel = lambda a: float(np.std(a ** 2) / np.mean(a ** 2))
    return {"frames": int(len(C)), "alpha0_bohr3": {"O": (float(aO.mean()), float(aO.std())),
                                                   "H": (float(aH.mean()), float(aH.std())),
                                                   "molecule": (float(amol.mean()), float(amol.std()))},
            "pins_alpha0_bohr3": tie["pins_alpha0_bohr3"], "C6_mbpol_au": C6, "omega_hartree": w,
            "cross_OH_over_mbpol": float(cross / C6["OH"]),
            "C6_spread_at_fixed_omega": {"O": rel(aO), "H": rel(aH), "molecule": rel(amol)}}


def run_read(a):
    sizes, notes, ties = {}, {}, {}
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
            if "c_iso" in d.files:                              # the tie reading (model files from v0.0.29's S1 patch on)
                meta = json.loads(str(d["meta"]))
                ties.setdefault(label, {"c_iso": {}, "tie": meta.get("tie")})["c_iso"][n] = d["c_iso"]
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
            extra = ""
            if k in b.get("parts", {}):
                p = b["parts"][k]
                extra = f"   (r - eps_inf(k) {pm(p['excess'])}; v_eta {pm(p['v_eta'])})"
            print(f"      {k:24s} {pm(v, 3 if name != 'S0.2' else 1)}{extra}")
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
    eft = eft_fits(sizes)
    print("\n  From the nuclear EFT note (pre-registration section 6; not registered): every shell")
    print("    v_eta = (4 pi beta / V) <|P_T - b_T Y_T|^2> per direction, and rho = (r - eps_inf(k)) / v_eta;")
    print("    the proper stiffnesses 1/A_T = 4 pi / T and 1/A_L = 4 pi x / (1 - eps_inf(k) x), x = 1/eps_inf(k) - L;")
    print("    S_nn and S_nZ / k^2 (e A^2) of the molecules' centres of mass and the charge density")
    for n, rd in sizes.items():
        print(f"  n {n}: 4 pi / (eps_s - eps_inf) from the cell's dipole {pm(rd['a_cell'])}")
        for sh, row in rd["shells"].items():
            c = row["candidates"]
            print(f"    ({sh}) k {row['k']:.4f}: 1/A_T {pm(row['inv_A_T'])}, 1/A_L {pm(row['inv_A_L'])}; "
                  f"S_nn {pm(row['S_nn'])}, S_nZ/k^2 {pm(row['S_nZ_over_k2'], 4)}")
            print("        v_eta (rho): " + "; ".join(f"{lab} {pm(v['v_eta'])} ({pm(v['rho'], 2)})" for lab, v in c.items()))
    for ch in ("T", "L"):
        if ch in eft:
            f = eft[ch]
            print(f"  1/A_{ch} = a + b k^2 over S0.1's {f['points']} shells: a {f['a']:.3f} +- {f['a_error']:.3f}, "
                  f"b {f['b']:+.2f} +- {f['b_error']:.2f} A^2, chi2/dof {f['chi2_dof']:.2f}")
    if "k_os" in eft:
        print(f"  the overscreening zero of 1/A_L: k_os = (-a_L / b_L)^(1/2) = {eft['k_os']:.3f} /A")
    tie_out = {}
    for label, t in ties.items():
        if not t["tie"]:
            continue
        tr = tie_out[label] = tie_reading(t["c_iso"], t["tie"])
        al, w, sp = tr["alpha0_bohr3"], tr["omega_hartree"], tr["C6_spread_at_fixed_omega"]
        print(f"\n  The tie of induction and dispersion (R10; notes/v2-one-field-proposal.md; not registered): {label}, "
              f"{tr['frames']} frames of {', '.join(str(n) for n in t['c_iso'])} molecules")
        print(f"    the order-0 alpha(0) = tr C / 3, mean +- sd over the liquid's atoms: O {al['O'][0]:.3f} +- "
              f"{al['O'][1]:.3f}, H {al['H'][0]:.3f} +- {al['H'][1]:.3f}, per molecule {al['molecule'][0]:.3f} +- "
              f"{al['molecule'][1]:.3f} bohr^3; the dispersion pins' alpha(0): O {tr['pins_alpha0_bohr3']['O']:.3f}, "
              f"H {tr['pins_alpha0_bohr3']['H']:.3f}")
        print(f"    one oscillator for MB-pol's long-range c6 on that alpha(0): omega_O {w['O']:.3f}, omega_H {w['H']:.3f}, "
              f"per molecule {w['mol']:.3f} hartree (the gas-phase monomer's 4 C6 / 3 alpha^2: 0.65-0.67); the cross "
              f"C6_OH it implies / MB-pol's {tr['cross_OH_over_mbpol']:.3f}")
        print(f"    the implied C6's spread over the liquid at fixed omega (sd / mean of alpha^2): O {100 * sp['O']:.1f} %, "
              f"H {100 * sp['H']:.1f} %, per molecule {100 * sp['molecule']:.1f} %")
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump({"bands": bands, "eft": eft, "tie": tie_out, "sizes": {str(n): v for n, v in sizes.items()},
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
