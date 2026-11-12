"""
E4's MD driver, for S1 and S2 (notes/e4-s1-preregistration.md; the project's path-integral-ir-sector.md section
4.2-4.3): rigid water at 300 K with the forces of a trained E4 model or of MB-pol, and the read of the sum rules on the
trajectories it writes.

    python scripts/e4_md.py time --engine model --config configs/e4_job5.yaml \\
        --params results/e4_job5/M_16C2pm_us_seed0.params.pkl --rung M_16C2pm_us --n 64 [--steps 40]
    python scripts/e4_md.py run --engine model --config configs/e4_job5.yaml \\
        --params results/e4_job5/M_16C2pm_us_seed0.params.pkl --rung M_16C2pm_us --n 64 --seed 0 --label Dp
    python scripts/e4_md.py run --engine mbpol --mbx external/mbx/build/MBX --n 64 --seed 0 --label mbpol
    python scripts/e4_md.py read [--out results/e4_s1]

run: sseft/rigidmd.py (velocity Verlet with SHAKE/RATTLE, the Bussi thermostat at 300 K, tau 100 fs, 2 fs steps, the
  centre-of-mass velocity removed every 100 steps) from the first frame of the seed-th quarter of data/e4's frames of
  the size; --equil_ps not read, then --sample_ps with a frame every --save_every steps: the positions, the site
  positions O, H, H, M, their charges and the polarisation (a model's -dE/dE_ext,i at every atom, as S0's model step;
  MB-pol's induced dipoles at zero field). Written to <out>/traj_<label>_n<n>_s<seed>.npz, atomically, every
  --ckpt_min minutes and at the end, with the integrator's state and the thermostat's random state: run the same
  command again and the trajectory continues exactly where it stopped. It stops itself before --max_hours, and at once
  if the energy or the forces stop being finite (recorded in the file's meta).
time: seconds a step (forces) and a saved frame (the polarisation), after compilation; and what a run would take.
read: S1.1 (|L_D' - L_MB-pol| <= 0.02 + 2 sigma on the 64 cell's (100) and the 128 cell's (100) and (110) shells) and
  S2.1 (the dagger's own L(k1) >= 0.90 on the 64 cell), with S0's estimators (scripts/e4_sumrules.py) on the pooled
  trajectories of each label and size, jackknife over blocks of 10 ps within a trajectory; beside them T, T/L, eps_s,
  the proper stiffnesses, the integrated longitudinal relaxation time tau_L of the first shell, and each trajectory's
  temperature, conserved-energy drift and constraint error. NumPy only.
"""
import argparse
import glob
import json
import math
import os
import sys
import time
import zlib

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import e4_sumrules as SR  # noqa: E402
from sseft import rigidmd as R  # noqa: E402

DT_FS, TAU_FS, T_K = 2.0, 100.0, 300.0
EQUIL_PS, SAMPLE_PS, SAVE_EVERY = 10.0, 100.0, 25          # 25 steps: a frame every 0.05 ps
LOG_EVERY = 50                                              # steps: the log every 0.1 ps
BLOCK_PS = 10.0                                             # the read's jackknife blocks, longer than tau_D
MIN_SAMPLE_PS = 50.0                                        # S1.1 needs two trajectories with at least this much
S1_SHELLS = {64: ("100",), 128: ("100", "110")}
BANDS = {"S1.1": 0.02, "S2.1": 0.90}


# ---------------------------------------------------------------- engines ------------------------------------------
class ModelEngine:
    """a trained E4 model as the Hamiltonian: its interaction energy and forces in a cell, and its own polarisation --
    the pinned site charges and -dE/dE_ext,i at every atom (none for a rung without a response)."""

    def __init__(self, config, params_path, rung_name, pos0, cell, site_q, gamma):
        import pickle
        import jax
        import jax.numpy as jnp
        import e4_water as E4
        from sseft import model as M, structure as S, water as W
        from sseft.units import ang_to_bohr, KCALMOL_PER_HARTREE, BOHR_PER_ANGSTROM
        self.jnp, self.ang_to_bohr = jnp, ang_to_bohr
        self.e_unit, self.f_unit = KCALMOL_PER_HARTREE, KCALMOL_PER_HARTREE * BOHR_PER_ANGSTROM
        cfg = E4.load_config(config)
        K, om = int(cfg["model"]["K"]), float(cfg["model"]["omega0"])
        st = W.water_structure(np.asarray(pos0, float), energy=0.0, forces=np.zeros_like(pos0), pinned_q=False,
                               info={"n_mol": len(pos0) // 3})
        st.cell, st.pbc = ang_to_bohr(np.asarray(cell, float)), True
        W.pin_rigid_site_model([st], site_q, gamma, K, om)
        rung_cell, _ = E4.rungs_for(rung_name, cfg, st.cell)
        r = E4.in_cell(rung_cell, st.cell)
        with open(params_path, "rb") as fh:
            self.params = jax.tree_util.tree_map(jnp.asarray, pickle.load(fh))
        b0 = {k: jnp.asarray(v[0]) for k, v in S.pad_batch([st], n_freq=r.K).items()}
        self.m = st.n_atoms
        self.q4 = np.tile(np.asarray(site_q, float), self.m // 3)
        self.gamma = float(gamma)
        self._ef = jax.jit(lambda p, x: M.energy_forces_single(p, r, {**b0, "positions": x}))
        self.response = r.band_fields >= 0 and r.bf_mutual != 0       # the order-0 sector's induced dipoles (aux mu_ind)
        dip = SR.dipole_fn(r)
        self._dip = jax.jit(lambda p, x: dip(p, {**b0, "positions": x})[0]) if self.response else None
        self.info = {"engine": "model", "config": config, "params": params_path, "rung": rung_name,
                     "solver": r.bf_solver if r.bf_mutual else None, "response": self.response}

    def force(self, pos, cell):
        E, F = self._ef(self.params, self.jnp.asarray(self.ang_to_bohr(pos)))
        return float(E) * self.e_unit, np.asarray(F)[:self.m] * self.f_unit

    def sites(self, pos, cell):
        from sseft.units import bohr_to_ang
        xyz = SR.site_positions(np.asarray(pos)[None], self.gamma)[0]
        if self._dip is None:
            return xyz, self.q4, np.zeros_like(xyz)
        mu = bohr_to_ang(np.asarray(self._dip(self.params, self.jnp.asarray(self.ang_to_bohr(pos))))[:self.m])
        return xyz, self.q4, SR.with_m_sites(mu)


class MBPolEngine:
    """MB-pol through MBX with the dynamics' settings (scripts/e4_data.py DYNAMICS), as E4's data: forces, and at a
    saved frame its sites, charges and induced dipoles at zero field."""

    def __init__(self, mbx):
        from sseft import mbpol as MB
        import e4_data
        self.dyn = MB.MBPol(mbx, settings=e4_data.DYNAMICS)
        self.info = {"engine": "mbpol", "mbx": mbx, "settings": e4_data.DYNAMICS}

    def force(self, pos, cell):
        e, gr = self.dyn.energy_gradient_kcal(pos, cell)
        return e, -gr

    def sites(self, pos, cell):
        s = self.dyn.sites(pos, cell)
        return s["xyz"], s["charges"], s["induced"]


# ---------------------------------------------------------------- the run ------------------------------------------
def start_frame(data, n, seed):
    """the first frame of the seed-th quarter of data/e4's frames of n molecules (train, valid, test together)."""
    fr = SR.load_frames(data, n)
    if fr is None:
        raise SystemExit(f"no data set of {n} molecules in {data}")
    idx = (seed % 4) * (len(fr["positions"]) // 4)
    return np.asarray(fr["positions"][idx], float), fr, int(idx)


def _save(path, arrays, meta):
    tmp = path + ".tmp.npz"
    np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)


def simulate(engine, pos0, cell, path, seed_seq, meta, equil_ps=EQUIL_PS, sample_ps=SAMPLE_PS, save_every=SAVE_EVERY,
             ckpt_min=10.0, max_hours=3.7, stop_after_steps=None, say=print):
    """the trajectory at `path`, continued from it if it is there; returns its meta. stop_after_steps (tests): stop
    after that many steps of this call, as the time limit does."""
    cell = np.asarray(cell, float)
    n_equil = int(round(equil_ps * 1000 / DT_FS))
    n_total = n_equil + int(round(sample_ps * 1000 / DT_FS))
    frames = {"positions": [], "xyz": [], "q": [], "mu": [], "time_ps": []}
    log = []
    if os.path.exists(path):
        d = np.load(path, allow_pickle=False)
        old = json.loads(str(d["meta"]))
        if old.get("completed") or old.get("stopped"):
            say(f"{path}: {'completed' if old.get('completed') else 'stopped: ' + old['stopped']}; nothing to do")
            return old
        md = R.RigidWaterMD(d["X"].reshape(-1, 3), cell, engine.force, dt_fs=DT_FS, T_K=T_K, tau_fs=TAU_FS, seed=0)
        md.V, md.time_fs, md.heat = d["V"].copy(), float(d["time_fs"]), float(d["heat"])
        md.d = d["d"].copy()                     # the constrained distances of the start, not re-read from the drifted X
        md.rng.bit_generator.state = json.loads(str(d["rng"]))
        step = int(d["step"])
        for k in frames:
            frames[k] = list(d[k])
        log = [tuple(r) for r in d["log"]]
        meta = {**old, "continued": old.get("continued", 0) + 1}
        say(f"{path}: continued at step {step} of {n_total} ({md.time_fs / 1000:.2f} ps)")
    else:
        md = R.RigidWaterMD(pos0, cell, engine.force, dt_fs=DT_FS, T_K=T_K, tau_fs=TAU_FS, seed=seed_seq)
        md.init_velocities()
        step = 0
        meta = {**meta, "dt_fs": DT_FS, "tau_fs": TAU_FS, "T_K": T_K, "equil_ps": equil_ps, "sample_ps": sample_ps,
                "save_every": save_every, "completed": False, "stopped": None, "continued": 0, "engine_info": engine.info}
        log.append((0.0, md.temperature(), md.E_pot, md.constraint_error(), md.conserved()))

    def write():
        arr = {k: np.asarray(v, np.float64 if k == "time_ps" else np.float32) for k, v in frames.items()}
        if not frames["time_ps"]:
            n_at, n_s = md.X.size // 3, 4 * md.n
            arr = {"positions": np.zeros((0, n_at, 3), np.float32), "xyz": np.zeros((0, n_s, 3), np.float32),
                   "q": np.zeros((0, n_s), np.float32), "mu": np.zeros((0, n_s, 3), np.float32),
                   "time_ps": np.zeros(0)}
        meta.update(step=step, sampled_ps=len(frames["time_ps"]) * save_every * DT_FS / 1000.0)
        _save(path, {**arr, "log": np.asarray(log, float).reshape(-1, 5), "X": md.X, "V": md.V, "d": md.d,
                     "time_fs": np.array(md.time_fs), "heat": np.array(md.heat), "step": np.array(step),
                     "rng": np.array(json.dumps(md.rng.bit_generator.state))}, meta)

    t_start = t_ckpt = time.time()
    done_here = 0
    while step < n_total:
        md.step()
        step += 1
        done_here += 1
        if step % 100 == 0:
            md.remove_com_velocity()
        if not (np.isfinite(md.E_pot) and np.all(np.isfinite(md.A)) and np.all(np.isfinite(md.V))):
            meta["stopped"] = f"non-finite energy or forces at step {step} ({md.time_fs / 1000:.3f} ps)"
            say(f"{path}: {meta['stopped']}")
            break
        if step % LOG_EVERY == 0:
            log.append((md.time_fs / 1000.0, md.temperature(), md.E_pot, md.constraint_error(), md.conserved()))
        if step > n_equil and (step - n_equil) % save_every == 0:
            pos = md.positions()
            xyz, q, mu = engine.sites(pos, cell)
            for k, v in zip(frames, (pos, xyz, q, mu, md.time_fs / 1000.0)):
                frames[k].append(v)
        over = (time.time() - t_start) > max_hours * 3600.0 or (stop_after_steps is not None and done_here >= stop_after_steps)
        if over or time.time() - t_ckpt > ckpt_min * 60.0:
            write()
            t_ckpt = time.time()
            if over:
                say(f"{path}: stopped for the time limit at step {step} of {n_total} ({md.time_fs / 1000:.2f} ps); "
                    f"run the same command again to continue")
                return meta
    meta["completed"] = meta["stopped"] is None and step >= n_total
    meta["seconds_last_job"] = time.time() - t_start
    write()
    say(f"{path}: {'completed' if meta['completed'] else 'stopped'} at step {step} ({md.time_fs / 1000:.2f} ps), "
        f"{len(frames['time_ps'])} frames")
    return meta


def build(a, n, seed):
    pos0, fr, idx = start_frame(a.data, n, seed)
    if a.engine == "mbpol":
        eng = MBPolEngine(a.mbx)
    else:
        eng = ModelEngine(a.config, a.params, a.rung, pos0, fr["cell"], fr["site_q"], fr["site_gamma"])
    meta = {"label": a.label, "n": n, "seed": seed, "start_frame": idx, "data": a.data,
            "site_gamma": float(fr["site_gamma"]), "cell": np.asarray(fr["cell"]).tolist()}
    return eng, pos0, fr["cell"], meta


def run_run(a):
    os.makedirs(a.out, exist_ok=True)
    eng, pos0, cell, meta = build(a, a.n, a.seed)
    seq = np.random.SeedSequence([a.seed, a.n, zlib.crc32(a.label.encode())])
    path = os.path.join(a.out, f"traj_{a.label}_n{a.n}_s{a.seed}.npz")
    simulate(eng, pos0, cell, path, seq, meta, equil_ps=a.equil_ps, sample_ps=a.sample_ps, save_every=a.save_every,
             ckpt_min=a.ckpt_min, max_hours=a.max_hours)


def run_time(a):
    eng, pos0, cell, _ = build(a, a.n, 0)
    t0 = time.time()
    md = R.RigidWaterMD(pos0, cell, eng.force, dt_fs=DT_FS, T_K=T_K, tau_fs=TAU_FS, seed=0)  # compiles the forces
    t_compile = time.time() - t0
    md.init_velocities()
    md.step()
    t0 = time.time()
    eng.sites(md.positions(), cell)                             # compiles the polarisation
    t_compile += time.time() - t0
    c0 = md.conserved()
    t0 = time.time()
    for _ in range(a.steps):
        md.step()
    t_step = (time.time() - t0) / a.steps
    dc = (md.conserved() - c0) / md.n
    k = max(1, a.steps // 10)
    t0 = time.time()
    for _ in range(k):
        eng.sites(md.positions(), cell)
    t_save = (time.time() - t0) / k
    n_steps = (EQUIL_PS + SAMPLE_PS) * 1000 / DT_FS
    total = n_steps * t_step + SAMPLE_PS * 1000 / DT_FS / SAVE_EVERY * t_save
    print(f"{a.engine} {a.label} n {a.n}: compilation {t_compile:.0f} s; {t_step:.3f} s a step, {t_save:.3f} s a saved "
          f"frame; {EQUIL_PS:g} + {SAMPLE_PS:g} ps: {total / 3600:.2f} h, {math.ceil(total / (3.7 * 3600))} job(s) of "
          f"3.7 h; T {md.temperature():.0f} K after {a.steps + 1} steps, the conserved energy moved {dc:+.1e} kcal/mol "
          f"per molecule", flush=True)


# ---------------------------------------------------------------- the read -----------------------------------------
def load_group(paths, gamma=None):
    """the pooled frames of one label and size: read_size's fr and mb, plus each trajectory's meta and log."""
    pos, xyz, q, mu, src, t, metas, logs = [], [], [], [], [], [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=False)
        m = json.loads(str(d["meta"]))
        metas.append(m); logs.append(d["log"])
        if len(d["time_ps"]) == 0:
            continue
        pos.append(d["positions"].astype(float)); xyz.append(d["xyz"].astype(float)); q.append(d["q"].astype(float))
        mu.append(d["mu"].astype(float)); t.append(d["time_ps"]); src.append(np.full(len(d["time_ps"]), m["seed"]))
    if not pos:
        return None
    m0 = metas[0]
    fr = {"positions": np.concatenate(pos), "cell": np.asarray(m0["cell"], float), "source": np.concatenate(src),
          "time_ps": np.concatenate(t), "site_gamma": m0["site_gamma"] if gamma is None else gamma, "n": m0["n"]}
    mb = {"xyz": np.concatenate(xyz), "q": np.concatenate(q), "mu": np.concatenate(mu)}
    return fr, mb, metas, logs


def tau_L(fr, mb, shell="100", window_ps=5.0):
    """the integrated relaxation time of P_L(k, t) on a shell, dt (1/2 + sum_l C(l dt)) over its normalised
    autocorrelation C averaged over the shell's vectors, the time origins and the trajectories, summed to C's first
    zero (at most window_ps)."""
    kv = SR.kvectors(fr["cell"], shell)
    com = SR.centres(fr["positions"])
    PL, _ = SR.channels(mb["xyz"], mb["q"], mb["mu"], com, kv)
    dt = np.median(np.diff(np.sort(fr["time_ps"][fr["source"] == fr["source"][0]])))
    lags = int(round(window_ps / dt))
    num, den = np.zeros(lags + 1), np.zeros(lags + 1)
    for s in np.unique(fr["source"]):
        idx = np.flatnonzero(fr["source"] == s)
        x = PL[idx[np.argsort(fr["time_ps"][idx])]]
        for l in range(min(lags, len(x) - 1) + 1):
            num[l] += np.sum(np.real(x[l:] * np.conj(x[:len(x) - l])))
            den[l] += x[l:].size
    C = (num / np.maximum(den, 1)) / (num[0] / den[0])
    zero = np.flatnonzero(C <= 0)
    end = zero[0] if len(zero) else len(C)
    return float(dt * (0.5 + C[1:end].sum())), C            # dt (1/2 + sum of C up to its first zero)


def stability(metas, logs):
    out = []
    for m, lg in zip(metas, logs):
        lg = np.asarray(lg)
        samp = lg[lg[:, 0] > m.get("equil_ps", EQUIL_PS)] if len(lg) else lg
        drift = float(np.polyfit(samp[:, 0], samp[:, 4], 1)[0] / m["n"]) if len(samp) > 2 else float("nan")
        out.append({"seed": m["seed"], "sampled_ps": m.get("sampled_ps", 0.0), "completed": m.get("completed"),
                    "stopped": m.get("stopped"), "T_mean": float(samp[:, 1].mean()) if len(samp) else float("nan"),
                    "drift_kcal_per_ps_per_molecule": drift,
                    "constraint_max": float(lg[:, 3].max()) if len(lg) else float("nan")})
    return out


def run_read(a):
    groups = {}
    for p in sorted(glob.glob(os.path.join(a.out, "traj_*_n*_s*.npz"))):
        base = os.path.basename(p)[len("traj_"):-len(".npz")]
        label, n = base.rsplit("_n", 1)[0], int(base.rsplit("_n", 1)[1].split("_s")[0])
        groups.setdefault((label, n), []).append(p)
    if not groups:
        raise SystemExit(f"no trajectories in {a.out}")
    block = int(round(BLOCK_PS / (SAVE_EVERY * DT_FS / 1000.0)))
    res = {}
    for (label, n), paths in sorted(groups.items()):
        g = load_group(paths)
        if g is None:
            continue
        fr, mb, metas, logs = g
        rd = SR.read_size(fr, mb, {}, block)
        rd["tau_L_100_ps"] = tau_L(fr, mb)[0]
        rd["trajectories"] = stability(metas, logs)
        rd["enough"] = sum(1 for s in rd["trajectories"] if s["sampled_ps"] >= a.min_sample_ps) >= 2
        res[(label, n)] = rd
    pm = lambda v, d=3: f"{v[0]:.{d}f} +- {v[1]:.{d}f}"
    print(f"E4, S1 and S2: the sum rules in the models' own liquids (notes/e4-s1-preregistration.md), blocks of "
          f"{BLOCK_PS:g} ps")
    bands = {}
    rows, ok = {}, True
    for n, shells in S1_SHELLS.items():
        A, B = res.get((a.model, n)), res.get((a.truth, n))
        for sh in shells:
            key = f"n{n} ({sh})"
            if A is None or B is None or not (A["enough"] and B["enough"]):
                rows[key] = None
                continue
            d = A["shells"][sh]["L"][0] - B["shells"][sh]["L"][0]
            sig = math.hypot(A["shells"][sh]["L"][1], B["shells"][sh]["L"][1])
            rows[key] = {"model": A["shells"][sh]["L"], "truth": B["shells"][sh]["L"], "difference": d, "sigma": sig,
                         "allowed": BANDS["S1.1"] + 2 * sig, "holds": bool(abs(d) <= BANDS["S1.1"] + 2 * sig)}
    if all(v is None for v in rows.values()):
        bands["S1.1"] = {"read": False}
    else:
        bands["S1.1"] = {"read": True, "rows": rows,
                         "holds": bool(all(v is not None and v["holds"] for v in rows.values()))}
    dg = res.get((a.dagger, 64))
    if dg is not None:
        v = dg["shells"]["100"]["L"]
        bands["S2.1"] = {"read": True, "value": v, "holds": bool(v[0] >= BANDS["S2.1"])}
    for name, b in bands.items():
        if not b.get("read"):
            print(f"  {name}: not read (too few trajectories of {MIN_SAMPLE_PS:g} ps or more)")
            continue
        print(f"  {name}: {'holds' if b['holds'] else 'FAILS'}")
        if name == "S1.1":
            for key, v in b["rows"].items():
                print(f"      {key:12s} " + ("not read" if v is None else
                      f"L {a.model} {pm(v['model'])}, {a.truth} {pm(v['truth'])}: difference {v['difference']:+.3f}, "
                      f"allowed {v['allowed']:.3f} ({'holds' if v['holds'] else 'FAILS'})"))
        else:
            print(f"      n64 (100): the dagger's own L {pm(b['value'])} (band >= {BANDS['S2.1']:g})")
    print("\n  Beside them, not registered")
    for (label, n), rd in res.items():
        tr = rd["trajectories"]
        print(f"  {label}, n {n}: {rd['frames']} frames, {rd['blocks']} blocks; eps_s by the tin-foil formula "
              f"{pm(rd['eps_s_tinfoil'], 1)}; tau_L(100) {rd['tau_L_100_ps']:.3f} ps")
        for s in tr:
            print(f"    seed {s['seed']}: {s['sampled_ps']:.1f} ps sampled, "
                  f"{'completed' if s['completed'] else ('STOPPED: ' + s['stopped'] if s['stopped'] else 'running')}; "
                  f"T {s['T_mean']:.1f} K, conserved-energy drift {s['drift_kcal_per_ps_per_molecule']:+.2e} "
                  f"kcal/mol/ps per molecule, largest constraint error {s['constraint_max']:.1e}")
        for sh, row in rd["shells"].items():
            print(f"    ({sh}) k {row['k']:.4f}: L {pm(row['L'])}, T {pm(row['T'], 2)}, T/L {pm(row['T_over_L'], 1)}; "
                  f"1/eps_s(k) {pm(row['inv_eps_s'])}; 1/A_T {pm(row['inv_A_T'])}, 1/A_L {pm(row['inv_A_L'])}")
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump({"bands": bands, "groups": {f"{l}_n{n}": v for (l, n), v in res.items()},
                       "constants": {"block_ps": BLOCK_PS, "min_sample_ps": a.min_sample_ps}}, fh, indent=1)
        print(f"\nwritten {a.json}")
    return bands, res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "time", "read"):
        s = sub.add_parser(name)
        s.add_argument("--data", default="data/e4")
        s.add_argument("--out", default="results/e4_s1")
    for name in ("run", "time"):
        s = sub.choices[name]
        s.add_argument("--engine", choices=["model", "mbpol"], required=True)
        s.add_argument("--config", default=None)
        s.add_argument("--params", default=None, help="a trained run's .params.pkl (scripts/e4_water.py train)")
        s.add_argument("--rung", default="M_16C2pm_us")
        s.add_argument("--mbx", default="external/mbx/build/MBX")
        s.add_argument("--n", type=int, required=True)
        s.add_argument("--label", default=None)
    s = sub.choices["run"]
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--equil_ps", type=float, default=EQUIL_PS)
    s.add_argument("--sample_ps", type=float, default=SAMPLE_PS)
    s.add_argument("--save_every", type=int, default=SAVE_EVERY)
    s.add_argument("--ckpt_min", type=float, default=10.0)
    s.add_argument("--max_hours", type=float, default=3.7)
    sub.choices["time"].add_argument("--steps", type=int, default=40)
    s = sub.choices["read"]
    s.add_argument("--model", default="Dp")
    s.add_argument("--truth", default="mbpol")
    s.add_argument("--dagger", default="dagger")
    s.add_argument("--min_sample_ps", type=float, default=MIN_SAMPLE_PS)
    s.add_argument("--json", default="results/e4_s1/s1.json")
    a = ap.parse_args()
    if a.cmd in ("run", "time"):
        if a.engine == "model" and not (a.config and a.params):
            raise SystemExit("--engine model needs --config and --params")
        a.label = a.label or a.engine
    {"run": run_run, "time": run_time, "read": run_read}[a.cmd](a)


if __name__ == "__main__":
    main()
