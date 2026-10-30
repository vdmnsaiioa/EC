"""E4's data (notes/e4-plan.md section 3): periodic water cells sampled by rigid-water MB-pol MD and labelled with E4's
truth, MB-pol through MBX (sseft/mbpol.py). Two steps.

    python scripts/e4_data.py md --mbx external/mbx/build/MBX --n_mol 128 --seed 0

One trajectory:
  - rigid monomers at the harness's geometry, randomly packed in a cubic cell at 0.997 g/cm^3;
  - a 1 ps warm-up at 0.5 fs under a tight thermostat (tau 50 fs);
  - then 2 fs steps under the Bussi thermostat at 300 K, tau 100 fs (sseft.rigidmd), with MBX's example PME settings
    (DYNAMICS below);
  - after --equil_ps, one frame every --every_ps (--frames of them).
The packing and the dynamics (velocities, thermostat) draw from two streams of SeedSequence([seed, n_mol]), so no
two trajectories share random numbers.
The frames are labelled at the end with E4's truth: sseft.mbpol's defaults, r_c 6.5 A, accurate PME, narrow cells on
their replica. It writes <out_dir>/traj_n<n>_s<seed>.npz and prints diagnostics:
  - temperature, constraints, energies, g_OO, timings;
  - the conserved energy K + E_pot + the thermostat's heat: its drift checks that the forces are the energy's
    gradient (at 64 molecules it stays within 0.003 kcal/mol per molecule over picoseconds);
  - the labels' size against the dynamics' energies.
A 64-molecule cell's temperature wanders by +-10 K over picoseconds: a canonical fluctuation of its total energy
(standard deviation about 15 kcal/mol), not a fault, as long as the conserved energy holds.

    python scripts/e4_data.py assemble [--dir data/e4] [--train 64 128 256] [--test 512 1024]

The data sets. For a training size, the last seed's frames are split into validation (first half) and test (second
half) and the other seeds are training; every frame of a test size is a test frame. It writes <split>_n<n>.npz
(scripts/e4_data.py's frame format, read by sseft.water.e4_structures) and <split>_n<n>.extxyz (eV and eV/A under
REF_energy / REF_forces, for MACE; the isolated monomer's energy is zero). It first prints one line per trajectory
(temperature, the conserved energy's drift, g_OO) and flags a drift above DRIFT_WARN.

The frame format, one file:
  - positions (F, 3n, 3) A, molecules whole, O inside the cell; cell (3, 3) A;
  - energy (F,) hartree and forces (F, 3n, 3) hartree/bohr, the truth's;
  - md_energy (F,) kcal/mol, the dynamics' potential energy;
  - time_ps (F,); source (F,), the seed;
  - site_q (4,) e and site_alpha (4,) A^3 for the sites O, H, H, M, and site_gamma (the M-site's place on the bisector);
  - log (K, 5): time (ps), T (K), the dynamics' potential energy, the largest constraint error and the conserved energy
    K + E_pot + the thermostat's heat (kcal/mol), every 0.1 ps;
  - meta, a json string of every setting, timing and diagnostic.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from sseft import mbpol as MB, rigidmd as R  # noqa: E402
from sseft.units import KCALMOL_PER_HARTREE, BOHR_PER_ANGSTROM  # noqa: E402

RHO = 0.997                                   # g/cm^3
FRAMES = {64: 60, 128: 30, 256: 15, 512: 15, 1024: 13}         # per trajectory; four seeds per size (run_e4_data.sbatch)
DYNAMICS = dict(alpha_ewald_elec=0.6, alpha_ewald_disp=0.6, grid_density_elec=2.5, grid_density_disp=2.5,
                spline_order_elec=6, spline_order_disp=6, dipole_tolerance=1e-10, replicate=False)
EV_PER_HARTREE = 27.211386245988
DRIFT_WARN = 1e-3            # kcal/mol/ps per molecule: a conserved-energy drift above this is flagged by assemble


def cubic_side(n_mol, rho=RHO):
    """the side (A) of a cubic cell of n_mol waters at density rho (g/cm^3)."""
    return (n_mol * 18.01528 / (rho * 0.6022140857)) ** (1.0 / 3.0)


def site_model(truth):
    """MB-pol's site charges and polarisabilities (O, H, H, M) and the M-site's gamma, read from MBX on the rigid
    monomer (the harness's geometry; the same for every rigid molecule)."""
    m = MB.monomer_ang()
    s = truth.sites(m)
    b = m[1] + m[2] - 2 * m[0]
    gamma = float(2 * np.dot(s["xyz"][3] - m[0], b) / np.dot(b, b))
    return s["charges"].copy(), s["polarizabilities"].copy(), gamma


def g_oo_summary(r, g):
    """the first peak of g_OO (position, height), the first minimum after it, and the coordination number to it."""
    i = int(np.argmax(g))
    j = i + int(np.argmin(g[i:i + int(1.5 / (r[1] - r[0]))]))
    return dict(r_peak=float(r[i]), g_peak=float(g[i]), r_min=float(r[j]), g_min=float(g[j]))


def run_md(mbx, n_mol, seed, frames=None, every_ps=0.5, equil_ps=10.0, warmup_ps=1.0, dt=2.0, tau_fs=100.0,
           warm_dt=0.5, warm_tau_fs=50.0, T_K=300.0, rho=RHO, out_dir=None, log_ps=0.1, verbose=True):
    """one trajectory and its labels; returns the frame dict (and writes it when out_dir is given)."""
    say = print if verbose else (lambda *a, **k: None)
    frames = FRAMES.get(n_mol, 15) if frames is None else frames
    L = cubic_side(n_mol, rho)
    cell = np.eye(3) * L
    pack_seq, md_seq = np.random.SeedSequence([seed, n_mol]).spawn(2)      # separate streams: packing, dynamics
    t0 = time.time()
    pos0 = MB.random_box(n_mol, cell, np.random.default_rng(pack_seq), min_oo=2.5, min_any=1.5)
    say(f"{n_mol} molecules in a {L:.4f} A cell ({rho} g/cm^3), seed {seed}: packed in {time.time() - t0:.1f} s")

    dyn = MB.MBPol(mbx, settings=DYNAMICS)

    def force(p, c):
        e, gr = dyn.energy_gradient_kcal(p, c)
        return e, -gr

    md = R.RigidWaterMD(pos0, cell, force, dt_fs=warm_dt, T_K=T_K, tau_fs=warm_tau_fs, seed=md_seq)
    md.init_velocities()
    log = []

    def rec(m):
        log.append((m.time_fs / 1000.0, m.temperature(), m.E_pot, m.constraint_error(), m.conserved()))
    rec(md)
    timings = {}
    t0 = time.time()
    md.run(int(round(warmup_ps * 1000 / warm_dt)), every=max(1, int(round(log_ps * 1000 / warm_dt))), callback=rec)
    timings["warmup_s"] = time.time() - t0
    say(f"warm-up {warmup_ps} ps at {warm_dt} fs (tau {warm_tau_fs} fs): T {log[0][1]:.0f} -> {log[-1][1]:.0f} K, "
        f"E_pot {log[0][2] / n_mol:.3f} -> {log[-1][2] / n_mol:.3f} kcal/mol per molecule, {timings['warmup_s']:.0f} s")

    md.dt, md.tau = dt, tau_fs
    n_warm = len(log)
    every_log = max(1, int(round(log_ps * 1000 / dt)))
    t0 = time.time()
    md.run(int(round(equil_ps * 1000 / dt)), every=every_log, callback=rec)
    timings["equil_s"] = time.time() - t0
    n_eq = len(log)
    say(f"equilibration {equil_ps} ps at {dt} fs (tau {tau_fs} fs): T {log[-1][1]:.0f} K, E_pot "
        f"{log[-1][2] / n_mol:.3f} kcal/mol per molecule, {timings['equil_s']:.0f} s "
        f"({timings['equil_s'] / max(1, round(equil_ps * 1000 / dt)):.3f} s per step)")

    X, E_md, t_ps = [], [], []
    t0 = time.time()
    steps = int(round(every_ps * 1000 / dt))
    for f in range(frames):
        md.run(steps, every=every_log, callback=rec)
        X.append(md.positions()); E_md.append(md.E_pot); t_ps.append(md.time_fs / 1000.0)
    timings["sample_s"] = time.time() - t0
    X = np.array(X)
    prod = np.array(log[n_eq:])
    # the conserved energy over the steps at dt: its drift is the check that the forces are the energy's gradient
    ec = np.array(log[n_warm - 1:])
    ec_drift = float(np.polyfit(ec[:, 0], ec[:, 4], 1)[0] / n_mol)
    ec_span = float((ec[:, 4].max() - ec[:, 4].min()) / n_mol)
    say(f"sampling {frames} frames every {every_ps} ps: T {prod[:, 1].mean():.1f} +- {prod[:, 1].std():.1f} K, E_pot "
        f"{prod[:, 2].mean() / n_mol:.3f} +- {prod[:, 2].std() / n_mol:.3f} kcal/mol per molecule, largest constraint "
        f"error {max(r[3] for r in log):.1e}, {timings['sample_s']:.0f} s")
    say(f"conserved energy at {dt} fs: drift {ec_drift:+.5f} kcal/mol/ps per molecule, range {ec_span:.4f} kcal/mol "
        f"per molecule over {ec[-1, 0] - ec[0, 0]:.1f} ps")

    truth = MB.MBPol(mbx)                                     # E4's truth: the driver's defaults
    t0 = time.time()
    E, F = [], []
    for p in X:
        e, fo = truth.energy_forces(p, cell)
        E.append(e); F.append(fo)
    E, F = np.array(E), np.array(F)
    timings["label_s"] = time.time() - t0
    sq, sa, gamma = site_model(truth)
    truth.finalize()

    r, g = R.rdf_oo(X, cell, dr=0.05)
    gs = g_oo_summary(r, g)
    E_kcal = E * KCALMOL_PER_HARTREE
    diff = E_kcal - np.array(E_md)
    F_rms = float(np.sqrt(np.mean((F * KCALMOL_PER_HARTREE * BOHR_PER_ANGSTROM) ** 2)))
    diag = dict(T_mean=float(prod[:, 1].mean()), T_std=float(prod[:, 1].std()),
                conserved_drift_per_mol_per_ps=ec_drift, conserved_range_per_mol=ec_span,
                E_md_per_mol=float(prod[:, 2].mean() / n_mol), constraint_max=float(max(r_[3] for r_ in log)),
                E_truth_per_mol=float(E_kcal.mean() / n_mol), E_truth_std_per_mol=float(E_kcal.std() / n_mol),
                truth_minus_md_per_mol_mean=float(diff.mean() / n_mol), truth_minus_md_per_mol_std=float(diff.std() / n_mol),
                F_rms_kcal_per_A=F_rms, g_oo=gs)
    say(f"labels (E4's truth) in {timings['label_s']:.0f} s: E {diag['E_truth_per_mol']:.4f} +- "
        f"{diag['E_truth_std_per_mol']:.4f} kcal/mol per molecule; truth - dynamics {diag['truth_minus_md_per_mol_mean']:+.4f} "
        f"+- {diag['truth_minus_md_per_mol_std']:.4f} per molecule; force rms {F_rms:.2f} kcal/mol/A")
    say(f"g_OO: first peak {gs['g_peak']:.2f} at {gs['r_peak']:.2f} A, first minimum {gs['g_min']:.2f} at "
        f"{gs['r_min']:.2f} A")

    meta = dict(n_mol=n_mol, seed=seed, rng="SeedSequence([seed, n_mol]).spawn(2): packing, dynamics", rho=rho, L=L,
                T_K=T_K, dt_fs=dt, tau_fs=tau_fs, warmup_ps=warmup_ps,
                warm_dt_fs=warm_dt, warm_tau_fs=warm_tau_fs, equil_ps=equil_ps, every_ps=every_ps, frames=frames,
                dynamics_settings=dict(dyn.settings), truth=truth.describe(), masses=R.MASSES.tolist(),
                timings=timings, diagnostics=diag)
    try:
        import sseft
        meta["sseft_version"] = sseft.__version__
    except Exception:
        pass
    out = dict(positions=X, cell=cell, energy=E, forces=F, md_energy=np.array(E_md), time_ps=np.array(t_ps),
               source=np.full(frames, seed), site_q=sq, site_alpha=sa, site_gamma=np.array(gamma),
               log=np.array(log), g_oo=np.stack([r, g]), meta=np.array(json.dumps(meta)))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"traj_n{n_mol}_s{seed}.npz")
        np.savez(path, **out)
        say(f"written {path}")
    return out


def write_extxyz(path, positions, cell, energy, forces, info=None):
    """frames as extended XYZ for MACE: species, positions (A), REF_forces (eV/A), REF_energy (eV), the lattice,
    pbc, and per-frame info (dict of lists)."""
    pos = np.asarray(positions, float)
    F, N = pos.shape[0], pos.shape[1]
    sym = np.tile(["O", "H", "H"], N // 3)
    cells = np.asarray(cell, float)
    cells = cells if cells.ndim == 3 else np.broadcast_to(cells, (F, 3, 3))
    E = np.asarray(energy, float) * EV_PER_HARTREE
    Fo = np.asarray(forces, float) * EV_PER_HARTREE * BOHR_PER_ANGSTROM
    info = info or {}
    with open(path, "w") as fh:
        for f in range(F):
            lat = " ".join(f"{x:.10f}" for x in cells[f].reshape(-1))
            extra = "".join(f" {k}={v[f]}" for k, v in info.items())
            fh.write(f"{N}\nLattice=\"{lat}\" Properties=species:S:1:pos:R:3:REF_forces:R:3 "
                     f"REF_energy={E[f]:.12f} pbc=\"T T T\"{extra}\n")
            for a in range(N):
                x, y, z = pos[f, a]; fx, fy, fz = Fo[f, a]
                fh.write(f"{sym[a]} {x:.10f} {y:.10f} {z:.10f} {fx:.12e} {fy:.12e} {fz:.12e}\n")


def assemble(dir_, train=(64, 128, 256), test=(512, 1024), verbose=True):
    """the data sets from the trajectories in dir_ (see the module docstring); returns {(split, n): n_frames}."""
    say = print if verbose else (lambda *a, **k: None)
    by_n = {}
    for p in sorted(glob.glob(os.path.join(dir_, "traj_n*_s*.npz"))):
        d = dict(np.load(p, allow_pickle=False))
        n = int(d["positions"].shape[1] // 3)
        by_n.setdefault(n, []).append((int(d["source"][0]), p, d))
    for n in sorted(by_n):                                   # one line per trajectory, from its diagnostics
        for seed, p, d in sorted(by_n[n], key=lambda t: t[0]):
            g = json.loads(str(d["meta"])).get("diagnostics", {})
            if not g:
                continue
            drift = g.get("conserved_drift_per_mol_per_ps")
            flag = "  <- check its log" if drift is not None and abs(drift) > DRIFT_WARN else ""
            say(f"n {n:5d} seed {seed}: {len(d['energy'])} frames, T {g['T_mean']:.1f} +- {g['T_std']:.1f} K, conserved "
                f"energy drift {'n/a' if drift is None else f'{drift:+.1e}'} kcal/mol/ps per molecule, g_OO peak "
                f"{g['g_oo']['g_peak']:.2f} at {g['g_oo']['r_peak']:.2f} A{flag}")
    written = {}
    keys = ("positions", "energy", "forces", "md_energy", "time_ps", "source")
    for n in sorted(by_n):
        if n not in train and n not in test:
            say(f"n = {n}: not asked for, skipped")
            continue
        runs = sorted(by_n[n], key=lambda t: t[0])
        ref = runs[0][2]
        for _, p, d in runs[1:]:
            if not (np.allclose(d["cell"], ref["cell"]) and np.allclose(d["site_q"], ref["site_q"])):
                raise ValueError(f"{p}: a different cell or site model from the other trajectories of {n} molecules")
        parts = {}
        if n in train:
            if len(runs) < 2:
                raise ValueError(f"{n} molecules: a training size needs at least two trajectories (the last one is "
                                 "the validation and test frames)")
            parts["train"] = [(d, slice(None)) for _, _, d in runs[:-1]]
            last = runs[-1][2]
            h = len(last["energy"]) // 2
            parts["valid"] = [(last, slice(0, h))]
            parts["test"] = [(last, slice(h, None))]
        else:
            parts["test"] = [(d, slice(None)) for _, _, d in runs]
        for split, items in parts.items():
            cat = {k: np.concatenate([d[k][sl] for d, sl in items]) for k in keys}
            meta = [json.loads(str(d["meta"])) for d, _ in items]
            out = dict(cat, cell=ref["cell"], site_q=ref["site_q"], site_alpha=ref["site_alpha"],
                       site_gamma=ref["site_gamma"], meta=np.array(json.dumps({"split": split, "n_mol": n,
                                                                              "trajectories": meta})))
            base = os.path.join(dir_, f"{split}_n{n}")
            np.savez(base + ".npz", **out)
            write_extxyz(base + ".extxyz", cat["positions"], ref["cell"], cat["energy"], cat["forces"],
                         info={"config_type": [f"{split}_n{n}"] * len(cat["energy"]),
                               "source": cat["source"].tolist(), "time_ps": [f"{t:.3f}" for t in cat["time_ps"]]})
            e = cat["energy"] * KCALMOL_PER_HARTREE / n
            written[(split, n)] = len(e)
            say(f"{split}_n{n}: {len(e)} frames from seeds {sorted(set(cat['source'].tolist()))}, E {e.mean():.4f} +- "
                f"{e.std():.4f} kcal/mol per molecule")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("md", help="one trajectory and its labels")
    m.add_argument("--mbx", required=True, help="the MBX build (MBX_HOME), e.g. external/mbx/build/MBX")
    m.add_argument("--n_mol", type=int, required=True)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--frames", type=int, default=None, help=f"frames to keep (default by size: {FRAMES})")
    m.add_argument("--every_ps", type=float, default=0.5)
    m.add_argument("--equil_ps", type=float, default=10.0)
    m.add_argument("--warmup_ps", type=float, default=1.0)
    m.add_argument("--dt", type=float, default=2.0, help="fs")
    m.add_argument("--tau", type=float, default=100.0, help="Bussi time constant, fs")
    m.add_argument("--T", type=float, default=300.0)
    m.add_argument("--rho", type=float, default=RHO)
    m.add_argument("--out_dir", default=os.path.join(REPO, "data", "e4"))
    a_ = sub.add_parser("assemble", help="the data sets from the trajectories")
    a_.add_argument("--dir", default=os.path.join(REPO, "data", "e4"))
    a_.add_argument("--train", type=int, nargs="*", default=[64, 128, 256])
    a_.add_argument("--test", type=int, nargs="*", default=[512, 1024])
    a = ap.parse_args()
    if a.cmd == "md":
        print(f"threads: {os.environ.get('OMP_NUM_THREADS', 'unset')}")
        run_md(a.mbx, a.n_mol, a.seed, frames=a.frames, every_ps=a.every_ps, equil_ps=a.equil_ps,
               warmup_ps=a.warmup_ps, dt=a.dt, tau_fs=a.tau, T_K=a.T, rho=a.rho, out_dir=a.out_dir)
    else:
        assemble(a.dir, train=tuple(a.train), test=tuple(a.test))


if __name__ == "__main__":
    main()
