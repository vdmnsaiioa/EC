"""
e4_water.py -- E4's training driver: the ladder on MB-pol's periodic water cells of one size L0 (notes/e4-plan.md
sections 4 and 9; the pre-registrations are notes/e4-x3-preregistration.md and, for job 2,
notes/e4-job2-preregistration.md).  Four steps:

    python scripts/e4_water.py probes --mbx external/mbx/build/MBX [--config configs/e4_water.yaml]
    python scripts/e4_water.py response --mbx external/mbx/build/MBX [--config configs/e4_job2b.yaml]
    python scripts/e4_water.py train --config configs/e4_water.yaml [--rungs M_16C2pm_us] [--seeds 0] [--task 5]
    python scripts/e4_water.py summary --out_dir results/e4_job1

probes   MB-pol's far-field probes through MBX, E3's recipe (sseft.water.probe_configurations): Metropolis-relaxed cores
         of 3-5 molecules and one molecule at 8-15 A from the core's centre, with each core's own energy and forces and,
         for the test probes, the truth's probe interaction and its non-additive part (less the probe's dimers).  The
         read-out is matched on the training probes whose probe the core's E_0 cannot see; the test probes read the far
         field.  Deterministic in its seeds: the file's hash is printed, and the train step prints it again.
response (job 2, v0.0.24) MB-pol's induced dipoles in liquid cells of the training size, clamped nuclei, and their
         response to a uniform external field, every molecule's (MBX's site dipoles summed): calibration frames spread
         over the training frames, held-out frames over the test frames (liquid.calibration_frames, heldout_frames).
         The train step reads eps_inf and the molecules' dipoles and response on the held-out frames of every rung with
         a read-out, and with training.readout.liquid matches the read-out on the calibration frames' cell sums too.
train    for every rung and seed (or one, --task i: rung i // n_seeds, seed i % n_seeds, for an array job):
           1. a rung with a read-out has it matched on the training probes, open, before training (E3-QP2's matching:
              every order, Gauss-Newton steps, the ridge chosen on held-out cores, energies only).  It depends on neither
              E_0 nor the seed (the species read-out's initialisation is fixed; E_0 sees nothing of a probe beyond r_c),
              so it is solved once per rung; with training.readout.liquid (job 2's arm B) a second stage adds the
              calibration cells' induced dipole and field response as rows, the probes' ridge kept (match_readout);
           2. the rung is trained on the cells of one size (Adam, then an L-BFGS polish; the read-out held), on the
              energies and on the forces' rigid-body part (sseft.train.rigid_force_sums: each molecule's net force and
              torque; the internal components carry the flux of MB-pol's geometry-dependent charges, which rigid pinned
              charges cannot represent and whose long-range part no local model can);
           3. it is read: energy per molecule and forces on the training, validation and test frames of every size in
              data.eval_sizes, the far-field test probes, and eps_inf on test frames of the training size (rungs with a
              response; the others have eps_inf = 1 by construction).
         One json and one parameter file per rung and seed in out_dir.
summary  seed means and spreads of every rung's readings, from the jsons in out_dir.

Energies are interaction energies (MBX's 1-body term off, every molecule's own energy zero), in hartree inside and in
kcal/mol in the readings; forces in kcal/mol/A.
"""
import argparse
import copy
import glob
import hashlib
import json
import math
import os
import pickle
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import jax                                    # noqa: E402
import jax.numpy as jnp                       # noqa: E402
import sseft                                  # noqa: E402
from sseft import model as M, train as T, water as W, periodic as PB, structure as S, linear as LIN  # noqa: E402
from sseft.units import ang_to_bohr, bohr_to_ang, KCALMOL_PER_HARTREE, BOHR_PER_ANGSTROM  # noqa: E402
import e3_water as E3                         # E3's rungs, their configuration and the probe readings  # noqa: E402

KCAL = KCALMOL_PER_HARTREE
FORCE_UNIT = KCALMOL_PER_HARTREE * BOHR_PER_ANGSTROM          # hartree/bohr -> kcal/mol/A
SPLITS = ("train", "valid", "test")
# job 2's band-edge sweep reads the frozen form at read-out order 0 beside order 2 (notes/e4-x3-preregistration.md)
E3.RUNGS.setdefault("M_16C0pm_us", M.with_rung(E3.RUNGS["M_16C2pm_us"], name="M_16C0pm_us", band_fields=0))
# job 2 (notes/e4-job2-preregistration.md): the dagger with the frozen form's field term (its self terms' reach and its
# cross terms by species) and no response -- the frozen form less its read-out, so that frozen against it is the response
E3.RUNGS.setdefault("M_16pm_us", M.with_rung(E3.RUNGS["M_16pm_w"], name="M_16pm_us", e0_field_self_on=ang_to_bohr(4.0),
                                             e0_field_self_off=ang_to_bohr(5.0), e0_field_cross_species=True))

DEFAULTS = {
    "data": {"dir": "data/e4", "train_size": 64, "eval_sizes": [64, 128, 256], "probes": "data/e4/probes_mbpol.npz",
             "response": "data/e4/response_mbpol.npz"},
    "liquid": {"calibration_frames": 30, "heldout_frames": 10, "delta": 1.0e-4},
    "model": {"r_cut_angstrom": 6.0, "features": 32, "band_edge_angstrom": 0.75, "response_edge_angstrom": 0.75,
              "mutual_edge_angstrom": None, "analytic_envelope_angstrom": 4.0, "rs_over_l1": 0.0,
              "charge_neutrality": "fragment", "ewald_b_max": 4.0, "solver": "gmres", "solver_tol": 1.0e-10, "K": 8,
              "omega0": 0.3},
    "rungs": ["M_inf", "M_1", "M_16", "M_16pm", "M_16C2pm", "M_16C2pm_us", "M_G"],
    "seeds": [0, 1, 2, 3],
    "training": {
        "adam": {"steps": 3000, "batch_size": 6, "learning_rate": 3.0e-3, "final_learning_rate_ratio": 0.01,
                 "beta1": 0.9, "beta2": 0.999, "epsilon": 1.0e-8},
        "lbfgs": {"steps": 50, "chunk_size": 12, "memory_size": 10, "max_linesearch_steps": 20},
        "loss": {"energy_weight": 100.0, "force_weight": 1.0, "forces": "rigid"},
        "readout": {"orders": "all", "solve_iterations": 4, "ridge": 1.0e-8, "ridge_selection": "groups",
                    "solve_chunk": 64, "liquid": None},
    },
    "probes": {"distances_angstrom": [8.0, 10.0, 12.0, 15.0], "train_per_distance": 2, "train_cores": 16,
               "train_core_sizes": [3, 4, 5], "test_per_distance": 10, "test_cores": 2, "seed": 400, "mc_steps": 300,
               "temperature_kelvin": 300.0, "min_oo_angstrom": 2.7, "min_atom_distance_angstrom": 1.7,
               "far_trimers": 40, "trimer_seed": 900, "trimer_min_angstrom": 6.5, "trimer_max_oo_angstrom": 12.0},
    "reading": {"eps_frames": 10, "splits": ["train", "valid", "test"]},
}


# ---------------------------------------------------------------- configuration -------------------------------------
def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path=None):
    if path is None:
        return copy.deepcopy(DEFAULTS)
    import yaml
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    unknown = set(cfg) - set(DEFAULTS) - {"experiment"}
    if unknown:
        raise ValueError(f"unknown sections in {path}: {sorted(unknown)}")
    return merge(DEFAULTS, {k: v for k, v in cfg.items() if k != "experiment"})


def e3_args(cfg):
    """the namespace E3's configure() reads: E_0's cutoff and width, the band and response edges, the neutrality."""
    m = cfg["model"]
    return SimpleNamespace(lA=m["analytic_envelope_angstrom"], rcut=m["r_cut_angstrom"], F=m["features"], charges=None,
                           charge_neutrality=m["charge_neutrality"], l1=m["band_edge_angstrom"],
                           rs_over_l1=m["rs_over_l1"], response_edge=m["response_edge_angstrom"])


def rungs_for(name, cfg, cell_bohr):
    """(the rung in a cell, with its k-grid, and the same rung open): E3's rung (E3-QP8's settings, so that
    M_16C2pm_us is E3's frozen form) made periodic, with the solver of the mutual response and the cell's n_max."""
    if name not in E3.RUNGS:
        raise ValueError(f"unknown rung {name}; E3's rungs: {sorted(E3.RUNGS)}")
    m = cfg["model"]
    base = E3.configure(E3.RUNGS[name], e3_args(cfg))
    base = replace(base, K=int(m["K"]), omega0=float(m["omega0"]))
    if m.get("mutual_edge_angstrom") and base.band_fields >= 0 and base.bf_mutual:
        base = replace(base, bf_l_mutual=ang_to_bohr(float(m["mutual_edge_angstrom"])))   # v0.0.23, Rung.bf_l_mutual
    cell = replace(base, periodic=True, ewald_b_max=float(m["ewald_b_max"]), bf_solver=m["solver"],
                   bf_tol=float(m["solver_tol"]), ewald_n_max=PB.n_max_for(np.asarray(cell_bohr), base.l1, m["ewald_b_max"]))
    return cell, replace(cell, periodic=False)


def in_cell(rung, cell_bohr):
    return replace(rung, ewald_n_max=PB.n_max_for(np.asarray(cell_bohr), rung.l1, rung.ewald_b_max))


def model_record(name, cfg, rung):
    rec = E3.model_record(name, e3_args(cfg))
    rec.update(periodic=True, ewald_b_max=float(rung.ewald_b_max), solver=rung.bf_solver if rung.bf_mutual else None,
               K=int(rung.K), omega0=float(rung.omega0))
    if rung.bf_l_mutual > 0:
        rec["mutual_edge_angstrom"] = round(float(bohr_to_ang(rung.bf_l_mutual)), 12)
    return rec


# ---------------------------------------------------------------- the probes ----------------------------------------
def make_probes(truth, cfg):
    """MB-pol's far-field probes, E3's recipe and seeds (configs/water_qaqua_pol8.yaml's probe settings)."""
    p = cfg["probes"]
    geom = dict(T_K=p["temperature_kelvin"], radius_ang=None, min_OO=p["min_oo_angstrom"],
                min_any=p["min_atom_distance_angstrom"])
    out = {}
    for split, seed, n_cores, count, sizes in (
            ("train", p["seed"], p["train_cores"], p["train_per_distance"], p["train_core_sizes"]),
            ("test", p["seed"] + 100, p["test_cores"], p["test_per_distance"], [3])):
        structs = []
        for k, n_core in enumerate(sizes):
            t0 = time.time()
            made = W.probe_configurations(n_core, p["distances_angstrom"], count, seed=seed + 1000 * k,
                                          mc_steps=p["mc_steps"], n_cores=n_cores, truth=truth, **geom)[0]
            for st in made:
                st.info.update(n_core=n_core, core_id=1000 * k + st.info["core"])
            structs += made
            print(f"  {split}: {len(made)} probes on {n_cores} cores of {n_core} molecules [{time.time() - t0:.0f} s]",
                  flush=True)
        out[split] = structs
    return out


def far_trimers(truth, p):
    """Theorem 2's T5 (les-vs-eft-what-is-provable.md section 7): three rigid molecules at random orientations whose
    every intermolecular atom pair lies beyond trimer_min_angstrom (> r_c), O-O at most trimer_max_oo_angstrom, with the
    truth's three-body energy E(ABC) - E(AB) - E(AC) - E(BC) -- MB-pol's 3-body polynomial is off there, so it is the
    many-body induction alone.  A model whose sources do not respond to a far field (fixed or environment-dependent
    charges, MACE-LES, the dagger) gives exactly zero: every molecule's environment within r_c is itself."""
    rng = np.random.default_rng(p["trimer_seed"])
    mono = W.monomer_positions() - W.monomer_positions().mean(axis=0)
    hi, lo = float(p["trimer_max_oo_angstrom"]), float(p["trimer_min_angstrom"])
    pos_all, e3, pairs = [], [], []
    while len(pos_all) < int(p["far_trimers"]):
        cen = rng.uniform(-hi / 2, hi / 2, size=(3, 3))
        pos = np.concatenate([c + mono @ W.random_rotation(rng).T for c in cen])
        d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        mol = np.repeat(np.arange(3), 3)
        inter = mol[:, None] != mol[None, :]
        if d[inter].min() <= lo or d[0::3, 0::3].max() > hi:
            continue
        E = truth.energy(pos, 3)
        Ep = [truth.energy(np.concatenate([pos[3 * i:3 * i + 3], pos[3 * j:3 * j + 3]]), 2)
              for i, j in ((0, 1), (0, 2), (1, 2))]
        pos_all.append(pos); pairs.append(Ep); e3.append(E - sum(Ep))
    return np.array(pos_all), np.array(pairs), np.array(e3)


def probes_hash(arrays):
    h = hashlib.sha256()
    for split in ("train", "test"):
        h.update(np.round(np.asarray(arrays[f"{split}_positions"], float), 9).tobytes())
    return h.hexdigest()[:16]


def save_probes(path, probes, truth, cfg):
    arrays = {}
    for split, structs in probes.items():
        F_core, pair_sum, cache = [], [], {}
        for st in structs:
            pos = bohr_to_ang(st.positions); n = len(pos) // 3; core = pos[:3 * (n - 1)]
            key = core.tobytes()
            if key not in cache:
                cache[key] = truth.energy_forces(core, n - 1)
            E_c, F_c = cache[key]
            if abs(E_c - st.info["E_core"]) > 1e-9:
                raise RuntimeError("a probe's core energy does not reproduce")
            F_core.append(np.asarray(F_c, float))
            if split == "test":
                pair_sum.append(sum(truth.energy(np.concatenate([pos[3 * i:3 * i + 3], pos[-3:]]), 2) for i in range(n - 1)))
        n_atoms = np.array([st.n_atoms for st in structs])
        arrays[f"{split}_offsets"] = np.concatenate([[0], np.cumsum(n_atoms)])
        arrays[f"{split}_positions"] = np.concatenate([bohr_to_ang(st.positions) for st in structs])     # A
        arrays[f"{split}_energy"] = np.array([st.energy for st in structs])                               # hartree
        arrays[f"{split}_forces"] = np.concatenate([np.asarray(st.forces, float) for st in structs])    # hartree/bohr
        arrays[f"{split}_E_core"] = np.array([st.info["E_core"] for st in structs])
        arrays[f"{split}_F_core"] = np.concatenate(F_core)
        for key in ("d", "core", "core_id", "n_core", "k"):
            arrays[f"{split}_{key}"] = np.array([st.info[key] for st in structs])
        if split == "test":
            arrays["test_pair_energy"] = np.array(pair_sum)
    arrays["site_q"] = np.asarray(truth.site_charges, float)
    arrays["site_gamma"] = np.array(float(truth.gamma_m))
    digest = probes_hash(arrays)
    trimers = {}
    if int(cfg["probes"].get("far_trimers", 0)) > 0:
        t0 = time.time()
        arrays["trimer_positions"], arrays["trimer_pairs"], arrays["trimer_three_body"] = far_trimers(truth, cfg["probes"])
        trimers = {"trimers_hash": hashlib.sha256(np.round(arrays["trimer_positions"], 9).tobytes()).hexdigest()[:16]}
        print(f"  far trimers: {len(arrays['trimer_three_body'])}, three-body rms "
              f"{np.sqrt(np.mean(arrays['trimer_three_body'] ** 2)) * KCAL:.5f} kcal/mol [{time.time() - t0:.0f} s]",
              flush=True)
    arrays["meta"] = np.array(json.dumps({"probes": cfg["probes"], "truth": truth.describe(), "hash": digest,
                                          "sseft": sseft.__version__, **trimers}))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **arrays)
    return digest


def load_probes(path, K=8, omega0=0.3):
    """the probes as Structures: {"train": interaction-labelled training probes (E - E_core, F - F_core, the core and
    the probe as subtraction groups, so that the model's energy is the same difference), "test": the test probes with
    their totals, "test_ref": the truth's readings (d, interaction, nonadditive), "hash", "meta"}, every structure
    with MB-pol's rigid site model and dispersion pins."""
    d = np.load(path, allow_pickle=False)
    out = {"hash": probes_hash(d), "meta": json.loads(str(d["meta"]))}
    for split in ("train", "test"):
        off = d[f"{split}_offsets"]
        structs = []
        for i in range(len(off) - 1):
            pos, F = d[f"{split}_positions"][off[i]:off[i + 1]], d[f"{split}_forces"][off[i]:off[i + 1]]
            n = len(pos) // 3
            info = {k: d[f"{split}_{k}"][i].item() for k in ("d", "core", "core_id", "n_core", "k")}
            info["E_core"] = float(d[f"{split}_E_core"][i])
            if split == "train":
                f_core = d["train_F_core"][off[i] - 3 * i:off[i + 1] - 3 * (i + 1)]
                F = np.array(F, float); F[:3 * (n - 1)] -= f_core
                st = W.water_structure(pos, energy=float(d[f"{split}_energy"][i]) - info["E_core"], forces=F,
                                       pinned_q=False, info={**info, "label": "interaction"})
                st.sub = np.repeat([0, 1], [3 * (n - 1), 3])
            else:
                st = W.water_structure(pos, energy=float(d[f"{split}_energy"][i]), forces=np.array(F, float),
                                       pinned_q=False, info=info)
            structs.append(st)
        W.pin_rigid_site_model(structs, d["site_q"], float(d["site_gamma"]), K, omega0)
        out[split] = structs
    inter = d["test_energy"] - d["test_E_core"]
    out["test_ref"] = {"d": d["test_d"].astype(float), "interaction": inter, "nonadditive": inter - d["test_pair_energy"]}
    out["site_q"], out["site_gamma"] = d["site_q"].copy(), float(d["site_gamma"])
    if "trimer_positions" in d.files:
        tri = [W.water_structure(pos, energy=None, pinned_q=False) for pos in d["trimer_positions"]]
        pairs = [W.water_structure(np.concatenate([pos[3 * i:3 * i + 3], pos[3 * j:3 * j + 3]]), pinned_q=False)
                 for pos in d["trimer_positions"] for i, j in ((0, 1), (0, 2), (1, 2))]
        W.pin_rigid_site_model(tri + pairs, d["site_q"], float(d["site_gamma"]), K, omega0)
        out["trimers"] = {"structures": tri, "pairs": pairs, "three_body": d["trimer_three_body"].copy()}
    return out


def run_probes(a):
    cfg = load_config(a.config)
    path = a.out or cfg["data"]["probes"]
    truth = W.MBPolTruth(a.mbx)
    print(f"MB-pol through MBX at {a.mbx}: {json.dumps(truth.describe())}", flush=True)
    t0 = time.time()
    probes = make_probes(truth, cfg)
    digest = save_probes(path, probes, truth, cfg)
    loaded = load_probes(path)
    ref = loaded["test_ref"]
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    print(f"written {path}: {len(loaded['train'])} training probes, {len(loaded['test'])} test probes, hash {digest} "
          f"[{time.time() - t0:.0f} s]", flush=True)
    for dist in np.unique(ref["d"]):
        sel = ref["d"] == dist
        print(f"  test probes at {dist:g} A: interaction rms {rms(ref['interaction'][sel]) * KCAL:.4f} kcal/mol, "
              f"non-additive rms {rms(ref['nonadditive'][sel]) * KCAL:.5f}", flush=True)
    import e4_far                                   # the same structures as extxyz, for the baselines (NumPy only)
    print(f"  the far-field test structures for any model: {e4_far.write_extxyz(path)} (scripts/e4_far.py)", flush=True)


# ---------------------------------------------------------------- the liquid's response (job 2) ---------------------
def liquid_response(mb, pos_ang, cell_ang, delta=1e-4):
    """MB-pol's induced dipoles in one periodic frame, clamped nuclei (MBX): every molecule's at zero external field
    (n, 3) e A and their response to a uniform field (n, 3, 3) A^3, by central differences of field delta e/A^2 (the
    response is linear: the step only sets the rounding).  Molecules in the frame's order, their sites O, H, H, M summed."""
    n = len(pos_ang) // 3
    mol = lambda: mb.sites(pos_ang, cell_ang)["induced"].reshape(n, 4, 3).sum(axis=1)
    mb.set_external_field(pos_ang, None, cell_ang)
    mu0 = mol()
    R = np.zeros((n, 3, 3))
    for k in range(3):
        d = np.zeros(3); d[k] = delta
        mb.set_external_field(pos_ang, d, cell_ang); plus = mol()
        mb.set_external_field(pos_ang, -d, cell_ang); minus = mol()
        R[:, :, k] = (plus - minus) / (2 * delta)
    mb.set_external_field(pos_ang, None, cell_ang)
    return mu0, R


def response_frames(cfg):
    """the frames of job 2's liquid response: calibration frames spread evenly over the training frames of the training
    size (every trajectory), held-out frames spread over its test frames; {split: (path, indices)}."""
    dcfg, lq = cfg["data"], cfg["liquid"]
    L0 = int(dcfg["train_size"])
    out = {}
    for split, key, count in (("calibration", "train", lq["calibration_frames"]), ("heldout", "test", lq["heldout_frames"])):
        path = os.path.join(dcfg["dir"], f"{key}_n{L0}.npz")
        n = len(np.load(path)["energy"])
        out[split] = (path, np.unique(np.linspace(0, n - 1, min(int(count), n)).round().astype(int)))
    return out


def frames_hash(positions):
    return hashlib.sha256(np.ascontiguousarray(np.round(np.asarray(positions, float), 8)).tobytes()).hexdigest()[:16]


def run_response(a):
    cfg = load_config(a.config)
    path = a.out or cfg["data"]["response"]
    truth = W.MBPolTruth(a.mbx)
    print(f"MB-pol through MBX at {a.mbx}: {json.dumps(truth.describe())}", flush=True)
    delta = float(cfg["liquid"]["delta"])
    arrays, meta = {}, {"truth": truth.describe(), "delta": delta, "sseft": sseft.__version__, "splits": {}}
    for split, (fpath, idx) in response_frames(cfg).items():
        d = np.load(fpath)
        pos, cell = d["positions"][idx], (d["cell"] if d["cell"].ndim == 2 else d["cell"][idx[0]])
        mus, Rs = [], []
        t0 = time.time()
        for k, p in enumerate(pos):
            mu, R = liquid_response(truth.mb, p, cell, delta)
            mus.append(mu); Rs.append(R)
            V = abs(np.linalg.det(cell))
            print(f"  {split} frame {idx[k]}: eps_inf {1 + 4 * np.pi * np.trace(R.sum(0)) / 3 / V:.4f}, molecular "
                  f"induced dipole rms {np.sqrt(np.mean(np.sum(mu ** 2, -1))) * 4.80320471:.3f} D  "
                  f"[{time.time() - t0:.0f} s]", flush=True)
        arrays[f"{split}_index"] = idx
        arrays[f"{split}_mu"] = np.array(mus)
        arrays[f"{split}_R"] = np.array(Rs)
        meta["splits"][split] = {"file": fpath, "frames": int(len(idx)), "positions_hash": frames_hash(pos)}
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez(path, **arrays)
    print(f"written {path}", flush=True)


def load_response(path, cfg):
    """job 2's liquid response file, checked against the data sets' frames: {split: {"structures", "mu", "R"}} in
    atomic units (e bohr, bohr^3), and its hash."""
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    K, om = int(cfg["model"]["K"]), float(cfg["model"]["omega0"])
    out = {"hash": hashlib.sha256(open(path, "rb").read()).hexdigest()[:8], "meta": meta}
    for split, info in meta["splits"].items():
        idx = d[f"{split}_index"]
        data = np.load(info["file"])
        if frames_hash(data["positions"][idx]) != info["positions_hash"]:
            raise ValueError(f"{path}: its {split} frames are not those of {info['file']} any more")
        structs = W.e4_structures(info["file"], pins=True, K=K, omega0=om)
        out[split] = {"structures": [structs[i] for i in idx], "index": idx,
                      "mu": ang_to_bohr(d[f"{split}_mu"]), "R": ang_to_bohr(ang_to_bohr(ang_to_bohr(d[f"{split}_R"])))}
    return out


def liquid_block(rung_cell, response, lq):
    """the calibration rows of job 2's matching (LIN.solve_readout's extra): every calibration cell's induced dipole M (3)
    and field response R (3, 3) against MB-pol's, with the dense response solve (64 molecules).  Weights: lq["weight"]
    times the mean over cells and components of the squared residual relative to n alpha_ref (R) and sqrt(n) mu_ref (M)
    -- weight 0.1 makes a 1 % error of the cells' response cost as much as the far probes' matched objective (~1e-5)."""
    cal = response["calibration"]
    r = in_cell(replace(rung_cell, bf_solver="direct"), cal["structures"][0].cell)
    n_mol = cal["mu"].shape[1]
    s_R = n_mol * ang_to_bohr(ang_to_bohr(ang_to_bohr(float(lq.get("alpha_ref_angstrom3", 1.4075)))))
    s_M = math.sqrt(n_mol) * ang_to_bohr(float(lq.get("mu_ref_eangstrom", 0.15)))
    nf = len(cal["structures"])
    w = float(lq["weight"]) / nf * np.concatenate([np.full(3, 1.0 / (3 * s_M ** 2)), np.full(9, 1.0 / (9 * s_R ** 2))])

    def fn(params, b):
        Mi, R = M.induced_response(params, r, b)
        return jnp.concatenate([Mi, R.reshape(-1)])
    batches = [{k: v[0] for k, v in S.pad_batch([st], n_freq=r.K).items()} for st in cal["structures"]]
    targets = [np.concatenate([mu.sum(0), R.sum(0).reshape(-1)]) for mu, R in zip(cal["mu"], cal["R"])]
    tag = hashlib.sha256(json.dumps({**lq, "response": response["hash"]}, sort_keys=True).encode()).hexdigest()[:8]
    return {"block": {"fn": fn, "batches": batches, "targets": targets, "weights": [w] * nf}, "hash": tag}


def read_liquid(params, rung_cell, response, split="heldout"):
    """on the held-out liquid cells: eps_inf of the model and of MB-pol on the same frames, the cells' induced dipole
    against MB-pol's, and (a rung with a mutual response) every molecule's induced dipole at zero field and response to
    a uniform field against MB-pol's (rmse / rms; the model's order-0 sector, model.molecular_response)."""
    h = response[split]
    r = in_cell(replace(rung_cell, bf_solver="direct"), h["structures"][0].cell)
    f = jax.jit(lambda p, b: M.molecular_response(p, r, b)) if r.bf_mutual else None
    g = jax.jit(lambda p, b: M.induced_response(p, r, b))
    eps_m, eps_t, dM, dmu, dR = [], [], [], [], []
    for st, mu_t, R_t in zip(h["structures"], h["mu"], h["R"]):
        b = {k: v[0] for k, v in S.pad_batch([st], n_freq=r.K).items()}
        V = abs(np.linalg.det(np.asarray(st.cell)))
        Mc, Rc = (np.asarray(x) for x in g(params, b))
        eps_m.append(float(1 + 4 * np.pi * np.trace(Rc) / 3 / V))
        eps_t.append(float(1 + 4 * np.pi * np.trace(R_t.sum(0)) / 3 / V))
        dM.append((Mc - mu_t.sum(0), mu_t.sum(0)))
        if f is not None:
            mu_m, R_m = (np.asarray(x)[:len(mu_t)] for x in f(params, b))
            dmu.append((mu_m - mu_t, mu_t)); dR.append((R_m - R_t, R_t))
    rel = lambda pairs: float(np.sqrt(np.mean([np.mean(e ** 2) for e, _ in pairs]) / np.mean([np.mean(t ** 2) for _, t in pairs])))
    out = {"frames": len(eps_m), "eps_inf_model": float(np.mean(eps_m)), "eps_inf_truth": float(np.mean(eps_t)),
           "eps_inf_model_frames": eps_m, "eps_inf_truth_frames": eps_t, "cell_dipole_rel_rmse": rel(dM)}
    if dmu:
        iso = lambda x: np.trace(x, axis1=-2, axis2=-1) / 3
        im, it = (np.concatenate([iso(e + t) for e, t in dR]), np.concatenate([iso(t) for _, t in dR]))
        mm, mt = (np.concatenate([(e + t).ravel() for e, t in dmu]), np.concatenate([t.ravel() for _, t in dmu]))
        out.update(molecular_dipole_rel_rmse=rel(dmu), molecular_response_rel_rmse=rel(dR),
                   molecular_response_iso_rel_rmse=rel([(iso(e), iso(t)) for e, t in dR]),
                   # the molecules' spread about their mean, the model's against MB-pol's, and how well they follow it
                   molecular_response_iso_sd_ratio=float(np.std(im) / np.std(it)),
                   molecular_response_iso_corr=float(np.corrcoef(im, it)[0, 1]),
                   molecular_dipole_corr=float(np.corrcoef(mm, mt)[0, 1]),
                   molecular_dipole_norm_ratio=float(np.sqrt(np.mean(mm ** 2) / np.mean(mt ** 2))))
    return out


def liquid_line(v):
    return (f"liquid cells (held out, {v['frames']} frames): eps_inf {v['eps_inf_model']:.4f} against MB-pol's "
            f"{v['eps_inf_truth']:.4f}; cell induced dipole {100 * v['cell_dipole_rel_rmse']:.1f} %"
            + (f"; per molecule: induced dipole {100 * v['molecular_dipole_rel_rmse']:.1f} %, field response "
               f"{100 * v['molecular_response_rel_rmse']:.1f} % (isotropic {100 * v['molecular_response_iso_rel_rmse']:.1f} %)"
               if "molecular_dipole_rel_rmse" in v else "") + " (rmse / rms)"
            + (f"; the isotropic response's spread {v['molecular_response_iso_sd_ratio']:.2f} of MB-pol's, r "
               f"{v['molecular_response_iso_corr']:.2f}; the dipoles' size {v['molecular_dipole_norm_ratio']:.3f} of "
               f"MB-pol's, r {v['molecular_dipole_corr']:.3f}" if "molecular_response_iso_sd_ratio" in v else ""))


# ---------------------------------------------------------------- the matching --------------------------------------
def with_readout(params, readout):
    """params with the read-out replaced (a matched one, as matched() returns it: NumPy arrays)."""
    return {**params, "readout": jax.tree_util.tree_map(jnp.asarray, readout)}


def match_readout(rung_open, probes, scale, cfg, verbose=True, liquid=None, start=None):
    """the read-out's parameters matched on the training probes beyond r_c of their cores (E3-QP2's far_first stage,
    E3.far_field_blind): energies only, every order, Gauss-Newton steps, the ridge chosen on held-out cores.  With
    liquid (job 2's arm B, liquid_block) a second stage from that solution -- start, (read-out, record) as matched()
    keeps it, or solved first -- with MB-pol's induced dipole and field response of liquid cells as further rows
    (LIN.solve_readout's extra), the probes' ridge kept."""
    ro = cfg["training"]["readout"]
    blind = E3.far_field_blind(probes["train"], cfg["model"]["r_cut_angstrom"])
    idx = np.flatnonzero(blind)
    if len(idx) == 0:
        raise ValueError("no training probe lies beyond r_c of its core: nothing to match the read-out on")
    structs = [probes["train"][i] for i in idx]
    data = S.pad_batch(structs, n_freq=rung_open.K)
    params0 = M.init_params(jax.random.PRNGKey(0), rung_open, energy_scale=scale)
    extra = {}
    if ro["ridge_selection"] == "groups":
        extra = dict(groups=np.array([st.info["core_id"] for st in structs]), ridge_grid=LIN.RIDGE_GRID)
    t0 = time.time()
    if start is None:
        params, report = LIN.solve_readout(params0, rung_open, data, scale, w_force=0.0, ridge=float(ro["ridge"]),
                                           chunk=int(ro["solve_chunk"]),
                                           w_energy=float(cfg["training"]["loss"]["energy_weight"]),
                                           orders=ro["orders"], iterations=int(ro["solve_iterations"]), **extra)
        hist = [(float(b), float(c)) for b, c in getattr(report, "history", [report])]
        ridges = [float(r) for r in getattr(report, "ridge", [])] if hasattr(report, "ridge") else []
    else:
        params = with_readout(params0, start[0])
        hist, ridges = [tuple(h) for h in start[1]["objective"]], list(start[1]["ridge"])
        t0 -= start[1]["time"]
    hist_x = []
    if liquid is not None:
        # job 2's calibration, a second stage from the far-field solution: the probes and the liquid cells' response
        # together, Gauss-Newton steps (halved where they overshoot), the probes' ridge kept
        rel = ridges[0] if ridges else float(ro["ridge"])
        params, report = LIN.solve_readout(params, rung_open, data, scale, w_force=0.0, ridge=rel,
                                           chunk=int(ro["solve_chunk"]),
                                           w_energy=float(cfg["training"]["loss"]["energy_weight"]),
                                           orders=ro["orders"], iterations=int(ro["solve_iterations"]),
                                           extra=liquid["block"])
        hist += [(float(b), float(c)) for b, c in report.history]
        hist_x = [(float(b), float(c)) for b, c in report.extra_history]
    if verbose:
        print(f"  read-out matched on {len(idx)} of {len(probes['train'])} training probes (beyond r_c of their cores)"
              + (f" and {len(liquid['block']['batches'])} liquid cells' response" if liquid is not None else "")
              + ": objective " + "; ".join(f"{b:.3e} -> {c:.3e}" for b, c in hist)
              + (" (the liquid rows: " + "; ".join(f"{b:.3e} -> {c:.3e}" for b, c in hist_x) + ")" if hist_x else "")
              + (f" (ridge {ridges[0]:.0e})" if ridges else "") + f"  [{time.time() - t0:.0f} s]", flush=True)
    out = {"probes": int(len(idx)), "objective": hist, "ridge": ridges, "time": time.time() - t0}
    if liquid is not None:
        out.update(liquid_objective=hist_x, liquid_cells=len(liquid["block"]["batches"]), liquid_hash=liquid["hash"])
    return params, out


def matched(name, rung_open, probes, scale, cfg, out_dir, liquid=None):
    """the rung's matched read-out, from out_dir's cache (written once per rung and probe set, atomically, so that the
    tasks of an array job share it) or solved now; with the far-field test probes read on it before any training
    (E_0 at its initialisation: the probes at 10-15 A lie beyond r_c of every core atom, those at 8 A not all).
    liquid (job 2's calibration, liquid_block): MB-pol's response of liquid cells as further matching rows, from the
    far-field read-out (itself from the cache, or matched and cached first)."""
    tag = "" if liquid is None else "_liq" + liquid["hash"]
    path = os.path.join(out_dir, f"matched_{name}_{probes['hash']}{tag}.pkl")
    key = repr(rung_open) + tag
    if os.path.exists(path):
        with open(path, "rb") as fh:
            got = pickle.load(fh)
        if got["rung"] == key and got["scale"] == scale:
            print(f"  read-out matched before ({path}): objective "
                  + "; ".join(f"{b:.3e} -> {c:.3e}" for b, c in got["match"]["objective"]), flush=True)
            return got["readout"], got["match"]
        print(f"  {path} was matched for another rung or scale: matched again", flush=True)
    start = None if liquid is None else matched(name, rung_open, probes, scale, cfg, out_dir)   # arm B: from arm A's
    params, match = match_readout(rung_open, probes, scale, cfg, liquid=liquid, start=start)
    t0 = time.time()
    match["probes_before_training"] = read_probes(params, rung_open, probes)
    print(f"  {name}: far-field probes on the matched read-out, before training: "
          f"{E3.probe_summary(match['probes_before_training'])}  [{time.time() - t0:.0f} s]", flush=True)
    if "trimers" in probes:
        match["far_trimers_before_training"] = v = read_trimers(params, rung_open, probes)
        print(f"  {name}: far trimers (T5) on the matched read-out: three-body rmse {v['three_body_rmse'] * KCAL:.5f} of "
              f"rms {v['three_body_rms'] * KCAL:.5f} kcal/mol, slope {v['slope']:.4f}", flush=True)
    readout = jax.tree_util.tree_map(np.asarray, params["readout"])
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump({"rung": key, "scale": scale, "readout": readout, "match": match}, fh)
    os.replace(tmp, path)
    return readout, match


# ---------------------------------------------------------------- the readings --------------------------------------
def rigid_errors(err_kcal, pos_ang, cell_ang, n_mol):
    """per molecule: the net force error (kcal/mol/A), the torque error about the centroid (kcal/mol) and the
    rigid-body projection's sum of squares and degrees of freedom (sseft.train.rigid_force_sums)."""
    frag = np.repeat(np.arange(n_mol), 3)
    fm = np.zeros((n_mol, 3 * n_mol), bool); fm[frag, np.arange(3 * n_mol)] = True
    s, c = T.rigid_force_sums(jnp.asarray(err_kcal), jnp.asarray(pos_ang), jnp.ones(3 * n_mol, bool), jnp.asarray(fm),
                              jnp.asarray(cell_ang))
    d = pos_ang.reshape(n_mol, 3, 3) - pos_ang.reshape(n_mol, 3, 3)[:, :1]
    d = d - np.round(d @ np.linalg.inv(cell_ang)) @ cell_ang
    d = d - d.mean(axis=1, keepdims=True)
    e = err_kcal.reshape(n_mol, 3, 3)
    return e.sum(axis=1), np.cross(d, e).sum(axis=1), float(s), float(c)


def read_cells(params, rung, structs, eps_frames=0):
    """energy per molecule and force errors on periodic frames of one size (one compilation), and eps_inf on the
    first eps_frames of them (the read-out's response to a uniform field at clamped nuclei; model.dielectric_inf)."""
    r = in_cell(rung, structs[0].cell)
    f = jax.jit(lambda p, b: M.energy_forces_single(p, r, b))
    n_mol = structs[0].n_atoms // 3
    dE, net, tau, ss, cc, fa, times = [], [], [], 0.0, 0.0, [], []
    for st in structs:
        b = {k: v[0] for k, v in S.pad_batch([st], n_freq=r.K).items()}
        t0 = time.time()
        E, F = f(params, b)
        E = float(E); F = np.asarray(F)[:st.n_atoms]
        times.append(time.time() - t0)
        dE.append((E - st.energy) / n_mol * KCAL)
        err = (F - np.asarray(st.forces)) * FORCE_UNIT
        fa.append(np.mean(err ** 2))
        n_, t_, s_, c_ = rigid_errors(err, bohr_to_ang(np.asarray(st.positions)), bohr_to_ang(np.asarray(st.cell)), n_mol)
        net.append(n_); tau.append(t_); ss += s_; cc += c_
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    out = {"frames": len(structs), "n_mol": n_mol, "n_max": int(r.ewald_n_max),
           "energy_per_molecule_rmse": rms(dE), "energy_per_molecule_mean_error": float(np.mean(dE)),
           "energy_per_molecule_errors": [float(x) for x in dE],
           "force_rigid_rmse": float(np.sqrt(ss / cc)), "net_force_rmse": rms(np.concatenate(net)),
           "torque_rmse": rms(np.concatenate(tau)), "force_atomic_rmse": float(np.sqrt(np.mean(fa))),
           "seconds_per_frame": float(np.median(times[1:] if len(times) > 1 else times))}
    if eps_frames > 0:
        g = jax.jit(lambda p, b: M.dielectric_inf(p, r, b))
        eps = []
        for st in structs[:eps_frames]:
            b = {k: v[0] for k, v in S.pad_batch([st], n_freq=r.K).items()}
            eps.append(np.asarray(g(params, b)))
        eps = np.array(eps)
        out["eps_inf"] = {"frames": len(eps), "mean_tensor": eps.mean(axis=0).tolist(),
                          "isotropic": [float(np.trace(e) / 3) for e in eps],
                          "isotropic_mean": float(np.mean([np.trace(e) / 3 for e in eps]))}
    return out


def read_probes(params, rung_open, probes):
    """the far-field test probes: the model's probe interaction and its non-additive part against the truth's
    (E3.probe_metrics), each sub-structure with MB-pol's pins."""
    cores, pairs, owner = E3.probe_parts(probes["test"])
    pin = lambda P: W.pin_rigid_site_model([W.water_structure(p, pinned_q=False) for p in P], probes["site_q"],
                                           probes["site_gamma"], rung_open.K, rung_open.omega0)
    inter = T.predict_energies(params, rung_open, probes["test"]) - T.predict_energies(params, rung_open, pin(cores))
    nadd = inter - np.bincount(owner, weights=T.predict_energies(params, rung_open, pin(pairs)), minlength=len(inter))
    return E3.probe_metrics((inter, nadd), probes["test_ref"])


def read_trimers(params, rung_open, probes):
    """T5: the model's three-body energy of the far trimers against the truth's: rmse and rms (hartree) and the
    slope of the model's against the truth's (1 for a model that has it, 0 for one that cannot)."""
    t = probes["trimers"]
    E = T.predict_energies(params, rung_open, t["structures"])
    Ep = T.predict_energies(params, rung_open, t["pairs"]).reshape(-1, 3).sum(axis=1)
    model, ref = E - Ep, t["three_body"]
    return {"count": int(len(ref)), "three_body_rmse": float(np.sqrt(np.mean((model - ref) ** 2))),
            "three_body_rms": float(np.sqrt(np.mean(ref ** 2))),
            "slope": float(np.dot(model, ref) / np.dot(ref, ref)), "model": [float(x) for x in model]}


def summary_line(name, seed, rd):
    parts = []
    for n, by in rd["cells"].items():
        for split in ("valid", "test"):
            if split in by and "energy_per_molecule_rmse" in by[split]:
                v = by[split]
                parts.append(f"{split}_n{n} E {v['energy_per_molecule_rmse']:.4f} F {v['force_rigid_rmse']:.3f}")
    if "far_trimers" in rd:
        v = rd["far_trimers"]
        parts.append(f"far trimers 3-body {100 * v['three_body_rmse'] / v['three_body_rms']:.0f} % (slope {v['slope']:.3f})")
    if "probes" in rd:
        b = rd["probes"]["by_distance_angstrom"]
        parts.append("probes non-add " + "/".join(f"{100 * v['nonadditive_rmse'] / v['nonadditive_rms']:.0f}"
                                                  for k, v in b.items() if float(k) >= 10) + " %")
    if "liquid" in rd:
        parts.append(f"liquid eps_inf {rd['liquid']['eps_inf_model']:.4f} (MB-pol {rd['liquid']['eps_inf_truth']:.4f})")
    return f"{name} seed {seed}: " + "; ".join(parts)


# ---------------------------------------------------------------- training ------------------------------------------
def divisor_at_most(n, k):
    return max(d for d in range(1, max(1, min(n, k)) + 1) if n % d == 0)


def run_train(a):
    cfg = load_config(a.config)
    rungs = a.rungs.split(",") if a.rungs else list(cfg["rungs"])
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else [int(s) for s in cfg["seeds"]]
    if a.task is not None:
        if not 0 <= a.task < len(rungs) * len(seeds):
            raise ValueError(f"--task must be in [0, {len(rungs) * len(seeds)}): {len(rungs)} rungs x {len(seeds)} seeds")
        rungs, seeds = [rungs[a.task // len(seeds)]], [seeds[a.task % len(seeds)]]
    for name in rungs:
        if name not in E3.RUNGS:
            raise ValueError(f"unknown rung {name}")
    out_dir = a.out_dir
    os.makedirs(out_dir, exist_ok=True)
    dcfg, tr = cfg["data"], cfg["training"]
    L0 = int(dcfg["train_size"])
    path = lambda split, n: os.path.join(dcfg["dir"], f"{split}_n{n}.npz")
    K, om = int(cfg["model"]["K"]), float(cfg["model"]["omega0"])
    train = W.e4_structures(path("train", L0), pins=True, K=K, omega0=om)
    scale = T.energy_scale(train)
    probes = load_probes(dcfg["probes"], K, om)
    print(f"sseft {sseft.__version__}, JAX {jax.__version__}, devices {jax.devices()}", flush=True)
    print(f"training cells: {len(train)} frames of {L0} molecules ({path('train', L0)}), energy scale "
          f"{scale * KCAL:.4f} kcal/mol per atom; probes {dcfg['probes']} (hash {probes['hash']}): "
          f"{len(probes['train'])} training, {len(probes['test'])} test", flush=True)
    chunk = divisor_at_most(len(train), int(tr["lbfgs"]["chunk_size"]))
    if chunk != int(tr["lbfgs"]["chunk_size"]):
        print(f"  the L-BFGS chunk {tr['lbfgs']['chunk_size']} does not divide {len(train)} frames: {chunk} instead",
              flush=True)
    evals = {}
    for n in dcfg["eval_sizes"]:
        for split in cfg["reading"]["splits"]:
            if os.path.exists(path(split, n)):
                evals[(int(n), split)] = path(split, n)
    lq, response = tr["readout"].get("liquid"), None
    if dcfg.get("response") and os.path.exists(dcfg["response"]):
        response = load_response(dcfg["response"], cfg)
        print(f"liquid response {dcfg['response']} (hash {response['hash']}, MB-pol through MBX): "
              + ", ".join(f"{len(response[k]['structures'])} {k}" for k in ("calibration", "heldout") if k in response)
              + " frames" + (f"; the read-out calibrated on them, {json.dumps(lq)}" if lq else ""), flush=True)
    elif lq:
        raise FileNotFoundError(f"training.readout.liquid needs MB-pol's liquid response {dcfg.get('response')}: "
                                "scripts/e4_water.py response --mbx ...")
    for name in rungs:
        rung, rung_open = rungs_for(name, cfg, train[0].cell)
        readout, match = None, None
        if rung.band_fields >= 0:
            liquid = liquid_block(rung, response, lq) if lq else None
            readout, match = matched(name, rung_open, probes, scale, cfg, out_dir, liquid=liquid)
            if response is not None and len(response.get("heldout", {}).get("structures", [])):
                t1 = time.time()
                p0 = with_readout(M.init_params(jax.random.PRNGKey(0), rung_open, energy_scale=scale), readout)
                match = {**match, "liquid_before_training": read_liquid(p0, rung, response)}
                print(f"  {name}: {liquid_line(match['liquid_before_training'])} on the matched read-out, before training"
                      f"  [{time.time() - t1:.0f} s]", flush=True)
        for seed in seeds:
            t0 = time.time()
            print(f"{name} seed {seed}: training on {len(train)} cells (n_max {rung.ewald_n_max}, solver "
                  f"{rung.bf_solver if rung.bf_mutual else '-'}, forces {tr['loss']['forces']})", flush=True)
            ad, lb, ls = tr["adam"], tr["lbfgs"], tr["loss"]
            params, info = T.fit(rung, train, seed=seed, steps=int(ad["steps"]), lr=float(ad["learning_rate"]),
                                 w_energy=float(ls["energy_weight"]), w_force=float(ls["force_weight"]),
                                 batch_size=int(ad["batch_size"]), lbfgs_steps=int(lb["steps"]), lbfgs_chunk=chunk,
                                 lr_final_ratio=float(ad["final_learning_rate_ratio"]), adam_b1=float(ad["beta1"]),
                                 adam_b2=float(ad["beta2"]), adam_eps=float(ad["epsilon"]), scale=scale,
                                 lbfgs_memory=int(lb["memory_size"]),
                                 lbfgs_max_linesearch_steps=int(lb["max_linesearch_steps"]), readout_init=readout,
                                 force_mode=ls["forces"], verbose=True)
            t_train = time.time() - t0
            stem = os.path.join(out_dir, f"{name}_seed{seed}")
            with open(stem + ".params.pkl", "wb") as fh:
                pickle.dump(jax.tree_util.tree_map(np.asarray, params), fh)
            rd = {"cells": {}}
            for (n, split), p in sorted(evals.items()):
                t1 = time.time()
                structs = train if (n == L0 and split == "train") else W.e4_structures(p, pins=True, K=K, omega0=om)
                eps = int(cfg["reading"]["eps_frames"]) if (n == L0 and split == "test" and rung.band_fields >= 0) else 0
                try:
                    v = read_cells(params, rung, structs, eps_frames=eps)
                except Exception as e:                                   # out of memory and the like: recorded
                    v = {"failed": f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"}
                rd["cells"].setdefault(str(n), {})[split] = v
                if "failed" in v:
                    print(f"  {name} seed {seed}: {split}_n{n} failed -- {v['failed']}", flush=True)
                else:
                    print(f"  {name} seed {seed}: {split}_n{n} ({v['frames']} frames) energy per molecule rmse "
                          f"{v['energy_per_molecule_rmse']:.4f} kcal/mol (mean {v['energy_per_molecule_mean_error']:+.4f}), "
                          f"rigid forces {v['force_rigid_rmse']:.3f} kcal/mol/A (net {v['net_force_rmse']:.3f}, torque "
                          f"{v['torque_rmse']:.3f} kcal/mol; atomic {v['force_atomic_rmse']:.3f})"
                          + (f", eps_inf {v['eps_inf']['isotropic_mean']:.4f} on {v['eps_inf']['frames']} frames"
                             if "eps_inf" in v else "") + f"  [{time.time() - t1:.0f} s]", flush=True)
            t1 = time.time()
            rd["probes"] = read_probes(params, rung_open, probes)
            print(f"  {name} seed {seed}: far-field probes {E3.probe_summary(rd['probes'])}  [{time.time() - t1:.0f} s]",
                  flush=True)
            if "trimers" in probes:
                rd["far_trimers"] = read_trimers(params, rung_open, probes)
                v = rd["far_trimers"]
                print(f"  {name} seed {seed}: far trimers (T5), three-body rmse {v['three_body_rmse'] * KCAL:.5f} of rms "
                      f"{v['three_body_rms'] * KCAL:.5f} kcal/mol, slope {v['slope']:.4f}", flush=True)
            if match is not None and "liquid_before_training" in match:
                t1 = time.time()
                rd["liquid"] = read_liquid(params, rung, response)
                print(f"  {name} seed {seed}: {liquid_line(rd['liquid'])}  [{time.time() - t1:.0f} s]", flush=True)
            result = {"rung": name, "seed": seed, "model": model_record(name, cfg, rung), "config": cfg,
                      "sseft": sseft.__version__, "jax": jax.__version__, "devices": str(jax.devices()),
                      "probes_hash": probes["hash"], "energy_scale": scale, "training_frames": len(train),
                      "matching": match, "training": {"time": t_train, "history": info["history"],
                                                      "options": info["training_options"]},
                      "readings": rd}
            with open(stem + ".json", "w") as fh:
                json.dump(result, fh, indent=1)
            print(f"  {summary_line(name, seed, rd)}  [{time.time() - t0:.0f} s in all; {stem}.json]", flush=True)


# ---------------------------------------------------------------- summary -------------------------------------------
def run_summary(a):
    runs = {}
    for p in sorted(glob.glob(os.path.join(a.out_dir, "*_seed*.json"))):
        with open(p) as fh:
            r = json.load(fh)
        runs.setdefault(r["rung"], []).append(r)
    if not runs:
        print(f"no results in {a.out_dir}"); return
    ms = lambda v: f"{np.mean(v):.4f} +- {np.std(v):.4f}" if len(v) > 1 else f"{np.mean(v):.4f}"
    print(f"{a.out_dir}: seed means +- spreads (kcal/mol per molecule; rigid forces kcal/mol/A)")
    for name, rs in runs.items():
        seeds = sorted(r["seed"] for r in rs)
        print(f"{name} (seeds {seeds}; probes hash {sorted({r['probes_hash'] for r in rs})})")
        sizes = sorted({n for r in rs for n in r["readings"]["cells"]}, key=int)
        for n in sizes:
            for split in SPLITS:
                vals = [r["readings"]["cells"][n].get(split, {}) for r in rs if n in r["readings"]["cells"]]
                vals = [v for v in vals if "energy_per_molecule_rmse" in v]
                if vals:
                    print(f"  {split:5s} n{n:>5s}: energy {ms([v['energy_per_molecule_rmse'] for v in vals])}, "
                          f"bias {ms([v['energy_per_molecule_mean_error'] for v in vals])}; forces "
                          f"{ms([v['force_rigid_rmse'] for v in vals])}"
                          + (f"; eps_inf {ms([v['eps_inf']['isotropic_mean'] for v in vals if 'eps_inf' in v])}"
                             if any("eps_inf" in v for v in vals) else ""))
        tri = [r["readings"]["far_trimers"] for r in rs if "far_trimers" in r["readings"]]
        if tri:
            print(f"  far trimers (T5): three-body rmse / rms {ms([100 * v['three_body_rmse'] / v['three_body_rms'] for v in tri])} %, "
                  f"slope {ms([v['slope'] for v in tri])}")
        by = [r["readings"]["probes"]["by_distance_angstrom"] for r in rs if "probes" in r["readings"]]
        if by:
            print("  far-field probes, non-additive rmse / rms: " + ", ".join(
                f"{d} A {ms([100 * b[d]['nonadditive_rmse'] / b[d]['nonadditive_rms'] for b in by])} %" for d in by[0]))
        liq = [r["readings"]["liquid"] for r in rs if "liquid" in r["readings"]]
        if liq:
            print(f"  liquid cells (held out, {liq[0]['frames']} frames): eps_inf {ms([v['eps_inf_model'] for v in liq])} "
                  f"(MB-pol {liq[0]['eps_inf_truth']:.4f}); cell induced dipole {ms([100 * v['cell_dipole_rel_rmse'] for v in liq])} %"
                  + (f"; per molecule: induced dipole {ms([100 * v['molecular_dipole_rel_rmse'] for v in liq])} %, field "
                     f"response {ms([100 * v['molecular_response_rel_rmse'] for v in liq])} %"
                     if "molecular_dipole_rel_rmse" in liq[0] else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probes", help="MB-pol's far-field probes through MBX")
    p.add_argument("--mbx", required=True, help="the MBX build (MBX_HOME), e.g. external/mbx/build/MBX")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=None, help="the probes file (default: the config's data.probes)")
    t = sub.add_parser("train", help="match, train and read the rungs")
    t.add_argument("--config", default=None)
    t.add_argument("--rungs", default=None, help="comma-separated (default: the config's)")
    t.add_argument("--seeds", default=None, help="comma-separated (default: the config's)")
    t.add_argument("--task", type=int, default=None, help="one (rung, seed): rung task // n_seeds, seed task % n_seeds")
    t.add_argument("--out_dir", default="results/e4")
    s = sub.add_parser("summary", help="seed means of the readings in out_dir")
    s.add_argument("--out_dir", default="results/e4")
    r = sub.add_parser("response", help="MB-pol's induced dipoles and field response of liquid cells (job 2), through MBX")
    r.add_argument("--mbx", required=True)
    r.add_argument("--config", default=None)
    r.add_argument("--out", default=None, help="the response file (default: the config's data.response)")
    a = ap.parse_args()
    {"probes": run_probes, "train": run_train, "summary": run_summary, "response": run_response}[a.cmd](a)


if __name__ == "__main__":
    main()
