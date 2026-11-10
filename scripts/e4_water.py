"""
e4_water.py -- E4's training driver: the ladder on MB-pol's periodic water cells of one size L0 (notes/e4-plan.md
sections 4 and 9; the pre-registrations are notes/e4-x3-preregistration.md and, for job 2,
notes/e4-job2-preregistration.md; for job 3, notes/e4-job3-preregistration.md; for job 4,
notes/e4-job4-preregistration.md).  Six steps:

    python scripts/e4_water.py probes --mbx external/mbx/build/MBX [--config configs/e4_water.yaml]
    python scripts/e4_water.py response --mbx external/mbx/build/MBX [--config configs/e4_job2b.yaml]
    python scripts/e4_water.py train --config configs/e4_water.yaml [--rungs M_16C2pm_us] [--seeds 0] [--task 5]
    python scripts/e4_water.py summary --out_dir results/e4_job1
    python scripts/e4_water.py response_k --mbx external/mbx/build/MBX --config configs/e4_job3.yaml [--part 0/2]
    python scripts/e4_water.py kread --config configs/e4_job3.yaml --out_dir results/e4_job3 [--only C,C0] [--out x.json]

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
              with training.readout.finite_k (job 4's arm D, v0.0.28) a third stage moves the orders >= 1 alone on the
              probes and the calibration cells' finite-k susceptibilities (k_stage);
           2. the rung is trained on the cells of one size (Adam, then an L-BFGS polish; the read-out held), on the
              energies and on the forces' rigid-body part (sseft.train.rigid_force_sums: each molecule's net force and
              torque; the internal components carry the flux of MB-pol's geometry-dependent charges, which rigid pinned
              charges cannot represent and whose long-range part no local model can);
           3. it is read: energy per molecule and forces on the training, validation and test frames of every size in
              data.eval_sizes, the far-field test probes, and eps_inf on test frames of the training size (rungs with a
              response; the others have eps_inf = 1 by construction).
         One json and one parameter file per rung and seed in out_dir.
summary  seed means and spreads of every rung's readings, from the jsons in out_dir.
response_k (job 3, v0.0.27) MB-pol's response of the same liquid cells to longitudinal plane waves of the cell
         (sseft.modes; liquid_k: axes, harmonics, phases), every molecule's induced dipole and each mode's
         susceptibility, through a field set at every site MBX evaluates (mbpol.set_site_fields); --part i/n and
         --merge n split the frames across tasks.
kread    (job 3) no training: the read-outs of kread.arms, held in their output directories, read on the finite-k and
         uniform responses with the far probes and T5 (model.mode_response); with the orders >= 1 off for
         kread.order_zero_of; and the locality test (kread.locality): the order-0 columns solved on every calibration
         molecule's dipole and response alone, no probe in the objective, the far probes then out of state.  v0.0.28
         (job 4): an arm's "set" overrides its config (another mutual edge, another stage), and --out names the json,
         so that tasks reading different arms write different files.  v0.0.29 (job 5): kread.contact adds every
         read-out's change at contact against a reference arm's (read_contact), and an arm whose finite_k has
         anchor true is matched through the anchored third stage (arm D').

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
from sseft import modes as MD                  # noqa: E402
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
             "response": "data/e4/response_mbpol.npz", "response_k": "data/e4/response_k_mbpol.npz"},
    "liquid": {"calibration_frames": 30, "heldout_frames": 10, "delta": 1.0e-4},
    # job 3 (v0.0.27): the plane-wave modes of the liquid cells (sseft.modes), on the frames of "liquid"
    "liquid_k": {"axes": [0, 1, 2], "harmonics": [1, 2], "phases": [0, 1], "delta": 1.0e-4},
    # job 3's reading (run_kread): the rung, the read-outs read (each arm's config and output directory, where its
    # matched read-out is cached, and v0.0.28 "set": overrides of that config), which of them also with the orders
    # >= 1 off (a list: label arm + "0"; a mapping: arm -> label), the locality test (None: off), and v0.0.29 the
    # contact reading (None: off; {reference, frames, max_frames}: every variant's change at contact, read_contact)
    "kread": {"rung": "M_16C2pm_us", "arms": {}, "order_zero_of": [], "splits": ["heldout", "calibration"],
              "locality": None, "contact": None},
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
                    "solve_chunk": 64, "liquid": None, "finite_k": None},
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
    -- weight 0.1 makes a 1 % error of the cells' response cost as much as the far probes' matched objective (~1e-5).
    lq["rows"] = "molecules" (v0.0.27, job 3's locality test): every molecule's induced dipole (3) and field response
    (3, 3) instead (model.molecular_response: the order-0 sector's, which is the whole response of a read-out at order
    0), relative to mu_ref and alpha_ref, the same weight per cell; their Jacobian by forward mode (many rows)."""
    cal = response["calibration"]
    r = in_cell(replace(rung_cell, bf_solver="direct"), cal["structures"][0].cell)
    n_mol = cal["mu"].shape[1]
    a_ref = ang_to_bohr(ang_to_bohr(ang_to_bohr(float(lq.get("alpha_ref_angstrom3", 1.4075)))))
    m_ref = ang_to_bohr(float(lq.get("mu_ref_eangstrom", 0.15)))
    nf = len(cal["structures"])
    batches = [{k: v[0] for k, v in S.pad_batch([st], n_freq=r.K).items()} for st in cal["structures"]]
    rows = lq.get("rows", "cells")
    if rows == "molecules":
        w = float(lq["weight"]) / nf * np.concatenate([np.full(3 * n_mol, 1.0 / (3 * n_mol * m_ref ** 2)),
                                                       np.full(9 * n_mol, 1.0 / (9 * n_mol * a_ref ** 2))])

        def fn(params, b):
            mu, R = M.molecular_response(params, r, b)
            return jnp.concatenate([mu[:n_mol].reshape(-1), R[:n_mol].reshape(-1)])
        targets = [np.concatenate([mu.reshape(-1), R.reshape(-1)]) for mu, R in zip(cal["mu"], cal["R"])]
    elif rows == "cells":
        s_R, s_M = n_mol * a_ref, math.sqrt(n_mol) * m_ref
        w = float(lq["weight"]) / nf * np.concatenate([np.full(3, 1.0 / (3 * s_M ** 2)), np.full(9, 1.0 / (9 * s_R ** 2))])

        def fn(params, b):
            Mi, R = M.induced_response(params, r, b)
            return jnp.concatenate([Mi, R.reshape(-1)])
        targets = [np.concatenate([mu.sum(0), R.sum(0).reshape(-1)]) for mu, R in zip(cal["mu"], cal["R"])]
    else:
        raise ValueError(f"liquid rows must be 'cells' or 'molecules', not {rows!r}")
    tag = hashlib.sha256(json.dumps({**lq, "response": response["hash"]}, sort_keys=True).encode()).hexdigest()[:8]
    # lq["orders"] (v0.0.25): the read-out orders the second stage moves -- "zero" for the order-0 columns alone, as the
    # derivative counting asks of k = 0 data (notes/e4-job2-results.md section 4b); unset, the readout's own orders
    block = {"fn": fn, "batches": batches, "targets": targets, "weights": [w] * nf}
    if rows == "molecules":
        block.update(jac="forward", jac_chunk=int(lq.get("jac_chunk", 32)))
    return {"block": block, "hash": tag, "orders": lq.get("orders")}


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
    out = {"split": split, "frames": len(eps_m), "eps_inf_model": float(np.mean(eps_m)),
           "eps_inf_truth": float(np.mean(eps_t)),
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
    which = {"heldout": "held out", None: "held out"}.get(v.get("split"), v.get("split"))
    return (f"liquid cells ({which}, {v['frames']} frames): eps_inf {v['eps_inf_model']:.4f} against MB-pol's "
            f"{v['eps_inf_truth']:.4f}; cell induced dipole {100 * v['cell_dipole_rel_rmse']:.1f} %"
            + (f"; per molecule: induced dipole {100 * v['molecular_dipole_rel_rmse']:.1f} %, field response "
               f"{100 * v['molecular_response_rel_rmse']:.1f} % (isotropic {100 * v['molecular_response_iso_rel_rmse']:.1f} %)"
               if "molecular_dipole_rel_rmse" in v else "") + " (rmse / rms)"
            + (f"; the isotropic response's spread {v['molecular_response_iso_sd_ratio']:.2f} of MB-pol's, r "
               f"{v['molecular_response_iso_corr']:.2f}; the dipoles' size {v['molecular_dipole_norm_ratio']:.3f} of "
               f"MB-pol's, r {v['molecular_dipole_corr']:.3f}" if "molecular_response_iso_sd_ratio" in v else ""))


# ---------------------------------------------------------------- the liquid's finite-k response (job 3) -----------
def k_modes(cfg):
    """job 3's plane-wave modes of the cell (sseft.modes): [(axis, harmonic, phase)], harmonic by harmonic."""
    kc = cfg["liquid_k"]
    return [(int(ax), int(n), int(ph)) for n in kc["harmonics"] for ax in kc["axes"] for ph in kc["phases"]]


def liquid_response_k(mb, pos_ang, cell_ang, modes, delta=1e-4):
    """MB-pol's response to plane-wave fields of the cell (sseft.modes), clamped nuclei (MBX): for every mode, every
    molecule's induced dipole per unit amplitude (n, 3) A^3 (its sites O, H, H, M summed) and the mode's susceptibility
    sum_s u_s . dmu_s/da over the cell's own sites (A^3), by central differences of amplitude delta e/A^2 (the response
    is linear: the step only sets the rounding).  The field is set at every site MBX evaluates, a replica's included.
    Molecules in the frame's order."""
    n = len(pos_ang) // 3
    mb.set_external_field(pos_ang, None, cell_ang)
    X = mb.evaluated_sites(pos_ang, cell_ang)
    own = 4 * n
    dmu, chi = np.zeros((len(modes), n, 3)), np.zeros(len(modes))
    for m, (axis, harm, phase) in enumerate(modes):
        u = MD.mode_field(X, cell_ang, axis, harm, phase)[0]

        def induced(sign):
            mb.set_site_fields(pos_ang, sign * delta * u, cell_ang)
            return mb.sites(pos_ang, cell_ang)["induced"]                      # the cell's own sites, (4n, 3)
        d = (induced(1.0) - induced(-1.0)) / (2.0 * delta)
        dmu[m] = d.reshape(n, 4, 3).sum(axis=1)
        chi[m] = float(np.sum(u[:own] * d))
    mb.set_site_fields(pos_ang, None, cell_ang)
    return dmu, chi


def _parts(spec):
    if not spec:
        return 0, 1
    i, n = (int(x) for x in spec.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"--part i/n needs 0 <= i < n, got {spec}")
    return i, n


def part_path(path, i, n):
    stem, ext = os.path.splitext(path)
    return f"{stem}.part{i}of{n}{ext}"


def run_response_k(a):
    """MB-pol's finite-k response of job 2's liquid frames (the same calibration and held-out frames as the uniform
    response), every mode of liquid_k; --part i/n computes the i-th of n contiguous shares of the frames into a part
    file, --merge n joins the n part files into the response file."""
    cfg = load_config(a.config)
    path = a.out or cfg["data"]["response_k"]
    modes = k_modes(cfg)
    delta = float(cfg["liquid_k"]["delta"])
    frames = response_frames(cfg)
    if a.merge:
        parts = [np.load(part_path(path, i, a.merge), allow_pickle=False) for i in range(a.merge)]
        metas = [json.loads(str(p["meta"])) for p in parts]
        if len({json.dumps({k: m[k] for k in ("modes", "delta", "sseft")}, sort_keys=True) for m in metas}) != 1:
            raise ValueError("the part files were computed with different modes, steps or versions")
        arrays, meta = {}, {k: metas[0][k] for k in ("truth", "delta", "sseft", "modes", "modes_k_inv_angstrom")}
        meta["splits"] = {}
        for split, (fpath, idx) in frames.items():
            got = [(p[f"{split}_index"], p[f"{split}_dmu"], p[f"{split}_chi"]) for p in parts if f"{split}_index" in p]
            ix = np.concatenate([g[0] for g in got]) if got else np.zeros(0, int)
            if not np.array_equal(ix, idx):
                raise ValueError(f"the parts' {split} frames {ix.tolist()} are not the config's {idx.tolist()}")
            arrays[f"{split}_index"] = ix
            arrays[f"{split}_dmu"] = np.concatenate([g[1] for g in got])
            arrays[f"{split}_chi"] = np.concatenate([g[2] for g in got])
            pos = np.load(fpath)["positions"][idx]
            meta["splits"][split] = {"file": fpath, "frames": int(len(idx)), "positions_hash": frames_hash(pos)}
        arrays["meta"] = np.array(json.dumps(meta))
        np.savez(path, **arrays)
        print(f"written {path} from {a.merge} parts", flush=True)
        return
    i_part, n_parts = _parts(a.part)
    jobs = [(split, fpath, int(i)) for split, (fpath, idx) in frames.items() for i in idx]
    lo, hi = (len(jobs) * i_part) // n_parts, (len(jobs) * (i_part + 1)) // n_parts
    truth = W.MBPolTruth(a.mbx)
    print(f"MB-pol through MBX at {a.mbx}: {json.dumps(truth.describe())}", flush=True)
    data = {}
    cell0 = None
    t0 = time.time()
    for split, fpath, i in jobs[lo:hi]:
        d = np.load(fpath)
        pos = d["positions"][i]
        cell = d["cell"] if d["cell"].ndim == 2 else d["cell"][i]
        cell0 = cell if cell0 is None else cell0
        dmu, chi = liquid_response_k(truth.mb, pos, cell, modes, delta)
        rec = data.setdefault(split, {"index": [], "dmu": [], "chi": []})
        rec["index"].append(i); rec["dmu"].append(dmu); rec["chi"].append(chi)
        V = abs(np.linalg.det(cell))
        line = "; ".join(f"eps_L(k{n}) {1 / (1 - 4 * np.pi * np.mean([c for c, md in zip(chi, modes) if md[1] == n]) * 2 / V):.4f}"
                         for n in sorted({md[1] for md in modes}))
        print(f"  {split} frame {i}: {line}  [{time.time() - t0:.0f} s]", flush=True)
    meta = {"truth": truth.describe(), "delta": delta, "sseft": sseft.__version__, "modes": modes,
            "modes_k_inv_angstrom": [m["k"] for m in MD.describe(cell0, modes)] if cell0 is not None else [],
            "part": [i_part, n_parts], "splits": {}}
    arrays = {}
    for split, rec in data.items():
        arrays[f"{split}_index"] = np.array(rec["index"], int)
        arrays[f"{split}_dmu"] = np.array(rec["dmu"])
        arrays[f"{split}_chi"] = np.array(rec["chi"])
    arrays["meta"] = np.array(json.dumps(meta))
    out = path if n_parts == 1 else part_path(path, i_part, n_parts)
    if n_parts == 1:                                       # the whole file: its frames' hash, as the uniform response's
        for split, (fpath, idx) in frames.items():
            meta["splits"][split] = {"file": fpath, "frames": int(len(idx)),
                                     "positions_hash": frames_hash(np.load(fpath)["positions"][idx])}
        arrays["meta"] = np.array(json.dumps(meta))
    np.savez(out, **arrays)
    print(f"written {out}", flush=True)


def load_response_k(path, cfg, response=None):
    """job 3's finite-k response file, checked against the data sets' frames: {split: {"structures", "index", "dmu"
    (frames, modes, n, 3) bohr^3, "chi" (frames, modes) bohr^3}, "modes", "hash", "meta"}; the structures are the
    uniform response's (load_response) when it reads the same frames."""
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    K, om = int(cfg["model"]["K"]), float(cfg["model"]["omega0"])
    out = {"hash": hashlib.sha256(open(path, "rb").read()).hexdigest()[:8], "meta": meta,
           "modes": [tuple(int(x) for x in m) for m in meta["modes"]]}
    for split, info in meta["splits"].items():
        idx = d[f"{split}_index"]
        data = np.load(info["file"])
        if frames_hash(data["positions"][idx]) != info["positions_hash"]:
            raise ValueError(f"{path}: its {split} frames are not those of {info['file']} any more")
        if response is not None and split in response and np.array_equal(response[split]["index"], idx):
            structs = response[split]["structures"]
        else:
            all_structs = W.e4_structures(info["file"], pins=True, K=K, omega0=om)
            structs = [all_structs[i] for i in idx]
        a3 = lambda x: ang_to_bohr(ang_to_bohr(ang_to_bohr(x)))
        out[split] = {"structures": structs, "index": idx, "dmu": a3(d[f"{split}_dmu"]), "chi": a3(d[f"{split}_chi"])}
    return out


def order_zero(readout, n0):
    """the read-out with its orders >= 1 switched off: the pair coefficients' output columns beyond the order-0 ones
    (n0 = 2 n_rbf_readout) at zero, the order-0 columns and the node term as they are."""
    th = readout["theta"][-1]
    w, b = np.array(th["w"]), np.array(th["b"])
    w[:, n0:] = 0.0; b[n0:] = 0.0
    theta = list(readout["theta"]); theta[-1] = {"w": w, "b": b}
    return {**readout, "theta": theta}


def k_stage(rung_cell, resp_k, kq):
    """job 4's arm D (v0.0.28; notes/e4-job4-preregistration.md): the read-out's third matching stage, its orders >= 1
    moved on the far probes and on every calibration cell's susceptibility in every mode of the finite-k response,
    chi_mode = -d^2E/da^2 (model.mode_chi), against MB-pol's -- the derivative counting's second step: the k = 0 data set
    the order-0 coefficient (job 2's arm C), the k-dependence the orders >= 1 (notes/e4-job3-results.md section 6).
    kq: training.readout.finite_k -- weight (0: no finite-k rows, the orders >= 1 re-solved on the probes alone, a
    control).  The rows' unit is MB-pol's mean chi_mode at the mode's harmonic, over the calibration cells and modes, and
    their weight kq["weight"] / (cells x modes): weight 0.1 makes a 1 % error of the cells' finite-k susceptibility cost
    about the far probes' matched objective (~1e-5), as job 2's liquid weight does for the uniform response.  12 rows a
    cell, their Jacobian by reverse mode.  Returns {"block" (LIN.solve_readout's extra, or None), "hash", "scale",
    "anchor"}.  v0.0.29 (job 5's arm D'): kq["anchor"] true anchors the stage at its start -- the ridge toward the
    starting orders >= 1 and the far probes' rows held at the starting read-out's own predictions (LIN.solve_readout's
    anchor and hold) -- so that the orders >= 1 move from arm C's only where the finite-k rows need it.  Settings
    without "anchor" hash as before (arm D's cache is unchanged)."""
    tag = hashlib.sha256(json.dumps({**kq, "response_k": resp_k["hash"]}, sort_keys=True).encode()).hexdigest()[:8]
    anchor = bool(kq.get("anchor", False))
    if float(kq["weight"]) <= 0:
        return {"block": None, "hash": tag, "scale": {}, "anchor": anchor}
    cal = resp_k["calibration"]
    modes = resp_k["modes"]
    r = in_cell(replace(rung_cell, bf_solver="direct"), cal["structures"][0].cell)
    nf, nm = len(cal["structures"]), len(modes)
    chi_t = np.asarray(cal["chi"], float)                                 # (frames, modes), bohr^3
    harm = np.array([m[1] for m in modes])
    scale = {int(n): float(np.mean(chi_t[:, harm == n])) for n in sorted(set(harm.tolist()))}
    w = float(kq["weight"]) / (nf * nm) / np.array([scale[int(n)] ** 2 for n in harm])
    batches = []
    for st in cal["structures"]:
        b = {k: v[0] for k, v in S.pad_batch([st], n_freq=r.K).items()}
        pos, cell = np.asarray(b["positions"]), np.asarray(st.cell)
        mask = np.asarray(b["mask"], float)[:, None]
        fields = [MD.mode_field(pos, cell, axis, n, phase) for axis, n, phase in modes]
        batches.append({**b, "k_u": np.array([u * mask for u, _, _ in fields]),
                        "k_g": np.array([g * mask[..., None] for _, g, _ in fields]),
                        "k_h": np.array([h * mask[..., None, None] for _, _, h in fields])})

    def fn(params, b):
        bb = {k: v for k, v in b.items() if k not in ("k_u", "k_g", "k_h")}
        return jax.lax.map(lambda x: M.mode_chi(params, r, bb, *x), (b["k_u"], b["k_g"], b["k_h"]))
    block = {"fn": fn, "batches": batches, "targets": [chi_t[i] for i in range(nf)], "weights": [w] * nf}
    return {"block": block, "hash": tag, "scale": scale, "anchor": anchor}


def read_response_k(params, rung_cell, resp_k, split="heldout", eps0=None):
    """on liquid cells, the model's response to the plane-wave modes (model.mode_response) against MB-pol's, harmonic by
    harmonic: the cells' longitudinal susceptibility chi(k) = (chi_cos + chi_sin) / V averaged over the axes and
    eps_L(k) = 1 / (1 - 4 pi chi(k)), the model's and MB-pol's, their relative difference (frame by frame, then the
    mean); every molecule's induced dipole response (rmse / rms over molecules and modes, r, the norms' ratio).  eps0:
    the frames' eps_inf at k = 0 (model, MB-pol), for the screened limit chi(0+) = (1 - 1/eps0) / 4 pi."""
    h = resp_k[split]
    r = in_cell(replace(rung_cell, bf_solver="direct"), h["structures"][0].cell)
    f = jax.jit(lambda p, b, u, g, hh: M.mode_response(p, r, b, u, g, hh))
    modes = resp_k["modes"]
    harmonics = sorted({m[1] for m in modes})
    acc = {n: {"chi_m": [], "chi_t": [], "d_m": [], "d_t": []} for n in harmonics}
    for st, dmu_t, chi_t in zip(h["structures"], h["dmu"], h["chi"]):
        b = {k: v[0] for k, v in S.pad_batch([st], n_freq=r.K).items()}
        pos, cell = np.asarray(b["positions"]), np.asarray(st.cell)
        mask = np.asarray(b["mask"], float)[:, None]
        V = abs(np.linalg.det(cell))
        cm, ct = {n: 0.0 for n in harmonics}, {n: 0.0 for n in harmonics}
        naxes = {n: len({m[0] for m in modes if m[1] == n}) for n in harmonics}
        for j, (axis, n, phase) in enumerate(modes):
            u, g, hh = MD.mode_field(pos, cell, axis, n, phase)
            d_m, c_m = f(params, b, u * mask, g * mask[..., None], hh * mask[..., None, None])
            cm[n] += float(c_m) / (naxes[n] * V); ct[n] += float(chi_t[j]) / (naxes[n] * V)
            acc[n]["d_m"].append(np.asarray(d_m)[:len(dmu_t[j])]); acc[n]["d_t"].append(dmu_t[j])
        for n in harmonics:
            acc[n]["chi_m"].append(cm[n]); acc[n]["chi_t"].append(ct[n])
    out = {"frames": len(h["structures"]), "harmonics": {}}
    k_of = {m[1]: kk for m, kk in zip(modes, resp_k["meta"].get("modes_k_inv_angstrom", [np.nan] * len(modes)))}
    epsL = lambda c: float(1.0 / (1.0 - 4.0 * np.pi * c))
    for n in harmonics:
        cmn, ctn = np.array(acc[n]["chi_m"]), np.array(acc[n]["chi_t"])
        dm, dt = np.concatenate([x.ravel() for x in acc[n]["d_m"]]), np.concatenate([x.ravel() for x in acc[n]["d_t"]])
        out["harmonics"][str(n)] = {
            "k_inv_angstrom": float(k_of.get(n, np.nan)),
            "chi_model": float(cmn.mean()), "chi_truth": float(ctn.mean()),
            "eps_L_model": epsL(cmn.mean()), "eps_L_truth": epsL(ctn.mean()),
            "chi_rel_diff": float(np.mean(cmn / ctn - 1.0)), "chi_rel_diff_frames": (cmn / ctn - 1.0).tolist(),
            "molecular_rel_rmse": float(np.sqrt(np.mean((dm - dt) ** 2) / np.mean(dt ** 2))),
            "molecular_corr": float(np.corrcoef(dm, dt)[0, 1]),
            "molecular_norm_ratio": float(np.sqrt(np.mean(dm ** 2) / np.mean(dt ** 2)))}
    if eps0 is not None:
        out["chi_screened_k0"] = {"model": float((1 - 1 / eps0[0]) / (4 * np.pi)), "truth": float((1 - 1 / eps0[1]) / (4 * np.pi))}
    return out


def k_line(v):
    return "; ".join(
        f"k = {h['k_inv_angstrom']:.3f}/A: eps_L {h['eps_L_model']:.4f} against MB-pol's {h['eps_L_truth']:.4f} (chi "
        f"{100 * h['chi_rel_diff']:+.2f} %), molecules {100 * h['molecular_rel_rmse']:.1f} % (r {h['molecular_corr']:.3f}, "
        f"size {h['molecular_norm_ratio']:.3f})" for h in v["harmonics"].values())


def match_liquid_alone(name, rung_cell, rung_open, response, probes, scale, lc, out_dir=None):
    """job 3's locality test (eft-of-lss-workflow.md section 5): the read-out's order-0 columns and node term solved on
    MB-pol's liquid response alone -- every calibration molecule's induced dipole and field response (liquid_block,
    rows "molecules") -- from a read-out at zero, with no probe in the objective (LIN.solve_readout with w_energy =
    w_force = 0, the ridge relative to these rows).  The far probes and T5 read on it are then out-of-state predictions.
    lc: kread.locality (weight, ridge, iterations, alpha_ref_angstrom3, mu_ref_eangstrom, jac_chunk).  Cached in out_dir
    by the rows' settings and the response file."""
    lq = {"weight": float(lc.get("weight", 1.0)), "rows": "molecules",
          "alpha_ref_angstrom3": float(lc.get("alpha_ref_angstrom3", 1.4075)),
          "mu_ref_eangstrom": float(lc.get("mu_ref_eangstrom", 0.15)), "jac_chunk": int(lc.get("jac_chunk", 32))}
    block = liquid_block(rung_cell, response, lq)
    key = {"rung": repr(rung_open), "liquid": block["hash"], "ridge": float(lc["ridge"]),
           "iterations": int(lc["iterations"]), "scale": scale}
    tag = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
    path = os.path.join(out_dir, f"liquid_alone_{name}_{tag}.pkl") if out_dir else None
    if path and os.path.exists(path):
        with open(path, "rb") as fh:
            got = pickle.load(fh)
        print(f"  the read-out calibrated on the liquid alone before ({path}): the liquid rows "
              + "; ".join(f"{b:.3e} -> {c:.3e}" for b, c in got["match"]["liquid_objective"]), flush=True)
        return got["readout"], got["match"]
    params0 = M.init_params(jax.random.PRNGKey(0), rung_open, energy_scale=scale)
    data = S.pad_batch([probes["train"][0]], n_freq=rung_open.K)             # sizes the solve only: no probe row
    t0 = time.time()
    params, report = LIN.solve_readout(params0, rung_open, data, scale, w_force=0.0, w_energy=0.0,
                                       ridge=float(lc["ridge"]), orders="zero", iterations=int(lc["iterations"]),
                                       extra=block["block"])
    hist = [(float(b), float(c)) for b, c in report.extra_history]
    print(f"  {name}: the read-out's order-0 columns calibrated on the liquid alone, every molecule of "
          f"{len(block['block']['batches'])} cells (no probe): the liquid rows "
          + "; ".join(f"{b:.3e} -> {c:.3e}" for b, c in hist) + f" (ridge {float(lc['ridge']):.0e})  "
          f"[{time.time() - t0:.0f} s]", flush=True)
    readout = jax.tree_util.tree_map(np.asarray, params["readout"])
    match = {"liquid_objective": hist, "ridge": float(lc["ridge"]), "liquid_cells": len(block["block"]["batches"]),
             "liquid_hash": block["hash"], "time": time.time() - t0}
    if path:
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({"key": key, "readout": readout, "match": match}, fh)
        os.replace(tmp, path)
    return readout, match


def switched_off(readout):
    """the read-out with its output layers (pair coefficients and node term) at zero: the model without the read-out's
    energy, against which a read-out's own energy and forces are read (job 5's contact reading)."""
    theta, iso = list(readout["theta"]), list(readout["iso"])
    theta[-1] = {k: np.zeros_like(np.asarray(v)) for k, v in readout["theta"][-1].items()}
    iso[-1] = {k: np.zeros_like(np.asarray(v)) for k, v in readout["iso"][-1].items()}
    return {**readout, "theta": theta, "iso": iso}


def read_contact(variants, ref, scale, frames, n0):
    """job 5 (v0.0.29; notes/e4-job5-preregistration.md): what each read-out changes at contact before training, read
    as scratch/e4_job2_calibration_orders.py reads it -- on liquid frames of the training size, the model's energy per
    molecule and rigid forces with the read-out, against the reference variant's (ref) and against the read-out switched
    off (its own), E_0 and the field term at their initialisation cancelling in the differences -- and the read-out's
    output columns' change from the reference's, the orders >= 1 and the order 0 (with the node term) apart, relative
    to the reference's norms.  variants: kread's (label, rung_cell, rung_open, readout, match); those of another rung
    than ref's are skipped.  n0: the order-0 columns (2 n_rbf_readout).  Returns {label: record}."""
    by = {v[0]: v for v in variants}
    _, rc, ro, ref_readout, _ = by[ref]
    p0 = M.init_params(jax.random.PRNGKey(0), ro, energy_scale=scale)
    r = in_cell(replace(rc, bf_solver="direct"), frames[0].cell)
    f = jax.jit(lambda p, b: M.energy_forces_single(p, r, b))
    n_mol = frames[0].n_atoms // 3
    batches = [{k: v[0] for k, v in S.pad_batch([st], n_freq=r.K).items()} for st in frames]

    def ef(readout):
        p = with_readout(p0, readout)
        out = []
        for st, b in zip(frames, batches):
            E, F = f(p, b)
            out.append((float(E), np.asarray(F)[:st.n_atoms]))
        return out

    def change(a, b):
        dE, ss, cc = [], 0.0, 0.0
        for (Ea, Fa), (Eb, Fb), st in zip(a, b, frames):
            dE.append((Ea - Eb) / n_mol * KCAL)
            *_, s_, c_ = rigid_errors((Fa - Fb) * FORCE_UNIT, bohr_to_ang(np.asarray(st.positions)),
                                      bohr_to_ang(np.asarray(st.cell)), n_mol)
            ss += s_; cc += c_
        return {"energy_per_molecule": float(np.mean(dE)), "rigid_rms": float(np.sqrt(ss / cc))}

    def cols(readout):
        th, iso = readout["theta"][-1], readout["iso"][-1]
        w, b = np.asarray(th["w"]), np.asarray(th["b"])
        lo = [w[:, :n0].ravel(), b[:n0]] + [np.asarray(iso[k]).ravel() for k in sorted(iso)]
        return np.concatenate([w[:, n0:].ravel(), b[n0:]]), np.concatenate(lo)

    off, base = ef(switched_off(ref_readout)), ef(ref_readout)
    hi_ref, lo_ref = cols(ref_readout)
    rel = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y)) if np.linalg.norm(y) > 0 else float("nan")
    out = {}
    for label, _, ro_v, readout, _ in variants:
        if repr(ro_v) != repr(ro):
            continue
        mine = base if label == ref else ef(readout)
        hi, lo = cols(readout)
        out[label] = {"reference": ref, "frames": len(frames), "molecules": n_mol,
                      "own": change(mine, off), "against_reference": change(mine, base),
                      "coefficients": {"orders_above_zero": rel(hi, hi_ref), "order_zero": rel(lo, lo_ref)}}
    return out


def kread_arms(kc):
    """kread's arms (jobs 3-4): [(arm, its config -- the arm's config file with its "set" overrides merged (v0.0.28) --,
    its output directory)], and {arm: label} for the arms read with their orders >= 1 off (kread.order_zero_of: a list,
    labelled arm + "0", or a mapping arm -> label)."""
    oz = kc.get("order_zero_of") or []
    oz = {arm: f"{arm}0" for arm in oz} if isinstance(oz, (list, tuple)) else dict(oz)
    unknown = set(oz) - set(kc["arms"])
    if unknown:
        raise ValueError(f"kread.order_zero_of names arms not in kread.arms: {sorted(unknown)}")
    arms = [(arm, merge(load_config(spec["config"]), spec.get("set") or {}), spec["out_dir"])
            for arm, spec in kc["arms"].items()]
    return arms, oz


def run_kread(a):
    """job 3: the read-outs of kread.arms (their configs and output directories, where their matched read-outs are
    cached), read on MB-pol's finite-k response and the uniform one, with the far probes and T5; the orders >= 1
    switched off for the arms in kread.order_zero_of; and the locality test (kread.locality).  No training.  Job 4
    (v0.0.28): an arm's "set" overrides its config (merge), e.g. {model: {mutual_edge_angstrom: 1.5}}; an arm whose
    config has training.readout.finite_k is matched through arm D's third stage (k_stage) and its record printed;
    --out names the json (default out_dir/kread.json).  Job 5 (v0.0.29): kread.contact ({reference, frames,
    max_frames}) adds every variant's change at contact against the reference's (read_contact), before training."""
    cfg = load_config(a.config)
    kc, dcfg = cfg["kread"], cfg["data"]
    os.makedirs(a.out_dir, exist_ok=True)
    out_path = getattr(a, "out", None) or os.path.join(a.out_dir, "kread.json")
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    K, om = int(cfg["model"]["K"]), float(cfg["model"]["omega0"])
    train = W.e4_structures(os.path.join(dcfg["dir"], f"train_n{dcfg['train_size']}.npz"), pins=True, K=K, omega0=om)
    scale = T.energy_scale(train)
    probes = load_probes(dcfg["probes"], K, om)
    response = load_response(dcfg["response"], cfg)
    resp_k = load_response_k(dcfg["response_k"], cfg, response)
    name = kc["rung"]
    print(f"sseft {sseft.__version__}, JAX {jax.__version__}, devices {jax.devices()}; probes {probes['hash']}; liquid "
          f"response {response['hash']}; finite-k response {resp_k['hash']}: modes {resp_k['modes']} at "
          f"{sorted(set(round(x, 4) for x in resp_k['meta']['modes_k_inv_angstrom']))} /A", flush=True)
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    out["meta"] = {"sseft": sseft.__version__, "probes": probes["hash"], "response": response["hash"],
                   "response_k": resp_k["hash"], "rung": name, "modes": resp_k["modes"], "energy_scale": scale}
    variants = []
    wanted = (lambda label: True) if not a.only else (lambda label: label in a.only.split(","))
    arms, oz = kread_arms(kc)
    for arm, cfg_arm, out_dir_arm in arms:
        if not (wanted(arm) or (arm in oz and wanted(oz[arm]))):
            continue
        rung_cell, rung_open = rungs_for(name, cfg_arm, train[0].cell)
        lq = cfg_arm["training"]["readout"].get("liquid")
        kq = cfg_arm["training"]["readout"].get("finite_k")
        liquid = liquid_block(rung_cell, response, lq) if lq else None
        fk = k_stage(rung_cell, resp_k, kq) if kq else None
        os.makedirs(out_dir_arm, exist_ok=True)                    # v0.0.29: the arm's cache directory, if new
        readout, match = matched(name, rung_open, probes, scale, cfg_arm, out_dir_arm, liquid=liquid, finite_k=fk)
        if fk is not None:
            steps = match["objective"][-int(match.get("finite_k_stage_steps") or 1):]
            print(f"  {arm}: " + (f"the orders >= 1 on the finite-k response ({json.dumps(kq)}; MB-pol's mean chi_mode "
                                  + ", ".join(f"{v:.4g} bohr^3 at harmonic {n}" for n, v in fk["scale"].items()) + ")"
                                  if fk["block"] is not None else "the orders >= 1 re-solved on the probes alone (the "
                                  "control: no finite-k rows)")
                  + ": the stage's objective " + "; ".join(f"{b:.3e} -> {c:.3e}" for b, c in steps)
                  + ("; the finite-k rows " + "; ".join(f"{b:.3e} -> {c:.3e}" for b, c in match["finite_k_objective"])
                     if match.get("finite_k_objective") else ""), flush=True)
        variants.append((arm, rung_cell, rung_open, readout, match))
        if arm in oz:
            n0 = 2 * int(rung_open.e0.get("n_rbf_readout", rung_open.e0["n_rbf"]))
            variants.append((oz[arm], rung_cell, rung_open, order_zero(readout, n0), {}))
    if kc.get("locality") and wanted("L"):
        rung_cell, rung_open = rungs_for(name, cfg, train[0].cell)
        readout, match = match_liquid_alone(name, rung_cell, rung_open, response, probes, scale, kc["locality"], a.out_dir)
        variants.append(("L", rung_cell, rung_open, readout, match))
    for label, rung_cell, rung_open, readout, match in variants:
        if not wanted(label):
            continue
        t0 = time.time()
        params = with_readout(M.init_params(jax.random.PRNGKey(0), rung_open, energy_scale=scale), readout)
        rec = {"match": {k: v for k, v in (match or {}).items() if k not in ("probes_before_training",
                                                                              "far_trimers_before_training")}}
        rec["probes"] = (match or {}).get("probes_before_training") or read_probes(params, rung_open, probes)
        rec["trimers"] = (match or {}).get("far_trimers_before_training") or read_trimers(params, rung_open, probes)
        print(f"{label}: far-field probes {E3.probe_summary(rec['probes'])}; T5 slope {rec['trimers']['slope']:.4f} "
              f"({100 * rec['trimers']['three_body_rmse'] / rec['trimers']['three_body_rms']:.1f} %)", flush=True)
        for split in kc["splits"]:
            rec[f"liquid_{split}"] = v = read_liquid(params, rung_cell, response, split)
            rec[f"k_{split}"] = w = read_response_k(params, rung_cell, resp_k, split,
                                                    (v["eps_inf_model"], v["eps_inf_truth"]))
            print(f"  {label} {split}: {liquid_line(v)}\n  {label} {split} finite k: {k_line(w)}", flush=True)
        rec["time"] = time.time() - t0
        out[label] = rec
        with open(out_path + ".tmp", "w") as fh:
            json.dump(out, fh, indent=1)
        os.replace(out_path + ".tmp", out_path)
    ct = kc.get("contact")
    if ct:
        # job 5 (v0.0.29): each read-out's change at contact against the reference arm's, before training
        ref = ct["reference"]
        read = [v for v in variants if wanted(v[0])]
        if ref not in {v[0] for v in read}:
            print(f"  the contact reading needs its reference {ref} among the variants read: not read", flush=True)
        else:
            t0 = time.time()
            fset = ct.get("frames", "test")
            frames = W.e4_structures(os.path.join(dcfg["dir"], f"{fset}_n{dcfg['train_size']}.npz"), pins=True, K=K,
                                     omega0=om)
            if ct.get("max_frames"):
                frames = frames[:int(ct["max_frames"])]
            ro_ref = {v[0]: v for v in read}[ref][2]
            n0 = 2 * int(ro_ref.e0.get("n_rbf_readout", ro_ref.e0["n_rbf"]))
            got = read_contact(read, ref, scale, frames, n0)
            for label, c in got.items():
                out[label]["contact"] = c
                print(f"  {label} at contact ({c['frames']} {fset} frames of {c['molecules']} molecules): its own energy "
                      f"{c['own']['energy_per_molecule']:+.3f} kcal/mol per molecule, rigid forces "
                      f"{c['own']['rigid_rms']:.3f} kcal/mol/A rms; against {ref}'s: energy "
                      f"{c['against_reference']['energy_per_molecule']:+.3f}, rigid forces changed by "
                      f"{c['against_reference']['rigid_rms']:.3f} rms; its orders >= 1 moved by "
                      f"{c['coefficients']['orders_above_zero']:.3f} of {ref}'s norm, its order 0 by "
                      f"{c['coefficients']['order_zero']:.2e}", flush=True)
            out["meta"]["contact"] = {"reference": ref, "frames": fset, "n_frames": len(frames),
                                      "time": time.time() - t0}
            with open(out_path + ".tmp", "w") as fh:
                json.dump(out, fh, indent=1)
            os.replace(out_path + ".tmp", out_path)
    print(f"written {out_path}", flush=True)


# ---------------------------------------------------------------- the matching --------------------------------------
def with_readout(params, readout):
    """params with the read-out replaced (a matched one, as matched() returns it: NumPy arrays)."""
    return {**params, "readout": jax.tree_util.tree_map(jnp.asarray, readout)}


def match_readout(rung_open, probes, scale, cfg, verbose=True, liquid=None, start=None, finite_k=None):
    """the read-out's parameters matched on the training probes beyond r_c of their cores (E3-QP2's far_first stage,
    E3.far_field_blind): energies only, every order, Gauss-Newton steps, the ridge chosen on held-out cores.  With
    liquid (job 2's arm B, liquid_block) a second stage from that solution -- start, (read-out, record) as matched()
    keeps it, or solved first -- with MB-pol's induced dipole and field response of liquid cells as further rows
    (LIN.solve_readout's extra), the probes' ridge kept.  With finite_k (job 4's arm D, k_stage; v0.0.28) a third stage
    from start: the orders >= 1 alone on the probes and the calibration cells' finite-k susceptibilities, the probes'
    ridge kept; the liquid's k = 0 rows are not in it (they constrain the orders >= 1 only through the cells' dipole at
    zero field, which is what moved arm B's).  finite_k["anchor"] (job 5's arm D'; v0.0.29): the third stage anchored
    at start, the ridge toward its orders >= 1 and the probes held at its predictions (LIN.solve_readout's anchor and
    hold)."""
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
    hist_x, hist_k = [], []
    if liquid is not None:
        # job 2's calibration, a second stage from the far-field solution: the probes and the liquid cells' response
        # together, Gauss-Newton steps (halved where they overshoot), the probes' ridge kept
        rel = ridges[0] if ridges else float(ro["ridge"])
        params, report = LIN.solve_readout(params, rung_open, data, scale, w_force=0.0, ridge=rel,
                                           chunk=int(ro["solve_chunk"]),
                                           w_energy=float(cfg["training"]["loss"]["energy_weight"]),
                                           orders=liquid.get("orders") or ro["orders"],
                                           iterations=int(ro["solve_iterations"]), extra=liquid["block"])
        hist += [(float(b), float(c)) for b, c in report.history]
        hist_x = [(float(b), float(c)) for b, c in report.extra_history]
    n_k = 0
    anchored = finite_k is not None and bool(finite_k.get("anchor"))
    if finite_k is not None:
        # job 4's arm D: a third stage from start (arm C's read-out), the orders >= 1 alone on the probes and the
        # calibration cells' finite-k susceptibilities (none with weight 0: the control), the probes' ridge kept.
        # Job 5's arm D' (v0.0.29, anchored): the same rows and ridge, the ridge toward the start's orders >= 1 and the
        # probes' rows held at the start's own predictions; the record's objective is then the probes' against MB-pol
        # (with the finite-k rows), as arm D's is, and the held one is kept beside it
        rel = ridges[0] if ridges else float(ro["ridge"])
        kw = {} if finite_k["block"] is None else {"extra": finite_k["block"]}
        n_k = 0 if finite_k["block"] is None else len(finite_k["block"]["batches"])
        params, report = LIN.solve_readout(params, rung_open, data, scale, w_force=0.0, ridge=rel,
                                           chunk=int(ro["solve_chunk"]),
                                           w_energy=float(cfg["training"]["loss"]["energy_weight"]),
                                           orders="higher", iterations=int(ro["solve_iterations"]),
                                           anchor=anchored, hold=anchored, **kw)
        hist += [(float(b), float(c)) for b, c in (report.data_history if anchored else report.history)]
        hist_k = [(float(b), float(c)) for b, c in report.extra_history] if kw else []
    if verbose:
        print(f"  read-out matched on {len(idx)} of {len(probes['train'])} training probes (beyond r_c of their cores)"
              + (f" and {len(liquid['block']['batches'])} liquid cells' response" if liquid is not None else "")
              + (f", then its orders >= 1 on {n_k} cells' finite-k response" if finite_k is not None and n_k else "")
              + (" (anchored at the start: the ridge toward it, the probes held at its predictions)" if anchored else "")
              + (", then its orders >= 1 on the probes alone" if finite_k is not None and not n_k else "")
              + ": objective " + "; ".join(f"{b:.3e} -> {c:.3e}" for b, c in hist)
              + (" (the liquid rows: " + "; ".join(f"{b:.3e} -> {c:.3e}" for b, c in hist_x) + ")" if hist_x else "")
              + (" (the finite-k rows: " + "; ".join(f"{b:.3e} -> {c:.3e}" for b, c in hist_k) + ")" if hist_k else "")
              + (f" (ridge {ridges[0]:.0e})" if ridges else "") + f"  [{time.time() - t0:.0f} s]", flush=True)
    out = {"probes": int(len(idx)), "objective": hist, "ridge": ridges, "time": time.time() - t0}
    if liquid is not None:
        out.update(liquid_objective=hist_x, liquid_cells=len(liquid["block"]["batches"]), liquid_hash=liquid["hash"])
    elif start is not None:                                          # arm D: arm C's liquid record, kept
        out.update({k: v for k, v in start[1].items() if k.startswith("liquid_")})
    if finite_k is not None:
        out.update(finite_k_objective=hist_k, finite_k_cells=n_k, finite_k_hash=finite_k["hash"],
                   finite_k_stage_steps=len(report.history))
        if anchored:
            out.update(finite_k_anchor=True,
                       finite_k_held_objective=[(float(b), float(c)) for b, c in report.history])
    return params, out


def matched(name, rung_open, probes, scale, cfg, out_dir, liquid=None, finite_k=None):
    """the rung's matched read-out, from out_dir's cache (written once per rung and probe set, atomically, so that the
    tasks of an array job share it) or solved now; with the far-field test probes read on it before any training
    (E_0 at its initialisation: the probes at 10-15 A lie beyond r_c of every core atom, those at 8 A not all).
    liquid (job 2's calibration, liquid_block): MB-pol's response of liquid cells as further matching rows, from the
    far-field read-out (itself from the cache, or matched and cached first).  finite_k (job 4's arm D, k_stage): the
    orders >= 1 then moved on the finite-k response, from the read-out matched with liquid (cached the same way)."""
    tag = ("" if liquid is None else "_liq" + liquid["hash"]) + ("" if finite_k is None else "_k" + finite_k["hash"])
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
    if finite_k is not None:                                       # arm D: from arm C's (or, without liquid, arm A's)
        start = matched(name, rung_open, probes, scale, cfg, out_dir, liquid=liquid)
        params, match = match_readout(rung_open, probes, scale, cfg, start=start, finite_k=finite_k)
    else:
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
    kq, resp_k = tr["readout"].get("finite_k"), None
    if kq:                                                           # job 4's arm D (v0.0.28)
        if not (dcfg.get("response_k") and os.path.exists(dcfg["response_k"])):
            raise FileNotFoundError(f"training.readout.finite_k needs MB-pol's finite-k response {dcfg.get('response_k')}:"
                                    " scripts/e4_water.py response_k --mbx ...")
        resp_k = load_response_k(dcfg["response_k"], cfg, response)
        print(f"finite-k response {dcfg['response_k']} (hash {resp_k['hash']}): modes {resp_k['modes']}; the read-out's "
              f"orders >= 1 then calibrated on its calibration cells, {json.dumps(kq)}", flush=True)
    for name in rungs:
        rung, rung_open = rungs_for(name, cfg, train[0].cell)
        readout, match = None, None
        if rung.band_fields >= 0:
            liquid = liquid_block(rung, response, lq) if lq else None
            fk = k_stage(rung, resp_k, kq) if kq else None
            readout, match = matched(name, rung_open, probes, scale, cfg, out_dir, liquid=liquid, finite_k=fk)
            if response is not None and len(response.get("heldout", {}).get("structures", [])):
                t1 = time.time()
                p0 = with_readout(M.init_params(jax.random.PRNGKey(0), rung_open, energy_scale=scale), readout)
                match = {**match, "liquid_before_training": read_liquid(p0, rung, response)}
                print(f"  {name}: {liquid_line(match['liquid_before_training'])} on the matched read-out, before training"
                      f"  [{time.time() - t1:.0f} s]", flush=True)
                if resp_k is not None:
                    t1 = time.time()
                    v = match["liquid_before_training"]
                    match["k_before_training"] = read_response_k(p0, rung, resp_k, "heldout",
                                                                 (v["eps_inf_model"], v["eps_inf_truth"]))
                    print(f"  {name}: finite k (held out) {k_line(match['k_before_training'])} on the matched read-out, "
                          f"before training  [{time.time() - t1:.0f} s]", flush=True)
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
    q = sub.add_parser("response_k", help="MB-pol's response of liquid cells to plane-wave fields (job 3), through MBX")
    q.add_argument("--mbx", default=None, help="the MBX build (not needed with --merge)")
    q.add_argument("--config", default=None)
    q.add_argument("--out", default=None, help="the finite-k response file (default: the config's data.response_k)")
    q.add_argument("--part", default=None, help="i/n: the i-th of n shares of the frames, into a part file")
    q.add_argument("--merge", type=int, default=0, help="n: join the n part files into the response file")
    k = sub.add_parser("kread", help="jobs 3-5: read-outs on the finite-k and uniform responses, the locality test, the "
                                         "change at contact")
    k.add_argument("--config", default=None)
    k.add_argument("--out_dir", default="results/e4_job3")
    k.add_argument("--only", default=None, help="comma-separated variants to read (default: all)")
    k.add_argument("--out", default=None, help="the json (default: out_dir/kread.json); v0.0.28")
    a = ap.parse_args()
    if a.cmd == "response_k" and not a.merge and not a.mbx:
        ap.error("response_k needs --mbx (or --merge n)")
    {"probes": run_probes, "train": run_train, "summary": run_summary, "response": run_response,
     "response_k": run_response_k, "kread": run_kread}[a.cmd](a)


if __name__ == "__main__":
    main()
