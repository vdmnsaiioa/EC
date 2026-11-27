"""
e4_job18.py -- E4 job 18 (notes/e4-job18-preregistration.md; v0.0.43): from a few molecules to many, MB-pol standing in
for CCSD(T).  Every model is trained on clusters of at most six molecules -- what CCSD(T) affords -- and read on clusters
of 6 to 256 molecules cut from the liquid.

    python scripts/e4_job18.py make --mbx external/mbx/build/MBX              # CPU: the clusters and MB-pol's labels
    python scripts/e4_job18.py train --config configs/e4_job18.yaml --rungs M_inf --seeds 0 --out_dir results/e4_job18
    python baselines/mace/energies.py <model> data/e4/job18_test.extxyz data/e4/probes_mbpol_far.extxyz --tag job18
    python scripts/e4_job18.py read                                         # NumPy only

`make` writes data/e4/job18_clusters.npz and, for MACE, data/e4/job18_{train,valid,test}.extxyz:
  - training clusters of 1-6 molecules from the 45 training frames of 256 molecules (data/e4/train_n256.npz): for each
    size from 2, half compact (a molecule and its nearest neighbours) and half sparse (a molecule and others drawn at
    random within SPARSE_RADIUS of it), and isolated monomers; MB-pol's interaction energies and forces; every tenth one
    a validation cluster (MACE's), the rest the training set of every model;
  - calibration clusters: compact training clusters of 3-6 molecules, with MB-pol's induced dipole and its response to
    a uniform field (clamped nuclei) -- the small-cluster response a CCSD calculation affords;
  - test clusters, compact, of 6, 16, 32, 64, 128 and 256 molecules, from the test frames of 256, 512 and 1024 molecules:
    MB-pol's interaction energy and its components, and up to 64 molecules the sum of its pair energies.

`train` trains harness rungs on the training clusters, open (scripts/e4_water.py's rungs, losses and fit): a rung with a
read-out has it matched first on the far probes (E3-QP2's matching, small clusters too) and, with job18.calibration set,
then on the calibration clusters' response (job 2's second stage on clusters instead of liquid cells), and held through
training.  Each run is read on the test clusters, the far probes and T5; one json and one .params.pkl per rung and seed.

`read` reads K18.1-K18.7 (the pre-registration's section 5) from the runs' jsons and MACE's energies.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_job17 as J17                                       # noqa: E402  (the cluster geometry, NumPy only)

KCAL = 627.5094740631
EV_PER_HARTREE = 27.211386245988
BOHR_PER_ANGSTROM = 1.8897259886
EV_ANG_PER_HARTREE_BOHR = EV_PER_HARTREE * BOHR_PER_ANGSTROM

TRAIN_SOURCE = "data/e4/train_n256.npz"
TEST_SOURCES = {6: "data/e4/test_n256.npz", 16: "data/e4/test_n256.npz", 32: "data/e4/test_n256.npz",
                64: "data/e4/test_n256.npz", 128: "data/e4/test_n512.npz", 256: "data/e4/test_n1024.npz"}
OUT = "data/e4/job18_clusters.npz"
XYZ = {"train": "data/e4/job18_train.extxyz", "valid": "data/e4/job18_valid.extxyz", "test": "data/e4/job18_test.extxyz"}
DONE = "data/e4/job18_done.json"
FAR_XYZ = "data/e4/probes_mbpol_far.extxyz"
PROBES = "data/e4/probes_mbpol.npz"

TRAIN_SIZES = (2, 3, 4, 5, 6)
TRAIN_PER_SIZE = 400                     # half compact, half sparse
MONOMERS = 20
SPARSE_RADIUS = 9.0                      # A, O-O from the cluster's first molecule
VALID_EVERY = 10
CALIBRATION = 60                         # compact training clusters of 3-6 molecules
CAL_SIZES = (3, 4, 5, 6)
TEST_PER_SIZE = 16
PAIR_SUM_MAX = 64                        # the truth's pair sums up to this size
SEED = 1800
DELTA = 1e-4                             # e/A^2, the field step of the response

TEST_SIZES = tuple(sorted(TEST_SOURCES))
FAR_SIZES = (128, 256)                   # K18.2's sizes
K18_3, K18_4, K18_5 = 2.0, 0.15, 2.0
# the runs read: role -> (directory, rung)
ROLES = {"model": ("results/e4_job18", "M_16C2pm_us_d20"),        # E4's model's rung, its response calibrated on clusters
         "model_probes": ("results/e4_job18p", "M_16C2pm_us_d20"),  # the same, its read-out matched on the probes alone
         "dagger": ("results/e4_job18", "M_16pm_us_d20"),
         "learned": ("results/e4_job18", "M_1w100"),
         "cutoff": ("results/e4_job18", "M_inf")}
MACE_GLOB = "results/e4_job18_mace/*_s*/*_stagetwo_job18.npz"


# ---------------------------------------------------------------- the clusters (NumPy) ------------------------------
def frames_of(path):
    d = np.load(path, allow_pickle=False)
    pos = np.asarray(d["positions"], float)
    cell = np.asarray(d["cell"], float)
    L = J17.cubic_length(cell if cell.ndim == 2 else cell[0])
    return pos, L, d


def placed(m, dO, center, idx):
    """whole molecules idx of a frame (m: (n, 3, 3) whole) at their minimum image about the centre's O."""
    return (m[idx] + (m[center, 0] + dO[idx] - m[idx, 0])[:, None, :]).reshape(-1, 3)


def cut_sparse(pos, L, center, n, rng, radius=SPARSE_RADIUS):
    """the molecule `center` and n - 1 others drawn at random among those whose O lies within radius of its O (minimum
    image), each whole at its minimum image about the centre; (3n, 3) A."""
    m = J17.whole_molecules(pos, L)
    dO = m[:, 0] - m[center, 0]
    dO -= np.round(dO / L) * L
    near = np.flatnonzero((np.linalg.norm(dO, axis=1) <= radius) & (np.arange(len(m)) != center))
    if len(near) < n - 1:
        raise ValueError(f"only {len(near)} molecules within {radius} A")
    idx = np.concatenate([[center], rng.choice(near, size=n - 1, replace=False)])
    return placed(m, dO, center, idx)


def compact(pos, L, center, n):
    """J17.cut_cluster, checked: every molecule within half the cell of the centre (the minimum image unambiguous)."""
    c = J17.cut_cluster(pos, L, center, n)
    r = np.linalg.norm(c.reshape(-1, 3, 3)[:, 0] - c[0], axis=1).max()
    if r >= 0.5 * L - 1.0:
        raise ValueError(f"a {n}-molecule cluster reaches {r:.1f} A in a cell of {L:.1f} A")
    return c


def training_clusters(pos, L, seed=SEED, sizes=TRAIN_SIZES, per_size=TRAIN_PER_SIZE, monomers=MONOMERS):
    """[(positions (3n, 3), kind, frame, centre)]: monomers, then per size half compact and half sparse."""
    rng = np.random.default_rng(seed)
    F, n_mol = len(pos), pos.shape[1] // 3
    out = []
    for _ in range(monomers):
        f, c = int(rng.integers(F)), int(rng.integers(n_mol))
        out.append((compact(pos[f], L, c, 1), "monomer", f, c))
    for n in sizes:
        for k in range(per_size):
            f, c = int(rng.integers(F)), int(rng.integers(n_mol))
            if k % 2 == 0:
                out.append((compact(pos[f], L, c, n), "compact", f, c))
            else:
                out.append((cut_sparse(pos[f], L, c, n, rng), "sparse", f, c))
    return out


def test_clusters(sources, seed, per_size):
    """[(positions, size, frame, centre)] per size, compact, from that size's source frames (distinct frame-centre
    pairs)."""
    rng = np.random.default_rng(seed)
    out = []
    cache = {}
    for n, path in sorted(sources.items()):
        if path not in cache:
            cache[path] = frames_of(path)[:2]
        pos, L = cache[path]
        F, n_mol = len(pos), pos.shape[1] // 3
        seen = set()
        while sum(1 for x in out if x[1] == n) < per_size:
            f, c = int(rng.integers(F)), int(rng.integers(n_mol))
            if (f, c) in seen:
                continue
            seen.add((f, c))
            out.append((compact(pos[f], L, c, n), n, f, c))
    return out


def ragged(arrs):
    """(concatenated (sum, ...), offsets (len + 1,))"""
    off = np.concatenate([[0], np.cumsum([len(a) for a in arrs])]).astype(int)
    return np.concatenate(arrs) if arrs else np.zeros((0, 3)), off


def unragged(cat, off):
    return [cat[off[k]:off[k + 1]] for k in range(len(off) - 1)]


def cluster_response(mb, pos, delta=DELTA):
    """MB-pol's induced dipole of an open cluster at zero field (3,) e A and its response to a uniform field (3, 3) A^3,
    clamped nuclei, by central differences (sseft.mbpol.MBPol: set_external_field, sites)."""
    ind = lambda: np.asarray(mb.sites(pos)["induced"]).sum(axis=0)
    mb.set_external_field(pos, None)
    M0 = ind()
    R = np.zeros((3, 3))
    for k in range(3):
        d = np.zeros(3); d[k] = delta
        mb.set_external_field(pos, d); plus = ind()
        mb.set_external_field(pos, -d); minus = ind()
        R[:, k] = (plus - minus) / (2 * delta)
    mb.set_external_field(pos, None)
    return M0, R


def write_extxyz(path, frames, energies_ev, forces_ev=None, tags=None):
    """open frames (pbc F F F) with REF_energy (eV) and, given, REF_forces (eV/A)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for k, pos in enumerate(frames):
            sym = np.tile(["O", "H", "H"], len(pos) // 3)
            props = "species:S:1:pos:R:3" + (":REF_forces:R:3" if forces_ev is not None else "")
            head = f'Properties={props} pbc="F F F" REF_energy={energies_ev[k]:.12f}'
            if tags is not None:
                head += "".join(f" {a}={b}" for a, b in tags[k].items())
            fh.write(f"{len(pos)}\n{head}\n")
            for i, (s, (x, y, z)) in enumerate(zip(sym, pos)):
                line = f"{s} {x:.10f} {y:.10f} {z:.10f}"
                if forces_ev is not None:
                    fx, fy, fz = forces_ev[k][i]
                    line += f" {fx:.10f} {fy:.10f} {fz:.10f}"
                fh.write(line + "\n")
    os.replace(tmp, path)
    return path


def run_make(a):
    from sseft.mbpol import MBPol
    make(MBPol(a.mbx), a)


def make(truth, a):
    """the clusters and the truth's labels (truth: energy, energy_forces, components, sites, set_external_field as
    sseft.mbpol.MBPol's)."""
    t0 = time.time()
    pos, L, d = frames_of(a.train_source)
    site_q, gamma = np.asarray(d["site_q"]), float(d["site_gamma"])
    tr = training_clusters(pos, L, SEED, TRAIN_SIZES, TRAIN_PER_SIZE, MONOMERS)
    E, F = [], []
    for k, (p, *_) in enumerate(tr):
        e, f = truth.energy_forces(p)
        E.append(e); F.append(np.asarray(f))
        if (k + 1) % 500 == 0:
            print(f"  training clusters labelled: {k + 1} of {len(tr)}  [{time.time() - t0:.0f} s]", flush=True)
    E = np.array(E)
    valid = np.array([k % VALID_EVERY == VALID_EVERY - 1 for k in range(len(tr))])
    n_of = np.array([len(p) // 3 for p, *_ in tr])
    kinds = np.array([k for _, k, _, _ in tr])
    cand = [k for k in range(len(tr)) if not valid[k] and kinds[k] == "compact" and n_of[k] in CAL_SIZES]
    rng = np.random.default_rng(SEED + 2)
    cal = np.sort(rng.choice(cand, size=min(CALIBRATION, len(cand)), replace=False))
    Mc, Rc = [], []
    for k in cal:
        m0, r = cluster_response(truth, tr[k][0], DELTA)
        Mc.append(m0); Rc.append(r)
    print(f"  {len(tr)} training clusters ({int(valid.sum())} for validation), {len(cal)} calibration clusters' response"
          f"  [{time.time() - t0:.0f} s]", flush=True)
    sources = getattr(a, "test_sources", TEST_SOURCES)
    te = test_clusters({n: sources[n] for n in TEST_SIZES}, SEED + 1, TEST_PER_SIZE)
    Et, comp, pair = [], {t: [] for t in J17.TERMS}, []
    for k, (p, n, f, c) in enumerate(te):
        if k % TEST_PER_SIZE == 0:
            print(f"  test clusters of {n} molecules: labelling  [{time.time() - t0:.0f} s]", flush=True)
        cm = truth.components(p)
        Et.append(truth.energy(p))
        for t in J17.TERMS:
            comp[t].append(cm[t])
        if n <= PAIR_SUM_MAX:
            m = p.reshape(-1, 3, 3)
            pair.append(sum(truth.energy(np.concatenate([m[i], m[j]])) for i in range(n) for j in range(i + 1, n)))
        else:
            pair.append(np.nan)
    Et, pair = np.array(Et), np.array(pair)
    for n in TEST_SIZES:
        s = [k for k, x in enumerate(te) if x[1] == n]
        e = Et[s] / n * KCAL
        nadd = ((Et[s] - pair[s]) / Et[s]) if n <= PAIR_SUM_MAX else None
        print(f"  test clusters of {n}: interaction per molecule {e.mean():.3f} +- {e.std():.3f} kcal/mol"
              + (f", beyond pairs {100 * np.mean(nadd):.1f} %" if nadd is not None else "")
              + f", induced {100 * np.mean(np.array(comp['induced'])[s] / Et[s]):.1f} %  [{time.time() - t0:.0f} s]",
              flush=True)
    P_tr, off_tr = ragged([p for p, *_ in tr])
    F_tr, _ = ragged(F)
    P_te, off_te = ragged([p for p, *_ in te])
    arrays = {"train_positions": P_tr, "train_offsets": off_tr, "train_energy": E, "train_forces": F_tr,
              "train_valid": valid, "train_kind": kinds.astype("U8"), "train_source": np.array([[f, c] for *_, f, c in tr]),
              "calibration_index": cal, "calibration_M": np.array(Mc), "calibration_R": np.array(Rc),
              "test_positions": P_te, "test_offsets": off_te, "test_energy": Et, "test_size": np.array([x[1] for x in te]),
              "test_source": np.array([[f, c] for _, _, f, c in te]), "test_pair_sum": pair,
              "site_q": site_q, "site_gamma": np.array(gamma)}
    for t in J17.TERMS:
        arrays[f"test_{t}"] = np.array(comp[t])
    digest = J17.arrays_hash({k: v for k, v in arrays.items() if k.endswith(("positions", "energy"))})
    meta = {"hash": digest, "train_source": a.train_source, "test_sources": {str(n): sources[n] for n in TEST_SIZES},
            "train_sizes": list(TRAIN_SIZES), "train_per_size": TRAIN_PER_SIZE, "monomers": MONOMERS,
            "sparse_radius": SPARSE_RADIUS, "test_sizes": list(TEST_SIZES), "test_per_size": TEST_PER_SIZE,
            "seed": SEED, "delta": DELTA}
    if hasattr(truth, "describe"):
        meta["truth"] = truth.describe()
    tmp = a.out + ".tmp.npz"
    np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, a.out)
    Fev = [f * EV_ANG_PER_HARTREE_BOHR for f in F]
    for split, sel in (("train", ~valid), ("valid", valid)):
        idx = np.flatnonzero(sel)
        write_extxyz(a.xyz[split], [tr[k][0] for k in idx], E[idx] * EV_PER_HARTREE, [Fev[k] for k in idx],
                     tags=[{"n_mol": int(n_of[k]), "kind": str(kinds[k])} for k in idx])
    write_extxyz(a.xyz["test"], [p for p, *_ in te], Et * EV_PER_HARTREE, tags=[{"n_mol": int(n)} for _, n, _, _ in te])
    if not os.path.exists(a.far_xyz) and os.path.exists(a.probes):
        import e4_far
        e4_far.write_extxyz(a.probes, a.far_xyz)
    with open(a.done, "w") as fh:
        json.dump({"hash": digest, "seconds": time.time() - t0}, fh)
    print(f"written {a.out} (hash {digest}), {', '.join(a.xyz.values())}; done  [{time.time() - t0:.0f} s]", flush=True)


def load_clusters(path=OUT):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    out = {"meta": meta, "hash": meta["hash"], "site_q": np.asarray(d["site_q"]), "site_gamma": float(d["site_gamma"])}
    P = unragged(np.asarray(d["train_positions"]), np.asarray(d["train_offsets"]))
    Fo = unragged(np.asarray(d["train_forces"]), np.asarray(d["train_offsets"]))
    valid = np.asarray(d["train_valid"], bool)
    out["train"] = [(P[k], float(d["train_energy"][k]), Fo[k]) for k in range(len(P)) if not valid[k]]
    out["train_index"] = np.flatnonzero(~valid)
    cal = np.asarray(d["calibration_index"], int)
    out["calibration"] = {"positions": [P[k] for k in cal], "M": np.asarray(d["calibration_M"]),
                          "R": np.asarray(d["calibration_R"])}
    out["test"] = {"positions": unragged(np.asarray(d["test_positions"]), np.asarray(d["test_offsets"])),
                   "energy": np.asarray(d["test_energy"]), "size": np.asarray(d["test_size"], int),
                   "pair_sum": np.asarray(d["test_pair_sum"]),
                   **{t: np.asarray(d[f"test_{t}"]) for t in J17.TERMS}}
    return out


# ---------------------------------------------------------------- train (JAX, GPU) ----------------------------------
def calibration_block(rung_open, C, cfg_cal, K, omega0):
    """job 2's calibration rows (scripts/e4_water.py liquid_block, rows "cells") on open clusters instead of liquid
    cells: every calibration cluster's induced dipole M (3) and field response R (3, 3) against MB-pol's, weighted per
    cluster relative to n alpha_ref (R) and sqrt(n) mu_ref (M), the block's weight shared over the clusters."""
    import jax.numpy as jnp
    from sseft import model as M, structure as S, water as W
    from sseft.units import ang_to_bohr
    structs = W.pin_rigid_site_model([W.water_structure(p, pinned_q=False) for p in C["calibration"]["positions"]],
                                     C["site_q"], C["site_gamma"], K, omega0)
    a_ref = ang_to_bohr(ang_to_bohr(ang_to_bohr(float(cfg_cal.get("alpha_ref_angstrom3", 1.4075)))))
    m_ref = ang_to_bohr(float(cfg_cal.get("mu_ref_eangstrom", 0.15)))
    nf = len(structs)
    batches = [{k: v[0] for k, v in S.pad_batch([st], n_freq=rung_open.K).items()} for st in structs]
    weights, targets = [], []
    for st, m0, r in zip(structs, C["calibration"]["M"], C["calibration"]["R"]):
        n = st.n_atoms // 3
        s_R, s_M = n * a_ref, math.sqrt(n) * m_ref
        weights.append(float(cfg_cal["weight"]) / nf * np.concatenate([np.full(3, 1.0 / (3 * s_M ** 2)),
                                                                         np.full(9, 1.0 / (9 * s_R ** 2))]))
        targets.append(np.concatenate([ang_to_bohr(np.asarray(m0)), ang_to_bohr(ang_to_bohr(ang_to_bohr(
            np.asarray(r)))).reshape(-1)]))

    def fn(params, b):
        Mi, R = M.induced_response(params, rung_open, b)
        return jnp.concatenate([Mi, R.reshape(-1)])
    tag = J17.arrays_hash({"M": C["calibration"]["M"], "R": C["calibration"]["R"],
                           "w": np.array([float(cfg_cal["weight"])])})[:8]
    return {"block": {"fn": fn, "batches": batches, "targets": targets, "weights": weights}, "hash": "cl" + tag,
            "orders": cfg_cal.get("orders")}


def read_calibration(params, rung_open, block):
    """the calibration clusters' response on a model: R's trace and M against MB-pol's, relative rms."""
    import jax
    f = jax.jit(block["block"]["fn"])
    got = np.array([np.asarray(f(params, b)) for b in block["block"]["batches"]])
    tgt = np.array(block["block"]["targets"])
    tr = lambda x: x[:, 3:].reshape(-1, 3, 3).trace(axis1=1, axis2=2)
    m_rms = float(np.sqrt(np.mean(tgt[:, :3] ** 2)))
    return {"trace_ratio_mean": float(np.mean(tr(got) / tr(tgt))),
            "M_rel_rms": float(np.sqrt(np.mean((got[:, :3] - tgt[:, :3]) ** 2))) / m_rms if m_rms > 0 else None}


def load_config(path):
    """(scripts/e4_water.py's configuration with the file's sections over its defaults, the file's job18 section)."""
    import yaml
    import e4_water as E4
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    j18 = raw.pop("job18", None) or {}
    unknown = set(raw) - set(E4.DEFAULTS) - {"experiment"}
    if unknown:
        raise ValueError(f"unknown sections in {path}: {sorted(unknown)}")
    return E4.merge(E4.DEFAULTS, {k: v for k, v in raw.items() if k != "experiment"}), j18


def run_train(a):
    import pickle
    import jax
    import e4_water as E4
    import sseft
    from sseft import train as T, water as W
    cfg, j18 = load_config(a.config)
    rungs = a.rungs.split(",") if a.rungs else list(cfg["rungs"])
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else [int(s) for s in cfg["seeds"]]
    C = load_clusters(j18.get("clusters", OUT))
    K, om = int(cfg["model"]["K"]), float(cfg["model"]["omega0"])
    pin = lambda P: W.pin_rigid_site_model([W.water_structure(p, pinned_q=False) for p in P], C["site_q"],
                                           C["site_gamma"], K, om)
    train = W.pin_rigid_site_model([W.water_structure(p, energy=e, forces=f, pinned_q=False)    # hartree, hartree/bohr
                                    for p, e, f in C["train"]], C["site_q"], C["site_gamma"], K, om)
    test = pin(C["test"]["positions"])
    scale = T.energy_scale(train)
    probes = E4.load_probes(cfg["data"]["probes"], K, om)
    src = j18.get("train_source", C["meta"]["train_source"])          # the cell only sizes the rung's periodic twin
    cell = np.eye(3) * J17.cubic_length(np.asarray(np.load(src)["cell"]).reshape(-1, 3, 3)[0]) * BOHR_PER_ANGSTROM
    tr = cfg["training"]
    ad, lb, ls, ro = tr["adam"], tr["lbfgs"], tr["loss"], tr["readout"]
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"sseft {sseft.__version__}, JAX {jax.__version__}, {jax.devices()}; {len(train)} training clusters "
          f"(hash {C['hash']}), energy scale {scale * KCAL:.4f} kcal/mol per atom; {len(test)} test clusters; probes "
          f"{cfg['data']['probes']} (hash {probes['hash']})", flush=True)
    for name in rungs:
        _, rung_open = E4.rungs_for(name, cfg, cell)
        rung_open = E4.warm_started(name, rung_open, train)
        readout, match, block = None, None, None
        if rung_open.band_fields >= 0:
            if j18.get("calibration"):
                block = calibration_block(rung_open, C, j18["calibration"], K, om)
            readout, match = E4.matched(name, rung_open, probes, scale, cfg, a.out_dir, liquid=block)
        for seed in seeds:
            t0 = time.time()
            stem = os.path.join(a.out_dir, f"{name}_seed{seed}")
            if os.path.exists(stem + ".json"):
                print(f"{name} seed {seed}: done already ({stem}.json)", flush=True)
                continue
            extra = {}
            if E4.CHARGE_LR.get(name, 1.0) != 1.0:
                extra["charge_lr_scale"] = float(E4.CHARGE_LR[name])
            print(f"{name} seed {seed}: training on {len(train)} clusters, open"
                  + (", the read-out held" + (" (calibrated on the clusters' response)" if block else "")
                     if readout is not None else ""), flush=True)
            params, info = T.fit(rung_open, train, seed=seed, steps=int(ad["steps"]), lr=float(ad["learning_rate"]),
                                 w_energy=float(ls["energy_weight"]), w_force=float(ls["force_weight"]),
                                 batch_size=int(ad["batch_size"]), lbfgs_steps=int(lb["steps"]),
                                 lbfgs_chunk=int(lb["chunk_size"]), lr_final_ratio=float(ad["final_learning_rate_ratio"]),
                                 adam_b1=float(ad["beta1"]), adam_b2=float(ad["beta2"]), adam_eps=float(ad["epsilon"]),
                                 scale=scale, lbfgs_memory=int(lb["memory_size"]),
                                 lbfgs_max_linesearch_steps=int(lb["max_linesearch_steps"]), readout_init=readout,
                                 force_mode=ls["forces"], verbose=True,
                                 lbfgs_max_seconds=60.0 * float(lb["max_minutes"]) if lb.get("max_minutes") else None,
                                 **extra)
            t_train = time.time() - t0
            t1 = time.time()
            E_test = np.zeros(len(test))
            for n in sorted(set(C["test"]["size"].tolist())):
                idx = np.flatnonzero(C["test"]["size"] == n)
                E_test[idx] = T.predict_energies(params, rung_open, [test[i] for i in idx],
                                                 chunk=max(1, int(a.chunk_atoms) // (3 * n)))
            rd = {"test_energy": E_test.tolist(), "test_seconds": time.time() - t1}
            err = (E_test - C["test"]["energy"]) * KCAL / C["test"]["size"]
            print(f"  {name} seed {seed}: the test clusters, error per molecule rms (mean) by size "
                  + ", ".join(f"{n} {np.sqrt(np.mean(err[C['test']['size'] == n] ** 2)):.4f} "
                              f"({np.mean(err[C['test']['size'] == n]):+.4f})" for n in TEST_SIZES)
                  + f" kcal/mol  [{rd['test_seconds']:.0f} s]", flush=True)
            rd["probes"] = E4.read_probes(params, rung_open, probes)
            if "trimers" in probes:
                rd["far_trimers"] = E4.read_trimers(params, rung_open, probes)
            if block is not None:
                rd["calibration"] = read_calibration(params, rung_open, block)
            E_tr = T.predict_energies(params, rung_open, train, chunk=256)
            rd["train_energy_rmse_per_molecule"] = float(np.sqrt(np.mean(
                ((E_tr - np.array([s.energy for s in train])) * KCAL / np.array([s.n_atoms // 3 for s in train])) ** 2)))
            result = {"rung": name, "seed": seed, "model": E4.model_record(name, cfg, rung_open), "config": cfg,
                      "job18": j18,
                      "sseft": sseft.__version__, "jax": jax.__version__, "devices": str(jax.devices()),
                      "clusters_hash": C["hash"], "probes_hash": probes["hash"], "energy_scale": scale,
                      "training_clusters": len(train), "matching": match,
                      "training": {"time": t_train, "fit_time": info["time"], "history": info["history"],
                                   "options": info["training_options"]},
                      "readings": rd}
            with open(stem + ".params.pkl", "wb") as fh:
                pickle.dump(jax.tree_util.tree_map(np.asarray, params), fh)
            with open(stem + ".json", "w") as fh:
                json.dump(result, fh, indent=1)
            print(f"  {name} seed {seed}: {E3_line(rd)}  [{time.time() - t0:.0f} s in all; {stem}.json]", flush=True)


def E3_line(rd):
    p = rd["probes"]["by_distance_angstrom"]
    s = "far probes non-additive left " + " / ".join(
        f"{100 * p[d]['nonadditive_rmse'] / p[d]['nonadditive_rms']:.1f}" for d in ("10", "12", "15") if d in p) + " %"
    if rd.get("far_trimers"):
        s += f", T5 slope {rd['far_trimers']['slope']:.4f}"
    if rd.get("calibration"):
        c = rd["calibration"]
        s += (f", the calibration clusters' response {c['trace_ratio_mean']:.4f} of MB-pol's (trace)"
              + (f", dipole {100 * c['M_rel_rms']:.1f} %" if c["M_rel_rms"] is not None else ""))
    return s + f", training energy rms {rd['train_energy_rmse_per_molecule']:.4f} kcal/mol per molecule"


# ---------------------------------------------------------------- the readings (NumPy) ------------------------------
def rms(v):
    v = np.asarray(v, float)
    return float(np.sqrt(np.mean(v * v))) if v.size else float("nan")


def by_size(E_model, C):
    """{size: {"rmse", "mean", "rel"}}: the error per molecule (kcal/mol) rms and mean over the clusters of each size,
    and the interaction's relative rmse."""
    t = C["test"]
    err = (np.asarray(E_model) - t["energy"]) * KCAL / t["size"]
    out = {}
    for n in sorted(set(t["size"].tolist())):
        s = t["size"] == n
        out[int(n)] = {"rmse": rms(err[s]), "mean": float(np.mean(err[s])),
                       "rel": rms(np.asarray(E_model)[s] - t["energy"][s]) / rms(t["energy"][s])}
    return out


def load_runs(roles=ROLES, C=None):
    out = {}
    for role, (d, rung) in roles.items():
        for path in sorted(glob.glob(os.path.join(d, f"{rung}_seed*.json"))):
            with open(path) as fh:
                r = json.load(fh)
            if C is not None and r.get("clusters_hash") != C["hash"]:
                raise ValueError(f"{path} was trained on other clusters ({r.get('clusters_hash')} against {C['hash']})")
            out.setdefault(role, {})[int(r["seed"])] = r
    return out


def mace_group(path):
    """('maceles', 0) for results/e4_job18_mace/maceles_s0/maceles_job18_s0_stagetwo_job18.npz"""
    base = os.path.basename(os.path.dirname(path))
    g, s = base.rsplit("_s", 1)
    return g, int(s)


def load_mace(pattern=MACE_GLOB):
    out = {}
    for path in sorted(glob.glob(pattern)):
        g, s = mace_group(path)
        out.setdefault(g, {})[s] = path
    return out


def seed_mean(by, n, key="rmse"):
    v = [r[n][key] for r in by.values() if n in r]
    return float(np.mean(v)) if v else None


def evaluate(C, runs, mace, far_path=PROBES):
    """the readings: C the clusters (load_clusters), runs the harness's (load_runs), mace MACE's npz paths by group and
    seed (load_mace)."""
    import e4_far
    t = C["test"]
    sizes = sorted(set(t["size"].tolist()))
    ev = {"sizes": sizes, "truth": {}}
    for n in sizes:
        s = t["size"] == n
        ev["truth"][n] = {"per_molecule": float(np.mean(t["energy"][s] / n)) * KCAL,
                          "induced_share": float(np.mean(t["induced"][s] / t["energy"][s])),
                          "beyond_pairs": (float(np.mean((t["energy"][s] - t["pair_sum"][s]) / t["energy"][s]))
                                           if np.all(np.isfinite(t["pair_sum"][s])) else None)}
    models = {}
    for role, by in runs.items():
        models[role] = {s: {"sizes": by_size(r["readings"]["test_energy"], C), "probes": r["readings"].get("probes"),
                            "far_trimers": r["readings"].get("far_trimers"),
                            "calibration": r["readings"].get("calibration")} for s, r in by.items()}
    stem_t = os.path.splitext(os.path.basename(XYZ["test"]))[0]
    stem_f = os.path.splitext(os.path.basename(FAR_XYZ))[0]
    for g, by in mace.items():
        for s, path in by.items():
            z = np.load(path, allow_pickle=False)
            E = np.asarray(z[f"{stem_t}__energy"], float) / EV_PER_HARTREE
            r = {"sizes": by_size(E, C)}
            if f"{stem_f}__energy" in z.files and os.path.exists(far_path):
                rf = e4_far.readings(far_path, z[f"{stem_f}__energy"])
                r["probes"] = rf["probes"]
                r["far_trimers"] = rf.get("far_trimers")
            models.setdefault(g, {})[s] = r
    ev["models"] = models
    sm = {g: {n: seed_mean({s: r["sizes"] for s, r in by.items()}, n) for n in sizes} for g, by in models.items()}
    sm_mean = {g: {n: seed_mean({s: r["sizes"] for s, r in by.items()}, n, "mean") for n in sizes}
               for g, by in models.items()}
    ev["seed_means"] = sm
    ours, les = sm.get("model"), sm.get("maceles")
    small, far = min(sizes), [n for n in FAR_SIZES if n in sizes]
    two = lambda g: g in models and len(models[g]) >= 2
    ev["K18.1"] = {"ours": ours.get(small) if ours else None, "maceles": les.get(small) if les else None}
    ev["K18.1"]["holds"] = (bool(les[small] <= ours[small]) if two("model") and two("maceles") else None)
    ev["K18.2"] = {"ours": [ours[n] for n in far] if ours else None, "maceles": [les[n] for n in far] if les else None}
    ev["K18.2"]["holds"] = (bool(all(ours[n] < les[n] for n in far)) if two("model") and two("maceles") else None)
    ev["K18.3"] = {"band": K18_3, "ratio": les[max(sizes)] / les[small] if les else None}
    ev["K18.3"]["holds"] = bool(ev["K18.3"]["ratio"] >= K18_3) if two("maceles") else None
    ev["K18.4"] = {"band": K18_4, "worst": max(ours.values()) if ours else None,
                   "by_seed": {s: {n: r["sizes"][n]["rmse"] for n in sizes} for s, r in models.get("model", {}).items()}}
    ev["K18.4"]["holds"] = (bool(all(v <= K18_4 for by in ev["K18.4"]["by_seed"].values() for v in by.values()))
                            if two("model") else None)
    dg = sm.get("dagger")
    ev["K18.5"] = {"band": K18_5, "ratio": dg[max(sizes)] / ours[max(sizes)] if dg and ours else None}
    ev["K18.5"]["holds"] = bool(ev["K18.5"]["ratio"] >= K18_5) if two("dagger") and two("model") else None
    mp = sm_mean.get("model_probes")
    mo = sm_mean.get("model")
    ev["K18.6"] = {"probes_only": mp[max(sizes)] if mp else None, "calibrated": mo[max(sizes)] if mo else None}
    ev["K18.6"]["holds"] = (bool(abs(mp[max(sizes)]) > abs(mo[max(sizes)])) if two("model_probes") and two("model")
                            else None)
    lr = sm.get("learned")
    ev["K18.7"] = {"learned": [lr[n] for n in far] if lr else None, "maceles": [les[n] for n in far] if les else None}
    ev["K18.7"]["holds"] = (bool(all(lr[n] <= les[n] for n in far)) if two("learned") and two("maceles") else None)
    return ev


def _verdict(h):
    return "not read" if h is None else ("holds" if h else "FAILS")


def report(ev):
    f = lambda x, d=4: "-" if x is None else f"{x:.{d}f}"
    gl = lambda v: "-" if v is None else " / ".join(f(x) for x in v)
    sizes = ev["sizes"]
    L = ["E4, job 18 (notes/e4-job18-preregistration.md): from a few molecules to many, MB-pol standing in for CCSD(T)",
         "  the error per molecule, rms over 16 clusters of each size (kcal/mol), seed means"]
    k = ev["K18.1"]
    L.append(f"  K18.1: {_verdict(k['holds'])} (in distribution, {min(sizes)} molecules: MACE-LES at most our model's): "
             f"ours {f(k['ours'])}, MACE-LES {f(k['maceles'])}")
    k = ev["K18.2"]
    L.append(f"  K18.2: {_verdict(k['holds'])} (at {', '.join(str(n) for n in FAR_SIZES)} molecules: ours below MACE-LES): "
             f"ours {gl(k['ours'])}, MACE-LES {gl(k['maceles'])}")
    k = ev["K18.3"]
    L.append(f"  K18.3: {_verdict(k['holds'])} (MACE-LES's error at {max(sizes)} molecules at least {k['band']:g} times its "
             f"error at {min(sizes)}): {f(k['ratio'], 2)}")
    k = ev["K18.4"]
    L.append(f"  K18.4: {_verdict(k['holds'])} (ours within {k['band']:g} kcal/mol per molecule at every size, every seed):"
             f" worst seed mean {f(k['worst'])}")
    k = ev["K18.5"]
    L.append(f"  K18.5: {_verdict(k['holds'])} (the dagger's error at {max(sizes)} molecules at least {k['band']:g} times "
             f"ours): {f(k['ratio'], 2)}")
    k = ev["K18.6"]
    L.append(f"  K18.6: {_verdict(k['holds'])} (at {max(sizes)} molecules the read-out on the probes alone biased more "
             f"than calibrated on the clusters' response): mean error {f(k['probes_only'])} against {f(k['calibrated'])}")
    k = ev["K18.7"]
    L.append(f"  K18.7: {_verdict(k['holds'])} (at {', '.join(str(n) for n in FAR_SIZES)} molecules our learned charges at "
             f"most MACE-LES): learned {gl(k['learned'])}, MACE-LES {gl(k['maceles'])}")
    L.append("\n  Beside them, not registered")
    L.append("    MB-pol: interaction per molecule " + ", ".join(f"{n} {f(ev['truth'][n]['per_molecule'], 3)}" for n in sizes)
             + " kcal/mol; induced share " + ", ".join(f"{n} {100 * ev['truth'][n]['induced_share']:.0f} %" for n in sizes)
             + "; beyond pairs " + ", ".join(f"{n} {100 * v['beyond_pairs']:.1f} %" for n, v in ev["truth"].items()
                                             if v["beyond_pairs"] is not None))
    for g, by in ev["models"].items():
        for s, r in sorted(by.items()):
            line = (f"    {g} seed {s}: rms (mean) " + ", ".join(
                f"{n} {f(r['sizes'][n]['rmse'])} ({r['sizes'][n]['mean']:+.4f})" for n in sizes)
                + "; relative " + " / ".join(f(r["sizes"][n]["rel"], 3) for n in sizes))
            if r.get("probes"):
                p = r["probes"]["by_distance_angstrom"]
                line += "; probes non-additive left " + " / ".join(
                    f(p[d]["nonadditive_rmse"] / p[d]["nonadditive_rms"], 3) for d in ("10", "12", "15") if d in p)
            if r.get("far_trimers"):
                line += f"; T5 slope {f(r['far_trimers']['slope'], 4)}"
            if r.get("calibration"):
                line += f"; calibration response {f(r['calibration']['trace_ratio_mean'], 4)} of MB-pol's"
            L.append(line)
    return "\n".join(L)


def run_read(a):
    C = load_clusters(a.clusters)
    ev = evaluate(C, load_runs(C=C), load_mace(a.mace), far_path=a.probes)
    print(report(ev))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(ev, fh, indent=1, default=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--mbx", required=True)
    m.add_argument("--train_source", default=TRAIN_SOURCE)
    m.add_argument("--out", default=OUT)
    m.add_argument("--done", default=DONE)
    m.add_argument("--probes", default=PROBES)
    m.add_argument("--far_xyz", default=FAR_XYZ)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--rungs", default=None)
    t.add_argument("--seeds", default=None)
    t.add_argument("--out_dir", required=True)
    t.add_argument("--chunk_atoms", type=int, default=3072, help="atoms per batch of the test clusters' energies")
    r = sub.add_parser("read")
    r.add_argument("--clusters", default=OUT)
    r.add_argument("--mace", default=MACE_GLOB)
    r.add_argument("--probes", default=PROBES)
    r.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "make":
        a.xyz = dict(XYZ)
    if a.cmd in ("make", "train"):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    {"make": run_make, "train": run_train, "read": run_read}[a.cmd](a)


if __name__ == "__main__":
    main()
