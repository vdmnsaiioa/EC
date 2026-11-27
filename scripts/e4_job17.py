"""
e4_job17.py -- E4 job 17 (notes/e4-job17-preregistration.md; v0.0.42): what a field-blind model with free latent
charges cannot do, read on every model at once -- E4's model, the dagger, our learned charges, the cutoff model and
MACE-LES -- on structures whose truth MB-pol gives through MBX.  No training: every model is one already trained.

    python scripts/e4_job17.py make --mbx external/mbx/build/MBX           # CPU: the structures and MB-pol's labels
    python scripts/e4_job17.py harness --run results/e4_job12/M_16C2pm_us_d20_seed0      # GPU: one harness run
    python baselines/mace/energies.py <model> data/e4/job17_clusters.extxyz data/e4/job17_slabs.extxyz \
        data/e4/probes_mbpol_far.extxyz                                     # GPU, mace-les env: one MACE-LES model
    python scripts/e4_job17.py read                                        # NumPy only

Three rows (the pre-registration's section 1):

  a  the response to a uniform field, eps_inf: E4's model's from its own runs (job 12's jsons, the read-out held since
     job 5); every model without a field term -- the dagger, our learned charges, the cutoff model, MACE-LES as
     deployed -- has eps_inf = 1 by construction.  Read, not computed here.
  b  the far probes and the far trimers (T5) of scripts/e4_water.py probes: the harness's runs read them in training
     (their jsons); MACE-LES through scripts/e4_far.py.
  c  far fields where nothing screens them, new here:
     c1  two clusters cut from the liquid (8 and 16 molecules), the second rotated and placed along a random direction
         at a smallest atom-atom distance of 7-30 A: the interaction E(AB) - E(A) - E(B), open;
     c2  a 64-molecule frame opened into a slab, its cell lengthened along z by a vacuum gap of 9-24 A: the energy's
         change with the gap, E(g) - E(24 A), periodic.

`make` writes, from the test frames of 64 molecules (data/e4/test_n64.npz):
  data/e4/job17_mbpol.npz        the clusters (positions, MB-pol's energies and their components by term)
  data/e4/job17_slabs_n64.npz    the slabs in E4's data format (sseft.water.e4_structures reads it), with MB-pol's
                                 energies, forces and the cell's dipole (permanent and induced)
  data/e4/job17_clusters.extxyz  the clusters for any model: per pair A, B, then AB at every gap (eV, open)
  data/e4/job17_slabs.extxyz     the slabs (eV, periodic)
and checks that MB-pol's polynomials (2-, 3- and 4-body) give nothing between the clusters at any gap: the far field
there is electrostatics (permanent and induced) and dispersion alone.

`harness` evaluates one trained harness run (its json and .params.pkl; the rung rebuilt from the json's config as
scripts/e4_water.py train built it) on the clusters and the slabs, and copies the run's own infrared readings (the far
probes, T5, the liquid's eps_inf) into results/e4_job17/<run directory>__<rung>__seed<seed>.json.

`read` reads K17.1-K17.9:

    K17.1  MACE-LES beyond r_c is its latent charges' Coulomb interaction (Proposition 1 of les-vs-eft): for every
           MACE-LES model, the clusters' interaction from its energies equals k sum_{i in A, j in B} q_i q_j
           erf(r/sqrt2 sigma)/r from its latent charges within K17_1 kcal/mol, at every gap
    K17.2  T5: every MACE-LES model's three-body energy of the far trimers exactly zero (within K17_2 kcal/mol)
    K17.3  the far probes: the default MACE-LES (both seeds) leaves at least K17_3 of the non-additive energy at 10, 12
           and 15 A
    K17.4  the clusters: E4's model's interaction within K17_4 (relative rmse over the pairs, both sizes) at every gap
           from 9 A, every seed
    K17.5  the clusters: the default MACE-LES's relative rmse at least K17_5 at every gap from 9 A, both seeds
    K17.6  the clusters: the default MACE-LES's free net charges -- its error with its latent charges as they are over
           its error with each molecule's net latent charge removed, at 25 and 30 A -- at least K17_6, both seeds
    K17.7  the slabs: E4's model's E(g) - E(24 A) within K17_7 (relative rmse over the frames) at 9, 12 and 16 A
    K17.8  the slabs: the default MACE-LES's relative rmse at least K17_8 at 9, 12 and 16 A, both seeds
    K17.9  the clusters: our learned charges (job 10's M_1w100, per-molecule neutral, no pins) at most the default
           MACE-LES's relative rmse at every gap from 9 A (seed means): the same data, no pins
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KCAL = 627.5094740631                                   # kcal/mol per hartree
EV_PER_HARTREE = 27.211386245988
KCAL_PER_EV = KCAL / EV_PER_HARTREE
LES_K = 90.4756 / (2.0 * math.pi)                       # LES's 1/(4 pi eps_0), eV A (its norm_factor / 2 pi)
LES_SIGMA = 1.0                                         # A, baselines/mace/les.yaml
MASS = np.array([15.999, 1.008, 1.008])

SOURCE = "data/e4/test_n64.npz"
TRUTH = "data/e4/job17_mbpol.npz"
SLABS = "data/e4/job17_slabs_n64.npz"
CLUSTERS_XYZ = "data/e4/job17_clusters.extxyz"
SLABS_XYZ = "data/e4/job17_slabs.extxyz"
PROBES = "data/e4/probes_mbpol.npz"
FAR_XYZ = "data/e4/probes_mbpol_far.extxyz"
OUT_DIR = "results/e4_job17"

SIZES = (8, 16)                                          # molecules per cluster
PAIRS = 12                                               # cluster pairs per size
GAPS = (7.0, 9.0, 12.0, 16.0, 20.0, 25.0, 30.0)          # A, the smallest atom-atom distance between the clusters
SLAB_FRAMES = 8
SLAB_GAPS = (9.0, 12.0, 16.0, 24.0)                      # A of vacuum added to the cell's z
SLAB_MIN_CLEAR = 6.5                                     # A: the slab's clearance from its image across the gap
SEED = 1700
FAR_GAP = 9.0                                            # the registered gaps: from 9 A (7 A beside)
TAIL_GAPS = (25.0, 30.0)                                 # K17.6's gaps
SLAB_READ = (9.0, 12.0, 16.0)                            # K17.7-K17.8's gaps, against the widest
PROBE_DISTANCES = ("10", "12", "15")                     # K17.3's (at 8 A some probes reach inside r_c: job 1)
K17_1, K17_2, K17_3, K17_4, K17_5, K17_6, K17_7, K17_8 = 1e-6, 1e-6, 0.5, 0.05, 0.30, 1.5, 0.10, 0.30
POLY = ("one_body", "two_body", "three_body", "four_body", "buckingham")
TERMS = POLY + ("dispersion", "electrostatics", "permanent", "induced")
# the harness's runs read: (directory, rung, seeds), by role
HARNESS = {"model": ("results/e4_job12", "M_16C2pm_us_d20", (0, 1, 2, 3)),     # E4's model, polished (job 12)
           "dagger": ("results/e4_job11", "M_16pm_us_d20", (0, 1)),             # pinned sources, no response (job 11)
           "learned": ("results/e4_job10", "M_1w100", (0, 1)),                  # learned charges, neutral (job 10)
           "cutoff": ("results/e4_job7", "M_inf", (0, 1))}                      # E_0 alone (job 7)
# MACE-LES: the default (the baseline, registered) and job 15's other units and releases (beside)
LES_GLOBS = {"maceles": "results/e4_baselines/maceles_n64_s*/maceles_n64_s*_stagetwo_job17.npz",
             "others": "results/e4_job15_les/*_n64_s*/*_stagetwo_job17.npz"}


# ---------------------------------------------------------------- geometry (NumPy) ----------------------------------
def cubic_length(cell):
    cell = np.asarray(cell, float).reshape(3, 3)
    L = float(cell[0, 0])
    if not np.allclose(cell, np.eye(3) * L, atol=1e-8):
        raise ValueError("job 17 cuts its structures from cubic cells")
    return L


def whole_molecules(pos, L):
    """(n, 3, 3): each molecule's H atoms taken to its O by the minimum image of a cubic cell of side L."""
    m = np.asarray(pos, float).reshape(-1, 3, 3).copy()
    d = m - m[:, :1]
    m = m[:, :1] + d - np.round(d / L) * L
    return m


def com(pos):
    m = np.asarray(pos, float).reshape(-1, 3, 3)
    w = np.broadcast_to(MASS, m.shape[:2])
    return (m * w[..., None]).sum(axis=(0, 1)) / w.sum()


def cut_cluster(pos, L, center, n):
    """n whole molecules of a liquid frame: the molecule `center` and its n - 1 nearest by O-O minimum-image distance,
    each placed at its minimum image about the centre; (3n, 3) A, the centre's molecule first, then by distance."""
    m = whole_molecules(pos, L)
    dO = m[:, 0] - m[center, 0]
    dO -= np.round(dO / L) * L
    order = np.argsort(np.linalg.norm(dO, axis=1), kind="stable")[:n]
    if order[0] != center:
        order = np.concatenate([[center], order[order != center]])[:n]
    out = m[order] + (m[center, 0] + dO[order] - m[order, 0])[:, None, :]
    return out.reshape(-1, 3)


def rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])


def unit_vector(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def placement(A, B0, u, d):
    """the smallest t >= 0 with every atom of B0 + t u at least d from every atom of A: for each pair w = b - a,
    |w + t u| >= d for every t beyond the larger root of t^2 + 2 t w.u + |w|^2 - d^2; at that t the smallest distance is
    d exactly."""
    w = (np.asarray(B0)[None, :, :] - np.asarray(A)[:, None, :]).reshape(-1, 3)
    wu = w @ u
    disc = wu * wu - np.einsum("ij,ij->i", w, w) + d * d
    roots = np.where(disc >= 0, -wu + np.sqrt(np.clip(disc, 0.0, None)), -np.inf)
    return float(max(0.0, roots.max()))


def min_distance(A, B):
    return float(np.sqrt(((np.asarray(A)[:, None, :] - np.asarray(B)[None, :, :]) ** 2).sum(-1)).min())


def cluster_pairs(positions, cell, sizes=SIZES, pairs=PAIRS, gaps=GAPS, seed=SEED):
    """{size: {"A" (P, 3n, 3), "B" (P, G, 3n, 3), "R" (P, G) the centres of mass' distance, "u" (P, 3), "src" (P, 4):
    frame and centre of A, of B}}: A and B cut from two different frames, B rotated about its centre of mass, its centre
    put on A's and moved along u until the smallest atom-atom distance is each gap."""
    rng = np.random.default_rng(seed)
    F = len(positions)
    L = cubic_length(cell if np.ndim(cell) == 2 else cell[0])
    n_mol = positions.shape[1] // 3
    out = {}
    for n in sizes:
        A_, B_, R_, U_, S_ = [], [], [], [], []
        for _ in range(pairs):
            fa, fb = rng.choice(F, size=2, replace=False)
            ca, cb = rng.integers(n_mol), rng.integers(n_mol)
            A = cut_cluster(positions[fa], L, ca, n)
            B = cut_cluster(positions[fb], L, cb, n)
            A = A - com(A)
            B = (B - com(B)) @ rotation(rng).T
            u = unit_vector(rng)
            Bs, Rs = [], []
            for d in gaps:
                t = placement(A, B, u, d)
                Bs.append(B + t * u)
                Rs.append(t)
            A_.append(A); B_.append(Bs); R_.append(Rs); U_.append(u); S_.append([fa, ca, fb, cb])
        out[n] = {"A": np.array(A_), "B": np.array(B_), "R": np.array(R_), "u": np.array(U_),
                  "src": np.array(S_, int)}
    return out


def slab(pos, L, gap):
    """a cubic frame opened into a slab: whole molecules, each O wrapped into [0, L)^3; the cell (L, L, L + gap)."""
    m = whole_molecules(pos, L)
    m -= (np.floor(m[:, :1] / L) * L)
    cell = np.diag([L, L, L + gap])
    return m.reshape(-1, 3), cell


def slab_clearance(pos, cell):
    """the smallest z distance between the slab and its image across the gap (the cell's z length less the atoms'
    z extent)."""
    z = np.asarray(pos)[:, 2]
    return float(cell[2, 2] - (z.max() - z.min()))


# ---------------------------------------------------------------- the frames' order (shared) ----------------------
def cluster_frames(T):
    """[(size, pair, kind, gap index, positions)] in the extxyz's order: per size and pair, A, B (at the first gap),
    then AB at every gap."""
    out = []
    for n in T["sizes"]:
        c = T["clusters"][n]
        for p in range(len(c["A"])):
            out.append((n, p, "A", -1, c["A"][p]))
            out.append((n, p, "B", -1, c["B"][p, 0]))
            for g in range(c["B"].shape[1]):
                out.append((n, p, "AB", g, np.concatenate([c["A"][p], c["B"][p, g]])))
    return out


def load_truth(path=TRUTH):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    sizes = [int(n) for n in meta["sizes"]]
    T = {"meta": meta, "hash": meta["hash"], "sizes": sizes, "gaps": np.asarray(d["gaps"], float), "clusters": {}}
    for n in sizes:
        c = {k: np.asarray(d[f"c{n}_{k}"]) for k in ("A", "B", "R", "u", "src", "E_A", "E_B", "E_AB")}
        c["terms"] = {t: np.asarray(d[f"c{n}_int_{t}"]) for t in TERMS}
        T["clusters"][n] = c
    return T


def arrays_hash(arrays):
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()[:16]


def write_extxyz(path, frames, cells=None, energies_ev=None, tags=None):
    """frames: [(3n, 3) A]; cells: [(3, 3)] or None (open); energies in eV or None."""
    with open(path, "w") as fh:
        for k, pos in enumerate(frames):
            sym = np.tile(["O", "H", "H"], len(pos) // 3)
            head = "Properties=species:S:1:pos:R:3"
            if cells is not None:
                head = f'Lattice="{" ".join(f"{x:.10f}" for x in np.asarray(cells[k]).reshape(-1))}" ' + head + ' pbc="T T T"'
            else:
                head += ' pbc="F F F"'
            if energies_ev is not None:
                head += f" REF_energy={energies_ev[k]:.12f}"
            if tags is not None:
                head += "".join(f" {a}={b}" for a, b in tags[k].items())
            fh.write(f"{len(pos)}\n{head}\n")
            for s, (x, y, z) in zip(sym, pos):
                fh.write(f"{s} {x:.10f} {y:.10f} {z:.10f}\n")
    return path


def read_extxyz(path):
    """[(info {key: str}, positions (n, 3) A)] of an extxyz written by write_extxyz (NumPy only, without ASE)."""
    import shlex
    out = []
    with open(path) as fh:
        lines = fh.read().splitlines()
    i = 0
    while i < len(lines):
        n = int(lines[i])
        info = dict(kv.split("=", 1) for kv in shlex.split(lines[i + 1]))
        pos = np.array([[float(x) for x in ln.split()[1:4]] for ln in lines[i + 2:i + 2 + n]])
        out.append((info, pos))
        i += 2 + n
    return out


# ---------------------------------------------------------------- make (MBX, CPU) -----------------------------------
def label_clusters(truth, C):
    """MB-pol's energies (hartree) of every A, B and AB, and the interaction's components by term (TERMS); raises if
    a polynomial (2-, 3-, 4-body) reaches between the clusters at any gap."""
    out = {}
    for n, c in C.items():
        P, G = c["B"].shape[:2]
        compA = [truth.components(c["A"][p]) for p in range(P)]
        compB = [truth.components(c["B"][p, 0]) for p in range(P)]
        E_A = np.array([truth.energy(c["A"][p]) for p in range(P)])
        E_B = np.array([truth.energy(c["B"][p, 0]) for p in range(P)])
        E_AB = np.zeros((P, G))
        terms = {t: np.zeros((P, G)) for t in TERMS}
        for p in range(P):
            for g in range(G):
                AB = np.concatenate([c["A"][p], c["B"][p, g]])
                cAB = truth.components(AB)
                E_AB[p, g] = truth.energy(AB)
                for t in TERMS:
                    terms[t][p, g] = cAB[t] - compA[p][t] - compB[p][t]
        for t in POLY:
            worst = float(np.abs(terms[t]).max()) * KCAL
            if worst > 1e-8:
                raise RuntimeError(f"MB-pol's {t} reaches between the {n}-molecule clusters ({worst:.2e} kcal/mol): "
                                   "the far field is not electrostatics and dispersion alone")
        out[n] = {"E_A": E_A, "E_B": E_B, "E_AB": E_AB, "terms": terms}
    return out


def run_make(a):
    from sseft.mbpol import MBPol
    make(MBPol(a.mbx), a)


def make(truth, a):
    """the structures and the truth's labels (a: run_make's arguments; truth: energy, energy_forces, components,
    sites as sseft.mbpol.MBPol's)."""
    t0 = time.time()
    d = np.load(a.source, allow_pickle=False)
    pos, cell = np.asarray(d["positions"], float), np.asarray(d["cell"], float)
    cells = cell if cell.ndim == 3 else np.broadcast_to(cell, (len(pos), 3, 3))
    print(f"truth {truth.describe() if hasattr(truth, 'describe') else type(truth).__name__}; {len(pos)} frames of "
          f"{pos.shape[1] // 3} molecules from {a.source}", flush=True)
    C = cluster_pairs(pos, cells[0], SIZES, PAIRS, GAPS, SEED)
    lab = label_clusters(truth, C)
    arrays = {"gaps": np.array(GAPS, float)}
    for n, c in C.items():
        for k in ("A", "B", "R", "u", "src"):
            arrays[f"c{n}_{k}"] = c[k]
        for k in ("E_A", "E_B", "E_AB"):
            arrays[f"c{n}_{k}"] = lab[n][k]
        for t in TERMS:
            arrays[f"c{n}_int_{t}"] = lab[n]["terms"][t]
        inter = (lab[n]["E_AB"] - lab[n]["E_A"][:, None] - lab[n]["E_B"][:, None]) * KCAL
        print(f"  {n}-molecule clusters, {len(c['A'])} pairs: the interaction's rms by gap "
              + ", ".join(f"{g:g} A {np.sqrt(np.mean(inter[:, k] ** 2)):.4f}" for k, g in enumerate(GAPS))
              + " kcal/mol; induced share "
              + ", ".join(f"{100 * np.sqrt(np.mean(lab[n]['terms']['induced'][:, k] ** 2)) * KCAL / max(np.sqrt(np.mean(inter[:, k] ** 2)), 1e-30):.0f} %"
                          for k in range(len(GAPS)))
              + f"  [{time.time() - t0:.0f} s]", flush=True)
    digest = arrays_hash(arrays)
    meta = {"sizes": list(SIZES), "pairs": PAIRS, "gaps": list(GAPS), "seed": SEED, "source": a.source,
            "hash": digest, "slab_frames": SLAB_FRAMES, "slab_gaps": list(SLAB_GAPS)}
    if hasattr(truth, "describe"):
        meta["truth"] = truth.describe()
    np.savez(a.out, meta=np.array(json.dumps(meta)), **arrays)
    T = load_truth(a.out)
    fr = cluster_frames(T)
    E_ev = []
    for n, p, kind, g, _ in fr:
        c = T["clusters"][n]
        E = c["E_A"][p] if kind == "A" else c["E_B"][p] if kind == "B" else c["E_AB"][p, g]
        E_ev.append(E * EV_PER_HARTREE)
    write_extxyz(a.clusters_xyz, [f[4] for f in fr], energies_ev=E_ev,
                 tags=[{"size": n, "pair": p, "kind": k, "gap": g} for n, p, k, g, _ in fr])
    print(f"written {a.out} (hash {digest}) and {a.clusters_xyz} ({len(fr)} frames)  [{time.time() - t0:.0f} s]",
          flush=True)
    # the slabs
    L = cubic_length(cells[0])
    S = {k: [] for k in ("positions", "cell", "energy", "forces", "slab_frame", "slab_gap", "dipole_permanent",
                         "dipole_induced")}
    for f in range(min(SLAB_FRAMES, len(pos))):
        for g in SLAB_GAPS:
            P_, C_ = slab(pos[f], L, g)
            if slab_clearance(P_, C_) < SLAB_MIN_CLEAR:
                raise RuntimeError(f"slab of frame {f} at {g} A: clearance {slab_clearance(P_, C_):.2f} A")
            E, Fo = truth.energy_forces(P_, C_)
            xyz, q, _, mu = truth.sites(P_, C_)
            S["positions"].append(P_); S["cell"].append(C_); S["energy"].append(E); S["forces"].append(Fo)
            S["slab_frame"].append(f); S["slab_gap"].append(g)
            S["dipole_permanent"].append((q[:, None] * xyz).sum(0)); S["dipole_induced"].append(mu.sum(0))
        print(f"  slab of frame {f}: E(g) - E({SLAB_GAPS[-1]:g} A) "
              + ", ".join(f"{g:g} A {(S['energy'][-len(SLAB_GAPS) + k] - S['energy'][-1]) * KCAL:+.4f}"
                          for k, g in enumerate(SLAB_GAPS[:-1]))
              + f" kcal/mol per cell  [{time.time() - t0:.0f} s]", flush=True)
    slabs = {k: np.array(v) for k, v in S.items()}
    slabs.update(site_q=np.asarray(d["site_q"]), site_gamma=np.asarray(d["site_gamma"]),
                 source=np.array(S["slab_frame"]), time_ps=np.zeros(len(S["energy"])))
    slabs["meta"] = np.array(json.dumps({"hash": arrays_hash({k: slabs[k] for k in ("positions", "cell", "energy")}),
                                         "source": a.source, "gaps": list(SLAB_GAPS), "frames": SLAB_FRAMES}))
    np.savez(a.slabs, **slabs)
    write_extxyz(a.slabs_xyz, list(slabs["positions"]), cells=list(slabs["cell"]),
                 energies_ev=list(slabs["energy"] * EV_PER_HARTREE),
                 tags=[{"frame": int(f), "gap": float(g)} for f, g in zip(slabs["slab_frame"], slabs["slab_gap"])])
    print(f"written {a.slabs} and {a.slabs_xyz} ({len(slabs['energy'])} slabs)", flush=True)
    if not os.path.exists(a.far_xyz) and os.path.exists(a.probes):
        import e4_far
        print(f"written {e4_far.write_extxyz(a.probes, a.far_xyz)} (the far probes and trimers)", flush=True)
    print(f"done  [{time.time() - t0:.0f} s]", flush=True)


# ---------------------------------------------------------------- harness (JAX, GPU) --------------------------------
def harness_tag(run):
    return f"{os.path.basename(os.path.dirname(os.path.abspath(run)))}__{os.path.basename(run).replace('_seed', '__seed')}"


def run_harness(a):
    import pickle
    import jax
    import e4_water as E4
    import sseft
    from sseft import water as W, train as T_
    t0 = time.time()
    with open(a.run + ".json") as fh:
        js = json.load(fh)
    with open(a.run + ".params.pkl", "rb") as fh:
        params = pickle.load(fh)
    cfg = E4.merge(E4.load_config(None), js["config"])          # the run's own settings; a key added since, its default
    name, seed = js["rung"], int(js["seed"])
    K, om = int(cfg["model"]["K"]), float(cfg["model"]["omega0"])
    src = W.e4_structures(a.source, pins=True, K=K, omega0=om)
    rung, rung_open = E4.rungs_for(name, cfg, src[0].cell)
    rung, rung_open = E4.warm_started(name, rung, src), E4.warm_started(name, rung_open, src)
    d = np.load(a.source, allow_pickle=False)
    site_q, gamma = np.asarray(d["site_q"]), float(d["site_gamma"])
    T = load_truth(a.truth)
    print(f"{a.run}: {name} seed {seed} (sseft {sseft.__version__}, JAX {jax.__version__}, {jax.devices()}); truth "
          f"{a.truth} (hash {T['hash']})", flush=True)
    pin = lambda P: W.pin_rigid_site_model([W.water_structure(p, pinned_q=False) for p in P], site_q, gamma, K, om)
    out = {"run": a.run, "rung": name, "seed": seed, "sseft": sseft.__version__, "truth_hash": T["hash"],
           "clusters": {}, "source_readings": {k: js.get("readings", {}).get(k)
                                               for k in ("probes", "far_trimers", "liquid")}}
    for n in T["sizes"]:
        c = T["clusters"][n]
        P, G = c["B"].shape[:2]
        EA = T_.predict_energies(params, rung_open, pin(c["A"]), chunk=a.chunk)
        EB = T_.predict_energies(params, rung_open, pin(c["B"][:, 0]), chunk=a.chunk)
        EAB = T_.predict_energies(params, rung_open, pin([np.concatenate([c["A"][p], c["B"][p, g]])
                                                          for p in range(P) for g in range(G)]), chunk=a.chunk)
        out["clusters"][str(n)] = {"E_A": EA.tolist(), "E_B": EB.tolist(), "E_AB": EAB.reshape(P, G).tolist()}
        inter = (EAB.reshape(P, G) - EA[:, None] - EB[:, None]) * KCAL
        ref = (c["E_AB"] - c["E_A"][:, None] - c["E_B"][:, None]) * KCAL
        print(f"  {n}-molecule clusters: relative rmse by gap "
              + ", ".join(f"{g:g} A {rel_rmse(inter[:, k], ref[:, k]):.3f}" for k, g in enumerate(T["gaps"]))
              + f"  [{time.time() - t0:.0f} s]", flush=True)
    slabs = W.e4_structures(a.slabs, pins=True, K=K, omega0=om)
    E = np.zeros(len(slabs))
    groups = {}
    for i, st in enumerate(slabs):
        groups.setdefault(tuple(np.round(np.asarray(st.cell).reshape(-1), 8)), []).append(i)
    for idx in groups.values():
        r = E4.in_cell(rung, slabs[idx[0]].cell)
        E[idx] = T_.predict_energies(params, r, [slabs[i] for i in idx], chunk=1)
    out["slabs"] = E.tolist()
    S = np.load(a.slabs, allow_pickle=False)
    dm, dt = slab_changes(E, S["slab_frame"], S["slab_gap"]), slab_changes(S["energy"], S["slab_frame"], S["slab_gap"])
    print("  slabs: relative rmse of E(g) - E(widest) by gap "
          + ", ".join(f"{g:g} A {rel_rmse(dm[g], dt[g]):.3f}" for g in sorted(dm)) + f"  [{time.time() - t0:.0f} s]",
          flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, harness_tag(a.run) + ".json")
    with open(path, "w") as fh:
        json.dump(out, fh)
    print(f"written {path}  [{time.time() - t0:.0f} s]", flush=True)


# ---------------------------------------------------------------- the readings (NumPy) ------------------------------
def rms(v):
    v = np.asarray(v, float)
    return float(np.sqrt(np.mean(v * v))) if v.size else float("nan")


def rel_rmse(model, ref):
    r = rms(ref)
    return rms(np.asarray(model) - np.asarray(ref)) / r if r > 0 else float("nan")


def slab_changes(E, frames, gaps):
    """{gap: E(frame, gap) - E(frame, widest gap) over the frames (kcal/mol per cell)} for every gap but the widest."""
    E, frames, gaps = np.asarray(E, float), np.asarray(frames), np.asarray(gaps, float)
    wide = gaps.max()
    out = {}
    for g in sorted(set(gaps.tolist())):
        if g == wide:
            continue
        rows = []
        for f in sorted(set(frames.tolist())):
            i = np.where((frames == f) & (gaps == g))[0]
            j = np.where((frames == f) & (gaps == wide))[0]
            if len(i) and len(j):
                rows.append((E[i[0]] - E[j[0]]) * KCAL)
        out[float(g)] = np.array(rows)
    return out


def truth_interactions(T):
    """{size: (P, G) kcal/mol} MB-pol's cluster interaction; and its parts by term."""
    out = {}
    for n in T["sizes"]:
        c = T["clusters"][n]
        out[n] = {"total": (c["E_AB"] - c["E_A"][:, None] - c["E_B"][:, None]) * KCAL,
                  **{t: c["terms"][t] * KCAL for t in ("permanent", "induced", "dispersion")}}
    return out


def harness_interactions(T, run):
    out = {}
    for n in T["sizes"]:
        c = run["clusters"][str(n)]
        out[n] = (np.asarray(c["E_AB"]) - np.asarray(c["E_A"])[:, None] - np.asarray(c["E_B"])[:, None]) * KCAL
    return out


def les_kernel(r):
    erf = np.vectorize(math.erf)
    return erf(r / (math.sqrt(2.0) * LES_SIGMA)) / r


def coulomb_cross(qa, pa, qb, pb):
    """LES's real-space interaction of two charge sets (eV): LES_K sum_ij qa_i qb_j erf(r/sqrt2 sigma)/r."""
    r = np.sqrt(((np.asarray(pa)[:, None, :] - np.asarray(pb)[None, :, :]) ** 2).sum(-1))
    return float(LES_K * np.einsum("i,j,ij->", qa, qb, les_kernel(r)))


def les_set(z, stem):
    """(energies (F,) eV, [latent charges per frame]) of one file in a baselines/mace/energies.py npz."""
    E = np.asarray(z[f"{stem}__energy"], float)
    q, off = np.asarray(z[f"{stem}__q"], float), np.asarray(z[f"{stem}__offsets"], int)
    return E, [q[off[k]:off[k + 1]] for k in range(len(E))]


def les_clusters(T, z):
    """MACE-LES on the clusters: its interaction from its energies (kcal/mol, (P, G) per size), the same from its
    latent charges (Proposition 1), with each molecule's net latent charge removed and with each cluster's removed;
    the largest change of A's and B's charges in AB (Proposition 1's other half); each cluster's net latent charge."""
    fr = cluster_frames(T)
    stem = os.path.splitext(os.path.basename(CLUSTERS_XYZ))[0]
    E, Q = les_set(z, stem)
    if len(E) != len(fr):
        raise ValueError(f"{len(E)} MACE-LES energies for {len(fr)} cluster frames")
    out, k = {}, 0
    for n in T["sizes"]:
        c = T["clusters"][n]
        P, G = c["B"].shape[:2]
        o = {key: np.zeros((P, G)) for key in ("energies", "charges", "molecule_neutral", "cluster_neutral")}
        o["net_A"], o["net_B"], o["dq"] = np.zeros(P), np.zeros(P), 0.0
        for p in range(P):
            eA, qA = E[k], Q[k]; eB, qB = E[k + 1], Q[k + 1]
            k += 2
            mol = lambda q: (q.reshape(-1, 3) - q.reshape(-1, 3).mean(axis=1, keepdims=True)).reshape(-1)
            o["net_A"][p], o["net_B"][p] = qA.sum(), qB.sum()
            for g in range(G):
                eAB, qAB = E[k], Q[k]
                k += 1
                o["dq"] = max(o["dq"], float(np.abs(qAB - np.concatenate([qA, qB])).max()))
                pa, pb = c["A"][p], c["B"][p, g]
                o["energies"][p, g] = (eAB - eA - eB) * KCAL_PER_EV
                o["charges"][p, g] = coulomb_cross(qA, pa, qB, pb) * KCAL_PER_EV
                o["molecule_neutral"][p, g] = coulomb_cross(mol(qA), pa, mol(qB), pb) * KCAL_PER_EV
                o["cluster_neutral"][p, g] = coulomb_cross(qA - qA.mean(), pa, qB - qB.mean(), pb) * KCAL_PER_EV
        out[n] = o
    return out


def les_slabs(z, S):
    stem = os.path.splitext(os.path.basename(SLABS_XYZ))[0]
    E, Q = les_set(z, stem)
    if len(E) != len(S["energy"]):
        raise ValueError(f"{len(E)} MACE-LES energies for {len(S['energy'])} slabs")
    pos = np.asarray(S["positions"])
    net = np.array([q.sum() for q in Q])
    dip = np.array([(q - q.mean()) @ p[:, 2] for q, p in zip(Q, pos)])
    return E / EV_PER_HARTREE, net, dip


def pooled(by_size, sizes, k):
    return np.concatenate([np.asarray(by_size[n])[:, k] for n in sizes])


def by_gap(model, ref, T):
    """[relative rmse at each gap, the sizes pooled]"""
    return [rel_rmse(pooled(model, T["sizes"], k), pooled(ref, T["sizes"], k)) for k in range(len(T["gaps"]))]


def load_harness(out_dir, T, roles=HARNESS):
    """{role: {seed: run json}} for the runs of `roles` found in out_dir with this truth."""
    out = {}
    for role, (d, rung, seeds) in roles.items():
        for s in seeds:
            path = os.path.join(out_dir, f"{os.path.basename(d)}__{rung}__seed{s}.json")
            if not os.path.exists(path):
                continue
            with open(path) as fh:
                run = json.load(fh)
            if run["truth_hash"] != T["hash"]:
                raise ValueError(f"{path} was read on another truth ({run['truth_hash']} against {T['hash']})")
            out.setdefault(role, {})[s] = run
    return out


def les_group(path):
    """('maceles', 0) for results/e4_baselines/maceles_n64_s0/..., ('maceles_o050', 1), ('release_o150', 0), ..."""
    base = os.path.basename(os.path.dirname(path))
    stem, seed = base.rsplit("_s", 1)
    return stem.replace("_n64", ""), int(seed)


def load_les(globs=LES_GLOBS):
    out = {}
    for pattern in globs.values():
        for path in sorted(glob.glob(pattern)):
            g, s = les_group(path)
            out.setdefault(g, {})[s] = path
    return out


def evaluate(T, S, H, L, far_path=PROBES):
    """the readings: T the truth (load_truth), S the slabs' npz, H the harness's runs (load_harness), L MACE-LES's
    npz paths by group and seed (load_les)."""
    gaps = [float(g) for g in T["gaps"]]
    far = [k for k, g in enumerate(gaps) if g >= FAR_GAP - 1e-9]
    tail = [gaps.index(g) for g in TAIL_GAPS if g in gaps]
    ref = truth_interactions(T)
    ref_total = {n: ref[n]["total"] for n in T["sizes"]}
    dt = slab_changes(S["energy"], S["slab_frame"], S["slab_gap"])
    ev = {"gaps": gaps, "truth": {
        "rms": [rms(pooled(ref_total, T["sizes"], k)) for k in range(len(gaps))],
        "shares": {t: [rms(pooled({n: ref[n][t] for n in T["sizes"]}, T["sizes"], k))
                       / max(rms(pooled(ref_total, T["sizes"], k)), 1e-300) for k in range(len(gaps))]
                   for t in ("permanent", "induced", "dispersion")},
        "slabs_rms": {g: rms(v) for g, v in dt.items()},
        "slab_dipole": {"permanent_rms": rms(np.asarray(S["dipole_permanent"])[:, 2]),
                        "induced_rms": rms(np.asarray(S["dipole_induced"])[:, 2])}}}
    # the harness's runs
    runs = {}
    for role, by in H.items():
        for s, run in by.items():
            inter = harness_interactions(T, run)
            dm = slab_changes(run["slabs"], S["slab_frame"], S["slab_gap"])
            sr = run.get("source_readings") or {}
            runs.setdefault(role, {})[s] = {
                "clusters": by_gap(inter, ref_total, T),
                "slabs": {g: rel_rmse(dm[g], dt[g]) for g in dt},
                "probes": sr.get("probes"), "far_trimers": sr.get("far_trimers"),
                "eps_inf": ({"model": sr["liquid"].get("eps_inf_model"), "truth": sr["liquid"].get("eps_inf_truth")}
                            if sr.get("liquid") else None)}
    ev["harness"] = runs
    # MACE-LES
    les = {}
    import e4_far
    for g, by in L.items():
        for s, path in by.items():
            z = np.load(path, allow_pickle=False)
            lc = les_clusters(T, z)
            E, net, dip = les_slabs(z, S)
            dm = slab_changes(E, S["slab_frame"], S["slab_gap"])
            r = {"clusters": by_gap({n: lc[n]["energies"] for n in T["sizes"]}, ref_total, T),
                 "molecule_neutral": by_gap({n: lc[n]["molecule_neutral"] for n in T["sizes"]}, ref_total, T),
                 "cluster_neutral": by_gap({n: lc[n]["cluster_neutral"] for n in T["sizes"]}, ref_total, T),
                 "prop1": max(float(np.abs(lc[n]["energies"] - lc[n]["charges"]).max()) for n in T["sizes"]),
                 "prop1_charges": max(lc[n]["dq"] for n in T["sizes"]),
                 "net_charge_rms": {str(n): rms(np.concatenate([lc[n]["net_A"], lc[n]["net_B"]])) for n in T["sizes"]},
                 "slabs": {gg: rel_rmse(dm[gg], dt[gg]) for gg in dt},
                 "slab_net_charge_rms": rms(net), "slab_dipole_rms": rms(dip),
                 "tail_ratio": {}}
            for n in T["sizes"]:                      # the interaction along each ray, 30 A against 16 A (the median)
                if 16.0 in gaps and 30.0 in gaps:
                    i, j = gaps.index(16.0), gaps.index(30.0)
                    r["tail_ratio"][str(n)] = float(np.median(lc[n]["energies"][:, j] / lc[n]["energies"][:, i]))
            stem = os.path.splitext(os.path.basename(FAR_XYZ))[0]
            if f"{stem}__energy" in z.files and os.path.exists(far_path):
                rf = e4_far.readings(far_path, z[f"{stem}__energy"])
                r["probes"] = rf["probes"]
                if "far_trimers" in rf:
                    r["far_trimers"] = rf["far_trimers"]
                    r["t5_max"] = float(np.abs(rf["far_trimers"]["model"]).max()) * KCAL
            les[g] = {**les.get(g, {}), s: r}
    ev["les"] = les
    if 16.0 in gaps and 30.0 in gaps:                 # the truth along each ray, and what a R^-3 or R^-1 tail gives
        i, j = gaps.index(16.0), gaps.index(30.0)
        ev["truth"]["tail_ratio"] = {str(n): float(np.median(ref_total[n][:, j] / ref_total[n][:, i])) for n in T["sizes"]}
        ev["truth"]["tail_reference"] = {
            str(n): {"dipolar": float(np.median((T["clusters"][n]["R"][:, i] / T["clusters"][n]["R"][:, j]) ** 3)),
                     "monopole": float(np.median(T["clusters"][n]["R"][:, i] / T["clusters"][n]["R"][:, j]))}
            for n in T["sizes"]}
    else:
        ev["truth"]["tail_ratio"], ev["truth"]["tail_reference"] = {}, {}
    # the registered readings
    dflt = les.get("maceles", {})
    two = len(dflt) >= 2
    ev["K17.1"] = {"band": K17_1, "worst": max((r["prop1"] for by in les.values() for r in by.values()), default=None),
                   "charges_worst": max((r["prop1_charges"] for by in les.values() for r in by.values()), default=None)}
    ev["K17.1"]["holds"] = None if not les else bool(ev["K17.1"]["worst"] <= K17_1)
    t5 = [r["t5_max"] for by in les.values() for r in by.values() if "t5_max" in r]
    ev["K17.2"] = {"band": K17_2, "worst": max(t5) if t5 else None, "holds": bool(max(t5) <= K17_2) if t5 else None}
    left = {s: {dd: r["probes"]["by_distance_angstrom"][dd]["nonadditive_rmse"]
                / r["probes"]["by_distance_angstrom"][dd]["nonadditive_rms"]
                for dd in PROBE_DISTANCES if dd in r["probes"]["by_distance_angstrom"]}
            for s, r in dflt.items() if r.get("probes")}
    ev["K17.3"] = {"band": K17_3, "left": left,
                   "holds": bool(all(v >= K17_3 for x in left.values() for v in x.values())) if two and len(left) >= 2 else None}
    mod = runs.get("model", {})
    ev["K17.4"] = {"band": K17_4, "by_seed": {s: [r["clusters"][k] for k in far] for s, r in mod.items()},
                   "holds": bool(all(r["clusters"][k] <= K17_4 for r in mod.values() for k in far)) if mod else None}
    ev["K17.5"] = {"band": K17_5, "by_seed": {s: [r["clusters"][k] for k in far] for s, r in dflt.items()},
                   "holds": bool(all(r["clusters"][k] >= K17_5 for r in dflt.values() for k in far)) if two else None}
    ratio = {s: [r["clusters"][k] / r["molecule_neutral"][k] if r["molecule_neutral"][k] > 0 else float("inf")
                 for k in tail] for s, r in dflt.items()}
    ev["K17.6"] = {"band": K17_6, "ratio": ratio,
                   "holds": bool(all(x >= K17_6 for v in ratio.values() for x in v)) if two and tail else None}
    ev["K17.7"] = {"band": K17_7, "by_seed": {s: [r["slabs"].get(g) for g in SLAB_READ] for s, r in mod.items()},
                   "holds": bool(all(r["slabs"][g] <= K17_7 for r in mod.values() for g in SLAB_READ)) if mod else None}
    ev["K17.8"] = {"band": K17_8, "by_seed": {s: [r["slabs"].get(g) for g in SLAB_READ] for s, r in dflt.items()},
                   "holds": bool(all(r["slabs"][g] >= K17_8 for r in dflt.values() for g in SLAB_READ)) if two else None}
    lr = runs.get("learned", {})
    k9 = {"learned": None, "maceles": None, "holds": None}
    if lr and two:
        k9["learned"] = [float(np.mean([r["clusters"][k] for r in lr.values()])) for k in far]
        k9["maceles"] = [float(np.mean([r["clusters"][k] for r in dflt.values()])) for k in far]
        k9["holds"] = bool(all(a <= b for a, b in zip(k9["learned"], k9["maceles"])))
    ev["K17.9"] = k9
    ev["far_gaps"] = [gaps[k] for k in far]
    return ev


def _verdict(h):
    return "not read" if h is None else ("holds" if h else "FAILS")


def report(ev):
    f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
    gl = lambda v: " / ".join(f(x) for x in v)
    L = ["E4, job 17 (notes/e4-job17-preregistration.md): what a field-blind model with free latent charges cannot do",
         f"  the clusters' gaps {', '.join(f'{g:g}' for g in ev['gaps'])} A; registered from {FAR_GAP:g} A"]
    k = ev["K17.1"]
    L.append(f"  K17.1: {_verdict(k['holds'])} (MACE-LES beyond r_c is its latent charges' Coulomb interaction, within "
             f"{k['band']:g} kcal/mol): worst {f(k['worst'], 9)}; the charges' largest change in AB {f(k['charges_worst'], 12)} e")
    k = ev["K17.2"]
    L.append(f"  K17.2: {_verdict(k['holds'])} (T5: MACE-LES's three-body energy of the far trimers within "
             f"{k['band']:g} kcal/mol of zero): worst {f(k['worst'], 9)}")
    k = ev["K17.3"]
    L.append(f"  K17.3: {_verdict(k['holds'])} (the far probes: the default MACE-LES leaves at least {k['band']:g} of the "
             "non-additive energy at 10 / 12 / 15 A): "
             + "; ".join(f"seed {s} " + " / ".join(f(v, 2) for v in x.values()) for s, x in k["left"].items()))
    for key, label in (("K17.4", f"the clusters: E4's model within {ev['K17.4']['band']:g}"),
                       ("K17.5", f"the clusters: the default MACE-LES at least {ev['K17.5']['band']:g}")):
        k = ev[key]
        L.append(f"  {key}: {_verdict(k['holds'])} ({label}, relative rmse at {', '.join(f'{g:g}' for g in ev['far_gaps'])}"
                 " A): " + "; ".join(f"seed {s} {gl(v)}" for s, v in k["by_seed"].items()))
    k = ev["K17.6"]
    L.append(f"  K17.6: {_verdict(k['holds'])} (the clusters: the default MACE-LES's error over its error with each "
             f"molecule's net charge removed, at least {k['band']:g} at {', '.join(f'{g:g}' for g in TAIL_GAPS)} A): "
             + "; ".join(f"seed {s} {gl(v)}" for s, v in k["ratio"].items()))
    for key, label in (("K17.7", f"the slabs: E4's model within {ev['K17.7']['band']:g}"),
                       ("K17.8", f"the slabs: the default MACE-LES at least {ev['K17.8']['band']:g}")):
        k = ev[key]
        L.append(f"  {key}: {_verdict(k['holds'])} ({label}, relative rmse of E(g) - E(widest) at "
                 f"{', '.join(f'{g:g}' for g in SLAB_READ)} A): " + "; ".join(f"seed {s} {gl(v)}" for s, v in k["by_seed"].items()))
    k = ev["K17.9"]
    L.append(f"  K17.9: {_verdict(k['holds'])} (the clusters: our learned charges at most the default MACE-LES, seed means): "
             f"learned {gl(k['learned']) if k['learned'] else '-'}; MACE-LES {gl(k['maceles']) if k['maceles'] else '-'}")
    L.append("\n  Beside them, not registered")
    t = ev["truth"]
    L.append("    MB-pol's cluster interaction, rms by gap (kcal/mol): " + gl([x for x in t["rms"]])
             + "; its shares, permanent " + gl(t["shares"]["permanent"]) + ", induced " + gl(t["shares"]["induced"])
             + ", dispersion " + gl(t["shares"]["dispersion"]))
    if t.get("tail_ratio"):
        L.append("    the interaction along a ray, 30 A over 16 A (median): MB-pol " + ", ".join(
            f"{n} molecules {f(v)} (R^-3 {f(t['tail_reference'][n]['dipolar'])}, R^-1 {f(t['tail_reference'][n]['monopole'])})"
            for n, v in t["tail_ratio"].items()))
    L.append("    the slabs: MB-pol's E(g) - E(widest), rms (kcal/mol per cell) " + ", ".join(
        f"{g:g} A {f(v, 4)}" for g, v in t["slabs_rms"].items()) + f"; the cell's dipole along z, rms: permanent "
             f"{f(t['slab_dipole']['permanent_rms'])}, induced {f(t['slab_dipole']['induced_rms'])} e A")
    for role, by in ev["harness"].items():
        for s, r in sorted(by.items()):
            line = (f"    {role} seed {s}: clusters {gl(r['clusters'])}; slabs "
                    + " / ".join(f(r['slabs'].get(g)) for g in sorted(r["slabs"])))
            if r.get("eps_inf"):
                line += f"; eps_inf {f(r['eps_inf']['model'], 4)} against {f(r['eps_inf']['truth'], 4)}"
            if r.get("probes"):
                b = r["probes"]["by_distance_angstrom"]
                line += "; probes non-additive left " + " / ".join(
                    f(b[dd]["nonadditive_rmse"] / b[dd]["nonadditive_rms"], 3) for dd in PROBE_DISTANCES if dd in b)
            if r.get("far_trimers"):
                line += f"; T5 slope {f(r['far_trimers']['slope'], 4)}"
            L.append(line)
    for g, by in ev["les"].items():
        for s, r in sorted(by.items()):
            line = (f"    MACE-LES {g} seed {s}: clusters {gl(r['clusters'])}; molecules neutral {gl(r['molecule_neutral'])};"
                    f" clusters neutral {gl(r['cluster_neutral'])}; net charge rms "
                    + ", ".join(f"{n} molecules {f(v, 4)} e" for n, v in r["net_charge_rms"].items())
                    + "; 30 A over 16 A " + ", ".join(f"{f(v)}" for v in r["tail_ratio"].values())
                    + "; slabs " + " / ".join(f(r['slabs'].get(gg)) for gg in sorted(r["slabs"]))
                    + f", net charge rms {f(r['slab_net_charge_rms'], 4)} e, dipole rms {f(r['slab_dipole_rms'], 3)} e A")
            if r.get("probes"):
                b = r["probes"]["by_distance_angstrom"]
                line += "; probes non-additive left " + " / ".join(
                    f(b[dd]["nonadditive_rmse"] / b[dd]["nonadditive_rms"], 3) for dd in PROBE_DISTANCES if dd in b)
                line += ", interaction left " + " / ".join(
                    f(b[dd]["interaction_rmse"] / b[dd]["interaction_rms"], 3) for dd in PROBE_DISTANCES if dd in b)
            if r.get("far_trimers"):
                line += f"; T5 slope {f(r['far_trimers']['slope'], 6)}"
            L.append(line)
    L.append("    eps_inf: every model without a field term (the dagger, the learned charges, the cutoff model, MACE-LES as "
             "deployed) has 1 by construction")
    return "\n".join(L)


def run_read(a):
    T = load_truth(a.truth)
    S = dict(np.load(a.slabs, allow_pickle=False))
    H = load_harness(a.out_dir, T)
    L = load_les({"maceles": a.les_default, "others": a.les_others})
    ev = evaluate(T, S, H, L, far_path=a.probes)
    print(report(ev))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(ev, fh, indent=1, default=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--mbx", required=True)
    m.add_argument("--source", default=SOURCE)
    m.add_argument("--out", default=TRUTH)
    m.add_argument("--slabs", default=SLABS)
    m.add_argument("--clusters_xyz", default=CLUSTERS_XYZ)
    m.add_argument("--slabs_xyz", default=SLABS_XYZ)
    m.add_argument("--probes", default=PROBES)
    m.add_argument("--far_xyz", default=FAR_XYZ)
    h = sub.add_parser("harness")
    h.add_argument("--run", required=True, help="a run's stem: <dir>/<rung>_seed<seed> (.json and .params.pkl)")
    h.add_argument("--source", default=SOURCE)
    h.add_argument("--truth", default=TRUTH)
    h.add_argument("--slabs", default=SLABS)
    h.add_argument("--out_dir", default=OUT_DIR)
    h.add_argument("--chunk", type=int, default=32)
    r = sub.add_parser("read")
    r.add_argument("--truth", default=TRUTH)
    r.add_argument("--slabs", default=SLABS)
    r.add_argument("--out_dir", default=OUT_DIR)
    r.add_argument("--probes", default=PROBES)
    r.add_argument("--les_default", default=LES_GLOBS["maceles"])
    r.add_argument("--les_others", default=LES_GLOBS["others"])
    r.add_argument("--out", default=None, help="also write the readings as json")
    a = ap.parse_args()
    if a.cmd in ("make", "harness"):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    {"make": run_make, "harness": run_harness, "read": run_read}[a.cmd](a)


if __name__ == "__main__":
    main()
