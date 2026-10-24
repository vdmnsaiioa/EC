"""
E3-QP6's numbers before training (notes/e3qp6-preregistration.md section 3), from the PES and the data alone, with the
harness's own field tensors (bandfields.local_field_tensors, every molecule whole: frag_src):

  (a) the field term's reach -- for every probe configuration of the run (E3-QP4's and E3-QP5's training and test
      probes, rebuilt bit for bit), whether any core atom sees the probe molecule (all its sites, the O, the two H and the
      M-site, within the taper's end r_c = 6 A, so that its weight is not zero) or any probe atom a core molecule; against
      E3-QP5's reach (any source within 5.5 A); and how often a molecule is in reach, whole or in part, in the test
      hexamers and the training clusters;
  (b) how much of the raw CCSD(T) trimers' 3-body (E3-QP4's 4673 frames, the rigid monomer's charges as the models read
      them) one effective polarisability per species on the field input's first-order cross term explains,
      label_3b ~ -sum_i a_Z(i) E_i(B) . E_i(C), with every molecule whole between 5 and 6 A -- against E3-QP5's per-source
      taper (4.5-5.5 A; 0.114 of 0.623 kcal/mol rms left) and none, by the smallest and the largest O-O.

    python scratch/e3qp6_preregistration.py [build_pol]          # ~3 min, one CPU
"""
import math, os, sys, time
import numpy as np
import jax, jax.numpy as jnp

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "external", "qaqua", "build_pol")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from sseft import water as W, bandfields as BF
from sseft.units import ang_to_bohr, bohr_to_ang
import e3_water as E3

K = 627.5094740631
t0 = time.time()
truth = W.qaqua_pol_truth(ROOT)
q_r, w_r, vq_r, _ = truth.site_model(W.monomer_positions())
RC = 6.0
ON, OFF = RC - E3.E0_FIELD_TAPER_WHOLE[0], RC - E3.E0_FIELD_TAPER_WHOLE[1]           # 5.0, 6.0 A
ON5, OFF5 = RC - E3.E0_FIELD_TAPER[0], RC - E3.E0_FIELD_TAPER[1]                    # E3-QP5's 4.5, 5.5 A
L1 = ang_to_bohr(0.75)
rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
print(f"the taper: every molecule whole, the product of its sites' tapers between {ON:g} and {OFF:g} A (E3-QP5: each "
      f"source by its own distance, {ON5:g}-{OFF5:g} A)", flush=True)


def sites(pos_ang):
    """atoms then M-sites (A), and each site's molecule."""
    n = len(pos_ang) // 3
    P = np.asarray(pos_ang, float)
    Ms = np.array([w_r[0] @ P[3 * m:3 * m + 3] for m in range(n)])
    return P, np.concatenate([P, Ms]), np.concatenate([np.repeat(np.arange(n), 3), np.arange(n)])


# (a) the reach
geometry = dict(T_K=300.0, radius_ang=None, min_OO=2.7, min_any=1.7)
probes = {"train": [], "test": []}
for key, seed, n_cores, count, sizes in (("train", 400, 16, 2, (3, 4, 5)), ("test", 500, 2, 10, (3,))):
    for k, n_core in enumerate(sizes):
        probes[key] += W.probe_configurations(n_core, (8.0, 10.0, 12.0, 15.0), count, seed=seed + 1000 * k, mc_steps=300,
                                              n_cores=n_cores, truth=truth, **geometry)[0]
print(f"(a) {len(probes['train'])} training and {len(probes['test'])} test probes rebuilt  [{time.time() - t0:.0f} s]", flush=True)
for key, structs in probes.items():
    rows = {}
    for st in structs:
        pos = bohr_to_ang(np.asarray(st.positions)); n = len(pos) // 3
        P, X, mol = sites(pos)
        atom_mol = np.repeat(np.arange(n), 3)
        r = np.linalg.norm(P[:, None] - X[None], axis=-1)                              # (atoms, sites)
        far = [np.max(np.where(mol[None, :] == m, r, 0.0), axis=1) for m in range(n)]   # each atom to each molecule's farthest site
        far = np.stack(far, axis=1)                                                     # (atoms, molecules)
        probe = n - 1
        whole = (far[atom_mol < probe, probe].min() < OFF) or (far[atom_mol == probe][:, :probe].min() < OFF)
        near = [np.min(np.where(mol[None, :] == m, r, np.inf), axis=1) for m in range(n)]
        near = np.stack(near, axis=1)
        per_source = (near[atom_mol < probe, probe].min() < OFF5) or (near[atom_mol == probe][:, :probe].min() < OFF5)
        gap = np.linalg.norm(pos[:-3, None] - pos[None, -3:], axis=-1).min()
        d = st.info["d"]
        row = rows.setdefault(d, [0, 0, 0, 0])
        row[0] += 1; row[1] += gap > RC; row[2] += not per_source; row[3] += not whole
    print(f"    {key} probes by d: " + "; ".join(f"{d:g} A: {n_} probes, {b} beyond r_c of the core (E_0 blind), {f5} beyond "
                                                 f"E3-QP5's reach, {f6} beyond E3-QP6's" for d, (n_, b, f5, f6) in sorted(rows.items())),
          flush=True)


def reach_in(structs, label):
    full = part = 0; total = 0
    for st in structs:
        pos = bohr_to_ang(np.asarray(st.positions)); n = len(pos) // 3
        P, X, mol = sites(pos)
        atom_mol = np.repeat(np.arange(n), 3)
        r = np.linalg.norm(P[:, None] - X[None], axis=-1)
        x = np.clip((r - ON) / (OFF - ON), 0.0, 1.0); t = 1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x ** 2)
        for m in range(n):
            wm = np.prod(np.where(mol[None, :] == m, t, 1.0), axis=1)[atom_mol != m]
            full += np.sum(wm == 1.0); part += np.sum((wm > 0.0) & (wm < 1.0)); total += wm.size
    print(f"    {label}: of the (atom, other molecule) pairs {100 * full / total:.0f} % whole in reach (weight 1), "
          f"{100 * part / total:.0f} % in the taper, {100 * (total - full - part) / total:.0f} % beyond", flush=True)


reach_in(W.cluster_dataset((6,), 24, seed=100, truth=truth, mc_steps=300, **geometry), "the 24 test hexamers")
train = W.cluster_dataset((3, 4, 5), 100, seed=0, sigma=1e-9, truth=truth, mc_steps=300, **geometry)
reach_in(train, f"the {len(train)} training clusters  [{time.time() - t0:.0f} s]")

# (b) the raw trimers' 3-body against the field input's first-order cross term, with the harness's tensors
a = type("A", (), dict(heldout="3b", heldout_oh_tol=0.03, heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02,
                       heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0, heldout_max=0))()
d3 = E3.prepare_heldout(a, truth)["3b"]
Pa = np.asarray(d3["pos"]); lab = np.asarray(d3["label"]) * K; soo = np.asarray(d3["soo"])
Oa = Pa.reshape(-1, 3, 3, 3)[:, :, 0]
loo = np.max(np.linalg.norm(Oa[:, :, None] - Oa[:, None], axis=-1).reshape(len(Pa), 9), axis=1)
q_src = jnp.asarray(np.concatenate([np.tile(q_r[:3], 3), np.full(3, vq_r[0])]))
src_mol = np.concatenate([np.repeat(np.arange(3), 3), np.arange(3)])
atom_mol = np.repeat(np.arange(3), 3)
frag_src = jnp.asarray(src_mol[None, :] == np.arange(3)[:, None])
src_by = [jnp.asarray((atom_mol[:, None] != src_mol[None, :]) & (src_mol[None, :] == B)) for B in range(3)]


def fields_by_molecule(P, on, off, whole):
    """(9 atoms, 3 molecules, 3): the field at every atom of each other molecule's pinned charges, as the rungs read it"""
    X = jnp.concatenate([P, jnp.einsum("a,max->mx", jnp.asarray(w_r[0][:3]), P.reshape(3, 3, 3))], axis=0)   # + M-sites
    out = [BF.local_field_tensors(P, X, q_src, s, L1, on, off, frag_src=frag_src if whole else None)[1] for s in src_by]
    return jnp.stack(out, axis=1)


def cross(on, off, whole):
    f = jax.jit(jax.vmap(lambda P: fields_by_molecule(P, on, off, whole)))
    E = np.asarray(f(jnp.asarray(ang_to_bohr(Pa.astype(float)))))
    X = np.zeros((len(Pa), 2))
    for m in range(3):
        y, z = [k for k in range(3) if k != m]
        c = np.sum(E[:, 3 * m:3 * m + 3, y] * E[:, 3 * m:3 * m + 3, z], axis=-1)
        X[:, 0] += c[:, 0]; X[:, 1] += c[:, 1] + c[:, 2]
    return X


S_BINS = ((0.0, 2.7), (2.7, 3.0), (3.0, 99.0))
L_BINS = ((0.0, 5.0), (5.0, 7.0), (7.0, 9.0), (9.0, 99.0))
cols = [f"S{lo:g}-{hi:g}" if hi < 99 else f"S{lo:g}-" for lo, hi in S_BINS] + \
       [f"L{lo:g}-{hi:g}" if hi < 99 else f"L{lo:g}-" for lo, hi in L_BINS]
masks = [(soo >= lo) & (soo < hi) for lo, hi in S_BINS] + [(loo >= lo) & (loo < hi) for lo, hi in L_BINS]
print(f"(b) {len(Pa)} raw trimers: the two-number fit's residual (kcal/mol rms), by the smallest (S) and the largest (L) "
      f"O-O; a_O, a_H (bohr^3)  [{time.time() - t0:.0f} s]", flush=True)
print(f"{'':>30}{'all':>8}" + "".join(f"{c:>9}" for c in cols))
print(f"{'the labels':>30}{rms(lab):8.4f}" + "".join(f"{rms(lab[m]):9.4f}" for m in masks))
print(f"{'n':>30}{len(lab):8d}" + "".join(f"{int(m.sum()):9d}" for m in masks))
for name, on, off, whole in (("no taper", 1e6, 2e6, False),
                             (f"per source {ON5:g}-{OFF5:g} A (E3-QP5)", ang_to_bohr(ON5), ang_to_bohr(OFF5), False),
                             (f"whole {ON:g}-{OFF:g} A (E3-QP6)", ang_to_bohr(ON), ang_to_bohr(OFF), True)):
    X = cross(on, off, whole)
    coef, *_ = np.linalg.lstsq(-X * K, lab, rcond=None)
    res = lab + (X * K) @ coef
    print(f"{name:>30}{rms(res):8.4f}" + "".join(f"{rms(res[m]):9.4f}" for m in masks) + f"   {coef[0]:.2f}, {coef[1]:.2f}",
          flush=True)
    if whole:
        for lo, hi in ((7.0, 9.0), (9.0, 99.0)):
            mm = (loo >= lo) & (loo < hi)
            pred = -(X[mm] * K) @ coef
            print(f"{'':>30}largest O-O {lo:g}-{hi:g} A: the fitted 3-body rms {rms(pred):.4f} (the labels' {rms(lab[mm]):.4f}), "
                  f"correlation {np.corrcoef(pred, lab[mm])[0, 1]:+.2f}", flush=True)

# (c) the raw dimers: where the whole-molecule term sees the partner, by O-O
a2 = type("A", (), dict(heldout="2b", heldout_oh_tol=0.03, heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02,
                        heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0, heldout_max=0))()
d2 = E3.prepare_heldout(a2, truth)["2b"]
P2 = np.asarray(d2["pos"], float).reshape(-1, 6, 3)
oo2 = np.linalg.norm(P2[:, 0] - P2[:, 3], axis=-1)
print(f"(c) {len(P2)} raw dimers by O-O: of the atoms, the partner whole in reach (weight 1) / in the taper / beyond", flush=True)
W2 = []
for P in P2:
    _, X, mol = sites(P)
    r = np.linalg.norm(P[:, None] - X[None], axis=-1)
    x = np.clip((r - ON) / (OFF - ON), 0.0, 1.0); t = 1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x ** 2)
    W2.append([np.prod(np.where(mol == 1 - i // 3, t[i], 1.0)) for i in range(6)])
W2 = np.array(W2)
for lo, hi in ((2.3, 2.7), (2.7, 2.9), (2.9, 3.0), (3.0, 3.2), (3.2, 5.0), (5.0, 7.0), (7.0, 99.0)):
    m = (oo2 >= lo) & (oo2 < hi)
    w = W2[m].ravel()
    print(f"    {lo:g}-{hi:g} A (n {int(m.sum())}): {100 * np.mean(w == 1.0):.1f} / {100 * np.mean((w > 0) & (w < 1)):.1f} / "
          f"{100 * np.mean(w == 0.0):.1f} %", flush=True)
print(f"done in {time.time() - t0:.0f} s")
