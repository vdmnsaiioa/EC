"""
E3-QP7's numbers before training (notes/e3qp7-preregistration.md section 3), from the PES and the data alone, with the
harness's own field tensors -- every molecule whole between 5 and 6 A, every molecule's share apart
(bandfields.local_field_tensors with frag_src and per_frag, as the cross rungs read them):

  (a) the two numbers that carry the raw CCSD(T) trimers' 3-body through the field input's first-order cross term,
      label_3b ~ -sum_i a_Z(i) E_i(B) . E_i(C)   (E3-QP6 pre-registration 3b: 0.078 of 0.623 kcal/mol left, a_O 4.66,
      a_H 4.18 bohr^3), refitted through the per-molecule tensors -- the same numbers, the same residual;
  (b) the same cross term, with those two numbers fixed, on what the cross rungs will be read on and trained on: the 24
      test hexamers' non-additive energy (their energy less their 15 pairs'), the 300 training clusters' non-additive
      energy by size (rebuilt bit for bit, as E3-QP4 to E3-QP6 built them); and the two numbers refitted on the clusters;
  (c) the self terms the two numbers bring, -1/2 sum_i a_Z(i) sum_B |E_i(B)|^2 -- the 2-body E3-QP6's degree-2 rung had to
      cancel (e3qp6-results.md section 3.4) and the cross rungs do not carry -- in the hexamers and the clusters, against
      the cross term; and on the 4937 raw dimers the cross term itself: zero on every one, to the last digit.

    python scratch/e3qp7_preregistration.py [build_pol]          # ~1.5 min, one CPU
"""
import itertools, math, os, sys, time
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
ON, OFF = (ang_to_bohr(RC - d) for d in E3.E0_FIELD_TAPER_WHOLE)                    # 5, 6 A
L1 = ang_to_bohr(0.75)
rms = lambda v: float(np.sqrt(np.mean(np.square(v)))) if len(v) else float("nan")
A_O, A_H = 4.66, 4.18                                                                # E3-QP6 pre-registration 3b


def cross_and_self(n):
    """for structures of n rigid waters (positions in bohr, (3n, 3)): per species (O, H) the sums over the atoms of the
    cross term sum_{B<C} E_i(B).E_i(C) and of the self terms sum_B |E_i(B)|^2, through the per-molecule tensors."""
    atom_mol = np.repeat(np.arange(n), 3)
    src_mol = np.concatenate([atom_mol, np.arange(n)])
    q_src = jnp.asarray(np.concatenate([np.tile(q_r[:3], n), np.full(n, vq_r[0])]))
    src = jnp.asarray(atom_mol[:, None] != src_mol[None, :])
    frag_src = jnp.asarray(src_mol[None, :] == np.arange(n)[:, None])
    is_O = jnp.asarray(np.tile([1.0, 0.0, 0.0], n))

    def one(P):
        X = jnp.concatenate([P, jnp.einsum("a,max->mx", jnp.asarray(w_r[0][:3]), P.reshape(n, 3, 3))], axis=0)
        _, E, _, (_, E_f, _, _) = BF.local_field_tensors(P, X, q_src, src, L1, ON, OFF, frag_src=frag_src, per_frag=True)
        tot = jnp.sum(E * E, axis=-1)
        selfs = jnp.sum(E_f * E_f, axis=(-2, -1))
        cross = 0.5 * (tot - selfs)
        return jnp.stack([jnp.sum(is_O * cross), jnp.sum((1 - is_O) * cross),
                          jnp.sum(is_O * selfs), jnp.sum((1 - is_O) * selfs)])
    return jax.jit(jax.vmap(one))


def run(structs_ang, n):
    return np.asarray(cross_and_self(n)(jnp.asarray(ang_to_bohr(np.asarray(structs_ang, float)))))


# (a) the raw trimers
a3 = type("A", (), dict(heldout="3b", heldout_oh_tol=0.03, heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02,
                        heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0, heldout_max=0))()
d3 = E3.prepare_heldout(a3, truth)["3b"]
P3 = np.asarray(d3["pos"], float); lab3 = np.asarray(d3["label"]) * K
Z3 = run(P3.reshape(-1, 9, 3), 3)
coef, *_ = np.linalg.lstsq(-Z3[:, :2] * K, lab3, rcond=None)
res3 = lab3 + (Z3[:, :2] * K) @ coef
print(f"(a) {len(P3)} raw trimers: the two-number fit through the per-molecule tensors leaves {rms(res3):.4f} of "
      f"{rms(lab3):.4f} kcal/mol rms; a_O {coef[0]:.2f}, a_H {coef[1]:.2f} bohr^3 (E3-QP6: 0.0775, 4.66, 4.18)  "
      f"[{time.time() - t0:.0f} s]", flush=True)
cross_pred = lambda Z, a=(A_O, A_H): -K * (a[0] * Z[:, 0] + a[1] * Z[:, 1])
self_pred = lambda Z, a=(A_O, A_H): -0.5 * K * (a[0] * Z[:, 2] + a[1] * Z[:, 3])


def nonadditive(structs):
    """the truth's non-additive energy (kcal/mol): the interaction energy less the sum of its pairs'."""
    out = []
    for st in structs:
        pos = bohr_to_ang(np.asarray(st.positions)); n = len(pos) // 3
        e2 = sum(truth.energy(np.concatenate([pos[3 * i:3 * i + 3], pos[3 * j:3 * j + 3]]), 2)
                 for i, j in itertools.combinations(range(n), 2))
        out.append((st.energy - e2) * K)
    return np.array(out)


geometry = dict(T_K=300.0, radius_ang=None, min_OO=2.7, min_any=1.7)
hexamers = W.cluster_dataset((6,), 24, seed=100, truth=truth, mc_steps=300, **geometry)
train = W.cluster_dataset((3, 4, 5), 100, seed=0, sigma=1e-9, truth=truth, mc_steps=300, **geometry)
print(f"    the 24 test hexamers and {len(train)} training clusters rebuilt  [{time.time() - t0:.0f} s]", flush=True)

# (b) the hexamers and the training clusters
rows = []
for label, structs in [("the 24 test hexamers", hexamers)] + [(f"the training clusters of {n}", [s for s in train if s.n_atoms == 3 * n])
                                                                 for n in (3, 4, 5)]:
    n = structs[0].n_atoms // 3
    na = nonadditive(structs)
    Z = run([bohr_to_ang(np.asarray(s.positions)) for s in structs], n)
    pc, ps = cross_pred(Z), self_pred(Z)
    rows.append((label, na, Z))
    print(f"(b) {label} ({len(structs)}): non-additive energy rms {rms(na):.3f} kcal/mol; the cross term with the trimers' two "
          f"numbers {rms(pc):.3f} rms, correlation {np.corrcoef(pc, na)[0, 1]:+.3f}, leaves {rms(na - pc):.3f}; "
          f"(c) the self terms: mean {np.mean(ps):+.2f}, spread (rms about the mean) {rms(ps - np.mean(ps)):.3f}  "
          f"[{time.time() - t0:.0f} s]", flush=True)
na_c = np.concatenate([r[1] for r in rows[1:]]); Z_c = np.concatenate([r[2] for r in rows[1:]])
coef_c, *_ = np.linalg.lstsq(-Z_c[:, :2] * K, na_c, rcond=None)
print(f"    refitted on the {len(na_c)} training clusters' non-additive energy: a_O {coef_c[0]:.2f}, a_H {coef_c[1]:.2f} "
      f"bohr^3, leaving {rms(na_c + (Z_c[:, :2] * K) @ coef_c):.3f} of {rms(na_c):.3f}; on the hexamers then "
      f"{rms(rows[0][1] - cross_pred(rows[0][2], coef_c)):.3f} of {rms(rows[0][1]):.3f}; on the raw trimers "
      f"{rms(lab3 - cross_pred(Z3, coef_c)):.4f}", flush=True)

# (c) the raw dimers: the cross term is zero on every one
a2 = type("A", (), dict(heldout="2b", heldout_oh_tol=0.03, heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02,
                        heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0, heldout_max=0))()
d2 = E3.prepare_heldout(a2, truth)["2b"]
Z2 = run(np.asarray(d2["pos"], float).reshape(-1, 6, 3), 2)
ps2 = self_pred(Z2)
print(f"(c) {len(Z2)} raw dimers: the cross term's largest magnitude {np.max(np.abs(Z2[:, :2])):.1e} (a.u.^2); the self "
      f"terms there mean {np.mean(ps2):+.3f}, spread {rms(ps2 - np.mean(ps2)):.3f} kcal/mol  [{time.time() - t0:.0f} s]",
      flush=True)
print(f"done in {time.time() - t0:.0f} s")
