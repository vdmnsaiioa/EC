"""
E3-QP8's numbers before training (notes/e3qp8-preregistration.md section 3), from the PES and the data alone, with the
harness's own field tensors -- every molecule whole, the term over 5-6 A and the self terms over 4-5 A, every
molecule's share apart (bandfields.local_field_tensors with frag_src, per_frag and self_taper, as the split rungs read
them):

  (a) where the self terms' reach cuts: on every atom-molecule pair in the term's reach, the self weight w (1 below
      4 A for every site of the molecule, 0 beyond 5 A for any) -- the share at w = 1, its mean, and the share of
      the potential and of the squared field the self terms read that falls outside it, sum (1 - w)|x| / sum |x| -- on
      the raw dimers by O-O, the raw trimers, the 24 test hexamers, the 300 training clusters and the 800 contact
      configurations (rebuilt bit for bit, as E3-QP4 to E3-QP7 built them);
  (b) the cross terms by species: the fit of -sum_i [a_Z(i) sum_{B<C} E_i(B).E_i(C) + c_Z(i) sum_{B<C} G_i(B):G_i(C)] to
      the raw trimers' 3-body labels -- four numbers, and the two of E3-QP6/E3-QP7 (a_Z alone: 0.0775 of 0.6225) --
      and with those numbers fixed, on the hexamers' and the training clusters' non-additive energy; refitted there;
  (c) with --e3qp7 <results/e3qp7> (the saved models, as the sbatch runs it before training): M_16C2pm_w's 2-body on
      the raw dimers by O-O as trained and with its self terms cut to 4-5 A, not retrained (the same parameters
      read as M_16C2pm_ws), with M_16C2pm's for reference -- what the cut does before E_0 adapts.

    python scratch/e3qp8_preregistration.py [build_pol]                                # ~3 min, one CPU
    python scratch/e3qp8_preregistration.py build_pol --e3qp7 results/e3qp7 --only_trained   # (c) alone
"""
import argparse, glob, itertools, json, math, os, pickle, re, sys, time
import numpy as np
import jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, HERE)
from sseft import water as W, bandfields as BF
from sseft.units import ang_to_bohr, bohr_to_ang
import e3_water as E3

K = 627.5094740631
RC = 6.0
ON, OFF = (ang_to_bohr(RC - d) for d in E3.E0_FIELD_TAPER_WHOLE)                    # 5, 6 A
S_ON, S_OFF = (ang_to_bohr(RC - d) for d in E3.E0_FIELD_SELF_TAPER)                 # 4, 5 A
L1 = ang_to_bohr(0.75)
DIMER_BINS = ((2.3, 2.7), (2.7, 2.9), (2.9, 3.0), (3.0, 3.2), (3.2, 4.0), (4.0, 5.0), (5.0, 6.0), (6.0, 7.0), (7.0, 99.0))
rms = lambda v: float(np.sqrt(np.mean(np.square(v)))) if len(v) else float("nan")


def tensors_fn(n, q_r, w_r, vq_r):
    """for structures of n rigid waters (bohr, (3n, 3)) with the rigid monomer's TTM3-F charges: per atom-molecule pair
    the term's reach, the self weight, |phi|, |E|^2 (N, n each) and per species (O, H) the cross sums
    sum_{B<C} E_i(B).E_i(C) and sum_{B<C} G_i(B):G_i(C)."""
    atom_mol = np.repeat(np.arange(n), 3)
    src_mol = np.concatenate([atom_mol, np.arange(n)])
    q_src = jnp.asarray(np.concatenate([np.tile(q_r[:3], n), np.full(n, vq_r[0])]))
    src = jnp.asarray(atom_mol[:, None] != src_mol[None, :])
    frag_src = jnp.asarray(src_mol[None, :] == np.arange(n)[:, None])
    is_O = jnp.asarray(np.tile([1.0, 0.0, 0.0], n))

    def one(P):
        X = jnp.concatenate([P, jnp.einsum("a,max->mx", jnp.asarray(w_r[0][:3]), P.reshape(n, 3, 3))], axis=0)
        _, E, G, (phi_f, E_f, G_f, reach, w_self) = BF.local_field_tensors(P, X, q_src, src, L1, ON, OFF, frag_src=frag_src,
                                                                          per_frag=True, self_taper=(S_ON, S_OFF))
        e_cross = 0.5 * (jnp.sum(E * E, -1) - jnp.sum(E_f * E_f, axis=(-2, -1)))
        g_cross = 0.5 * (jnp.sum(G * G, axis=(-2, -1)) - jnp.sum(G_f * G_f, axis=(-3, -2, -1)))
        sums = jnp.stack([jnp.sum(is_O * e_cross), jnp.sum((1 - is_O) * e_cross),
                          jnp.sum(is_O * g_cross), jnp.sum((1 - is_O) * g_cross)])
        return reach, w_self, jnp.abs(phi_f), jnp.sum(E_f * E_f, -1), sums
    return jax.jit(jax.vmap(one))


def run(fn_cache, structs_ang, n, site):
    if n not in fn_cache:
        fn_cache[n] = tensors_fn(n, *site)
    out = fn_cache[n](jnp.asarray(ang_to_bohr(np.asarray(structs_ang, float))))
    return [np.asarray(x) for x in out]


def reach_reading(label, reach, w, phi, e2):
    """the self weights on the atom-molecule pairs in the term's reach."""
    m = reach > 0
    if not m.any():
        return f"{label}: none in reach"
    return (f"{label}: {int(m.sum())} pairs in reach, w = 1 {100 * np.mean(w[m] > 1 - 1e-12):5.1f} %, mean w {w[m].mean():.3f}; "
            f"cut: the potential's share {100 * np.sum((reach - w) * phi) / np.sum(reach * phi):5.1f} %, "
            f"the squared field's {100 * np.sum((reach - w) * e2) / np.sum(reach * e2):5.2f} %")


def nonadditive(truth, structs):
    """the truth's non-additive energy (kcal/mol): the interaction energy less the sum of its pairs'."""
    out = []
    for st in structs:
        pos = bohr_to_ang(np.asarray(st.positions)); n = len(pos) // 3
        e2 = sum(truth.energy(np.concatenate([pos[3 * i:3 * i + 3], pos[3 * j:3 * j + 3]]), 2)
                 for i, j in itertools.combinations(range(n), 2))
        out.append((st.energy - e2) * K)
    return np.array(out)


def heldout(truth, key):
    a = type("A", (), dict(heldout=key, heldout_oh_tol=0.03, heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02,
                           heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0, heldout_max=0))()
    return E3.prepare_heldout(a, truth)[key]


def trained_cut(args, truth, t0):
    """(c): M_16C2pm_w's saved models as trained and read as M_16C2pm_ws (its self terms cut, the same parameters), and
    M_16C2pm's, on the raw dimers' 2-body by O-O against the labels (the pins as trained)."""
    from e3qp3_shortrange_diagnostic import check_record
    import jax.tree_util as jtu
    run_json = json.load(open(os.path.join(args.e3qp7, "clusters.json")))
    a = type("A", (), dict(lA=4.0, rcut=RC, F=32, l1=0.75, rs_over_l1=0.0, response_edge=0.75, charges=None,
                           charge_neutrality="fragment"))()
    r_w, r_ws, r_b = (E3.configure(E3.RUNGS[n], a) for n in ("M_16C2pm_w", "M_16C2pm_ws", "M_16C2pm"))
    check_record("M_16C2pm_w", r_w, (run_json.get("M_16C2pm_w") or {}).get("model"))
    check_record("M_16C2pm", r_b, (run_json.get("M_16C2pm") or {}).get("model"))
    frames = os.path.join(args.e3qp7, "shortrange_diagnostic_frames.npz")
    if os.path.exists(frames):
        Z = np.load(frames); P, lab, oo = Z["2b__P"], Z["2b__label"] * K, Z["2b__soo"]
    else:
        d = heldout(truth, "2b"); P, lab, oo = np.asarray(d["pos"]), np.asarray(d["label"]) * K, np.asarray(d["soo"])
    q = [W.rigid_molecule_charges(truth, 2)] * len(P)
    seeds = lambda name: sorted(int(re.search(r"_seed(\d+)\.params\.pkl$", p).group(1))
                                for p in glob.glob(os.path.join(args.e3qp7, f"clusters_{name}_seed*.params.pkl")))
    load = lambda name, s: jtu.tree_map(jnp.asarray, pickle.load(open(os.path.join(args.e3qp7, f"clusters_{name}_seed{s}.params.pkl"), "rb")))
    errs = {"M_16C2pm": [], "M_16C2pm_w": [], "cut": []}
    for s in seeds("M_16C2pm_w"):
        p = load("M_16C2pm_w", s)
        errs["M_16C2pm_w"].append(E3.model_nbody(p, r_w, P, q, 2, truth, rigid=True) * K - lab)
        errs["cut"].append(E3.model_nbody(p, r_ws, P, q, 2, truth, rigid=True) * K - lab)
        print(f"    M_16C2pm_w seed {s} read as trained and with its self terms cut  [{time.time() - t0:.0f} s]", flush=True)
    for s in seeds("M_16C2pm"):
        errs["M_16C2pm"].append(E3.model_nbody(load("M_16C2pm", s), r_b, P, q, 2, truth, rigid=True) * K - lab)
    print(f"(c) the raw dimers' 2-body error (kcal/mol rms, seed means): M_16C2pm / M_16C2pm_w as trained / M_16C2pm_w with "
          f"its self terms cut to {RC - E3.E0_FIELD_SELF_TAPER[0]:g}-{RC - E3.E0_FIELD_SELF_TAPER[1]:g} A, not retrained "
          f"(the same parameters read as M_16C2pm_ws)")
    bins = (("all", np.ones(len(P), bool)),) + tuple((f"O-O {lo:g}-{hi:g}", (oo >= lo) & (oo < hi))
                                                     for lo, hi in ((2.3, 2.7), (2.7, 2.9), (2.9, 3.0), (3.0, 3.2), (3.2, 5.0), (5.0, 7.0)))
    for label, m in bins:
        if m.any():
            cols = [np.mean([rms(e[m]) for e in errs[k]]) if errs[k] else float("nan") for k in ("M_16C2pm", "M_16C2pm_w", "cut")]
            print(f"    {label:<14} (n {int(m.sum()):4d}): " + " / ".join(f"{c:.4f}" for c in cols), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=os.path.join(REPO, "external", "qaqua", "build_pol"))
    ap.add_argument("--e3qp7", default=None, help="E3-QP7's results directory: (c) on its saved models")
    ap.add_argument("--only_trained", action="store_true", help="(c) alone")
    args = ap.parse_args()
    t0 = time.time()
    truth = W.qaqua_pol_truth(args.root)
    if args.e3qp7 and args.only_trained:
        trained_cut(args, truth, t0)
        return
    q_r, w_r, vq_r, _ = truth.site_model(W.monomer_positions())
    site = (q_r, w_r, vq_r)
    cache = {}

    # (a) the raw dimers by O-O, the raw trimers
    d2 = heldout(truth, "2b")
    P2, oo2 = np.asarray(d2["pos"], float), np.asarray(d2["soo"])
    reach, w, phi, e2, _ = run(cache, P2.reshape(-1, 6, 3), 2, site)
    print(f"(a) the self weights, {RC - E3.E0_FIELD_SELF_TAPER[0]:g}-{RC - E3.E0_FIELD_SELF_TAPER[1]:g} A, every molecule whole "
          f"(the term's reach {RC - E3.E0_FIELD_TAPER_WHOLE[0]:g}-{RC:g} A)  [{time.time() - t0:.0f} s]", flush=True)
    for lo, hi in DIMER_BINS:
        m = (oo2 >= lo) & (oo2 < hi)
        if m.any():
            print("    " + reach_reading(f"raw dimers O-O {lo:g}-{hi:g} ({int(m.sum())})", reach[m], w[m], phi[m], e2[m]), flush=True)
    d3 = heldout(truth, "3b")
    P3, lab3 = np.asarray(d3["pos"], float), np.asarray(d3["label"]) * K
    reach, w, phi, e2, sums3 = run(cache, P3.reshape(-1, 9, 3), 3, site)
    print("    " + reach_reading(f"raw trimers ({len(P3)})", reach, w, phi, e2), flush=True)

    geometry = dict(T_K=300.0, radius_ang=None, min_OO=2.7, min_any=1.7)
    hexamers = W.cluster_dataset((6,), 24, seed=100, truth=truth, mc_steps=300, **geometry)
    train = W.cluster_dataset((3, 4, 5), 100, seed=0, sigma=1e-9, truth=truth, mc_steps=300, **geometry)
    contacts = (W.contact_configurations(2, 600, (2.3, 3.0), seed=700, truth=truth, max_pair_kcal=2.0, min_any=1.5)
                + W.contact_configurations(3, 200, (2.3, 3.0), seed=701, truth=truth, max_pair_kcal=2.0, min_any=1.5))
    print(f"    the 24 test hexamers, {len(train)} training clusters and {len(contacts)} contact configurations rebuilt  "
          f"[{time.time() - t0:.0f} s]", flush=True)
    groups = [("the 24 test hexamers", hexamers)] + [(f"the training clusters of {n}", [s for s in train if s.n_atoms == 3 * n])
                                                     for n in (3, 4, 5)]
    groups += [(f"the contact {'dimers' if n == 2 else 'trimers'}", [s for s in contacts if s.n_atoms == 3 * n]) for n in (2, 3)]
    sums = {}
    for label, structs in groups:
        n = structs[0].n_atoms // 3
        reach, w, phi, e2, S = run(cache, [bohr_to_ang(np.asarray(s.positions)) for s in structs], n, site)
        sums[label] = S
        print("    " + reach_reading(f"{label} ({len(structs)})", reach, w, phi, e2), flush=True)

    # (b) the cross terms by species
    X3 = -sums3 * K                                     # columns: a_O, a_H, c_O, c_H
    fit2, *_ = np.linalg.lstsq(X3[:, :2], lab3, rcond=None)
    fit4, *_ = np.linalg.lstsq(X3, lab3, rcond=None)
    print(f"(b) {len(P3)} raw trimers' 3-body (labels rms {rms(lab3):.4f}): two numbers (a_O {fit2[0]:.2f}, a_H {fit2[1]:.2f} "
          f"bohr^3) leave {rms(lab3 - X3[:, :2] @ fit2):.4f}; four (a_O {fit4[0]:.2f}, a_H {fit4[1]:.2f} bohr^3, c_O {fit4[2]:.1f}, "
          f"c_H {fit4[3]:.1f} bohr^5) leave {rms(lab3 - X3 @ fit4):.4f} kcal/mol  [{time.time() - t0:.0f} s]", flush=True)
    by_bin = [(f"closest O-O {lo:g}-{hi:g}", (np.asarray(d3["soo"]) >= lo) & (np.asarray(d3["soo"]) < hi))
              for lo, hi in ((0.0, 2.7), (2.7, 3.0), (3.0, 99.0))]
    print("    by closest O-O: " + "; ".join(f"{lab} ({int(m.sum())}) {rms(lab3[m]):.3f}: two {rms((lab3 - X3[:, :2] @ fit2)[m]):.3f}, "
                                          f"four {rms((lab3 - X3 @ fit4)[m]):.3f}" for lab, m in by_bin), flush=True)
    na, Xs = {}, {}
    for label, structs in groups[:4]:
        na[label] = nonadditive(truth, structs); Xs[label] = -sums[label] * K
        print(f"    {label}: non-additive rms {rms(na[label]):.3f}; with the trimers' numbers left: two {rms(na[label] - Xs[label][:, :2] @ fit2):.3f}, "
              f"four {rms(na[label] - Xs[label] @ fit4):.3f}  [{time.time() - t0:.0f} s]", flush=True)
    na_c = np.concatenate([na[g[0]] for g in groups[1:4]]); X_c = np.concatenate([Xs[g[0]] for g in groups[1:4]])
    fitc, *_ = np.linalg.lstsq(X_c, na_c, rcond=None)
    hex_label = groups[0][0]
    print(f"    refitted on the {len(na_c)} training clusters' non-additive energy (four): a_O {fitc[0]:.2f}, a_H {fitc[1]:.2f}, "
          f"c_O {fitc[2]:.1f}, c_H {fitc[3]:.1f}, leaving {rms(na_c - X_c @ fitc):.3f} of {rms(na_c):.3f}; the hexamers then "
          f"{rms(na[hex_label] - Xs[hex_label] @ fitc):.3f} of {rms(na[hex_label]):.3f}; the raw trimers {rms(lab3 - X3 @ fitc):.4f}",
          flush=True)
    if args.e3qp7:
        trained_cut(args, truth, t0)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
