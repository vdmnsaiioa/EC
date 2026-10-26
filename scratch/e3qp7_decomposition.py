"""
Where a run's many-body errors live: every saved model's n-body energies split by component and by body order, from the
saved parameters (E3-QP7 section 4; CPU minutes a model, or a short GPU job).

    python scratch/e3qp7_decomposition.py results/e3qp7 external/qaqua/build_pol --seeds all
    python scratch/e3qp7_decomposition.py results/e3qp6 external/qaqua/build_pol --seeds all

A model's interaction energy is the sum of its components (sseft.model._total_energy's aux, each E(all) - sum_f E(f)):
    E0         E_0's site energies
    bands      the pinned bands: the site model's Coulomb band and the dispersion band -- pairwise, no 3-body
    read-out   the band-field read-out (the C rungs' self-consistent response)
    field      the field term (v0.0.14-16), split two ways where the rung has one with every molecule whole:
      self / cross   its self terms -- h on each source molecule alone, sum_B h(s_i; x_i(B)) -- and its cross terms, the
                     rest: the same parameters read as the cross rung (Rung.e0_field_cross) reads them.  A cross rung
                     (M_16C2pm_wx, _w2x) has the cross terms alone; v0.0.17's rungs their self terms at their own reach,
                     sum_B w_iB h(s_i; x_i(B)), beside the MLP's cross terms (M_16C2pm_ws) or degree-2 cross terms with
                     coefficients by species (M_16C2pm_us: no coefficient part, no 4-body).
      fixed / coef   the term at the isolated molecule's features (E_0's s, v, t of every atom with its own molecule
                     alone: the coefficients fixed), and the rest -- what the other molecules do to the coefficients.  At
                     fixed coefficients the self terms are a 2-body exactly and the degree-2 cross terms a 3-body
                     exactly, so all they carry beyond is their coefficients'.
Every quantity's subset energies over a frame's sub-clusters are inverted to its k-body terms (the truth's expansion,
E3.model_nbody's), on
    hex          the run's test hexamers (clusters_cache.pkl): the body orders 2-6 per hexamer, the non-additive part
                 (3-6), and the 20 trimers' 3-body and the 15 tetramers' 4-body terms apart (by their closest O-O),
                 against the truth's own terms (q-AQUA-pol: what the models were trained on); the pins as the run reads
                 the hexamers (the charges at their own geometry, which is the rigid monomer's);
    4b, 3b, 2b   the raw CCSD(T) sets on the run's short-range diagnostic frames (<results>/shortrange_diagnostic_
                 frames.npz, written with --save_frames from v0.0.14; without it the sets are read again from the build),
                 their top order against the labels, the pins as trained (the rigid monomer's charges: the
                 diagnostic's fixed_labels and the E3-QP7 readings).
Readings per rung (seed means, the error's range over the seeds in brackets): the error's rms; every component's k-body
as "size / without": its rms, and the rms of the error with that component's k-body taken out -- below the error where
the component's part is the error's, above where it carries the truth's.  Paired by seed against the base rung
(M_16C2pm) and against M_16C2pm_w: every component's difference (rms), the error with that difference alone ("only":
the base's error plus it), and the correlation of the field's difference with E_0's -- E_0 trained beside the term.
Checks: the components sum to the energy; the field term as read here is the model's; the bands carry no 3-body; at
fixed coefficients the self terms no 3-body and the degree-2 cross terms no 4-body; the energies are the diagnostic's
on the same frames (its __fix and hex__E / hex__2b readings), and the truth's terms its cached hexamer energies and
2-body parts.  Prints the readings; writes <results>/<--out> (json) and every frame's terms to <out stem>_frames.npz.
"""
import argparse, copy, glob, itertools, json, math, os, pickle, re, sys, time
import numpy as np
import jax
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, HERE)
from sseft import bandfields as BF, e0 as E0, heads as H, model as M, water as W
from sseft.structure import pad_batch
from sseft.units import ang_to_bohr, bohr_to_ang
import e3_water as E3

K = 627.5094740631
CONTACT_BINS = ((0.0, 2.7), (2.7, 3.0), (3.0, 99.0))                              # the closest O-O (A)
DIMER_BINS = ((2.3, 2.7), (2.7, 2.9), (2.9, 3.0), (3.0, 3.2), (3.2, 5.0), (5.0, 99.0))
FIELD_PARTS = ("self", "self_fixed", "self_coef", "cross", "cross_fixed", "cross_coef", "field_fixed", "field_coef")
rms = lambda v: float(np.sqrt(np.mean(np.square(v)))) if len(v) else float("nan")
TOL = 1e-7                                     # kcal/mol: the checks of the decomposition itself
TOL_RUN = 1e-6                                 # kcal/mol: against the diagnostic's readings (another device)


# ------------------------------------------------------------------------------- the many-body expansion ----------
def subsets_of(n):
    """the sub-clusters of 2..n molecules, by size (E3.model_nbody's order; the last is the whole frame)."""
    return [c for k in range(2, n + 1) for c in itertools.combinations(range(n), k)]


def mobius(subs):
    """(S, S): row c gives sub-cluster c's own k-body term from the subset interaction energies,
    sum_{d <= c, |d| >= 2} (-1)^{|c|-|d|} E(d) -- the recursion of E3.model_nbody (nothing on a single molecule)."""
    idx = {c: j for j, c in enumerate(subs)}
    out = np.zeros((len(subs), len(subs)))
    for c in subs:
        for k in range(2, len(c) + 1):
            for d in itertools.combinations(c, k):
                out[idx[c], idx[d]] = (-1.0) ** (len(c) - k)
    return out


def smallest_oo(P, subs):
    """(frames, S): every sub-cluster's closest O-O distance (A)."""
    O = P[:, 0::3]
    d = np.linalg.norm(O[:, :, None] - O[:, None, :], axis=-1)
    return np.stack([np.min([d[:, i, j] for i, j in itertools.combinations(c, 2)], axis=0) for c in subs], axis=1)


# --------------------------------------------------------------------------------------- the components ----------
def field_tensors(rung, b):
    """the pinned sources' potential, field and field gradient at every atom with every molecule's share apart, as
    model._total_energy builds them for the field term (whole molecules)."""
    pos, mask = b["positions"], b["mask"]
    fm = b["frag_mask"].astype(pos.dtype)
    present = (fm @ mask.astype(pos.dtype)) > 0.5
    vs_on = b["vs_mask"] & ((present.astype(pos.dtype) @ b["vs_frag_mask"].astype(pos.dtype)) > 0.5)
    X_src = jnp.concatenate([pos, b["vs_w"] @ pos], axis=0)
    q_src = jnp.concatenate([b["pin_q_site"], b["vs_q"]])
    on_src = jnp.concatenate([mask, vs_on])
    frag_src = jnp.concatenate([b["frag_mask"], b["vs_frag_mask"]], axis=1)
    same_src = (fm.T @ frag_src.astype(pos.dtype)) > 0.5
    src = mask[:, None] & on_src[None, :] & ~same_src
    phi, E, G, parts = BF.local_field_tensors(pos, X_src, q_src, src, rung.l1, rung.e0_field_on, rung.e0_field_off,
                                              frag_src=frag_src, per_frag=True,
                                              self_taper=(rung.e0_field_self_on, rung.e0_field_self_off)
                                              if rung.e0_field_self_off > 0 else None)
    if len(parts) == 4:
        parts = parts + (parts[3],)                     # the self terms at the term's reach (v0.0.17's w_self = reach)
    return phi, E, G, parts


def field_energies(p, rung, b, tensors, s, v, t):
    """the field term summed over the atoms with E_0's features (s, v, t), as (its self terms, its cross terms) in the
    rung's own form: the MLP or degree 2 with its self terms (h(x_i) - sum_B h(x_i(B)) the cross terms), or the cross
    terms alone (Rung.e0_field_cross); v0.0.17: the MLP's self terms at their own reach, sum_B w_iB h(x_i(B)), beside the
    MLP's cross terms (Rung.e0_field_self_on / _off) or degree-2 cross terms by species (Rung.e0_field_cross_species)."""
    phi, Eloc, Gloc, parts = tensors
    e_scale, unit, mask = p["heads"]["e_scale"], rung.e0_field_unit, b["mask"]
    if rung.e0_field_form == "quadratic":
        f = lambda prt: jnp.sum(H.field_energy2(p["field"], s, v, t, Eloc, Gloc, mask, e_scale, unit, parts=prt))
    else:
        f = lambda prt: jnp.sum(H.field_energy(p["field"], s, v, t, phi, Eloc, Gloc, mask, e_scale, unit, parts=prt))
    if rung.e0_field_self_off > 0 or rung.e0_field_cross_species:
        self_ = jnp.sum(H.field_energy_split(p["field"], s, v, t, phi, Eloc, Gloc, mask, e_scale, unit, parts,
                                             mlp_cross=False))
        if rung.e0_field_cross_species:
            cross = jnp.sum(H.field_cross_species(p["field_x"], b["numbers"], Eloc, Gloc, mask, e_scale, unit, parts))
        else:
            cross = f(parts[:4])
        return self_, cross
    cross = f(parts[:4])
    return (jnp.zeros(()) if rung.e0_field_cross else f(None) - cross), cross


def components_single(p, rung, b):
    """one structure's interaction energy (hartree) by component, as model.energy_single: E(all) - sum_f E(f alone).
    For a field rung with whole molecules also the field term read here as its self terms (F_self) and its cross terms
    (F_cross) -- their sum to check against the model's -- and both at the isolated molecule's features (F_self1,
    F_cross1); on a molecule alone the term is zero, so these need the whole structure only."""
    groups = b["frag_mask"]
    E_all, a_all = M._total_energy(p, rung, b, b["mask"], subtract_atoms=False)
    E_fr, a_fr = jax.vmap(lambda fm: M._total_energy(p, rung, b, fm, subtract_atoms=False))(groups)
    valid = jnp.any(groups, axis=1)
    less = lambda x_all, x_fr: x_all - jnp.sum(jnp.where(valid, x_fr, 0.0))
    E_self = jnp.sum(b["pin_eself"] * b["mask"])
    out = {"E": less(E_all, E_fr) + E_self, "E_self": E_self}
    for k in ("E0", "E_coul", "E_disp", "E_disp8", "E_A", "E_G", "E_bf", "E_field"):
        if k in a_all:
            out[k] = less(a_all[k], a_fr[k])
    if rung.e0_field and rung.e0_field_whole:
        apply = lambda mask: E0.apply_e0(p["e0"], rung.e0, b["positions"], b["numbers"], mask, b["cell"], b["pbc"])[:3]
        s, v, t = apply(b["mask"])
        s1, v1, t1 = (jnp.sum(x, axis=0) for x in jax.vmap(apply)(groups))     # every atom with its own molecule alone
        tensors = field_tensors(rung, b)
        out["F_self"], out["F_cross"] = field_energies(p, rung, b, tensors, s, v, t)
        out["F_self1"], out["F_cross1"] = field_energies(p, rung, b, tensors, s1, v1, t1)
    return out


_FNS = {}


def component_fn(rung):
    """the batched components, compiled once per rung and shape; the parameters an argument (reused across seeds)."""
    key = repr(rung)
    if key not in _FNS:
        _FNS[key] = jax.jit(lambda p, bb: jax.vmap(lambda b: components_single(p, rung, b))(bb))
    return _FNS[key]


def batches_for(structs, n_freq, chunk=256):
    """padded batches grouped by atom count, one shape per group (the last chunk padded with repeats of its first
    structure): [(indices, batch)]."""
    groups = {}
    for i, s in enumerate(structs):
        groups.setdefault(s.n_atoms, []).append(i)
    out = []
    for idx in groups.values():
        size = min(chunk, len(idx))
        for i0 in range(0, len(idx), size):
            sel = idx[i0:i0 + size]
            batch = [structs[i] for i in sel]
            batch += [batch[0]] * (size - len(batch))
            out.append((np.array(sel), pad_batch(batch, n_freq=n_freq)))
    return out


def evaluate(params, rung, batches, n):
    """every structure's components (hartree): {key: (n,)}."""
    fn = component_fn(rung)
    out = {}
    for sel, b in batches:
        r = fn(params, b)
        for k, v in r.items():
            out.setdefault(k, np.zeros(n))[sel] = np.asarray(v)[:len(sel)]
    return out


def quantities(c, rung):
    """the components of one evaluation as the readings' quantities (kcal/mol, per structure), with the field's splits
    and the checks' residuals."""
    q = {"E": c["E"] * K, "E0": c["E0"] * K,
         "bands": (c.get("E_coul", 0.0) + c.get("E_disp", 0.0) + c.get("E_disp8", 0.0) + c["E_self"]) * K}
    rest = [k for k in ("E_A", "E_G") if k in c]
    if rest:
        q["other"] = sum(c[k] for k in rest) * K
    if "E_bf" in c:
        q["readout"] = c["E_bf"] * K
    if "E_field" in c:
        q["field"] = c["E_field"] * K
    if "F_cross" in c:
        f = q["field"]
        cross, cross1, self1 = c["F_cross"] * K, c["F_cross1"] * K, c["F_self1"] * K
        if rung.e0_field_cross:
            q.update(cross=f, cross_fixed=cross1, cross_coef=f - cross1, field_fixed=cross1, field_coef=f - cross1)
        else:
            q.update(self=f - cross, cross=cross, self_fixed=self1, cross_fixed=cross1, self_coef=(f - cross) - self1,
                     cross_coef=cross - cross1, field_fixed=self1 + cross1, field_coef=f - (self1 + cross1))
        q["_check_field"] = (c["F_self"] + c["F_cross"] - c["E_field"]) * K
    q["_check_sum"] = q["E"] - sum(q[k] for k in ("E0", "bands", "other", "readout", "field") if k in q)
    return q


# ------------------------------------------------------------------------------------------- the sets -------------
def build_structures(rung, P, n, truth, rigid):
    """every frame's sub-clusters as structures with the rung's pins (E3.model_nbody's): the rigid monomer's charges
    (rigid) or each molecule's at its own geometry."""
    subs = subsets_of(n)
    atoms = {c: np.concatenate([np.arange(3 * i, 3 * i + 3) for i in c]) for c in subs}
    if rigid:
        qr = W.rigid_molecule_charges(truth, n)
        out = [W.water_structure(p[atoms[c]], charges=qr[atoms[c]]) for p in P for c in subs]
    else:
        out = [W.water_structure(p[atoms[c]], charges=truth.molecule_charges(p[atoms[c]])) for p in P for c in subs]
    E3.pin_for(rung, out, truth, rigid=rigid)
    return out


def load_sets(args, truth, cached, frames):
    """the frames of every set: positions (A), molecules, the pins' kind, the reference (the truth's terms of every
    sub-cluster for the hexamers, the labels of the raw sets' top order) and the closest O-O of every sub-cluster."""
    sets = {}
    keys = args.sets.split(",")
    if "hex" in keys:
        hexamers = cached["tests"][6][:args.max_hex or None]
        P = np.array([bohr_to_ang(np.asarray(s.positions)) for s in hexamers])
        subs = subsets_of(6)
        atoms = {c: np.concatenate([np.arange(3 * i, 3 * i + 3) for i in c]) for c in subs}
        t_sub = np.array([[truth.qa.interaction_energy(ang_to_bohr(p[atoms[c]])) for c in subs] for p in P])
        terms = t_sub @ mobius(subs).T
        sets["hex"] = dict(n=6, P=P, rigid=False, idx=np.arange(len(P)), ref_terms=terms * K,
                           soo=smallest_oo(P, subs), cached_energy=np.array([s.energy for s in hexamers]) * K,
                           truth_total=t_sub[:, -1] * K)
        print(f"hexamers: {len(P)}, the truth's terms of their {len(subs)} sub-clusters each", flush=True)
    raw = [k for k in keys if k in ("2b", "3b", "4b")]
    if raw:
        if frames is not None:
            data = {k: dict(n=int(k[0]), pos=frames[f"{k}__P"], label=frames[f"{k}__label"], soo=frames[f"{k}__soo"],
                            roo=frames[f"{k}__roo"]) for k in raw if f"{k}__P" in frames}
            print(f"the raw sets' frames from {args.frames_path}: " + ", ".join(f"{k} {len(d['pos'])}" for k, d in data.items()),
                  flush=True)
        else:
            a = type("A", (), dict(heldout=",".join(raw), heldout_oh_tol=0.03, heldout_angle_tol=5.0,
                                   heldout_strict_oh_tol=0.02, heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0,
                                   heldout_max=0))()
            data = E3.prepare_heldout(a, truth)
            print("the raw sets read again from the build (no frames file): " + ", ".join(f"{k} {len(d['pos'])}"
                                                                                          for k, d in data.items()), flush=True)
        for k in raw:
            if k not in data:
                print(f"  {k}: not among the frames; skipped", flush=True)
                continue
            d = data[k]
            idx = np.arange(len(d["pos"]))
            if args.max_frames and len(idx) > args.max_frames:
                idx = np.sort(np.random.default_rng(0).choice(idx, args.max_frames, replace=False))
            n = d["n"]
            sets[k] = dict(n=n, P=np.asarray(d["pos"])[idx], rigid=True, idx=idx, label=np.asarray(d["label"])[idx] * K,
                           soo_frame=np.asarray(d["soo"])[idx], roo_frame=np.asarray(d["roo"])[idx],
                           soo=smallest_oo(np.asarray(d["pos"])[idx], subsets_of(n)))
    return sets


def units_of(key, s):
    """the units read on a set: [(name, select, bins)], select mapping a quantity's terms (frames, S) to the values
    (U,), the bins {label: (U,) mask}."""
    n = s["n"]; subs = subsets_of(n)
    size = np.array([len(c) for c in subs])
    contact = lambda soo: {"all": np.ones(len(soo), bool),
                           **{f"closest O-O {lo:g}-{hi:g}": (soo >= lo) & (soo < hi) for lo, hi in CONTACT_BINS}}
    if key == "hex":
        out = []
        for k in range(2, n + 1):
            out.append((f"{k}-body", lambda T, k=k: T[:, size == k].sum(axis=1), {"all": np.ones(len(s["P"]), bool)}))
        out.append(("non-additive (3-6)", lambda T: T[:, size >= 3].sum(axis=1), {"all": np.ones(len(s["P"]), bool)}))
        for k, what in ((3, "trimers"), (4, "tetramers")):
            out.append((f"their {what}' {k}-body ({math.comb(n, k)} each)", lambda T, k=k: T[:, size == k].reshape(-1),
                        contact(s["soo"][:, size == k].reshape(-1))))
        return out
    top = lambda T: T[:, -1]
    if key == "2b":
        soo = s["soo_frame"]
        bins = {"all": np.ones(len(soo), bool), **{f"O-O {lo:g}-{hi:g}": (soo >= lo) & (soo < hi) for lo, hi in DIMER_BINS}}
        return [("2-body", top, bins)]
    return [(f"{n}-body", top, contact(s["soo_frame"]))]


def reference_of(key, s, select):
    return select(s["ref_terms"]) if key == "hex" else s["label"]


# ------------------------------------------------------------------------------------------- the readings ---------
def model_reading(Q, err, mask):
    """the error's rms and every component's size and the error without it, on one bin."""
    m = mask
    out = {"error": rms(err[m]), "mean_error": float(np.mean(err[m])) if m.any() else float("nan")}
    for name, v in Q.items():
        if name in ("E",) or name.startswith("_"):
            continue
        out[name] = {"size": rms(v[m]), "without": rms((err - v)[m])}
    return out


def pair_reading(QX, QB, errX, errB, mask):
    """X against B on the same seed: every component's difference (rms), the error with that difference alone, and the
    correlation of the field's difference with E_0's."""
    m = mask
    out = {"error": rms(errX[m]), "base_error": rms(errB[m])}
    names = [k for k in QX if not k.startswith("_") and k != "E"]
    names += [k for k in QB if not k.startswith("_") and k != "E" and k not in names]
    for name in names:
        d = QX.get(name, 0.0) - QB.get(name, 0.0)
        d = np.broadcast_to(d, errX.shape)
        out[name] = {"difference": rms(d[m]), "only": rms((errB + d)[m])}
    if "field" in QX or "field" in QB:
        df = np.broadcast_to(QX.get("field", 0.0) - QB.get("field", 0.0), errX.shape)[m]
        de = (QX["E0"] - QB["E0"])[m]
        out["corr_field_E0"] = float(np.corrcoef(df, de)[0, 1]) if m.sum() > 2 and np.std(df) > 0 and np.std(de) > 0 else float("nan")
    return out


def seed_stats(vals):
    v = np.array([x for x in vals if x == x])
    return (float(v.mean()), float(v.min()), float(v.max())) if len(v) else (float("nan"),) * 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results"); ap.add_argument("root")
    ap.add_argument("--seeds", default="all", help="comma-separated seeds, or all")
    ap.add_argument("--rungs", default=None, help="comma-separated; omitted: the rungs of <results>/clusters.json")
    ap.add_argument("--sets", default="hex,4b,3b,2b")
    ap.add_argument("--pairs", default=None, help="X:B,... (default: every field rung against its rung without the term, "
                                                       "e.g. M_16C2pm, and against that rung's _w, e.g. M_16C2pm_w)")
    ap.add_argument("--max_frames", type=int, default=0, help="at most this many frames of each raw set (0: all)")
    ap.add_argument("--max_hex", type=int, default=0, help="at most this many hexamers (0: all)")
    ap.add_argument("--frames", default=None, help="the diagnostic's frames (default <results>/shortrange_diagnostic_frames.npz)")
    ap.add_argument("--l1", type=float, default=0.75); ap.add_argument("--response_edge", type=float, default=0.75)
    ap.add_argument("--F", type=int, default=32); ap.add_argument("--rcut", type=float, default=6.0)
    ap.add_argument("--lA", type=float, default=4.0)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--out", default="decomposition.json")
    args = ap.parse_args()
    t0 = time.time()
    from e3qp3_shortrange_diagnostic import check_record
    truth = W.qaqua_pol_truth(args.root)
    run = json.load(open(os.path.join(args.results, "clusters.json")))
    run_rungs = [k for k, v in run.items() if isinstance(v, dict) and "clusters_by_size" in v]
    rungs = (args.rungs or ",".join(run_rungs)).split(",")
    args.frames_path = args.frames or os.path.join(args.results, "shortrange_diagnostic_frames.npz")
    frames = dict(np.load(args.frames_path)) if os.path.exists(args.frames_path) else None
    with open(os.path.join(args.results, "clusters_cache.pkl"), "rb") as fh:
        cached = pickle.load(fh)
    a = type("A", (), dict(lA=args.lA, rcut=args.rcut, F=args.F, l1=args.l1, rs_over_l1=0.0,
                           response_edge=args.response_edge, charges=None, charge_neutrality="fragment"))()
    configured = {}
    for name in rungs:
        configured[name] = E3.configure(E3.RUNGS[name], a)
        check_record(name, configured[name], (run.get(name) or {}).get("model"))
    sets = load_sets(args, truth, cached, frames)
    print(f"rungs: {', '.join(rungs)}; sets: {', '.join(sets)}  [{time.time() - t0:.0f} s]", flush=True)
    checks = {}
    if "hex" in sets:                                              # the truth's terms against what the run read
        s = sets["hex"]
        checks["truth"] = {"hexamer energies": float(np.max(np.abs(s["truth_total"] - s["cached_energy"])))}
        if frames is not None and "hex__truth_2b" in frames:
            t2 = s["ref_terms"][:, [len(c) == 2 for c in subsets_of(6)]].sum(axis=1)
            checks["truth"]["hexamer 2-body (the diagnostic's)"] = float(np.max(np.abs(t2 - frames["hex__truth_2b"][s["idx"]] * K)))
        print("the truth's hexamer terms against the cached energies and the diagnostic's 2-body: "
              + ", ".join(f"{k} {v:.1e}" for k, v in checks["truth"].items()) + " kcal/mol", flush=True)

    # every model's quantities, every set's terms
    terms = {}                                                     # (name, seed) -> set -> quantity -> (frames, S) kcal/mol
    built = {}
    mob = {k: mobius(subsets_of(s["n"])) for k, s in sets.items()}
    for name in rungs:
        rung = configured[name]
        if args.seeds == "all":
            seeds = sorted(int(re.search(r"_seed(\d+)\.params\.pkl$", p).group(1))
                           for p in glob.glob(os.path.join(args.results, f"clusters_{name}_seed*.params.pkl")))
        else:
            seeds = [int(x) for x in args.seeds.split(",")]
        for seed in seeds:
            path = os.path.join(args.results, f"clusters_{name}_seed{seed}.params.pkl")
            if not os.path.exists(path):
                print(f"  {name} seed {seed}: no {path}", flush=True); continue
            params = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(path, "rb")))
            terms[(name, seed)] = {}
            chk = {}
            for key, s in sets.items():
                pin_key = (key, bool(rung.pin_mu), bool(rung.pin_alpha), bool(rung.msites), rung.K, args.chunk)
                if pin_key not in built:
                    built[pin_key] = batches_for(build_structures(rung, s["P"], s["n"], truth, s["rigid"]), rung.K, args.chunk)
                n_sub = len(subsets_of(s["n"]))
                c = evaluate(params, rung, built[pin_key], len(s["P"]) * n_sub)
                Q = {k: v.reshape(len(s["P"]), n_sub) for k, v in quantities(c, rung).items()}
                T = {k: v @ mob[key].T for k, v in Q.items() if not k.startswith("_")}
                terms[(name, seed)][key] = T
                size = np.array([len(c_) for c_ in subsets_of(s["n"])])
                chk[f"{key}: components sum"] = float(np.max(np.abs(Q["_check_sum"])))
                if "_check_field" in Q:
                    chk[f"{key}: field term as read here"] = float(np.max(np.abs(Q["_check_field"])))
                if s["n"] >= 3:
                    chk[f"{key}: bands' 3-body and beyond"] = float(np.max(np.abs(T["bands"][:, size >= 3])))
                    if "self_fixed" in T:
                        chk[f"{key}: self terms at fixed coefficients, 3-body and beyond"] = float(np.max(np.abs(T["self_fixed"][:, size >= 3])))
                degree2 = rung.e0_field_form == "quadratic" or rung.e0_field_cross_species
                if s["n"] >= 4 and "cross_fixed" in T and degree2:
                    chk[f"{key}: degree-2 cross terms at fixed coefficients, 4-body and beyond"] = float(np.max(np.abs(T["cross_fixed"][:, size >= 4])))
                if "cross_coef" in T and rung.e0_field_cross_species:
                    chk[f"{key}: the species cross terms' coefficient part"] = float(np.max(np.abs(T["cross_coef"])))
                if frames is not None:
                    if key == "hex" and f"{name}__s{seed}__hex__E" in frames:
                        chk["hex: energies (the diagnostic's)"] = float(np.max(np.abs(Q["E"][:, -1] - frames[f"{name}__s{seed}__hex__E"][s["idx"]] * K)))
                        chk["hex: 2-body (the diagnostic's)"] = float(np.max(np.abs(T["E"][:, size == 2].sum(axis=1)
                                                                                   - frames[f"{name}__s{seed}__hex__2b"][s["idx"]] * K)))
                    elif f"{name}__s{seed}__{key}__fix" in frames:
                        chk[f"{key}: top order (the diagnostic's fixed)"] = float(np.max(np.abs(T["E"][:, -1] - frames[f"{name}__s{seed}__{key}__fix"][s["idx"]] * K)))
            checks[f"{name} seed {seed}"] = chk
            bad = [k for k, v in chk.items() if not (v <= (TOL_RUN if "diagnostic" in k else TOL))]
            print(f"  {name} seed {seed}: checks (max |difference|, kcal/mol) " + ", ".join(f"{k} {v:.1e}" for k, v in chk.items())
                  + (f"  ** CHECK FAILED: {'; '.join(bad)} **" if bad else "") + f"  [{time.time() - t0:.0f} s]", flush=True)

    # the readings
    models = sorted(terms)
    seeds_of = {name: [sd for nm, sd in models if nm == name] for name in rungs}
    if args.pairs:
        pairs = [tuple(p.split(":")) for p in args.pairs.split(",")]
    else:                                  # every field rung against its rung without the term, and against its _w rung
        family = lambda x: "_".join(x.split("_")[:2])
        pairs = [(x, family(x)) for x in rungs if configured[x].e0_field and family(x) in rungs]
        pairs += [(x, family(x) + "_w") for x in rungs if configured[x].e0_field and x != family(x) + "_w"
                  and family(x) + "_w" in rungs]
    report = {"checks": checks, "sets": {}, "rungs": rungs, "pairs": [list(p) for p in pairs]}
    order = ("E0", "bands", "other", "readout", "field")
    label = {"E0": "E0", "bands": "bands", "other": "other", "readout": "read-out", "field": "field"}
    for key, s in sets.items():
        report["sets"][key] = {}
        for unit, select, bins in units_of(key, s):
            ref = reference_of(key, s, select)
            rep_u = report["sets"][key][unit] = {}
            what = "the truth's" if key == "hex" else "the labels'"
            print(f"\n{key} -- {unit}: {len(ref)} values; {what} rms "
                  + ", ".join(f"{b} {rms(ref[m]):.4f} (n {int(m.sum())})" for b, m in bins.items() if m.any()) + " kcal/mol", flush=True)
            print("  size / without (kcal/mol, seed means): the component's part, and the error with it taken out", flush=True)
            for b, m in bins.items():
                if not m.any():
                    continue
                rep_b = rep_u[b] = {"n": int(m.sum()), "reference_rms": rms(ref[m]), "models": {}, "pairs": {}}
                print(f"  [{b}, n {int(m.sum())}]", flush=True)
                for name in rungs:
                    rows = []
                    for seed in seeds_of[name]:
                        Tm = terms[(name, seed)][key]
                        Qu = {k: select(v) for k, v in Tm.items()}
                        rows.append(model_reading(Qu, Qu["E"] - ref, m))
                    if not rows:
                        continue
                    rep_b["models"][name] = {"seeds": seeds_of[name], "per_seed": rows}
                    e = seed_stats([r["error"] for r in rows])
                    pct = f" ({100 * e[0] / rep_b['reference_rms']:.1f} %)" if rep_b["reference_rms"] > 0 else ""
                    line = f"    {name:<15} error {e[0]:.4f}{pct} [{e[1]:.4f} {e[2]:.4f}]"
                    for comp in order:
                        if comp in rows[0] and (comp != "bands" or unit == "2-body"):        # pairwise: nothing beyond
                            sz, wo = (seed_stats([r[comp][x] for r in rows])[0] for x in ("size", "without"))
                            line += f"  {label[comp]} {sz:.4f} / {wo:.4f}"
                    print(line, flush=True)
                    if "field_fixed" in rows[0]:
                        sub = []
                        for part in ("self", "cross"):
                            if part in rows[0]:
                                sz, wo = (seed_stats([r[part][x] for r in rows])[0] for x in ("size", "without"))
                                fx, cf = (seed_stats([r[f"{part}_{x}"]["size"] for r in rows])[0] for x in ("fixed", "coef"))
                                sub.append(f"{part} {sz:.4f} / {wo:.4f} (fixed {fx:.4f}, coef {cf:.4f})")
                        print(f"    {'':<15} the field term: " + "; ".join(sub), flush=True)
                for X, B in pairs:
                    common = [sd for sd in seeds_of.get(X, []) if sd in seeds_of.get(B, [])]
                    if not common:
                        continue
                    rows = []
                    for seed in common:
                        TX, TB = terms[(X, seed)][key], terms[(B, seed)][key]
                        QX = {k: select(v) for k, v in TX.items()}; QB = {k: select(v) for k, v in TB.items()}
                        rows.append(pair_reading(QX, QB, QX["E"] - ref, QB["E"] - ref, m))
                    rep_b["pairs"][f"{X} - {B}"] = {"seeds": common, "per_seed": rows}
                    eX, eB = (seed_stats([r[x] for r in rows])[0] for x in ("error", "base_error"))
                    rel = seed_stats([100 * (r["error"] / r["base_error"] - 1) for r in rows])
                    line = f"    {X} - {B}: error {eX:.4f} against {eB:.4f} (paired {rel[0]:+.0f} % [{rel[1]:+.0f} {rel[2]:+.0f}]); difference / only:"
                    for comp in ("E0", "readout", "field", "self", "cross"):
                        if comp in rows[0]:
                            d, o = (seed_stats([r[comp][x] for r in rows])[0] for x in ("difference", "only"))
                            line += f"  {label.get(comp, comp)} {d:.4f} / {o:.4f}"
                    if "corr_field_E0" in rows[0]:
                        cf = seed_stats([r["corr_field_E0"] for r in rows])[0]
                        line += "; corr(field's, E0's difference) " + (f"{cf:+.2f}" if cf == cf else "n/a")
                    print(line, flush=True)
    with open(os.path.join(args.results, args.out), "w") as fh:
        json.dump(report, fh, indent=1)
    npz = os.path.join(args.results, os.path.splitext(args.out)[0] + "_frames.npz")
    out = {}
    for key, s in sets.items():
        out[f"{key}__P"] = s["P"]; out[f"{key}__index"] = s["idx"]
        out[f"{key}__subsets"] = np.array([[*c, *([-1] * (s["n"] - len(c)))] for c in subsets_of(s["n"])])
        out[f"{key}__reference"] = s["ref_terms"] if key == "hex" else s["label"]
    for (name, seed), T in terms.items():
        for key, Tk in T.items():
            for q, v in Tk.items():
                out[f"{name}__s{seed}__{key}__{q}"] = v if key == "hex" else v[:, -1]
    np.savez_compressed(npz, **out)
    print(f"\nevery frame's terms (kcal/mol; the raw sets' top order, the hexamers' every sub-cluster): {npz}")
    print(f"done [{time.time() - t0:.0f} s]; {os.path.join(args.results, args.out)}")


if __name__ == "__main__":
    main()
