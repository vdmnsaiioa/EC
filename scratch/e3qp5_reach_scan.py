"""
E3-QP5's reading of its field term's taper (notes/e3qp5-results.md section 3.5), from the PES and the data alone.

E3-QP5's pre-registration (section 3c, scratch/e3qp5_preregistration.py) fitted one effective polarisability per species
on the field input's first-order cross term over E3-QP4's 4673 raw CCSD(T) trimers (the rigid monomer's charges, as the
models read them),

    label_3b ~ -sum_i a_Z(i) E_i(B) . E_i(C)        (every atom i of A, B and C the other two; and cyclically),

every source's share of E_i tapered by its own distance to i between r_c - 1.5 and r_c - 0.5 A (4.5-5.5 A), as
bandfields.local_field_tensors reads it: 0.114 of the 0.623 kcal/mol rms left.  Here the same two-number fit, refitted
for every weighting, by the smallest and the largest O-O:

  (1) the per-source taper at other reaches, and none;
  (2) every source molecule whole -- one weight for all of its sources (its two H atoms and its M-site), so that no
      molecule enters part charged: the product of its sources' tapers (no source beyond the taper's end in reach, as
      E3-QP5's), a taper on the distance from the atom to the molecule's O, a taper on the two molecules' O-O; and,
      on the cross term rather than the field, a taper on all three O-O of the triple (its three cross terms -- one at
      every molecule -- kept or dropped together);
  (3) what the per-source taper does: how many (atom, other molecule) pairs straddle it and the net charge it leaves
      such a molecule; the fitted 3-body where the farthest molecule is 5-7, 7-9 and beyond 9 A from another, against
      the labels; how often the partner straddles it in the raw dimers, by O-O; and how often a molecule straddles it
      in the training clusters and the test hexamers (rebuilt bit for bit, as E3-QP5's run built them).

    python scratch/e3qp5_reach_scan.py [build_pol]          # ~1.5 min, one CPU
"""
import math, os, sys, time
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "external", "qaqua", "build_pol")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from scipy.special import erf
from sseft import water as W
from sseft.units import ang_to_bohr, bohr_to_ang
import e3_water as E3

K = 627.5094740631
L1 = ang_to_bohr(0.75)
t0 = time.time()
truth = W.qaqua_pol_truth(ROOT)
q_r, w_r, vq_r, _ = truth.site_model(W.monomer_positions())
qs = np.array([q_r[0], q_r[1], q_r[2], vq_r[0]])                     # O, H, H, M-site (the O carries none)
rms = lambda v: float(np.sqrt(np.mean(np.square(v))))


def taper(x, on, off):
    """bandfields.taper, numpy: 1 below on, 0 beyond off, a quintic C^2 smoothstep between (x, on, off in one unit)"""
    u = np.clip((x - on) / (off - on), 0.0, 1.0)
    return 1.0 - u ** 3 * (10.0 - 15.0 * u + 6.0 * u ** 2)


def heldout(kind):
    a = type("A", (), dict(heldout=kind, heldout_oh_tol=0.03, heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02,
                           heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0, heldout_max=0))()
    return E3.prepare_heldout(a, truth)[kind]


def geometry(pos_ang, n):
    """bohr positions (F, n molecules, 3 atoms, 3), the sources with the M-site (F, n, 4, 3), and per atom and source the
    displacement (F, 3n, n, 4, 3) and the distance (F, 3n, n, 4); an atom as its own source at distance 1 (unused)"""
    P = ang_to_bohr(np.asarray(pos_ang, float)).reshape(-1, n, 3, 3)
    S = np.concatenate([P, np.einsum("a,fmax->fmx", w_r[0][:3], P)[:, :, None]], axis=2)
    d = P.reshape(len(P), 3 * n, 1, 1, 3) - S[:, None]
    r = np.linalg.norm(d, axis=-1)
    return P, d, np.where(r > 0, r, 1.0)


# --- the raw trimers ---------------------------------------------------------------------------------------------------
d3 = heldout("3b")
lab = np.asarray(d3["label"]) * K; soo = np.asarray(d3["soo"])
P, d, r = geometry(d3["pos"], 3)
F = len(P)
Ob = P[:, :, 0]                                                       # the O atoms (F, 3, 3), bohr
oo = np.linalg.norm(Ob[:, :, None] - Ob[:, None], axis=-1)            # (F, 3, 3)
loo = bohr_to_ang(np.max(oo.reshape(F, 9), axis=1))
dg = (2.0 / (L1 * math.sqrt(math.pi))) * np.exp(-(r / L1) ** 2) / r - erf(r / L1) / r ** 2
with np.errstate(divide="ignore", invalid="ignore"):
    E_src = -(qs * dg)[..., None] * d / r[..., None]                  # every source's field at every atom (F, 9, 3, 4, 3)
mol_of = np.repeat(np.arange(3), 3)
print(f"{F} raw trimers  [{time.time() - t0:.0f} s]", flush=True)


def molecule_terms(w_src, coef=None):
    """with every source's share weighted by w_src (F, 9, 3, 4): per frame and molecule m, sum over m's O atom and over
    its H atoms of E_i(Y) . E_i(Z), Y and Z the other two molecules (F, 3 molecules, 2 species; a.u.^2); with coef, the
    fitted 3-body at m's atoms, -sum a_Z(i) E_i(Y) . E_i(Z) (F, 3; kcal/mol)"""
    E = np.einsum("fiys,fiysx->fiyx", w_src, np.nan_to_num(E_src))
    c = np.zeros((F, 3, 2))
    for m in range(3):
        y, z = [k for k in range(3) if k != m]
        e = np.sum(E[:, 3 * m:3 * m + 3, y] * E[:, 3 * m:3 * m + 3, z], axis=-1)
        c[:, m, 0] = e[:, 0]; c[:, m, 1] = e[:, 1] + e[:, 2]
    return c if coef is None else -K * (c @ np.asarray(coef))


S_BINS = ((0.0, 2.7), (2.7, 3.0), (3.0, 99.0))
L_BINS = ((0.0, 5.0), (5.0, 7.0), (7.0, 9.0), (9.0, 99.0))
cols = [f"S{lo:g}-{hi:g}" if hi < 99 else f"S{lo:g}-" for lo, hi in S_BINS] + \
       [f"L{lo:g}-{hi:g}" if hi < 99 else f"L{lo:g}-" for lo, hi in L_BINS]
masks = [(soo >= lo) & (soo < hi) for lo, hi in S_BINS] + [(loo >= lo) & (loo < hi) for lo, hi in L_BINS]
fits = {}


def fit(name, w_src, w_frame=None):
    """refit the two numbers with every source's share of the field weighted by w_src (F, 9, 3, 4), or, with w_frame (F,),
    every cross term of the frame weighted by it (a triple's three cross terms kept or dropped together)"""
    X = molecule_terms(w_src).sum(axis=1)                             # (F, 2)
    if w_frame is not None:
        X = X * w_frame[:, None]
    coef, *_ = np.linalg.lstsq(-X * K, lab, rcond=None)
    res = lab + (X * K) @ coef
    fits[name] = dict(coef=coef, w=w_src)
    print(f"{name:>26}{rms(res):8.4f}" + "".join(f"{rms(res[m]):9.4f}" for m in masks) + f"   {coef[0]:.2f}, {coef[1]:.2f}",
          flush=True)


print("(1, 2) the two-number fit's residual (kcal/mol rms), by the smallest (S) and the largest (L) O-O (A); a_O, a_H (bohr^3)")
print(f"{'':>26}{'all':>8}" + "".join(f"{c:>9}" for c in cols))
print(f"{'the labels':>26}{rms(lab):8.4f}" + "".join(f"{rms(lab[m]):9.4f}" for m in masks))
print(f"{'n':>26}{F:8d}" + "".join(f"{int(m.sum()):9d}" for m in masks))
A = ang_to_bohr
fit("no taper", np.ones_like(r))
for on, off in ((4.5, 5.5), (4.0, 5.0), (3.5, 4.5), (3.0, 4.0)):
    fit(f"per source {on:g}-{off:g}" + (" (E3-QP5)" if on == 4.5 else ""), taper(r, A(on), A(off)))
for on, off in ((4.5, 5.5), (5.0, 6.0)):
    fit(f"whole: product {on:g}-{off:g}", np.broadcast_to(np.prod(taper(r, A(on), A(off)), axis=-1)[..., None], r.shape))
r_iO = np.linalg.norm(P.reshape(F, 9, 1, 3) - Ob[:, None], axis=-1)   # (F, 9 atoms, 3 molecules)
for on, off in ((4.0, 5.0), (4.5, 5.5)):
    fit(f"whole: atom-O {on:g}-{off:g}", np.broadcast_to(taper(r_iO, A(on), A(off))[..., None], r.shape))
for on, off in ((5.0, 6.0), (5.5, 6.5)):
    fit(f"whole: O-O pair {on:g}-{off:g}", np.broadcast_to(taper(oo[:, mol_of], A(on), A(off))[..., None], r.shape))
for on, off in ((5.0, 6.0), (5.5, 6.5)):
    fit(f"whole: O-O triple {on:g}-{off:g}", np.ones_like(r), np.prod(taper(oo[:, [0, 0, 1], [1, 2, 2]], A(on), A(off)), axis=1))

# (3) what the per-source taper does
w_ps = taper(r, A(4.5), A(5.5))
own = mol_of[:, None] == np.arange(3)[None, :]                         # (9, 3): the atom's own molecule
part = (w_ps.min(axis=-1) < 1.0) & (w_ps.max(axis=-1) > 0.0) & ~own    # a molecule straddling the taper, seen from an atom
qnet = np.einsum("fiys,s->fiy", w_ps, qs)
print(f"(3) the per-source taper 4.5-5.5 A in the raw trimers: {100 * part.sum() / (F * (~own).sum()):.0f} % of the (atom, "
      f"other molecule) pairs straddle it; such a molecule enters with a net charge rms {rms(qnet[part]):.3f} e, largest "
      f"{np.abs(qnet[part]).max():.3f} e (its M-site {vq_r[0]:+.3f}, each H {q_r[1]:+.3f})", flush=True)
for lo, hi in ((5.0, 7.0), (7.0, 9.0), (9.0, 99.0)):
    mm = (loo >= lo) & (loo < hi)
    s = f"    largest O-O {lo:g}-{hi:g} A (n {int(mm.sum())}): the labels' 3-body rms {rms(lab[mm]):.4f}; the fitted 3-body"
    for name in ("no taper", "per source 4.5-5.5 (E3-QP5)", "whole: product 5-6"):
        tot = molecule_terms(fits[name]["w"], fits[name]["coef"])[mm].sum(axis=1)
        s += f"; {name} rms {rms(tot):.4f}, correlation {np.corrcoef(tot, lab[mm])[0, 1]:+.2f}"
    print(s, flush=True)
d2 = heldout("2b")
P2, _, r2 = geometry(d2["pos"], 2)
w2 = taper(r2, A(4.5), A(5.5))
other = np.repeat(np.arange(2), 3)[:, None] != np.arange(2)[None, :]   # (6, 2): the atom's partner
oo2 = bohr_to_ang(np.linalg.norm(P2[:, 0, 0] - P2[:, 1, 0], axis=-1))
st2 = ((w2.min(axis=-1) < 1.0) & (w2.max(axis=-1) > 0.0))[:, other]   # (F, 6)
q2 = np.einsum("fiys,s->fiy", w2, qs)[:, other]
print(f"    the {len(P2)} raw dimers, by O-O: the atoms whose partner straddles the taper (%); the partner's net charge as seen "
      f"(rms over the atoms, e)", flush=True)
for lo, hi in ((2.3, 2.7), (2.7, 2.9), (2.9, 3.0), (3.0, 3.2), (3.2, 5.0), (5.0, 7.0), (7.0, 99.0)):
    mm = (oo2 >= lo) & (oo2 < hi)
    print(f"      {lo:g}-{hi:g} A (n {int(mm.sum())}): {100 * st2[mm].mean():.1f} %, {rms(q2[mm]):.3f} e", flush=True)
geo = dict(mc_steps=300, T_K=300.0, radius_ang=None, min_OO=2.7, min_any=1.7)     # configs/water_qaqua_pol5.yaml


def cluster_straddle(structs, label):
    st, qn = [], []
    for c in structs:
        Pc = np.asarray(c.positions).reshape(-1, 3, 3)                 # bohr
        n = len(Pc)
        Sc = np.concatenate([Pc, np.einsum("a,max->mx", w_r[0][:3], Pc)[:, None]], axis=1)
        wc = taper(np.linalg.norm(Pc.reshape(3 * n, 1, 1, 3) - Sc[None], axis=-1), A(4.5), A(5.5))   # (3n, n, 4)
        ownc = np.repeat(np.arange(n), 3)[:, None] == np.arange(n)[None, :]
        sc = (wc.min(-1) < 1.0) & (wc.max(-1) > 0.0) & ~ownc
        st.append(sc[~ownc]); qn.append(np.einsum("iys,s->iy", wc, qs)[sc])
    st, qn = np.concatenate(st), np.concatenate(qn)
    print(f"    {label}: {100 * st.mean():.0f} % of the (atom, other molecule) pairs straddle it, net charge rms {rms(qn):.3f} e  "
          f"[{time.time() - t0:.0f} s]", flush=True)


print("    relaxed clusters, as E3-QP5's run built them:", flush=True)
cluster_straddle(W.cluster_dataset((6,), 24, seed=100, truth=truth, **geo), "the 24 test hexamers")
train = W.cluster_dataset((3, 4, 5), 100, seed=0, sigma=1e-9, truth=truth, **geo)
for n in (3, 4, 5):
    cluster_straddle([c for c in train if len(c.positions) == 3 * n], f"the 100 training clusters of {n} molecules")
print(f"done in {time.time() - t0:.0f} s")
