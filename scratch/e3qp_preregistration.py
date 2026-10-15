"""
The numbers of notes/e3qp-preregistration.md, computed from q-AQUA-pol before any training:

    python scratch/e3qp_preregistration.py [external/qaqua/build_pol]

(a) the dimer: R_e and D_e at phi = 0, the window, the identifying nodes at r_c = 5 and 6 A, the monomer's dipole-surface
    charges (the pinned charges) and TTM3-F's sites; the truth's tail at the reading rows split into TTM3-F's pieces
    (the site charges' electrostatics, the induction, the O-O C6 term) against the point charges; the daggers' exact
    tail errors, evaluated with the harness's own M_1p and M_16p (E_0 has nothing beyond r_c); the class minimum of
    M_1's charge scale on the identifying nodes under the training objective (as for E3-Q);
(c) the clusters of configs/water_qaqua_pol.yaml (same seeds): interaction and non-additive energies, the 3-, 4- and
    higher-body shares, TTM3-F's share of the non-additive energy, and the floor of a fixed-coefficient order-0
    read-out (the first-order cross terms sum_{j != k} E_ij . E_ik of the band field, l_1 = 0.75 A, of the other
    molecules' pinned charges, per species) on 200 more hexamers and on the test hexamers;
(e) the probe configurations of the config: the truth's probe interaction, its non-additive part and its induction by
    distance, and the same fixed-coefficient form on the probes' non-additive energy;
(d) the held-out raw CCSD(T) sets under the near-rigid filter and q-AQUA-pol's training domain: counts, label rms, the
    PES's own rmse, by largest O-O, and the fixed-coefficient form on the 3-body set by distance.
"""
import math, os, sys, time
import numpy as np
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
import jax
from scipy.special import erf
from sseft import water as W, qaqua as QA, data as D, model as M, train as T
from sseft.units import ang_to_bohr, bohr_to_ang
import e3_water as E3

K = 627.5094740631
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "external", "qaqua", "build_pol")
truth = W.qaqua_pol_truth(root)
rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
t_start = time.time()


def coulomb_band(pos_ang, q, l1_ang, frag):
    """the Coulomb band of pinned charges between different fragments, erf(r/l)/r, and its forces (hartree, hartree/bohr)."""
    x = ang_to_bohr(np.asarray(pos_ang, float)); l = ang_to_bohr(l1_ang)
    E = 0.0; F = np.zeros_like(x)
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if frag[i] == frag[j]:
                continue
            d = x[i] - x[j]; r = np.linalg.norm(d)
            g = erf(r / l) / r
            dg = (2 / math.sqrt(math.pi) * math.exp(-(r / l) ** 2) / l - g) / r     # dg/dr
            E += q[i] * q[j] * g
            F[i] -= q[i] * q[j] * dg * d / r; F[j] += q[i] * q[j] * dg * d / r
    return E, F


print(f"q-AQUA-pol at {truth.qa.root}: {truth.describe()}")
print("\n=== (a) the dimer at phi = 0 (rigid monomers r_OH 0.9572 A, HOH 104.52 deg)")
Re = truth.r_e; De = -truth.energy(W.dimer_positions(Re), 2)
lo, hi = 1.35 * Re, 1.35 * Re * 1.8
print(f"R_e = {Re:.5f} A, D_e = {De * K:.4f} kcal/mol; window [{lo:.4f}, {hi:.4f}] A")
R_train = bohr_to_ang(D.window_design(lo, hi, 40))
nodes = W.dimer_scan(R_train, truth=truth)
smallest = np.array([E3.min_intermolecular(s) for s in nodes])
for rc in (5.0, 6.0):
    m = smallest > rc
    print(f"  r_c = {rc}: {m.sum()} identifying nodes, R >= {R_train[m].min() if m.any() else float('nan'):.3f} A")
ident = [s for s, m in zip(nodes, smallest > 5.0) if m]
mono = W.monomer_positions()
q = truth.molecule_charges(mono)
mu = np.linalg.norm(W.molecular_dipoles(truth, mono)[0]) * 2.541746473
t1 = truth.qa.ttm(ang_to_bohr(mono))
offset = np.linalg.norm(bohr_to_ang(t1["msites"][0]) - mono[0])
print(f"dipole-surface charges at the rigid geometry: q (O, H, H) = {np.round(q, 7)}, dipole {mu:.4f} D; TTM3-F's sites: "
      f"q_H = {t1['site_charges'][0, 0]:+.5f}, q_M = {t1['site_charges'][0, 2]:+.5f} e, the M-site {offset:.4f} A from O on the "
      f"bisector; point-dipole coefficient {W.dipole_dipole_coefficient(truth):+.4f} hartree bohr^3")
alpha, c6 = truth.alpha, truth.c6
print(f"TTM3-F's polarisability {alpha:.4f} bohr^3 (1.444 A^3, on the M-site), O-O C6 {c6:.3f} hartree bohr^6")

# the tail: the truth against its pieces and against the point charges
R_eval = bohr_to_ang(D.eval_grid(hi, 500.0, 61))                  # the dimer part's evaluation grid and its reading rows
R_rows = np.array([R_eval[int(np.argmin(np.abs(R_eval - Rv)))] for Rv in (8, 10, 12, 15, 20, 30, 50, 100, 300)])
print("the reading rows (the evaluation grid's points nearest 8, 10, 12, 15, 20, 30, 50, 100, 300 A): "
      + ", ".join(f"{R:.3f}" for R in R_rows))
rows = W.dimer_scan(R_rows, truth=truth)
frag = np.repeat([0, 1], 3)
print(f"the tail ({'R':>6s} A: E_truth (hartree); its pieces as shares: site-charge electrostatics, induction, C6 + exponential; "
      "the point charges' Coulomb)")
for R, s in zip(R_rows, rows):
    t = truth.qa.ttm(s.positions)
    ec, _ = coulomb_band(bohr_to_ang(s.positions), s.pinned["q"], 1e-6, frag)
    print(f"  {R:7.3f}: {s.energy:+.6e}; elec {t['elec'] / s.energy:+.5f}, ind {t['ind'] / s.energy:+.5f}, "
          f"vdw {t['vdw'] / s.energy:+.5f}; point charges {ec / s.energy:+.6f}; rest (the CCSD(T) corrections) "
          f"{(s.energy - t['total'] + t['intra']) / s.energy:+.1e}")


def dagger_tail(name, structs):
    """the harness's dagger on the reading rows (E_0 has nothing beyond r_c = 5 A: any parameters give its tail)."""
    a = type("A", (), dict(rcut=5.0, F=16, lA=4.0, l1=0.0, rs_over_l1=0.0, charges=None, charge_neutrality="fragment"))()
    rung = E3.configure(E3.RUNGS[name], a)
    structs = [W.water_structure(bohr_to_ang(s.positions), energy=s.energy, charges=s.pinned["q"]) for s in structs]
    if rung.pin_alpha:
        E3.pin_dispersion(structs, truth)
    params = M.init_params(jax.random.PRNGKey(0), rung)
    return T.predict_energies(params, rung, structs)


for name in ("M_1p", "M_16p"):
    E = dagger_tail(name, rows)
    err = (E - np.array([s.energy for s in rows])) / np.array([s.energy for s in rows])
    print(f"{name}'s exact relative error at the reading rows: " + ", ".join(f"{v:+.3e}" for v in err))
omega, w = M.kn.casimir_polder_grid(8)
aO = E3.pinned_alpha_oscillator([8], alpha=alpha, c6=c6)[0]
print(f"  (the pinned oscillator's C6 by the K = 8 quadrature: {float(M.kn.c6_from_alpha(aO, aO, w)):.4f} vs TTM3-F's {c6:.4f})")

# M_1's class beyond r_c, as for E3-Q: lambda^2 x the band of the truth's charges, with a linear charge response
Ec, Fc, Et, Ft, beyond = [], [], [], [], []
for s in ident:
    pos = bohr_to_ang(s.positions)
    e, f = coulomb_band(pos, s.pinned["q"], 1.5, frag)
    Ec.append(e); Fc.append(f); Et.append(s.energy); Ft.append(s.forces)
    beyond.append((s.energy - e) / s.energy)
Ec, Et = np.array(Ec), np.array(Et); Fc, Ft = np.array(Fc), np.array(Ft)
wE = 1.0 / 36.0; wF = 1.0 / 6.0


def charge_response_basis(pos_ang, qq, l1_ang):
    """the gradients (18,) of the band energy of a dimer through a linear response of the charges to each monomer's
    internal coordinates (r1, r2, theta): three parameters by the H1 <-> H2 symmetry (as for E3-Q)."""
    x = ang_to_bohr(np.asarray(pos_ang, float)); l = ang_to_bohr(l1_ang)
    Phi = np.zeros(6)
    for i in range(6):
        for j in range(6):
            if (i < 3) != (j < 3):
                r = np.linalg.norm(x[i] - x[j]); Phi[i] += qq[j] * erf(r / l) / r
    basis = np.zeros((3, 18))
    for m in (0, 1):
        O, H1, H2 = x[3 * m], x[3 * m + 1], x[3 * m + 2]
        v1, v2 = H1 - O, H2 - O; r1, r2 = np.linalg.norm(v1), np.linalg.norm(v2); e1, e2 = v1 / r1, v2 / r2
        cos = e1 @ e2; sin = math.sqrt(1 - cos ** 2)
        B = np.zeros((3, 9))
        B[0, 3:6] = e1; B[0, 0:3] = -e1
        B[1, 6:9] = e2; B[1, 0:3] = -e2
        dth1 = -(e2 - cos * e1) / (r1 * sin); dth2 = -(e1 - cos * e2) / (r2 * sin)
        B[2, 3:6] = dth1; B[2, 6:9] = dth2; B[2, 0:3] = -(dth1 + dth2)
        P = Phi[3 * m:3 * m + 3]
        for p in range(3):
            dq = np.zeros((3, 3))
            if p == 0: dq[1, 0] = 1; dq[2, 1] = 1
            if p == 1: dq[1, 1] = 1; dq[2, 0] = 1
            if p == 2: dq[1, 2] = 1; dq[2, 2] = 1
            dq[0] = -(dq[1] + dq[2])
            basis[p, 9 * m:9 * m + 9] = (P @ dq) @ B
    return basis


G = np.array([charge_response_basis(bohr_to_ang(s.positions), s.pinned["q"], 1.5) for s in ident])
rows_E = np.sqrt(wE) * np.column_stack([Ec, np.zeros((len(Ec), 3))])
rows_F = np.sqrt(wF) * np.concatenate([np.column_stack([f.ravel(), -g.T]) for f, g in zip(Fc, G)])
coef, *_ = np.linalg.lstsq(np.vstack([rows_E, rows_F]), np.concatenate([np.sqrt(wE) * Et, np.sqrt(wF) * Ft.reshape(-1)]), rcond=None)
lam2_resp = coef[0]
lam2_all = (wE * np.sum(Ec * Et) + wF * np.sum(Fc * Ft)) / (wE * np.sum(Ec ** 2) + wF * np.sum(Fc ** 2))
lam2_E = np.sum(Ec * Et) / np.sum(Ec ** 2)
print(f"identifying nodes: the truth's energy beyond the point charges is {100 * np.min(beyond):+.1f} to {100 * np.max(beyond):+.1f} % "
      "of the energy")
print(f"class minimum of M_1's charge scale on the {len(ident)} identifying nodes (lambda^2 -> learned dipole lambda x {mu:.4f} D):")
for label, l2 in (("energies only", lam2_E),
                  ("energies + forces, charges with a linear response to the monomer geometry", lam2_resp),
                  ("energies + all force components (charges fixed)", lam2_all)):
    print(f"   {label:76s} lambda^2 = {l2:.4f}, dipole {math.sqrt(l2) * mu:.4f} D ({100 * (math.sqrt(l2) - 1):+.1f} %), "
          f"tail error |lambda^2 - 1| = {abs(l2 - 1):.3f}")
Eq3 = np.array([coulomb_band(bohr_to_ang(s.positions), s.pinned["q"], 1e-6, frag)[0] for s in rows])
Etr = np.array([s.energy for s in rows])
print("M_1 at its class minimum on the tail (lambda^2 x the point charges against the truth): relative error at the reading "
      "rows: " + ", ".join(f"{abs(lam2_resp * e / t - 1):.3f}" for e, t in zip(Eq3, Etr)))
P0 = W.dimer_positions(Re); v = np.random.default_rng(0).normal(size=P0.shape); v /= np.linalg.norm(v)
tt = np.arange(-100, 101) * 1e-7
Es = np.array([truth.energy(P0 + s * v, 2) for s in tt])
print(f"the truth's rounding noise at R_e: {rms(Es - np.polyval(np.polyfit(tt, Es, 3), tt)):.1e} hartree (label noise sigma 1e-9)")

print(f"\n=== (c) the clusters of configs/water_qaqua_pol.yaml  [{time.time() - t_start:.0f} s]")
geo = dict(mc_steps=300, T_K=300.0, radius_ang=None, min_OO=2.7, min_any=1.7)
t0 = time.time()
train = W.cluster_dataset([3, 4, 5], 100, seed=0, sigma=1e-9, truth=truth, **geo)
print(f"training: {len(train)} clusters, interaction energy rms {rms([s.energy for s in train]) * K:.2f} kcal/mol "
      f"(generated in {time.time() - t0:.0f} s); their pair energies span "
      f"{min(truth.energy(np.concatenate([bohr_to_ang(s.positions)[3*i:3*i+3], bohr_to_ang(s.positions)[3*j:3*j+3]]), 2) for s in train for i in range(len(s.positions)//3) for j in range(i+1, len(s.positions)//3)) * K:+.2f} to "
      f"{max(truth.energy(np.concatenate([bohr_to_ang(s.positions)[3*i:3*i+3], bohr_to_ang(s.positions)[3*j:3*j+3]]), 2) for s in train for i in range(len(s.positions)//3) for j in range(i+1, len(s.positions)//3)) * K:+.2f} kcal/mol")
tests = {6: W.cluster_dataset([6], 24, seed=100, truth=truth, **geo), 2: W.cluster_dataset([2], 24, seed=200, truth=truth, **geo)}


def cross_terms(st, l1=0.75):
    """the first-order cross terms of the band field of the other molecules' pinned charges, summed per species (O, H)."""
    x = st.positions; qq = st.pinned["q"]; n = len(x) // 3; l = ang_to_bohr(l1)
    mol = np.repeat(np.arange(n), 3)
    d = x[:, None, :] - x[None, :, :]; r = np.linalg.norm(d, axis=-1) + np.eye(len(x))
    dg = (2 / math.sqrt(math.pi) * np.exp(-(r / l) ** 2) / l - erf(r / l) / r) / r
    f = -(qq[None, :] * dg)[..., None] * d / r[..., None]                         # field at i from atom a
    f[mol[:, None] == mol[None, :]] = 0.0
    Ef = np.stack([f[:, mol == j].sum(axis=1) for j in range(n)], axis=1)         # (atoms, molecules, 3)
    cross = np.einsum("ija,ika->i", Ef, Ef) - np.einsum("ija,ija->i", Ef, Ef)
    Z = np.tile([8, 1, 1], n)
    return np.array([cross[Z == 8].sum(), cross[Z == 1].sum()])


def cluster_parts(structs):
    """per cluster: the non-additive energy (E minus the truth's dimer energies), its 3- and 4-body parts, TTM3-F's
    non-additive part (its induction beyond the pairs) and the cross terms."""
    out = []
    for s in structs:
        n = len(s.positions) // 3; pos = bohr_to_ang(s.positions)
        nadd = s.energy - W.pair_additive_energy(pos, n, truth)
        nb = truth.qa.nbody(s.positions, max_order=4)
        ind_pairs = sum(truth.qa.ttm(ang_to_bohr(np.concatenate([pos[3*i:3*i+3], pos[3*j:3*j+3]])))["ind"]
                        for i in range(n) for j in range(i + 1, n))
        out.append((nadd, nb.get(3, 0.0), nb.get(4, 0.0), truth.qa.ttm(s.positions)["ind"] - ind_pairs, *cross_terms(s)))
    return np.array(out)


h = cluster_parts(tests[6])
Eh = np.array([s.energy for s in tests[6]])
print(f"test hexamers (24): interaction rms {rms(Eh) * K:.2f} kcal/mol; non-additive rms {rms(h[:, 0]) * K:.3f} "
      f"({100 * rms(h[:, 0]) / rms(Eh):.1f} % of the interaction): 3-body rms {rms(h[:, 1]) * K:.3f}, 4-body {rms(h[:, 2]) * K:.3f}, "
      f"5- and 6-body {rms(h[:, 0] - h[:, 1] - h[:, 2]) * K:.4f}; TTM3-F's many-body induction {rms(h[:, 3]) * K:.3f} "
      f"(the CCSD(T) corrections' non-additive part {rms(h[:, 0] - h[:, 3]) * K:.3f}); dimers: interaction rms "
      f"{rms([s.energy for s in tests[2]]) * K:.3f}")
more = cluster_parts(W.cluster_dataset([6], 200, seed=900, truth=truth, **geo))


def fixed_fit(arr, label):
    A = -0.5 * arr[:, 4:6]
    c, *_ = np.linalg.lstsq(A, arr[:, 0], rcond=None)
    res = arr[:, 0] - A @ c
    print(f"  {label}: fixed per-species order-0 coefficients a_O = {c[0]:.2f}, a_H = {c[1]:.2f} a.u.; residual "
          f"{100 * rms(res) / rms(arr[:, 0]):.0f} % of the non-additive rms ({rms(arr[:, 0]) * K:.3f} kcal/mol)")
    return c


c_more = fixed_fit(more, "200 hexamers")
res_t = h[:, 0] - (-0.5 * h[:, 4:6]) @ c_more
print(f"  the 24 test hexamers with those coefficients: residual {100 * rms(res_t) / rms(h[:, 0]):.0f} % of their non-additive rms")

print(f"\n=== (e) the probe configurations of the config  [{time.time() - t_start:.0f} s]")
core_geo = {k: v for k, v in geo.items() if k != "mc_steps"}
dists = (8.0, 10.0, 12.0, 15.0)
for label, count, seed, cores in (("training", 5, 400, 8), ("test", 10, 500, 2)):
    probes = W.probe_configurations(3, dists, count, seed=seed, mc_steps=300, n_cores=cores, truth=truth, **core_geo)[0]
    ref = E3.probe_truth(probes, truth)
    cross = np.array([cross_terms(s) - cross_terms(W.water_structure(bohr_to_ang(s.positions)[:9], charges=s.pinned["q"][:9]))
                      for s in probes])
    pred = -0.5 * cross @ c_more
    gap = np.array([float(np.min(np.linalg.norm(bohr_to_ang(s.positions)[:9, None] - bohr_to_ang(s.positions)[None, 9:], axis=-1)))
                    for s in probes])
    print(f"{label} probes: the probe's nearest core atom (A) by d: " + "; ".join(
        f"{d:g} A {gap[ref['d'] == d].min():.2f}-{gap[ref['d'] == d].max():.2f}, {100 * np.mean(gap[ref['d'] == d] > 6.0):.0f} % beyond r_c = 6 A"
        for d in dists))
    print(f"{label} probes ({cores} cores x {len(dists)} distances x {count}): by d -- interaction rms, non-additive rms, induction rms "
          "(kcal/mol); the non-additive part with the hexamers' fixed coefficients (residual % of its rms)")
    # the daggers' probe interaction where E_0 cannot see the probe: the pinned charges' Coulomb (M_1p), + the C6 band (M_16p)
    fr = np.repeat([0, 0, 0, 1], 3)
    pc = np.array([coulomb_band(bohr_to_ang(s.positions), s.pinned["q"], 1e-6, fr)[0] for s in probes])
    disp = np.array([-c6 * sum(np.linalg.norm(s.positions[9] - s.positions[3 * i]) ** -6 for i in range(3)) for s in probes])
    for d in dists:
        m = ref["d"] == d
        print(f"  {d:4.0f} A: {rms(ref['interaction'][m]) * K:.5f}, {rms(ref['nonadditive'][m]) * K:.5f}, {rms(ref['induction'][m]) * K:.5f}; "
              f"fixed form {100 * rms(ref['nonadditive'][m] - pred[m]) / rms(ref['nonadditive'][m]):.0f} %; the pinned charges' "
              f"Coulomb {100 * rms(pc[m] - ref['interaction'][m]) / rms(ref['interaction'][m]):.0f} % of the interaction, with the "
              f"C6 term {100 * rms(pc[m] + disp[m] - ref['interaction'][m]) / rms(ref['interaction'][m]):.0f} %")

print(f"\n=== (d) the held-out raw CCSD(T) sets: every monomer within 0.03 A / 5 deg of the rigid geometry, every pair's\n"
      f"    q-AQUA-pol 2-body energy at most +2 kcal/mol  [{time.time() - t_start:.0f} s]")
bins = ((0, 5), (5, 7), (7, 9), (9, math.inf))
for key, (fname, n) in QA.RAW_SETS.items():
    pos, lab = QA.read_raw(os.path.join(truth.qa.root, "q-AQUA", fname), n)
    keep = QA.near_rigid(pos, W.R_OH, math.degrees(W.THETA), 0.03, 5.0)
    strict_all = QA.near_rigid(pos, W.R_OH, math.degrees(W.THETA), 0.02, 3.0)
    idx = np.flatnonzero(keep)
    worst = np.array([max(truth.energy(np.concatenate([pos[f][3 * i:3 * i + 3], pos[f][3 * j:3 * j + 3]]), 2)
                          for i in range(n) for j in range(i + 1, n)) for f in idx]) * K
    sel = idx[worst <= 2.0]
    P, L, strict = pos[sel], lab[sel], strict_all[sel]
    pes = np.array([truth.qa.nbody(ang_to_bohr(p), max_order=n)[n] for p in P])
    roo = QA.largest_oo(P)
    print(f"{key}: {len(sel)} frames ({keep.sum()} of {len(pos)} near-rigid; {strict.sum()} strict); labels rms {rms(L) * K:.4f} "
          f"kcal/mol, PES rmse {rms(pes - L) * K:.4f} ({100 * rms(pes - L) / rms(L):.0f} %); strict: labels {rms(L[strict]) * K:.4f}, "
          f"PES {rms(pes[strict] - L[strict]) * K:.4f}")
    for a_, b_ in bins:
        m = (roo >= a_) & (roo < b_)
        if m.any():
            print(f"    largest O-O {a_}-{b_} A: n = {m.sum():5d}, labels rms {rms(L[m]) * K:.4f}, PES rmse {rms(pes[m] - L[m]) * K:.4f} "
                  f"({100 * rms(pes[m] - L[m]) / rms(L[m]):.0f} %), PES mean error {np.mean(pes[m] - L[m]) * K:+.4f}")
    if key == "3b":
        A = -0.5 * np.array([cross_terms(W.water_structure(p, charges=truth.molecule_charges(p))) for p in P])
        for a_, b_ in bins:
            m = (roo >= a_) & (roo < b_)
            if m.sum() > 3:
                c, *_ = np.linalg.lstsq(A[m], L[m], rcond=None)
                print(f"    3-body by first-order cross terms, fitted per bin {a_}-{b_} A: residual {100 * rms(L[m] - A[m] @ c) / rms(L[m]):.0f} % "
                      f"(a_O {c[0]:.1f}, a_H {c[1]:.1f}); with the hexamer coefficients {100 * rms(L[m] - A[m] @ c_more) / rms(L[m]):.0f} %")
print(f"\ndone in {time.time() - t_start:.0f} s")
