"""
The numbers of notes/e3q-preregistration.md, computed from the q-AQUA PES before any training:

    python scratch/e3q_preregistration.py [external/qaqua/build]

(a) the dimer: R_e and D_e at phi = 0, the window, the identifying nodes at r_c = 5 and 6 A, the truth's tail (point
    dipoles of the LTP2011 charges), the class minimum of M_1's charge scale on the identifying nodes under the training
    objective (energies per atom and forces, weight 1), M_1-dagger's exact error at the reading rows, the share of the
    identifying nodes' energy beyond the point charges;
(c) the clusters of configs/water_qaqua.yaml (same seeds): interaction and non-additive energies, the 3- and 4-body
    shares, and the floor of a fixed-coefficient order-0 read-out -- the non-additive energy fitted by the first-order
    cross terms sum_i sum_{j != k} E_ij . E_ik of the band field (l_1 = 0.75 A) of the other molecules' LTP2011 charges,
    per species -- on the test hexamers and on 200 more hexamers of the same protocol;
(d) the held-out raw CCSD(T) sets under the near-rigid filter: counts, label rms, the PES's own rmse, by largest O-O,
    and the same first-order-cross-term fit on the 3-body set by distance (how much of the CCSD(T) 3-body energy that the
    PES drops beyond 7-9 A a fixed-coefficient order-0 term could carry).
"""
import itertools, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sseft import water as W, qaqua as QA, data as D
from sseft.units import ang_to_bohr, bohr_to_ang
from scipy.special import erf

K = 627.5094740631
root = sys.argv[1] if len(sys.argv) > 1 else "external/qaqua/build"
truth = W.qaqua_truth(root)
rms = lambda v: float(np.sqrt(np.mean(np.square(v))))


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


print("=== (a) the dimer at phi = 0 (rigid monomers r_OH 0.9572 A, HOH 104.52 deg)")
Re = truth.r_e; De = -truth.energy(W.dimer_positions(Re), 2)
lo, hi = 1.35 * Re, 1.35 * Re * 1.8
print(f"R_e = {Re:.5f} A, D_e = {De * K:.4f} kcal/mol; window [{lo:.4f}, {hi:.4f}] A")
R_train = bohr_to_ang(D.window_design(lo, hi, 40))
nodes = W.dimer_scan(R_train, truth=truth)
smallest = np.array([R - W.R_OH for R in R_train])
for rc in (5.0, 6.0):
    m = smallest > rc
    print(f"  r_c = {rc}: {m.sum()} identifying nodes, R >= {R_train[m].min() if m.any() else float('nan'):.3f} A")
ident = [s for s, m in zip(nodes, smallest > 5.0) if m]
mu, q = (np.linalg.norm(W.molecular_dipoles(truth, W.monomer_positions())[0]) * 2.541746473,
         truth.molecule_charges(W.monomer_positions()))
print(f"LTP2011 at the rigid geometry: q (O, H, H) = {np.round(q, 7)}, dipole {mu:.4f} D; point-dipole coefficient "
      f"{W.dipole_dipole_coefficient(truth):+.4f} hartree bohr^3")
# M_1's class beyond r_c: lambda^2 x the band of the truth's charges (a symmetric neutral monomer has one free charge)
frag = np.repeat([0, 1], 3)
Ec, Fc, Et, Ft, beyond = [], [], [], [], []
for s in ident:
    pos = bohr_to_ang(s.positions)
    e, f = coulomb_band(pos, s.pinned["q"], 1.5, frag)
    Ec.append(e); Fc.append(f); Et.append(s.energy); Ft.append(s.forces)
    beyond.append((s.energy - e) / s.energy)
Ec, Et = np.array(Ec), np.array(Et); Fc, Ft = np.array(Fc), np.array(Ft)
# the objective: mean over structures of (dE / 6 atoms)^2 + w_force * sum |dF|^2 / (6 atoms per structure)
wE = 1.0 / 36.0; wF = 1.0 / 6.0


def rigid_projector(pos_bohr):
    """projector (9n x 9n) onto the rigid-body motions (translation, rotation) of every molecule."""
    n = len(pos_bohr) // 3; P = np.zeros((3 * len(pos_bohr),) * 2)
    for m in range(n):
        x = pos_bohr[3 * m:3 * m + 3]; c = x.mean(0); B = []
        for k in range(3):
            t = np.zeros((3, 3)); t[:, k] = 1.0; B.append(t.ravel())
            w = np.zeros(3); w[k] = 1.0; B.append(np.cross(w, x - c).ravel())
        Q, _ = np.linalg.qr(np.array(B).T)
        sl = slice(9 * m, 9 * m + 9); P[sl, sl] = Q @ Q.T
    return P


Fc_rb = np.array([rigid_projector(s.positions) @ f.ravel() for s, f in zip(ident, Fc)])
Ft_rb = np.array([rigid_projector(s.positions) @ f.ravel() for s, f in zip(ident, Ft)])
lam2_all = (wE * np.sum(Ec * Et) + wF * np.sum(Fc * Ft)) / (wE * np.sum(Ec ** 2) + wF * np.sum(Fc ** 2))
lam2_rb = (wE * np.sum(Ec * Et) + wF * np.sum(Fc_rb * Ft_rb)) / (wE * np.sum(Ec ** 2) + wF * np.sum(Fc_rb ** 2))
lam2_E = np.sum(Ec * Et) / np.sum(Ec ** 2)
print(f"identifying nodes: the truth's energy beyond the point charges is {100 * np.min(beyond):+.1f} to {100 * np.max(beyond):+.1f} % "
      f"of the energy (the PIP / switch region below 7.8 A)")
def charge_response_basis(pos_ang, q, l1_ang):
    """the gradients (18,) of the band energy of a dimer through a linear response of the charges to each monomer's
    internal coordinates (r1, r2, theta): three parameters by the H1 <-> H2 symmetry, a = dq_H1/dr1 = dq_H2/dr2,
    b = dq_H1/dr2 = dq_H2/dr1, c = dq_H1/dtheta = dq_H2/dtheta, the O taking minus their sum (neutral monomers).
    dE/dx = sum_i (dq_i/dx) Phi_i with Phi_i the band potential at atom i of the other monomer's charges."""
    x = ang_to_bohr(np.asarray(pos_ang, float)); l = ang_to_bohr(l1_ang)
    Phi = np.zeros(6)
    for i in range(6):
        for j in range(6):
            if (i < 3) != (j < 3):
                r = np.linalg.norm(x[i] - x[j]); Phi[i] += q[j] * erf(r / l) / r
    basis = np.zeros((3, 18))
    for m in (0, 1):
        O, H1, H2 = x[3 * m], x[3 * m + 1], x[3 * m + 2]
        v1, v2 = H1 - O, H2 - O; r1, r2 = np.linalg.norm(v1), np.linalg.norm(v2); e1, e2 = v1 / r1, v2 / r2
        cos = e1 @ e2; sin = math.sqrt(1 - cos ** 2)
        B = np.zeros((3, 9))                                              # d(r1, r2, theta) / d(O, H1, H2)
        B[0, 3:6] = e1; B[0, 0:3] = -e1
        B[1, 6:9] = e2; B[1, 0:3] = -e2
        dth1 = -(e2 - cos * e1) / (r1 * sin); dth2 = -(e1 - cos * e2) / (r2 * sin)
        B[2, 3:6] = dth1; B[2, 6:9] = dth2; B[2, 0:3] = -(dth1 + dth2)
        P = Phi[3 * m:3 * m + 3]                                           # O, H1, H2
        D = {0: ((-1, -1), (1, 0), (0, 1)), 1: ((-1, -1), (0, 1), (1, 0)), 2: ((-1, -1), (1, 1), (0, 0))}
        for p in range(3):                                                 # p: a, b, c
            # dq_i/dxi_k for parameter p: H1 responds to (r1, r2, theta) with (a, b, c), H2 with (b, a, c)
            dq = np.zeros((3, 3))                                          # [atom O/H1/H2, coordinate r1/r2/theta]
            if p == 0: dq[1, 0] = 1; dq[2, 1] = 1
            if p == 1: dq[1, 1] = 1; dq[2, 0] = 1
            if p == 2: dq[1, 2] = 1; dq[2, 2] = 1
            dq[0] = -(dq[1] + dq[2])
            basis[p, 9 * m:9 * m + 9] = (P @ dq) @ B                       # dE/dx of this monomer's atoms
    return basis


G = np.array([charge_response_basis(bohr_to_ang(s.positions), s.pinned["q"], 1.5) for s in ident])   # (nodes, 3, 18)
rows_E = np.sqrt(wE) * np.column_stack([Ec, np.zeros((len(Ec), 3))])
rows_F = np.sqrt(wF) * np.concatenate([np.column_stack([f.ravel(), -g.T]) for f, g in zip(Fc, G)])
coef, *_ = np.linalg.lstsq(np.vstack([rows_E, rows_F]), np.concatenate([np.sqrt(wE) * Et, np.sqrt(wF) * Ft.reshape(-1)]), rcond=None)
lam2_resp = coef[0]
print(f"class minimum of M_1's charge scale on the {len(ident)} identifying nodes (lambda^2 -> learned dipole lambda x {mu:.4f} D):")
for label, l2 in (("energies only", lam2_E),
                  ("energies + forces, charges with a linear response to the monomer geometry", lam2_resp),
                  ("energies + rigid-body force components (all intramolecular components absorbed)", lam2_rb),
                  ("energies + all force components (charges fixed)", lam2_all)):
    print(f"   {label:80s} lambda^2 = {l2:.4f}, dipole {math.sqrt(l2) * mu:.4f} D ({100 * (math.sqrt(l2) - 1):+.1f} %), "
          f"tail error |lambda^2 - 1| = {abs(l2 - 1):.3f}")
print(f"   (the response: dq_H1/dr1, dq_H1/dr2, dq_H1/dtheta = {np.round(coef[1:] / lam2_resp, 4)} e/bohr, e/bohr, e/rad)")
R_rows = np.array([8, 10, 12, 15, 20, 30, 50, 100, 300.0])
rows = W.dimer_scan(R_rows, truth=truth)
errs = []
for s in rows:
    e, _ = coulomb_band(bohr_to_ang(s.positions), s.pinned["q"], 1.5, frag)
    errs.append(abs(e - s.energy) / abs(s.energy))
print("M_1-dagger relative error at " + ", ".join(f"{R:g}" for R in R_rows) + " A: " + ", ".join(f"{v:.1e}" for v in errs))
noise = []
P = W.dimer_positions(Re); v = np.random.default_rng(0).normal(size=P.shape); v /= np.linalg.norm(v)
t = np.arange(-100, 101) * 1e-7
E = np.array([truth.energy(P + s * v, 2) for s in t])
print(f"the truth's rounding noise at R_e: {rms(E - np.polyval(np.polyfit(t, E, 3), t)):.1e} hartree (label noise sigma 1e-9)")

print("\n=== (c) the clusters of configs/water_qaqua.yaml")
geo = dict(mc_steps=300, T_K=300.0, radius_ang=None, min_OO=2.7, min_any=1.7)
train = W.cluster_dataset([3, 4, 5], 12, seed=0, sigma=1e-9, truth=truth, **geo)
tests = {6: W.cluster_dataset([6], 8, seed=100, truth=truth, **geo), 2: W.cluster_dataset([2], 8, seed=200, truth=truth, **geo)}
print(f"training: 36 clusters, interaction energy rms {rms([s.energy for s in train]) * K:.2f} kcal/mol")


def nadd_and_cross(structs, n, l1=0.75):
    """the truth's non-additive energy (3-b + 4-b) and the first-order cross terms of the band field per species."""
    out = []
    for s in structs:
        pos = bohr_to_ang(s.positions); nb = truth.qa.nbody(s.positions)
        x = s.positions; qq = s.pinned["q"]; l = ang_to_bohr(l1)
        Ef = np.zeros((len(x), n, 3))                         # field at atom i from molecule j
        for i in range(len(x)):
            for j in range(n):
                if j == i // 3:
                    continue
                for a in range(3 * j, 3 * j + 3):
                    d = x[i] - x[a]; r = np.linalg.norm(d)
                    dg = (2 / math.sqrt(math.pi) * math.exp(-(r / l) ** 2) / l - erf(r / l) / r) / r
                    Ef[i, j] -= qq[a] * dg * d / r
        cross = np.einsum("ija,ika->i", Ef, Ef) - np.einsum("ija,ija->i", Ef, Ef)
        Z = np.tile([8, 1, 1], n)
        out.append((nb.get(3, 0.0) + nb.get(4, 0.0), nb.get(3, 0.0), nb.get(4, 0.0), cross[Z == 8].sum(), cross[Z == 1].sum()))
    return np.array(out)


h = nadd_and_cross(tests[6], 6)
print(f"test hexamers: interaction rms {rms([s.energy for s in tests[6]]) * K:.2f} kcal/mol; non-additive rms {rms(h[:, 0]) * K:.3f} "
      f"({100 * rms(h[:, 0]) / rms([s.energy for s in tests[6]]):.1f} % of the interaction), 3-body rms {rms(h[:, 1]) * K:.3f}, "
      f"4-body rms {rms(h[:, 2]) * K:.3f} kcal/mol; dimers: interaction rms {rms([s.energy for s in tests[2]]) * K:.3f}")
more = nadd_and_cross(W.cluster_dataset([6], 200, seed=900, truth=truth, **geo), 6)


def fixed_fit(arr, label):
    A = -0.5 * arr[:, 3:5]
    c, *_ = np.linalg.lstsq(A, arr[:, 0], rcond=None)
    res = arr[:, 0] - A @ c
    print(f"  {label}: fixed per-species order-0 coefficients a_O = {c[0]:.2f}, a_H = {c[1]:.2f} a.u.; residual "
          f"{100 * rms(res) / rms(arr[:, 0]):.0f} % of the non-additive rms ({rms(arr[:, 0]) * K:.3f} kcal/mol)")
    return c


c_more = fixed_fit(more, "200 hexamers")
res8 = h[:, 0] - (-0.5 * h[:, 3:5]) @ c_more
print(f"  the 8 test hexamers with those coefficients: residual {100 * rms(res8) / rms(h[:, 0]):.0f} % of their non-additive rms")

print("\n=== (d) the held-out raw CCSD(T) sets: every monomer within 0.03 A / 5 deg of the rigid geometry, every pair's PES\n"
      "    2-body energy at most +2 kcal/mol (the training domain: the training clusters' pairs span -4.8 to +1.2)")
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
                  f"({100 * rms(pes[m] - L[m]) / rms(L[m]):.0f} %)")
    if key == "3b":
        cross = []
        for p in P:
            s = W.water_structure(p, charges=truth.molecule_charges(p))
            cross.append(nadd_and_cross([s], 3)[0])
        cross = np.array(cross); A = -0.5 * cross[:, 3:5]
        for a_, b_ in bins:
            m = (roo >= a_) & (roo < b_)
            if m.sum() > 3:
                c, *_ = np.linalg.lstsq(A[m], L[m], rcond=None)
                print(f"    3-body by first-order cross terms, fitted per bin {a_}-{b_} A: residual {100 * rms(L[m] - A[m] @ c) / rms(L[m]):.0f} % "
                      f"(a_O {c[0]:.1f}, a_H {c[1]:.1f}); with the hexamer coefficients {100 * rms(L[m] - A[m] @ c_more) / rms(L[m]):.0f} %")
