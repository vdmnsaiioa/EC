"""
What a pinned-source response can reach on the probes' non-additive energy, before any training of the harness:

    python scratch/e3qp_response_floor.py [external/qaqua/build_pol]

Per-species polarisabilities (O, H) fitted on the 160 TRAINING probes' non-additive energy -- the probe's interaction
minus its three dimers, the quantity probe-difference labels hand the read-out -- and read on the 80 test probes and the
24 test hexamers (against TTM3-F's many-body induction there), for:

  sources   q3: the dipole-surface charges (E3-QP's pins); mp: TTM3-F's multipoles on the atoms (the M-site charge on O
            with an O dipole q_M (r_M - r_O), the H charges)
  response  first order: -1/2 sum_i a_Z |E_i|^2 (E3-QP's order-0 read-out with constant coefficients);
            scf: induced dipoles mu = (1 - A T)^-1 A E through the band kernel's dipole tensor between molecules
  edge      the band field and tensor at l = 0.75 A (E3-QP's) and larger l (closer to TTM3-F's Thole damping of the
            near field: 1 - exp(-0.175 (r/A)^3) is 0.89 at the hydrogen bond's 2.0 A)

and the sources' own accuracy: the dimer tail at the reading rows and the probes' pair electrostatics.
"""
import math, os, sys, time
import numpy as np
from scipy.special import erf
from scipy.optimize import least_squares

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from sseft import water as W, qaqua as QA, data as D
from sseft.units import ang_to_bohr, bohr_to_ang
import e3_water as E3

K = 627.5094740631
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "external", "qaqua", "build_pol")
truth = W.qaqua_pol_truth(root)
rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
t_start = time.time()
GAMMA_HALF = 0.23                                       # TTM3-F: r_M = r_O + 0.23 (r_H1 + r_H2 - 2 r_O)


def sources(pos_ang, kind):
    """(X bohr, q e, mu e bohr) of the pinned sources on the atoms."""
    pos_ang = np.asarray(pos_ang, float); X = ang_to_bohr(pos_ang); n = len(X) // 3
    if kind == "q3":
        return X, truth.molecule_charges(pos_ang), np.zeros_like(X)
    qs = truth.qa.ttm(X)["site_charges"]                 # (n, 3): H1, H2, M
    q = np.zeros(3 * n); mu = np.zeros_like(X)
    for m in range(n):
        q[3 * m:3 * m + 3] = [qs[m, 2], qs[m, 0], qs[m, 1]]
        mu[3 * m] = qs[m, 2] * GAMMA_HALF * (X[3 * m + 1] + X[3 * m + 2] - 2 * X[3 * m])
    return X, q, mu


def kernel(r, l):
    g = erf(r / l) / r
    gp = (2 / math.sqrt(math.pi) * np.exp(-(r / l) ** 2) / l - g) / r
    gpp = -4 / math.sqrt(math.pi) * np.exp(-(r / l) ** 2) / l ** 3 - 2 * gp / r
    return g, gp, gpp


def fields_and_tensor(X, q, mu, l):
    """the band field at every atom of the other molecules' sources (N, 3), the band dipole tensor between atoms of
    different molecules (3N, 3N), and the band Coulomb energy between molecules (charges and dipoles)."""
    N = len(X); mol = np.arange(N) // 3
    R = X[:, None, :] - X[None, :, :]; r = np.linalg.norm(R, axis=-1)
    other = mol[:, None] != mol[None, :]
    r_ = np.where(other, r, 1.0); n = R / r_[..., None]
    g, gp, gpp = kernel(r_, l)
    Hs = gpp[..., None, None] * n[..., :, None] * n[..., None, :] + (gp / r_)[..., None, None] * (np.eye(3) - n[..., :, None] * n[..., None, :])
    Hs = np.where(other[..., None, None], Hs, 0.0)
    E = -np.einsum("ij,j,ija->ia", np.where(other, gp, 0.0), q, n) + np.einsum("ijab,jb->ia", Hs, mu)
    T = Hs.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
    # band electrostatics: q-q g, q-mu and mu-mu through the same kernel (for the dimer tail / pair checks)
    gq = np.where(other, g, 0.0)
    Ecoul = 0.5 * q @ gq @ q
    Ecoul -= np.einsum("i,ij,ija,ja->", q, np.where(other, gp, 0.0), n, mu)      # q_i in mu_j's potential -mu_j . grad g
    Ecoul += -0.5 * np.einsum("ia,iajb,jb->", mu, Hs.transpose(0, 2, 1, 3), mu)
    return E, T, Ecoul


def response(E, T, a, scf):
    """induction energy of per-atom isotropic polarisabilities a (N,): first order or self-consistent."""
    av = np.repeat(a, 3); e = E.reshape(-1)
    if not scf:
        return -0.5 * float(np.sum(av * e * e))
    mu = np.linalg.solve(np.eye(len(e)) - av[:, None] * T, av * e)
    return -0.5 * float(mu @ e)


def subsystems(pos_ang, n_core):
    """(all, core, core molecule k + probe ...) for a probe configuration."""
    n = n_core + 1
    mols = [list(range(n)), list(range(n_core))] + [[k, n_core] for k in range(n_core)]
    return [np.concatenate([pos_ang[3 * m:3 * m + 3] for m in ms]) for ms in mols], [1, -1] + [-1] * n_core


def cluster_subsystems(pos_ang, n):
    pairs = [[i, j] for i in range(n) for j in range(i + 1, n)]
    return ([pos_ang] + [np.concatenate([pos_ang[3 * m:3 * m + 3] for m in p]) for p in pairs]), [1] + [-1] * len(pairs)


def precompute(confs, split, kind, l):
    """per configuration: the signed list of (E, T, Z) of its subsystems."""
    out = []
    for pos in confs:
        subs, signs = split(pos)
        items = []
        for p, sgn in zip(subs, signs):
            X, q, mu = sources(p, kind)
            E, T, _ = fields_and_tensor(X, q, mu, l)
            items.append((sgn, E, T, np.tile([0, 1, 1], len(p) // 3)))
        out.append(items)
    return out


def predict(pre, aO, aH, scf):
    return np.array([sum(sgn * response(E, T, np.where(Z == 0, aO, aH), scf) for sgn, E, T, Z in items) for items in pre])


# ---------------------------------------------------------------- the sources: the dimer tail and the pair electrostatics
print(f"q-AQUA-pol at {truth.qa.root}")
Re = truth.r_e
R_eval = bohr_to_ang(D.eval_grid(1.35 * Re * 1.8, 500.0, 61))
rows = [R_eval[int(np.argmin(np.abs(R_eval - Rv)))] for Rv in (8, 10, 12, 15, 20, 30, 50, 100, 300)]
print("the dimer tail at the reading rows (phi = 0): relative error of the pinned sources' Coulomb (point limit, l -> 0) "
      "against the truth")
for kind in ("q3", "mp"):
    errs = []
    for R in rows:
        pos = W.dimer_positions(R)
        X, q, mu = sources(pos, kind)
        errs.append(fields_and_tensor(X, q, mu, 1e-3)[2] / truth.energy(pos, 2) - 1)
    print(f"  {kind}: " + ", ".join(f"{e:+.4f}" for e in errs))

hexa_geo = dict(mc_steps=300, T_K=300.0, radius_ang=None, min_OO=2.7, min_any=1.7)
core_geo = {k: v for k, v in hexa_geo.items() if k != "mc_steps"}
dists = (8.0, 10.0, 12.0, 15.0)
sets = {}
for label, count, seed, cores in (("train", 5, 400, 8), ("test", 10, 500, 2)):
    st = W.probe_configurations(3, dists, count, seed=seed, mc_steps=300, n_cores=cores, truth=truth, **core_geo)[0]
    sets[label] = (st, E3.probe_truth(st, truth))
print(f"probes generated  [{time.time() - t_start:.0f} s]")
st, ref = sets["test"]
_, pairs = E3.probe_parts(st)
for kind in ("q3", "mp"):
    pc = []
    for s in st:
        pos = bohr_to_ang(s.positions)
        X, q, mu = sources(pos, kind)
        X0, q0, mu0 = sources(pos[:9], kind)
        pc.append(fields_and_tensor(X, q, mu, 1e-3)[2] - fields_and_tensor(X0, q0, mu0, 1e-3)[2])
    pc = np.array(pc)
    c6 = truth.c6
    disp = np.array([-c6 * sum(np.linalg.norm(s.positions[9] - s.positions[3 * i]) ** -6 for i in range(3)) for s in st])
    print(f"  test probes, {kind}: the probe's interaction minus [the sources' Coulomb + the O-O C6 term], % of the interaction rms: "
          + ", ".join(f"{d:g} A {100 * rms(ref['interaction'][ref['d'] == d] - pc[ref['d'] == d] - disp[ref['d'] == d]) / rms(ref['interaction'][ref['d'] == d]):.1f}"
                      for d in dists) + "   (the remainder is the induction: "
          + ", ".join(f"{100 * rms(ref['induction'][ref['d'] == d]) / rms(ref['interaction'][ref['d'] == d]):.1f}" for d in dists) + ")")

# ---------------------------------------------------------------- the response floors -------------------------------
hexamers = W.cluster_dataset([6], 24, seed=100, truth=truth, **hexa_geo)
hex_pos = [bohr_to_ang(s.positions) for s in hexamers]
hex_ind = np.array([truth.qa.ttm(ang_to_bohr(p))["ind"] - sum(truth.qa.ttm(ang_to_bohr(np.concatenate([p[3*i:3*i+3], p[3*j:3*j+3]])))["ind"]
                                                             for i in range(6) for j in range(i + 1, 6)) for p in hex_pos])
probe_pos = {k: [bohr_to_ang(s.positions) for s in v[0]] for k, v in sets.items()}
y_tr = sets["train"][1]["nonadditive"]; y_te = ref["nonadditive"]
print(f"\nthe response floors: per-species polarisabilities fitted on the {len(y_tr)} training probes' non-additive energy, read on "
      f"the {len(y_te)} test probes (% of the non-additive rms by distance) and on the test hexamers (% of TTM3-F's many-body "
      f"induction, rms {rms(hex_ind) * K:.3f} kcal/mol)  [{time.time() - t_start:.0f} s]")
for kind in ("q3", "mp"):
    for l_ang in (0.75, 1.0, 1.25):
        l = ang_to_bohr(l_ang)
        tr = precompute(probe_pos["train"], lambda p: subsystems(p, 3), kind, l)
        te = precompute(probe_pos["test"], lambda p: subsystems(p, 3), kind, l)
        hx = precompute(hex_pos, lambda p: cluster_subsystems(p, 6), kind, l)
        for scf in (False, True):
            if not scf:
                A = np.column_stack([predict(tr, 1.0, 0.0, False), predict(tr, 0.0, 1.0, False)])
                aO, aH = np.linalg.lstsq(A, y_tr, rcond=None)[0]
            else:
                fit = least_squares(lambda a: (predict(tr, a[0], a[1], True) - y_tr) * 1e4, x0=[8.0, 1.0], bounds=([0, 0], [20, 10]))
                aO, aH = fit.x
            p_tr = predict(tr, aO, aH, scf); p_te = predict(te, aO, aH, scf); p_hx = predict(hx, aO, aH, scf)
            d_te = sets["test"][1]["d"]
            print(f"  {kind:2s} l = {l_ang:4.2f} A {'scf  ' if scf else 'first'}: a_O {aO:5.2f}, a_H {aH:5.2f}; train "
                  f"{100 * rms(p_tr - y_tr) / rms(y_tr):4.0f} %; test by d "
                  + ", ".join(f"{100 * rms(p_te[d_te == d] - y_te[d_te == d]) / rms(y_te[d_te == d]):3.0f}" for d in dists)
                  + f" %; hexamers {100 * rms(p_hx - hex_ind) / rms(hex_ind):3.0f} %", flush=True)
print(f"\ndone in {time.time() - t_start:.0f} s")
