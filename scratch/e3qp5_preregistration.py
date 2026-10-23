"""
E3-QP5's numbers before training (notes/e3qp5-preregistration.md section 3), from the PES and the data alone:

  (a) the field input's reach -- the M-site's offset from its O, and for every probe configuration of the run (E3-QP4's
      training and test probes, rebuilt bit for bit) whether any probe source lies within the taper's end of a core atom
      or any core source within it of a probe atom: where the field term can see a probe at all;
  (b) the field input's size on the training data (the 300 relaxed clusters and the 800 contact configurations): the
      potential, field and field gradient of the other molecules' pinned charges at the O and H atoms, tapered as the
      field rungs read them;
  (c) how much of the raw CCSD(T) trimers' 3-body (E3-QP4's 4673 held-out frames, the rigid monomer's charges as the
      models read them) one effective polarisability per species on the field input's first-order cross term explains:
      label_3b ~ -sum_i a_Z(i) E_i(B) . E_i(C) over every atom i of A (and cyclically), least squares -- the 3-body that a
      term quadratic in the field input carries at first order, against TTM3-F's 3-body induction (0.355 of 0.623 kcal/mol
      rms; scratch/e3qp3_response_contact.py).

    python scratch/e3qp5_preregistration.py [build_pol]          # ~10 min, one CPU
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
L1, ON, OFF, RC = ang_to_bohr(0.75), ang_to_bohr(4.5), ang_to_bohr(5.5), 6.0
q_r, w_r, vq_r, _ = truth.site_model(W.monomer_positions())
mono = W.monomer_positions()
m_site = w_r[0] @ mono
print(f"(a) the M-site: gamma {truth.gamma_m:.4f}, {np.linalg.norm(m_site - mono[0]):.3f} A from its O; site charges H {q_r[1]:+.4f}, "
      f"M {vq_r[0]:+.4f} (rigid monomer)  [{time.time() - t0:.0f} s]", flush=True)


def sources(pos_ang):
    """atoms then M-sites (bohr), their rigid charges and molecules."""
    n = len(pos_ang) // 3
    P = ang_to_bohr(np.asarray(pos_ang, float))
    M = np.array([w_r[0] @ P[3 * m:3 * m + 3] for m in range(n)])
    X = np.concatenate([P, M]); q = np.concatenate([np.tile(q_r[:3], n), np.full(n, vq_r[0])])
    mol = np.concatenate([np.repeat(np.arange(n), 3), np.arange(n)])
    return P, X, q, mol


# (a) the probes' reach
geometry = dict(T_K=300.0, radius_ang=None, min_OO=2.7, min_any=1.7)
probes = {"train": [], "test": []}
for key, seed, n_cores, count, sizes in (("train", 400, 16, 2, (3, 4, 5)), ("test", 500, 2, 10, (3,))):
    for k, n_core in enumerate(sizes):
        probes[key] += W.probe_configurations(n_core, (8.0, 10.0, 12.0, 15.0), count, seed=seed + 1000 * k, mc_steps=300,
                                              n_cores=n_cores, truth=truth, **geometry)[0]
print(f"    {len(probes['train'])} training and {len(probes['test'])} test probes rebuilt  [{time.time() - t0:.0f} s]", flush=True)
for key, structs in probes.items():
    rows = {}
    for st in structs:
        pos = bohr_to_ang(st.positions); n = len(pos) // 3
        P, X, q, mol = sources(pos)
        core_atoms, probe_atoms = np.arange(3 * (n - 1)), np.arange(3 * (n - 1), 3 * n)
        probe_src, core_src = np.flatnonzero(mol == n - 1), np.flatnonzero(mol < n - 1)
        gap = np.linalg.norm(pos[:-3, None] - pos[None, -3:], axis=-1).min()
        reach = min(np.linalg.norm(P[core_atoms][:, None] - X[probe_src][None], axis=-1).min(),
                    np.linalg.norm(P[probe_atoms][:, None] - X[core_src][None], axis=-1).min())
        d = st.info["d"]
        r = rows.setdefault(d, [0, 0, 0])
        r[0] += 1; r[1] += gap > RC; r[2] += reach >= OFF
    print(f"    {key} probes by d: " + "; ".join(f"{d:g} A: {n_} probes, {b} beyond r_c of the core (E_0 blind), {f} beyond the "
                                                 f"field's reach" for d, (n_, b, f) in sorted(rows.items())), flush=True)

# (b) the field input on the training data
clusters = W.cluster_dataset((3, 4, 5), 100, seed=0, sigma=1e-9, truth=truth, mc_steps=300, **geometry)
data = {"contact_dimers": 600, "contact_trimers": 200, "contact_oo_angstrom": [2.3, 3.0], "contact_seed": 700,
        "contact_max_pair_kcal": 2.0, "contact_min_atom_distance_angstrom": 1.5}
contacts = E3.contact_data(data, truth)
print(f"(b) {len(clusters)} training clusters and {len(contacts)} contact configurations rebuilt  [{time.time() - t0:.0f} s]", flush=True)


@jax.jit
def field_of(P, X, q, src):
    return BF.local_field_tensors(P, X, q, src, L1, ON, OFF)


def field_stats(structs):
    out = {8: [], 1: []}
    for st in structs:
        pos = bohr_to_ang(st.positions)
        P, X, q, mol = sources(pos)
        src = np.repeat(np.arange(len(pos) // 3), 3)[:, None] != mol[None, :]
        phi, E, G = (np.asarray(x) for x in field_of(jnp.asarray(P), jnp.asarray(X), jnp.asarray(q), jnp.asarray(src)))
        for i, z in enumerate(st.numbers):
            out[int(z)].append((abs(phi[i]), np.linalg.norm(E[i]), np.linalg.norm(G[i])))
    return {z: np.array(v) for z, v in out.items()}


for label, structs in (("relaxed clusters", clusters), ("contact configurations", contacts)):
    st_ = field_stats(structs)
    print(f"    {label}: " + "; ".join(
        f"{'O' if z == 8 else 'H'} |phi| rms {np.sqrt(np.mean(v[:, 0] ** 2)):.4f}, |E| rms {np.sqrt(np.mean(v[:, 1] ** 2)):.4f} "
        f"(99 % {np.quantile(v[:, 1], 0.99):.4f}), |G| rms {np.sqrt(np.mean(v[:, 2] ** 2)):.4f} a.u."
        for z, v in st_.items()) + f"  [{time.time() - t0:.0f} s]", flush=True)

# (c) the raw trimers' 3-body against the field input's first-order cross term
a = type("A", (), dict(heldout="3b", heldout_oh_tol=0.03, heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02,
                       heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0, heldout_max=0))()
d3 = E3.prepare_heldout(a, truth)["3b"]
Pf = np.asarray(d3["pos"]); lab = np.asarray(d3["label"]) * K; soo = np.asarray(d3["soo"])


def fields_by_molecule(Pa, taper=True):
    """the field (a.u.) at every atom of every frame (F, 9, 3 A) of each molecule's pinned charges (atoms and M-site,
    the rigid monomer's charges), the band kernel at l_1 and each share tapered as the field rungs read it: (F, 9, 3, 3),
    [frame, atom, source molecule, axis].  numpy, as bandfields.local_field_tensors."""
    from scipy.special import erf
    P = ang_to_bohr(np.asarray(Pa, float)).reshape(len(Pa), 3, 3, 3)                 # frame, molecule, atom, axis
    S = np.concatenate([P, np.einsum("a,fmax->fmx", w_r[0][:3], P)[:, :, None]], axis=2)   # + the M-site
    qs = np.array([q_r[0], q_r[1], q_r[2], vq_r[0]])
    d = P.reshape(len(Pa), 9, 1, 1, 3) - S[:, None]                                  # (F, 9, 3 molecules, 4 sources, 3)
    r = np.linalg.norm(d, axis=-1)
    r = np.where(r > 0, r, 1.0)                                                      # an atom as its own source
    dg = (2.0 / (L1 * math.sqrt(math.pi))) * np.exp(-(r / L1) ** 2) / r - erf(r / L1) / r ** 2
    w = np.ones_like(r)
    if taper:
        x = np.clip((r - ON) / (OFF - ON), 0.0, 1.0); w = 1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):                             # the own molecule's shares: unused
        return -np.einsum("fiys,fiysx->fiyx", w * qs * dg, d / r[..., None])


def cross_features(Pa, taper=True):
    """per frame: sum over the O atoms and over the H atoms of E_i(Y) . E_i(Z), Y and Z the other two molecules (a.u.^2)."""
    E = fields_by_molecule(Pa, taper)
    out = np.zeros((len(Pa), 2))
    for m in range(3):
        y, z = [k for k in range(3) if k != m]
        c = np.sum(E[:, 3 * m:3 * m + 3, y] * E[:, 3 * m:3 * m + 3, z], axis=-1)
        out[:, 0] += c[:, 0]; out[:, 1] += c[:, 1] + c[:, 2]
    return out


Xf = cross_features(Pf)
Xu = cross_features(Pf[soo < 3.0], taper=False)
print(f"(c) {len(Pf)} raw trimers, their first-order field cross terms  [{time.time() - t0:.0f} s]", flush=True)
rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
coef, *_ = np.linalg.lstsq(-Xf * K, lab, rcond=None)
res = lab + (Xf * K) @ coef
print(f"    one effective polarisability per species (bohr^3): O {coef[0]:.3f}, H {coef[1]:.3f}; the raw 3-body rms {rms(lab):.4f} "
      f"kcal/mol, after the fit {rms(res):.4f} ({100 * (1 - rms(res) / rms(lab)):.0f} % of its rms explained)", flush=True)
for lo, hi in ((0.0, 2.7), (2.7, 3.0), (3.0, 99.0)):
    m = (soo >= lo) & (soo < hi)
    print(f"      smallest O-O {lo:g}-{hi:g} A (n {int(m.sum())}): {rms(lab[m]):.4f} -> {rms(res[m]):.4f} kcal/mol", flush=True)
coef_u, *_ = np.linalg.lstsq(-Xu * K, lab[soo < 3.0], rcond=None)
res_u = lab[soo < 3.0] + (Xu * K) @ coef_u
print(f"    without the taper (below O-O 3 A): {rms(lab[soo < 3.0]):.4f} -> {rms(res_u):.4f} kcal/mol (taper: "
      f"{rms(res[soo < 3.0]):.4f})", flush=True)
print(f"done in {time.time() - t0:.0f} s")
