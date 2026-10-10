"""
Water: rigid gas-phase monomer, the fixed-orientation dimer, rigid clusters, the monomer's point-charge
multipoles for the dagger rungs, and a synthetic polarisable truth for rehearsing E3 before the CCSD(T) data
exist (phase-d-build-plan.md section 3).

Geometry (gas phase): r_OH = 0.9572 A, HOH = 104.52 deg.  Point charges on the atoms reproducing the gas-phase
dipole 1.855 D exactly: q_H = +0.32949 e, q_O = -2 q_H (one free parameter after neutrality and symmetry; the
quadrupole is then whatever three charges give).

The dimer: donor O at the origin with its H-bonded O-H along +z, the second H in the xz-plane; acceptor O at
(0, 0, R), its bisector tilted by phi from +z in the xz-plane (H's pointing away from the donor), its molecular
plane perpendicular to the donor's (Cs symmetry).  R = R_OO is the scan coordinate.  The scan orientation is
phi = 0 (acceptor dipole along the O-O axis): its dipole-dipole coefficient is large, E R^3 -> -0.652 a.u.,
so p* = 3 is clean.  At the global-minimum-like tilt phi = 57 deg the dipole-dipole term happens to vanish
(1 - 3 cos 52 cos 57 ~ 0) and the curve is dipole-quadrupole, p = 4, out to ~900 A -- a trap for a p* = 3 test,
recorded here so that the real-data scan is set up at phi = 0 as well.

The synthetic truth PW ("polarisable water"): intermolecular smeared Coulomb of the point charges, an isotropic
polarisable site on each O (alpha = 9.72 a.u. = 1.44 A^3) responding self-consistently to the field of the other
molecules' charges and induced dipoles (smeared kernels, so no polarisation catastrophe at the sampled distances),
an O-O / O-H / H-H exponential repulsion and a damped O-O C6 dispersion (C6 = 45.4 a.u.).  Everything in JAX, so
the forces are the exact derivatives.  Its many-body energy is pure induction, with known answers: the dimer tail
is -C3/R^3 with C3 from the point-charge dipoles, the induction tail -C6_ind/R^6 with C6_ind = alpha (|E| from a
dipole)^2 R^6, and the non-additive cluster energy is the self-consistent dipole part.
"""
import math
import numpy as np
import jax
import jax.numpy as jnp
from .structure import Structure
from .units import ang_to_bohr, bohr_to_ang
from . import kernels as kn

R_OH = 0.9572
THETA = math.radians(104.52)
DEBYE_PER_E_ANG = 4.80320
MU_GAS = 1.855
Q_H = MU_GAS / (2.0 * R_OH * math.cos(THETA / 2) * DEBYE_PER_E_ANG)
Q_O = -2.0 * Q_H
ALPHA_O = 9.72                      # a.u.
C6_OO = 45.4                        # a.u.
PW = dict(lam=ang_to_bohr(0.5), A_OO=4000.0, b_OO=2.7, A_OH=8.0, b_OH=2.6, A_HH=4.0, b_HH=2.4, l_disp=3.0)   # R_e = 2.86 A, D_e = 4.1 kcal/mol at phi = 0


def monomer_positions():
    """(3,3) in A: O at the origin, bisector along +z, H's in the xz-plane."""
    s, c = math.sin(THETA / 2), math.cos(THETA / 2)
    return np.array([[0.0, 0.0, 0.0], [R_OH * s, 0.0, R_OH * c], [-R_OH * s, 0.0, R_OH * c]])


def monomer_charges():
    return np.array([Q_O, Q_H, Q_H])


def rotation(axis, angle):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def random_rotation(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])


def dimer_positions(R_OO_ang, phi_deg=0.0):
    """(6,3) in A: donor (O, H_bonded, H), acceptor (O, H, H) at the fixed orientation described in the module docstring."""
    s, c = math.sin(THETA), math.cos(THETA)
    donor = np.array([[0, 0, 0], [0, 0, R_OH], [R_OH * s, 0, R_OH * c]])
    phi = math.radians(phi_deg)
    b = np.array([math.sin(phi), 0.0, math.cos(phi)])
    y = np.array([0.0, 1.0, 0.0])
    Oa = np.array([0.0, 0.0, R_OO_ang])
    hs, hc = math.sin(THETA / 2), math.cos(THETA / 2)
    acceptor = np.array([Oa, Oa + R_OH * (hc * b + hs * y), Oa + R_OH * (hc * b - hs * y)])
    return np.concatenate([donor, acceptor])


def water_structure(pos_ang, energy=None, forces=None, pinned_q=True, info=None):
    n = len(pos_ang) // 3
    Z = np.tile([8, 1, 1], n)
    st = Structure(ang_to_bohr(np.asarray(pos_ang, float)), Z, energy=energy, forces=forces, info=info or {},
                   frag=np.repeat(np.arange(n), 3))
    if pinned_q:
        st.pinned = {"q": np.tile(monomer_charges(), n)}
    return st


# ---------------------------------------------------------------- the synthetic truth PW (JAX) ---------------
def _mol_index(n_mol):
    return jnp.repeat(jnp.arange(n_mol), 3)


def pw_energy(pos_bohr, n_mol, p=PW):
    """total intermolecular energy (hartree) of n_mol rigid monomers at positions (3 n_mol, 3) bohr, atoms ordered
    O, H, H per molecule.  Differentiable in the positions."""
    N = 3 * n_mol
    mol = _mol_index(n_mol)
    q = jnp.tile(jnp.asarray(monomer_charges()), n_mol)
    Z = jnp.tile(jnp.asarray([8, 1, 1]), n_mol)
    inter = (mol[:, None] != mol[None, :])
    D = pos_bohr[:, None, :] - pos_bohr[None, :, :]
    r = jnp.sqrt(jnp.sum(D * D, axis=-1) + jnp.where(inter, 0.0, 1.0))      # intramolecular pairs: dummy distance 1
    rhat = D / r[..., None]
    lam = p["lam"]
    # electrostatics (smeared), each pair once
    E_es = 0.5 * jnp.sum(jnp.where(inter, q[:, None] * q[None, :] * kn.g_long(r, lam), 0.0))
    # fields at the oxygens from the other molecules' charges: E = -q grad g = -q g'(r) rhat (rhat = (r_i - r_j)/r)
    isO = Z == 8
    dg = kn.dg_long(r, lam); d2g = kn.d2g_long(r, lam)
    Eq = -jnp.sum(jnp.where(inter[:, :, None], (q[None, :] * dg)[:, :, None] * rhat, 0.0), axis=1)       # (N,3)
    # dipole-dipole tensor between oxygens of different molecules: T = grad grad g = g'' rr + (g'/r)(1 - rr)
    T = (d2g[..., None, None] * rhat[..., :, None] * rhat[..., None, :]
         + (dg / r)[..., None, None] * (jnp.eye(3)[None, None] - rhat[..., :, None] * rhat[..., None, :]))
    T = jnp.where((inter & isO[:, None] & isO[None, :])[..., None, None], T, 0.0)
    # self-consistent induced dipoles on the oxygens: mu = alpha (Eq + T mu)  ->  (I - alpha T) mu = alpha Eq
    # build the 3n x 3n system on all atoms with alpha = 0 on hydrogens (keeps shapes static)
    alpha = jnp.where(isO, ALPHA_O, 0.0)
    A = jnp.eye(3 * N) - (alpha[:, None, None, None] * T).transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
    rhs = (alpha[:, None] * Eq).reshape(-1)
    mu = jnp.linalg.solve(A, rhs).reshape(N, 3)
    E_ind = -0.5 * jnp.sum(mu * Eq)
    # repulsion and dispersion (intermolecular)
    OO = inter & isO[:, None] & isO[None, :]
    OH = inter & (isO[:, None] != isO[None, :])
    HH = inter & ~isO[:, None] & ~isO[None, :]
    E_rep = 0.5 * jnp.sum(jnp.where(OO, p["A_OO"] * jnp.exp(-p["b_OO"] * r), 0.0)
                          + jnp.where(OH, p["A_OH"] * jnp.exp(-p["b_OH"] * r), 0.0)
                          + jnp.where(HH, p["A_HH"] * jnp.exp(-p["b_HH"] * r), 0.0))
    E_disp = -0.5 * jnp.sum(jnp.where(OO, C6_OO * kn.f6_long(r, p["l_disp"]), 0.0))
    return E_es + E_ind + E_rep + E_disp


def pw_energy_forces(pos_ang, n_mol):
    """energy (hartree) and forces (hartree/bohr) of the truth for positions in A."""
    x = jnp.asarray(ang_to_bohr(np.asarray(pos_ang, float)))
    E, g = jax.value_and_grad(lambda y: pw_energy(y, n_mol))(x)
    return float(E), -np.array(g)


def pw_components(pos_ang, n_mol):
    """the four pieces, for diagnostics."""
    return _pw_components(jnp.asarray(ang_to_bohr(np.asarray(pos_ang, float))), n_mol)


def _pw_components(x, n_mol, p=PW):
    N = 3 * n_mol; mol = _mol_index(n_mol)
    q = jnp.tile(jnp.asarray(monomer_charges()), n_mol); Z = jnp.tile(jnp.asarray([8, 1, 1]), n_mol)
    inter = (mol[:, None] != mol[None, :]); D = x[:, None, :] - x[None, :, :]
    r = jnp.sqrt(jnp.sum(D * D, axis=-1) + jnp.where(inter, 0.0, 1.0)); rhat = D / r[..., None]; lam = p["lam"]
    E_es = 0.5 * jnp.sum(jnp.where(inter, q[:, None] * q[None, :] * kn.g_long(r, lam), 0.0))
    isO = Z == 8; dg = kn.dg_long(r, lam); d2g = kn.d2g_long(r, lam)
    Eq = -jnp.sum(jnp.where(inter[:, :, None], (q[None, :] * dg)[:, :, None] * rhat, 0.0), axis=1)
    T = (d2g[..., None, None] * rhat[..., :, None] * rhat[..., None, :] + (dg / r)[..., None, None] * (jnp.eye(3)[None, None] - rhat[..., :, None] * rhat[..., None, :]))
    T = jnp.where((inter & isO[:, None] & isO[None, :])[..., None, None], T, 0.0)
    alpha = jnp.where(isO, ALPHA_O, 0.0)
    A = jnp.eye(3 * N) - (alpha[:, None, None, None] * T).transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
    mu = jnp.linalg.solve(A, (alpha[:, None] * Eq).reshape(-1)).reshape(N, 3)
    E_ind = -0.5 * jnp.sum(mu * Eq)
    E_ind1 = -0.5 * jnp.sum(alpha[:, None] * Eq * Eq)                        # first-order (pairwise-additive) induction
    OO = inter & isO[:, None] & isO[None, :]; OH = inter & (isO[:, None] != isO[None, :]); HH = inter & ~isO[:, None] & ~isO[None, :]
    E_rep = 0.5 * jnp.sum(jnp.where(OO, p["A_OO"] * jnp.exp(-p["b_OO"] * r), 0.0) + jnp.where(OH, p["A_OH"] * jnp.exp(-p["b_OH"] * r), 0.0)
                          + jnp.where(HH, p["A_HH"] * jnp.exp(-p["b_HH"] * r), 0.0))
    E_disp = -0.5 * jnp.sum(jnp.where(OO, C6_OO * kn.f6_long(r, p["l_disp"]), 0.0))
    return {k: float(v) for k, v in dict(es=E_es, ind=E_ind, ind1=E_ind1, rep=E_rep, disp=E_disp).items()} | {"mu_ind": np.array(mu)}


# ---------------------------------------------------------------- data sets --------------------------------
def dimer_scan(R_OO_ang, sigma=0.0, rng=None):
    """fixed-orientation dimers at the O-O separations given (A), with the truth's energies and forces."""
    rng = rng or np.random.default_rng(0)
    out = []
    for R in np.atleast_1d(R_OO_ang):
        pos = dimer_positions(float(R))
        E, F = pw_energy_forces(pos, 2)
        if sigma > 0: E += rng.normal() * sigma
        out.append(water_structure(pos, energy=E, forces=F, info={"R": float(ang_to_bohr(R)), "R_A": float(R)}))
    return out


def monomer_structure():
    return water_structure(monomer_positions(), energy=0.0, forces=np.zeros((3, 3)), info={"R": np.inf})


def random_cluster(n_mol, rng, radius_ang=None, min_OO=2.7, min_any=1.7, mc_steps=300, T_K=300.0):
    """n rigid monomers placed at random in a sphere and relaxed by a short rigid-body Metropolis run on the truth
    (so that the clusters are hydrogen-bonded rather than random packings).  Returns positions (3n, 3) in A."""
    radius_ang = radius_ang or (1.9 * n_mol ** (1 / 3) + 0.8)
    def place():
        Os = []
        while len(Os) < n_mol:
            x = rng.uniform(-radius_ang, radius_ang, 3)
            if np.linalg.norm(x) > radius_ang: continue
            if Os and np.min(np.linalg.norm(np.array(Os) - x, axis=1)) < min_OO: continue
            Os.append(x)
        return np.array(Os)
    def assemble(Os, Rs):
        return np.concatenate([Os[m] + monomer_positions() @ Rs[m].T for m in range(n_mol)])
    def ok(pos):
        mol = np.repeat(np.arange(n_mol), 3)
        d = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1); inter = mol[:, None] != mol[None, :]
        return np.all(d[inter] >= min_any)
    Os = place(); Rs = [random_rotation(rng) for _ in range(n_mol)]
    pos = assemble(Os, Rs)
    while not ok(pos):
        Os = place(); Rs = [random_rotation(rng) for _ in range(n_mol)]; pos = assemble(Os, Rs)
    kT = T_K * 3.166811563e-6
    E = pw_energy_forces(pos, n_mol)[0]
    for step in range(mc_steps):
        m = rng.integers(n_mol)
        Os2 = Os.copy(); Rs2 = list(Rs)
        Os2[m] = Os[m] + rng.normal(size=3) * 0.15
        Rs2[m] = rotation(rng.normal(size=3), rng.normal() * 0.25) @ Rs[m]
        if np.linalg.norm(Os2[m]) > radius_ang: continue
        pos2 = assemble(Os2, Rs2)
        if not ok(pos2): continue
        E2 = pw_energy_forces(pos2, n_mol)[0]
        if E2 < E or rng.random() < math.exp(-(E2 - E) / kT):
            Os, Rs, pos, E = Os2, Rs2, pos2, E2
    return pos


def cluster_dataset(sizes, n_per_size, seed=0, sigma=0.0, **kw):
    """clusters with the truth's interaction energies and forces (monomers at zero)."""
    rng = np.random.default_rng(seed); out = []
    for n in sizes:
        for k in range(n_per_size):
            pos = random_cluster(n, rng, **kw)
            E, F = pw_energy_forces(pos, n)
            if sigma > 0: E += rng.normal() * sigma
            out.append(water_structure(pos, energy=E, forces=F, info={"n": n, "k": k}))
    return out


def probe_configurations(n_core, distances_ang, n_per_distance, seed=0, mc_steps=300, n_cores=1):
    """relaxed core clusters of n_core molecules (n_cores different ones) plus one probe monomer at a distance d (A, O
    to the core's centre of mass) in a random direction and orientation: the far-field induction test.  Beyond
    E_0's cutoff the probe's interaction with the core is electrostatics (which pinned charges give exactly), the
    interaction of the core's environment-induced dipoles with the probe (environment-dependent sources), and the
    responses to the far field (the band-field read-out).  Returns (structures, core_energies) with the core's own
    interaction energy so that E - E_core is the probe's interaction."""
    rng = np.random.default_rng(seed); out = []; cores = []
    for c in range(n_cores):
        core = random_cluster(n_core, rng, mc_steps=mc_steps)
        com = core[0::3].mean(axis=0)
        E_core = pw_energy_forces(core, n_core)[0]
        for d in distances_ang:
            for k in range(n_per_distance):
                u = rng.normal(size=3); u /= np.linalg.norm(u)
                probe = com + d * u + monomer_positions() @ random_rotation(rng).T
                pos = np.concatenate([core, probe])
                E, F = pw_energy_forces(pos, n_core + 1)
                out.append(water_structure(pos, energy=E, forces=F, info={"d": float(d), "k": k, "core": c, "E_core": E_core}))
                cores.append(E_core)
    return out, np.array(cores)


def pair_additive_energy(pos_ang, n_mol):
    """sum over molecule pairs of the truth's dimer energies (for the non-additive energy E - E_pairs)."""
    pos = np.asarray(pos_ang, float); tot = 0.0
    for a in range(n_mol):
        for b in range(a + 1, n_mol):
            p = np.concatenate([pos[3 * a:3 * a + 3], pos[3 * b:3 * b + 3]])
            tot += pw_energy_forces(p, 2)[0]
    return tot


# ---------------------------------------------------------------- real data ---------------------------------
def read_extxyz(path, energy_key="energy", forces_key="forces", energy_unit="hartree", length_unit="angstrom"):
    """water clusters from an extended-XYZ file: frames of 3n atoms ordered O, H, H per molecule, a comment line
    with `energy=...` (interaction energy) and optional per-atom forces columns 5-7 (`forces` property).
    Units converted to hartree / bohr; fragments assigned per triple.  The monomer's point charges are pinned."""
    import ase.io  # noqa: F401 -- only if ASE is available
    frames = ase.io.read(path, index=":")
    eu = {"hartree": 1.0, "ev": 1.0 / 27.211386245988, "kcal/mol": 1.0 / 627.5094740631}[energy_unit.lower()]
    out = []
    for fr in frames:
        n = len(fr) // 3
        E = float(fr.info[energy_key]) * eu if energy_key in fr.info else None
        F = fr.arrays[forces_key] * eu * bohr_to_ang(1.0) if forces_key in fr.arrays else None    # per A -> per bohr
        out.append(water_structure(fr.get_positions(), energy=E, forces=F, info=dict(fr.info)))
    return out
