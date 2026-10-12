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


# a sodium-like cation for the band-field response test: a non-polarisable unit charge with exponential repulsion
# to O and H (tuned to R_e(ion-O) = 2.25 A, D_e = 24 kcal/mol for the ion-water dimer, see ion_dimer_scan)
ION = dict(Q=1.0, Z=11, A_O=500.0, b_O=2.4, A_H=30.0, b_H=2.5)      # R_e(ion-O) = 2.39 A, D_e = 21.5 kcal/mol


def _site_arrays(n_mol, n_ion=0, p=PW, ion=ION):
    """species-resolved arrays of the truth's sites: fragment index, charges, polarisabilities, pair repulsion
    A exp(-b r) and C6 matrices; water atoms ordered O, H, H per molecule, then the ions (one fragment each)."""
    N = 3 * n_mol + n_ion
    mol = jnp.concatenate([_mol_index(n_mol), n_mol + jnp.arange(n_ion)])
    q = jnp.concatenate([jnp.tile(jnp.asarray(monomer_charges()), n_mol), jnp.full((n_ion,), ion["Q"])])
    Z = jnp.concatenate([jnp.tile(jnp.asarray([8, 1, 1]), n_mol), jnp.full((n_ion,), ion["Z"])])
    isO = Z == 8; isH = Z == 1; isI = Z == ion["Z"]
    alpha = jnp.where(isO, ALPHA_O, 0.0)
    def pair(ma, mb): return (ma[:, None] & mb[None, :]) | (mb[:, None] & ma[None, :])
    A = (jnp.where(pair(isO, isO), p["A_OO"], 0.0) + jnp.where(pair(isO, isH), p["A_OH"], 0.0) + jnp.where(pair(isH, isH), p["A_HH"], 0.0)
         + jnp.where(pair(isI, isO), ion["A_O"], 0.0) + jnp.where(pair(isI, isH), ion["A_H"], 0.0))
    b = (jnp.where(pair(isO, isO), p["b_OO"], 0.0) + jnp.where(pair(isO, isH), p["b_OH"], 0.0) + jnp.where(pair(isH, isH), p["b_HH"], 0.0)
         + jnp.where(pair(isI, isO), ion["b_O"], 0.0) + jnp.where(pair(isI, isH), ion["b_H"], 0.0))
    c6 = jnp.where(pair(isO, isO), C6_OO, 0.0)
    return mol, q, alpha, A, b, c6


def _pw_general(x, mol, q, alpha, A, b, c6, lam, l_disp, components=False):
    """intermolecular energy of charged, polarisable point sites in fragments: smeared Coulomb between sites of
    different fragments, self-consistent induced dipoles on the sites with alpha > 0, pairwise exponential
    repulsion and damped C6 dispersion on fragment-external pairs.  Differentiable in x."""
    N = x.shape[0]
    inter = (mol[:, None] != mol[None, :])
    D = x[:, None, :] - x[None, :, :]
    r = jnp.sqrt(jnp.sum(D * D, axis=-1) + jnp.where(inter, 0.0, 1.0))      # intrafragment pairs: dummy distance 1
    rhat = D / r[..., None]
    E_es = 0.5 * jnp.sum(jnp.where(inter, q[:, None] * q[None, :] * kn.g_long(r, lam), 0.0))
    dg = kn.dg_long(r, lam); d2g = kn.d2g_long(r, lam)
    Eq = -jnp.sum(jnp.where(inter[:, :, None], (q[None, :] * dg)[:, :, None] * rhat, 0.0), axis=1)       # (N,3)
    T = (d2g[..., None, None] * rhat[..., :, None] * rhat[..., None, :]
         + (dg / r)[..., None, None] * (jnp.eye(3)[None, None] - rhat[..., :, None] * rhat[..., None, :]))
    pol = alpha > 0
    T = jnp.where((inter & pol[:, None] & pol[None, :])[..., None, None], T, 0.0)
    Amat = jnp.eye(3 * N) - (alpha[:, None, None, None] * T).transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
    mu = jnp.linalg.solve(Amat, (alpha[:, None] * Eq).reshape(-1)).reshape(N, 3)
    E_ind = -0.5 * jnp.sum(mu * Eq)
    E_rep = 0.5 * jnp.sum(jnp.where(inter, A * jnp.exp(-b * r), 0.0))
    E_disp = -0.5 * jnp.sum(jnp.where(inter, c6 * kn.f6_long(r, l_disp), 0.0))
    if components:
        E_ind1 = -0.5 * jnp.sum(alpha[:, None] * Eq * Eq)
        return dict(es=E_es, ind=E_ind, ind1=E_ind1, rep=E_rep, disp=E_disp, mu_ind=mu, Eq=Eq)
    return E_es + E_ind + E_rep + E_disp


def pw_energy(pos_bohr, n_mol, p=PW, n_ion=0):
    """total intermolecular energy (hartree) of n_mol rigid monomers (atoms O, H, H per molecule) [+ n_ion ions,
    last rows] at positions (3 n_mol + n_ion, 3) bohr.  Differentiable in the positions."""
    mol, q, alpha, A, b, c6 = _site_arrays(n_mol, n_ion, p)
    return _pw_general(pos_bohr, mol, q, alpha, A, b, c6, p["lam"], p["l_disp"])


def pw_energy_forces(pos_ang, n_mol, n_ion=0):
    """energy (hartree) and forces (hartree/bohr) of the truth for positions in A."""
    x = jnp.asarray(ang_to_bohr(np.asarray(pos_ang, float)))
    E, g = jax.value_and_grad(lambda y: pw_energy(y, n_mol, n_ion=n_ion))(x)
    return float(E), -np.array(g)


def pw_components(pos_ang, n_mol, n_ion=0):
    """the pieces (es, ind, ind1, rep, disp) and the induced dipoles / charge fields, for diagnostics."""
    mol, q, alpha, A, b, c6 = _site_arrays(n_mol, n_ion)
    c = _pw_general(jnp.asarray(ang_to_bohr(np.asarray(pos_ang, float))), mol, q, alpha, A, b, c6, PW["lam"], PW["l_disp"], components=True)
    return {k: (float(v) if v.ndim == 0 else np.array(v)) for k, v in c.items()}


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
    if n_mol > 1 and 2 * radius_ang <= min_OO:
        raise ValueError("cluster radius must exceed half the minimum oxygen separation")
    def place():
        Os = []
        attempts = 0
        while len(Os) < n_mol:
            attempts += 1
            if attempts > 10000 * n_mol:
                raise ValueError("could not place water molecules; increase radius_ang or reduce min_OO")
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
    attempts = 0
    while not ok(pos):
        attempts += 1
        if attempts > 1000:
            raise ValueError("could not generate a non-overlapping water cluster; increase radius_ang or reduce min_any")
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


# ---------------------------------------------------------------- the ion probe (band-field response test) -----
def ion_structure(pos_ang, n_mol, energy=None, forces=None, info=None):
    """n_mol waters (O, H, H each) followed by one ion: fragments per molecule and one for the ion, each with its own
    total charge (the waters neutral, the ion Q; learned charges are shifted to these fragment by fragment); pinned
    charges are the truth's (the monomer charges and the ion's Q)."""
    Z = np.concatenate([np.tile([8, 1, 1], n_mol), [ION["Z"]]])
    frag = np.concatenate([np.repeat(np.arange(n_mol), 3), [n_mol]])
    st = Structure(ang_to_bohr(np.asarray(pos_ang, float)), Z, energy=energy, forces=forces, info=info or {}, frag=frag,
                   total_charge=ION["Q"], frag_charges=np.concatenate([np.zeros(n_mol), [ION["Q"]]]))
    st.pinned = {"q": np.concatenate([np.tile(monomer_charges(), n_mol), [ION["Q"]]])}
    return st


def ion_monomer_scan(distances_ang, n_orient, seed=0):
    """the ion at distance R from the oxygen of a monomer, in random directions with random monomer orientations:
    teaches E_0 the ion-water short range.  Returns structures with the truth's energy and forces."""
    rng = np.random.default_rng(seed); out = []
    for R in distances_ang:
        for k in range(n_orient):
            u = rng.normal(size=3); u /= np.linalg.norm(u)
            pos = np.vstack([monomer_positions() @ random_rotation(rng).T, R * u])
            E, F = pw_energy_forces(pos, 1, n_ion=1)
            out.append(ion_structure(pos, 1, energy=E, forces=F, info={"R": float(R), "k": k}))
    return out


def ion_configurations(n_core, distances_ang, n_per_distance, seed=0, mc_steps=300, n_cores=1):
    """relaxed water cores plus the ion at distance d (A) from the core's centre of mass in a random direction:
    the band-field response test.  Beyond E_0's cutoff the ion's interaction with the core is the electrostatics of
    the core's charges and embedded induced dipoles with the ion (exact with pinned embedded sources) plus the
    core's response to the ion's field -- the term the band-field read-out has to carry.  Returns (structures,
    core_energies)."""
    rng = np.random.default_rng(seed); out = []; cores = []
    for c in range(n_cores):
        core = random_cluster(n_core, rng, mc_steps=mc_steps)
        com = core[0::3].mean(axis=0)
        E_core = pw_energy_forces(core, n_core)[0]
        for d in distances_ang:
            for k in range(n_per_distance):
                u = rng.normal(size=3); u /= np.linalg.norm(u)
                pos = np.vstack([core, com + d * u])
                E, F = pw_energy_forces(pos, n_core, n_ion=1)
                out.append(ion_structure(pos, n_core, energy=E, forces=F, info={"d": float(d), "k": k, "core": c, "E_core": E_core}))
                cores.append(E_core)
    return out, np.array(cores)


def ion_response_decomposition(st):
    """for an ion configuration: the ion's interaction with the core, its point-charge electrostatics + repulsion,
    piece (2) = the core's embedded induced dipoles (core alone) against the ion's field, and the remainder -- the
    core's self-consistent response to the ion, the band-field response.  All in hartree."""
    pos = bohr_to_ang(st.positions); n_mol = (len(pos) - 1) // 3
    c_all = pw_components(pos, n_mol, n_ion=1); c_core = pw_components(pos[:-1], n_mol)
    y = st.energy - st.info["E_core"]
    es = (c_all["es"] - c_core["es"]) + (c_all["rep"] - c_core["rep"]) + (c_all["disp"] - c_core["disp"])
    # the ion's smeared field at the core's atoms
    x = st.positions; lam = PW["lam"]; Q = ION["Q"]
    D = x[:-1] - x[-1]; r = np.linalg.norm(D, axis=1)
    E_ion = -Q * np.array(kn.dg_long(jnp.asarray(r), lam))[:, None] * D / r[:, None]
    piece2 = -np.sum(c_core["mu_ind"] * E_ion)
    return dict(y=y, es=es, piece2=piece2, response=y - es - piece2)


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
