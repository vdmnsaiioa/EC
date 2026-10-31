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
import os
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


def water_structure(pos_ang, energy=None, forces=None, pinned_q=True, info=None, charges=None):
    """n waters (O, H, H each) with fragments per molecule; pinned charges: `charges` (3n,) when given (a truth's own,
    e.g. q-AQUA's LTP2011 charges at each molecule's geometry), else the PW truth's monomer charges."""
    n = len(pos_ang) // 3
    Z = np.tile([8, 1, 1], n)
    st = Structure(ang_to_bohr(np.asarray(pos_ang, float)), Z, energy=energy, forces=forces, info=info or {},
                   frag=np.repeat(np.arange(n), 3))
    if pinned_q:
        st.pinned = {"q": np.tile(monomer_charges(), n) if charges is None else np.asarray(charges, float).copy()}
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


# ---------------------------------------------------------------- truths --------------------------------------
class PWTruth:
    """the synthetic polarisable water truth of the rehearsal, the default of every generator below (its calls are
    exactly the ones the generators always made, so that every PW data set is bit-for-bit what it was)."""
    name = "pw"
    r_e = 2.86                     # A: the dimer minimum at phi = 0 as recorded for the rehearsal (rounded, and kept so)

    def energy(self, pos_ang, n_mol):
        return pw_energy_forces(pos_ang, n_mol)[0]

    def energy_forces(self, pos_ang, n_mol):
        return pw_energy_forces(pos_ang, n_mol)

    def molecule_charges(self, pos_ang):
        return np.tile(monomer_charges(), len(pos_ang) // 3)

    def describe(self):
        return {"name": self.name, "r_e_angstrom": self.r_e}


PW_TRUTH = PWTruth()


class QAquaTruth:
    """q-AQUA v1 (sseft.qaqua; external/qaqua/build.sh builds it): interaction energies (its 1-body term removed) and
    forces of rigid or flexible monomers, and the monomers' LTP2011 charges -- the charges its 2-body term reduces
    to beyond 7.8 A -- for the pinned-charge rungs.  r_e is the rigid dimer's minimum at phi = 0, computed."""
    name = "qaqua"

    def __init__(self, root):
        from .qaqua import QAqua
        self.qa = QAqua(root)
        self.r_e = dimer_minimum(self)

    def energy(self, pos_ang, n_mol):
        return self.qa.interaction_energy(ang_to_bohr(np.asarray(pos_ang, float)))

    def energy_forces(self, pos_ang, n_mol):
        return self.qa.interaction_energy_forces(ang_to_bohr(np.asarray(pos_ang, float)))

    def molecule_charges(self, pos_ang):
        return self.qa.molecule_charges(ang_to_bohr(np.asarray(pos_ang, float)))

    def describe(self):
        return {"name": self.name, "root": self.qa.root, "commit": _git_commit(os.path.join(self.qa.root, "q-AQUA")),
                "r_e_angstrom": self.r_e}


class QAquaPolTruth(QAquaTruth):
    """q-AQUA-pol (sseft.qaqua.QAquaPol; external/qaqua/build_pol.sh builds it): interaction energies (its
    Partridge-Schwenke 1-body term removed) and forces, the monomers' dipole-surface charges on O, H, H (the charges
    TTM3-F moves to its H atoms and M-site, keeping their dipole) for the pinned-charge rungs, and TTM3-F's static
    polarisability and O-O C6 (atomic units) for the pinned dispersion band.  Beyond 7 A the truth is TTM3-F:
    electrostatics, many-body induction and dispersion, at every distance."""
    name = "qaqua_pol"

    def __init__(self, root):
        from .qaqua import QAquaPol
        self.qa = QAquaPol(root)
        self.alpha, self.c6 = self.qa.ttm_constants()
        # TTM3-F's M-site, r_M = (1 - gamma) r_O + gamma/2 (r_H1 + r_H2), read back from the library on the monomer
        X = ang_to_bohr(monomer_positions()); rM = self.qa.ttm(X)["msites"][0]
        b = X[1] + X[2] - 2 * X[0]
        self.gamma_m = float(2 * np.dot(rM - X[0], b) / np.dot(b, b))
        if np.linalg.norm(rM - X[0] - 0.5 * self.gamma_m * b) > 1e-10:
            raise RuntimeError("TTM3-F's M-site is not on the bisector as expected")
        self.r_e = dimer_minimum(self)

    def sampling_energy(self, pos_ang, n_mol):
        """the library's energy-only routine, for the Metropolis relaxation of geometries only (sseft.qaqua.QAquaPol:
        off in the 3-body blend region at 5.0-5.5 A; it drove E3-QP's relaxations, so the data sets reproduce)."""
        return self.qa.interaction_energy_sampling(ang_to_bohr(np.asarray(pos_ang, float)))

    def site_model(self, pos_ang):
        """TTM3-F's permanent charges at the molecules' geometry -- the truth's own permanent sources, exactly: the H
        charges on the H atoms (none on O) and one virtual site per molecule, the M-site, with its charge.  Returns
        (q_site (3n,), vsite_w (n, 3n): each M-site as a combination of its molecule's atoms, vsite_q (n,), vsite_frag (n,))."""
        pos = np.asarray(pos_ang, float); n = len(pos) // 3
        qs = self.qa.ttm(ang_to_bohr(pos))["site_charges"]                    # (n, 3): H1, H2, M
        q_site = np.zeros(3 * n); w = np.zeros((n, 3 * n))
        for m in range(n):
            q_site[3 * m + 1:3 * m + 3] = qs[m, :2]
            w[m, 3 * m:3 * m + 3] = [1 - self.gamma_m, 0.5 * self.gamma_m, 0.5 * self.gamma_m]
        return q_site, w, qs[:, 2].copy(), np.arange(n)

    def describe(self):
        return {"name": self.name, "root": self.qa.root, "commit": _git_commit(os.path.join(self.qa.root, "q-AQUA-pol")),
                "data_commit": _git_commit(os.path.join(self.qa.root, "q-AQUA")), "r_e_angstrom": self.r_e,
                "alpha_bohr3": self.alpha, "c6_oo": self.c6, "gamma_m": self.gamma_m}


class MBPolTruth:
    """MB-pol through MBX (sseft.mbpol; external/mbx/build.sh builds it), E4's truth (notes/e4-plan.md section 2).

    It gives interaction energies (the 1-body polynomial off) and forces of clusters and periodic cells (cell in A,
    rows = lattice vectors, keyword `cell`), in hartree and hartree/bohr, with E4's settings (sseft.mbpol.DEFAULTS).
    It also gives the site model.
      - Charges: on the H atoms and on an M-site at r_O + gamma/2 (r_H1 + r_H2 - 2 r_O), none on O.
      - Point polarisabilities on O and H.
    All of these are read from MBX on the rigid monomer and are the same for every rigid molecule."""
    name = "mbpol"

    def __init__(self, root, settings=None):
        from .mbpol import MBPol
        self.mb = MBPol(root, settings=settings)
        m = monomer_positions()
        s = self.mb.sites(m)
        b = m[1] + m[2] - 2 * m[0]
        self.gamma_m = float(2 * np.dot(s["xyz"][3] - m[0], b) / np.dot(b, b))
        if np.linalg.norm(s["xyz"][3] - m[0] - 0.5 * self.gamma_m * b) > 1e-10:
            raise RuntimeError("MB-pol's M-site is not on the bisector as expected")
        self.site_charges = s["charges"].copy()                  # O, H, H, M (e)
        self.site_alpha = s["polarizabilities"].copy()           # O, H, H, M (A^3)
        self.r_e = None

    def energy(self, pos_ang, n_mol=None, cell=None):
        return self.mb.energy(pos_ang, cell)

    def energy_forces(self, pos_ang, n_mol=None, cell=None):
        return self.mb.energy_forces(pos_ang, cell)

    def site_model(self, pos_ang):
        """the rigid monomer's site charges at every molecule of pos_ang (3n, 3): (q_site (3n,) on the atoms, vsite_w
        (n, 3n) each M-site as a combination of its molecule's atoms, vsite_q (n,), vsite_frag (n,))."""
        n = len(pos_ang) // 3
        q_site = np.tile(self.site_charges[:3], n)
        w = np.zeros((n, 3 * n))
        for k in range(n):
            w[k, 3 * k:3 * k + 3] = [1 - self.gamma_m, 0.5 * self.gamma_m, 0.5 * self.gamma_m]
        return q_site, w, np.full(n, float(self.site_charges[3])), np.arange(n)

    def describe(self):
        return {"name": self.name, **self.mb.describe(), "gamma_m": self.gamma_m,
                "site_charges": self.site_charges.tolist(), "site_alpha": self.site_alpha.tolist()}


# MB-pol's long-range dispersion: beyond MBX's real-space cutoff it is c6_i c6_j / r^6 by particle-mesh Ewald, with
# per-site c6 in (kcal/mol)^1/2 A^3 (MBX v1.4.0, src/bblock/sys_tools.cpp; C6_OO = c6_O^2 = 237.32 kcal/mol A^6)
MBPOL_C6_LR = {8: 15.40523357222455098728, 1: 4.48258697649551357815}


def mbpol_dispersion_pins(numbers, K=8, omega0=0.3, w_osc=0.5):
    """alpha_i(i w_k) (N, K) on the rung's Casimir-Polder grid (kernels.casimir_polder_grid(K, omega0)) whose
    C6_ij = (3/pi) sum_k w_k alpha_i alpha_j is MB-pol's long-range c6_i c6_j, rank one, on O and H.  A common oscillator
    shape 1 / (1 + (w / w_osc)^2) is normalised so that (3/pi) sum_k w_k phi_k^2 = 1 and scaled by c6_i in atomic units.
    With these pins the dispersion band beyond the truth's real-space cutoff (6.5 A) is the truth's (e4-plan.md section
    2).  The static value is not MB-pol's polarisability: the pins carry the dispersion band only."""
    from .units import KCALMOL_PER_HARTREE, BOHR_PER_ANGSTROM
    om, w = (np.asarray(x) for x in kn.casimir_polder_grid(K, omega0))
    phi = 1.0 / (1.0 + (om / w_osc) ** 2)
    phi = phi / math.sqrt(3.0 / math.pi * float(np.sum(w * phi ** 2)))
    unit = math.sqrt(1.0 / KCALMOL_PER_HARTREE) * BOHR_PER_ANGSTROM ** 3
    return np.array([MBPOL_C6_LR[int(z)] * unit * phi for z in numbers])


def e4_structures(path, pins=False, K=8, omega0=0.3):
    """the frames of an E4 data file (scripts/e4_data.py: a trajectory or an assembled set) as harness Structures.
    Each carries positions in bohr, the cell in bohr with pbc, the interaction energy in hartree and forces in
    hartree/bohr, one fragment per molecule, and info (n_mol, seed, time, the frame's source).

    pins=True also pins the file's site model (q_site, vsite_w, vsite_q, vsite_frag), as pin_site_model does for
    q-AQUA-pol, and the dispersion band's alpha (mbpol_dispersion_pins on the grid of K and omega0, the rung's). The
    virtual-site matrix is dense, (n, 3n) per frame, which is fine for the training cells and costly for 1024
    molecules."""
    d = np.load(path, allow_pickle=False)
    pos, cell = d["positions"], d["cell"]
    n_frames, n_atoms = pos.shape[0], pos.shape[1]
    n = n_atoms // 3
    cells = cell if cell.ndim == 3 else np.broadcast_to(cell, (n_frames, 3, 3))
    src = d["source"] if "source" in d.files else np.zeros(n_frames, int)
    t = d["time_ps"] if "time_ps" in d.files else np.zeros(n_frames)
    gamma, sq = float(d["site_gamma"]), d["site_q"]
    alpha = mbpol_dispersion_pins(np.tile([8, 1, 1], n), K, omega0) if pins else None
    out = []
    for f in range(n_frames):
        st = water_structure(pos[f], energy=float(d["energy"][f]), forces=np.asarray(d["forces"][f], float),
                             pinned_q=False, info={"n_mol": n, "source": int(src[f]), "time_ps": float(t[f])})
        st.cell = ang_to_bohr(np.asarray(cells[f], float))
        st.pbc = True
        if pins:
            w = np.zeros((n, 3 * n))
            for k in range(n):
                w[k, 3 * k:3 * k + 3] = [1 - gamma, 0.5 * gamma, 0.5 * gamma]
            st.pinned = {"q_site": np.tile(sq[:3], n), "vsite_w": w, "vsite_q": np.full(n, float(sq[3])),
                         "vsite_frag": np.arange(n), "alpha": alpha}
        out.append(st)
    return out


def _git_commit(repo):
    """the short commit of a clone (read from .git, without git), or None."""
    try:
        ref = open(os.path.join(repo, ".git", "HEAD")).read().strip()
        path = os.path.join(repo, ".git", ref[5:]) if ref.startswith("ref: ") else None
        if path and os.path.exists(path):
            return open(path).read().strip()[:7]
        if path:                                          # a packed ref (a fresh clone)
            for line in open(os.path.join(repo, ".git", "packed-refs")):
                if line.strip().endswith(ref[5:]):
                    return line.split()[0][:7]
        return ref[:7]
    except OSError:
        return None


_QAQUA = {}


def qaqua_truth(root):
    """one QAquaTruth per build directory and process (the library keeps module state)."""
    key = os.path.abspath(os.path.expanduser(root))
    if key not in _QAQUA:
        _QAQUA[key] = QAquaTruth(key)
    return _QAQUA[key]


def qaqua_pol_truth(root):
    """one QAquaPolTruth per build directory and process."""
    key = ("pol", os.path.abspath(os.path.expanduser(root)))
    if key not in _QAQUA:
        _QAQUA[key] = QAquaPolTruth(key[1])
    return _QAQUA[key]


def mbpol_truth(root):
    """one MBPolTruth per MBX build and process (MBX keeps one system)."""
    key = ("mbpol", os.path.abspath(os.path.expanduser(root)))
    if key not in _QAQUA:
        _QAQUA[key] = MBPolTruth(key[1])
    return _QAQUA[key]


def dimer_minimum(truth, phi_deg=0.0, bracket=(2.5, 3.5), tol=1e-7):
    """the O-O distance (A) minimising the truth's rigid fixed-orientation dimer energy: golden section on `bracket`."""
    f = lambda R: truth.energy(dimer_positions(R, phi_deg), 2)
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = bracket
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc; c = b - g * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd; d = a + g * (b - a); fd = f(d)
    return 0.5 * (a + b)


def molecular_dipoles(truth, pos_ang):
    """(n, 3) dipole vectors (e bohr) of the truth's charges on each molecule (neutral, so origin-free)."""
    pos = ang_to_bohr(np.asarray(pos_ang, float)).reshape(-1, 3, 3)
    q = np.asarray(truth.molecule_charges(pos_ang)).reshape(-1, 3)
    return np.einsum("mi,mia->ma", q, pos)


def dipole_dipole_coefficient(truth, phi_deg=0.0):
    """lim E R^3 (hartree bohr^3) of the fixed-orientation dimer for point dipoles of the truth's charges:
    mu_d.mu_a - 3 (mu_d.z)(mu_a.z)."""
    mu = molecular_dipoles(truth, dimer_positions(10.0, phi_deg))
    return float(mu[0] @ mu[1] - 3.0 * mu[0, 2] * mu[1, 2])


# ---------------------------------------------------------------- data sets --------------------------------
def dimer_scan(R_OO_ang, sigma=0.0, rng=None, truth=None, phi_deg=0.0):
    """fixed-orientation dimers at the O-O separations given (A), with the truth's energies and forces (PW unless a
    truth is given) and its charges pinned."""
    truth = truth or PW_TRUTH
    rng = rng or np.random.default_rng(0)
    out = []
    for R in np.atleast_1d(R_OO_ang):
        pos = dimer_positions(float(R), phi_deg)
        E, F = truth.energy_forces(pos, 2)
        if sigma > 0: E += rng.normal() * sigma
        out.append(water_structure(pos, energy=E, forces=F, info={"R": float(ang_to_bohr(R)), "R_A": float(R)},
                                   charges=truth.molecule_charges(pos)))
    return out


def monomer_structure(truth=None):
    """an isolated rigid monomer: interaction energy and forces exactly zero for any truth; the truth's charges pinned."""
    truth = truth or PW_TRUTH
    pos = monomer_positions()
    return water_structure(pos, energy=0.0, forces=np.zeros((3, 3)), info={"R": np.inf},
                           charges=truth.molecule_charges(pos))


def random_cluster(n_mol, rng, radius_ang=None, min_OO=2.7, min_any=1.7, mc_steps=300, T_K=300.0, energy=None):
    """n rigid monomers placed at random in a sphere and relaxed by a short rigid-body Metropolis run on the truth
    (so that the clusters are hydrogen-bonded rather than random packings).  energy(pos_ang, n_mol): the truth's
    energy (default the PW truth's).  Returns positions (3n, 3) in A."""
    energy = energy or PW_TRUTH.energy
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
    E = energy(pos, n_mol)
    for step in range(mc_steps):
        m = rng.integers(n_mol)
        Os2 = Os.copy(); Rs2 = list(Rs)
        Os2[m] = Os[m] + rng.normal(size=3) * 0.15
        Rs2[m] = rotation(rng.normal(size=3), rng.normal() * 0.25) @ Rs[m]
        if np.linalg.norm(Os2[m]) > radius_ang: continue
        pos2 = assemble(Os2, Rs2)
        if not ok(pos2): continue
        E2 = energy(pos2, n_mol)
        if E2 < E or rng.random() < math.exp(-(E2 - E) / kT):
            Os, Rs, pos, E = Os2, Rs2, pos2, E2
    return pos


def cluster_dataset(sizes, n_per_size, seed=0, sigma=0.0, truth=None, **kw):
    """clusters relaxed on the truth (PW unless given) with its interaction energies and forces (monomers at zero)
    and its charges pinned."""
    truth = truth or PW_TRUTH
    rng = np.random.default_rng(seed); out = []
    for n in sizes:
        for k in range(n_per_size):
            pos = random_cluster(n, rng, energy=getattr(truth, "sampling_energy", truth.energy), **kw)
            E, F = truth.energy_forces(pos, n)
            if sigma > 0: E += rng.normal() * sigma
            out.append(water_structure(pos, energy=E, forces=F, info={"n": n, "k": k}, charges=truth.molecule_charges(pos)))
    return out


def probe_configurations(n_core, distances_ang, n_per_distance, seed=0, mc_steps=300, n_cores=1, truth=None, **kw):
    """relaxed core clusters of n_core molecules (n_cores different ones) plus one probe monomer at a distance d (A, O
    to the core's centre of mass) in a random direction and orientation: the far-field induction test.  Beyond
    E_0's cutoff the probe's interaction with the core is electrostatics (which pinned charges give exactly), the
    interaction of the core's environment-induced dipoles with the probe (environment-dependent sources), and the
    responses to the far field (the band-field read-out).  Returns (structures, core_energies) with the core's own
    interaction energy so that E - E_core is the probe's interaction.  truth: PW unless given (the same calls as
    before for PW); kw: random_cluster's geometry settings for the cores."""
    truth = truth or PW_TRUTH
    rng = np.random.default_rng(seed); out = []; cores = []
    for c in range(n_cores):
        core = random_cluster(n_core, rng, mc_steps=mc_steps, energy=getattr(truth, "sampling_energy", truth.energy), **kw)
        com = core[0::3].mean(axis=0)
        E_core = truth.energy(core, n_core)
        for d in distances_ang:
            for k in range(n_per_distance):
                u = rng.normal(size=3); u /= np.linalg.norm(u)
                probe = com + d * u + monomer_positions() @ random_rotation(rng).T
                pos = np.concatenate([core, probe])
                E, F = truth.energy_forces(pos, n_core + 1)
                out.append(water_structure(pos, energy=E, forces=F, info={"d": float(d), "k": k, "core": c, "E_core": E_core},
                                           charges=truth.molecule_charges(pos)))
                cores.append(E_core)
    return out, np.array(cores)


def contact_configurations(n_mol, count, oo_range_ang=(2.3, 3.0), seed=0, truth=None, max_pair_kcal=2.0, min_any=1.5):
    """rigid dimers (n_mol = 2) or trimers (n_mol = 3) at a hydrogen-bond contact or closer, for the training set: molecule
    1's O at an O-O distance from molecule 0's drawn uniformly from oo_range_ang (A), in a random direction; a trimer's
    molecule 2 likewise from molecule 0 or 1 (at random), no nearer than the range's lower end to the other; every
    orientation uniform; kept when every pair's truth 2-body energy is at most max_pair_kcal (the raw CCSD(T) sets'
    training domain; None: no cap) and no two atoms of different molecules are nearer than min_any (A).  Labels: the
    truth's interaction energies and forces, its charges pinned.  The Metropolis-relaxed clusters rarely visit the
    contact -- 1.5 % of their pairs below O-O 2.7 A and no two atoms of different molecules nearer than 1.7 A, against
    5.7 % of the raw dimers, whose closest atoms come to 1.53 A (e3qp2-results.md section 3.5)."""
    if n_mol not in (2, 3):
        raise ValueError("contact configurations are dimers (n_mol 2) or trimers (n_mol 3)")
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("count must be an integer >= 0")
    lo, hi = (float(v) for v in oo_range_ang)
    if not (0 < lo <= hi and math.isfinite(hi)):
        raise ValueError("oo_range_ang must be (lower, upper) with 0 < lower <= upper")
    if not (math.isfinite(min_any) and min_any >= 0):
        raise ValueError("min_any must be finite and >= 0")
    truth = truth or PW_TRUTH
    rng = np.random.default_rng(seed)
    mono = monomer_positions()
    mol = np.repeat(np.arange(n_mol), 3); inter = mol[:, None] != mol[None, :]
    direction = lambda: (lambda u: u / np.linalg.norm(u))(rng.normal(size=3))
    out, tries = [], 0
    while len(out) < count:
        tries += 1
        if tries > 100000 + 1000 * count:
            raise ValueError("could not place contact configurations within the caps; widen oo_range_ang, raise "
                             "max_pair_kcal or lower min_any")
        Os = [np.zeros(3)]
        Os.append(Os[0] + rng.uniform(lo, hi) * direction())
        if n_mol == 3:
            k = int(rng.integers(2))
            O2 = Os[k] + rng.uniform(lo, hi) * direction()
            if np.linalg.norm(O2 - Os[1 - k]) < lo:
                continue
            Os.append(O2)
        pos = np.concatenate([O + mono @ random_rotation(rng).T for O in Os])
        if np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)[inter].min() < min_any:
            continue
        if max_pair_kcal is not None:
            pairs = [np.concatenate([pos[3 * i:3 * i + 3], pos[3 * j:3 * j + 3]]) for i in range(n_mol) for j in range(i + 1, n_mol)]
            if max(truth.energy(p, 2) for p in pairs) * 627.5094740631 > max_pair_kcal:
                continue
        E, F = truth.energy_forces(pos, n_mol)
        out.append(water_structure(pos, energy=E, forces=F, info={"n": n_mol, "k": len(out), "contact": True},
                                   charges=truth.molecule_charges(pos)))
    return out


def rigid_molecule_charges(truth, n_mol):
    """the truth's pinned charges (O, H, H) of the rigid monomer on n_mol molecules: the charges a model trained on rigid
    monomers carries, whatever the geometry it is read at (e3qp2-results.md section 3.4)."""
    return np.tile((truth or PW_TRUTH).molecule_charges(monomer_positions()), n_mol)


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


def pin_site_model(structs, truth, rigid=False):
    """pin a truth's site model (truth.site_model: q-AQUA-pol's TTM3-F charges on the H atoms and the M-sites) on water
    structures, for the rungs whose sources are the truth's permanent charges exactly (Rung.msites).  rigid: every
    molecule carries the rigid monomer's site charges, at its own sites -- the pins of a model trained on rigid monomers,
    read on near-rigid frames (the charges recomputed at each frame's geometry carry the flux of the truth's charges,
    which E_0 learned from the internal forces: counted twice, e3qp2-results.md sections 3.4-3.5)."""
    if rigid:
        q_r, _, vq_r, _ = truth.site_model(monomer_positions())
    for s in structs:
        q, w, vq, vf = truth.site_model(bohr_to_ang(s.positions))
        if rigid:
            n = len(vq); q = np.tile(q_r, n); vq = np.full(n, float(vq_r[0]))
        s.pinned = {**s.pinned, "q_site": q, "vsite_w": w, "vsite_q": vq, "vsite_frag": vf}
    return structs


def pair_additive_energy(pos_ang, n_mol, truth=None):
    """sum over molecule pairs of the truth's dimer energies (for the non-additive energy E - E_pairs)."""
    truth = truth or PW_TRUTH
    pos = np.asarray(pos_ang, float); tot = 0.0
    for a in range(n_mol):
        for b in range(a + 1, n_mol):
            p = np.concatenate([pos[3 * a:3 * a + 3], pos[3 * b:3 * b + 3]])
            tot += truth.energy(p, 2)
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
