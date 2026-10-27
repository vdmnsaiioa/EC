"""
MB-pol through MBX as the truth for E4 (notes/e4-plan.md). MB-pol is the many-body water potential of V. Babin,
C. Leforestier and F. Paesani (J. Chem. Theory Comput. 9, 5395 (2013); 10, 1599 and 2906 (2014)). MBX v1.4.0
evaluates it (M. Riera et al., J. Chem. Phys. 159, 054802 (2023); github.com/paesanilab/MBX). We call MBX through the
C entry points of its shared library (src/bblock/external_call.cpp), built by external/mbx/build.sh.

What the truth is:
  - Electrostatics as in TTM4-F. Geometry-dependent charges sit on the H atoms and on an M-site on the bisector; they
    come from the Partridge-Schwenke dipole surface.
  - Point polarisabilities on O and H, Thole-damped. The induced dipoles are iterated to self-consistency (conjugate
    gradient, to a tolerance set below).
  - 2-body and 3-body permutationally invariant polynomials fitted to CCSD(T), switched off inside 6.5 and 4.5 A.
  - A damped C6 dispersion.
  - Periodic cells use particle-mesh Ewald for electrostatics and dispersion, with tin-foil boundary conditions
    (scratch/e4_mbx_checks.py measures the dipole image term).
  - In a cell, the real-space cutoff is part of the dispersion's Hamiltonian, and MBX takes the minimum image. So E4
    fixes the cutoff at 6.5 A and evaluates narrow cells on a replica (DEFAULTS below).

Conventions:
  - Positions are (3n, 3) in Angstrom, atoms O, H, H per molecule: the harness's order, and MBX's for "h2o". Cells
    are (3, 3) in Angstrom, rows = lattice vectors.
  - Energies are in hartree and forces in hartree/bohr. MBX's kcal/mol and Angstrom are converted with its own
    constants, which are the harness's (sseft.units: CODATA 2010 hartree, CODATA 2014 bohr).
  - Site quantities stay in MBX's units: e, Angstrom^3, e Angstrom, e/Angstrom, e/Angstrom^2. MBX's site order is
    O, H, H, M per molecule.

The interaction energy is MBX's energy with the 1-body polynomial switched off ("ignore_1b_poly": ["h2o"]). An
isolated monomer then gives exactly zero at any geometry: no intramolecular electrostatics, and no induced dipoles
without an external field. That is E3's convention: the truth less its monomer energy.

The library keeps one system as global state. It is re-initialised whenever the number of molecules, the boundary
(open / periodic) or the settings change. It is not thread-safe and runs OpenMP inside (OMP_NUM_THREADS).

    python sseft/mbpol.py --check external/mbx/build/MBX      # the checks of check(): references, gradients, cells

This module imports only numpy, ctypes and the standard library, so the build script can run the check without JAX.
"""
import ctypes
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

try:
    from .units import KCALMOL_PER_HARTREE, BOHR_PER_ANGSTROM
except ImportError:                                          # run as a script (python sseft/mbpol.py --check ...)
    KCALMOL_PER_HARTREE = 627.5094740631
    BOHR_PER_ANGSTROM = 1.8897259886

SITES_PER_WATER = 4                  # O, H, H, M
WATER_ATOMS = ("O", "H", "H")
# The truth's settings for E4 (notes/e4-plan.md section 2; scratch/e4_mbx_checks.py measured them). The 2-body
# polynomial reaches 6.5 A and the 3-body 4.5 A for water alone (MBX's README).
#
# In a cell, MBX's dispersion is not an Ewald sum of one fixed kernel:
#   - Inside the real-space cutoff r_c it is MB-pol's damped pair C6 (O-O 237.32, O-H 83.50, H-H 20.09
#     kcal/mol A^6), switched off over r_c - 1 to r_c.
#   - Beyond r_c it is particle-mesh Ewald of per-site long-range c6 (O 15.405, H 4.483), whose O-H product is 69.06,
#     17 % below the pair value (O-O and H-H agree).
# So r_c is part of the Hamiltonian (between r_c 6.2 and 9 A a 64-molecule cell's dispersion moves by about
# 1 kcal/mol). E4 fixes r_c = 6.5 A, the 2-body reach, for every cell, so beyond 6.5 A the truth's dispersion is the
# rank-1 c6 model that a pinned dispersion band reproduces. alpha 0.7 / A puts the real-space truncation at erfc(4.55),
# and a grid of 6 / A with splines of order 8 converges the reciprocal part to about 1e-7 kcal/mol for 32 molecules.
# Open structures sum everything in real space with the pair C6 at every distance, so they differ from a cell's
# energy in their O-H pairs beyond 5.5 A. Their non-additive energies do not (dispersion is pairwise in MB-pol).
DEFAULTS = dict(
    twobody_cutoff=6.5,
    threebody_cutoff=4.5,
    dipole_tolerance=1e-16,          # |mu(t+1) - mu(t)|^2 per site, (e A)^2: the examples' value
    dipole_max_it=500,
    dipole_method="cg",
    # periodic cells only (open structures sum everything in real space, Ewald off):
    realspace_cutoff=6.5,            # A; part of the Hamiltonian in a cell (above)
    alpha_ewald_elec=0.7, grid_density_elec=6.0, spline_order_elec=8,
    alpha_ewald_disp=0.7, grid_density_disp=6.0, spline_order_disp=8,
    # MBX takes the minimum image for its 2-body list and its real-space sums (e4-plan.md section 2: a 64-molecule
    # cell, 12.43 A, misses 1.6e-2 kcal/mol of 2-body against its 2x2x2 replica). A cell narrower than twice the larger
    # of the 2-body reach and r_c is therefore evaluated on its smallest replica that clears both, and every quantity
    # is reported per cell. False: as MBX does, with r_c capped at the cell's half-width.
    replicate=True,
)
# MBX's own example settings (examples/PEFs/001_mbpol/python/mbx_pbc.json), for the check against its printed values
EXAMPLE_PBC = dict(twobody_cutoff=9.0, threebody_cutoff=5.0, realspace_cutoff=9.0, alpha_ewald_elec=0.6,
                   alpha_ewald_disp=0.6, grid_density_elec=2.5, grid_density_disp=2.5, spline_order_elec=6,
                   spline_order_disp=6)
OPEN_REALSPACE = 1000.0              # A: everything in real space for open structures (MBX's README)
_SYSTEM = {"key": None}              # what MBX's one global system was initialised for (shared by every instance)


def load_library(root):
    """ctypes handle of root/lib/libmbx.so (MBX installs into its source tree: MBX_HOME = root)."""
    for name in ("libmbx.so", "libmbx.dylib"):
        path = os.path.join(root, "lib", name)
        if os.path.exists(path):
            return ctypes.CDLL(path, mode=getattr(os, "RTLD_LAZY", 1))
    raise OSError(f"no libmbx in {os.path.join(root, 'lib')}: build MBX with external/mbx/build.sh "
                  "(./configure --enable-shared; make; make install)")


def _commit(repo):
    try:
        return subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def cell_widths(cell):
    """the distances between opposite faces of a cell (rows = lattice vectors): the minimum image holds for a cutoff
    up to half the smallest."""
    cell = np.asarray(cell, float)
    V = abs(np.linalg.det(cell))
    return np.array([V / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)])


class MBPol:
    """MB-pol interaction energies, forces and site quantities of water clusters and periodic cells (see the module
    docstring). One instance per process: MBX keeps a single system."""
    name = "mbpol"

    def __init__(self, root, settings=None, one_body=False):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.lib = load_library(self.root)
        self._bind(self.lib)
        self.settings = {**DEFAULTS, **(settings or {})}
        self.one_body = one_body            # True: MBX's total energy (1-body polynomial on), for its own references
        self._ext = None                    # the uniform external field set on the sites, e/A^2, or None
        self._tmp = tempfile.mkdtemp(prefix="mbx_")

    # ---- the library ----
    @staticmethod
    def _bind(lib):
        D, I = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
        C = ctypes.POINTER(ctypes.c_char_p)
        sig = {
            "initialize_system_py_": [D, I, C, C, I, ctypes.c_char_p],
            "get_energy_": [D, I, D], "get_energy_g_": [D, I, D, D],
            "get_energy_pbc_": [D, I, D, D], "get_energy_pbc_g_": [D, I, D, D, D],
            "get_energy_decomp_": [D, I, D, D, D, D, D, D, D],
            "get_energy_decomp_pbc_": [D, I, D, D, D, D, D, D, D, D],
            "get_electrostatic_energy_decomp_": [D, I, D, D],
            "get_electrostatic_energy_decomp_pbc_": [D, I, D, D, D],
            "get_total_dipole_": [D], "get_xyz_": [D], "get_charges_": [D], "get_polarizabilities_": [D],
            "get_induced_dipoles_": [D], "get_potential_and_electric_field_on_points_": [D, D, D, I],
            "set_potential_and_electric_field_on_sites_": [D, D, I], "finalize_system_": [],
        }
        for name, args in sig.items():
            f = getattr(lib, name)
            f.argtypes = args
            f.restype = None

    @staticmethod
    def _dp(a):
        return np.ascontiguousarray(a, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def config(self, periodic, cell=None, settings=None):
        """the MBX json for open (periodic False) or periodic structures."""
        s = self.settings if settings is None else settings
        mbx = {"box": [], "twobody_cutoff": s["twobody_cutoff"], "threebody_cutoff": s["threebody_cutoff"],
               "dipole_tolerance": s["dipole_tolerance"], "dipole_max_it": s["dipole_max_it"],
               "dipole_method": s["dipole_method"], "realspace_cutoff": OPEN_REALSPACE,
               "alpha_ewald_elec": 0.0, "grid_density_elec": s["grid_density_elec"],
               "spline_order_elec": s["spline_order_elec"], "alpha_ewald_disp": 0.0,
               "grid_density_disp": s["grid_density_disp"], "spline_order_disp": s["spline_order_disp"],
               "ff_mons": [], "connectivity_file": "", "ttm_pairs": [], "ignore_dispersion": [],
               "use_lennard_jones": [], "nonbonded_file": "", "monomers_file": "",
               "ignore_1b_poly": [] if self.one_body else ["h2o"], "ignore_2b_poly": [], "ignore_3b_poly": []}
        if periodic:
            mbx.update(box=[float(x) for x in np.asarray(cell, float).reshape(-1)],
                       realspace_cutoff=s["realspace_cutoff"], alpha_ewald_elec=s["alpha_ewald_elec"],
                       alpha_ewald_disp=s["alpha_ewald_disp"])
        return {"Note": "written by sseft.mbpol", "MBX": mbx}

    def _expand(self, pos, cell):
        """what MBX evaluates: (positions, cell, copies) -- the cell itself, or its smallest replica whose half-width
        clears the 2-body reach (settings["replicate"]); the first copy is the cell's own atoms, in order."""
        if cell is None or not self.settings["replicate"]:
            return pos, cell, 1
        reach = 2.0 * max(self.settings["twobody_cutoff"], self.settings["realspace_cutoff"])
        reps = tuple(max(1, math.ceil(reach / w - 1e-9)) for w in cell_widths(cell))
        if reps == (1, 1, 1):
            return pos, cell, 1
        P, C = replicate(pos, cell, reps)
        return np.ascontiguousarray(P), np.ascontiguousarray(C), int(np.prod(reps))

    def _ensure(self, pos, cell):
        n = len(pos) // 3
        if n * 3 != len(pos):
            raise ValueError("positions must be whole water molecules, O, H, H each")
        periodic = cell is not None
        eff = dict(self.settings)
        if periodic:                                         # MBX's real-space sums take the minimum image
            eff["realspace_cutoff"] = min(eff["realspace_cutoff"], 0.5 * cell_widths(cell).min() - 1e-9)
        key = (n, periodic, json.dumps(eff, sort_keys=True), self.one_body)
        if key == _SYSTEM["key"]:
            return n
        self.finalize()
        path = os.path.join(self._tmp, "mbx.json")
        with open(path, "w") as f:
            json.dump(self.config(periodic, cell, eff), f, indent=1)
        nats = (ctypes.c_int * n)(*([3] * n))
        ats = (ctypes.c_char_p * (3 * n))(*([a.encode() for a in WATER_ATOMS] * n))
        mons = (ctypes.c_char_p * n)(*([b"h2o"] * n))
        self.lib.initialize_system_py_(self._dp(pos.reshape(-1)), nats, ats, mons, ctypes.byref(ctypes.c_int(n)),
                                       path.encode())
        _SYSTEM["key"] = key
        self._ext = None
        return n

    def finalize(self):
        """release MBX's system (the next evaluation initialises a new one)."""
        if _SYSTEM["key"] is not None:
            self.lib.finalize_system_()
            _SYSTEM["key"] = None

    def _prep(self, pos_ang, cell_ang):
        """(positions, cell, copies) as MBX evaluates them, its system initialised for them."""
        pos = np.ascontiguousarray(np.asarray(pos_ang, float).reshape(-1, 3))
        cell = None if cell_ang is None else np.ascontiguousarray(np.asarray(cell_ang, float).reshape(3, 3))
        pos, cell, k = self._expand(pos, cell)
        self._ensure(pos, cell)
        return pos, cell, k

    # ---- energies and forces ----
    def energy_kcal(self, pos_ang, cell_ang=None):
        pos, cell, k = self._prep(pos_ang, cell_ang)
        nat, e = ctypes.c_int(len(pos)), ctypes.c_double(0.0)
        if cell is None:
            self.lib.get_energy_(self._dp(pos.reshape(-1)), ctypes.byref(nat), ctypes.byref(e))
        else:
            self.lib.get_energy_pbc_(self._dp(pos.reshape(-1)), ctypes.byref(nat), self._dp(cell.reshape(-1)),
                                     ctypes.byref(e))
        return e.value / k

    def energy_gradient_kcal(self, pos_ang, cell_ang=None):
        """MBX's energy (kcal/mol) and gradient (kcal/mol/A, (3n, 3)), per cell."""
        n = len(np.asarray(pos_ang).reshape(-1, 3))
        pos, cell, k = self._prep(pos_ang, cell_ang)
        nat, e = ctypes.c_int(len(pos)), ctypes.c_double(0.0)
        g = np.zeros(pos.size)
        if cell is None:
            self.lib.get_energy_g_(self._dp(pos.reshape(-1)), ctypes.byref(nat), ctypes.byref(e), self._dp(g))
        else:
            self.lib.get_energy_pbc_g_(self._dp(pos.reshape(-1)), ctypes.byref(nat), self._dp(cell.reshape(-1)),
                                       ctypes.byref(e), self._dp(g))
        return e.value / k, g.reshape(-1, 3)[:n].copy()

    def energy(self, pos_ang, cell_ang=None):
        """the interaction energy in hartree."""
        return self.energy_kcal(pos_ang, cell_ang) / KCALMOL_PER_HARTREE

    def energy_forces(self, pos_ang, cell_ang=None):
        """the interaction energy (hartree) and forces (hartree/bohr, (3n, 3))."""
        e, g = self.energy_gradient_kcal(pos_ang, cell_ang)
        return e / KCALMOL_PER_HARTREE, -g / (KCALMOL_PER_HARTREE * BOHR_PER_ANGSTROM)

    def components(self, pos_ang, cell_ang=None):
        """MBX's energy by term, in hartree: one_body, two_body, three_body, four_body (the polynomials), dispersion,
        buckingham, electrostatics, and electrostatics split into permanent and induced. Their sum is energy()."""
        pos, cell, k = self._prep(pos_ang, cell_ang)
        nat = ctypes.c_int(len(pos))
        out = [ctypes.c_double(0.0) for _ in range(7)]
        refs = [ctypes.byref(o) for o in out]
        if cell is None:
            self.lib.get_energy_decomp_(self._dp(pos.reshape(-1)), ctypes.byref(nat), *refs)
        else:
            self.lib.get_energy_decomp_pbc_(self._dp(pos.reshape(-1)), ctypes.byref(nat),
                                            self._dp(cell.reshape(-1)), *refs)
        names = ("one_body", "two_body", "three_body", "four_body", "dispersion", "buckingham", "electrostatics")
        comp = {name: o.value / (k * KCALMOL_PER_HARTREE) for name, o in zip(names, out)}
        perm, ind = ctypes.c_double(0.0), ctypes.c_double(0.0)
        if cell is None:
            self.lib.get_electrostatic_energy_decomp_(self._dp(pos.reshape(-1)), ctypes.byref(nat),
                                                      ctypes.byref(perm), ctypes.byref(ind))
        else:
            self.lib.get_electrostatic_energy_decomp_pbc_(self._dp(pos.reshape(-1)), ctypes.byref(nat),
                                                          self._dp(cell.reshape(-1)), ctypes.byref(perm),
                                                          ctypes.byref(ind))
        comp.update(permanent=perm.value / (k * KCALMOL_PER_HARTREE), induced=ind.value / (k * KCALMOL_PER_HARTREE))
        return comp

    # ---- the site model ----
    def _site_arrays(self):
        ns = SITES_PER_WATER * _SYSTEM["key"][0]
        xyz, q, a, mu = np.zeros(3 * ns), np.zeros(ns), np.zeros(ns), np.zeros(3 * ns)
        self.lib.get_xyz_(self._dp(xyz)); self.lib.get_charges_(self._dp(q))
        self.lib.get_polarizabilities_(self._dp(a)); self.lib.get_induced_dipoles_(self._dp(mu))
        return xyz.reshape(-1, 3), q, a, mu.reshape(-1, 3)

    def sites(self, pos_ang, cell_ang=None):
        """the electrostatic sites at this geometry, after an energy evaluation there (the induced dipoles are the
        converged ones): xyz (4n, 3) A, charges (4n,) e, polarisabilities (4n,) A^3, induced dipoles (4n, 3) e A, in
        MBX's order O, H, H, M per molecule; for a cell evaluated on a replica, the cell's own (the first copy's).
        With an external field set, the induced dipoles include its response."""
        n = len(np.asarray(pos_ang).reshape(-1, 3)) // 3
        self.energy_kcal(pos_ang, cell_ang)
        xyz, q, a, mu = (x[:SITES_PER_WATER * n] for x in self._site_arrays())
        return {"xyz": xyz, "charges": q, "polarizabilities": a, "induced": mu}

    def induced_dipole(self, pos_ang, cell_ang=None):
        """the sum of the induced dipoles (e A) after an energy evaluation, per cell."""
        _, _, k = self._prep(pos_ang, cell_ang)
        self.energy_kcal(pos_ang, cell_ang)
        return self._site_arrays()[3].sum(axis=0) / k

    def field_at_points(self, pos_ang, points_ang, cell_ang=None):
        """the electrostatic potential (e/A) and field (e/A^2) of the molecules' permanent charges and induced dipoles
        at the points, after an energy evaluation."""
        self.energy_kcal(pos_ang, cell_ang)
        pts = np.ascontiguousarray(np.asarray(points_ang, float).reshape(-1, 3))
        phi, ef = np.zeros(len(pts)), np.zeros(pts.size)
        self.lib.get_potential_and_electric_field_on_points_(self._dp(pts.reshape(-1)), self._dp(phi), self._dp(ef),
                                                             ctypes.byref(ctypes.c_int(len(pts))))
        return phi, ef.reshape(-1, 3)

    def set_external_field(self, pos_ang, field, cell_ang=None):
        """a uniform external field (e/A^2, 3-vector) on every site, which the induced dipoles answer self-consistently.
        The potential on the sites is left at zero: MBX counts an external potential with a factor 1/2 in its
        permanent energy (electrostatics.cpp, CalculateElecEnergy), so energies with a field set are not the energy of
        the charges in that field. Use the induced dipoles (induced_dipole, induced_response), not the energy.
        field=None clears it. Any later re-initialisation of MBX's system clears it too."""
        self._prep(pos_ang, cell_ang)
        ns = SITES_PER_WATER * _SYSTEM["key"][0]
        E = np.zeros(3) if field is None else np.asarray(field, float)
        self.lib.set_potential_and_electric_field_on_sites_(self._dp(np.zeros(ns)), self._dp(np.tile(E, ns)),
                                                            ctypes.byref(ctypes.c_int(ns)))
        self._ext = None if field is None else E.copy()

    def induced_response(self, pos_ang, cell_ang=None, delta=1e-4):
        """dM_ind/dE (3, 3), A^3, per cell: the induced dipoles' response to a uniform external field at clamped nuclei
        (central differences, field steps delta e/A^2; the response is linear, so the step only sets the rounding).
        For one molecule it is the molecular polarisability; for a periodic cell of volume V the clamped-nuclei
        dielectric tensor under tin-foil boundary conditions is 1 + (4 pi / V) times it."""
        R = np.zeros((3, 3))
        for b in range(3):
            d = np.zeros(3); d[b] = delta
            self.set_external_field(pos_ang, d, cell_ang)
            mp = self.induced_dipole(pos_ang, cell_ang)
            self.set_external_field(pos_ang, -d, cell_ang)
            mm = self.induced_dipole(pos_ang, cell_ang)
            R[:, b] = (mp - mm) / (2 * delta)
        self.set_external_field(pos_ang, None, cell_ang)
        return R

    def describe(self):
        return {"name": self.name, "root": self.root, "commit": _commit(self.root), "settings": dict(self.settings),
                "one_body": self.one_body}


# ---- the checks ----
def _read_xyz(path):
    with open(path) as f:
        n = int(f.readline().split()[0]); f.readline()
        rows = [f.readline().split() for _ in range(n)]
    return [r[0] for r in rows], np.array([[float(x) for x in r[1:4]] for r in rows])


def _parse_expected(path):
    """examples/PEFs/001_mbpol/python/expected_output(_pbc): the energy with gradients and the gradients."""
    e = g = None
    with open(path) as f:
        for line in f:
            if line.startswith("Energy(grad)"):
                e = float(line.split(":")[1])
            elif line.startswith("Grads:"):
                g = np.array(json.loads(line.split(":", 1)[1]))
    return e, g.reshape(-1, 3)


def monomer_ang():
    """the harness's rigid monomer (sseft.water: r_OH 0.9572 A, HOH 104.52 deg), O at the origin."""
    t = math.radians(104.52) / 2
    return np.array([[0.0, 0.0, 0.0], [0.9572 * math.sin(t), 0.0, 0.9572 * math.cos(t)],
                     [-0.9572 * math.sin(t), 0.0, 0.9572 * math.cos(t)]])


def _rotation(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)], [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def random_box(n_mol, cell, rng, min_oo=2.6, min_any=1.5, tries=400000):
    """rigid monomers at random positions and orientations in a cell (rows = lattice vectors, A): O-O at least min_oo
    and every intermolecular atom pair at least min_any apart by the minimum image. A test configuration, not a
    liquid."""
    cell = np.asarray(cell, float).reshape(3, 3)
    inv = np.linalg.inv(cell)
    m = monomer_ang() - monomer_ang()[0]
    X = np.zeros((0, 3, 3))                                      # the molecules placed, (k, 3 atoms, 3)
    while len(X) < n_mol and tries:
        tries -= 1
        cand = rng.uniform(0, 1, 3) @ cell + m @ _rotation(rng).T
        if len(X):
            d = cand[None, :, None, :] - X[:, None, :, :]        # (k, 3, 3, 3)
            d = d - np.round(d @ inv) @ cell
            r = np.linalg.norm(d, axis=-1)
            if r[:, 0, 0].min() < min_oo or r.min() < min_any:
                continue
        X = np.concatenate([X, cand[None]])
    if len(X) < n_mol:
        raise RuntimeError("could not place the molecules")
    return X.reshape(-1, 3)


def replicate(pos_ang, cell_ang, reps=(2, 2, 2)):
    """the cell's contents repeated reps times along its lattice vectors, and the supercell."""
    cell = np.asarray(cell_ang, float).reshape(3, 3)
    shifts = [i * cell[0] + j * cell[1] + k * cell[2] for i in range(reps[0]) for j in range(reps[1])
              for k in range(reps[2])]
    return np.concatenate([np.asarray(pos_ang, float) + s for s in shifts]), cell * np.asarray(reps)[:, None]


def check(root, verbose=True):
    """MBX's MB-pol against its own references and the identities the harness relies on. Returns a dict of
    readings; raises AssertionError on a failure."""
    out = {}
    say = print if verbose else (lambda *a, **k: None)
    ex = os.path.join(root, "examples", "PEFs", "001_mbpol", "python")
    # 1. MBX's example trimer, 1-body on, open and in a 20 A box with the example's settings: energy and gradients
    names, pos = _read_xyz(os.path.join(ex, "input.xyz"))
    assert names == ["O", "H", "H"] * 3
    t = MBPol(root, one_body=True, settings=dict(twobody_cutoff=100.0, threebody_cutoff=7.0))
    e_ref, g_ref = _parse_expected(os.path.join(ex, "expected_output"))
    e, g = t.energy_gradient_kcal(pos)
    out["example_open"] = (e - e_ref, float(np.abs(g - g_ref).max()))
    say(f"example trimer, open: E {e:.12f} kcal/mol (MBX's {e_ref:.12f}), max |grad diff| {out['example_open'][1]:.1e}")
    assert abs(e - e_ref) < 1e-8 and out["example_open"][1] < 1e-6
    t.finalize()
    t = MBPol(root, one_body=True, settings=EXAMPLE_PBC)
    e_ref, g_ref = _parse_expected(os.path.join(ex, "expected_output_pbc"))
    e, g = t.energy_gradient_kcal(pos, np.eye(3) * 20.0)
    out["example_pbc"] = (e - e_ref, float(np.abs(g - g_ref).max()))
    say(f"example trimer, 20 A box: E {e:.12f} kcal/mol (MBX's {e_ref:.12f}), max |grad diff| {out['example_pbc'][1]:.1e}")
    assert abs(e - e_ref) < 1e-8 and out["example_pbc"][1] < 1e-6
    t.finalize()
    # 2. the interaction energy: an isolated monomer at any geometry is zero; a dimer's is its binding energy
    t = MBPol(root)
    rng = np.random.default_rng(0)
    m = monomer_ang() + rng.normal(scale=0.05, size=(3, 3))
    out["monomer"] = t.energy_kcal(m)
    say(f"distorted monomer, 1-body off: {out['monomer']:.3e} kcal/mol")
    assert abs(out["monomer"]) < 1e-12
    # 3. forces against central differences, open
    p = pos.copy()
    e0, F = t.energy_forces(p)
    h = 1e-5
    worst = 0.0
    for i, k in [(0, 0), (1, 2), (4, 1), (8, 0)]:
        pp, pm = p.copy(), p.copy(); pp[i, k] += h; pm[i, k] -= h
        fd = -(t.energy(pp) - t.energy(pm)) / (2 * h * BOHR_PER_ANGSTROM)
        worst = max(worst, abs(fd - F[i, k]) / max(1e-6, abs(F[i, k])))
    out["forces_fd_rel"] = worst
    say(f"open forces against central differences: worst relative difference {worst:.1e}")
    assert worst < 1e-5
    # 4. the monomer's polarisability through the external field (MB-pol: about 1.43-1.47 A^3)
    R = t.induced_response(monomer_ang())
    out["alpha_monomer"] = float(np.trace(R) / 3)
    say(f"monomer polarisability, isotropic: {out['alpha_monomer']:.4f} A^3; tensor eigenvalues "
        f"{np.round(np.linalg.eigvalsh(0.5 * (R + R.T)), 4)}")
    assert 1.2 < out["alpha_monomer"] < 1.7
    return out


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--check":
        check(sys.argv[2])
        print("MB-pol check passed")
    else:
        print(__doc__)
