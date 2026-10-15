"""
q-AQUA v1 as a truth for E3 (E3-Q): the many-body CCSD(T) water potential of Q. Yu, C. Qu, P. L. Houston, R. Conte,
A. Nandi and J. M. Bowman, J. Phys. Chem. Lett. 13, 5068 (2022) (github.com/jmbowma/q-AQUA, MIT licence), called
through the C interface external/qaqua/qaqua_capi.f90 of the shared library built by external/qaqua/build.sh.

What the truth is (pes_shell.f90 of q-AQUA v1): a 4-body-truncated many-body expansion -- the Partridge-Schwenke
monomer, a 2-body permutationally invariant polynomial fit to CCSD(T)/CBS for O-O < 6.5 A, switched over 6.5-7.8 A to
the bare point-charge Coulomb of the monomers' LTP2011 charges and that alone beyond 7.8 A (no induction, no
dispersion); a 3-body fit (CCSD(T)-F12a, BSSE-corrected) switched to zero as the largest O-O distance goes from 7 to
9 A; a 4-body fit (CCSD(T)-F12) switched to zero over 5.8-7.5 A.  So beyond 7.8 A the interaction of two monomers is
exactly sum q_i q_j / r_ij with the truth's own charges, which a pinned-charge Coulomb band reproduces.

Conventions here: positions as (3n, 3) arrays, atoms ordered O, H, H per molecule (the harness's order), in bohr
unless a name says otherwise; energies in hartree, forces in hartree/bohr.  The interaction energy is
E(all) - sum_i E_1b(i): q-AQUA without its 1-body term, so separated monomers give exactly zero (at any geometry)
and the forces are the gradient of the n >= 2 body energy.  q-AQUA's own gradient is analytic except the switching
functions' and the charges' derivatives (central differences, h = 1e-3 bohr): the forces are exact to ~1e-7
relative.  The library keeps module state and is not thread-safe.

    python sseft/qaqua.py --check external/qaqua/build      # the authors' benchmarks (README of qAQUA_v1)
    python sseft/qaqua.py --check external/qaqua/build_pol  # q-AQUA-pol's benchmark and TTM3-F's long range

This module imports only numpy, ctypes and the standard library, so that the build script can run the check without JAX.
"""
import ctypes
import glob
import itertools
import math
import os
import re
import sys

import numpy as np

AUANG = 0.5291772083                 # q-AQUA's bohr in Angstrom (constants.f90); used to reproduce its benchmarks
AUKCAL = 627.51                      # q-AQUA-pol's hartree in kcal/mol (its constants.f90): TTM3-F's kcal/mol over this
DEBYE_PER_E_BOHR = 2.541746473       # e a0 in Debye
RUNTIME_DIR = "runtime"              # build.sh copies the compiler's run-time libraries here (in libqaqua.so's RPATH)
_RUNTIME_NAME = re.compile(r"(libgfortran|libquadmath|libifcore|libifport|libimf|libsvml|libintlc|libirng)\.so")
_preloaded = []                      # handles of run-time libraries loaded by load_library, kept for the process

_RUNTIME_HELP = """\
{lib} is linked against the run-time library of the Fortran compiler that built it (libgfortran, libquadmath),
which this environment does not provide.  Relink the build so that it carries that runtime, in a shell where the
compiler you built with is available (the same module or conda env as for the build script), then run again:
    sh {script} --relink        # seconds: reuses the compiled objects
or make the compiler's runtime visible to this process, e.g.
    export LD_LIBRARY_PATH="$(dirname "$(gfortran -print-file-name=libgfortran.so.5)"):$LD_LIBRARY_PATH\""""


def load_library(root, name="libqaqua.so", script="external/qaqua/build.sh"):
    """ctypes handle of root/<name>.  The library finds the compiler's run-time libraries that the build script copies
    to root/runtime through its RPATH; should the loader not find them that way (a build without that RPATH), they
    are loaded here first, in an order that satisfies their dependencies on one another.  If the library still does
    not load, the error says how to repair the build or the environment."""
    path = os.path.join(root, name)
    try:
        return ctypes.CDLL(path)
    except OSError as e:
        first = e
    pending = sorted(glob.glob(os.path.join(root, RUNTIME_DIR, "*.so*")))
    while pending:                                # load what loads, retry the rest while that makes progress
        failed = []
        for p in pending:
            try:
                _preloaded.append(ctypes.CDLL(p))
            except OSError:
                failed.append(p)
        if len(failed) == len(pending):
            break
        pending = failed
    try:
        return ctypes.CDLL(path)
    except OSError as e:
        raise OSError(f"cannot load {path}: {e}\n{_RUNTIME_HELP.format(lib=name, script=script)}") from first


def loaded_runtime():
    """the files of the Fortran run-time libraries mapped into this process (Linux; empty elsewhere)."""
    try:
        with open("/proc/self/maps") as fh:
            files = {line.split(None, 5)[5].strip() for line in fh if len(line.split(None, 5)) == 6}
    except OSError:
        return []
    return sorted(f for f in files if _RUNTIME_NAME.match(os.path.basename(f)))      # not wheels' renamed copies


class QAqua:
    """the q-AQUA library at `root` (the build directory: libqaqua.so, qAQUA_v1/{src,coef}, q-AQUA/*_data.xyz)."""
    LIBRARY = "libqaqua.so"
    SCRIPT = "external/qaqua/build.sh"
    NAME = "q-AQUA"

    def _layout(self):
        """the source directory (pes_init's working directory) and the coefficient files it reads."""
        coef = os.path.join(self.root, "qAQUA_v1", "coef")
        return os.path.join(self.root, "qAQUA_v1", "src"), [os.path.join(coef, f) for f in (
            "coef_2b_v8_cbs.dat", "coef_3b_v6_5.0-9.5.dat", "coef_3b_v9_2.0-7.0.dat", "coef_4b_v3.dat")]

    def __init__(self, root):
        self.root = os.path.abspath(os.path.expanduser(root))
        lib_path = os.path.join(self.root, self.LIBRARY)
        self.src, coef_files = self._layout()
        missing = [p for p in [lib_path] + coef_files if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(f"incomplete {self.NAME} build at {self.root} (missing {missing[0]}); "
                                    f"run: sh {self.SCRIPT}")
        lib = load_library(self.root, self.LIBRARY, self.SCRIPT)
        self._bind(lib)
        self.lib = lib
        if not lib.qaqua_ready():
            cwd = os.getcwd()
            os.chdir(self.src)                    # pes_init reads its coefficients relative to the source directory
            try:
                lib.qaqua_init(2)
            finally:
                os.chdir(cwd)

    @staticmethod
    def _bind(lib):
        f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        dp = ctypes.POINTER(ctypes.c_double)
        lib.qaqua_init.argtypes = [ctypes.c_int]; lib.qaqua_init.restype = ctypes.c_int
        lib.qaqua_ready.argtypes = []; lib.qaqua_ready.restype = ctypes.c_int
        lib.qaqua_energy.argtypes = [ctypes.c_int, f64, dp]; lib.qaqua_energy.restype = None
        lib.qaqua_energy_gradient.argtypes = [ctypes.c_int, f64, dp, f64]; lib.qaqua_energy_gradient.restype = None
        lib.qaqua_onebody.argtypes = [f64, dp]; lib.qaqua_onebody.restype = None
        lib.qaqua_onebody_gradient.argtypes = [f64, dp, f64]; lib.qaqua_onebody_gradient.restype = None
        lib.qaqua_charges.argtypes = [f64, f64, f64]; lib.qaqua_charges.restype = None

    # ------------------------------------------------------------------ atom order --------------------------------
    @staticmethod
    def to_qaqua_order(pos):
        """(3n, 3) O, H, H per molecule -> flat (9n,) in q-AQUA's order H1a H2a H1b H2b ... Oa Ob ..."""
        P = np.asarray(pos, dtype=np.float64).reshape(-1, 3, 3)
        return np.ascontiguousarray(np.concatenate([P[:, 1:, :].reshape(-1, 3), P[:, 0, :]]).ravel())

    @staticmethod
    def from_qaqua_order(flat, n_mol):
        """flat (9n,) in q-AQUA's order -> (3n, 3) O, H, H per molecule."""
        G = np.asarray(flat).reshape(3 * n_mol, 3)
        H = G[:2 * n_mol].reshape(n_mol, 2, 3)
        return np.concatenate([G[2 * n_mol:][:, None, :], H], axis=1).reshape(3 * n_mol, 3)

    @staticmethod
    def _monomer(pos_mono):
        """(3, 3) O, H, H -> flat (9,) H1, H2, O."""
        P = np.asarray(pos_mono, dtype=np.float64)
        return np.ascontiguousarray(np.concatenate([P[1], P[2], P[0]]))

    # ------------------------------------------------------------------ raw calls --------------------------------
    def total_energy(self, pos_bohr):
        """q-AQUA's total energy (1- to 4-body) of the waters at pos_bohr (3n, 3)."""
        n = len(pos_bohr) // 3
        e = ctypes.c_double()
        self.lib.qaqua_energy(n, self.to_qaqua_order(pos_bohr), ctypes.byref(e))
        return e.value

    def total_energy_gradient(self, pos_bohr):
        n = len(pos_bohr) // 3
        e = ctypes.c_double(); g = np.zeros(9 * n)
        self.lib.qaqua_energy_gradient(n, self.to_qaqua_order(pos_bohr), ctypes.byref(e), g)
        return e.value, self.from_qaqua_order(g, n)

    def onebody(self, mono_bohr):
        """the 1-body (Partridge-Schwenke) energy of one monomer (3, 3) O, H, H."""
        e = ctypes.c_double()
        self.lib.qaqua_onebody(self._monomer(mono_bohr), ctypes.byref(e))
        return e.value

    def onebody_gradient(self, mono_bohr):
        e = ctypes.c_double(); g = np.zeros(9)
        self.lib.qaqua_onebody_gradient(self._monomer(mono_bohr), ctypes.byref(e), g)
        g = g.reshape(3, 3)
        return e.value, np.stack([g[2], g[0], g[1]])

    def charges(self, mono_bohr):
        """the LTP2011 charges (O, H, H; e) of one monomer and its dipole vector (e bohr)."""
        q = np.zeros(3); d = np.zeros(3)
        self.lib.qaqua_charges(self._monomer(mono_bohr), q, d)
        return np.array([q[2], q[0], q[1]]), d

    # ------------------------------------------------------------------ the truth ---------------------------------
    def interaction_energy(self, pos_bohr):
        """E(all) - sum_i E_1b(i): zero for separated monomers at any geometry."""
        P = np.asarray(pos_bohr, dtype=np.float64)
        return self.total_energy(P) - sum(self.onebody(P[3 * i:3 * i + 3]) for i in range(len(P) // 3))

    def interaction_energy_forces(self, pos_bohr):
        """the interaction energy and its forces (hartree/bohr, (3n, 3))."""
        P = np.asarray(pos_bohr, dtype=np.float64)
        E, G = self.total_energy_gradient(P)
        for i in range(len(P) // 3):
            e1, g1 = self.onebody_gradient(P[3 * i:3 * i + 3])
            E -= e1; G[3 * i:3 * i + 3] -= g1
        return E, -G

    def molecule_charges(self, pos_bohr):
        """(3n,) the LTP2011 charges of every molecule at its own geometry, O, H, H per molecule."""
        P = np.asarray(pos_bohr, dtype=np.float64)
        return np.concatenate([self.charges(P[3 * i:3 * i + 3])[0] for i in range(len(P) // 3)])

    def nbody(self, pos_bohr, max_order=4):
        """the many-body decomposition of the interaction energy by the subset energies: {k: sum of the k-body terms}
        for k = 2..max_order (all of it for q-AQUA v1, a 4-body-truncated expansion; q-AQUA-pol's induction has
        terms of every order)."""
        P = np.asarray(pos_bohr, dtype=np.float64)
        n = len(P) // 3
        sub = lambda idx: np.concatenate([P[3 * i:3 * i + 3] for i in idx])
        terms = {}                                                # frozenset(idx) -> k-body term
        for k in range(2, min(max_order, n) + 1):
            for idx in itertools.combinations(range(n), k):
                e = self.interaction_energy(sub(idx))
                for j in range(2, k):
                    e -= sum(terms[frozenset(s)] for s in itertools.combinations(idx, j))
                terms[frozenset(idx)] = e
        return {k: float(sum(v for s, v in terms.items() if len(s) == k)) for k in range(2, min(max_order, n) + 1)}


class QAquaPol(QAqua):
    """q-AQUA-pol (Q. Yu, C. Qu, P. L. Houston, A. Nandi, R. Conte, J. M. Bowman, J. Chem. Theory Comput. 19, 3446
    (2023); github.com/jmbowma/q-AQUA-pol, CC0) at `root`, the build directory of external/qaqua/build_pol.sh:
    libqaqua_pol.so, q-AQUA-pol/q-AQUA-pol/{src, coef_qAQUA-pol} and q-AQUA/*_data.xyz (the raw CCSD(T) sets of the
    q-AQUA v1 repository).  TTM3-F everywhere -- the Partridge-Schwenke monomer, geometry-dependent charges on the
    H atoms and the M-site, one Thole-smeared polarisable site per molecule (the M-site) with self-consistent induced
    dipoles and no cutoff, an O-O C6/R^6 + exponential term -- plus CCSD(T) corrections switched off over
    6.5-7.0 A (2- and 3-body) and 5.5-6.5 A (4-body).  So beyond 7 A the interaction is TTM3-F's: permanent
    electrostatics, many-body induction and dispersion.  The interface is QAqua's (the 1-body term is TTM3-F's
    Partridge-Schwenke energy, the monomer charges its dipole-surface charges on O, H, H) plus TTM3-F's pieces."""
    LIBRARY = "libqaqua_pol.so"
    SCRIPT = "external/qaqua/build_pol.sh"
    NAME = "q-AQUA-pol"

    def _layout(self):
        base = os.path.join(self.root, "q-AQUA-pol", "q-AQUA-pol")
        coef = os.path.join(base, "coef_qAQUA-pol")
        return os.path.join(base, "src"), [os.path.join(coef, f) for f in (
            "coeff_diff_2b.dat", "coeff_diff_3b.dat", "coeff_diff_3b_lr_4th.dat", "coeff_grp200_diff")]

    @staticmethod
    def _bind(lib):
        QAqua._bind(lib)
        f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        dp = ctypes.POINTER(ctypes.c_double)
        lib.qaqua_ttm.argtypes = [ctypes.c_int, f64, f64, f64, f64, f64]; lib.qaqua_ttm.restype = None
        lib.qaqua_ttm_constants.argtypes = [dp, dp]; lib.qaqua_ttm_constants.restype = None

    def ttm(self, pos_bohr):
        """TTM3-F alone at pos_bohr (3n, 3), O, H, H per molecule: its total, the Partridge-Schwenke sum (intra), the
        electrostatics of its site charges (elec), the induction -1/2 sum mu.E_q (ind) and the O-O C6 + exponential
        term (vdw), all in hartree; its induced dipoles mu (n, 3) on the M-sites (e bohr), the M-sites (n, 3, bohr)
        and the site charges (n, 3: H1, H2, M; e)."""
        P = np.asarray(pos_bohr, dtype=np.float64)
        n = len(P) // 3
        comp = np.zeros(5); mu = np.zeros(3 * n); rm = np.zeros(3 * n); qs = np.zeros(3 * n)
        self.lib.qaqua_ttm(n, self.to_qaqua_order(P), comp, mu, rm, qs)
        return dict(zip(("total", "intra", "elec", "ind", "vdw"), comp.tolist()),
                    mu=mu.reshape(n, 3), msites=rm.reshape(n, 3), site_charges=qs.reshape(n, 3))

    def ttm_constants(self):
        """TTM3-F's static polarisability on the M-site and its O-O dispersion coefficient, in atomic units with the
        potential's own constants (bohr = 0.5291772083 A, hartree = 627.51 kcal/mol): (alpha bohr^3, C6 E_h bohr^6)."""
        a = ctypes.c_double(); c6 = ctypes.c_double()
        self.lib.qaqua_ttm_constants(ctypes.byref(a), ctypes.byref(c6))
        return a.value / AUANG ** 3, c6.value / AUKCAL / AUANG ** 6


# ---------------------------------------------------------------------- the raw CCSD(T) n-body sets ----------------
RAW_SETS = {"2b": ("2b_data.xyz", 2), "3b": ("3b_data.xyz", 3), "4b": ("4b_data.xyz", 4)}


def read_raw(path, n_mol):
    """the frames of a q-AQUA raw n-body file: positions (frames, 3n, 3) in A, atoms reordered to O, H, H per molecule,
    and the n-body energies (hartree, printed to 1e-8; the 2-body set is CCSD(T)/CBS, the 3-body BSSE-corrected
    CCSD(T)-F12a/aVTZ, the 4-body CCSD(T)-F12/haTZ)."""
    natm = 3 * n_mol
    with open(path) as fh:
        lines = fh.read().split("\n")
    nfr = len([l for l in lines if l.strip()]) // (natm + 2)
    pos = np.empty((nfr, natm, 3)); E = np.empty(nfr)
    for f in range(nfr):
        b = f * (natm + 2)
        if int(lines[b].split()[0]) != natm:
            raise ValueError(f"{path}: frame {f} has {lines[b].strip()} atoms, expected {natm}")
        E[f] = float(lines[b + 1].split()[0])
        rows = [lines[b + 2 + i].split() for i in range(natm)]
        if [r[0] for r in rows] != ["H"] * (2 * n_mol) + ["O"] * n_mol:
            raise ValueError(f"{path}: frame {f} is not in the order H...H O...O")
        x = np.array([[float(v) for v in r[1:4]] for r in rows])
        H = x[:2 * n_mol].reshape(n_mol, 2, 3)
        pos[f] = np.concatenate([x[2 * n_mol:][:, None, :], H], axis=1).reshape(natm, 3)
    return pos, E


def monomer_internal(pos_ang):
    """(frames, 3n, 3) A -> the O-H distances (frames, n, 2) in A and the H-O-H angles (frames, n) in degrees."""
    P = np.asarray(pos_ang).reshape(len(pos_ang), -1, 3, 3)
    v1 = P[:, :, 1] - P[:, :, 0]; v2 = P[:, :, 2] - P[:, :, 0]
    r1 = np.linalg.norm(v1, axis=-1); r2 = np.linalg.norm(v2, axis=-1)
    cos = np.clip(np.sum(v1 * v2, axis=-1) / (r1 * r2), -1.0, 1.0)
    return np.stack([r1, r2], axis=-1), np.degrees(np.arccos(cos))


def near_rigid(pos_ang, r_oh, theta_deg, tol_r, tol_theta):
    """frames whose every monomer is within tol_r (A) of r_oh in both O-H distances and within tol_theta (degrees) of
    theta_deg: the configurations a rigid-monomer model can be read on."""
    r, th = monomer_internal(pos_ang)
    return np.all(np.abs(r - r_oh) <= tol_r, axis=(1, 2)) & np.all(np.abs(th - theta_deg) <= tol_theta, axis=1)


def largest_oo(pos_ang):
    """the largest O-O distance (A) of every frame (q-AQUA's switching variable for the 3- and 4-body terms)."""
    O = np.asarray(pos_ang)[:, 0::3]
    d = np.linalg.norm(O[:, :, None] - O[:, None, :], axis=-1)
    return d.reshape(len(O), -1).max(axis=1)


# ---------------------------------------------------------------------- the authors' benchmarks --------------------
_BENCHMARKS = {   # README of qAQUA_v1: total energy (hartree) and gradient (hartree/bohr, q-AQUA order), 8 decimals
    "trimer.xyz": (-0.02503134, [
        [0.00018393, -0.00005228, 0.00019626], [-0.00047441, 0.00010057, 0.00038482],
        [-0.00007470, 0.00002213, -0.00007683], [0.00050444, 0.00038716, 0.00011350],
        [-0.00025724, 0.00012446, -0.00009281], [-0.00001201, -0.00051519, -0.00027230],
        [-0.00034540, -0.00045345, -0.00052707], [-0.00031678, 0.00026620, 0.00040260],
        [0.00079217, 0.00012041, -0.00012818]]),
    "t01_prism.xyz": (-0.07331554, [
        [0.00484048, -0.00226665, 0.00074756], [-0.00214426, -0.00122522, 0.00364118],
        [-0.00118297, -0.00056298, 0.00526862], [-0.00286030, -0.00122854, -0.00133340],
        [-0.00192362, -0.00225279, 0.00284307], [-0.00181841, -0.00169786, -0.00294382],
        [-0.00007456, -0.00088599, -0.00432267], [0.00306595, 0.00355145, -0.00045939],
        [-0.00422437, 0.00194335, -0.00077380], [0.00155425, 0.00343632, 0.00037329],
        [-0.00425150, 0.00244677, -0.00010802], [0.00266379, 0.00308012, -0.00039121],
        [-0.00246301, 0.00350676, -0.00465070], [0.00414398, 0.00176588, -0.00379145],
        [0.00391320, 0.00363723, 0.00021019], [-0.00303911, -0.00234440, 0.00464375],
        [0.00258923, -0.00534872, 0.00055884], [0.00121122, -0.00555467, 0.00048796]]),
}


_POL_BENCHMARK = ("hexamer.xyz", -45.7738923286911, [    # README of q-AQUA-pol: the total energy x 627.51 (kcal/mol, as
    # their getpot prints it) and the gradient (hartree/bohr, q-AQUA order, 8 decimals)
    [0.00394754, -0.00182171, 0.00069273], [-0.00228976, -0.00142203, 0.00400568], [-0.00075949, -0.00061204, 0.00420559],
    [-0.00292673, -0.00130259, -0.00122316], [-0.00168868, -0.00239567, 0.00269531], [-0.00187900, -0.00180394, -0.00272606],
    [0.00003051, -0.00049893, -0.00338875], [0.00253757, 0.00353316, -0.00065256], [-0.00338173, 0.00147785, -0.00057620],
    [0.00113714, 0.00272610, 0.00042893], [-0.00328271, 0.00208605, -0.00000976], [0.00267610, 0.00330605, -0.00039869],
    [-0.00129751, 0.00319493, -0.00475530], [0.00379564, 0.00193308, -0.00310067], [0.00382317, 0.00406095, 0.00004663],
    [-0.00307279, -0.00295002, 0.00401148], [0.00213928, -0.00433354, 0.00042017], [0.00049144, -0.00517769, 0.00032463]])
TTM3F_CHARGECON = 18.22234397655801030455      # TTM3-F's sqrt(kcal/mol A) per e (model_mod.f90)
TTM3F_VDW_D, TTM3F_VDW_E = 0.10211829e6, 0.37170376e1    # its O-O exponential, kcal/mol and 1/A (ttm3f_mod.f90)

def _read_xyz_qaqua_order(path):
    with open(path) as fh:
        rows = [l.split() for l in fh.read().split("\n")[2:] if l.strip()]
    return np.array([[float(v) for v in r[1:4]] for r in rows])


def check(root):
    """the README benchmarks: energies to 1e-8 hartree and gradients to 1e-8 hartree/bohr (the printed precision),
    positions converted with q-AQUA's own bohr; and the self-consistency of the driver (order round trip, the
    long range against the LTP2011 point charges).  Returns True if everything passes.  A q-AQUA-pol build
    (libqaqua_pol.so at root) is checked by check_pol."""
    if os.path.isfile(os.path.join(os.path.abspath(os.path.expanduser(root)), QAquaPol.LIBRARY)):
        return check_pol(root)
    qa = QAqua(root)
    print(f"  library        {os.path.join(qa.root, 'libqaqua.so')}")
    for f in loaded_runtime():
        print(f"  runtime        {f}")
    ok = True
    for name, (E_ref, G_ref) in _BENCHMARKS.items():
        xq = _read_xyz_qaqua_order(os.path.join(qa.src, name)) / AUANG          # q-AQUA order, bohr
        n = len(xq) // 3
        pos = qa.from_qaqua_order(xq.ravel(), n)                                   # harness order
        E, G = qa.total_energy_gradient(pos)
        Gq = qa.to_qaqua_order(G).reshape(-1, 3)
        dE = abs(E - E_ref); dG = float(np.max(np.abs(Gq - np.array(G_ref))))
        passed = dE <= 1.5e-8 and dG <= 1.5e-8
        ok &= passed
        print(f"  {name:14s} E = {E:.8f} (README {E_ref:.8f}, |dE| {dE:.1e}); max |dG| {dG:.1e}  {'ok' if passed else 'FAIL'}")
    # the long range: two monomers 12 A apart interact exactly by the point charges of their LTP2011 charges
    from_mono = np.array([[0.0, 0.0, 0.0], [0.7572, 0.0, 0.5865], [-0.7572, 0.0, 0.5865]]) / AUANG
    rot = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    other = from_mono @ rot.T + np.array([0.3, -0.2, 12.0]) / AUANG
    P = np.concatenate([from_mono, other])
    q = qa.molecule_charges(P)
    r = np.linalg.norm(P[:3, None] - P[None, 3:], axis=-1)
    coul = float(np.sum(np.outer(q[:3], q[3:]) / r))
    E = qa.interaction_energy(P)
    passed = abs(E - coul) <= 1e-12 * abs(coul)
    ok &= passed
    print(f"  long range     E_int at 12 A = {E:.6e}, point charges {coul:.6e}, rel. diff {abs(E - coul) / abs(coul):.1e}  "
          f"{'ok' if passed else 'FAIL'}")
    print("q-AQUA check " + ("passed" if ok else "FAILED"))
    return ok


def check_pol(root):
    """q-AQUA-pol: the README benchmark (the hexamer's total energy to 1e-6 kcal/mol -- the polynomials' rounding noise
    is ~1e-10 hartree -- and its gradient to the printed 1e-8 hartree/bohr), and the long range: two monomers 12 A
    apart interact by TTM3-F alone (the CCSD(T) corrections vanish beyond 7 A), and its pieces are recomputed here from
    the site charges, the M-sites and the polarisability -- the electrostatics of point charges, the self-consistent
    point-dipole induction (no smearing left at 12 A) and the O-O C6 term -- to 1e-10 relative.  Returns True if
    everything passes."""
    qa = QAquaPol(root)
    print(f"  library        {os.path.join(qa.root, qa.LIBRARY)}")
    for f in loaded_runtime():
        print(f"  runtime        {f}")
    ok = True
    name, E_ref, G_ref = _POL_BENCHMARK
    xq = _read_xyz_qaqua_order(os.path.join(qa.src, name)) / AUANG
    n = len(xq) // 3
    pos = qa.from_qaqua_order(xq.ravel(), n)
    E, G = qa.total_energy_gradient(pos)
    Gq = qa.to_qaqua_order(G).reshape(-1, 3)
    dE = abs(E * AUKCAL - E_ref); dG = float(np.max(np.abs(Gq - np.array(G_ref))))
    passed = dE <= 1e-6 and dG <= 1.5e-8
    ok &= passed
    print(f"  {name:14s} E = {E * AUKCAL:.10f} kcal/mol (README {E_ref:.10f}, |dE| {dE:.1e}); max |dG| {dG:.1e}  "
          f"{'ok' if passed else 'FAIL'}")
    # the long range: TTM3-F's pieces at 12 A against their definitions, in TTM3-F's units (A, e, kcal/mol)
    mono = np.array([[0.0, 0.0, 0.0], [0.7572, 0.0, 0.5865], [-0.7572, 0.0, 0.5865]])
    rot = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    A = np.concatenate([mono, mono @ rot.T + np.array([0.3, -0.2, 12.0])])          # A
    P = A / AUANG
    E_int = qa.interaction_energy(P)
    t = qa.ttm(P)
    k = TTM3F_CHARGECON ** 2
    sites = [np.vstack([A[3 * m + 1], A[3 * m + 2], t["msites"][m] * AUANG]) for m in range(2)]     # H1, H2, M
    q = t["site_charges"]
    elec = k * sum(q[0, i] * q[1, j] / np.linalg.norm(sites[0][i] - sites[1][j]) for i in range(3) for j in range(3))
    alpha_A3 = qa.ttm_constants()[0] * AUANG ** 3
    field = [sum(q[1 - m, j] * (sites[m][2] - sites[1 - m][j]) / np.linalg.norm(sites[m][2] - sites[1 - m][j]) ** 3
                 for j in range(3)) for m in range(2)]                                # at each M-site, e / A^2
    R = sites[0][2] - sites[1][2]; r = np.linalg.norm(R)
    Tdd = (3.0 * np.outer(R, R) / r ** 2 - np.eye(3)) / r ** 3
    M = np.block([[np.eye(3), -alpha_A3 * Tdd], [-alpha_A3 * Tdd, np.eye(3)]])
    mu = np.linalg.solve(M, alpha_A3 * np.concatenate(field)).reshape(2, 3)           # e A
    ind = -0.5 * k * float(np.sum(mu * np.array(field)))
    roo = np.linalg.norm(A[0] - A[3])
    vdw = -qa.ttm_constants()[1] * AUKCAL * AUANG ** 6 / roo ** 6 + TTM3F_VDW_D * np.exp(-TTM3F_VDW_E * roo)
    rel = lambda a, b: abs(a - b) / abs(b)
    checks = {"TTM3-F alone": (E_int, t["total"] - t["intra"], 1e-12), "electrostatics": (t["elec"], elec / AUKCAL, 1e-10),
              "induction": (t["ind"], ind / AUKCAL, 1e-10), "dispersion": (t["vdw"], vdw / AUKCAL, 1e-10),
              "induced dipoles": (float(np.linalg.norm(t["mu"] * AUANG)), float(np.linalg.norm(mu)), 1e-10)}
    for label, (got, want, tol) in checks.items():
        passed = rel(got, want) <= tol
        ok &= passed
        print(f"  12 A {label:16s} {got:+.6e} vs {want:+.6e} (rel. diff {rel(got, want):.1e})  {'ok' if passed else 'FAIL'}")
    print("q-AQUA-pol check " + ("passed" if ok else "FAILED"))
    return ok

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        sys.exit(0 if check(sys.argv[2]) else 1)
    print(__doc__.strip().split("\n\n")[-2])
    sys.exit(2)
