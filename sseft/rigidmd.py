"""
Rigid-water molecular dynamics, for sampling configurations (E4's data, notes/e4-plan.md section 3).

The integrator is velocity Verlet with SHAKE/RATTLE. Every molecule is held at a fixed geometry by its three distance
constraints, O-H, O-H and H-H, vectorised over the molecules. The thermostat is the Bussi-Donadio-Parrinello
stochastic velocity rescaling (J. Chem. Phys. 126, 014101 (2007)), which samples the canonical ensemble. The kinetic
energy the thermostat (and the removal of the centre-of-mass drift) takes out is kept in `heat`, so that
conserved() = K + E_pot + heat is constant up to the integrator's error, with or without the thermostat: its drift is
the trajectory's check that the forces are the gradient of the energy.

The module is NumPy only. The forces come from a callable: MB-pol through MBX in scripts/e4_data.py, toy potentials
in tests/test_rigidmd.py.

Units: positions in A, velocities in A/fs, time in fs, masses in amu, energies in kcal/mol, forces in kcal/mol/A.
1 amu A^2/fs^2 = 1e7 J/mol = 2390.057 kcal/mol.

Atoms are ordered O, H, H per molecule. Along the trajectory molecules stay whole and unwrapped. wrap_molecules()
moves each whole molecule so that its O lies in the cell; that is the form the force callable receives and the form
frames are saved in. A rigid molecule has 6 degrees of freedom; with the centre-of-mass momentum held at zero the
system has 6n - 3.
"""
import math

import numpy as np

MASSES = np.array([15.999, 1.008, 1.008])            # amu: O, H, H (the dynamics only; configurations do not depend on them)
KCAL_PER_AMU_A2_FS2 = 1.0e7 / 4184.0                 # 1 amu A^2/fs^2 in kcal/mol
KB = 8.314462618 / 4184.0                             # kcal/mol/K
PAIRS = ((0, 1), (0, 2), (1, 2))                       # the three constraints of a water molecule


def monomer_distances(monomer):
    """the constrained distances (O-H1, O-H2, H1-H2) of a (3, 3) monomer, A."""
    return np.array([np.linalg.norm(monomer[i] - monomer[j]) for i, j in PAIRS])


def wrap_molecules(pos, cell):
    """(3n, 3) positions with every molecule moved whole by a lattice vector so that its O lies in the cell
    (fractional coordinates in [0, 1)); rows of cell are the lattice vectors."""
    cell = np.asarray(cell, float)
    X = np.asarray(pos, float).reshape(-1, 3, 3)
    frac = X[:, 0] @ np.linalg.inv(cell)
    shift = np.floor(frac) @ cell
    return (X - shift[:, None, :]).reshape(-1, 3)


def bussi_factor(K, K_target, n_dof, dt_over_tau, rng):
    """the velocity scale factor of one Bussi step: the kinetic energy K resampled towards K_target over dt / tau
    (the paper's Appendix: alpha^2 from one normal deviate and a chi-square of n_dof - 1 degrees of freedom)."""
    c = math.exp(-dt_over_tau)
    r1 = rng.normal()
    s = rng.chisquare(n_dof - 1) if n_dof > 1 else 0.0
    ratio = K_target / (n_dof * K)
    a2 = c + (1.0 - c) * (s + r1 * r1) * ratio + 2.0 * r1 * math.sqrt(c * (1.0 - c) * ratio)
    sign = 1.0 if r1 + math.sqrt(c / ((1.0 - c) * ratio)) >= 0 else -1.0
    return sign * math.sqrt(max(a2, 0.0))


class RigidWaterMD:
    """velocity Verlet with SHAKE/RATTLE for rigid water and a Bussi thermostat (tau_fs None: microcanonical).

    force(pos (3n, 3), cell) -> (energy kcal/mol, forces (3n, 3) kcal/mol/A) is called on the molecule-wrapped
    positions. The initial positions must satisfy the constraints (rigid monomers of the given geometry). seed is
    anything numpy.random.default_rng takes (an int, a SeedSequence); it drives the velocities and the thermostat."""

    def __init__(self, pos, cell, force, dt_fs=2.0, T_K=300.0, tau_fs=100.0, seed=0, tol=1e-12, max_iter=500):
        self.cell = np.asarray(cell, float).reshape(3, 3)
        self.X = np.asarray(pos, float).reshape(-1, 3, 3).copy()
        self.n = len(self.X)
        self.d = monomer_distances(self.X[0])
        if np.abs(np.array([monomer_distances(m) for m in self.X]) - self.d).max() > 1e-8:
            raise ValueError("every molecule must have the same rigid geometry as the first")
        self.m = MASSES
        self.force_fn = force
        self.dt, self.T, self.tau = float(dt_fs), float(T_K), tau_fs
        self.rng = np.random.default_rng(seed)
        self.tol, self.max_iter = tol, max_iter
        self.n_dof = 6 * self.n - 3
        self.time_fs = 0.0
        self.heat = 0.0                              # kcal/mol taken out by the thermostat and the COM removal
        self.V = np.zeros_like(self.X)
        self.E_pot, F = self._force(self.X)
        self.A = self._acc(F)

    # ---- pieces ----
    def _force(self, X):
        E, F = self.force_fn(wrap_molecules(X.reshape(-1, 3), self.cell), self.cell)
        return float(E), np.asarray(F, float).reshape(-1, 3, 3)

    def _acc(self, F):
        return F / (self.m[None, :, None] * KCAL_PER_AMU_A2_FS2)

    def kinetic(self):
        return 0.5 * KCAL_PER_AMU_A2_FS2 * float(np.sum(self.m[None, :, None] * self.V ** 2))

    def temperature(self):
        return 2.0 * self.kinetic() / (self.n_dof * KB)

    def conserved(self):
        """K + E_pot + the energy the thermostat has taken out: constant up to the integrator's error."""
        return self.kinetic() + self.E_pot + self.heat

    def remove_com_velocity(self):
        K = self.kinetic()
        p = np.sum(self.m[None, :, None] * self.V, axis=(0, 1))
        self.V -= p / (self.n * self.m.sum())
        self.heat += K - self.kinetic()

    def shake(self, Xn, Xold, V):
        """SHAKE: positions Xn moved along the old bond vectors until every constraint holds; V gets the same moves / dt."""
        inv = 1.0 / self.m
        for it in range(self.max_iter):
            worst = 0.0
            for k, (i, j) in enumerate(PAIRS):
                s = Xn[:, i] - Xn[:, j]
                diff = self.d[k] ** 2 - np.sum(s * s, axis=-1)
                worst = max(worst, float(np.abs(diff).max()) / self.d[k] ** 2)
                r_old = Xold[:, i] - Xold[:, j]
                g = diff / (2.0 * np.sum(s * r_old, axis=-1) * (inv[i] + inv[j]))
                Xn[:, i] += (g * inv[i])[:, None] * r_old
                Xn[:, j] -= (g * inv[j])[:, None] * r_old
                V[:, i] += (g * inv[i] / self.dt)[:, None] * r_old
                V[:, j] -= (g * inv[j] / self.dt)[:, None] * r_old
            if worst < self.tol:
                return Xn, V
        raise RuntimeError(f"SHAKE did not converge in {self.max_iter} iterations (relative error {worst:.1e})")

    def rattle(self, X, V):
        """RATTLE: velocities projected so that no constrained distance changes (v_ij . r_ij = 0)."""
        inv = 1.0 / self.m
        for it in range(self.max_iter):
            worst = 0.0
            for k, (i, j) in enumerate(PAIRS):
                r = X[:, i] - X[:, j]
                vr = np.sum((V[:, i] - V[:, j]) * r, axis=-1)
                worst = max(worst, float(np.abs(vr).max()) / self.d[k] ** 2)
                g = vr / (self.d[k] ** 2 * (inv[i] + inv[j]))
                V[:, i] -= (g * inv[i])[:, None] * r
                V[:, j] += (g * inv[j])[:, None] * r
            if worst < self.tol:
                return V
        raise RuntimeError(f"RATTLE did not converge in {self.max_iter} iterations")

    def constraint_error(self):
        """the largest relative deviation of a constrained distance."""
        return max(float(np.abs(np.linalg.norm(self.X[:, i] - self.X[:, j], axis=-1) / self.d[k] - 1).max())
                   for k, (i, j) in enumerate(PAIRS))

    # ---- dynamics ----
    def init_velocities(self, T_K=None):
        """Maxwell-Boltzmann velocities at T_K, projected onto the constraints, the centre of mass at rest, and scaled
        to the temperature exactly. The conserved energy starts again from here (heat = 0)."""
        T = self.T if T_K is None else T_K
        sig = np.sqrt(KB * T / (self.m * KCAL_PER_AMU_A2_FS2))
        self.V = self.rng.normal(size=self.X.shape) * sig[None, :, None]
        self.V = self.rattle(self.X, self.V)
        self.remove_com_velocity()
        self.V *= math.sqrt(T / self.temperature())
        self.heat = 0.0

    def step(self):
        dt = self.dt
        V = self.V + 0.5 * dt * self.A
        Xn = self.X + dt * V
        Xn, V = self.shake(Xn, self.X, V)
        self.E_pot, F = self._force(Xn)
        self.A = self._acc(F)
        V = V + 0.5 * dt * self.A
        self.X = Xn
        self.V = self.rattle(Xn, V)
        if self.tau is not None:
            K = self.kinetic()
            a = bussi_factor(K, 0.5 * self.n_dof * KB * self.T, self.n_dof, dt / self.tau, self.rng)
            self.V *= a
            self.heat += K * (1.0 - a * a)
        self.time_fs += dt

    def run(self, n_steps, every=0, callback=None):
        """n_steps steps; callback(self) after every `every` steps (0: never). The centre-of-mass velocity is
        removed every 100 steps (the forces of a periodic solve sum to zero only to its accuracy)."""
        for s in range(1, n_steps + 1):
            self.step()
            if s % 100 == 0:
                self.remove_com_velocity()
            if every and s % every == 0 and callback is not None:
                callback(self)

    def positions(self):
        """the current positions, molecule-wrapped, (3n, 3) A."""
        return wrap_molecules(self.X.reshape(-1, 3), self.cell)


def rdf_oo(frames, cell, r_max=None, dr=0.05):
    """the O-O radial distribution function of frames (F, 3n, 3) in a cell, by the minimum image: (r, g)."""
    cell = np.asarray(cell, float)
    inv = np.linalg.inv(cell)
    V = abs(np.linalg.det(cell))
    widths = V / np.array([np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)])
    r_max = 0.5 * widths.min() if r_max is None else r_max
    edges = np.arange(0.0, r_max + dr, dr)
    hist = np.zeros(len(edges) - 1)
    frames = np.asarray(frames, float)
    n = frames.shape[1] // 3
    iu = np.triu_indices(n, 1)
    for fr in frames:
        O = fr[0::3]
        d = O[:, None, :] - O[None, :, :]
        d -= np.round(d @ inv) @ cell
        r = np.linalg.norm(d, axis=-1)[iu]
        hist += np.histogram(r, bins=edges)[0]
    shell = 4.0 / 3.0 * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    ideal = len(frames) * 0.5 * n * (n - 1) / V * shell
    return 0.5 * (edges[1:] + edges[:-1]), hist / ideal
