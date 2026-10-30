"""The rigid-water integrator (sseft/rigidmd.py) on a toy potential: constraints, energy conservation without a
thermostat, the Bussi thermostat's temperature, the initial velocities, the molecule wrapping, the O-O g(r)."""
import math

import numpy as np
import pytest

from sseft import rigidmd as R

W = {(8, 8): 1.0, (8, 1): -0.5, (1, 8): -0.5, (1, 1): 0.3}       # kcal/mol: repulsive O-O, attractive O-H, weak H-H


def toy_force(n_mol, s=1.0):
    """a smooth intermolecular toy: sum over atom pairs of different molecules of w_ab exp(-r^2 / s^2), by the minimum
    image (the cell at least 11 s wide, so that the minimum image holds to 1e-50)."""
    Z = np.tile([8, 1, 1], n_mol)
    w = np.array([[W[(a, b)] for b in Z] for a in Z])
    mol = np.repeat(np.arange(n_mol), 3)
    w[mol[:, None] == mol[None, :]] = 0.0

    def f(pos, cell):
        inv = np.linalg.inv(cell)
        d = pos[:, None, :] - pos[None, :, :]
        d -= np.round(d @ inv) @ cell
        r2 = np.sum(d * d, axis=-1)
        e = w * np.exp(-r2 / s ** 2)
        E = 0.5 * e.sum()
        F = np.sum((2.0 * e / s ** 2)[:, :, None] * d, axis=1)            # -dE/dr_i
        return E, F
    return f


@pytest.fixture
def box():
    from sseft.mbpol import random_box
    cell = np.eye(3) * 12.0
    return random_box(16, cell, np.random.default_rng(0), min_oo=2.6, min_any=1.5), cell


def test_toy_force_is_the_gradient(box):
    pos, cell = box
    f = toy_force(16)
    E, F = f(pos, cell)
    h = 1e-6
    for i, k in [(0, 0), (5, 2), (40, 1)]:
        p, q = pos.copy(), pos.copy(); p[i, k] += h; q[i, k] -= h
        assert abs(-(f(p, cell)[0] - f(q, cell)[0]) / (2 * h) - F[i, k]) < 1e-6


def test_constraints_and_energy_conservation(box):
    pos, cell = box
    md = R.RigidWaterMD(pos, cell, toy_force(16), dt_fs=1.0, T_K=300.0, tau_fs=None, seed=1, tol=1e-13)
    md.init_velocities()
    E0 = md.E_pot + md.kinetic()
    Es = []
    md.run(400, every=10, callback=lambda m: Es.append(m.E_pot + m.kinetic()))
    assert md.constraint_error() < 1e-10
    drift = np.abs(np.array(Es) - E0).max()
    assert drift < 2e-3 * md.kinetic(), (drift, md.kinetic())                  # a symplectic integrator: bounded, small
    for k, (i, j) in enumerate(R.PAIRS):                                         # no velocity along any constraint
        r = md.X[:, i] - md.X[:, j]
        assert np.abs(np.sum((md.V[:, i] - md.V[:, j]) * r, axis=-1)).max() < 1e-9


def test_thermostat_temperature(box):
    pos, cell = box
    md = R.RigidWaterMD(pos, cell, toy_force(16), dt_fs=1.0, T_K=250.0, tau_fs=50.0, seed=2)
    md.init_velocities(400.0)
    E0 = md.conserved()
    md.run(300)                                                                  # relax towards 250 K
    Ts, Ec = [], []
    md.run(3000, every=5, callback=lambda m: (Ts.append(m.temperature()), Ec.append(m.conserved())))
    assert abs(np.mean(Ts) / 250.0 - 1.0) < 0.1
    assert md.constraint_error() < 1e-10
    # the thermostat took out about 150 K of kinetic energy and keeps exchanging it; K + E_pot + heat stays put
    assert md.heat > 0.5 * 0.5 * md.n_dof * R.KB * 150.0
    assert np.abs(np.array(Ec) - E0).max() < 2e-3 * md.kinetic()


def test_bussi_factor_limits():
    rng = np.random.default_rng(3)
    assert abs(R.bussi_factor(2.0, 1.0, 50, 1e-12, rng) - 1.0) < 1e-5          # tau >> dt: no change
    K, Kt, n = 3.0, 1.0, 40                                                       # tau << dt: K resampled canonically
    samples = [K * R.bussi_factor(K, Kt, n, 50.0, rng) ** 2 for _ in range(20000)]
    assert abs(np.mean(samples) / Kt - 1.0) < 0.02
    assert abs(np.var(samples) / (2 * Kt ** 2 / n) - 1.0) < 0.1                  # Gamma(n/2): variance 2 Kt^2 / n


def test_initial_velocities(box):
    pos, cell = box
    md = R.RigidWaterMD(pos, cell, toy_force(16), seed=4)
    md.init_velocities(300.0)
    assert abs(md.temperature() - 300.0) < 1e-9
    assert np.abs(np.sum(R.MASSES[None, :, None] * md.V, axis=(0, 1))).max() < 1e-12
    assert md.n_dof == 6 * 16 - 3


def test_wrap_molecules():
    cell = np.array([[10.0, 0, 0], [2.0, 9.0, 0], [1.0, 1.5, 8.0]])
    m = np.array([[0.0, 0, 0], [0.75, 0, 0.59], [-0.75, 0, 0.59]])
    X = np.concatenate([m + np.array([25.3, -7.1, 3.3]), m + np.array([-0.1, 4.0, 17.0])])
    Y = R.wrap_molecules(X, cell).reshape(-1, 3, 3)
    frac = Y[:, 0] @ np.linalg.inv(cell)
    assert np.all((frac >= 0) & (frac < 1))
    for a, b in zip(X.reshape(-1, 3, 3), Y):                                     # whole, moved by lattice vectors
        assert np.allclose(b - b[0], a - a[0])
        n = (a[0] - b[0]) @ np.linalg.inv(cell)
        assert np.allclose(n, np.round(n))


def test_rdf_of_an_ideal_gas():
    rng = np.random.default_rng(5)
    cell = np.eye(3) * 15.0
    frames = rng.uniform(0, 15.0, size=(200, 3 * 100, 3))
    r, g = R.rdf_oo(frames, cell, r_max=7.0, dr=0.5)
    assert abs(np.mean(g[r > 3.0]) - 1.0) < 0.05
