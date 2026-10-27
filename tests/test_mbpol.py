"""MB-pol through MBX as E4's truth (sseft/mbpol.py). The geometry helpers run everywhere; the rest is skipped unless
a build exists, at $MBX_HOME or external/mbx/build/MBX (sh external/mbx/build.sh)."""
import math
import os
from pathlib import Path

import numpy as np
import pytest

from sseft import mbpol as M
from sseft.units import KCALMOL_PER_HARTREE, BOHR_PER_ANGSTROM

REPO = Path(__file__).resolve().parents[1]
ROOT = os.environ.get("MBX_HOME", str(REPO / "external" / "mbx" / "build" / "MBX"))
needs_mbx = pytest.mark.skipif(not os.path.isfile(os.path.join(ROOT, "lib", "libmbx.so")),
                               reason="no MBX build (sh external/mbx/build.sh, or set MBX_HOME)")


# ---- no library needed ----
def test_cell_widths_and_replicate():
    cell = np.array([[10.0, 0, 0], [3.0, 12.0, 0], [0, 0, 8.0]])
    w = M.cell_widths(cell)
    V = abs(np.linalg.det(cell))
    faces = [np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)]
    assert np.allclose(w, V / np.array(faces)) and np.isclose(w[2], 8.0) and w[0] < 10.0
    pos = np.random.default_rng(0).normal(size=(6, 3))
    P, C = M.replicate(pos, cell, (2, 1, 3))
    assert P.shape == (36, 3) and np.allclose(C, cell * np.array([2, 1, 3])[:, None])
    assert np.allclose(P[:6], pos)                                  # the first copy is the cell's own atoms, in order
    assert np.allclose(P[6:12], pos + cell[2])                       # then along the last axis fastest


def test_random_box_constraints():
    cell = np.eye(3) * 12.0
    pos = M.random_box(20, cell, np.random.default_rng(1), min_oo=2.6, min_any=1.5)
    assert pos.shape == (60, 3)
    inv = np.linalg.inv(cell)
    X = pos.reshape(20, 3, 3)
    mono = M.monomer_ang()
    for m in X:                                                    # rigid monomers: the harness's geometry
        d = lambda Y: sorted(np.linalg.norm(Y[i] - Y[j]) for i in range(3) for j in range(i + 1, 3))
        assert np.allclose(d(m), d(mono))
    for i in range(20):
        for j in range(i + 1, 20):
            D = X[i][:, None] - X[j][None]
            D = D - np.round(D @ inv) @ cell
            r = np.linalg.norm(D, axis=-1)
            assert r[0, 0] >= 2.6 - 1e-12 and r.min() >= 1.5 - 1e-12


# ---- MBX ----
@pytest.fixture(scope="module")
def truth():
    return M.MBPol(ROOT)


@needs_mbx
def test_check():
    """MBX's MB-pol example to its printed digits, the monomer's zero, forces, the monomer's polarisability."""
    out = M.check(ROOT, verbose=False)
    assert 1.3 < out["alpha_monomer"] < 1.5


@needs_mbx
def test_site_model(truth):
    m = M.monomer_ang() + 1.0
    s = truth.sites(m)
    assert s["xyz"].shape == (4, 3) and np.allclose(s["xyz"][:3], m)
    assert abs(s["charges"].sum()) < 1e-12 and s["charges"][0] == 0.0          # TTM4-F: no charge on O
    assert np.allclose(s["polarizabilities"], [1.310, 0.294, 0.294, 0.0])
    b = 0.5 * (m[1] + m[2]) - m[0]                                              # the M-site on the bisector, 0.25 A out
    assert np.allclose(s["xyz"][3] - m[0], 0.25 * b / np.linalg.norm(b), atol=1e-6)
    assert np.allclose(s["induced"], 0.0)                                       # nothing to induce it


@needs_mbx
def test_units_and_forces_in_a_cell(truth):
    cell = np.eye(3) * 13.2
    pos = M.random_box(24, cell, np.random.default_rng(3))
    e_k, g_k = truth.energy_gradient_kcal(pos, cell)
    e, F = truth.energy_forces(pos, cell)
    # each call solves the induced dipoles again, from MBX's last ones: repeat calls agree to ~1e-14 hartree
    assert np.isclose(e, e_k / KCALMOL_PER_HARTREE, rtol=0, atol=1e-12)
    assert np.allclose(F, -g_k / (KCALMOL_PER_HARTREE * BOHR_PER_ANGSTROM), rtol=1e-9, atol=1e-12)
    assert np.isclose(truth.energy(pos, cell), e, rtol=0, atol=1e-12)
    h = 1e-5
    for i, k in [(0, 0), (7, 1), (41, 2)]:
        p, q = pos.copy(), pos.copy(); p[i, k] += h; q[i, k] -= h
        fd = -(truth.energy(p, cell) - truth.energy(q, cell)) / (2 * h * BOHR_PER_ANGSTROM)
        assert abs(fd - F[i, k]) <= 1e-5 * max(abs(F[i, k]), 1e-4)


@needs_mbx
def test_narrow_cell_through_its_replica(truth):
    """a cell narrower than twice the 6.5 A reach: the driver's value (its 2x2x2 replica) is the 3x3x3 replica's."""
    cell = np.eye(3) * 10.0
    pos = M.random_box(14, cell, np.random.default_rng(4))
    e2, g2 = truth.energy_gradient_kcal(pos, cell)
    P, C = M.replicate(pos, cell, (3, 3, 3))
    e3, g3 = truth.energy_gradient_kcal(P, C)
    assert abs(e3 / 27 - e2) < 1e-6 * max(1.0, abs(e2))
    assert np.abs(g3[:len(pos)] - g2).max() < 1e-5


@needs_mbx
def test_tin_foil_dipole_image(truth):
    """one molecule in a cubic cell: tin-foil's -(2 pi / 3) |mu|^2 / L^3 to its O(L^-2) remainder (vacuum: 0)."""
    m = M.monomer_ang()
    s = truth.sites(m)
    mu = (s["charges"][:, None] * s["xyz"]).sum(0)
    coulomb = 1e-7 * (299792458.0 * 1.602176565e-19) ** 2 / 1e-10 * 6.02214129e23 / 4184.0
    for L, lo, hi in ((30.0, 0.975, 0.995), (60.0, 0.993, 0.999)):
        ratio = truth.energy_kcal(m, np.eye(3) * L) / (-(2 * math.pi / 3) * coulomb * (mu @ mu) / L ** 3)
        assert lo < ratio < hi


@needs_mbx
def test_external_field_response(truth):
    m = M.monomer_ang()
    R1 = truth.induced_response(m, delta=1e-4)
    R2 = truth.induced_response(m, delta=2e-3)
    assert np.allclose(R1, R2, atol=1e-6)                                       # linear
    assert np.allclose(R1, R1.T, atol=1e-6) and np.all(np.linalg.eigvalsh(R1) > 1.0)
    assert np.allclose(truth.sites(m)["induced"], 0.0)                          # cleared afterwards
