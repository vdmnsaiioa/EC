"""q-AQUA-pol as the E3-QP truth (sseft/qaqua.py QAquaPol, sseft.water.QAquaPolTruth, the probe reading of
scripts/e3_water.py).  Skipped unless a build exists: $QAQUA_POL_ROOT or external/qaqua/build_pol
(sh external/qaqua/build_pol.sh)."""
import importlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ROOT = os.environ.get("QAQUA_POL_ROOT", str(REPO / "external" / "qaqua" / "build_pol"))
pytestmark = pytest.mark.skipif(not os.path.isfile(os.path.join(ROOT, "libqaqua_pol.so")),
                                reason="no q-AQUA-pol build (sh external/qaqua/build_pol.sh, or set QAQUA_POL_ROOT)")

from sseft import qaqua as Q, water as W  # noqa: E402
from sseft.units import ang_to_bohr, bohr_to_ang  # noqa: E402


@pytest.fixture(scope="module")
def truth():
    return W.qaqua_pol_truth(ROOT)


@pytest.fixture(scope="module")
def hexamer(truth):
    xq = Q._read_xyz_qaqua_order(os.path.join(truth.qa.src, "hexamer.xyz")) / Q.AUANG
    return truth.qa.from_qaqua_order(xq.ravel(), 6)                # bohr, O, H, H per molecule


def test_authors_benchmark_and_the_long_range():
    """the README hexamer (energy and gradient) and TTM3-F's pieces at 12 A recomputed from its sites (check_pol)."""
    assert Q.check(ROOT)


@pytest.mark.skipif(shutil.which("ldd") is None, reason="no ldd")
def test_library_carries_its_runtime():
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    out = subprocess.run(["ldd", os.path.join(ROOT, "libqaqua_pol.so")], env=env, capture_output=True, text=True).stdout
    assert "not found" not in out, out


def test_interaction_energy_is_ttm3f_beyond_seven_angstrom(truth, hexamer):
    """the CCSD(T) corrections vanish once every O-O exceeds 7 A: the interaction energy is TTM3-F's (minus its
    Partridge-Schwenke monomers) to rounding, and a monomer alone has none."""
    qa = truth.qa
    E, F = qa.interaction_energy_forces(hexamer[:3])
    assert E == 0.0 and np.all(F == 0.0)
    spread = np.concatenate([hexamer[3 * m:3 * m + 3] - hexamer[3 * m] + np.array([0.0, 0.0, 16.0 * m]) for m in range(3)])  # O-O 8.5 A
    t = qa.ttm(spread)
    assert abs(qa.interaction_energy(spread) - (t["total"] - t["intra"])) < 1e-15
    assert abs(t["elec"] + t["ind"] + t["vdw"] - (t["total"] - t["intra"])) < 1e-15


def test_forces_integrate_to_the_energy(truth, hexamer):
    rng = np.random.default_rng(5)
    for n in (2, 3, 6):
        P = hexamer[:3 * n]
        v = rng.normal(size=P.shape); v /= np.linalg.norm(v)
        s = np.linspace(0.0, 0.05, 201)
        g = np.array([-np.vdot(truth.qa.interaction_energy_forces(P + t * v)[1], v) for t in s])
        w = np.ones(len(s)); w[1:-1:2] = 4; w[2:-1:2] = 2
        integral = (s[1] - s[0]) / 3 * np.sum(w * g)
        dE = truth.qa.interaction_energy(P + s[-1] * v) - truth.qa.interaction_energy(P)
        assert abs(dE - integral) < 1e-9, (n, dE, integral)


def test_truth_object_and_pre_registered_numbers(truth):
    assert abs(truth.r_e - 2.92988) < 2e-5                        # the rigid dimer's minimum at phi = 0
    assert abs(W.dipole_dipole_coefficient(truth) + 0.6571) < 5e-4
    q = truth.molecule_charges(W.monomer_positions())
    np.testing.assert_allclose(q, [-0.6617307, 0.3308653, 0.3308653], atol=1e-6)
    np.testing.assert_array_equal(W.monomer_structure(truth).pinned["q"], q)
    assert abs(truth.alpha - 9.7446) < 1e-3 and abs(truth.c6 - 52.469) < 1e-2      # TTM3-F's, in atomic units
    t = truth.qa.ttm(ang_to_bohr(W.monomer_positions()))
    # TTM3-F's sites keep the dipole of the dipole-surface charges
    d_sites = (t["site_charges"][0, 0] * ang_to_bohr(W.monomer_positions()[1]) + t["site_charges"][0, 1] * ang_to_bohr(W.monomer_positions()[2])
               + t["site_charges"][0, 2] * t["msites"][0])
    d_q3 = (q[:, None] * ang_to_bohr(W.monomer_positions())).sum(0)
    np.testing.assert_allclose(d_sites, d_q3, atol=1e-12)


def test_probe_reading_on_the_truth(truth, monkeypatch):
    """probe configurations on q-AQUA-pol: energies and forces are the truth's, the non-additive part is the interaction
    minus the three dimer energies, and beyond 7 A it is TTM3-F's induction beyond the pairs."""
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    water = importlib.import_module("e3_water")
    st, E_core = W.probe_configurations(3, (10.0,), 2, seed=3, mc_steps=10, truth=truth)
    for s in st:
        E, F = truth.energy_forces(bohr_to_ang(s.positions), 4)
        assert abs(E - s.energy) < 1e-15 and np.allclose(F, s.forces, rtol=1e-12, atol=1e-15)
    ref = water.probe_truth(st, truth)
    cores, pairs = water.probe_parts(st)
    for k, s in enumerate(st):
        assert abs(ref["interaction"][k] - (s.energy - truth.energy(cores[k], 3))) < 1e-15
        e_pairs = sum(truth.energy(p, 2) for p in pairs[3 * k:3 * k + 3])
        assert abs(ref["nonadditive"][k] - (ref["interaction"][k] - e_pairs)) < 1e-18
    assert np.all(np.abs(ref["nonadditive"]) > 0) and np.all(np.abs(ref["induction"]) > 0)


def test_heldout_counts_on_pol(monkeypatch):
    """the near-rigid and training-domain filters on q-AQUA-pol's pair energies (counts of notes/e3qp-preregistration.md)."""
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    water = importlib.import_module("e3_water")
    truth = W.qaqua_pol_truth(ROOT)
    a = SimpleNamespace(heldout="2b,4b", heldout_oh_tol=0.03, heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02,
                        heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0, heldout_max=0)
    sets = water.prepare_heldout(a, truth)
    assert (sets["2b"]["near_rigid"], sets["2b"]["in_domain"]) == (5952, 4937)
    assert (sets["4b"]["near_rigid"], sets["4b"]["in_domain"]) == (275, 267)
