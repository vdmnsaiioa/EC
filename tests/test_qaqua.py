"""q-AQUA v1 as the E3-Q truth (sseft/qaqua.py, sseft.water.QAquaTruth, the held-out reading of scripts/e3_water.py).
Skipped unless a build exists: $QAQUA_ROOT or external/qaqua/build (sh external/qaqua/build.sh)."""
import importlib
import math
import os
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ROOT = os.environ.get("QAQUA_ROOT", str(REPO / "external" / "qaqua" / "build"))
pytestmark = pytest.mark.skipif(not os.path.isfile(os.path.join(ROOT, "libqaqua.so")),
                                reason="no q-AQUA build (sh external/qaqua/build.sh, or set QAQUA_ROOT)")

from sseft import qaqua as Q, water as W, model as M, structure as S  # noqa: E402
from sseft.units import ang_to_bohr  # noqa: E402


@pytest.fixture(scope="module")
def qa():
    return Q.QAqua(ROOT)


@pytest.fixture(scope="module")
def prism(qa):
    xq = Q._read_xyz_qaqua_order(os.path.join(qa.src, "t01_prism.xyz")) / Q.AUANG
    return qa.from_qaqua_order(xq.ravel(), 6)                    # bohr, O, H, H per molecule


def test_authors_benchmarks_and_long_range(capsys):
    """the README's trimer and prism hexamer (energies and gradients to the printed 1e-8), and two monomers 12 A apart
    interacting by exactly the point charges of their LTP2011 charges."""
    assert Q.check(ROOT)


def test_atom_order_round_trip(qa, prism):
    flat = qa.to_qaqua_order(prism)
    np.testing.assert_array_equal(qa.from_qaqua_order(flat, 6), prism)
    assert np.allclose(flat.reshape(-1, 3)[:2], prism[1:3]) and np.allclose(flat.reshape(-1, 3)[12], prism[0])


def test_interaction_energy_of_monomers_is_zero(qa, prism):
    E, F = qa.interaction_energy_forces(prism[:3])
    assert E == 0.0 and np.all(F == 0.0)
    # q-AQUA's Coulomb has no cutoff: two monomers 1e4 bohr apart interact by their point charges (~1e-13 E_h)
    far = np.concatenate([prism[:3], prism[3:6] + np.array([0.0, 0.0, 1e4])])
    q = qa.molecule_charges(far)
    coul = np.sum(np.outer(q[:3], q[3:]) / np.linalg.norm(far[:3, None] - far[None, 3:], axis=-1))
    # the interaction energy is the total minus the 1-body terms: its rounding floor is eps x |total| (these prism
    # monomers are distorted, |total| ~ 2e-3 E_h; for the rigid monomers of E3-Q it is ~1.5e-6 E_h)
    assert abs(qa.interaction_energy(far) - coul) < 4 * np.finfo(float).eps * abs(qa.total_energy(far))


def test_forces_integrate_to_the_energy(qa, prism):
    """the energy carries ~1e-10 E_h of rounding noise (cancellation in the polynomial sums), which defeats small-step
    finite differences; the gradient is checked by integrating it along a path instead."""
    rng = np.random.default_rng(3)
    for n in (2, 3, 6):
        P = prism[:3 * n]
        v = rng.normal(size=P.shape); v /= np.linalg.norm(v)
        s = np.linspace(0.0, 0.05, 201)
        g = np.array([-np.vdot(qa.interaction_energy_forces(P + t * v)[1], v) for t in s])
        w = np.ones(len(s)); w[1:-1:2] = 4; w[2:-1:2] = 2
        integral = (s[1] - s[0]) / 3 * np.sum(w * g)
        dE = qa.interaction_energy(P + s[-1] * v) - qa.interaction_energy(P)
        assert abs(dE - integral) < 1e-9, (n, dE, integral)


def test_nbody_decomposition_is_exact(qa, prism):
    nb = qa.nbody(prism)
    assert set(nb) == {2, 3, 4}
    assert abs(sum(nb.values()) - qa.interaction_energy(prism)) < 1e-12


def test_truth_object_and_pre_registered_numbers():
    truth = W.qaqua_truth(ROOT)
    assert abs(truth.r_e - 2.93459) < 2e-5                      # the rigid dimer's minimum at phi = 0 (pre-registered)
    assert abs(W.dipole_dipole_coefficient(truth) + 0.6491) < 5e-4
    q = truth.molecule_charges(W.monomer_positions())
    np.testing.assert_allclose(q, [-0.6577222, 0.3288611, 0.3288611], atol=1e-6)
    st = W.monomer_structure(truth)
    np.testing.assert_array_equal(st.pinned["q"], q)
    # the dimer tail beyond 7.8 A is the pinned-charge Coulomb exactly
    d = W.dimer_scan(np.array([9.0]), truth=truth)[0]
    pos = d.positions; qq = d.pinned["q"]
    r = np.linalg.norm(pos[:3, None] - pos[None, 3:], axis=-1)
    assert abs(d.energy - np.sum(np.outer(qq[:3], qq[3:]) / r)) < 1e-12 * abs(d.energy)


def test_cluster_generation_on_the_truth():
    truth = W.qaqua_truth(ROOT)
    a = W.cluster_dataset([3], 2, seed=1, truth=truth, mc_steps=20)
    b = W.cluster_dataset([3], 2, seed=1, truth=truth, mc_steps=20)
    from sseft.units import bohr_to_ang
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.positions, y.positions)
        assert x.energy == y.energy and x.energy < 0
        E, F = truth.energy_forces(bohr_to_ang(x.positions), 3)
        assert abs(E - x.energy) < 1e-12 * abs(E) and np.allclose(F, x.forces, rtol=1e-9, atol=1e-13)
        np.testing.assert_allclose(x.pinned["q"], np.tile(truth.molecule_charges(W.monomer_positions()), 3), atol=1e-9)


def test_raw_sets_and_the_near_rigid_filter():
    pos, lab = Q.read_raw(os.path.join(ROOT, "q-AQUA", "3b_data.xyz"), 3)
    assert pos.shape == (45332, 9, 3) and len(lab) == 45332
    keep = Q.near_rigid(pos, W.R_OH, math.degrees(W.THETA), 0.03, 5.0)
    assert int(keep.sum()) == 4699                               # pre-registered count
    r, th = Q.monomer_internal(pos[keep])
    assert np.all(np.abs(r - W.R_OH) <= 0.03) and np.all(np.abs(th - 104.52) <= 5.0)


def test_heldout_preparation_matches_the_registered_counts(monkeypatch):
    """the near-rigid and training-domain filters of the held-out reading (counts of notes/e3q-preregistration.md),
    the PES's n-body values and the charges of the frames, the frame cap."""
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    water = importlib.import_module("e3_water")
    truth = W.qaqua_truth(ROOT)
    a = SimpleNamespace(heldout="2b,4b", heldout_oh_tol=0.03, heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02,
                        heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0, heldout_max=40)
    sets = water.prepare_heldout(a, truth)
    assert (sets["2b"]["near_rigid"], sets["2b"]["in_domain"]) == (5952, 4940)
    assert (sets["4b"]["near_rigid"], sets["4b"]["in_domain"]) == (275, 267)
    d = sets["4b"]
    assert len(d["label"]) == 40 and d["pos"].shape == (40, 12, 3) and d["charges"].shape == (40, 12)
    for p, pes, q in zip(d["pos"][:3], d["pes"][:3], d["charges"][:3]):
        assert pes == truth.qa.nbody(ang_to_bohr(p), max_order=4)[4]
        np.testing.assert_array_equal(q, truth.molecule_charges(p))
    pairs = [truth.energy(np.concatenate([p[3 * i:3 * i + 3], p[3 * j:3 * j + 3]]), 2) for p in d["pos"]
             for i in range(4) for j in range(i + 1, 4)]
    assert max(pairs) * 627.5094740631 <= 2.0
    a.heldout_max_pair_kcal = 0.0                                  # no domain cap
    assert water.prepare_heldout(SimpleNamespace(**{**vars(a), "heldout": "4b"}), truth)["4b"]["in_domain"] == 275


def test_model_nbody_vanishes_for_a_pairwise_model(monkeypatch):
    """the held-out reading's subset expansion: a model that is exactly pairwise additive (the Coulomb band of pinned
    charges, no E_0, no fragment references) has zero 3- and 4-body energies to rounding."""
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    water = importlib.import_module("e3_water")
    import jax
    # fragments on (as every real rung): E_int has no 1-body term, which the subset expansion assumes
    rung = M.with_rung(M.RUNGS["M_1"], pin_q=True, site_energies=False)
    params = M.init_params(jax.random.PRNGKey(0), rung)
    truth = W.qaqua_truth(ROOT)
    frames = np.array([W.cluster_dataset([4], 1, seed=s, truth=truth, mc_steps=5)[0].positions / 1.8897259886 for s in (1, 2)])
    charges = np.array([truth.molecule_charges(f) for f in frames])
    for n in (3, 4):
        e = water.model_nbody(params, rung, frames[:, :3 * n], charges[:, :3 * n], n)
        assert np.all(np.abs(e) < 1e-14), e
    e2 = water.model_nbody(params, rung, frames[:, :6], charges[:, :6], 2)
    assert np.all(np.abs(e2) > 1e-6)                             # the 2-body term itself is not zero
