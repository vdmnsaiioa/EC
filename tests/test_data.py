"""
The published pair potentials: each parameter set must reproduce its paper's well depth and position to the
quoted digits (the provenance check, see sseft/data.py), and the Rostock form must reduce to its asymptotic
series at long range.  The Tang-Toennies truth must reproduce the TT 2003 well of Ar2.
"""
import numpy as np
from scipy.optimize import minimize_scalar
import pytest
from sseft import data as D
from sseft.units import ang_to_bohr, bohr_to_ang


@pytest.mark.parametrize("name", list(D.PUBLISHED))
def test_published_well(name):
    p = D.PUBLISHED[name]; V = D.published_potential(name)
    res = minimize_scalar(V, bracket=(ang_to_bohr(2.5), ang_to_bohr(10 * p["Re_nm"]), ang_to_bohr(6.0)))
    eps_K = -res.fun * D.KELVIN_PER_HARTREE
    assert abs(eps_K - p["eps_K"]) < 1.5e-3, (name, eps_K)                # the paper's eps/k_B to the last digit
    assert abs(res.x * D.NM_PER_BOHR - p["Re_nm"]) < 1.5e-5, (name, res.x * D.NM_PER_BOHR)
    # asymptotics: V R^6 -> -C6 (the potential's own C6), within the C8 term, at 40 A
    R = ang_to_bohr(40.0)
    c6 = D.published_c2n(name, 6); c8 = D.published_c2n(name, 8)
    assert abs(V(R) * R ** 6 + c6) < 1.2 * c8 / R ** 2


def test_published_c6_near_dosd():
    """the ab initio C6's sit within 1 % of DOSD (a sanity check on the unit conversion, not a test of the papers)."""
    assert abs(D.published_c2n("Ar2", 6) / D.DOSD_C6[("Ar", "Ar")] - 1) < 0.01
    assert abs(D.published_c2n("Ne2", 6) / D.DOSD_C6[("Ne", "Ne")] - 1) < 0.01


def test_tt_ar2_well():
    V = D.tt_potential("Ar2")
    res = minimize_scalar(V, bracket=(ang_to_bohr(3.0), ang_to_bohr(3.75), ang_to_bohr(5.0)))
    assert abs(bohr_to_ang(res.x) - 3.7565) < 2e-3
    assert abs(-res.fun * 219474.63 - 99.55) < 0.3          # D_e in cm^-1


def test_water_truth_and_fragments():
    """the synthetic water truth: dipole 1.855 D by construction, E R^3 -> -0.652 at phi = 0, forces against finite
    differences; the fragment reference makes two far monomers exactly non-interacting in the model."""
    import math, jax
    from sseft import water as W, model as M, train as T
    assert abs(2 * W.Q_H * W.R_OH * math.cos(W.THETA / 2) * W.DEBYE_PER_E_ANG - 1.855) < 1e-9
    E200 = W.pw_energy_forces(W.dimer_positions(200.0), 2)[0]
    assert abs(E200 * ang_to_bohr(200.0) ** 3 + 0.652) < 0.01
    pos = W.dimer_positions(3.0); E0, F = W.pw_energy_forces(pos, 2); h = 1e-4
    p1 = pos.copy(); p1[1, 2] += h; p2 = pos.copy(); p2[1, 2] -= h
    fd = -(W.pw_energy_forces(p1, 2)[0] - W.pw_energy_forces(p2, 2)[0]) / (2 * h) * bohr_to_ang(1.0)
    assert abs(fd - F[1, 2]) < 1e-7 * abs(fd)
    rung = M.RUNGS["M_1"]; params = M.init_params(jax.random.PRNGKey(0), rung, 1e-3)
    far = W.dimer_scan(np.array([300.0])); E, _ = T.predict(params, rung, far)
    aux = T.predict_aux(params, rung, far)
    # beyond r_c the interaction energy is the Coulomb band of the learned charges and nothing else: E_0's part and
    # the intramolecular band cancel exactly against the fragment references (no constant offset)
    assert abs(E[0] - float(aux["E0"][0] + aux["E_coul"][0] - np.sum(aux["E_frag"][0]))) < 1e-12 and abs(E[0]) < 1e-8


def test_embedded_dagger_identity():
    """pinned self-consistent induced dipoles + the pinned polarisation work 1/2 |mu|^2 / alpha make the Coulomb band
    reproduce the truth's electrostatics + induction exactly when the band's kernel is the truth's (l_1 = lambda)."""
    import jax, sys, os
    from sseft import water as W, model as M, train as T
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts")); import e3_water as E3
    rng = np.random.default_rng(3)
    pos = W.random_cluster(3, rng, mc_steps=5)                                   # a relaxed trimer, A
    st = W.water_structure(pos, energy=W.pw_energy_forces(pos, 3)[0]); E3.pin_embedded([st])
    c = W.pw_components(pos, 3)
    rung = M.with_rung(E3.RUNGS["M_1pemb"], l1=W.PW["lam"], site_energies=False)        # PW["lam"] is in bohr
    params = M.init_params(jax.random.PRNGKey(0), rung, 1e-3)
    E, _ = T.predict(params, rung, [st])
    assert abs(E[0] - (c["es"] + c["ind"])) < 1e-10, (E[0], c["es"] + c["ind"])
    # without the self energy the band double counts the polarisation work: off by exactly 1/2 sum mu^2 / alpha - 2 sum_{i<j} mu T mu,
    # i.e. by more than the many-body energy itself
    E3.pin_embedded([st], self_energy=False); E0, _ = T.predict(params, rung, [st])
    assert abs(E0[0] - (c["es"] + c["ind"])) > abs(c["ind"] - c["ind1"])
