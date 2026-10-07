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
