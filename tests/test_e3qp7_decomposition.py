"""E3-QP7's decomposition (scratch/e3qp7_decomposition.py): every model's n-body energies by component -- E_0, the pinned
bands, the read-out, the field term -- and the field term split into its self and cross terms and into its value at the
isolated molecule's features (the coefficients fixed) and the rest.  Without a build: the stand-in site model of
tests/test_e0_field.py, on a tetramer whose every atom pair lies within the field term's full weight (below 5 A).

  - the Moebius inversion is E3.model_nbody's recursion;
  - the components sum to the model's interaction energy (train.predict_energies), and the field term read there is the
    model's own;
  - the cross part read with a rung's parameters is the cross rung's field term with the same parameters (v0.0.16), and
    nothing on a dimer;
  - the pinned bands carry no 3-body; at fixed coefficients the self terms carry no 3-body and the degree-2 cross terms
    no 4-body -- what they carry beyond is their coefficients' (E_0's features of the whole environment) -- while the
    MLP form's cross terms carry a 4-body of their own;
  - v0.0.17's rungs: M_16C2pm_ws's cross terms are the cross rung's with the same parameters and its self terms the
    weighted ones; M_16C2pm_us's cross terms (by species) have no coefficient part at all."""
import importlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

from sseft import model as M, train as T
from sseft.units import ang_to_bohr

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
RUNGS = ("M_16C2pm", "M_16C2pm_w", "M_16C2pm_w2", "M_16C2pm_wx", "M_16C2pm_w2x", "M_16C2pm_ws", "M_16C2pm_us")


@pytest.fixture(scope="module")
def mods():
    for p in (ROOT / "scripts", ROOT / "scratch", TESTS):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    E3 = importlib.import_module("e3_water")
    tef = importlib.import_module("test_e0_field")
    tew = importlib.import_module("test_e0_field_whole")
    dec = importlib.import_module("e3qp7_decomposition")
    return E3, tef, tew, dec


def tetramer(tef):
    """tests/test_e0_field.py's contact trimer and a fourth molecule: O-O 2.9-3.8 A, every intermolecular atom pair
    below 4.8 A."""
    return np.concatenate([tef.A, tef.B, tef.C, tef.molecule((1, 0, 1), 0.9, (0.2, 1.6, -2.6))])


def decompose(mods, r, params):
    E3, tef, tew, dec = mods
    structs = dec.build_structures(r, tetramer(tef)[None], 4, tef.SiteTruth(), rigid=False)
    c = dec.evaluate(params, r, dec.batches_for(structs, r.K), len(structs))
    q = dec.quantities(c, r)
    subs = dec.subsets_of(4)
    terms = {k: v[None] @ dec.mobius(subs).T for k, v in q.items() if not k.startswith("_")}
    return structs, c, q, {k: v[0] for k, v in terms.items()}, np.array([len(s) for s in subs])


def test_mobius_is_the_recursion(mods):
    dec = mods[3]
    subs = dec.subsets_of(5)
    E = np.random.default_rng(0).normal(size=len(subs))
    term = {}
    for j, c in enumerate(subs):
        term[c] = E[j] - sum(term[d] for k in range(2, len(c)) for d in itertools.combinations(c, k))
    np.testing.assert_allclose(dec.mobius(subs) @ E, [term[c] for c in subs], rtol=0, atol=1e-12)
    assert subs[-1] == tuple(range(5))


@pytest.mark.parametrize("name", RUNGS)
def test_components(mods, name):
    E3, tef, tew, dec = mods
    r = tew.rung(E3, tef, name)
    p = tef.perturbed_params(r)
    structs, c, q, terms, size = decompose(mods, r, p)
    np.testing.assert_allclose(c["E"], T.predict_energies(p, r, structs), rtol=0, atol=1e-13)
    assert np.max(np.abs(q["_check_sum"])) < 1e-9                       # kcal/mol
    assert np.max(np.abs(terms["bands"][size >= 3])) < 1e-9
    assert np.max(np.abs(terms["E0"][size == 4])) > 1e-6                 # E_0 has every order
    if not r.e0_field:
        assert "field" not in q
        return
    assert np.max(np.abs(q["_check_field"])) < 1e-9
    np.testing.assert_allclose(terms["cross"][size == 2], 0.0, atol=1e-12)
    assert np.max(np.abs(terms["cross"][size == 3])) > 1e-4
    np.testing.assert_allclose(terms["field"], terms["field_fixed"] + terms["field_coef"], rtol=0, atol=1e-12)
    if r.e0_field_cross:
        assert "self" not in q
        np.testing.assert_allclose(terms["cross"], terms["field"], rtol=0, atol=0)
    else:
        np.testing.assert_allclose(terms["field"], terms["self"] + terms["cross"], rtol=0, atol=1e-12)
        assert np.max(np.abs(terms["self"][size == 2])) > 1e-4
        assert np.max(np.abs(terms["self_fixed"][size >= 3])) < 1e-9      # a 2-body at fixed coefficients
        assert np.max(np.abs(terms["self"][size == 3])) > 1e-7           # its 3-body: the coefficients'
    if r.e0_field_cross_species:
        np.testing.assert_allclose(terms["cross_coef"], 0.0, rtol=0, atol=1e-12)          # no coefficient to move
        assert np.max(np.abs(terms["cross"][size == 4])) < 1e-9                         # strictly 3-body
    elif r.e0_field_form == "quadratic":
        assert np.max(np.abs(terms["cross_fixed"][size == 4])) < 1e-9     # a 3-body at fixed coefficients
        assert np.max(np.abs(terms["cross"][size == 4])) > 1e-7          # its 4-body: the coefficients'
    else:
        assert np.max(np.abs(terms["cross_fixed"][size == 4])) > 1e-7    # the MLP's own higher orders in the field


@pytest.mark.parametrize("name", ("M_16C2pm_w", "M_16C2pm_w2"))
def test_cross_part_is_the_cross_rungs(mods, name):
    """the cross part read with a rung's parameters is what the cross rung (Rung.e0_field_cross) gives with them."""
    E3, tef, tew, dec = mods
    r = tew.rung(E3, tef, name)
    rx = M.with_rung(r, e0_field_cross=True)
    p = tef.perturbed_params(r)
    _, c, _, _, _ = decompose(mods, r, p)
    _, cx, _, _, _ = decompose(mods, rx, p)
    np.testing.assert_allclose(c["F_cross"], cx["E_field"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(c["F_cross1"], cx["F_cross1"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(c["E_field"] - c["F_cross"], c["F_self"], rtol=0, atol=1e-15)
    assert not np.any(cx["F_self"]) and not np.any(cx["F_self1"])


def test_split_rungs_read_as_their_parts(mods):
    """v0.0.17: M_16C2pm_ws's cross terms are the cross rung's field term with the same parameters, and its self terms
    M_16C2pm_us's with the same MLP -- the two rungs' channels read apart are the same functions."""
    E3, tef, tew, dec = mods
    ws = M.with_rung(tew.rung(E3, tef, "M_16C2pm_ws"), e0_field_self_on=ang_to_bohr(4.0), e0_field_self_off=ang_to_bohr(5.0))
    us = M.with_rung(ws, name="M_16C2pm_us", e0_field_cross_species=True)
    pu = tef.perturbed_params(us)
    p = {k: v for k, v in pu.items() if k != "field_x"}
    _, c_ws, _, _, _ = decompose(mods, ws, p)
    _, c_us, _, _, _ = decompose(mods, us, pu)
    _, c_wx, _, _, _ = decompose(mods, M.with_rung(ws, e0_field_self_on=0.0, e0_field_self_off=0.0, e0_field_cross=True), p)
    np.testing.assert_allclose(c_ws["F_cross"], c_wx["E_field"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(c_ws["F_self"], c_us["F_self"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(c_ws["F_self1"], c_us["F_self1"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(c_us["F_cross"], c_us["F_cross1"], rtol=0, atol=0)       # nothing of E_0's features in it
    assert np.max(np.abs(c_us["F_cross"])) > 1e-6
