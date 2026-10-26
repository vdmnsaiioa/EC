"""E3-QP8 (v0.0.17): the field term's self and cross terms apart.

Rung.e0_field_self_on / _off: the MLP form's self terms weighted by every molecule's own whole-molecule taper over a
shorter reach than the term's (heads.field_energy_split),
    eps_i = h(s_i; x_i) - sum_B (reach_iB - w_iB) h(s_i; x_i(B)),
the cross terms h(x_i) - sum_B h(x_i(B)) at the term's reach as before -- M_16C2pm_ws.  Rung.e0_field_cross_species: the
MLP's cross terms replaced by degree-2 cross terms with coefficients by species alone (heads.field_cross_species),
    -1/2 e_scale sum_{B != C} [a_Z(i) E_i(B).E_i(C) + c_Z(i) G_i(B):G_i(C)],
beside those self terms -- M_16C2pm_us: strictly 3-body, nothing on a dimer, no gradient into E_0 through them.  Without a
build: the stand-in site model of tests/test_e0_field.py.

E3-QP7's cross rungs (M_16C2pm_wx, M_16C2pm_w2x) must be untouched, to the last digits: GOLDEN_CROSS holds their energies
and force checksums at fixed perturbed parameters, computed with v0.0.16 before any v0.0.17 change
(scratch/e3qp8_golden.py); tests/test_e0_field_cross.py, test_e0_field_whole.py and test_e0_field.py keep the rungs
before them."""
import importlib
import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sseft import bandfields as BF, heads as H, model as M, structure as S, train as T, water as W
from sseft.units import ang_to_bohr

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"
SPLIT = ("M_16C2pm_ws", "M_16C2pm_us")
BASE = "M_16C2pm"

# v0.0.16's readings (scratch/e3qp8_golden.py, run on 16dac8f = v0.0.16 with the E3-QP7 reads, before any v0.0.17 change)
GOLDEN_CROSS = {
    'M_16C2pm_wx': [(-0.004841650102647588, 0.0032856445525385313, 0.0008707232459842071),
               (-0.007760624926943294, 0.013448750643636315, 0.0002851200349664163),
               (5.5516162856594775e-05, -2.9180521711281488e-05, 1.6977030921761296e-07)],
    'M_16C2pm_w2x': [(-0.0069080491871975624, 0.0015132637915419023, 0.0010177640966982091),
               (-0.007826157473763917, 0.01370637885281183, 0.00028635282714551145),
               (5.5516162856150686e-05, -2.918052171136195e-05, 1.6977030921761428e-07)],
}


@pytest.fixture(scope="module")
def E3():
    import sys
    sys.path.insert(0, str(SCRIPTS))
    return importlib.import_module("e3_water")


@pytest.fixture(scope="module")
def tef():
    import sys
    sys.path.insert(0, str(TESTS))
    return importlib.import_module("test_e0_field")


@pytest.fixture(scope="module")
def tew():
    import sys
    sys.path.insert(0, str(TESTS))
    return importlib.import_module("test_e0_field_whole")


def rung(E3, tef, tew, name, **kw):
    """the rung as the field tests build it (width 8), its tapers as E3-QP8 runs them at r_c = 6 A: the term over 5-6 A,
    the self terms over 4-5 A (E3.configure)."""
    r = tew.rung(E3, tef, name, **kw)
    if r.e0_field_self_off > 0:
        r = M.with_rung(r, e0_field_self_on=ang_to_bohr(4.0), e0_field_self_off=ang_to_bohr(5.0))
    return r


def pinned(E3, tef, *geometries):
    out = [W.water_structure(np.asarray(g)) for g in geometries]
    W.pin_site_model(out, tef.SiteTruth()); E3.pin_dispersion(out)
    return out


def tetramer(tef):
    """the contact trimer and a fourth molecule: O-O 2.9-3.8 A, every intermolecular atom pair below 4.8 A."""
    return np.concatenate([tef.A, tef.B, tef.C, tef.molecule((1, 0, 1), 0.9, (0.2, 1.6, -2.6))])


def field_energies(r, p, structs):
    return np.asarray(T.predict_aux(p, r, structs)["E_field"])


def test_e3qp7_cross_rungs_unchanged(E3, tef, tew):
    """v0.0.17 leaves E3-QP7's cross rungs untouched: their energies and forces at fixed parameters as v0.0.16's."""
    structs = tef.structures(E3)
    for name, ref_rows in GOLDEN_CROSS.items():
        got = tef.readings(E3, tew.rung(E3, tef, name), structs)
        for g, ref in zip(got, ref_rows):
            np.testing.assert_allclose(g, ref, rtol=1e-10, atol=1e-14, err_msg=name)


def test_split_rungs_configured_and_recorded(E3, tef, tew):
    a = tef.run_args()
    ws, us, w = (E3.configure(E3.RUNGS[n], a) for n in ("M_16C2pm_ws", "M_16C2pm_us", "M_16C2pm_w"))
    for r in (ws, us):
        assert r.e0_field and r.e0_field_whole and r.e0_field_form == "mlp" and not r.e0_field_cross
        assert abs(r.e0_field_self_on - ang_to_bohr(4.0)) < 1e-12 and abs(r.e0_field_self_off - ang_to_bohr(5.0)) < 1e-12
        assert (r.e0_field_on, r.e0_field_off) == (w.e0_field_on, w.e0_field_off)          # the cross terms' reach kept
    assert M.with_rung(ws, name=w.name, e0_field_self_on=0.0, e0_field_self_off=0.0) == w     # M_16C2pm_w, its self terms cut
    assert not ws.e0_field_cross_species and us.e0_field_cross_species
    assert M.with_rung(us, name=ws.name, e0_field_cross_species=False) == ws
    rec_ws, rec_us, rec_w = (E3.model_record(n, a) for n in ("M_16C2pm_ws", "M_16C2pm_us", "M_16C2pm_w"))
    assert rec_ws["e0_field_self_taper_angstrom"] == [4.0, 5.0] and "e0_field_cross_form" not in rec_ws
    assert rec_us["e0_field_cross_form"] == "species" and rec_us["e0_field_self_taper_angstrom"] == [4.0, 5.0]
    assert {k: v for k, v in rec_ws.items() if k != "e0_field_self_taper_angstrom"} == rec_w
    # the rungs before keep their records (no new key)
    for name in ("M_16pm_f", "M_16C2pm_f", "M_inf_f", "M_16pm_w", "M_16C2pm_w", "M_16C2pm_w2", "M_16C2pm_wx", "M_16C2pm_w2x"):
        rec = E3.model_record(name, a)
        assert "e0_field_self_taper_angstrom" not in rec and "e0_field_cross_form" not in rec
    # the self taper follows r_c
    r5 = E3.configure(E3.RUNGS["M_16C2pm_ws"], tef.run_args(rcut=5.0))
    assert abs(r5.e0_field_self_on - ang_to_bohr(3.0)) < 1e-12 and abs(r5.e0_field_self_off - ang_to_bohr(4.0)) < 1e-12


def test_split_rungs_refused_where_undefined(E3, tef, tew):
    ws = rung(E3, tef, tew, "M_16C2pm_ws")
    key = jax.random.PRNGKey(0)
    for bad in (dict(e0_field_form="quadratic"), dict(e0_field_cross=True), dict(e0_field_whole=False),
                dict(e0_field_self_on=ang_to_bohr(5.0), e0_field_self_off=ang_to_bohr(4.0)),
                dict(e0_field_self_off=ang_to_bohr(6.5))):
        with pytest.raises(ValueError, match="e0_field_self|e0_field_cross_species"):
            M.init_params(key, M.with_rung(ws, **bad), 0.02)
    us = rung(E3, tef, tew, "M_16C2pm_us")
    with pytest.raises(ValueError, match="e0_field_cross_species"):
        M.init_params(key, M.with_rung(us, e0_field_cross=True), 0.02)
    st = tef.structures(E3, ("trimer",))[0]
    pos, X, q, frag, src, frag_src = tew.sites(st)
    with pytest.raises(ValueError, match="per_frag"):
        BF.local_field_tensors(pos, X, q, src, ang_to_bohr(0.75), tew.ON, tew.OFF, frag_src=frag_src,
                               self_taper=(ang_to_bohr(4.0), ang_to_bohr(5.0)))


def test_split_rungs_start_as_their_base(E3, tef, tew):
    """the term's output layer and the species coefficients start at zero: at initialisation both rungs are M_16C2pm,
    digit for digit, and M_16C2pm_ws's parameters are M_16C2pm_w's."""
    structs = tef.structures(E3)
    base = rung(E3, tef, tew, BASE)
    p0 = M.init_params(jax.random.PRNGKey(3), base, 0.02)
    E0, F0 = T.predict(p0, base, structs)
    pw = M.init_params(jax.random.PRNGKey(3), rung(E3, tef, tew, "M_16C2pm_w"), 0.02)
    for name in SPLIT:
        r = rung(E3, tef, tew, name)
        p = M.init_params(jax.random.PRNGKey(3), r, 0.02)
        for x, y in zip(jax.tree_util.tree_leaves({k: v for k, v in p.items() if k != "field_x"}), jax.tree_util.tree_leaves(pw)):
            assert np.array_equal(np.asarray(x), np.asarray(y))
        assert ("field_x" in p) == (name == "M_16C2pm_us")
        if "field_x" in p:
            assert all(not np.any(np.asarray(x)) for x in jax.tree_util.tree_leaves(p["field_x"]))
        E, F = T.predict(p, r, structs)
        assert np.array_equal(E, E0) and all(np.array_equal(f, g) for f, g in zip(F, F0))


def test_heads_split_and_species():
    """heads.field_energy_split: with w = reach it is field_energy's term (no parts), with w = 0 its cross terms
    (field_energy with parts), and in between the cross terms plus sum_B w_iB h(x_i(B)) -- each h from the head on the
    part alone; without the MLP's cross terms the self terms alone.  heads.field_cross_species: -1/2 e_scale times the
    cross terms sum_{B != C} written out from the coefficients; zero to the last digit where one fragment is in reach."""
    key = jax.random.PRNGKey(11)
    F, N, K = 8, 5, 4
    k = jax.random.split(key, 9)
    s, v, t = (jax.random.normal(k[0], (N, F)), jax.random.normal(k[1], (N, 3, F)), jax.random.normal(k[2], (N, 3, 3, F)))
    reach = (jax.random.uniform(k[3], (N, K)) > 0.35).astype(float).at[:, 0].set(1.0)
    phi_f = 0.02 * jax.random.normal(k[4], (N, K)) * reach
    E_f = 0.02 * jax.random.normal(k[5], (N, K, 3)) * reach[..., None]
    G_f = 0.02 * jax.random.normal(k[6], (N, K, 3, 3)) * reach[..., None, None]
    w = jax.random.uniform(k[7], (N, K)) * reach
    phi, E, G = phi_f.sum(1), E_f.sum(1), G_f.sum(1)
    mask, e_scale, unit = jnp.ones(N).at[-1].set(0.0), 0.02, 0.01
    move = lambda p: jax.tree_util.tree_map(lambda x: x + 0.1 * jnp.cos(jnp.arange(x.size, dtype=x.dtype)).reshape(x.shape), p)
    p1 = move(H.init_field_head(k[8], {"F": F}))
    parts4 = (phi_f, E_f, G_f, reach)
    split = lambda ww, cross=True: np.asarray(H.field_energy_split(p1, s, v, t, phi, E, G, mask, e_scale, unit, (*parts4, ww),
                                                                   mlp_cross=cross))
    total = np.asarray(H.field_energy(p1, s, v, t, phi, E, G, mask, e_scale, unit))
    crossonly = np.asarray(H.field_energy(p1, s, v, t, phi, E, G, mask, e_scale, unit, parts=parts4))
    h_B = [np.asarray(H.field_energy(p1, s, v, t, phi_f[:, B], E_f[:, B], G_f[:, B], mask, e_scale, unit)) for B in range(K)]
    np.testing.assert_allclose(split(reach), total, rtol=1e-11, atol=1e-18)
    np.testing.assert_allclose(split(0.0 * reach), crossonly, rtol=1e-11, atol=1e-18)
    selfs = sum(np.asarray(w[:, B]) * h_B[B] for B in range(K))
    np.testing.assert_allclose(split(w), crossonly + selfs, rtol=1e-11, atol=1e-18)
    np.testing.assert_allclose(split(w, cross=False), selfs, rtol=1e-11, atol=1e-18)
    assert np.max(np.abs(selfs)) > 1e-5 and np.max(np.abs(crossonly)) > 1e-5
    # the species cross terms
    Z = jnp.array([8, 1, 1, 8, 1])
    px = {"a": jnp.zeros(56).at[1].set(0.7).at[8].set(-1.3), "c": jnp.zeros(56).at[1].set(0.4).at[8].set(2.1)}
    got = np.asarray(H.field_cross_species(px, Z, E, G, mask, e_scale, unit, (*parts4, w)))
    Eu, Gu = np.asarray(E_f) / unit, np.asarray(G_f) / unit
    a, c = np.asarray(px["a"])[np.asarray(Z)], np.asarray(px["c"])[np.asarray(Z)]
    cross = sum(a * np.sum(Eu[:, B] * Eu[:, C], -1) + c * np.sum(Gu[:, B] * Gu[:, C], axis=(-2, -1))
                for B in range(K) for C in range(K) if B != C)
    np.testing.assert_allclose(got, -0.5 * e_scale * cross * np.asarray(mask), rtol=1e-11, atol=1e-18)
    assert np.max(np.abs(got)) > 1e-5
    one = jnp.zeros((N, K)).at[:, 2].set(1.0)
    parts1 = (phi_f * one, E_f * one[..., None], G_f * one[..., None, None], one, one)
    tot = (parts1[0].sum(1), parts1[1].sum(1), parts1[2].sum(1))
    assert np.array_equal(np.asarray(H.field_cross_species(px, Z, tot[1], tot[2], mask, e_scale, unit, parts1)), np.zeros(N))


def test_self_weights(E3, tef, tew):
    """local_field_tensors' w_self: every molecule's own whole-molecule taper over the self reach where it is in reach
    (computed apart here), never above its reach; the parts and totals are v0.0.16's."""
    st = tef.structures(E3, ("taper",))[0]
    pos, X, q, frag, src, frag_src = tew.sites(st)
    l, s_on, s_off = ang_to_bohr(0.75), ang_to_bohr(4.0), ang_to_bohr(5.0)
    phi, E, G, parts = BF.local_field_tensors(pos, X, q, src, l, tew.ON, tew.OFF, frag_src=frag_src, per_frag=True)
    phi2, E2, G2, parts2 = BF.local_field_tensors(pos, X, q, src, l, tew.ON, tew.OFF, frag_src=frag_src, per_frag=True,
                                                  self_taper=(s_on, s_off))
    for x, y in zip((phi, E, G, *parts), (phi2, E2, G2, *parts2[:4])):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    reach, w_self = np.asarray(parts2[3]), np.asarray(parts2[4])
    r = np.linalg.norm(np.asarray(pos)[:, None] - np.asarray(X)[None], axis=-1)
    x = np.clip((r - s_on) / (s_off - s_on), 0.0, 1.0)
    t = 1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x ** 2)
    for B in range(3):
        wB = np.where(st.frag == B, 0.0, np.prod(np.where(frag[None, :] == B, t, 1.0), axis=1)) * reach[:, B]
        np.testing.assert_allclose(w_self[:, B], wB, rtol=1e-12, atol=1e-15)
    assert np.all(w_self <= reach) and np.any((w_self == 0) & (reach == 1)) and np.any(w_self == 1)


def test_self_terms_beyond_their_reach(E3, tef, tew):
    """a dimer whose second molecule is in the term's reach of one H atom and beyond the self reach of every atom: the
    field term of M_16C2pm_w is on, M_16C2pm_ws's and M_16C2pm_us's zero -- energy and every force their base's."""
    A = W.monomer_positions()
    st = pinned(E3, tef, np.concatenate([A, A + np.array([5.8, 0.0, 0.0])]))[0]
    pos, X, q, frag, src, frag_src = tew.sites(st)
    _, _, _, parts = BF.local_field_tensors(pos, X, q, src, ang_to_bohr(0.75), tew.ON, tew.OFF, frag_src=frag_src,
                                            per_frag=True, self_taper=(ang_to_bohr(4.0), ang_to_bohr(5.0)))
    assert np.asarray(parts[3]).sum() >= 1 and not np.any(np.asarray(parts[4]))
    base = rung(E3, tef, tew, BASE)
    E0, F0 = T.predict(tef.perturbed_params(base), base, [st])
    r_w = rung(E3, tef, tew, "M_16C2pm_w")
    assert abs(field_energies(r_w, tef.perturbed_params(r_w), [st])[0]) > 1e-9
    for name in SPLIT:
        r = rung(E3, tef, tew, name)
        p = tef.perturbed_params(r)
        assert field_energies(r, p, [st])[0] == 0.0
        E, F = T.predict(p, r, [st])
        np.testing.assert_allclose(E, E0, rtol=0, atol=1e-15)
        np.testing.assert_allclose(F[0], F0[0], rtol=0, atol=1e-15)


def test_on_a_dimer_the_self_terms_weighted(E3, tef, tew):
    """on a dimer the cross terms vanish: M_16C2pm_ws's and M_16C2pm_us's field term is M_16C2pm_w's atom by atom times
    the self weight (here M_16C2pm_w with w = 1, M_16C2pm_ws the same parameters), and the two split rungs agree."""
    far = tef.molecule((1, 0, 1), 0.7, (0.4, -1.0, 5.1))
    dimers = pinned(E3, tef, np.concatenate([tef.A, tef.B]), np.concatenate([tef.A, far]))
    r_w, r_ws, r_us = (rung(E3, tef, tew, n) for n in ("M_16C2pm_w", "M_16C2pm_ws", "M_16C2pm_us"))
    pu = tef.perturbed_params(r_us)
    p = {k: v for k, v in pu.items() if k != "field_x"}
    f_ws, f_us = field_energies(r_ws, p, dimers), field_energies(r_us, pu, dimers)
    np.testing.assert_allclose(f_us, f_ws, rtol=1e-12, atol=1e-18)
    for st, f in zip(dimers, f_ws):
        b = {k: jnp.asarray(v[0]) for k, v in S.pad_batch([st], n_freq=r_w.K).items()}
        from sseft import e0 as E0m
        s_, v_, t_, _ = E0m.apply_e0(p["e0"], r_w.e0, b["positions"], b["numbers"], b["mask"], b["cell"], b["pbc"])
        pos, X, q, frag, src, frag_src = tew.sites(st)
        phi, E, G, parts = BF.local_field_tensors(pos, X, q, src, r_w.l1, tew.ON, tew.OFF, frag_src=frag_src, per_frag=True,
                                                  self_taper=(r_ws.e0_field_self_on, r_ws.e0_field_self_off))
        per_atom = np.asarray(H.field_energy(p["field"], s_, v_, t_, phi, E, G, b["mask"], p["heads"]["e_scale"], r_w.e0_field_unit))
        np.testing.assert_allclose(f, np.sum(np.asarray(parts[4]).sum(1) * per_atom), rtol=1e-11, atol=1e-18)
    assert abs(f_ws[0]) > 1e-6


def test_species_cross_strictly_three_body(E3, tef, tew):
    """M_16C2pm_us's species cross terms (its field term less the same with a = c = 0) on every sub-cluster of a
    tetramer: zero on the dimers, a 3-body on the trimers, and no 4-body -- to the last digits; its self terms carry a
    4-body through E_0's features."""
    P = tetramer(tef)
    subs = [c for k in range(2, 5) for c in itertools.combinations(range(4), k)]
    structs = pinned(E3, tef, *[np.concatenate([P[3 * i:3 * i + 3] for i in c]) for c in subs])
    r = rung(E3, tef, tew, "M_16C2pm_us")
    pu = tef.perturbed_params(r)
    p0 = {**pu, "field_x": jax.tree_util.tree_map(jnp.zeros_like, pu["field_x"])}
    f, f0 = field_energies(r, pu, structs), field_energies(r, p0, structs)
    terms = {}
    for j, c in enumerate(subs):
        terms[c] = (f[j] - f0[j]) - sum(terms[d] for k in range(2, len(c)) for d in itertools.combinations(c, k))
    selfs = {}
    for j, c in enumerate(subs):
        selfs[c] = f0[j] - sum(selfs[d] for k in range(2, len(c)) for d in itertools.combinations(c, k))
    assert max(abs(terms[c]) for c in subs if len(c) == 2) < 1e-17               # the parameters fold as constants
    assert max(abs(terms[c]) for c in subs if len(c) == 3) > 1e-6
    assert abs(terms[(0, 1, 2, 3)]) < 1e-17
    assert abs(selfs[(0, 1, 2, 3)]) > 1e-10


def test_species_cross_no_gradient_into_e0(E3, tef, tew):
    """the species cross terms read no feature of E_0: E_0's gradient of M_16C2pm_us's energy is the same with the
    coefficients and without them."""
    st = tef.structures(E3, ("trimer",))[0]
    r = rung(E3, tef, tew, "M_16C2pm_us")
    pu = tef.perturbed_params(r)
    b = {k: jnp.asarray(v[0]) for k, v in S.pad_batch([st], n_freq=r.K).items()}
    grad = lambda fx: jax.grad(lambda e0: M.energy_single({**pu, "e0": e0, "field_x": fx}, r, b)[0])(pu["e0"])
    g1, g0 = grad(pu["field_x"]), grad(jax.tree_util.tree_map(jnp.zeros_like, pu["field_x"]))
    for x, y in zip(jax.tree_util.tree_leaves(g1), jax.tree_util.tree_leaves(g0)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=0, atol=1e-16)
    gx = jax.grad(lambda fx: M.energy_single({**pu, "field_x": fx}, r, b)[0])(pu["field_x"])
    assert float(jnp.max(jnp.abs(gx["a"]))) > 1e-8 and float(jnp.max(jnp.abs(gx["c"]))) > 1e-10


def test_split_terms_local(E3, tef, tew):
    """a probe 7.3 A from its core interacts as with the base rung, energies and every force; on the trimer at contact
    and on the trimer with a molecule straddling the taper the term is on, and M_16C2pm_ws differs from M_16C2pm_w."""
    structs = tef.structures(E3)
    base = rung(E3, tef, tew, BASE)
    E0, F0 = T.predict(tef.perturbed_params(base), base, structs)
    r_w = rung(E3, tef, tew, "M_16C2pm_w")
    Ew, _ = T.predict(tef.perturbed_params(r_w), r_w, structs)
    for name in SPLIT:
        r = rung(E3, tef, tew, name)
        E, F = T.predict(tef.perturbed_params(r), r, structs)
        np.testing.assert_allclose(E[2], E0[2], rtol=0, atol=1e-15)
        np.testing.assert_allclose(F[2], F0[2], rtol=0, atol=1e-15)
        assert abs(E[0] - E0[0]) > 1e-6 and abs(E[1] - E0[1]) > 1e-8
        assert abs(E[1] - Ew[1]) > 1e-8


@pytest.mark.parametrize("name", SPLIT)
def test_split_rung_forces_are_the_energy_gradient(E3, tef, tew, name):
    """autodiff forces against central differences of the energy (h = 1e-5 bohr), with the third molecule straddling the
    taper, on atoms of every molecule."""
    r = rung(E3, tef, tew, name)
    p = tef.perturbed_params(r)
    st = tef.structures(E3, ("taper",))[0]
    b = {k: jnp.asarray(v[0]) for k, v in S.pad_batch([st], n_freq=r.K).items()}
    energy = lambda pos: M.energy_single(p, r, {**b, "positions": pos})[0]
    f, g = jax.jit(energy), jax.jit(jax.grad(energy))
    pos0 = b["positions"]
    F = -np.asarray(g(pos0))
    _, F_pred = T.predict(p, r, [st])
    np.testing.assert_allclose(F_pred[0], F, rtol=1e-12, atol=1e-16)
    h = 1e-5
    for atom, axis in ((1, 0), (2, 2), (6, 1), (7, 0), (8, 2)):
        e = jnp.zeros_like(pos0).at[atom, axis].set(h)
        fd = -(float(f(pos0 + e)) - float(f(pos0 - e))) / (2 * h)
        assert abs(fd - F[atom, axis]) < 2e-9, (atom, axis, fd, F[atom, axis])


def test_split_rung_invariant(E3, tef, tew):
    """rotated, translated and with its molecules reordered, the trimer with a straddling molecule has the same energy
    under both rungs, and the forces rotate and reorder with it."""
    P = tef.GEOMETRIES["taper"]
    R = W.rotation((0.3, -1.0, 0.7), 1.234)
    order = [2, 0, 1]
    Q = np.concatenate([(P[3 * m:3 * m + 3] @ R.T + np.array([1.0, -2.0, 0.5])) for m in order])
    a, b = pinned(E3, tef, P, Q)
    for name in SPLIT:
        r = rung(E3, tef, tew, name)
        (Ea, Eb), (Fa, Fb) = T.predict(tef.perturbed_params(r), r, [a, b])
        np.testing.assert_allclose(Eb, Ea, rtol=1e-11)
        Fa_moved = np.concatenate([Fa[3 * m:3 * m + 3] @ R.T for m in order])
        np.testing.assert_allclose(Fb, Fa_moved, rtol=1e-8, atol=1e-12)
