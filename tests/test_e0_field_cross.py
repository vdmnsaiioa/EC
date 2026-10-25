"""E3-QP7 (v0.0.16): the field term without its 2-body.

Rung.e0_field_cross: the term less its value on each source molecule alone,
    eps_i = h(s_i; x_i) - sum_B h(s_i; x_i(B)),      x_i = sum_B x_i(B)   (every molecule whole, Rung.e0_field_whole),
so that it gives nothing to any dimer -- the self terms, a 2-body tied to the cross terms at degree 2 (e3qp6-results.md
section 3.4), are gone -- and carries only what the molecules' fields do together.  At degree 2 (heads.field_energy2) the
cross terms -1/2 sum_{B != C} E_i(B).A_i.E_i(C) + the same in G; for the MLP form (heads.field_energy) h less its value on
each molecule.  bandfields.local_field_tensors(per_frag=True) gives every molecule's share apart.  Without a build: the
stand-in site model of tests/test_e0_field.py.

E3-QP6's field rungs (M_16pm_w, M_16C2pm_w, M_16C2pm_w2) must be untouched, to the last digits: GOLDEN_WHOLE holds their
energies and force checksums at fixed perturbed parameters, computed with v0.0.15 before any v0.0.16 change
(scratch/e3qp7_golden.py); tests/test_e0_field_whole.py and tests/test_e0_field.py keep the rungs before them."""
import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sseft import bandfields as BF, heads as H, model as M, train as T, water as W
from sseft.units import ang_to_bohr

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"
CROSS = ("M_16C2pm_wx", "M_16C2pm_w2x")
PARENT = {"M_16C2pm_wx": "M_16C2pm_w", "M_16C2pm_w2x": "M_16C2pm_w2"}
BASE = "M_16C2pm"

# v0.0.15's readings (scratch/e3qp7_golden.py, run on 217d647 = v0.0.15 before any v0.0.16 change)
GOLDEN_WHOLE = {
    'M_16pm_w': [(-0.006247986311879616, 0.002170285735582782, 0.0008706772465990334),
               (-0.00793816550185844, 0.013456359515603998, 0.00028053335721867675),
               (4.5614331690924104e-05, -2.927428326902172e-05, 1.8000062544591434e-07)],
    'M_16C2pm_w': [(-0.005577869858434781, 0.002046270046024645, 0.0008748939182662804),
               (-0.007398295971076774, 0.013293597458140193, 0.0002671110889886165),
               (5.5516162856150686e-05, -2.9180521711265035e-05, 1.697703092176058e-07)],
    'M_16C2pm_w2': [(-0.006526008843513731, 0.0011635988513225015, 0.001082244264688533),
               (-0.007396502617611489, 0.014041215068646199, 0.00030850228999754296),
               (5.5516162856150686e-05, -2.9180521711288616e-05, 1.6977030921761028e-07)],
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
    return tew.rung(E3, tef, name, **kw)


def pinned(E3, tef, *geometries):
    out = [W.water_structure(np.asarray(g)) for g in geometries]
    W.pin_site_model(out, tef.SiteTruth()); E3.pin_dispersion(out)
    return out


def test_e3qp6_field_rungs_unchanged(E3, tef, tew):
    """v0.0.16 leaves E3-QP6's field rungs untouched: their energies and forces at fixed parameters as v0.0.15's."""
    structs = tef.structures(E3)
    for name, ref_rows in GOLDEN_WHOLE.items():
        got = tef.readings(E3, rung(E3, tef, tew, name), structs)
        for g, ref in zip(got, ref_rows):
            np.testing.assert_allclose(g, ref, rtol=1e-10, atol=1e-14, err_msg=name)


def test_cross_rungs_configured_and_recorded(E3, tef, tew):
    a = tef.run_args()
    for name in CROSS:
        r, parent = E3.configure(E3.RUNGS[name], a), E3.configure(E3.RUNGS[PARENT[name]], a)
        assert r.e0_field and r.e0_field_whole and r.e0_field_cross and not parent.e0_field_cross
        assert r.e0_field_form == ("quadratic" if name == "M_16C2pm_w2x" else "mlp")
        assert M.with_rung(r, name=parent.name, e0_field_cross=False) == parent       # its E3-QP6 rung, less the self terms
        record = E3.model_record(name, a)
        assert record["e0_field_cross"] is True and record["e0_field_weights"] == "molecule"
        assert record["e0_field_taper_angstrom"] == [5.0, 6.0]
        assert {k: v for k, v in record.items() if k != "e0_field_cross"} == E3.model_record(PARENT[name], a)
    # the rungs before keep their records (no new key)
    for name in ("M_16pm_f", "M_16C2pm_f", "M_inf_f", "M_16pm_w", "M_16C2pm_w", "M_16C2pm_w2"):
        assert "e0_field_cross" not in E3.model_record(name, a)
    # a fragment's own value needs its sources under one weight
    with pytest.raises(ValueError, match="e0_field_cross"):
        M.init_params(jax.random.PRNGKey(0), M.with_rung(rung(E3, tef, tew, "M_16C2pm_wx"), e0_field_whole=False), 0.02)
    st = tef.structures(E3, ("trimer",))[0]
    pos, X, q, frag, src, frag_src = tew.sites(st)
    with pytest.raises(ValueError, match="per_frag"):
        BF.local_field_tensors(pos, X, q, src, ang_to_bohr(0.75), tew.ON, tew.OFF, per_frag=True)


def test_cross_rungs_start_as_their_base(E3, tef, tew):
    """the term's output layer starts at zero: at initialisation a cross rung is M_16C2pm, digit for digit."""
    structs = tef.structures(E3)
    base = rung(E3, tef, tew, BASE)
    p0 = M.init_params(jax.random.PRNGKey(3), base, 0.02)
    E0, F0 = T.predict(p0, base, structs)
    for name in CROSS:
        r = rung(E3, tef, tew, name)
        p = M.init_params(jax.random.PRNGKey(3), r, 0.02)
        assert {k: v for k, v in p.items() if k != "field"}.keys() == p0.keys()
        for x, y in zip(jax.tree_util.tree_leaves({k: v for k, v in p.items() if k != "field"}), jax.tree_util.tree_leaves(p0)):
            assert np.array_equal(np.asarray(x), np.asarray(y))
        E, F = T.predict(p, r, structs)
        assert np.array_equal(E, E0) and all(np.array_equal(f, g) for f, g in zip(F, F0))


def test_per_fragment_tensors(E3, tef, tew):
    """local_field_tensors(per_frag=True): every molecule's share is its whole field times its one weight (computed apart,
    as tests/test_e0_field_whole.py does), the totals are v0.0.15's, and a molecule is in reach where its weight is not
    zero -- never the atom's own, never one beyond the taper."""
    st = tef.structures(E3, ("taper",))[0]
    pos, X, q, frag, src, frag_src = tew.sites(st)
    l = ang_to_bohr(0.75)
    tot = BF.local_field_tensors(pos, X, q, src, l, tew.ON, tew.OFF, frag_src=frag_src)
    phi, E, G, (phi_f, E_f, G_f, reach) = BF.local_field_tensors(pos, X, q, src, l, tew.ON, tew.OFF, frag_src=frag_src,
                                                                 per_frag=True)
    for got, ref in zip((phi, E, G), tot):
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-12, atol=1e-16)
    r = np.linalg.norm(np.asarray(pos)[:, None] - np.asarray(X)[None], axis=-1)
    x = np.clip((r - tew.ON) / (tew.OFF - tew.ON), 0.0, 1.0)
    t = 1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x ** 2)
    for B in range(3):
        only_B = src & jnp.asarray(frag[None, :] == B)
        parts = BF.local_field_tensors(pos, X, q, only_B, l, 1e6, 2e6)                 # B's whole field, no taper
        wB = np.where(st.frag == B, 0.0, np.prod(np.where(frag[None, :] == B, t, 1.0), axis=1))
        for got, ref in zip((phi_f[:, B], E_f[:, B], G_f[:, B]), parts):
            np.testing.assert_allclose(np.asarray(got), wB.reshape((-1,) + (1,) * (ref.ndim - 1)) * np.asarray(ref),
                                       rtol=1e-12, atol=1e-16)
        np.testing.assert_array_equal(np.asarray(reach[:, B]), (wB > 0).astype(float))
    assert np.any((np.asarray(reach) == 0) & (np.asarray(st.frag)[:, None] != np.arange(3)[None, :]))   # one beyond reach


def _quadratic_form(c, F, v, t, Ex, Ey, Gx, Gy):
    """the bilinear form behind heads.field_energy2's quadratic, per atom (numpy, from its coefficients)."""
    a, b, bp, cg, dg = c[:, 0], c[:, 1:F + 1], c[:, F + 1:2 * F + 1], c[:, 2 * F + 1], c[:, 2 * F + 2:]
    ts = 0.5 * (t + np.swapaxes(t, 1, 2))
    e = (a * np.sum(Ex * Ey, -1) + np.sum(b * np.einsum("ia,iaf->if", Ex, v) * np.einsum("ia,iaf->if", Ey, v), -1)
         + np.sum(bp * np.einsum("ia,iabf,ib->if", Ex, ts, Ey), -1))
    g = cg * np.sum(Gx * Gy, axis=(-2, -1)) + np.sum(dg * np.einsum("iabf,iab->if", t, Gx) * np.einsum("iabf,iab->if", t, Gy), -1)
    return e + g


def test_heads_less_each_fragment_alone():
    """the heads with parts: degree 2 is -1/2 e_scale times the cross terms sum_{B != C} (written out here from the
    coefficients); the MLP form is h(total) less h of each fragment in reach (each from the head without parts).  Where
    one fragment is in reach, the term is zero to the last digit."""
    key = jax.random.PRNGKey(7)
    F, N, K = 8, 5, 4
    k = jax.random.split(key, 9)
    s, v, t = (jax.random.normal(k[0], (N, F)), jax.random.normal(k[1], (N, 3, F)), jax.random.normal(k[2], (N, 3, 3, F)))
    reach = (jax.random.uniform(k[3], (N, K)) > 0.35).astype(float).at[:, 0].set(1.0)
    phi_f = 0.02 * jax.random.normal(k[4], (N, K)) * reach
    E_f = 0.02 * jax.random.normal(k[5], (N, K, 3)) * reach[..., None]
    G_f = 0.02 * jax.random.normal(k[6], (N, K, 3, 3)) * reach[..., None, None]
    phi, E, G = phi_f.sum(1), E_f.sum(1), G_f.sum(1)
    mask, e_scale, unit = jnp.ones(N).at[-1].set(0.0), 0.02, 0.01
    move = lambda p: jax.tree_util.tree_map(lambda x: x + 0.1 * jnp.cos(jnp.arange(x.size, dtype=x.dtype)).reshape(x.shape), p)
    # degree 2
    p2 = move(H.init_field_head2(k[7], {"F": F}))
    got = H.field_energy2(p2, s, v, t, E, G, mask, e_scale, unit, parts=(phi_f, E_f, G_f, reach))
    c = np.asarray(H.mlp(p2["coef"], s))
    Eu, Gu = np.asarray(E_f) / unit, np.asarray(G_f) / unit
    cross = sum(_quadratic_form(c, F, np.asarray(v), np.asarray(t), Eu[:, B], Eu[:, C], Gu[:, B], Gu[:, C])
                for B in range(K) for C in range(K) if B != C)
    np.testing.assert_allclose(np.asarray(got), -0.5 * e_scale * cross * np.asarray(mask), rtol=1e-11, atol=1e-18)
    assert np.max(np.abs(np.asarray(got))) > 1e-5
    # the MLP form
    p1 = move(H.init_field_head(k[8], {"F": F}))
    got = H.field_energy(p1, s, v, t, phi, E, G, mask, e_scale, unit, parts=(phi_f, E_f, G_f, reach))
    want = H.field_energy(p1, s, v, t, phi, E, G, mask, e_scale, unit) - sum(
        reach[:, B] * H.field_energy(p1, s, v, t, phi_f[:, B], E_f[:, B], G_f[:, B], mask, e_scale, unit) for B in range(K))
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-11, atol=1e-18)
    assert np.max(np.abs(np.asarray(got))) > 1e-5
    # one fragment in reach: nothing
    one = jnp.zeros((N, K)).at[:, 2].set(1.0)
    parts = (phi_f * one, E_f * one[..., None], G_f * one[..., None, None], one)
    tot = (parts[0].sum(1), parts[1].sum(1), parts[2].sum(1))
    assert np.array_equal(np.asarray(H.field_energy2(p2, s, v, t, tot[1], tot[2], mask, e_scale, unit, parts=parts)), np.zeros(N))
    assert np.array_equal(np.asarray(H.field_energy(p1, s, v, t, *tot, mask, e_scale, unit, parts=parts)), np.zeros(N))


def test_nothing_to_any_dimer(E3, tef, tew):
    """with every parameter switched on, a cross rung's dimer -- at contact, straddling the taper, beyond it, and the
    monomer -- is M_16C2pm's, energy and every force: the field term has no 2-body.  Its parent rung's is not."""
    A, B = tef.A, tef.B
    far = tef.molecule((1, 0, 1), 0.7, (0.4, -1.0, 5.1))                       # straddling the taper (the 'taper' third)
    beyond = tef.molecule((1, 2, 0), 1.9, (8.5, 1.0, 0.5))
    dimers = pinned(E3, tef, np.concatenate([A, B]), np.concatenate([A, far]), np.concatenate([B, far]),
                    np.concatenate([A, beyond]), A)
    base = rung(E3, tef, tew, BASE)
    E0, F0 = T.predict(tef.perturbed_params(base), base, dimers)
    for name in CROSS:
        r = rung(E3, tef, tew, name)
        E, F = T.predict(tef.perturbed_params(r), r, dimers)
        np.testing.assert_allclose(E, E0, rtol=0, atol=1e-15, err_msg=name)
        for f, g in zip(F, F0):
            np.testing.assert_allclose(f, g, rtol=0, atol=1e-15, err_msg=name)
        rp = rung(E3, tef, tew, PARENT[name])
        Ep, _ = T.predict(tef.perturbed_params(rp), rp, dimers)
        assert abs(Ep[0] - E0[0]) > 1e-6 and abs(Ep[1] - E0[1]) > 1e-7            # the parent's self terms


def test_cross_term_local_and_on(E3, tef, tew):
    """a probe 7.3 A from its core interacts as with the base rung, energies and every force; on the trimer at contact
    and on the trimer with a molecule straddling the taper the term is on."""
    structs = tef.structures(E3)
    base = rung(E3, tef, tew, BASE)
    E0, F0 = T.predict(tef.perturbed_params(base), base, structs)
    for name in CROSS:
        r = rung(E3, tef, tew, name)
        E, F = T.predict(tef.perturbed_params(r), r, structs)
        np.testing.assert_allclose(E[2], E0[2], rtol=0, atol=1e-15)
        np.testing.assert_allclose(F[2], F0[2], rtol=0, atol=1e-15)
        assert abs(E[0] - E0[0]) > 1e-6 and abs(E[1] - E0[1]) > 1e-8


def test_quadratic_cross_is_degree_two(E3, tef, tew):
    """in the rung, the degree-2 cross term's energy scales as the square of the pinned charges; the MLP form's does not."""
    from sseft import structure as S
    st = tef.structures(E3, ("trimer",))[0]
    for name, quadratic in (("M_16C2pm_w2x", True), ("M_16C2pm_wx", False)):
        r = rung(E3, tef, tew, name)
        pp = tef.perturbed_params(r)
        b = {k_: jnp.asarray(v_[0]) for k_, v_ in S.pad_batch([st], n_freq=r.K).items()}
        b2 = {**b, "pin_q_site": 2.0 * b["pin_q_site"], "vs_q": 2.0 * b["vs_q"]}
        f1 = float(M.energy_single(pp, r, b)[1]["E_field"])
        f2 = float(M.energy_single(pp, r, b2)[1]["E_field"])
        assert abs(f1) > 1e-7
        assert (abs(f2 / f1 - 4.0) < 1e-10) == quadratic


@pytest.mark.parametrize("name", CROSS)
def test_cross_rung_forces_are_the_energy_gradient(E3, tef, tew, name):
    """autodiff forces against central differences of the energy (h = 1e-5 bohr), with the third molecule straddling the
    taper, on atoms of every molecule."""
    from sseft import structure as S
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


def test_cross_rung_invariant(E3, tef, tew):
    """rotated, translated and with its molecules reordered, the trimer with a straddling molecule has the same energy
    under both cross rungs, and the forces rotate and reorder with it."""
    P = tef.GEOMETRIES["taper"]
    R = W.rotation((0.3, -1.0, 0.7), 1.234)
    order = [2, 0, 1]
    Q = np.concatenate([(P[3 * m:3 * m + 3] @ R.T + np.array([1.0, -2.0, 0.5])) for m in order])
    a, b = pinned(E3, tef, P, Q)
    for name in CROSS:
        r = rung(E3, tef, tew, name)
        (Ea, Eb), (Fa, Fb) = T.predict(tef.perturbed_params(r), r, [a, b])
        np.testing.assert_allclose(Eb, Ea, rtol=1e-11)
        Fa_moved = np.concatenate([Fa[3 * m:3 * m + 3] @ R.T for m in order])
        np.testing.assert_allclose(Fb, Fa_moved, rtol=1e-8, atol=1e-12)
