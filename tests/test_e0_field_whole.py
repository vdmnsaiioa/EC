"""E3-QP6 (v0.0.15): the field term with every molecule whole, and at degree 2 in the field.

Rung.e0_field_whole: all of a molecule's sources share one weight at an atom, the product of their tapers (between r_c - 1
and r_c in E3-QP6), so that a molecule straddling the taper enters neutral -- never part charged, as each source tapered
by its own distance left it in E3-QP5.  Rung.e0_field_form "quadratic": the term at degree 2 in the field, the read-out's
order-0 form on the local field and its square in the field gradient (heads.field_energy2).  Without a build: the stand-in
site model of tests/test_e0_field.py.

E3-QP5's field rungs (M_16pm_f, M_16C2pm_f, M_inf_f) must be untouched, to the last digits: GOLDEN_FIELD holds their
energies and force checksums at fixed perturbed parameters, computed with v0.0.14 before any v0.0.15 change
(scratch/e3qp6_golden.py); tests/test_e0_field.py keeps the rungs before them."""
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
WHOLE = ("M_16pm_w", "M_16C2pm_w", "M_16C2pm_w2")
BASE_OF = {"M_16pm_w": "M_16pm", "M_16C2pm_w": "M_16C2pm", "M_16C2pm_w2": "M_16C2pm"}
ON, OFF = ang_to_bohr(5.0), ang_to_bohr(6.0)          # E3-QP6's taper at r_c = 6 A

# v0.0.14's readings (scratch/e3qp6_golden.py, run on 7be7b4a = v0.0.14 before any v0.0.15 change)
GOLDEN_FIELD = {
    'M_16pm_f': [(-0.006393595930000773, 0.0018159986461179938, 0.000861987566214571),
               (-0.005426995096145415, 0.010850350400727998, 0.000265411479354612),
               (4.5614331690924104e-05, -2.9274283269015566e-05, 1.80000625445925e-07)],
    'M_16C2pm_f': [(-0.005723479476555937, 0.0016919829565599114, 0.0008653332392397131),
               (-0.004887125565363748, 0.010687588343264207, 0.0002535309453865434),
               (5.5516162856150686e-05, -2.918052171129798e-05, 1.6977030921760978e-07)],
    'M_inf_f': [(-0.005737829878763312, 0.013706399384964266, 0.0007658965772980207),
               (-0.0008746862076661754, 0.011748074560146902, 0.00023728980916210517),
               (5.551115123125783e-17, 1.5468750816101667e-17, 1.2501735069300274e-33)],
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


def rung(E3, tef, name, **kw):
    """the rung as tests/test_e0_field.py builds it (width 8, band edge 0.75 A), the field term's taper as E3-QP6 runs it
    (E3.configure: r_c - 1 to r_c for the whole-molecule rungs)."""
    r = tef.rung(E3, name, **kw)
    if r.e0_field and r.e0_field_whole:
        r = M.with_rung(r, e0_field_on=ON, e0_field_off=OFF)
    return r


def test_e3qp5_field_rungs_unchanged(E3, tef):
    """v0.0.15 leaves E3-QP5's field rungs untouched: their energies and forces at fixed parameters as v0.0.14's."""
    structs = tef.structures(E3)
    for name, ref_rows in GOLDEN_FIELD.items():
        got = tef.readings(E3, tef.rung(E3, name), structs)
        for g, ref in zip(got, ref_rows):
            np.testing.assert_allclose(g, ref, rtol=1e-10, atol=1e-14, err_msg=name)


def test_whole_rungs_configured_and_recorded(E3, tef):
    a = tef.run_args()
    for name in WHOLE:
        r, base = E3.configure(E3.RUNGS[name], a), E3.configure(E3.RUNGS[BASE_OF[name]], a)
        assert r.e0_field and r.e0_field_whole and r.msites and not base.e0_field
        assert r.e0_field_form == ("quadratic" if name.endswith("_w2") else "mlp")
        assert abs(r.e0_field_on - ang_to_bohr(5.0)) < 1e-12 and r.e0_field_off == r.e0["r_cut"]
        # the whole-molecule rung is its base with the field term, nothing else
        assert M.with_rung(r, name=base.name, e0_field=False, e0_field_whole=False, e0_field_form="mlp",
                           e0_field_on=base.e0_field_on, e0_field_off=base.e0_field_off) == base
        record = E3.model_record(name, a)
        assert record["e0_field_taper_angstrom"] == [5.0, 6.0] and record["e0_field_unit"] == 0.01
        assert record["e0_field_weights"] == "molecule"
        assert record.get("e0_field_form") == ("quadratic" if name.endswith("_w2") else None)
    # E3-QP5's rungs keep their records (no new keys) and their taper
    for name in ("M_16pm_f", "M_16C2pm_f", "M_inf_f"):
        record = E3.model_record(name, a)
        assert record["e0_field_taper_angstrom"] == [4.5, 5.5]
        assert "e0_field_weights" not in record and "e0_field_form" not in record
    # the taper follows E_0's cutoff
    r5 = E3.configure(E3.RUNGS["M_16C2pm_w2"], tef.run_args(rcut=5.0))
    assert abs(r5.e0_field_on - ang_to_bohr(4.0)) < 1e-12 and r5.e0_field_off == r5.e0["r_cut"]
    with pytest.raises(ValueError, match="e0_field_form"):
        M.init_params(jax.random.PRNGKey(0), M.with_rung(rung(E3, tef, "M_16C2pm_w"), e0_field_form="cubic"), 0.02)


def test_whole_rungs_start_as_their_base(E3, tef):
    """the term's output layer starts at zero: at initialisation a whole-molecule rung is its base rung, digit for digit."""
    structs = tef.structures(E3)
    for name in WHOLE:
        r, base = rung(E3, tef, name), rung(E3, tef, BASE_OF[name])
        p, p0 = M.init_params(jax.random.PRNGKey(3), r, 0.02), M.init_params(jax.random.PRNGKey(3), base, 0.02)
        assert {k: v for k, v in p.items() if k != "field"}.keys() == p0.keys()
        for x, y in zip(jax.tree_util.tree_leaves({k: v for k, v in p.items() if k != "field"}), jax.tree_util.tree_leaves(p0)):
            assert np.array_equal(np.asarray(x), np.asarray(y))
        E, F = T.predict(p, r, structs); E0, F0 = T.predict(p0, base, structs)
        assert np.array_equal(E, E0) and all(np.array_equal(f, g) for f, g in zip(F, F0))


def sites(st):
    pos = jnp.asarray(st.positions)
    X = jnp.concatenate([pos, jnp.asarray(st.pinned["vsite_w"]) @ pos])
    q = jnp.concatenate([jnp.asarray(st.pinned["q_site"]), jnp.asarray(st.pinned["vsite_q"])])
    frag = np.concatenate([st.frag, st.pinned["vsite_frag"]])
    src = jnp.asarray(st.frag[:, None] != frag[None, :])
    frag_src = jnp.asarray(frag[None, :] == np.arange(st.frag.max() + 1)[:, None])
    return pos, X, q, frag, src, frag_src


def test_every_molecule_enters_whole(E3, tef):
    """the tensors with frag_src are the sum over the other molecules of each one's whole field (no taper) times one weight,
    the product of its sites' tapers (computed here apart): a molecule straddling the taper enters neutral, with all its
    charges in their proportions; the per-source taper of v0.0.14 is not that."""
    st = tef.structures(E3, ("taper",))[0]
    pos, X, q, frag, src, frag_src = sites(st)
    l = ang_to_bohr(0.75)
    phi, E, G = BF.local_field_tensors(pos, X, q, src, l, ON, OFF, frag_src=frag_src)
    r = np.linalg.norm(np.asarray(pos)[:, None] - np.asarray(X)[None], axis=-1)
    x = np.clip((r - ON) / (OFF - ON), 0.0, 1.0)
    t = 1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x ** 2)
    want = [np.zeros(phi.shape), np.zeros(E.shape), np.zeros(G.shape)]
    weights = []
    for B in range(3):
        only_B = src & jnp.asarray(frag[None, :] == B)
        parts = BF.local_field_tensors(pos, X, q, only_B, l, 1e6, 2e6)                 # B's whole field, no taper
        wB = np.prod(np.where(frag[None, :] == B, t, 1.0), axis=1)                     # (N,): one weight per atom
        wB = np.where(st.frag == B, 0.0, wB)
        weights.append(wB)
        for k in range(3):
            want[k] += wB.reshape((-1,) + (1,) * (parts[k].ndim - 1)) * np.asarray(parts[k])
    for got, ref in zip((phi, E, G), want):
        np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-12, atol=1e-16)
    weights = np.array(weights)
    assert np.sum((weights > 0.05) & (weights < 0.95)) >= 3                # the test straddles the taper
    # every molecule neutral (the stand-in's +Q_H, +Q_H, -2 Q_H): as seen from any atom its weighted charges sum to zero,
    # while the per-source taper leaves the straddling molecule a net charge
    q_np, Q = np.asarray(q), np.zeros((len(pos), 3))
    for B in range(3):
        Q[:, B] = np.sum(np.where(frag[None, :] == B, t * q_np[None, :], 0.0), axis=1)
    assert np.max(np.abs(Q[np.asarray(st.frag)[:, None] != np.arange(3)[None, :]])) > 0.05
    # and nothing from a molecule with any site beyond r_off, everything from one with all within r_on
    assert np.all(weights[np.max(np.where(frag[None, None, :] == np.arange(3)[:, None, None], r[None], 0.0), axis=2) >= OFF] == 0.0)


def test_whole_term_local(E3, tef):
    """with every parameter switched on, a probe 7.3 A from its core interacts as with the base rung -- energies and every
    force -- and an isolated molecule has no field term; with the third molecule straddling the taper the term is on."""
    structs = tef.structures(E3)
    alone = W.water_structure(tef.A)
    W.pin_site_model([alone], tef.SiteTruth()); E3.pin_dispersion([alone])
    for name in WHOLE:
        r, base = rung(E3, tef, name), rung(E3, tef, BASE_OF[name])
        E, F = T.predict(tef.perturbed_params(r), r, structs + [alone])
        E0, F0 = T.predict(tef.perturbed_params(base), base, structs + [alone])
        np.testing.assert_allclose(E[2], E0[2], rtol=0, atol=1e-15)
        np.testing.assert_allclose(F[2], F0[2], rtol=0, atol=1e-15)
        np.testing.assert_allclose(E[3], E0[3], rtol=0, atol=1e-15)
        assert abs(E[0] - E0[0]) > 1e-6 and abs(E[1] - E0[1]) > 1e-6


def test_quadratic_term_is_degree_two(E3, tef):
    """heads.field_energy2 is exactly quadratic in (E, G) and zero at zero field; in the rung, the term's energy scales as
    the square of the pinned charges (the MLP form does not)."""
    key = jax.random.PRNGKey(5)
    F = 8
    cfg = {"F": F}
    p = H.init_field_head2(key, cfg)
    p = jax.tree_util.tree_map(lambda x: x + 0.1 * jnp.cos(jnp.arange(x.size, dtype=x.dtype)).reshape(x.shape), p)
    k = jax.random.split(key, 5)
    N = 4
    s, v, t = (jax.random.normal(k[0], (N, F)), jax.random.normal(k[1], (N, 3, F)), jax.random.normal(k[2], (N, 3, 3, F)))
    E, G = 0.01 * jax.random.normal(k[3], (N, 3)), 0.01 * jax.random.normal(k[4], (N, 3, 3))
    mask = jnp.ones(N)
    e1 = H.field_energy2(p, s, v, t, E, G, mask, 0.02, 0.01)
    e3 = H.field_energy2(p, s, v, t, 3.0 * E, 3.0 * G, mask, 0.02, 0.01)
    assert np.max(np.abs(np.asarray(e1))) > 1e-6
    np.testing.assert_allclose(e3, 9.0 * e1, rtol=1e-12)
    assert np.array_equal(np.asarray(H.field_energy2(p, s, v, t, 0 * E, 0 * G, mask, 0.02, 0.01)), np.zeros(N))
    # in the rung: the site charges scaled by 2 scale the term's energy by 4
    from sseft import structure as S
    st = tef.structures(E3, ("trimer",))[0]
    for name, quadratic in (("M_16C2pm_w2", True), ("M_16C2pm_w", False)):
        r = rung(E3, tef, name)
        pp = tef.perturbed_params(r)
        b = {k_: jnp.asarray(v_[0]) for k_, v_ in S.pad_batch([st], n_freq=r.K).items()}
        b2 = {**b, "pin_q_site": 2.0 * b["pin_q_site"], "vs_q": 2.0 * b["vs_q"]}
        f1 = float(M.energy_single(pp, r, b)[1]["E_field"])
        f2 = float(M.energy_single(pp, r, b2)[1]["E_field"])
        assert abs(f1) > 1e-7
        if quadratic:
            assert abs(f2 / f1 - 4.0) < 1e-10
        else:
            assert abs(f2 / f1 - 4.0) > 1e-3


@pytest.mark.parametrize("name", ["M_16C2pm_w", "M_16C2pm_w2"])
def test_whole_rung_forces_are_the_energy_gradient(E3, tef, name):
    """autodiff forces against central differences of the energy (h = 1e-5 bohr), with the third molecule straddling the
    taper (weights 0.4-0.9 at several atoms), on atoms of every molecule."""
    from sseft import structure as S
    r = rung(E3, tef, name)
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


def test_whole_rung_invariant(E3, tef):
    """rotated, translated and with its molecules reordered, the trimer with a straddling molecule has the same energy
    under M_16C2pm_w and M_16C2pm_w2, and the forces rotate and reorder with it."""
    P = tef.GEOMETRIES["taper"]
    R = W.rotation((0.3, -1.0, 0.7), 1.234)
    order = [2, 0, 1]
    Q = np.concatenate([(P[3 * m:3 * m + 3] @ R.T + np.array([1.0, -2.0, 0.5])) for m in order])
    a, b = W.water_structure(P), W.water_structure(Q)
    W.pin_site_model([a, b], tef.SiteTruth()); E3.pin_dispersion([a, b])
    for name in ("M_16C2pm_w", "M_16C2pm_w2"):
        r = rung(E3, tef, name)
        (Ea, Eb), (Fa, Fb) = T.predict(tef.perturbed_params(r), r, [a, b])
        np.testing.assert_allclose(Eb, Ea, rtol=1e-11)
        Fa_moved = np.concatenate([Fa[3 * m:3 * m + 3] @ R.T for m in order])
        np.testing.assert_allclose(Fb, Fa_moved, rtol=1e-8, atol=1e-12)
