"""E3-QP5 (v0.0.14): E_0's view of the pinned field.  The rungs with Rung.e0_field read, at every atom, the potential,
field and field gradient of the other molecules' pinned charges and M-sites (the band kernel at the band edge, every
source's share tapered to zero between e0_field_on and e0_field_off, inside E_0's cutoff) through a site-energy term
that vanishes at zero field.  Without a build: a stand-in site model of TTM3-F's form.

The rungs that existed before (M_inf, M_16pm, M_16C2pm, M_16C2pm_r1) must be untouched, to the last digits: GOLDEN holds
their energies and force checksums at fixed perturbed parameters, computed with v0.0.13 before any v0.0.14 change."""
import importlib
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sseft import model as M, train as T, water as W
from sseft.units import ang_to_bohr

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
GAMMA = 0.46                                  # TTM3-F's M-site: (1 - g) r_O + g/2 (r_H1 + r_H2)
Q_H = 0.6
E0CFG = dict(F=8, T=2, n_rbf=20, r_cut=ang_to_bohr(6.0), n_species=56, n_rbf_readout=8)
BASE = ("M_inf", "M_16pm", "M_16C2pm", "M_16C2pm_r1")


class SiteTruth:
    """a site model of TTM3-F's form with fixed charges: +Q_H on the H atoms, none on O, -2 Q_H on the M-site."""
    gamma_m = GAMMA

    def site_model(self, pos_ang):
        n = len(pos_ang) // 3
        q = np.zeros(3 * n); w = np.zeros((n, 3 * n))
        for m in range(n):
            q[3 * m + 1:3 * m + 3] = Q_H
            w[m, 3 * m:3 * m + 3] = [1 - GAMMA, GAMMA / 2, GAMMA / 2]
        return q, w, np.full(n, -2 * Q_H), np.arange(n)

    def molecule_charges(self, pos_ang):
        return np.tile([0.0, Q_H, Q_H], len(pos_ang) // 3)


def molecule(axis, angle, shift):
    return W.monomer_positions() @ W.rotation(axis, angle).T + np.asarray(shift, float)


A = molecule((0, 0, 1), 0.0, (0.0, 0.0, 0.0))
B = molecule((1, 1, 0), 2.1, (2.85, 0.3, -0.2))
C = molecule((0, 1, 1), -1.3, (1.1, 2.7, 0.5))
GEOMETRIES = {
    "trimer": np.concatenate([A, B, C]),                                              # every pair at contact
    "taper": np.concatenate([A, B, molecule((1, 0, 1), 0.7, (0.4, -1.0, 5.1))]),      # one molecule in the taper
    "probe": np.concatenate([A, B, C, molecule((1, 2, 0), 1.9, (10.5, 1.0, 0.5))]),   # a probe 7.3 A from the core
}


@pytest.fixture(scope="module")
def E3():
    import sys
    sys.path.insert(0, str(SCRIPTS))
    return importlib.import_module("e3_water")


def structures(E3, names=tuple(GEOMETRIES), truth=None):
    out = []
    for name in names:
        st = W.water_structure(GEOMETRIES[name])
        if name == "probe":
            st.sub = np.repeat([0, 1], [9, 3])
        out.append(st)
    W.pin_site_model(out, truth or SiteTruth())
    E3.pin_dispersion(out)
    return out


def rung(E3, name, **kw):
    r = E3.RUNGS[name]
    extra = dict(e0={**r.e0, **E0CFG}, l1=ang_to_bohr(0.75))
    if r.band_fields >= 0:
        extra["bf_l"] = ang_to_bohr(E3.RUNG_RESPONSE_EDGE.get(name, 0.75))
    return M.with_rung(r, **{**extra, **kw})


def perturbed_params(r, seed=0):
    """the rung's initial parameters with every leaf moved by a fixed pattern, so that every term is switched on (the
    read-out's and the new terms' zero output layers too)."""
    p = M.init_params(jax.random.PRNGKey(seed), r, energy_scale=0.02)
    leaves, tree = jax.tree_util.tree_flatten(p)
    moved = [x + 0.03 * jnp.sin(1.0 + jnp.arange(x.size, dtype=x.dtype)).reshape(x.shape)
             if jnp.issubdtype(x.dtype, jnp.floating) else x for x in leaves]
    return jax.tree_util.tree_unflatten(tree, moved)


def readings(E3, r, structs, seed=0):
    """energies (hartree) and two force checksums per structure."""
    E, F = T.predict(perturbed_params(r, seed), r, structs)
    out = []
    for e, f in zip(E, F):
        c = np.cos(1.0 + np.arange(f.size)).reshape(f.shape)
        out.append((float(e), float(np.sum(f * c)), float(np.sum(f * f))))
    return out


# v0.0.13's readings (scratch/e3qp5_golden.py, run on 5308f99 before any v0.0.14 change): energy, force checksums
GOLDEN = {
    'M_inf': [(-0.006007533251910391, 0.01622885398436255, 0.0007745576173646347),
               (-0.0038798293923437144, 0.014786271366725957, 0.00024331053974938798),
               (5.551115123125783e-17, 3.5414259272922147e-17, 4.458179874928608e-34)],
    'M_16pm': [(-0.006663299303147907, 0.004338453245516267, 0.0009015325958863774),
               (-0.00843213828082301, 0.013888547207307048, 0.0003081574597374514),
               (4.5614331690924104e-05, -2.9274283268930998e-05, 1.800006254459145e-07)],
    'M_16C2pm': [(-0.0059931828497030715, 0.004214437555958144, 0.0008942581262731433),
               (-0.007892268750041342, 0.013725785149843281, 0.00028739837682842227),
               (5.5516162856594775e-05, -2.9180521711309352e-05, 1.6977030921761507e-07)],
    'M_16C2pm_r1': [(-0.00603773572985955, 0.004464338039295093, 0.0009332288261754427),
               (-0.007779705281578231, 0.013584138911343876, 0.00029235634595307794),
               (5.971776641722215e-05, -3.1755136870521796e-05, 1.688594736867108e-07)],
}


def test_existing_rungs_unchanged(E3):
    """v0.0.14 leaves the rungs that existed untouched: their energies and forces at fixed parameters as v0.0.13's."""
    structs = structures(E3)
    for name in BASE:
        got = readings(E3, rung(E3, name), structs)
        for g, ref in zip(got, GOLDEN[name]):
            np.testing.assert_allclose(g, ref, rtol=1e-10, atol=1e-14, err_msg=name)


# ---------------------------------------------------------------- the field term ----------------------------------
FIELD = ("M_16pm_f", "M_16C2pm_f", "M_inf_f")
BASE_OF = {"M_16pm_f": "M_16pm", "M_16C2pm_f": "M_16C2pm", "M_inf_f": "M_inf"}


def run_args(**kw):
    from types import SimpleNamespace
    return SimpleNamespace(**{**dict(lA=4.0, rcut=6.0, F=32, l1=0.75, rs_over_l1=0.0, response_edge=0.75), **kw})


def test_field_rungs_configured_and_recorded(E3):
    a = run_args()
    for name in FIELD:
        r, base = E3.configure(E3.RUNGS[name], a), E3.configure(E3.RUNGS[BASE_OF[name]], a)
        assert r.e0_field and r.msites and not base.e0_field
        assert abs(r.e0_field_on - ang_to_bohr(4.5)) < 1e-12 and abs(r.e0_field_off - ang_to_bohr(5.5)) < 1e-12
        assert r.e0_field_off < r.e0["r_cut"]
        # the field rung is its base with the field term, nothing else
        assert M.with_rung(r, name=base.name, e0_field=False, e0_field_on=base.e0_field_on,
                           e0_field_off=base.e0_field_off) == (base if name != "M_inf_f" else
                                                               M.with_rung(r, name=base.name, e0_field=False,
                                                                           e0_field_on=base.e0_field_on, e0_field_off=base.e0_field_off))
        record = E3.model_record(name, a)
        assert record["e0_field_taper_angstrom"] == [4.5, 5.5] and record["e0_field_unit"] == 0.01
        assert "e0_field_taper_angstrom" not in E3.model_record(BASE_OF[name], a)
    # the taper follows E_0's cutoff
    r5 = E3.configure(E3.RUNGS["M_16pm_f"], run_args(rcut=5.0))
    assert abs(r5.e0_field_off - ang_to_bohr(4.5)) < 1e-12
    inf_f = E3.model_record("M_inf_f", a)
    assert (inf_f["charges"], inf_f["polarisabilities"], inf_f["coulomb_band"], inf_f["site_model"]) == ("pinned", "off", False, True)
    assert "coulomb_band" not in E3.model_record("M_16pm_f", a)
    inf = E3.configure(E3.RUNGS["M_inf_f"], a)
    assert not (inf.coulomb_energy or inf.dispersion) and inf.band_fields < 0


def test_field_rungs_start_as_their_base(E3):
    """the field term's output layer starts at zero: at initialisation a field rung is its base rung, digit for digit."""
    structs = structures(E3)
    for name in FIELD:
        r, base = rung(E3, name), rung(E3, BASE_OF[name])
        p, p0 = M.init_params(jax.random.PRNGKey(3), r, 0.02), M.init_params(jax.random.PRNGKey(3), base, 0.02)
        assert {k: v for k, v in p.items() if k != "field"}.keys() == p0.keys()
        for x, y in zip(jax.tree_util.tree_leaves({k: v for k, v in p.items() if k != "field"}), jax.tree_util.tree_leaves(p0)):
            assert np.array_equal(np.asarray(x), np.asarray(y))
        E, F = T.predict(p, r, structs); E0, F0 = T.predict(p0, base, structs)
        assert np.array_equal(E, E0) and all(np.array_equal(f, g) for f, g in zip(F, F0))


def test_field_term_local(E3):
    """with every parameter switched on, a probe 7.3 A from its core (beyond r_c) interacts as with the base rung --
    energies and every force -- and an isolated molecule has no field term; at contact the term is on."""
    structs = structures(E3)
    alone = W.water_structure(A)
    W.pin_site_model([alone], SiteTruth()); E3.pin_dispersion([alone])
    for name in FIELD:
        r, base = rung(E3, name), rung(E3, BASE_OF[name])
        E, F = T.predict(perturbed_params(r), r, structs + [alone])
        E0, F0 = T.predict(perturbed_params(base), base, structs + [alone])
        np.testing.assert_allclose(E[2], E0[2], rtol=0, atol=1e-15)
        np.testing.assert_allclose(F[2], F0[2], rtol=0, atol=1e-15)
        np.testing.assert_allclose(E[3], E0[3], rtol=0, atol=1e-15)
        assert abs(E[0] - E0[0]) > 1e-6 and abs(E[1] - E0[1]) > 1e-6


def test_local_field_tensors_are_the_band_field(E3):
    """without the taper the shares sum to the band potential, its field and gradient (bandfields.band_field_tensors_from,
    by autodiff); with it, a molecule beyond e0_field_off of every atom adds nothing."""
    from sseft import bandfields as BF, kernels as kn
    st = structures(E3, ("taper",))[0]
    pos = jnp.asarray(st.positions)
    w, vq = st.pinned["vsite_w"], st.pinned["vsite_q"]
    X = jnp.concatenate([pos, jnp.asarray(w) @ pos]); q = jnp.concatenate([jnp.asarray(st.pinned["q_site"]), jnp.asarray(vq)])
    frag = np.concatenate([st.frag, st.pinned["vsite_frag"]])
    src = jnp.asarray(st.frag[:, None] != frag[None, :])
    l = ang_to_bohr(0.75)
    phi, E, G = BF.local_field_tensors(pos, X, q, src, l, 1e6, 2e6)
    Eb, gEb, _ = BF.band_field_tensors_from(pos, X, q, src, l, 1)
    r = jnp.linalg.norm(pos[:, None] - X[None], axis=-1)
    Vb = jnp.sum(jnp.where(src, q[None] * kn.g_long(jnp.where(src, r, 1.0), l), 0.0), axis=1)
    np.testing.assert_allclose(phi, Vb, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(E, Eb, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(G, gEb, rtol=1e-10, atol=1e-14)
    # the taper: only the third molecule (4.6-7 A from the others) is in or beyond it
    on, off = ang_to_bohr(4.5), ang_to_bohr(5.5)
    phi_t, E_t, G_t = BF.local_field_tensors(pos, X, q, src, l, on, off)
    assert not np.allclose(E_t, E)                                  # its shares are cut
    near = jnp.asarray(frag[None, :] != 2) & src                    # the same without the third molecule's sources
    phi_n, E_n, G_n = BF.local_field_tensors(pos, X, q, near, l, on, off)
    far_from_3 = np.min(np.asarray(r)[:, frag == 2], axis=1) >= off
    assert far_from_3.sum() >= 1
    for a, b in ((phi_t, phi_n), (E_t, E_n), (G_t, G_n)):
        np.testing.assert_allclose(np.asarray(a)[far_from_3], np.asarray(b)[far_from_3], rtol=0, atol=1e-15)


def test_taper_is_c2():
    from sseft import bandfields as BF
    on, off = 4.0, 5.0
    f = lambda r: BF.taper(r, on, off)
    d1, d2 = jax.grad(f), jax.grad(jax.grad(f))
    assert f(on) == 1.0 and f(off) == 0.0 and f(3.0) == 1.0 and f(6.0) == 0.0
    for r in (on, off):
        for h in (-1e-7, 1e-7):
            assert abs(d1(r + h)) < 1e-6 and abs(d2(r + h)) < 1e-5
    xs = np.linspace(on, off, 11)[1:-1]
    assert np.all(np.diff([f(x) for x in xs]) < 0) and all(0 < f(x) < 1 for x in xs)


def test_field_rung_forces_are_the_energy_gradient(E3):
    """autodiff forces of M_16C2pm_f against central differences of its energy (h = 1e-5 bohr: the difference falls as
    h^2, 2.5e-8 at 1e-4, as the base rung's), with the third molecule in the taper."""
    from sseft import structure as S
    r = rung(E3, "M_16C2pm_f")
    p = perturbed_params(r)
    st = structures(E3, ("taper",))[0]
    b = {k: jnp.asarray(v[0]) for k, v in S.pad_batch([st], n_freq=r.K).items()}
    energy = lambda pos: M.energy_single(p, r, {**b, "positions": pos})[0]
    f, g = jax.jit(energy), jax.jit(jax.grad(energy))
    pos0 = b["positions"]
    F = -np.asarray(g(pos0))
    _, F_pred = T.predict(p, r, [st])
    np.testing.assert_allclose(F_pred[0], F, rtol=1e-12, atol=1e-16)
    h = 1e-5
    for atom, axis in ((0, 0), (2, 2), (6, 1), (7, 0), (8, 2)):
        e = jnp.zeros_like(pos0).at[atom, axis].set(h)
        fd = -(float(f(pos0 + e)) - float(f(pos0 - e))) / (2 * h)
        assert abs(fd - F[atom, axis]) < 2e-9, (atom, axis, fd, F[atom, axis])


def test_field_rung_invariant(E3):
    """rotated, translated and with its molecules reordered, M_16C2pm_f's trimer has the same energy, and its forces
    rotate and reorder with it."""
    r = rung(E3, "M_16C2pm_f")
    p = perturbed_params(r)
    P = GEOMETRIES["trimer"]
    R = W.rotation((0.3, -1.0, 0.7), 1.234)
    order = [2, 0, 1]
    Q = np.concatenate([(P[3 * m:3 * m + 3] @ R.T + np.array([1.0, -2.0, 0.5])) for m in order])
    a, b = W.water_structure(P), W.water_structure(Q)
    W.pin_site_model([a, b], SiteTruth()); E3.pin_dispersion([a, b])
    (Ea, Eb), (Fa, Fb) = T.predict(p, r, [a, b])
    np.testing.assert_allclose(Eb, Ea, rtol=1e-11)
    Fa_moved = np.concatenate([Fa[3 * m:3 * m + 3] @ R.T for m in order])
    np.testing.assert_allclose(Fb, Fa_moved, rtol=1e-8, atol=1e-12)
