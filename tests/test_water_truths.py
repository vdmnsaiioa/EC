"""The truth abstraction of sseft.water: the PW truth stays the default, and the generators' PW output is the output of
the code before the abstraction (values recorded from v0.0.8 at commit c3f7c95)."""
import hashlib

import numpy as np

from sseft import water as W


def _digest(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


def test_pw_generators_reproduce_the_recorded_data():
    st = W.cluster_dataset([3], 1, seed=5, sigma=1e-9, mc_steps=20)[0]
    # a last-bit change in the JAX truth could not flip any of the 20 acceptances, so a different cluster is a change
    np.testing.assert_allclose(st.energy, -0.0032475532665750153, rtol=1e-12, atol=0)
    if _digest(st.positions) != "301c8b6e0c472100":                         # bit-for-bit on the recording platform
        reference = W.cluster_dataset([3], 1, seed=5, sigma=1e-9, mc_steps=20, truth=W.PW_TRUTH)[0]
        np.testing.assert_array_equal(st.positions, reference.positions)
    d = W.dimer_scan(np.array([3.9, 6.5]), sigma=1e-9, rng=np.random.default_rng(0))
    np.testing.assert_allclose([s.energy for s in d], [-0.002691706121814661, -0.00044447889697436663], rtol=1e-12, atol=0)
    np.testing.assert_array_equal(d[1].pinned["q"], np.tile(W.monomer_charges(), 2))
    pos = np.concatenate([W.dimer_positions(3.1), W.monomer_positions() + [0, 0, 6.0]])
    np.testing.assert_allclose(W.pair_additive_energy(pos, 3), -0.011733425603343018, rtol=1e-12, atol=0)


def test_explicit_pw_truth_is_the_default():
    a = W.cluster_dataset([2], 1, seed=3, mc_steps=10)[0]
    b = W.cluster_dataset([2], 1, seed=3, mc_steps=10, truth=W.PW_TRUTH)[0]
    np.testing.assert_array_equal(a.positions, b.positions)
    assert a.energy == b.energy and np.array_equal(a.forces, b.forces) and np.array_equal(a.pinned["q"], b.pinned["q"])
    m = W.monomer_structure()
    np.testing.assert_array_equal(m.pinned["q"], W.monomer_charges())


def test_predict_energies_matches_predict():
    """the energies-only, chunked, size-grouped evaluation of the held-out reading against the ordinary prediction,
    for a band-field rung with a non-zero read-out and structures of three sizes (chunks padded with repeats)."""
    import jax
    from sseft import model as M, train as T
    rung = M.with_rung(M.RUNGS["M_1"], pin_q=True, band_fields=0, l1=1.4,
                       e0={**M.RUNGS["M_1"].e0, "T": 1, "n_rbf": 4, "n_rbf_readout": 4, "F": 8})
    params = M.init_params(jax.random.PRNGKey(4), rung)
    kk = jax.random.split(jax.random.PRNGKey(9), 4)
    for key_, kw, kb in (("theta", kk[0], kk[1]), ("iso", kk[2], kk[3])):
        last = params["readout"][key_][-1]
        params["readout"][key_][-1] = {"w": 0.3 * jax.random.normal(kw, last["w"].shape), "b": 0.3 * jax.random.normal(kb, last["b"].shape)}
    rng = np.random.default_rng(0)
    structs = []
    for n in (2, 3, 2, 4, 3, 2, 2):
        pos = np.concatenate([W.monomer_positions() @ W.random_rotation(rng).T + rng.uniform(-1, 1, 3) + [3.1 * m, 0, 0] for m in range(n)])
        structs.append(W.water_structure(pos))
    E_ref = np.concatenate([T.predict(params, rung, [s])[0] for s in structs])
    E = T.predict_energies(params, rung, structs, chunk=2)
    np.testing.assert_allclose(E, E_ref, rtol=1e-12, atol=1e-15)
    assert np.any(np.abs(E) > 0)


def test_pw_truth_constants():
    assert W.PW_TRUTH.r_e == 2.86
    assert abs(W.dipole_dipole_coefficient(W.PW_TRUTH) + 0.652) < 5e-4   # the rehearsal's p* = 3 coefficient
    assert abs(W.dimer_minimum(W.PW_TRUTH) - 2.86) < 0.02                 # the recorded value is the rounded minimum
