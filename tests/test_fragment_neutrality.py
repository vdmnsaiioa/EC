"""
The neutrality shift of the learned charges is taken per fragment (v0.0.8): every molecule of a structure is
neutral and an ion carries its own charge whatever the environment, so that the charges of a fragment beyond r_c
of everything else are exactly the fragment's own and the fragment reference E_int = E(all) - sum_f E(f) cancels
the intramolecular band energy exactly.  Structures without fragments (atoms) keep one shift to the total charge.
"""
import numpy as np
import jax, jax.numpy as jnp
import pytest
import sseft
from sseft import model as M, structure as S, water as W, bands as B, e0 as E0


def _single(structures, **kw):
    return {k: v[0] for k, v in S.pad_batch(structures, **kw).items()}


def _learned_q_rung(**kw):
    return M.with_rung(M.RUNGS["M_1"], **kw)


def _fragment_sums(q, frag):
    return np.array([float(np.sum(np.array(q)[:len(frag)][frag == f])) for f in range(int(frag.max()) + 1)])


def test_each_molecule_of_a_cluster_is_neutral():
    """a hydrogen-bonded dimer inside r_c: the two molecules see different environments, so their raw dq differ;
    the shift per fragment leaves each neutral, the shift over the structure (frag_neutral=False) does not."""
    st = W.water_structure(W.dimer_positions(2.9), pinned_q=False)
    b = _single([st])
    params = M.init_params(jax.random.PRNGKey(5), _learned_q_rung())
    _, aux = M.energy_single(params, _learned_q_rung(), b)
    sums = _fragment_sums(aux["q"], st.frag)
    assert np.all(np.abs(sums) < 1e-12), sums
    _, aux_old = M.energy_single(params, _learned_q_rung(frag_neutral=False), b)
    sums_old = _fragment_sums(aux_old["q"], st.frag)
    assert abs(float(np.sum(aux_old["q"]))) < 1e-12 and np.max(np.abs(sums_old)) > 1e-6, sums_old


def test_separated_fragments_carry_their_isolated_charges_and_the_ion_its_charge():
    """a monomer and a unit charge 40 A apart (beyond r_c of each other): the monomer's charges are exactly the
    isolated monomer's, the ion's exactly +1, and the interaction energy is exactly the Coulomb band between the
    two fragments (the fragment reference cancels the rest) -- none of which holds with one shift over the structure."""
    pos = np.vstack([W.monomer_positions(), [[0.0, 0.0, 40.0]]])
    ion = W.ion_structure(pos, 1)
    ion.pinned = {}
    rung = _learned_q_rung()
    params = M.init_params(jax.random.PRNGKey(11), rung)
    b = _single([ion])
    E, aux = M.energy_single(params, rung, b)
    q = np.array(aux["q"])[:4]
    _, aux_mono = M.energy_single(params, rung, _single([W.water_structure(W.monomer_positions(), pinned_q=False)]))
    np.testing.assert_allclose(q[:3], np.array(aux_mono["q"])[:3], atol=1e-12)
    assert abs(q[3] - W.ION["Q"]) < 1e-12 and abs(np.sum(q[:3])) < 1e-12
    # the interaction energy: the Coulomb band over the pairs between the two fragments, nothing else
    s_, _, _, (D, r, pmask) = E0.apply_e0(params["e0"], rung.e0, b["positions"], b["numbers"], b["mask"], b["cell"], b["pbc"])
    fm = b["frag_mask"].astype(float); same = (fm.T @ fm) > 0.5
    E_inter = float(B.coulomb_band(aux["q"], None, D, r, pmask & ~same, rung.l1))
    assert abs(float(E) - E_inter) < 1e-12 * max(1.0, abs(E_inter)), (float(E), E_inter)
    # with the shift over the structure the ion's charge is spread over the water and the cancellation fails
    E_old, aux_old = M.energy_single(params, _learned_q_rung(frag_neutral=False), b)
    q_old = np.array(aux_old["q"])[:4]
    assert abs(q_old[3] - W.ION["Q"]) > 1e-3 and abs(np.sum(q_old[:3])) > 1e-3
    assert abs(float(E_old) - E_inter) > 1e-6


def test_structures_without_fragments_keep_one_shift_in_a_mixed_batch():
    """an atomic cluster (no fragments) padded together with a water dimer: the atoms are not treated as neutral
    fragments; their charges sum to the total charge and equal those obtained without any fragment in the batch."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(-4, 4, (5, 3)) * 1.8897; Z = rng.choice([10, 18], 5)
    atoms = S.Structure(pos, Z, total_charge=1.0)
    dimer = W.water_structure(W.dimer_positions(2.9), pinned_q=False)
    rung = _learned_q_rung()
    params = M.init_params(jax.random.PRNGKey(2), rung)
    batch = S.pad_batch([dimer, atoms])
    assert "frag_mask" in batch and bool(batch["has_frag"][0]) and not bool(batch["has_frag"][1])
    _, aux = M.energy_single(params, rung, {k: v[1] for k, v in batch.items()})
    q_mixed = np.array(aux["q"])[:5]
    assert abs(np.sum(q_mixed) - 1.0) < 1e-12
    _, aux_alone = M.energy_single(params, rung, _single([atoms]))
    np.testing.assert_allclose(q_mixed, np.array(aux_alone["q"])[:5], atol=1e-14)
    assert np.max(np.abs(q_mixed)) > 1e-6     # the atoms do carry charges (not per-atom neutrality)


def test_fragment_charges_are_validated_when_batching():
    pos = np.vstack([W.monomer_positions(), [[0.0, 0.0, 6.0]]])
    ion = W.ion_structure(pos, 1)
    np.testing.assert_array_equal(ion.frag_charges, [0.0, W.ION["Q"]])
    b = S.pad_batch([ion])
    assert bool(b["has_frag"][0]) and np.allclose(np.array(b["frag_charge"][0]), [0.0, 1.0])
    charged_without = S.Structure(ion.positions, ion.numbers, frag=ion.frag, total_charge=1.0)
    with pytest.raises(ValueError, match="frag_charges"):
        S.pad_batch([charged_without])
    wrong_length = S.Structure(ion.positions, ion.numbers, frag=ion.frag, total_charge=1.0, frag_charges=np.array([1.0]))
    with pytest.raises(ValueError, match="one entry per fragment"):
        S.pad_batch([wrong_length])
    wrong_sum = S.Structure(ion.positions, ion.numbers, frag=ion.frag, total_charge=1.0, frag_charges=np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="sum to total_charge"):
        S.pad_batch([wrong_sum])
    neutral = W.water_structure(W.dimer_positions(3.0))
    assert neutral.frag_charges is None and np.allclose(np.array(S.pad_batch([neutral])["frag_charge"][0]), 0.0)


def test_forces_finite_difference_with_fragment_shift():
    """the shift per fragment is differentiable: forces of M_1 with learned charges and dipoles on a water dimer
    against the four-point stencil (the test of tests/test_harness.py on a fragment structure)."""
    st = W.water_structure(W.dimer_positions(2.9), pinned_q=False)
    b = _single([st])
    rung = _learned_q_rung(dipoles=True)
    params = M.init_params(jax.random.PRNGKey(1), rung)
    _, F = M.energy_forces_single(params, rung, b)
    def E_at(i, a, dx):
        return float(M.energy_single(params, rung, {**b, "positions": b["positions"].at[i, a].add(dx)})[0])
    for (i, a) in [(0, 2), (4, 0)]:
        h = 3e-5
        fd = -(-E_at(i, a, 2 * h) + 8 * E_at(i, a, h) - 8 * E_at(i, a, -h) + E_at(i, a, -2 * h)) / (12 * h)
        assert abs(fd - float(F[i, a])) < 1e-6 * abs(fd) + 1e-10, (i, a, fd, float(F[i, a]))
