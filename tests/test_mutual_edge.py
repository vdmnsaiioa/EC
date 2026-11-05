"""
Rung.bf_l_mutual (v0.0.23): the mutual response's dipole tensor smeared on its own edge, the band field the read-out
reads left at the response's edge -- as MB-pol damps the dipole-dipole coupling at contact (Thole a_DD 0.055) and
hardly the charge-dipole field (a_CD 0.4).  0, or the response's own edge, is the model as before to the bit; first
order does not see it; open and periodic agree with it (a cluster in large cells against its open energy plus the
dipole image term, as test_periodic's frozen-model test); and a wider edge weakens the coupling at contact.
"""
import math
import os
import sys
from dataclasses import replace

import numpy as np
import jax
import jax.numpy as jnp

import sseft  # noqa: F401  (x64)
from sseft import model as M
from sseft.units import ang_to_bohr as A2B

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_periodic import frozen, switched_on, cluster, water_structure, one, in_cell  # noqa: E402


def test_default_and_own_edge_are_the_model_as_before():
    pos = cluster(4, seed=1)
    rung = frozen()
    params = switched_on(rung)
    b = one(water_structure(pos))
    E = lambda r: float(M.energy_single(params, r, b)[0])
    e0 = E(rung)
    assert E(replace(rung, bf_l_mutual=rung.bf_l)) == e0
    assert abs(E(replace(rung, bf_l_mutual=A2B(2.0))) - e0) > 1e-6          # it does act on the mutual part
    first = replace(rung, bf_mutual=1, bf_solver="direct")
    assert E(replace(first, bf_l_mutual=A2B(2.0))) == E(first)                # first order has no mutual tensor
    bc = one(water_structure(pos + 10.0, np.eye(3) * 20.0))
    rc = in_cell(rung, np.asarray(bc["cell"]), 4.0)
    Ec = lambda r: float(jax.jit(lambda pp, bb: M.energy_single(pp, r, bb)[0])(params, bc))
    assert Ec(replace(rc, bf_l_mutual=rc.bf_l)) == Ec(rc)


def test_wider_mutual_edge_in_a_large_cell_is_the_open_one():
    """with bf_l_mutual = 2 A: E_0 and the field term the open ones exactly, the rest the open energy plus the tin-foil
    dipole image term -(2 pi / 3V) |M_perm + M_ind|^2, extrapolated in 1 / L^2 from 30 and 40 A cells."""
    pos = cluster(4, seed=0)
    rung = replace(frozen(), bf_l_mutual=A2B(2.0))
    params = switched_on(rung)
    st = water_structure(pos)
    bo = one(st)
    E_open, aux_o = M.energy_single(params, rung, bo)
    p = np.asarray(bo["positions"])
    Mp = st.pinned["q_site"] @ p + st.pinned["vsite_q"] @ (st.pinned["vsite_w"] @ p)
    Mtot = Mp + np.asarray(aux_o["mu_ind"]).sum(0)
    vals = []
    for L in (30.0, 40.0):
        b = one(water_structure(pos + L / 2, np.eye(3) * L))
        rc = in_cell(rung, np.asarray(b["cell"]), 5.0)
        E, aux = jax.jit(lambda pp, bb: M.energy_single(pp, rc, bb))(params, b)
        assert abs(float(aux["E0"]) - float(aux_o["E0"])) < 1e-12 * abs(float(aux_o["E0"]))
        vals.append(float(jnp.linalg.det(b["cell"])) * (float(E) - float(E_open)))
    limit = (16 * vals[1] - 9 * vals[0]) / 7
    assert abs(limit / (-(2 * math.pi / 3) * Mtot @ Mtot) - 1) < 3e-3, (vals, limit, -(2 * math.pi / 3) * Mtot @ Mtot)


def test_wider_mutual_edge_weakens_the_coupling_at_contact():
    """the second-order part of the induced dipoles, C T C E, is smaller with the tensor smeared at 2 A than at the
    0.75 A edge on a compact cluster (O-O 2.6-3 A), and the first-order part C E is the same."""
    pos = cluster(4, seed=2, box=5.0)
    rung = replace(frozen(), bf_mutual=2, bf_solver="direct")
    params = switched_on(rung)
    b = one(water_structure(pos))
    mu = lambda r: np.asarray(M.energy_single(params, r, b)[1]["mu_ind"])
    mu1 = mu(replace(rung, bf_mutual=1))
    d_bare = np.linalg.norm(mu(rung) - mu1)
    d_wide = np.linalg.norm(mu(replace(rung, bf_l_mutual=A2B(2.0))) - mu1)
    assert np.allclose(mu(replace(rung, bf_mutual=1, bf_l_mutual=A2B(2.0))), mu1, rtol=0, atol=1e-14)
    assert d_wide < 0.8 * d_bare, (d_wide, d_bare)


def test_the_driver_sets_it_from_the_config():
    """scripts/e4_water.py: model.mutual_edge_angstrom sets bf_l_mutual on the cell and open rungs of a rung with a
    mutual response (and on no other), and the results' model record carries it; null leaves job 1's rungs as they were."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import e4_water as E4
    cfg = E4.load_config(None)
    cell = np.eye(3) * A2B(12.43)
    c0, o0 = E4.rungs_for("M_16C2pm_us", cfg, cell)
    assert c0.bf_l_mutual == 0 and o0.bf_l_mutual == 0
    cfg2 = E4.merge(cfg, {"model": {"mutual_edge_angstrom": 2.0}})
    c2, o2 = E4.rungs_for("M_16C2pm_us", cfg2, cell)
    assert c2.bf_l_mutual == A2B(2.0) and o2.bf_l_mutual == A2B(2.0) and replace(c2, bf_l_mutual=0.0) == c0
    assert E4.rungs_for("M_16pm", cfg2, cell)[0].bf_l_mutual == 0
    assert E4.model_record("M_16C2pm_us", cfg2, c2)["mutual_edge_angstrom"] == 2.0
    assert "mutual_edge_angstrom" not in E4.model_record("M_16C2pm_us", cfg, c0)
