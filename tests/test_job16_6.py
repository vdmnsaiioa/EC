"""E4 job 16.6 (v0.0.46; notes/e4-job16.6-preregistration.md): LES's reciprocal sum in NumPy -- by hand, against LES
itself where it is installed, a replica's invariance, the tail -- and K16.6.1-K16.6.4 on constructed models."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import e4_job16_6 as J                                     # noqa: E402
import e4_job16_5 as J5                                    # noqa: E402


def test_les_sum_by_hand():
    """a cubic cell of 10 A with k_max 0.7 / A holds the six k vectors of |n| = 1 alone: two charges +-0.5."""
    L, kmax = 10.0, 0.7
    q, r = np.array([0.5, -0.5]), np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    k0 = 2 * np.pi / L
    s2 = [0.5 - 0.5 * np.cos(k0 * d) for d in (1.0, 2.0, 3.0)]           # |S(k)|^2 along x, y, z (and their negatives)
    by_hand = J.NORM / L ** 3 * 2 * sum(np.exp(-k0 ** 2 / 2) / k0 ** 2 * s for s in s2) \
        - J.NORM * 0.5 / (2 * np.pi) ** 1.5
    assert J.les_energy(q, r, np.eye(3) * L, dl=2 * np.pi / kmax) == pytest.approx(by_hand, rel=1e-12)
    assert len(J.kvectors(np.eye(3) * L, 0.0, kmax)[0]) == 3


def test_replica_translation_and_tail():
    """a 2 x 2 x 2 replica holds eight times the energy and the tail; a translation changes neither; the tail is the
    energy at half dl less the energy at dl."""
    rng = np.random.default_rng(3)
    L = 7.0
    q = rng.normal(size=9) * 0.3
    r = rng.uniform(0, L, size=(9, 3))
    cell = np.eye(3) * L
    shifts = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) * L
    Q, R = np.tile(q, 8), np.concatenate([r + s for s in shifts])
    for fn in (J.les_energy, J.tail):
        assert fn(Q, R, cell * 2) == pytest.approx(8 * fn(q, r, cell), rel=1e-10)
        assert fn(q, r + 1.234, cell) == pytest.approx(fn(q, r, cell), rel=1e-10)
    assert J.tail(q, r, cell) == pytest.approx(J.les_energy(q, r, cell, dl=J.DL / 2) - J.les_energy(q, r, cell),
                                               rel=1e-10)
    tri = np.array([[7.0, 0.0, 0.0], [1.5, 6.5, 0.0], [0.5, -1.0, 8.0]])
    assert J.tail(q, r, tri) > 0


def test_against_les_itself():
    """the same energy as les 0.2.0's own Ewald (its legacy path), where it is installed (the mace-les env)."""
    torch = pytest.importorskip("torch")
    ewald = pytest.importorskip("les.module.ewald")
    rng = np.random.default_rng(5)
    for cell in (np.eye(3) * 12.42, np.diag([15.65, 15.65, 24.0]), np.array([[11.0, 0, 0], [2.0, 12.0, 0], [0, 1.0, 13.0]])):
        r = rng.uniform(0, 11, size=(24, 3))
        q = rng.normal(size=24) * 0.3
        ref = ewald.Ewald(dl=2.0, sigma=1.0)(torch.tensor(q), torch.tensor(r), torch.tensor(cell)[None])[0].item()
        assert J.les_energy(q, r, cell) == pytest.approx(ref, rel=1e-8, abs=1e-8)
        t_numpy = J.tail(q, r, cell)
        J.BACKEND["torch"] = torch                         # the torch path (the GPU's, on the CPU here)
        try:
            assert J.tail(q, r, cell) == pytest.approx(t_numpy, rel=1e-12)
            assert J.les_energy(q, r, cell) == pytest.approx(ref, rel=1e-8, abs=1e-8)
        finally:
            J.BACKEND["torch"] = None


TAIL = {"64": 0.0500, "128": 0.0544, "256": 0.0577, "512": 0.0516, "1024": 0.0533}   # kcal/mol per molecule
FR = {"64": {"train": 6, "valid": 3, "test": 3}, "128": {"train": 4, "valid": 2, "test": 2},
      "256": {"train": 2, "valid": 1, "test": 2}, "512": {"test": 2}, "1024": {"test": 2}}


def _world(tmp, g=None):
    """truth energies; MACE-LES at L_0 = 64 and 128 whose error per molecule is g(L) + Tbar(L_0) - T(L), T the tail
    above on every frame (and its tails file), g zero unless given."""
    rng = np.random.default_rng(0)
    g = g or {n: 0.0 for n in TAIL}
    data, base, out = tmp / "data", tmp / "baselines", tmp / "tails"
    for d in (data, base, out):
        d.mkdir()
    truth = {}
    for n, by in FR.items():
        for split, k in by.items():
            truth[(n, split)] = rng.normal(size=k) * 0.1
            np.savez(data / f"{split}_n{n}.npz", energy=truth[(n, split)])
    for L0 in J5.L0S:
        for s in J5.SEEDS:
            stem = f"maceles_n{L0}_s{s}"
            (base / stem).mkdir()
            z, t = {}, {}
            for (n, split), E in truth.items():
                err = g[n] + TAIL[str(L0)] - TAIL[n] + rng.normal(size=len(E)) * 1e-5
                z[f"{split}_n{n}__energy"] = (E + err * int(n) / J5.KCAL) * J5.EV_PER_HARTREE
                t[f"{split}_n{n}__tail"] = np.full(len(E), TAIL[n] * int(n) / J.KCAL_PER_EV)
            np.savez(base / stem / f"{stem}_stagetwo_{J5.TAG}.npz", **z)
            np.savez(out / f"{stem}_tails.npz", **t)
    return {"data": str(data), "baselines": str(base), "dagger64": str(tmp / "none"), "dagger128": str(tmp / "none"),
            "out": str(out)}


def _slabs(scale=1.0):
    """four frames, gaps 9 / 12 / 16 / 24; MB-pol's E(g) - E(24) about 6.7 / 4.4 / 2.4 kcal/mol per cell; the tail's
    change 0.3 / 0.2 / 0.1 times scale."""
    rng = np.random.default_rng(2)
    gaps = np.array([9.0, 12.0, 16.0, 24.0] * 4)
    frame = np.repeat(np.arange(4), 4)
    base = {9.0: -6.7, 12.0: -4.4, 16.0: -2.4, 24.0: 0.0}
    E = np.array([base[g] + rng.normal() * 0.3 for g in gaps])            # kcal/mol per cell
    T = np.array([{9.0: 0.3, 12.0: 0.2, 16.0: 0.1, 24.0: 0.0}[g] * scale for g in gaps]) + 3.2
    S = {"slab_frame": frame, "slab_gap": gaps, "energy": E / J.KCAL_PER_EV / 27.211386245988}
    z = {s: {"job17_slabs__energy": (E * 0.5 - T) / J.KCAL_PER_EV} for s in J5.SEEDS}
    t = {s: {"job17_slabs__tail": T / J.KCAL_PER_EV} for s in J5.SEEDS}
    return {"S": S, "z": z, "t": t}


def test_the_job16_6_readings(tmp_path):
    """a MACE-LES whose size dependence is the tail alone: K16.6.1-K16.6.3 hold, b + T flat at Tbar(L_0); the slabs'
    tail at 3-5 % of MB-pol's change holds K16.6.4; then the edges."""
    dirs = _world(tmp_path)
    ev = J.evaluate(J5.load(dirs), J.load_tails(dirs["out"], dirs), dirs["data"], _slabs())
    for k in ("K16.6.1", "K16.6.2", "K16.6.3", "K16.6.4"):
        assert ev[k]["holds"], (k, ev[k])
    assert ev["b"][64]["256"] == pytest.approx(-0.0077, abs=1e-4) and ev["c"][64]["256"] == pytest.approx(0.05, abs=1e-4)
    assert ev["K16.6.1"]["uncorrected"] == pytest.approx(0.0044, abs=1e-4)
    assert ev["K16.6.3"]["absorbed"] == pytest.approx(0.0044) and ev["K16.6.3"]["uncorrected"]["512"] == pytest.approx(
        0.0044, abs=1e-4)
    text = J.report(ev)
    for k in ("K16.6.1: holds", "K16.6.4: holds", "beside: K17.8", "L_0 = 128 (its training frames' tail +0.05440)"):
        assert k in text, k
    # the edges: a genuine drift the tail does not explain; a slab tail that matters; nothing there
    d2 = tmp_path / "drift"; d2.mkdir()
    dirs2 = _world(d2, g={"64": 0.0, "128": -0.004, "256": -0.010, "512": -0.002, "1024": -0.004})
    ev2 = J.evaluate(J5.load(dirs2), J.load_tails(dirs2["out"], dirs2), dirs2["data"], _slabs(scale=10.0))
    assert ev2["K16.6.1"]["holds"] is False and ev2["K16.6.2"]["holds"] is False and ev2["K16.6.4"]["holds"] is False
    assert ev2["K16.6.3"]["holds"]                          # the same g at both L_0: the shift is still the tail's
    ev0 = J.evaluate(J5.load(dirs), {}, dirs["data"])
    assert all(ev0[k]["holds"] is None for k in ("K16.6.1", "K16.6.2", "K16.6.3", "K16.6.4"))
    assert "not read" in J.report(ev0)
