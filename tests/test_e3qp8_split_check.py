"""E3-QP8's split check (scratch/e3qp8_split_check.py), the parts that need no build and no saved model:

  - the random rotations are rotations, spread over SO(3);
  - the scan's dimers are the training monomer twice, at the O-O distances asked, every intermolecular atom pair at least
    the minimum apart and every energy within the cap, as many per distance as asked;
  - the nearest-training-pair distance is zero for a training pair in any of its orderings, and the diagnostic's measure;
  - the readings: the seeds' disagreement, and per twin pair S (the self terms' difference), R (the energy difference)
    and the check that the energy difference is E_0's and the field term's."""
import importlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent


@pytest.fixture(scope="module")
def sc():
    for p in (ROOT / "scripts", ROOT / "scratch"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return importlib.import_module("e3qp8_split_check")


def test_random_rotations(sc):
    R = sc.random_rotations(np.random.default_rng(0), 4000)
    assert np.abs(np.einsum("nij,nkj->nik", R, R) - np.eye(3)).max() < 1e-12
    assert np.abs(np.linalg.det(R) - 1.0).max() < 1e-12
    assert np.abs(R.mean(axis=0)).max() < 0.05                       # uniform on SO(3): every entry averages to zero


def test_scan_dimers(sc):
    from sseft import water as W
    mono = W.monomer_positions()
    energy = lambda p: -1.0 / np.linalg.norm(p[0] - p[3]) + 3.0 * np.exp(-np.min(
        np.linalg.norm(p[:3, None] - p[None, 3:], axis=-1)))          # a stand-in: repulsive at the closest contact
    oo = (2.6, 3.0, 4.5)
    P, E, O, tally = sc.scan_dimers(energy, oo, 25, np.random.default_rng(1), cap=0.2, min_any=1.5)
    assert P.shape == (75, 6, 3) and E.shape == (75,) and sorted(set(O)) == list(oo)
    assert all(tally[d][0] == 25 and tally[d][1] >= 25 for d in oo)
    assert np.allclose(np.linalg.norm(P[:, 0] - P[:, 3], axis=-1), O)
    assert np.all(np.linalg.norm(P[:, :3, None] - P[:, None, 3:], axis=-1).reshape(len(P), -1).min(axis=1) >= 1.5)
    assert np.all(E <= 0.2) and np.allclose(E, [energy(p) for p in P])
    for p in P:                                                      # the training monomer, rigid, twice
        for m in (p[:3], p[3:]):
            d = lambda X: sorted(np.linalg.norm(X[i] - X[j]) for i, j in itertools.combinations(range(3), 2))
            assert np.allclose(d(m), d(mono))


def test_nearest(sc):
    rng = np.random.default_rng(2)
    pairs = [rng.normal(size=(6, 3)) * 2 for _ in range(5)]
    tp = np.array([sc.pair_desc(p[:3], p[3:])[0] for p in pairs])
    swapped = [np.concatenate([p[3:], p[[0, 2, 1]]]) for p in pairs]  # B before A, A's hydrogens exchanged
    assert np.allclose(sc.nearest(tp, np.array(pairs)), 0.0) and np.allclose(sc.nearest(tp, np.array(swapped)), 0.0)
    q = pairs[0].copy(); q[3:] += 0.3                                 # B shifted: every intermolecular distance moves
    assert sc.nearest(tp, q[None])[0] > 0.0


def test_readings(sc):
    rng = np.random.default_rng(3)
    n, seeds = 50, (0, 1, 2, 3)
    ref = rng.normal(size=n)
    bands = {s: rng.normal(size=n) for s in seeds}                   # the seed's own bands and read-out, every rung
    models = {}
    for name, scale in (("M_16C2pm", 0.0), ("M_16C2pm_ws", 0.3), ("M_16C2pm_us", 1.3)):
        models[name] = {}
        for s in seeds:
            self_ = scale * rng.normal(size=n)
            E0 = -self_ + 0.05 * rng.normal(size=n)
            q = {"E": bands[s] + E0 + self_, "E0": E0}
            if scale:
                q.update(field=self_, self=self_)
            models[name][s] = q
    masks = {"all": np.ones(n, bool), "half": np.arange(n) < n // 2}
    out = sc.readings(models, ref, masks)
    us = out["models"]["M_16C2pm_us"]["all"]
    assert us["n"] == n and len(us["error"]) == 4
    assert np.isclose(us["spread"], sc.rms(np.std([models["M_16C2pm_us"][s]["E"] for s in seeds], axis=0, ddof=1)))
    for row in out["twins"]["M_16C2pm_us - M_16C2pm_ws"]["half"]:
        a, b = models["M_16C2pm_us"][row["seed"]], models["M_16C2pm_ws"][row["seed"]]
        m = masks["half"]
        assert np.isclose(row["S"], sc.rms((a["self"] - b["self"])[m])) and np.isclose(row["R"], sc.rms((a["E"] - b["E"])[m]))
        assert row["check"] < 1e-12
    for row in out["twins"]["M_16C2pm_us - M_16C2pm"]["all"]:      # against the base: the self terms whole
        assert np.isclose(row["S"], sc.rms(models["M_16C2pm_us"][row["seed"]]["self"])) and row["check"] < 1e-12
