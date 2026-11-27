"""E4 job 16.5 (v0.0.44; notes/e4-job16.5-preregistration.md): MACE-LES at L_0 = 128, the readings K16.5.1-K16.5.4
(scripts/e4_job16_5.py) on constructed energies, the dagger's runs and eval.py's jsons, and their edges."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import e4_job16_5 as J                                     # noqa: E402

FRAMES = {"64": {"train": 12, "valid": 4, "test": 5}, "128": {"train": 6, "valid": 3, "test": 4},
          "256": {"train": 3, "valid": 2, "test": 2}, "512": {"test": 3}, "1024": {"test": 2}}
INV_L = {"64": 1 / 12.42, "128": 1 / 15.65, "256": 1 / 19.72, "512": 1 / 24.84, "1024": 1 / 31.30}


def _world(tmp, carry=(0.0, -0.010, -0.0176), carry128=None, c=(0.60, 0.61), c128=(0.61, 0.60), f128=1.01, dag=0.0,
           group="maceles"):
    """truth energies; MACE-LES at L_0 = 64 and 128 whose bias per molecule at sizes 64 / 128 / 256 is carry (L_0 = 64)
    and carry128 (default: the same carried energy, shifted so that it is zero at 128); the daggers' errors dag
    everywhere; eval.py's jsons with latent dipoles c (64) and c128 (128) times 1.864 D and forces 0.21 (64), times
    f128 at 128."""
    rng = np.random.default_rng(0)
    data = tmp / "data"; data.mkdir(exist_ok=True)
    truth = {}
    for n, by in FRAMES.items():
        for split, k in by.items():
            truth[(n, split)] = rng.normal(size=k) * 0.1
            np.savez(data / f"{split}_n{n}.npz", energy=truth[(n, split)])
    if carry128 is None:
        carry128 = tuple(x - carry[1] for x in carry)
    base = tmp / "baselines"
    big = lambda b, n: b[2] + (b[2] - b[1]) * (INV_L[n] - INV_L["256"]) / (INV_L["256"] - INV_L["128"])
    for L0, b, cc, ff in ((64, carry, c, 1.0), (128, carry128, c128, f128)):
        for s in J.SEEDS:
            stem = J.STEMS[group].format(L0=L0, seed=s)
            d = base / stem; d.mkdir(parents=True)
            z = {}
            for (n, split), E in truth.items():
                err = (b[J.SIZES.index(n)] if n in J.SIZES else big(b, n)) + rng.normal(size=len(E)) * 1e-4
                z[f"{split}_n{n}__energy"] = (E + err * int(n) / J.KCAL) * J.EV_PER_HARTREE
            np.savez(d / f"{stem}_stagetwo_{J.TAG}.npz", **z)
            sets = {f"test_n{n}": {"f_rmse_kcal_per_mol_A": 0.21 * ff, "latent_q_per_molecule_rms": 0.022,
                                   "latent_dipole_debye_mean": cc[s] * J.PERMANENT_DEBYE} for n in J.SIZES}
            json.dump({"sets": sets}, open(d / f"{stem}_stagetwo_eval.json", "w"))
    for key in ("dagger64", "dagger128"):
        d = tmp / key; d.mkdir(exist_ok=True)
        for s in J.SEEDS:
            cells = {n: {split: {"energy_per_molecule_errors": [dag] * k} for split, k in by.items()}
                     for n, by in FRAMES.items() if n in J.SIZES}
            json.dump({"rung": J.DAGGER, "seed": s, "readings": {"cells": cells}}, open(d / f"{J.DAGGER}_seed{s}.json", "w"))
    dirs = {"data": str(data), "baselines": str(base), "dagger64": str(tmp / "dagger64"), "dagger128": str(tmp / "dagger128")}
    return dirs


def test_the_job16_5_readings(tmp_path):
    """a MACE-LES whose error is one function of the cells, in 1/L (-0.010 from 64 to 128, -0.0176 to 256): K16.5.1
    holds with the step reversed exactly, K16.5.2 reads "1/L" (r = 0.0076 / 0.0176), K16.5.3 and K16.5.4 hold; the
    unseen frames are the test split at L_0 and every split elsewhere."""
    dirs = _world(tmp_path)
    runs = J.load(dirs)
    ev = J.evaluate(runs, dirs["data"])
    st = ev["steps"]
    assert st["s64(128)"] == pytest.approx(-0.010, abs=2e-4) and st["s128(64)"] == pytest.approx(0.010, abs=2e-4)
    assert st["s64(256)"] == pytest.approx(-0.0176, abs=2e-4) and st["s128(256)"] == pytest.approx(-0.0076, abs=2e-4)
    for k in ("K16.5.1", "K16.5.3", "K16.5.4"):
        assert ev[k]["holds"], (k, ev[k])
    assert ev["K16.5.2"]["verdict"] == "1/L" and ev["K16.5.2"]["ratio"] == pytest.approx(0.0076 / 0.0176, abs=0.02)
    assert ev["K16.5.2"]["from_64"] == pytest.approx(ev["K16.5.2"]["ratio"], abs=0.02)
    for n in J.SIZES + J.BIG:                              # one function of the cells: the L_0's difference everywhere
        assert ev["between"][n]["mean"] == pytest.approx(0.010, abs=2e-4), n
    assert ev["K16.5.1"]["ratio"] == pytest.approx(1.0, abs=0.03)
    assert ev["K16.5.3"]["relative"][0] == pytest.approx(0.61 / 0.60 - 1)
    assert ev["K16.5.4"]["relative"]["64"] == pytest.approx(0.01)
    z = runs[64][0]["z"]
    assert len(J.model_errors(z, dirs["data"], "64", 64)) == FRAMES["64"]["test"]
    assert len(J.model_errors(z, dirs["data"], "128", 64)) == sum(FRAMES["128"].values())
    assert len(J.model_errors(z, dirs["data"], "1024", 64)) == FRAMES["1024"]["test"]
    text = J.report(ev)
    for k in ("K16.5.1: holds", "K16.5.2: 1/L", "a 1/L law gives 0.44", "L_0 = 128: n 64", "n 1024 +0.0",
              "K16.5.5: not read (the control"):
        assert k in text, k
    _world(tmp_path, carry=(0.0, -0.0098, -0.0113), carry128=(0.0098, 0.0, -0.0026), group="mace")   # M_inf's steps
    mace = J.evaluate(J.load(dirs, "mace"), dirs["data"])
    assert J.control(mace) == {"verdict": "carried", "holds": True}
    assert "K16.5.5: holds (the control: MACE without LES reads \"carried\"): carried, r 0.23" in J.report(ev, mace)
    json.dumps(ev, default=float)


def test_the_job16_5_edges(tmp_path):
    """the edges: trained at 128 with no error anywhere fails K16.5.1 (near zero); K16.5.2's classes (carried, between,
    persists, reversed); a scale and forces that move fail K16.5.3 and K16.5.4; a step too small is not read; a missing
    model reads nothing."""
    for name, kw, key in (("flat", dict(carry128=(0.0, 0.0, 0.0)), "K16.5.1"),
                          ("scale", dict(c128=(0.70, 0.60)), "K16.5.3"),
                          ("forces", dict(f128=1.08), "K16.5.4")):
        d = tmp_path / name; d.mkdir()
        dirs = _world(d, **kw)
        ev = J.evaluate(J.load(dirs), dirs["data"])
        assert ev[key]["holds"] is False, (name, ev[key])
    for name, r, verdict in (("carried", 0.23, "carried"), ("between", 0.32, "between"), ("persists", 0.9, "persists"),
                             ("reversed", -0.2, "reversed")):
        d = tmp_path / name; d.mkdir()
        dirs = _world(d, carry128=(0.010, 0.0, -0.0176 * r))
        ev = J.evaluate(J.load(dirs), dirs["data"])
        assert ev["K16.5.2"]["verdict"] == verdict, (name, ev["K16.5.2"])
    d = tmp_path / "small"; d.mkdir()
    dirs = _world(d, carry=(0.0, -0.001, -0.002))
    ev = J.evaluate(J.load(dirs), dirs["data"])
    assert ev["K16.5.1"]["holds"] is None and ev["K16.5.1"]["too_small"]
    os.remove(os.path.join(dirs["baselines"], "maceles_n128_s1", f"maceles_n128_s1_stagetwo_{J.TAG}.npz"))
    ev = J.evaluate(J.load(dirs), dirs["data"])
    assert ev["K16.5.1"]["holds"] is None and ev["K16.5.2"]["verdict"] is None and "not read" in J.report(ev)
