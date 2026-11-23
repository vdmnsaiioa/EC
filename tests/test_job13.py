"""
E4 job 13 (v0.0.37; notes/e4-job13-preregistration.md): learned charges by species alone (train.fit's
freeze_charge_head), pinned charges released into learned ones at the same scale (e4_water.released), the start from
another rung's run (training.start_from.from_rung), the two configs and the readings K13.1-K13.5
(scripts/e4_job13.py).
"""
import json
import os
import pickle
import sys

import numpy as np
import jax
import pytest

import sseft  # noqa: F401  (x64)
from sseft import model as M, train as T, structure as S

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))
import e4_water as E4                                     # noqa: E402
import e4_job13 as J13                                    # noqa: E402
from test_e4_train import _fake_cells, _small_cfg, FakeMBPol   # noqa: E402
from test_job7 import _run                                      # noqa: E402
from test_job9 import _cells, _rung                             # noqa: E402
from test_job10 import _train                                   # noqa: E402

KW = dict(seed=0, lr=3e-3, w_energy=100.0, w_force=1.0, batch_size=2, force_mode="rigid", lbfgs_chunk=2)
leaves = jax.tree_util.tree_leaves


def _same(a, b):
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(leaves(a), leaves(b)))


def test_the_charge_head_held_leaves_charges_by_species(tmp_path):
    """freeze_charge_head: the charge head's layers do not move through Adam or the polish, its species offsets do, and
    from a warm start (the last layer at zero) every atom carries its species' charge in every environment.  It needs
    learned charges."""
    st = _cells(tmp_path)
    rung = _rung("M_1z150", st)
    assert rung.q_unit == E4.Q_UNIT_WARM and rung.q_ref_start and "M_1z150" in E4.SPECIES_ONLY
    p0 = M.init_params(jax.random.PRNGKey(0), rung, energy_scale=1.0)
    p, info = T.fit(rung, st, steps=2, lbfgs_steps=2, freeze_charge_head=True, **KW)
    assert _same(p["heads"]["charge"], p0["heads"]["charge"])
    assert not np.array_equal(np.asarray(p["heads"]["q_ref"]), np.asarray(p0["heads"]["q_ref"]))
    assert not _same(p["e0"], p0["e0"]) and info["training_options"]["freeze_charge_head"] is True
    q = np.asarray(M.aux_batched(p, rung, S.pad_batch(st, n_freq=rung.K))["q"])[:, :st[0].n_atoms]
    assert np.ptp(q[:, 0::3]) < 1e-14 and np.ptp(q[:, 1::3]) < 1e-14 and np.ptp(q[:, 2::3]) < 1e-14
    assert q[0, 0] == pytest.approx(-2 * q[0, 1], rel=1e-12)
    p1, i1 = T.fit(rung, st, steps=2, lbfgs_steps=2, **KW)                 # the head learned: environments differ
    assert not _same(p1["heads"]["charge"], p0["heads"]["charge"]) and "freeze_charge_head" not in i1["training_options"]
    with pytest.raises(ValueError, match="learned"):
        T.fit(_rung("M_1q070", st), st, steps=1, freeze_charge_head=True, **KW)
    with pytest.raises(ValueError, match="boolean"):
        T.fit(rung, st, steps=1, freeze_charge_head=1, **KW)


def test_a_released_start_is_the_pinned_model(tmp_path):
    """released: a pinned run's parameters started as learned charges at the pins' scale -- the species offsets at the
    start charges, the head's last layer at zero -- give the pinned model's energies and forces, with the head or by
    species; a rung without start charges cannot be released into."""
    st = _cells(tmp_path)
    pinned = _rung("M_1q070", st)
    p0, _ = T.fit(pinned, st, steps=2, lbfgs_steps=0, **KW)
    data = S.pad_batch(st, n_freq=pinned.K)
    E0, F0 = (np.asarray(x) for x in M.energy_forces(p0, pinned, data))
    for name in ("M_1r070", "M_1zr070"):
        rung = _rung(name, st)
        assert dict(rung.q_ref_start)[8] == pytest.approx(0.7 * float(st[0].pinned["q"][0]))
        p = E4.released(p0, rung)
        E1, F1 = (np.asarray(x) for x in M.energy_forces(p, rung, data))
        assert np.max(np.abs(E1 - E0)) <= 1e-12 * np.max(np.abs(E0)) and np.max(np.abs(F1 - F0)) <= 1e-12 * np.max(np.abs(F0))
        assert not np.any(np.asarray(p["heads"]["charge"][-1]["w"])) and not np.any(np.asarray(p["heads"]["charge"][-1]["b"]))
        assert _same(p["e0"], p0["e0"]) and _same(p["heads"]["charge"][:-1], p0["heads"]["charge"][:-1])
    with pytest.raises(ValueError, match="start charges"):
        E4.released(p0, _rung("M_1", st))


def test_driver_starts_from_another_rung(tmp_path):
    """training.start_from.from_rung: a run starts from another rung's run of the same seed -- released when that rung
    pins the charges this one learns (the same objective at the start), as it is otherwise (the transplant: the pinned
    charges at another scale) -- records where from, and holds the charge head for the rungs by species.  A source
    whose json records another model is refused, as is an unknown source."""
    cfg = _small_cfg(tmp_path)
    truth = FakeMBPol()
    E4.save_probes(cfg["data"]["probes"], E4.make_probes(truth, cfg), truth, cfg)
    for split, nf in (("train", 2), ("test", 1)):
        _fake_cells(tmp_path / f"{split}_n16.npz", nf, 16, 12.5, seed=len(split))
    cfg["seeds"] = [0]
    cfg["model"]["features"] = 8
    cfg["training"]["adam"].update(steps=2, batch_size=2)
    cfg["training"]["lbfgs"].update(steps=0, chunk_size=2)
    first = tmp_path / "first"
    cfg["rungs"] = ["M_1q070"]
    _train(tmp_path, cfg, first)
    cfg["rungs"] = ["M_1zr070", "M_1r070", "M_1t070"]
    cfg["training"]["adam"]["steps"] = 0
    cfg["training"]["lbfgs"]["steps"] = 2
    cfg["training"]["start_from"] = {"dir": str(first), "from_rung": {n: "M_1q070" for n in cfg["rungs"]}}
    cfg["training"]["trace"] = {"every": 100, "frames": 1, "lbfgs_every": 1}
    out = tmp_path / "out"
    _train(tmp_path, cfg, out)
    r0 = json.load(open(first / "M_1q070_seed0.json"))
    rz, rr, rt = (json.load(open(out / f"{n}_seed0.json")) for n in cfg["rungs"])
    for r, rel in ((rz, True), (rr, True), (rt, False)):
        b = r["training"]["start"]
        assert b["from"] == str(first / "M_1q070_seed0.params.pkl") and b["from_rung"] == "M_1q070" and b["released"] is rel
    for r in (rz, rr):                                               # released: the pinned model at the start
        b = r["training"]["start"]
        assert b["final_loss"]["total"] == pytest.approx(r0["training"]["final_loss"]["total"], rel=1e-10)
        assert b["charges"]["dipole_debye"] == pytest.approx(rz["training"]["start"]["charges"]["dipole_debye"],
                                                             rel=1e-12) and b["charges"]["dipole_debye"] > 0
        assert r["training"]["charges"]["lbfgs_trace"][0]["objective"] == pytest.approx(b["final_loss"]["total"], rel=1e-10)
    assert rt["training"]["start"]["final_loss"]["total"] > r0["training"]["final_loss"]["total"]   # moved to 1.1
    assert "charges" not in rt["training"] and rt["model"]["pinned_charge_scale"] == 1.1
    assert rz["training"]["options"]["freeze_charge_head"] is True and "freeze_charge_head" not in rr["training"]["options"]
    with open(out / "M_1zr070_seed0.params.pkl", "rb") as fh:
        pz = pickle.load(fh)
    assert not np.any(pz["heads"]["charge"][-1]["w"]) and not np.any(pz["heads"]["charge"][-1]["b"])
    with open(out / "M_1r070_seed0.params.pkl", "rb") as fh:
        assert np.any(pickle.load(fh)["heads"]["charge"][-1]["w"])
    assert rz["model"] == json.loads(json.dumps(E4.model_record("M_1zr070", cfg, _rung_cfg("M_1zr070", cfg, tmp_path))))
    cfg["rungs"] = ["M_1zr070"]
    cfg["training"]["start_from"]["from_rung"] = {"M_1zr070": "M_1q100"}
    os.rename(first / "M_1q070_seed0.json", first / "M_1q100_seed0.json")
    os.rename(first / "M_1q070_seed0.params.pkl", first / "M_1q100_seed0.params.pkl")
    with pytest.raises(ValueError, match="another model"):
        _train(tmp_path, cfg, tmp_path / "again")
    cfg["training"]["start_from"]["from_rung"] = {"M_1zr070": "M_1q999"}
    with pytest.raises(ValueError, match="unknown rung"):
        _train(tmp_path, cfg, tmp_path / "again2")


def _rung_cfg(name, cfg, tmp_path):
    from sseft import water as W
    train = W.e4_structures(str(tmp_path / "train_n16.npz"), pins=True)
    rung, _ = E4.rungs_for(name, cfg, train[0].cell)
    return E4.warm_started(name, rung, train)


def test_job13s_configs_rungs_and_reader_agree():
    """job 13's first stage is job 9's settings for the species-only warm starts; its polishes are job 11's settings
    from job 11's polished profile (R, T) and the first stage (S).  The rungs: R and S learned at job 7's unit, T job 7's
    M_1q110 under other names; the reader's arms are the driver's."""
    s = E4.load_config(os.path.join(REPO, "configs", "e4_job13s.yaml"))
    j9 = E4.load_config(os.path.join(REPO, "configs", "e4_job9.yaml"))
    assert s["rungs"] == ["M_1z050", "M_1z150"] and s["seeds"] == [0, 1]
    assert s["training"]["trace"] == {"every": 100, "frames": 4, "lbfgs_every": 25}
    for key in ("data", "model", "probes", "reading"):
        assert s[key] == j9[key]
    for key in ("adam", "lbfgs", "loss", "readout"):
        assert s["training"][key] == j9["training"][key]
    p = E4.load_config(os.path.join(REPO, "configs", "e4_job13.yaml"))
    j11 = E4.load_config(os.path.join(REPO, "configs", "e4_job11.yaml"))
    assert p["rungs"] == ["M_1r070", "M_1r145", "M_1zr070", "M_1zr145", "M_1t070", "M_1t145", "M_1z050", "M_1z150"]
    st = p["training"]["start_from"]
    assert st["dir"] == "results/e4_job11" and st["by_rung"] == {"M_1z050": "results/e4_job13s",
                                                                 "M_1z150": "results/e4_job13s"}
    assert st["from_rung"] == {"M_1r070": "M_1q070", "M_1r145": "M_1q145", "M_1zr070": "M_1q070",
                               "M_1zr145": "M_1q145", "M_1t070": "M_1q070", "M_1t145": "M_1q145"}
    for key in ("data", "model", "probes", "reading"):
        assert p[key] == j11[key] == j9[key]
    for key in ("adam", "lbfgs", "loss", "readout", "trace"):
        assert p["training"][key] == j11["training"][key]
    assert {**J13.BY_SPECIES, **J13.WITH_HEAD} == E4.RELEASE_C and J13.SPECIES == E4.SPECIES_C
    assert set(E4.SPECIES_ONLY) == set(J13.BY_SPECIES) | set(J13.SPECIES) and tuple(J13.TRANSPLANT) == E4.TRANSPLANT
    for name in list(E4.RELEASE_C) + list(E4.SPECIES_C):
        assert E4.E3.RUNGS[name] == M.with_rung(E4.E3.RUNGS["M_1w050"], name=name) and E4.STARTS[name] == \
            {**E4.RELEASE_C, **E4.SPECIES_C}[name]
    for name in E4.TRANSPLANT:
        assert E4.E3.RUNGS[name] == M.with_rung(E4.E3.RUNGS["M_1q110"], name=name)
    assert {J13.SOURCE[c] for c in J13.TRANSPLANT.values()} == {"M_1q070", "M_1q145"}


def _job13(rs=(1.05, 1.08, 1.06, 1.09), rf=(0.88, 1.15, 0.90, 1.17), t_loss=(21.5, 22.0, 21.8, 22.2),
           sp=(1.04, 1.10, 1.05, 1.09), missing=()):
    """job 13's runs and its references (the first stage, jobs 9, 10 and 11) on constructed jsons; rs, rf, sp: the
    ends in c by (start low seed 0, start high seed 0, start low seed 1, start high seed 1)."""
    sizes = J13.SIZES
    runs, stage, job9, job10, job11 = {}, {}, {}, {}, {}
    add = lambda d, r: d.setdefault(r["rung"], {}).__setitem__(r["seed"], r)
    dip = lambda c: c * J13.PERMANENT_DEBYE
    for s in J13.SEEDS:
        for name, L in (("M_1q070", 30.75), ("M_1q100", 23.63), ("M_1q145", 31.44)):
            add(job11, _run(name, s, {n: 0.6 for n in sizes}, loss=L))
        add(job10, _run("M_1q110", s, {n: 0.52 for n in sizes}, loss=(21.0, 21.9)[s]))
        for name, d in (("M_1w050", (1.504, 1.648)), ("M_1w150", (2.108, 2.184))):
            add(job10, _run(name, s, {n: 0.55 for n in sizes}, dipole=d[s], loss=25.0))
            add(job9, _run(name, s, {n: 0.78 for n in sizes}, dipole=(1.2, 2.5)[name == "M_1w150"], loss=46.0))
        add(job11, _run("M_1w050", s, {n: 0.56 for n in sizes}, dipole=(1.608, 1.588)[s], loss=23.8))
        for k, (zr, rr, z) in enumerate(zip(J13.BY_SPECIES, J13.WITH_HEAD, J13.SPECIES)):
            i = 2 * s + k
            for name, c in ((zr, rs[i]), (rr, rf[i]), (z, sp[i])):
                if (name, s) in missing:
                    continue
                r = _run(name, s, {n: 0.53 for n in sizes}, dipole=dip(c), loss=22.0)
                r["training"]["charges"] = {"lbfgs_trace": [{"step": t, "dipole_debye": dip(c) - 0.1 * (1 - t / 1000),
                                                             "objective": None} for t in (0, 250, 500, 750, 1000)],
                                            "scale_curve": {lab: {"s": [0.99, 1.0, 1.01], "loss": [L + 1e2, L, L + 1e2]}
                                                            for lab, L in (("start", 30.0), ("final", 22.0))}}
                r["training"]["lbfgs_trace"] = [[0, 30.0], [250, 25.0], [1000, None]]
                add(runs, r)
            add(stage, _run(z, s, {n: 0.75 for n in sizes}, dipole=dip(sp[i] - 0.1), loss=45.0))
        for k, name in enumerate(J13.TRANSPLANT):
            r = _run(name, s, {n: 0.52 for n in sizes}, loss=t_loss[2 * s + k])
            r["training"]["start"] = {"final_loss": {"total": 1e4}}
            add(runs, r)
    return runs, stage, job9, job10, job11


def test_the_job13_readings():
    """K13.1-K13.5 on constructed jsons, then each band's edge and a missing run."""
    ev = J13.evaluate(*_job13())
    assert ev["profile"]["c_min"] == pytest.approx(1.0704, abs=1e-3)
    assert ev["K13.1"]["holds"] and ev["K13.1"]["runs"]["M_1zr145 seed 0"]["off"] == pytest.approx(1.08 - 1.07)
    assert ev["K13.2"]["holds"] is False and ev["K13.5"]["holds"]
    assert ev["K13.3"]["holds"] and ev["K13.3"]["species_nearer"] == 4
    assert ev["K13.4"]["holds"] and ev["K13.4"]["runs"]["M_1t145 seed 1"]["ratio"] == pytest.approx(22.2 / 21.9 - 1)
    assert ev["paths"]["M_1zr070 seed 0"]["last_250"]["moved"] == pytest.approx(0.025)
    assert ev["paths"]["M_1zr070 seed 0"]["curvature"]["start"] == pytest.approx(2e6)
    assert ev["against_profile"]["M_1t070 seed 0"]["c"] == 1.1
    assert ev["stage"]["c0 0.5 seed 0"]["flexible_debye"] == 1.2 and ev["polish"]["c0 1.5 seed 1"]["flexible_job10_debye"] == 2.184
    text = J13.report(ev)
    for k in ("K13.1: holds", "K13.2: FAILS", "K13.3: holds", "K13.4: holds", "K13.5: holds", "c*_lb = 1.07"):
        assert k in text
    assert J13.evaluate(*_job13(rs=(1.019, 1.08, 1.06, 1.09)))["K13.1"]["holds"] is False      # 0.051 short
    assert J13.evaluate(*_job13(rf=(1.05, 1.08, 1.06, 1.09)))["K13.2"]["holds"]
    ev = J13.evaluate(*_job13(rf=(1.06, 1.15, 1.07, 1.17)))                                   # 2 of 4 nearer
    assert ev["K13.3"]["species_nearer"] == 2 and ev["K13.3"]["holds"] is False
    assert J13.evaluate(*_job13(t_loss=(21.5, 22.1, 21.8, 22.2)))["K13.4"]["holds"] is False   # +5.2 % on seed 0
    assert J13.evaluate(*_job13(sp=(0.85, 1.15, 0.88, 1.16)))["K13.5"]["holds"] is False
    ev = J13.evaluate(*_job13(missing=(("M_1zr145", 1),)))
    assert ev["K13.1"]["holds"] is None and ev["K13.3"]["holds"] is None
