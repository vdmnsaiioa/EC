"""
E4 job 12 (v0.0.36; notes/e4-job12-preregistration.md): D''s response precomputed -- the read-out's energy and forces on
every training frame computed once as D' less the field-term dagger (model.response_constant, model.is_response_free,
scripts/e4_water.py's response_constants) and the dagger trained against the remainder (training.response_precomputed)
-- and the readings K12.0-K12.3 and the model rule (scripts/e4_job12.py).
"""
import json
import os
import pickle
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import jax
import pytest

import sseft  # noqa: F401  (x64)
from sseft import model as M, train as T, water as W

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))
import e4_water as E4                                     # noqa: E402
import e4_job12 as J12                                    # noqa: E402
from test_e4_train import _fake_cells, _small_cfg, FakeMBPol   # noqa: E402
from test_job7 import _run                                      # noqa: E402
from test_job11 import _infra                                   # noqa: E402

KW = dict(seed=0, lr=3e-3, w_energy=100.0, w_force=1.0, batch_size=2, force_mode="rigid", lbfgs_chunk=2)


def _perturbed(readout):
    """a read-out moved off its initialisation (whose output layers are zero), as a matched one is."""
    return jax.tree_util.tree_map(lambda x: 1.1 * np.asarray(x) + 0.01 if np.issubdtype(np.asarray(x).dtype, np.floating)
                                  else np.asarray(x), readout)


def _setup(tmp_path, frames=2):
    cfg = _small_cfg(tmp_path)
    _fake_cells(tmp_path / "train_n16.npz", frames, 16, 12.5, seed=5)
    cfg["model"].update(features=8, mutual_edge_angstrom=2.0)
    train = W.e4_structures(str(tmp_path / "train_n16.npz"), pins=True)
    scale = T.energy_scale(train)
    rung, _ = E4.rungs_for(J12.DPRIME, cfg, train[0].cell)
    dagger, _ = E4.rungs_for(J12.FIELD, cfg, train[0].cell)
    p = M.init_params(jax.random.PRNGKey(3), rung, energy_scale=scale)
    return cfg, train, scale, rung, dagger, {**p, "readout": _perturbed(p["readout"])}


def test_the_response_is_a_constant_of_the_rest_and_the_dagger_is_dprime_less_it(tmp_path):
    """D''s read-out reads the species and the pinned sources' band field only: its energy and forces (D' less the
    field-term dagger) do not change when every other parameter does, and the dagger against the remainder is D''s
    objective.  The dagger at the 2.0 A band is D' less its response; the one at the run's edge and D' itself are not;
    learned sources or a read-out from E_0's features make the response depend on the rest."""
    cfg, train, scale, rung, dagger, p = _setup(tmp_path)
    assert M.response_constant(rung) and M.is_response_free(rung, dagger)
    assert not M.is_response_free(rung, E4.rungs_for("M_16pm_us", cfg, train[0].cell)[0])     # its c6 band at 0.75 A
    assert not M.is_response_free(rung, rung)
    assert M.response_constant(E4.rungs_for("M_16S2pm", cfg, train[0].cell)[0])                 # first order, species
    assert not M.response_constant(replace(rung, bf_species=False))                             # E_0's features
    assert not M.response_constant(replace(rung, msites=False, pin_q=False))                    # learned charges
    assert not M.response_constant(dagger)
    alt = M.init_params(jax.random.PRNGKey(7), rung, energy_scale=scale)
    RE, RF, rec = E4.response_constants(p, rung, dagger, train, alt=alt)
    assert rec["frames"] == 2 and rec["energy_rms"] > 1e-4 and rec["force_rms"] > 1e-5
    assert rec["invariance"]["frames"] == 2 and rec["invariance"]["energy"] < 1e-12 and rec["invariance"]["forces"] < 1e-12
    rem = E4.remainder(train, RE, RF)
    assert rem[0].energy == pytest.approx(train[0].energy - RE[0], abs=1e-15) and train[0].forces is not rem[0].forces
    full = E4.final_loss(p, rung, train, scale, 100.0, 1.0, 2, "rigid")
    red = E4.final_loss(E4.shared_params(p), dagger, rem, scale, 100.0, 1.0, 2, "rigid")
    for k in ("total", "energy", "forces"):
        assert red[k] == pytest.approx(full[k], rel=1e-12)
    # an E_0 that reads the read-out's input would break it: the read-out's coefficients from E_0's features
    learned = replace(rung, bf_species=False)
    q = M.init_params(jax.random.PRNGKey(3), learned, energy_scale=scale)
    q = {**q, "readout": _perturbed(q["readout"])}                     # its output layers start at zero
    _, _, rec2 = E4.response_constants(q, learned, dagger, train[:1], alt=M.init_params(jax.random.PRNGKey(7), learned,
                                                                                          energy_scale=scale), check=1)
    assert rec2["energy_rms"] > 1e-6 and rec2["invariance"]["energy"] > 1e-6


def test_the_polish_against_the_remainder_is_dprimes_polish(tmp_path):
    """train.fit on the dagger against the remainder, from D''s parameters less the read-out, takes the L-BFGS steps D''s
    own polish takes with the read-out held: the same objective at every step and the same parameters after."""
    _, train, scale, rung, dagger, p = _setup(tmp_path)
    RE, RF, _ = E4.response_constants(p, rung, dagger, train)
    pd, i_d = T.fit(dagger, E4.remainder(train, RE, RF), steps=0, lbfgs_steps=2, params_init=E4.shared_params(p),
                    scale=scale, trace_fn=lambda q: {}, lbfgs_trace_every=1, **KW)
    pf, i_f = T.fit(rung, train, steps=0, lbfgs_steps=2, params_init=p, readout_init=p["readout"], scale=scale,
                    trace_fn=lambda q: {}, lbfgs_trace_every=1, **KW)
    a, b = [v for _, _, v in i_d["lbfgs_trace"]], [v for _, _, v in i_f["lbfgs_trace"]]
    assert a[-1] is None and b[-1] is None and a[:-1] == pytest.approx(b[:-1], rel=1e-12)
    assert a[1] < a[0]
    la, lb = jax.tree_util.tree_leaves(pd), jax.tree_util.tree_leaves(E4.shared_params(pf))
    assert len(la) == len(lb)
    top = max(float(np.max(np.abs(np.asarray(x)))) for x in lb)
    assert max(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(la, lb)) <= 1e-12 * top
    for x, y in zip(jax.tree_util.tree_leaves(pf["readout"]), jax.tree_util.tree_leaves(p["readout"])):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_driver_polishes_with_the_response_precomputed(tmp_path):
    """scripts/e4_water.py train with training.response_precomputed: D' started from an earlier run, its response
    precomputed and checked, the dagger polished against the remainder, D' put back together with the held read-out
    and read; the json records the check, the objectives at the start and the end and the trace from D''s objective
    at the start.  A dagger that is not D' less its response, or a rung whose response is not a constant, is refused."""
    import yaml
    cfg = _small_cfg(tmp_path)
    truth = FakeMBPol()
    E4.save_probes(cfg["data"]["probes"], E4.make_probes(truth, cfg), truth, cfg)
    for split, nf in (("train", 2), ("test", 1)):
        _fake_cells(tmp_path / f"{split}_n16.npz", nf, 16, 12.5, seed=len(split))
    cfg["model"].update(features=8, mutual_edge_angstrom=2.0)
    cfg["training"]["adam"].update(steps=1, batch_size=2)
    cfg["training"]["lbfgs"].update(steps=0, chunk_size=2)
    cfg["reading"]["eps_frames"] = 0
    ref_dir = tmp_path / "job5"
    cfg["training"]["readout"]["held_from"] = {"rung": "M_16C2pm_us", "dir": str(ref_dir)}
    cfg["rungs"], cfg["seeds"] = [J12.DPRIME], [0]
    train = W.e4_structures(str(tmp_path / "train_n16.npz"), pins=True)
    scale = T.energy_scale(train)
    probes = E4.load_probes(cfg["data"]["probes"])
    _, ref_open = E4.rungs_for("M_16C2pm_us", cfg, train[0].cell)
    p0 = M.init_params(jax.random.PRNGKey(3), ref_open, energy_scale=scale)
    readout = _perturbed(p0["readout"])
    os.makedirs(ref_dir)
    with open(ref_dir / f"matched_M_16C2pm_us_{probes['hash']}.pkl", "wb") as fh:
        pickle.dump({"rung": E4.rung_key(ref_open), "scale": scale, "readout": readout,
                     "match": {"objective": [(1.0, 0.5)]}}, fh)

    def train_with(c, out):
        with open(tmp_path / "c.yaml", "w") as fh:
            yaml.safe_dump(c, fh)
        E4.run_train(SimpleNamespace(config=str(tmp_path / "c.yaml"), rungs=None, seeds=None, task=None,
                                     out_dir=str(out)))

    first = tmp_path / "first"
    train_with(cfg, first)
    cfg["training"]["adam"]["steps"] = 0
    cfg["training"]["lbfgs"]["steps"] = 2
    cfg["training"]["start_from"] = {"dir": str(first)}
    cfg["training"]["trace"] = {"every": 100, "frames": 1, "lbfgs_every": 1}
    cfg["training"]["response_precomputed"] = {"dagger": J12.FIELD}
    out = tmp_path / "out"
    train_with(cfg, out)
    r0, r = json.load(open(first / f"{J12.DPRIME}_seed0.json")), json.load(open(out / f"{J12.DPRIME}_seed0.json"))
    pre = r["training"]["response_precomputed"]
    assert pre["dagger"] == J12.FIELD and pre["frames"] == 2
    assert pre["invariance"]["energy"] <= E4.RESPONSE_TOL and pre["invariance"]["forces"] <= E4.RESPONSE_TOL
    st, en = pre["objective_start"], pre["objective_end"]
    assert st["dagger"] == pytest.approx(st["rung"], rel=1e-10) and en["dagger"] == pytest.approx(en["rung"], rel=1e-10)
    assert st["rung"] == pytest.approx(r0["training"]["final_loss"]["total"], rel=1e-10)
    assert en["rung"] == r["training"]["final_loss"]["total"] < st["rung"]
    tr = r["training"]["lbfgs_trace"]
    assert [s for s, _ in tr] == [0, 1, 2] and tr[0][1] == pytest.approx(st["rung"], rel=1e-10)
    assert r["model"] == r0["model"] and r["training"]["fit_time"] > 0
    with open(out / f"{J12.DPRIME}_seed0.params.pkl", "rb") as fh:
        trained = pickle.load(fh)
    for a, b in zip(jax.tree_util.tree_leaves(trained["readout"]), jax.tree_util.tree_leaves(readout)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert set(trained) == {"e0", "heads", "readout", "field", "field_x"}
    assert "16" in r["readings"]["cells"] and "probes" in r["readings"]
    cfg["training"]["response_precomputed"] = {"dagger": "M_16pm_us"}
    with pytest.raises(ValueError, match="less its response"):
        train_with(cfg, tmp_path / "again")
    cfg["training"]["response_precomputed"] = {"dagger": J12.FIELD}
    cfg["rungs"] = [J12.FIELD]
    cfg["training"]["start_from"] = None
    with pytest.raises(ValueError, match="not a constant"):
        train_with(cfg, tmp_path / "again")


def test_job12s_config_and_reader_agree():
    """job 12's config is job 11's D' config with seeds 0-3, a 360-minute budget and the response precomputed."""
    cfg = E4.load_config(os.path.join(REPO, "configs", "e4_job12.yaml"))
    d11 = E4.load_config(os.path.join(REPO, "configs", "e4_job11d.yaml"))
    assert cfg["rungs"] == [J12.DPRIME] and cfg["seeds"] == list(J12.SEEDS)
    tr = cfg["training"]
    assert tr["response_precomputed"] == {"dagger": J12.FIELD} and tr["start_from"] == {"dir": "results/e4_job8"}
    assert tr["adam"]["steps"] == 0 and tr["lbfgs"]["steps"] == J12.STEPS and tr["lbfgs"]["max_minutes"] == 360
    assert tr["trace"] == d11["training"]["trace"] and tr["readout"] == d11["training"]["readout"]
    assert {k: v for k, v in tr["lbfgs"].items() if k != "max_minutes"} == \
        {k: v for k, v in d11["training"]["lbfgs"].items() if k != "max_minutes"}
    for key in ("data", "model", "probes", "reading", "liquid", "liquid_k"):
        assert cfg[key] == d11[key]
    assert tr["loss"] == d11["training"]["loss"] and tr["adam"] == d11["training"]["adam"]
    assert E4.DEFAULTS["training"]["response_precomputed"] is None


def _pre(start, end, inv=(1e-15, 1e-14), dagger=J12.FIELD):
    return {"dagger": dagger, "invariance": {"energy": inv[0], "forces": inv[1]},
            "objective_start": {"rung": start, "dagger": start * (1 + 1e-12)},
            "objective_end": {"rung": end, "dagger": end * (1 - 1e-12)}}


def _job12(dprime=(0.440, 0.440, 0.442), field=(0.440, 0.453, 0.456), infra=(1.6893, (2.23, 1.98, 2.66), 4.31),
           start11=18.9, inv=(1e-15, 1e-14), missing=()):
    """job 12's runs and its references (jobs 8 and 11) on constructed jsons."""
    sizes = J12.SIZES
    runs, job8, job11 = {}, {}, {}
    add = lambda d, r: d.setdefault(r["rung"], {}).__setitem__(r["seed"], r)
    for s in J12.SEEDS:
        add(job8, _infra(_run(J12.DPRIME, s, dict(zip(sizes, (0.478, 0.485, 0.487))), loss=18.9),
                         1.6893, (2.23, 1.98, 2.66), 4.31))
        if s in missing:
            continue
        r = _infra(_run(J12.DPRIME, s, dict(zip(sizes, dprime)), loss=15.0), infra[0], infra[1], infra[2])
        r["training"].update(response_precomputed=_pre(18.9, 15.0, inv), lbfgs_trace=[[0, 18.9], [25, 18.0], [50, 17.5],
                                                                                         [1000, None]],
                             options={"lbfgs_steps": 1000}, fit_time=12000.0)
        add(runs, r)
    for s in J12.PAIRED:
        add(job11, _run(J12.FIELD, s, dict(zip(sizes, field)), loss=24.0))
        r = _run(J12.DPRIME, s, dict(zip(sizes, (0.461, 0.464, 0.466))), loss=16.0)
        r["training"]["lbfgs_trace"] = [[0, start11], [25, 18.0], [50, 17.5], [277, None]]
        add(job11, r)
    return runs, job8, job11


def test_the_job12_readings():
    """K12.0-K12.3 and the model rule on constructed jsons, then each band's edge."""
    runs, job8, job11 = _job12()
    ev = J12.evaluate(runs, job8, job11)
    assert ev["K12.0"]["holds"] and ev["K12.0"]["runs"]["seed 0"]["job11_step0"] == pytest.approx(0.0, abs=1e-15)
    assert "job11_step0" not in ev["K12.0"]["runs"]["seed 2"]
    assert ev["K12.1"]["holds"] and ev["K12.1"]["values"]["64"][2] == pytest.approx(0.440 / 0.478 - 1)
    assert ev["K12.2"]["holds"] and ev["model_rule"]["fires"]
    assert ev["K12.3"]["holds"] and set(ev["K12.3"]["values"]) == {"128", "256"}
    assert ev["K12.3"]["values"]["128"][2] == pytest.approx(0.440 / 0.453 - 1)
    assert ev["equal_steps_64"]["64"][2] == pytest.approx(0.0, abs=1e-15)
    assert ev["paths"]["seed 0"]["seconds_per_step"] == pytest.approx(12.0)
    assert ev["paths"]["seed 0"]["job11_dprime"]["0"][2] == pytest.approx(0.0, abs=1e-15)
    text = J12.report(ev)
    for k in ("K12.0: holds", "K12.1: holds", "K12.2: holds", "model rule: FIRES", "12.0 s a step"):
        assert k in text
    ev = J12.evaluate(*_job12(dprime=(0.440, 0.448, 0.450)))                    # -1.1 / -1.3 % against the dagger
    assert ev["K12.3"]["holds"] is False and ev["model_rule"]["fires"]
    ev = J12.evaluate(*_job12(dprime=(0.466, 0.440, 0.442)))                    # -2.5 % at 64
    assert ev["K12.1"]["holds"] is False and not ev["model_rule"]["fires"]
    ev = J12.evaluate(*_job12(infra=(1.6893, (2.23, 1.98, 3.3), 4.31)))         # the probes at 15 A move
    assert ev["K12.2"]["holds"] is False and not ev["model_rule"]["fires"]
    ev = J12.evaluate(*_job12(start11=18.95))                                   # not job 11's start
    assert ev["K12.0"]["holds"] is False and not ev["model_rule"]["fires"] and "does not fire" in J12.report(ev)
    ev = J12.evaluate(*_job12(inv=(1e-15, 1e-3)))                               # the response moved with the rest
    assert ev["K12.0"]["holds"] is False
    ev = J12.evaluate(*_job12(missing=(3,)))
    assert ev["K12.0"]["holds"] is None and ev["K12.1"]["holds"] is None and not ev["model_rule"]["read"]
