"""E3-QP3's configuration and its v0.0.12 pieces without a build: the contact configurations, the rungs with their own band
edge, the held-out readings with the pins as trained, the readings by the closest contact, and the shipped
configs/water_qaqua_pol3.yaml."""
import importlib
import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
REPO = SCRIPTS.parent


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("experiment_config"), importlib.import_module("e3_water")


def parse_script(water, config, argv, monkeypatch):
    monkeypatch.setattr("sys.argv", ["e3_water.py", *argv])
    out = {}

    class Parsed(Exception):
        pass

    def capture(parser, kind, **kwargs):
        out["args"] = config.parse_selection(parser, kind, **kwargs)
        raise Parsed
    monkeypatch.setattr(water, "parse_selection", capture)
    with pytest.raises(Parsed):
        water.main()
    return out["args"]


def fake_build(tmp_path):
    root = tmp_path / "build_pol"
    (root / "q-AQUA").mkdir(parents=True)
    (root / "libqaqua_pol.so").write_bytes(b"")
    for name in ("2b_data.xyz", "3b_data.xyz", "4b_data.xyz"):
        (root / "q-AQUA" / name).write_text("")
    return str(root)


def test_shipped_config_resolves(modules, monkeypatch, tmp_path):
    config, water = modules
    path = str(REPO / "configs" / "water_qaqua_pol3.yaml")
    assert config.config_parts(config.load_config(path)) == ["clusters"]
    a = parse_script(water, config, ["--config", path, "--qaqua_root", fake_build(tmp_path)], monkeypatch)
    assert a.part == "clusters"
    assert a.rungs.split(",") == ["M_inf", "M_16pm", "M_16pm_e1", "M_16C2pm", "M_16C2pm_e1"]
    assert (a.w_energy, a.w_force, a.steps, a.lbfgs, a.batch, a.lbfgs_chunk) == (100.0, 1.0, 4000, 1000, 32, 64)
    assert (a.readout_stage, a.readout_orders, a.readout_solve_iterations, a.readout_ridge_selection) == ("far_first", "all", 4, "groups")
    assert (a.heldout_pin_charges, a.l1, a.response_edge, a.F, a.rcut) == ("rigid", 0.75, 0.75, 16, 6.0)
    settings = water.cluster_data_settings(a)
    assert (settings["contact_dimers"], settings["contact_trimers"], settings["contact_oo_angstrom"]) == (600, 200, [2.3, 3.0])
    assert (settings["contact_seed"], settings["contact_max_pair_kcal"], settings["contact_min_atom_distance_angstrom"]) == (700, 2.0, 1.5)
    # E3-QP2's data otherwise: its clusters, probes and test sets
    assert (settings["n_train"], settings["n_test"], settings["probe_train"], settings["probe_test"]) == (100, 24, 2, 10)
    assert settings["probe_train_core_sizes"] == [3, 4, 5]


def test_runs_without_contacts_keep_their_dataset_settings(modules):
    _, water = modules
    a = SimpleNamespace(**{k: v for k, v in water.CLUSTER_DEFAULTS.items()})
    assert not any(k.startswith("contact") for k in water.cluster_data_settings(a))
    a.contact_dimers = 2
    assert water.cluster_data_settings(a)["contact_oo_angstrom"] == [2.3, 3.0]


@pytest.mark.parametrize("n_mol", [2, 3])
def test_contact_configurations(n_mol):
    from sseft import water as W
    from sseft.units import bohr_to_ang
    lo, hi = 2.4, 3.0
    out = W.contact_configurations(n_mol, 5, (lo, hi), seed=3, max_pair_kcal=2.0, min_any=1.5)
    again = W.contact_configurations(n_mol, 5, (lo, hi), seed=3, max_pair_kcal=2.0, min_any=1.5)
    assert len(out) == 5
    for s, t in zip(out, again):
        np.testing.assert_array_equal(s.positions, t.positions)
        assert s.info["contact"] and s.n_atoms == 3 * n_mol and s.energy is not None and s.forces.shape == (3 * n_mol, 3)
        p = bohr_to_ang(s.positions)
        O = p[0::3]
        assert lo - 1e-9 <= np.linalg.norm(O[1] - O[0]) <= hi + 1e-9
        if n_mol == 3:
            near = [np.linalg.norm(O[2] - O[k]) for k in (0, 1)]
            assert min(near) <= hi + 1e-9 and max(near) >= lo - 1e-9 and min(near) >= lo - 1e-9
        mol = np.repeat(np.arange(n_mol), 3)
        d = np.linalg.norm(p[:, None] - p[None], axis=-1)[mol[:, None] != mol[None]]
        assert d.min() >= 1.5
        for i, j in itertools.combinations(range(n_mol), 2):
            pair = np.concatenate([p[3 * i:3 * i + 3], p[3 * j:3 * j + 3]])
            assert W.PW_TRUTH.energy(pair, 2) * 627.5094740631 <= 2.0 + 1e-9
        # rigid monomers: the training monomer's geometry
        for m in range(n_mol):
            r = np.linalg.norm(p[3 * m + 1:3 * m + 3] - p[3 * m], axis=1)
            np.testing.assert_allclose(r, W.R_OH, atol=1e-9)


def test_contact_configurations_refusals():
    from sseft import water as W
    with pytest.raises(ValueError, match="dimers"):
        W.contact_configurations(4, 1)
    with pytest.raises(ValueError, match="oo_range"):
        W.contact_configurations(2, 1, (3.0, 2.3))
    with pytest.raises(ValueError, match="count"):
        W.contact_configurations(2, -1)


@pytest.mark.parametrize("data,message", [
    ({"contact_oo_angstrom": [3.0, 2.3]}, "contact_oo_angstrom"),
    ({"contact_oo_angstrom": [2.3]}, "contact_oo_angstrom"),
    ({"contact_dimers": -1}, "contact_dimers"),
    ({"contact_trimers": 1.5}, "contact_trimers"),
    ({"contact_max_pair_kcal": -1.0}, "contact_max_pair_kcal"),
    ({"contact_min_atom_distance_angstrom": "far"}, "contact_min_atom_distance_angstrom"),
])
def test_invalid_contact_fields(modules, data, message):
    config, _ = modules
    with pytest.raises(ValueError, match=message):
        config.validate_cluster_data(data)


def test_contact_configurations_belong_to_the_clusters_part(modules, monkeypatch, capsys):
    config, water = modules
    with pytest.raises(SystemExit):
        parse_script(water, config, ["--part", "dimer", "--rungs", "M_1p", "--contact_dimers", "4"], monkeypatch)
    assert "clusters part" in capsys.readouterr().err


def test_heldout_pin_charges_field(modules):
    config, _ = modules
    assert config.validate_truth({"heldout": {"sets": ["2b"], "pin_charges": "rigid"}}, "water")["heldout_pin_charges"] == "rigid"
    with pytest.raises(ValueError, match="pin_charges"):
        config.validate_truth({"heldout": {"sets": ["2b"], "pin_charges": "own"}}, "water")


def test_rungs_with_their_own_band_edge(modules):
    _, water = modules
    from sseft.units import ang_to_bohr
    for response_edge in (0.75, 0.0):
        a = SimpleNamespace(lA=4.0, rcut=6.0, F=16, l1=0.75, rs_over_l1=0.0, response_edge=response_edge)
        base, own = water.configure(water.RUNGS["M_16C2pm"], a), water.configure(water.RUNGS["M_16C2pm_e1"], a)
        assert abs(own.l1 - ang_to_bohr(1.0)) < 1e-12 and abs(base.l1 - ang_to_bohr(0.75)) < 1e-12
        # the read-out at the run's band edge: the response the base rung matched
        assert abs(own.bf_l - ang_to_bohr(0.75)) < 1e-12 and own.bf_rs == base.bf_rs == 0.0
        assert own.bf_mutual == base.bf_mutual == -1 and own.msites and own.bf_species
        dagger = water.configure(water.RUNGS["M_16pm_e1"], a)
        assert abs(dagger.l1 - ang_to_bohr(1.0)) < 1e-12 and dagger.bf_l == 0.0
    assert water.model_record("M_16pm_e1", a)["band_edge_angstrom"] == 1.0
    assert water.model_record("M_16pm", a)["band_edge_angstrom"] == 0.75


class SiteTruth:
    """a truth whose charges follow each molecule's geometry (the O-H lengths), with a site model (H charges and an
    M-site), for the readings' pins."""
    name = "fake"

    @staticmethod
    def _scale(pos):
        p = np.asarray(pos, float)
        return np.array([np.linalg.norm(p[3 * m + 1] - p[3 * m]) + np.linalg.norm(p[3 * m + 2] - p[3 * m])
                         for m in range(len(p) // 3)])

    def molecule_charges(self, pos):
        s = self._scale(pos)
        return np.concatenate([[-0.6 * k, 0.3 * k, 0.3 * k] for k in s])

    def site_model(self, pos):
        s = self._scale(pos); n = len(s)
        q = np.concatenate([[0.0, 0.5 * k, 0.5 * k] for k in s])
        w = np.zeros((n, 3 * n))
        for m in range(n):
            w[m, 3 * m:3 * m + 3] = [0.6, 0.2, 0.2]
        return q, w, -s.copy(), np.arange(n)


def test_readings_with_the_pins_as_trained(modules, monkeypatch):
    """model_nbody(rigid=True): every molecule carries the rigid monomer's charges (q3 and the site model's), at the
    frame's own sites; the default keeps the charges at the frame's geometry (v0.0.11)."""
    _, water = modules
    from sseft import water as W
    truth = SiteTruth()
    seen = []
    monkeypatch.setattr(water.T, "predict_energies", lambda params, rung, structs: seen.extend(structs) or np.zeros(len(structs)))
    stretched = W.monomer_positions().copy(); stretched[1] *= 1.02                  # a near-rigid, distorted monomer
    frame = np.concatenate([stretched, W.monomer_positions() + [3.0, 0.0, 0.0]])
    q_frame = truth.molecule_charges(frame)
    for name in ("M_16pm", "M_16p"):
        rung = water.RUNGS[name]
        seen.clear()
        water.model_nbody(None, rung, frame[None], q_frame[None], 2, truth=truth, rigid=True)
        (s,) = seen
        rigid_q = truth.molecule_charges(W.monomer_positions())
        np.testing.assert_allclose(s.pinned["q"], np.tile(rigid_q, 2))
        if rung.msites:
            q_r, _, vq_r, _ = truth.site_model(W.monomer_positions())
            np.testing.assert_allclose(s.pinned["q_site"], np.tile(q_r, 2))
            np.testing.assert_allclose(s.pinned["vsite_q"], np.full(2, vq_r[0]))
            np.testing.assert_allclose(s.pinned["vsite_w"], truth.site_model(frame)[1])      # the frame's own sites
        seen.clear()
        water.model_nbody(None, rung, frame[None], q_frame[None], 2, truth=truth)
        (s,) = seen
        np.testing.assert_allclose(s.pinned["q"], q_frame)
        if rung.msites:
            np.testing.assert_allclose(s.pinned["vsite_q"], truth.site_model(frame)[2])
    with pytest.raises(ValueError, match="truth"):
        water.model_nbody(None, water.RUNGS["M_16pm"], frame[None], q_frame[None], 2, truth=None, rigid=True)


def test_readings_by_the_closest_contact(modules):
    _, water = modules
    from sseft import qaqua as QA
    from sseft import water as W
    frames = np.array([np.concatenate([W.monomer_positions(), W.monomer_positions() + [d, 0.0, 0.0],
                                       W.monomer_positions() + [0.0, 6.0, 0.0]]) for d in (2.5, 2.8, 3.5)])
    np.testing.assert_allclose(QA.smallest_oo(frames), [2.5, 2.8, 3.5])
    np.testing.assert_allclose(QA.largest_oo(frames), [np.hypot(d, 6.0) for d in (2.5, 2.8, 3.5)])
    d = dict(label=np.array([1.0, 2.0, 3.0]), pes=np.array([1.0, 2.0, 3.0]), strict=np.ones(3, bool),
             roo=QA.largest_oo(frames), soo=QA.smallest_oo(frames))
    m = water.heldout_metrics(np.array([1.5, 2.0, 3.0]), d)
    near = m["by_smallest_oo_angstrom"]
    assert [near[b]["count"] for b in ("0-2.7", "2.7-3", "3-inf")] == [1, 1, 1]
    assert near["0-2.7"]["rmse_model"] == pytest.approx(0.5) and near["2.7-3"]["rmse_model"] == 0.0
    assert "by_smallest_oo_angstrom" not in water.heldout_metrics(np.zeros(3), {k: v for k, v in d.items() if k != "soo"})
