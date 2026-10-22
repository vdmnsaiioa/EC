"""E3-QP4's configurations and its v0.0.13 pieces without a build: the contact configurations' energy weight (in the loss
and in the read-out's solve), the rung with the read-out's own band edge, and the shipped configs/water_qaqua_pol4a.yaml
and pol4b.yaml (E3-QP3 at E_0's width 32; job B the same with the contacts' energies weighted down)."""
import copy
import importlib
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


@pytest.mark.parametrize("job, weight", [("4a", 1.0), ("4b", 0.125)])
def test_shipped_configs_resolve(modules, monkeypatch, tmp_path, job, weight):
    config, water = modules
    path = str(REPO / "configs" / f"water_qaqua_pol{job}.yaml")
    assert config.config_parts(config.load_config(path)) == ["clusters"]
    a = parse_script(water, config, ["--config", path, "--qaqua_root", fake_build(tmp_path)], monkeypatch)
    assert a.part == "clusters"
    assert a.rungs.split(",") == ["M_inf", "M_16pm", "M_16C2pm", "M_16C2pm_r1"]
    assert (a.F, a.rcut, a.l1, a.response_edge, a.heldout_pin_charges) == (32, 6.0, 0.75, 0.75, "rigid")
    assert a.w_contact_energy == weight
    assert (a.w_energy, a.w_force, a.steps, a.lbfgs, a.batch, a.lbfgs_chunk) == (100.0, 1.0, 4000, 1000, 32, 64)
    assert (a.readout_stage, a.readout_orders, a.readout_solve_iterations, a.readout_ridge_selection) == ("far_first", "all", 4, "groups")
    settings = water.cluster_data_settings(a)
    assert (settings["contact_dimers"], settings["contact_trimers"], settings["contact_oo_angstrom"]) == (600, 200, [2.3, 3.0])
    assert (settings["n_train"], settings["n_test"], settings["probe_train"], settings["probe_test"]) == (100, 24, 2, 10)


def test_each_job_is_one_change(modules):
    """job A is E3-QP3 at E_0's width 32 (its rungs aside); job B is job A with the contacts' energies weighted down."""
    config, _ = modules
    load = lambda name: config.load_config(str(REPO / "configs" / name))
    qp3, a, b = load("water_qaqua_pol3.yaml"), load("water_qaqua_pol4a.yaml"), load("water_qaqua_pol4b.yaml")
    a_as_qp3 = copy.deepcopy(a)
    a_as_qp3["model"]["features"] = 16
    a_as_qp3["parts"]["clusters"]["rungs"] = qp3["parts"]["clusters"]["rungs"]
    assert a_as_qp3 == qp3
    b_as_a = copy.deepcopy(b)
    assert b_as_a["training"]["loss"].pop("contact_energy_weight") == 0.125
    assert b_as_a == a


def test_rung_with_the_readouts_own_band_edge(modules):
    _, water = modules
    from sseft.units import ang_to_bohr
    for response_edge in (0.75, 0.0):
        a = SimpleNamespace(lA=4.0, rcut=6.0, F=32, l1=0.75, rs_over_l1=0.0, response_edge=response_edge)
        base, own = water.configure(water.RUNGS["M_16C2pm"], a), water.configure(water.RUNGS["M_16C2pm_r1"], a)
        assert abs(own.bf_l - ang_to_bohr(1.0)) < 1e-12                    # its own, over the run's response edge
        assert abs(own.l1 - ang_to_bohr(0.75)) < 1e-12 and own.l1 == base.l1  # the pinned bands at the run's
        assert own.bf_rs == base.bf_rs == 0.0
        assert own.bf_mutual == base.bf_mutual == -1 and own.msites and own.bf_species and own.e0["F"] == 32
    record = water.model_record("M_16C2pm_r1", a)
    assert (record["response_edge_angstrom"], record["band_edge_angstrom"], record["features"]) == (1.0, 0.75, 32)
    assert "response_edge_angstrom" not in water.model_record("M_16pm", a)


def test_contact_energy_weight_field(modules):
    config, _ = modules
    assert config.validate_training({"loss": {"contact_energy_weight": 0.125}}) == {"w_contact_energy": 0.125}
    for bad in (0, -1.0, float("inf"), "x", None):
        with pytest.raises(ValueError, match="contact_energy_weight"):
            config.validate_training({"loss": {"contact_energy_weight": bad}})


def test_contact_energy_weight_needs_contacts(modules, monkeypatch, tmp_path):
    config, water = modules
    path = str(REPO / "configs" / "water_qaqua_pol4b.yaml")
    a = parse_script(water, config, ["--config", path, "--qaqua_root", fake_build(tmp_path)], monkeypatch)
    water.validate_water_args(a)
    a.contact_dimers = a.contact_trimers = 0
    with pytest.raises(ValueError, match="contact_energy_weight"):
        water.validate_water_args(a)
    a.w_contact_energy = 1.0
    water.validate_water_args(a)


def test_weight_contact_energies(modules, capsys):
    _, water = modules
    from sseft import water as W
    clusters = W.cluster_dataset((3,), 2, seed=0, mc_steps=2)
    contacts = W.contact_configurations(2, 2, seed=1) + W.contact_configurations(3, 1, seed=2)
    monomers = [W.monomer_structure()]
    structs = clusters + monomers + contacts
    assert water.weight_contact_energies(structs, 0.125) == contacts
    assert [s.energy_weight for s in structs] == [1.0] * 3 + [0.125] * 3
    assert "share of the energy term" in capsys.readouterr().out
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="contact_energy_weight"):
            water.weight_contact_energies(structs, bad)


def test_energy_weight_in_the_loss_and_the_solve():
    """the energy term is a weighted mean over structures with weight x energy_weight; the forces keep the weight alone;
    the read-out's linear solve counts its energy rows the same way."""
    import jax
    from sseft import model as M, structure as S, train as T, water as W
    rung = M.with_rung(M.RUNGS["M_inf"], e0={**M.RUNGS["M_inf"].e0, "F": 4})
    structs = W.cluster_dataset((2,), 3, seed=3, mc_steps=2)
    params = M.init_params(jax.random.PRNGKey(0), rung)
    sums = T.make_loss_sums(rung, 1.0, w_force=1.0)
    plain = [float(x) for x in sums(params, S.pad_batch(structs))]
    for s, w in zip(structs, (1.0, 0.25, 3.0)):
        s.energy_weight = w
    data = S.pad_batch(structs)
    assert np.allclose(np.array(data["energy_weight"]), [1.0, 0.25, 3.0])
    weighted = [float(x) for x in sums(params, data)]
    E, _ = M.energy_forces(params, rung, data)
    n = np.array(data["mask"]).sum(1)
    e2 = ((np.array(E) - np.array(data["energy"])) / n) ** 2
    assert np.isclose(weighted[0], np.dot([1.0, 0.25, 3.0], e2)) and np.isclose(weighted[1], 4.25)
    assert np.isclose(plain[0], e2.sum()) and np.isclose(plain[1], 3.0)
    assert np.isclose(weighted[2], plain[2]) and np.isclose(weighted[3], plain[3])          # the forces' sums unchanged
    # structures pickled before v0.0.13 carry no energy_weight attribute of their own: weight 1
    old = W.cluster_dataset((2,), 1, seed=4, mc_steps=2)[0]
    old.__dict__.pop("energy_weight", None)
    assert float(S.pad_batch([old])["energy_weight"][0]) == 1.0
    # the read-out's solve weights its energy rows by the same product
    import inspect
    from sseft import linear as L
    assert 'data.get("energy_weight", 1.0)' in inspect.getsource(L.solve_readout)


def test_run_clusters_weights_the_contacts_energies_only(modules, monkeypatch, tmp_path):
    _, water = modules
    fits = []

    def structure(n, k=0, contact=False):
        pos = np.array([[2.9 * m, atom * 0.5, float(k)] for m in range(n) for atom in range(3)])
        return SimpleNamespace(positions=water.ang_to_bohr(pos), n_atoms=3 * n, energy=-0.001 * n, forces=None,
                               info={"n": n, **({"contact": True} if contact else {})}, pinned={})

    monkeypatch.setattr(water.W, "cluster_dataset", lambda sizes, n, **kw: [structure(m, k) for m in sizes for k in range(n)])
    monkeypatch.setattr(water.W, "monomer_structure", lambda truth=None: structure(1))
    monkeypatch.setattr(water, "contact_data", lambda settings, truth: [structure(2, k, True) for k in range(3)] + [structure(3, 0, True)])
    monkeypatch.setattr(water.W, "water_structure", lambda positions, **kw: SimpleNamespace(
        positions=water.ang_to_bohr(positions), n_atoms=6, energy=0.0, info={"pair": True}, pinned={}))
    monkeypatch.setattr(water.W, "pair_additive_energy", lambda positions, n, truth=None: 0.0)
    monkeypatch.setattr(water.T, "fit", lambda rung, structs, **kw: fits.append(list(structs)) or ({}, {"time": 0.0}))
    monkeypatch.setattr(water.T, "predict_energies", lambda params, rung, structs, chunk=256: np.zeros(len(structs)))
    monkeypatch.setattr(water, "configure", lambda rung, args: SimpleNamespace(pin_mu=False, pin_alpha=False, msites=False,
                                                                                band_fields=-1, bf_mutual=0))
    monkeypatch.setattr(water, "_save", lambda *args: None)
    a = SimpleNamespace(**{**water.CLUSTER_DEFAULTS, "train_sizes": "3", "test_sizes": "2", "n_train": 2, "n_test": 1,
                           "mc_steps": 1, "monomer_references": 1, "cache": "", "rungs": "M_inf", "seeds": 1, "steps": 1,
                           "lbfgs": 0, "batch": 2, "lbfgs_chunk": 2, "out": str(tmp_path / "out.json"),
                           "contact_dimers": 3, "contact_trimers": 1, "w_contact_energy": 0.125})
    water.run_clusters(a)
    structs = fits[0]
    weights = [getattr(s, "energy_weight", 1.0) for s in structs]
    assert [w for s, w in zip(structs, weights) if s.info.get("contact")] == [0.125] * 4
    assert all(w == 1.0 for s, w in zip(structs, weights) if not s.info.get("contact"))
