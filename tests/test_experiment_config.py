"""Configuration validation and routing, without expensive training runs."""
import argparse
import importlib.util
from pathlib import Path
import sys
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("experiment_config", SCRIPTS / "experiment_config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


def write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return str(path)


def test_defaults_and_cli_override(tmp_path):
    path = write(tmp_path, 'experiment:\n  type: water\n  part: ion\n  rungs: [M_1p]\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', default='dimer')
    parser.add_argument('--rungs', default='M_inf')
    args = config.parse_selection(parser, 'water', ['--config', path, '--part', 'far'], rungs=['M_1p'])
    assert (args.part, args.rungs) == ('far', 'M_1p')


@pytest.mark.parametrize('text', [
    'experiment: {type: unknown}',
    'experiment: {type: water, systems: [Ar2]}',
    'experiment: {type: water, part: wrong}',
    'experiment: {type: dimer_ladder, truth: wrong}',
    'experiment: {type: x2b_toy, orders: [1]}',
    'experiment: {type: x2b_toy, orders: [false]}',
    'experiment: {type: water, rungs: []}',
    'experiment: {type: water}\ntraining: {unknown: 1}',
])
def test_invalid_config(tmp_path, text):
    with pytest.raises(ValueError):
        config.load_selection(write(tmp_path, text))


def test_dispatch(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import run_experiment
    path = write(tmp_path, 'experiment: {type: x2b_toy, orders: [0, 2]}')
    monkeypatch.setattr(sys, 'argv', ['run_experiment.py', '--config', path, '--steps', '5'])
    calls = []
    monkeypatch.setattr(run_experiment.runpy, 'run_path', lambda target, **kw: calls.append((target, sys.argv[:], kw)))
    run_experiment.main()
    assert Path(calls[0][0]).name == 'x2b_toy_regression.py'
    assert calls[0][1][1:] == ['--config', path, '--steps', '5']


def test_existing_cli_defaults():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orders', default='0,2')
    assert config.parse_selection(parser, 'x2b_toy', []).orders == '0,2'


def test_wrong_script(tmp_path):
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        config.parse_selection(parser, 'water', ['--config', write(tmp_path, 'experiment: {type: x2b_toy}')])


@pytest.mark.parametrize('script, text, expected', [
    ('dimer_ladder', 'experiment: {type: dimer_ladder, systems: [Ne2, Ar2], truth: published, rungs: [M_6]}', {'system': 'Ne2,Ar2', 'truth': 'published', 'rungs': 'M_6'}),
    ('e3_water', 'experiment: {type: water, part: ion, rungs: [M_16S0p]}', {'part': 'ion', 'rungs': 'M_16S0p'}),
    ('x2b_toy_regression', 'experiment: {type: x2b_toy, orders: [2]}', {'orders': '2'}),
])
def test_script_selection_before_computation(tmp_path, monkeypatch, script, text, expected):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(script, SCRIPTS / (script + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, 'argv', [script, '--config', write(tmp_path, text)])

    class Selected(Exception):
        pass

    def capture(parser, kind, **kwargs):
        args = config.parse_selection(parser, kind, **kwargs)
        assert all(getattr(args, key) == value for key, value in expected.items())
        raise Selected

    monkeypatch.setattr(module, 'parse_selection', capture)
    with pytest.raises(Selected):
        module.main()
