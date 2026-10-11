"""Small differentiable objectives verify the optimizer stage boundaries."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sseft import linear, train


@pytest.fixture
def tiny_fit(monkeypatch):
    """Use real Adam with two parameter groups and a known quadratic loss."""
    initial = {"gnn": {"weight": jnp.array(0.0)}, "readout": {"weight": jnp.array(0.0)}}
    monkeypatch.setattr(train.M, "init_params", lambda *args, **kwargs: initial)
    monkeypatch.setattr(train, "pad_batch", lambda *args, **kwargs: {"x": jnp.ones((2, 1))})

    def loss(params, batch):
        gnn_loss = (params["gnn"]["weight"] - 3.0) ** 2
        readout_loss = (params["readout"]["weight"] - 4.0) ** 2
        return gnn_loss + readout_loss, (gnn_loss, readout_loss)

    monkeypatch.setattr(train, "make_loss", lambda *args, **kwargs: loss)
    monkeypatch.setattr(train, "make_chunked_loss", lambda *args, **kwargs: lambda params: loss(params, None)[0])
    return SimpleNamespace(K=1), [object(), object()], initial


@pytest.mark.parametrize("readout_adam", [False, True])
def test_adam_freezes_only_readout(tiny_fit, monkeypatch, readout_adam):
    rung, structures, initial = tiny_fit
    boundary = []

    def polish(params, *args, **kwargs):
        boundary.append(params)
        return params

    monkeypatch.setattr(train, "polish_lbfgs", polish)
    params, info = train.fit(
        rung, structures, steps=5, lr=0.1, scale=1.0, lbfgs_steps=1,
        readout_adam=readout_adam, readout_linear_solve=False,
    )
    assert len(info["history"]) == 5
    assert len(boundary) == 1
    assert float(params["gnn"]["weight"]) > 0.0
    if readout_adam:
        assert float(params["readout"]["weight"]) > 0.0
    else:
        np.testing.assert_array_equal(params["readout"]["weight"], initial["readout"]["weight"])


def test_linear_solve_is_between_frozen_adam_and_polish(tiny_fit, monkeypatch):
    rung, structures, _ = tiny_fit
    events = []

    def solve(params, rung, data, scale, **kwargs):
        events.append(("solve", params, kwargs))
        assert float(params["gnn"]["weight"]) > 0.0
        assert float(params["readout"]["weight"]) == 0.0
        return {**params, "readout": {"weight": jnp.array(2.0)}}, (16.0, 4.0)

    def polish(params, *args, **kwargs):
        events.append(("polish", params, kwargs))
        assert float(params["readout"]["weight"]) == 2.0
        return params

    monkeypatch.setattr(linear, "solve_readout", solve)
    monkeypatch.setattr(train, "polish_lbfgs", polish)
    params, info = train.fit(
        rung, structures, steps=3, lr=0.1, scale=1.0, lbfgs_steps=2,
        w_energy=2.5, w_force=0.5, readout_ridge=2e-6, lbfgs_chunk=1,
        lbfgs_lr=0.4, lbfgs_memory=5, lbfgs_scale_init_precond=False,
        lbfgs_max_linesearch_steps=7, lbfgs_readout_only=True,
    )
    assert [event[0] for event in events] == ["solve", "polish"]
    assert events[0][2] == {"w_energy": 2.5, "w_force": 0.5, "ridge": 2e-6, "chunk": 1}
    assert events[1][2] == {
        "learning_rate": 0.4, "memory_size": 5, "scale_init_precond": False,
        "max_linesearch_steps": 7, "readout_only": True,
    }
    assert info["readout_solve"] == (16.0, 4.0)
    np.testing.assert_array_equal(params["gnn"]["weight"], events[0][1]["gnn"]["weight"])


def test_zero_adam_steps_runs_subsequent_optimizer(tiny_fit, monkeypatch):
    rung, structures, initial = tiny_fit
    calls = []
    monkeypatch.setattr(train, "polish_lbfgs", lambda params, *args, **kwargs: calls.append(params) or params)
    params, info = train.fit(
        rung, structures, steps=0, scale=1.0, lbfgs_steps=1,
        readout_linear_solve=False,
    )
    assert info["history"] == []
    assert len(calls) == 1
    for leaf, expected in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(initial)):
        np.testing.assert_array_equal(leaf, expected)


def test_adam_uses_configured_schedule_moments_epsilon_and_clipping(tiny_fit):
    rung, structures, _ = tiny_fit
    steps, lr, final_ratio = 4, 0.08, 0.35
    beta1, beta2, epsilon, epsilon_root, clip_norm = 0.7, 0.8, 0.2, 0.3, 2.0
    params, _ = train.fit(
        rung, structures, steps=steps, lr=lr, scale=1.0, readout_adam=True,
        readout_linear_solve=False, lr_final_ratio=final_ratio,
        adam_b1=beta1, adam_b2=beta2, adam_eps=epsilon, adam_eps_root=epsilon_root,
        gradient_clip_norm=clip_norm,
    )
    # Independently evaluate the Adam equations for the known two-variable loss.
    expected = np.zeros(2)
    first_moment = np.zeros(2)
    second_moment = np.zeros(2)
    for step in range(steps):
        gradient = 2 * (expected - np.array([3.0, 4.0]))
        gradient *= min(1.0, clip_norm / np.linalg.norm(gradient))
        first_moment = beta1 * first_moment + (1 - beta1) * gradient
        second_moment = beta2 * second_moment + (1 - beta2) * gradient ** 2
        corrected_first = first_moment / (1 - beta1 ** (step + 1))
        corrected_second = second_moment / (1 - beta2 ** (step + 1))
        scheduled_lr = lr * (final_ratio + (1 - final_ratio) * (1 + np.cos(np.pi * step / steps)) / 2)
        expected -= scheduled_lr * corrected_first / (np.sqrt(corrected_second + epsilon_root) + epsilon)
    np.testing.assert_allclose(
        [params["gnn"]["weight"], params["readout"]["weight"]], expected, rtol=1e-12, atol=1e-12,
    )


@pytest.mark.parametrize("readout_only", [False, True])
def test_lbfgs_response_only_preserves_gnn_exactly(readout_only):
    initial = {"gnn": {"weight": jnp.array([1.0, -2.0])}, "readout": {"weight": jnp.array(0.0)}}
    objective = lambda params: jnp.sum((params["gnn"]["weight"] - jnp.array([5.0, 6.0])) ** 2) + (params["readout"]["weight"] - 3.0) ** 2
    params = train.polish_lbfgs(initial, objective, steps=6, readout_only=readout_only)
    np.testing.assert_allclose(params["readout"]["weight"], 3.0, atol=1e-8)
    if readout_only:
        np.testing.assert_array_equal(params["gnn"]["weight"], initial["gnn"]["weight"])
    else:
        np.testing.assert_allclose(params["gnn"]["weight"], [5.0, 6.0], atol=1e-8)
    assert float(objective(params)) < float(objective(initial))


def test_configurable_linesearch_preserves_optax_lbfgs_initial_guess(monkeypatch):
    original = train.optax.scale_by_zoom_linesearch
    choices = []

    def linesearch(**kwargs):
        choices.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(train.optax, "scale_by_zoom_linesearch", linesearch)
    params = train.polish_lbfgs(
        {"weight": jnp.array(0.0)}, lambda p: (p["weight"] - 3.0) ** 2,
        steps=2, max_linesearch_steps=7,
    )
    assert choices == [{"max_linesearch_steps": 7, "initial_guess_strategy": "one"}]
    np.testing.assert_allclose(params["weight"], 3.0, atol=1e-8)


def _linear_problem(monkeypatch):
    """One active output coefficient fits conflicting energy and force targets."""
    params = {
        "gnn": {"weight": jnp.array(7.0)},
        "readout": {
            "theta": [{"w": jnp.zeros((1, 2)), "b": jnp.zeros(2)}],
            "iso": [{"w": jnp.zeros((1, 1)), "b": jnp.zeros(1)}],
        },
    }
    data = {
        "mask": jnp.ones((2, 1)), "weight": jnp.array([1.0, 2.0]),
        "has_energy": jnp.ones(2), "has_forces": jnp.ones(2),
        "energy": jnp.array([2.0, 4.0]),
        "forces": jnp.array([[[8.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]]]),
    }

    def energy_forces(params, rung, batch):
        coefficient = params["readout"]["theta"][-1]["b"][0]
        n = batch["mask"].shape[0]
        energies = jnp.full((n,), coefficient)
        forces = jnp.zeros((n, 1, 3)).at[:, 0, 0].set(coefficient)
        return energies, forces

    monkeypatch.setattr(linear.M, "energy_forces", energy_forces)
    return params, SimpleNamespace(e0={"n_rbf": 1}), data


@pytest.mark.parametrize("w_energy,w_force", [(1.0, 1.0), (4.0, 0.5), (0.0, 2.0), (2.0, 0.0)])
def test_linear_solve_minimizes_configured_weighted_objective(monkeypatch, w_energy, w_force):
    params, rung, data = _linear_problem(monkeypatch)
    scale = 2.0
    expected = (w_energy * (2.0 + 2 * 4.0) / 3 + w_force * (8.0 + 2 * 10.0) / 3) / (w_energy + w_force)
    solved, reported = linear.solve_readout(
        params, rung, data, scale, w_energy=w_energy, w_force=w_force, ridge=1e-10, chunk=1,
    )
    np.testing.assert_allclose(solved["readout"]["theta"][-1]["b"][0], expected, rtol=1e-8)
    objective = train.make_loss(rung, scale, w_energy=w_energy, w_force=w_force)
    # The zero-force case uses energy_single rather than energy_forces.
    monkeypatch.setattr(train.M, "energy_single", lambda p, r, b: (p["readout"]["theta"][-1]["b"][0], {}))
    actual = (float(objective(params, data)[0]), float(objective(solved, data)[0]))
    np.testing.assert_allclose(reported, actual, rtol=1e-12, atol=1e-12)
    assert reported[1] < reported[0]
    np.testing.assert_array_equal(solved["gnn"]["weight"], params["gnn"]["weight"])
