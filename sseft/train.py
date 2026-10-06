"""
Training: energies and forces, targets scaled (never shifted), Adam with a cosine schedule and an optional
L-BFGS polish, float64, independent seeds for ensembles.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from .structure import pad_batch
from . import model as M


def energy_scale(structures):
    """std of the per-atom reference energies; the targets are divided by it, never shifted."""
    e = np.array([s.energy / s.n_atoms for s in structures if s.energy is not None])
    return float(np.std(e) + 1e-30) if len(e) > 1 else float(abs(e[0]) + 1e-30)


def make_loss_sums(rung, scale, w_force=10.0):
    """per-batch sums and counts of the squared scaled residuals: (sum_e, cnt_e, sum_f, cnt_f)."""
    def sums(params, batch):
        n = jnp.maximum(jnp.sum(batch["mask"], axis=1), 1.0)
        if w_force > 0:
            E, F = M.energy_forces(params, rung, batch)
            f_err = (F - batch["forces"]) / scale
            fm = batch["mask"][:, :, None] * batch["has_forces"][:, None, None]
            sf, cf = jnp.sum(f_err ** 2 * fm), jnp.sum(fm)
        else:
            E = jax.vmap(lambda bb: M.energy_single(params, rung, bb)[0])(batch)
            sf, cf = jnp.zeros(()), jnp.zeros(())
        e_err = (E - batch["energy"]) / n / scale
        return jnp.sum(e_err ** 2 * batch["has_energy"]), jnp.sum(batch["has_energy"]), sf, cf
    return sums


def make_loss(rung, scale, w_energy=1.0, w_force=10.0):
    sums = make_loss_sums(rung, scale, w_force)
    def loss_fn(params, batch):
        se, ce, sf, cf = sums(params, batch)
        le = se / jnp.maximum(ce, 1.0); lf = sf / jnp.maximum(cf, 1.0)
        return w_energy * le + w_force * lf, (le, lf)
    return loss_fn


def make_chunked_loss(rung, scale, data, w_energy=1.0, w_force=10.0, chunk=48):
    """the full-data loss evaluated chunk by chunk (exact: sums and counts are combined after the chunks), so
    that the full-batch gradient of L-BFGS does not hold the whole data set's forward graph in memory."""
    sums = make_loss_sums(rung, scale, w_force)
    n = int(data["mask"].shape[0])
    bounds = list(range(0, n, chunk)) + [n]
    chunks = [{k: v[a:b] for k, v in data.items()} for a, b in zip(bounds[:-1], bounds[1:])]
    def loss_fn(params):
        se = ce = sf = cf = 0.0
        for c in chunks:
            a, b, cc, d = sums(params, c)
            se, ce, sf, cf = se + a, ce + b, sf + cc, cf + d
        le = se / jnp.maximum(ce, 1.0); lf = sf / jnp.maximum(cf, 1.0)
        return w_energy * le + w_force * lf
    return loss_fn


def fit(rung, structures, seed=0, steps=2000, lr=3e-3, w_energy=1.0, w_force=10.0, batch_size=None,
        lbfgs_steps=0, verbose=False, scale=None, lbfgs_chunk=48):
    """train one model; returns (params, info)."""
    key = jax.random.PRNGKey(seed)
    scale = scale or energy_scale(structures)
    params = M.init_params(key, rung, energy_scale=scale)
    loss_fn = make_loss(rung, scale, w_energy, w_force)
    data = pad_batch(structures, n_freq=rung.K)
    n_data = len(structures)
    batch_size = batch_size or n_data
    sched = optax.cosine_decay_schedule(lr, steps, alpha=0.01)
    opt = optax.adam(sched)
    state = opt.init(params)

    @jax.jit
    def step(params, state, batch):
        (l, parts), g = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, l, parts

    rng = np.random.default_rng(seed)
    t0 = time.time(); hist = []
    for it in range(steps):
        idx = rng.choice(n_data, size=min(batch_size, n_data), replace=False) if batch_size < n_data else np.arange(n_data)
        batch = {k: v[idx] for k, v in data.items()}
        params, state, l, parts = step(params, state, batch)
        if it % max(steps // 10, 1) == 0 or it == steps - 1:
            hist.append((it, float(l), float(parts[0]), float(parts[1])))
            if verbose:
                print(f"    step {it:5d}  loss {float(l):.3e}  (E {float(parts[0]):.2e}, F {float(parts[1]):.2e})  [{time.time() - t0:.0f} s]", flush=True)
    if lbfgs_steps > 0:
        f = make_chunked_loss(rung, scale, data, w_energy, w_force, chunk=lbfgs_chunk)
        params = polish_lbfgs(params, f, lbfgs_steps, verbose)
    return params, {"scale": scale, "history": hist, "time": time.time() - t0}


def polish_lbfgs(params, f, steps, verbose=False):
    """f(params) -> scalar loss over the whole data set."""
    opt = optax.lbfgs()
    state = opt.init(params)
    vg = optax.value_and_grad_from_state(f)

    @jax.jit
    def step(params, state):
        value, grad = vg(params, state=state)
        upd, state = opt.update(grad, state, params, value=value, grad=grad, value_fn=f)
        return optax.apply_updates(params, upd), state, value
    for it in range(steps):
        params, state, value = step(params, state)
        if verbose and (it % max(steps // 5, 1) == 0 or it == steps - 1):
            print(f"    lbfgs {it:4d}  loss {float(value):.3e}", flush=True)
    return params


def fit_ensemble(rung, structures, seeds, **kw):
    return [fit(rung, structures, seed=s, **kw) for s in seeds]


def predict(params, rung, structures):
    """energies (hartree) and forces (hartree/bohr) per structure."""
    data = pad_batch(structures, n_freq=rung.K)
    E, F = jax.jit(lambda b: M.energy_forces(params, rung, b))(data)
    E = np.array(E); F = np.array(F)
    return E, [F[i, :s.n_atoms] for i, s in enumerate(structures)]


def predict_aux(params, rung, structures):
    data = pad_batch(structures, n_freq=rung.K)
    aux = jax.jit(lambda b: M.aux_batched(params, rung, b))(data)
    return {k: np.array(v) for k, v in aux.items()}
