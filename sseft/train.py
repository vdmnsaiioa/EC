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
        wgt = batch.get("weight", jnp.ones_like(batch["energy"]))          # per-structure weights (default 1)
        if w_force > 0:
            E, F = M.energy_forces(params, rung, batch)
            f_err = (F - batch["forces"]) / scale
            fm = batch["mask"][:, :, None] * (batch["has_forces"] * wgt)[:, None, None]
            sf, cf = jnp.sum(f_err ** 2 * fm), jnp.sum(fm)
        else:
            E = jax.vmap(lambda bb: M.energy_single(params, rung, bb)[0])(batch)
            sf, cf = jnp.zeros(()), jnp.zeros(())
        e_err = (E - batch["energy"]) / n / scale
        we = batch["has_energy"] * wgt
        return jnp.sum(e_err ** 2 * we), jnp.sum(we), sf, cf
    return sums


def make_loss(rung, scale, w_energy=1.0, w_force=10.0):
    sums = make_loss_sums(rung, scale, w_force)
    def loss_fn(params, batch):
        se, ce, sf, cf = sums(params, batch)
        le = se / jnp.maximum(ce, 1.0); lf = sf / jnp.maximum(cf, 1.0)
        return w_energy * le + w_force * lf, (le, lf)
    return loss_fn


def make_chunked_loss(rung, scale, data, w_energy=1.0, w_force=10.0, chunk=48):
    """the full-data loss evaluated chunk by chunk (exact: sums and counts are combined after the chunks) with
    a scan over rematerialised chunks, so that the full-batch gradient of L-BFGS holds the forward graph of one
    chunk only.  The data are padded to a multiple of the chunk size with inert structures (no energy, no
    forces, no atoms), which contribute nothing to the sums."""
    sums = make_loss_sums(rung, scale, w_force)
    n = int(data["mask"].shape[0])
    n_chunks = -(-n // chunk)
    pad = n_chunks * chunk - n
    def padded(v):
        if pad == 0: return v
        tail = jnp.zeros((pad,) + v.shape[1:], dtype=v.dtype)
        if v.dtype == jnp.bool_: tail = jnp.zeros((pad,) + v.shape[1:], dtype=bool)
        return jnp.concatenate([v, tail], axis=0)
    stacked = {k: padded(v).reshape((n_chunks, chunk) + v.shape[1:]) for k, v in data.items()}
    # padded structures: cell must stay invertible, and positions / numbers are zero -> masked everywhere
    if pad > 0:
        stacked["cell"] = stacked["cell"].at[-1, chunk - pad:].set(jnp.eye(3) * 1e6)
    chunk_sums = jax.checkpoint(lambda params, c: jnp.stack(sums(params, c)))
    def loss_fn(params):
        out = jax.lax.map(lambda c: chunk_sums(params, c), stacked)          # (n_chunks, 4)
        se, ce, sf, cf = jnp.sum(out, axis=0)
        le = se / jnp.maximum(ce, 1.0); lf = sf / jnp.maximum(cf, 1.0)
        return w_energy * le + w_force * lf
    return loss_fn


def _integer_option(name, value, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite_option(name, value, minimum=0.0, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    if not np.isfinite(value) or value < minimum or (positive and value == minimum):
        sign = ">" if positive else ">="
        raise ValueError(f"{name} must be finite and {sign} {minimum}")


def fit(rung, structures, seed=0, steps=2000, lr=3e-3, w_energy=1.0, w_force=10.0, batch_size=None,
        lbfgs_steps=0, verbose=False, scale=None, lbfgs_chunk=48, readout_adam=False,
        lr_final_ratio=0.01, adam_b1=0.9, adam_b2=0.999, adam_eps=1e-8, adam_eps_root=0.0,
        gradient_clip_norm=None, readout_linear_solve=None, readout_ridge=1e-8, lbfgs_lr=None,
        lbfgs_memory=10, lbfgs_scale_init_precond=True, lbfgs_max_linesearch_steps=20,
        lbfgs_readout_only=False):
    """train one model; returns (params, info).  The band-field read-out (when the rung has one) is held at its
    zero initialisation during the Adam stage unless readout_adam=True, and switched on in the L-BFGS polish: a
    quadratic read-out in the band field spans four orders of magnitude in |E|^2 across a training set with near and
    far sources, so that Adam's fixed-size steps on its coefficients either do nothing in the far field (coefficients
    in physical units) or wreck the near configurations (coefficients O(1)); L-BFGS, being curvature-aware, handles
    the stiff directions (e3r-bandfield-response-preregistration.md section 8).  The polish is therefore mandatory
    when the read-out is frozen during Adam.  Set lbfgs_readout_only=True to preserve the Adam-trained model
    outside the read-out during that polish.  readout_linear_solve=None follows the frozen-read-out default;
    True or False explicitly enables or disables the solve on rungs with a read-out.  steps=0 runs only the
    subsequent stages."""
    _integer_option("seed", seed)
    _integer_option("steps", steps)
    _integer_option("lbfgs_steps", lbfgs_steps)
    _integer_option("lbfgs_chunk", lbfgs_chunk, 1)
    _integer_option("lbfgs_memory", lbfgs_memory, 1)
    _integer_option("lbfgs_max_linesearch_steps", lbfgs_max_linesearch_steps, 1)
    if batch_size is not None:
        _integer_option("batch_size", batch_size, 1)
    if steps + lbfgs_steps == 0:
        raise ValueError("at least one Adam or L-BFGS optimization step is required")
    for name, value in (("lr", lr), ("adam_eps", adam_eps), ("adam_eps_root", adam_eps_root),
                        ("w_energy", w_energy), ("w_force", w_force), ("readout_ridge", readout_ridge)):
        _finite_option(name, value, positive=name == "lr")
    for name, value in (("adam_b1", adam_b1), ("adam_b2", adam_b2)):
        _finite_option(name, value)
        if value >= 1:
            raise ValueError(f"{name} must be < 1")
    _finite_option("lr_final_ratio", lr_final_ratio)
    if lr_final_ratio > 1:
        raise ValueError("lr_final_ratio must be <= 1")
    if adam_eps + adam_eps_root == 0:
        raise ValueError("adam_eps and adam_eps_root cannot both be zero")
    if w_energy + w_force == 0:
        raise ValueError("w_energy and w_force cannot both be zero")
    for name, value in (("scale", scale), ("gradient_clip_norm", gradient_clip_norm), ("lbfgs_lr", lbfgs_lr)):
        if value is not None:
            _finite_option(name, value, positive=True)
    for name, value in (("readout_adam", readout_adam), ("lbfgs_scale_init_precond", lbfgs_scale_init_precond),
                        ("lbfgs_readout_only", lbfgs_readout_only)):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean")
    if readout_linear_solve is not None and not isinstance(readout_linear_solve, (bool, np.bool_)):
        raise ValueError("readout_linear_solve must be a boolean or None")
    if len(structures) == 0:
        raise ValueError("training structures must not be empty")
    key = jax.random.PRNGKey(seed)
    scale = energy_scale(structures) if scale is None else scale
    params = M.init_params(key, rung, energy_scale=scale)
    has_readout = "readout" in params
    freeze_readout = has_readout and not readout_adam
    linear_solve = has_readout and (freeze_readout if readout_linear_solve is None else readout_linear_solve)
    if lbfgs_readout_only and (not has_readout or lbfgs_steps == 0):
        raise ValueError("lbfgs_readout_only requires a band-field read-out and lbfgs_steps > 0")
    if freeze_readout and lbfgs_steps == 0:
        raise ValueError("a band-field rung needs the L-BFGS polish (lbfgs_steps > 0): the read-out is frozen during Adam")
    loss_fn = make_loss(rung, scale, w_energy, w_force)
    data = pad_batch(structures, n_freq=rung.K)
    n_data = len(structures)
    batch_size = n_data if batch_size is None else batch_size
    sched = optax.cosine_decay_schedule(lr, max(steps, 1), alpha=lr_final_ratio)
    opt = optax.adam(sched, b1=adam_b1, b2=adam_b2, eps=adam_eps, eps_root=adam_eps_root)
    if gradient_clip_norm is not None:
        opt = optax.chain(optax.clip_by_global_norm(gradient_clip_norm), opt)
    state = opt.init(params)

    @jax.jit
    def step(params, state, batch):
        (l, parts), g = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        if freeze_readout:
            g = {**g, "readout": jax.tree_util.tree_map(jnp.zeros_like, g["readout"])}
        upd, state = opt.update(g, state, params)
        if freeze_readout:
            upd = {**upd, "readout": jax.tree_util.tree_map(jnp.zeros_like, upd["readout"])}
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
    solve = None
    if linear_solve:
        # the read-out's output layers are linear in the energies and forces: set them to the weighted least-squares
        # optimum of the objective given the Adam-trained rest of the model (sseft.linear), then polish everything
        from .linear import solve_readout
        params, solve = solve_readout(params, rung, data, scale, w_force=w_force, ridge=readout_ridge,
                                     chunk=lbfgs_chunk, w_energy=w_energy)
        print(f"    read-out linear solve: objective {solve[0]:.3e} -> {solve[1]:.3e}  [{time.time() - t0:.0f} s]", flush=True)
    if lbfgs_steps > 0:
        f = make_chunked_loss(rung, scale, data, w_energy, w_force, chunk=lbfgs_chunk)
        params = polish_lbfgs(params, f, lbfgs_steps, verbose, learning_rate=lbfgs_lr,
                              memory_size=lbfgs_memory, scale_init_precond=lbfgs_scale_init_precond,
                              max_linesearch_steps=lbfgs_max_linesearch_steps, readout_only=lbfgs_readout_only)
    options = dict(seed=int(seed), steps=int(steps), lr=float(lr), lr_final_ratio=float(lr_final_ratio),
                   adam_b1=float(adam_b1), adam_b2=float(adam_b2), adam_eps=float(adam_eps),
                   adam_eps_root=float(adam_eps_root),
                   gradient_clip_norm=None if gradient_clip_norm is None else float(gradient_clip_norm),
                   w_energy=float(w_energy), w_force=float(w_force), batch_size=int(min(batch_size, n_data)),
                   scale=float(scale), readout_adam=bool(readout_adam), freeze_readout=bool(freeze_readout),
                   readout_linear_solve=bool(linear_solve), readout_ridge=float(readout_ridge),
                   lbfgs_steps=int(lbfgs_steps), lbfgs_chunk=int(lbfgs_chunk),
                   lbfgs_lr=None if lbfgs_lr is None else float(lbfgs_lr),
                   lbfgs_memory=int(lbfgs_memory), lbfgs_scale_init_precond=bool(lbfgs_scale_init_precond),
                   lbfgs_max_linesearch_steps=int(lbfgs_max_linesearch_steps), lbfgs_readout_only=bool(lbfgs_readout_only),
                   verbose=bool(verbose))
    return params, {"scale": scale, "history": hist, "time": time.time() - t0, "readout_solve": solve,
                    "training_options": options}


def polish_lbfgs(params, f, steps, verbose=False, learning_rate=None, memory_size=10,
                 scale_init_precond=True, max_linesearch_steps=20, readout_only=False):
    """f(params) -> scalar loss over the whole data set.  With readout_only, preserve all other parameters."""
    _integer_option("steps", steps)
    _integer_option("memory_size", memory_size, 1)
    _integer_option("max_linesearch_steps", max_linesearch_steps, 1)
    for name, value in (("scale_init_precond", scale_init_precond), ("readout_only", readout_only)):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean")
    if learning_rate is not None:
        _finite_option("learning_rate", learning_rate, positive=True)
    if readout_only and ("readout" not in params or steps == 0):
        raise ValueError("readout_only requires read-out parameters and steps > 0")
    frozen = params
    if readout_only:
        objective = lambda p: f({**frozen, "readout": p["readout"]})
        def mask(tree):
            return {key: value if key == "readout" else jax.tree_util.tree_map(jnp.zeros_like, value)
                    for key, value in tree.items()}
    else:
        objective = f
        mask = lambda tree: tree
    opt = optax.lbfgs(learning_rate=learning_rate, memory_size=memory_size,
                      scale_init_precond=scale_init_precond,
                      linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=max_linesearch_steps,
                                                               initial_guess_strategy="one"))
    state = opt.init(params)
    vg = optax.value_and_grad_from_state(objective)

    @jax.jit
    def step(params, state):
        value, grad = vg(params, state=state)
        grad = mask(grad)
        upd, state = opt.update(grad, state, params, value=value, grad=grad, value_fn=objective)
        new = optax.apply_updates(params, mask(upd))
        if readout_only:
            new = {**frozen, "readout": new["readout"]}
        return new, state, value
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


_ENERGY_FNS = {}


def predict_energies(params, rung, structures, chunk=256):
    """interaction energies (hartree) only -- no forces -- for many structures: grouped by atom count and evaluated in
    chunks of one shape per group (the last chunk padded with repeats of its first structure), so that one compilation
    serves each group; params enter as an argument, so the compilation is reused across seeds."""
    key = repr(rung)
    if key not in _ENERGY_FNS:
        _ENERGY_FNS[key] = jax.jit(lambda p, b: jax.vmap(lambda bb: M.energy_single(p, rung, bb)[0])(b))
    f = _ENERGY_FNS[key]
    out = np.zeros(len(structures))
    groups = {}
    for i, s in enumerate(structures):
        groups.setdefault(s.n_atoms, []).append(i)
    for idx in groups.values():
        size = min(chunk, len(idx))
        for i0 in range(0, len(idx), size):
            sel = idx[i0:i0 + size]
            batch = [structures[i] for i in sel]
            batch += [batch[0]] * (size - len(batch))
            out[sel] = np.asarray(f(params, pad_batch(batch, n_freq=rung.K)))[:len(sel)]
    return out


def predict_aux(params, rung, structures):
    data = pad_batch(structures, n_freq=rung.K)
    aux = jax.jit(lambda b: M.aux_batched(params, rung, b))(data)
    return {k: np.array(v) for k, v in aux.items()}
