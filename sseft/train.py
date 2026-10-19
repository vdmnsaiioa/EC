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
        lbfgs_readout_only=False, readout_orders="zero", readout_solve_iterations=1, readout_ridge_selection="fixed",
        readout_solve_weights=None, readout_solve_groups=None, lbfgs_freeze_readout=False, readout_first=False,
        readout_solve_subset=None, readout_solve_w_force=None):
    """train one model; returns (params, info).  The band-field read-out (when the rung has one) is held at its
    zero initialisation during the Adam stage unless readout_adam=True, and switched on in the L-BFGS polish: a
    quadratic read-out in the band field spans four orders of magnitude in |E|^2 across a training set with near and
    far sources, so that Adam's fixed-size steps on its coefficients either do nothing in the far field (coefficients
    in physical units) or wreck the near configurations (coefficients O(1)); L-BFGS, being curvature-aware, handles
    the stiff directions (e3r-bandfield-response-preregistration.md section 8).  The polish is therefore mandatory
    when the read-out is frozen during Adam.  Set lbfgs_readout_only=True to preserve the Adam-trained model
    outside the read-out during that polish.  readout_linear_solve=None follows the frozen-read-out default;
    True or False explicitly enables or disables the solve on rungs with a read-out.  steps=0 runs only the
    subsequent stages.

    The solve (sseft.linear.solve_readout): readout_orders "zero" (the order-0 columns, as before) or "all";
    readout_solve_iterations Gauss-Newton steps (a read-out not linear in its coefficients: the self-consistent
    response); readout_ridge_selection "fixed" (readout_ridge) or "groups" (chosen from linear.RIDGE_GRID on held-out
    groups of readout_solve_groups, one per structure); readout_solve_weights: the structures' weights in the solve
    instead of their training weights (e.g. far-field probes weighted up: the response matched on the far field, the
    clusters keeping it sane in the near field); readout_solve_subset: the indices of the structures it is solved on.
    lbfgs_freeze_readout keeps the solved read-out through the polish (everything else polished).  readout_first: the
    solve before Adam, on readout_solve_subset -- a read-out that does not depend on E_0 (Rung.bf_species) matched on
    configurations E_0 does not see (far-field probes, their interaction energies), then held through Adam and the
    polish while E_0 learns the rest with the response in place.  readout_solve_w_force: the forces' weight in the solve
    (default w_force; 0: energies only -- for rigid monomers whose truth has geometry-dependent charges, the atomic forces'
    internal components carry the charge flux that pinned charges cannot represent)."""
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
    if readout_orders not in ("zero", "all"):
        raise ValueError("readout_orders must be 'zero' or 'all'")
    if readout_ridge_selection not in ("fixed", "groups"):
        raise ValueError("readout_ridge_selection must be 'fixed' or 'groups'")
    _integer_option("readout_solve_iterations", readout_solve_iterations, 1)
    if not isinstance(lbfgs_freeze_readout, (bool, np.bool_)):
        raise ValueError("lbfgs_freeze_readout must be a boolean")
    if lbfgs_freeze_readout and lbfgs_readout_only:
        raise ValueError("lbfgs_freeze_readout and lbfgs_readout_only exclude each other")
    if not isinstance(readout_first, (bool, np.bool_)):
        raise ValueError("readout_first must be a boolean")
    if readout_solve_w_force is not None:
        _finite_option("readout_solve_w_force", readout_solve_w_force)
        if w_energy == 0 and readout_solve_w_force == 0:
            raise ValueError("the solve needs energies or forces: w_energy and readout_solve_w_force are both zero")
    if readout_first and (readout_adam or not lbfgs_freeze_readout or readout_solve_subset is None):
        raise ValueError("readout_first solves the read-out on readout_solve_subset before Adam and keeps it: it needs "
                         "the read-out frozen in Adam and in the polish (lbfgs_freeze_readout)")
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

    t0 = time.time(); hist = []
    solve = None

    def run_solve(params):
        from .linear import solve_readout, RIDGE_GRID
        extra = {}
        if readout_orders != "zero": extra["orders"] = readout_orders
        if readout_solve_iterations != 1: extra["iterations"] = int(readout_solve_iterations)
        if readout_ridge_selection == "groups":
            if readout_solve_groups is None:
                raise ValueError("readout_ridge_selection 'groups' needs readout_solve_groups")
            extra.update(ridge_grid=RIDGE_GRID, groups=np.asarray(readout_solve_groups))
        solve_data, n_solve = data, n_data
        if readout_solve_subset is not None:
            subset = np.asarray(readout_solve_subset, dtype=int)
            if subset.ndim != 1 or len(subset) == 0 or subset.min() < 0 or subset.max() >= n_data:
                raise ValueError("readout_solve_subset must index the training structures")
            solve_data, n_solve = {k: v[subset] for k, v in data.items()}, len(subset)
        if "groups" in extra and len(extra["groups"]) != n_solve:
            raise ValueError("readout_solve_groups must give one group per structure the read-out is solved on")
        if readout_solve_weights is not None:
            sw = np.asarray(readout_solve_weights, dtype=float)
            if sw.shape != (n_solve,) or not np.all(np.isfinite(sw)) or np.any(sw < 0) or not np.any(sw > 0):
                raise ValueError("readout_solve_weights must give a finite weight >= 0 per structure of the solve, not all zero")
            solve_data = {**solve_data, "weight": jnp.asarray(sw)}
        params, report = solve_readout(params, rung, solve_data, scale,
                                       w_force=w_force if readout_solve_w_force is None else readout_solve_w_force,
                                       ridge=readout_ridge, chunk=lbfgs_chunk, w_energy=w_energy, **extra)
        steps_txt = "; ".join(f"{b:.3e} -> {a:.3e}" + (f" (ridge {r:.0e})" if r is not None and "ridge_grid" in extra else "")
                              for (b, a), r in zip(getattr(report, "history", [report]), getattr(report, "ridge", [None])))
        print(f"    read-out linear solve" + (f" on {n_solve} structures" if readout_solve_subset is not None else "")
              + (" (structure weights of the solve)" if readout_solve_weights is not None else "")
              + f": objective {steps_txt}  [{time.time() - t0:.0f} s]", flush=True)
        return params, report

    if linear_solve and readout_first:
        params, solve = run_solve(params)
        state = opt.init(params)
    rng = np.random.default_rng(seed)
    for it in range(steps):
        idx = rng.choice(n_data, size=min(batch_size, n_data), replace=False) if batch_size < n_data else np.arange(n_data)
        batch = {k: v[idx] for k, v in data.items()}
        params, state, l, parts = step(params, state, batch)
        if it % max(steps // 10, 1) == 0 or it == steps - 1:
            hist.append((it, float(l), float(parts[0]), float(parts[1])))
            if verbose:
                print(f"    step {it:5d}  loss {float(l):.3e}  (E {float(parts[0]):.2e}, F {float(parts[1]):.2e})  [{time.time() - t0:.0f} s]", flush=True)
    if linear_solve and not readout_first:
        # the read-out's output layers are linear in the energies and forces: set them to the weighted least-squares
        # optimum of the objective given the Adam-trained rest of the model (sseft.linear), then polish everything
        params, solve = run_solve(params)
    if lbfgs_steps > 0:
        f = make_chunked_loss(rung, scale, data, w_energy, w_force, chunk=lbfgs_chunk)
        extra = {"freeze_readout": True} if lbfgs_freeze_readout and has_readout else {}
        params = polish_lbfgs(params, f, lbfgs_steps, verbose, learning_rate=lbfgs_lr,
                              memory_size=lbfgs_memory, scale_init_precond=lbfgs_scale_init_precond,
                              max_linesearch_steps=lbfgs_max_linesearch_steps, readout_only=lbfgs_readout_only, **extra)
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
    if (readout_orders, readout_solve_iterations, readout_ridge_selection, lbfgs_freeze_readout, readout_first) != \
            ("zero", 1, "fixed", False, False) or readout_solve_weights is not None or readout_solve_subset is not None \
            or readout_solve_w_force is not None:
        options.update(readout_orders=readout_orders, readout_solve_iterations=int(readout_solve_iterations),
                       readout_ridge_selection=readout_ridge_selection, lbfgs_freeze_readout=bool(lbfgs_freeze_readout),
                       readout_first=bool(readout_first), readout_solve_w_force=None if readout_solve_w_force is None
                       else float(readout_solve_w_force),
                       readout_solve_structures=None if readout_solve_subset is None else int(len(readout_solve_subset)),
                       readout_solve_weights=None if readout_solve_weights is None
                       else sorted(set(float(w) for w in readout_solve_weights)))
        if solve is not None and hasattr(solve, "ridge"):
            options["readout_solve_ridge"] = [float(r) for r in solve.ridge]
    return params, {"scale": scale, "history": hist, "time": time.time() - t0, "readout_solve": solve,
                    "training_options": options}


def polish_lbfgs(params, f, steps, verbose=False, learning_rate=None, memory_size=10,
                 scale_init_precond=True, max_linesearch_steps=20, readout_only=False, freeze_readout=False):
    """f(params) -> scalar loss over the whole data set.  With readout_only, preserve all other parameters; with
    freeze_readout, preserve the read-out (polish everything else)."""
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
    if not isinstance(freeze_readout, (bool, np.bool_)) or (freeze_readout and ("readout" not in params or readout_only)):
        raise ValueError("freeze_readout requires read-out parameters and excludes readout_only")
    frozen = params
    if freeze_readout:
        objective = lambda p: f({**p, "readout": frozen["readout"]})
        def mask(tree):
            return {key: jax.tree_util.tree_map(jnp.zeros_like, value) if key == "readout" else value
                    for key, value in tree.items()}
    elif readout_only:
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
        elif freeze_readout:
            new = {**new, "readout": frozen["readout"]}
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
