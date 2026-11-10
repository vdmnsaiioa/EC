"""
The band-field read-out's output layers are linear in their coefficients, and so are the energies and forces of the
model in them (the rest of the model held fixed).  Given a model trained without the read-out, the optimal output
layers under the training objective are therefore a weighted linear least-squares problem -- solved here in one
step before the L-BFGS polish of everything.  This is the analogue of the column-normalised analytic family of M_A
(kernels.analytic_family_scales): a linear-in-coefficients problem inside a non-linear model, which gradient
training with a fixed step does not solve across the four orders of magnitude the band field spans
(e3r-bandfield-response-preregistration.md sections 6-9).

v0.0.11: the solve may take every order's output columns (orders="all"; E3-QP left the orders >= 1 to the polish,
where they did not move: M_S2p no better than M_S0p), with the ridge chosen on held-out groups of structures (the
cores of the training probes) rather than fixed; and it may be iterated (Gauss-Newton) for a read-out that is not
linear in its coefficients -- the self-consistent response, (1 - C T)^-1 C, whose first step is the first-order solve.
"""
import numpy as np
import jax
import jax.numpy as jnp
from . import model as M

RIDGE_GRID = tuple(10.0 ** k for k in range(-12, 0))      # relative to the average diagonal of the normal matrix


class SolveReport(tuple):
    """(objective before, objective after) of the solve, with the steps' details: history [(before, after), ...] and
    the relative ridge of each step."""
    def __new__(cls, history, ridge):
        obj = super().__new__(cls, (history[0][0], history[-1][1] if history[-1][1] < history[-1][0] or len(history) == 1
                                    else history[-1][0]))
        obj.history, obj.ridge = history, ridge
        return obj


def _flat_last(readout, n0, start=0, node=True):
    """the output-layer parameters of the read-out as one vector -- the output columns start..n0 of the pair
    coefficients (order 0: the two order-0 invariants' radial columns; all columns: every order; v0.0.28, from
    the order-0 columns' end: the orders >= 1 alone) and, with node, the node term -- and the inverse.  At order 0 alone
    the orders >= 1 stay at zero and switch on in the L-BFGS polish: with ~2000 output coefficients at order 2 against
    ~1700 equations (E3-R's 36 clusters) the plain solve is under-determined and over-fits the training set (126 % of
    the non-additive energy against 44-51 % at order 0)."""
    th = readout["theta"][-1]
    leaves = [th["w"][:, start:n0], th["b"][start:n0]]
    if node:
        leaves += [readout["iso"][-1]["w"], readout["iso"][-1]["b"]]
    shapes = [l.shape for l in leaves]; sizes = [int(np.prod(s)) for s in shapes]
    def unflat(c):
        out = []; i = 0
        for sh, n in zip(shapes, sizes):
            out.append(c[i:i + n].reshape(sh)); i += n
        return out
    unflat.columns, unflat.node = (start, n0), node
    return jnp.concatenate([l.reshape(-1) for l in leaves]), unflat


def _with_last(params, c, unflat):
    parts = unflat(c)
    a, z = unflat.columns
    ro = dict(params["readout"]); th = ro["theta"][-1]
    ro["theta"] = list(ro["theta"])
    ro["theta"][-1] = {"w": th["w"].at[:, a:z].set(parts[0]), "b": th["b"].at[a:z].set(parts[1])}
    if unflat.node:
        ro["iso"] = list(ro["iso"]); ro["iso"][-1] = {"w": parts[2], "b": parts[3]}
    return {**params, "readout": ro}


def _validation_split(groups, n):
    """structures held out for the ridge: every fifth group (sorted), at least one, never all."""
    if groups is None:
        return np.zeros(n, dtype=bool)
    groups = np.asarray(groups)
    if groups.shape != (n,):
        raise ValueError("groups must give one group per structure")
    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise ValueError("the ridge selection needs at least two groups")
    held = uniq[::5] if len(uniq) >= 5 else uniq[:1]
    return np.isin(groups, held)


def solve_readout(params, rung, data, scale, w_force=10.0, ridge=1e-8, chunk=12, *, w_energy=1.0, orders="zero",
                  iterations=1, groups=None, ridge_grid=None, extra=None, anchor=False, hold=False):
    """set the read-out's output layers to the weighted least-squares optimum of the training objective (energies per
    atom and forces, scaled as in train.make_loss_sums, per-structure weights included), the rest of the model fixed.
    w_energy and w_force match train.make_loss; ridge is relative to the average diagonal of the normal matrix.
    orders: "zero" (the order-0 columns), "all" (every order's) or "higher" (v0.0.28: the orders >= 1 columns alone,
    the order-0 columns and the node term held -- arm D's third stage).  iterations > 1: Gauss-Newton steps, each the
    regularised least-squares problem of the model linearised at the current coefficients (one step is exact for the
    first-order read-out), stopped when a step does not lower the objective.  ridge_grid with groups: the ridge is the
    grid value whose solution, fitted without every fifth group, has the lowest objective on those groups (chosen at
    the first step and kept through the Gauss-Newton steps); the final solve uses all structures.  Returns the new params and a SolveReport: (the unregularised
    objective before, after), with .history (before and after of each step) and .ridge (the relative ridge of each).
    extra (v0.0.24): further rows of the objective, sum_k sum_i w_ki (t_ki - fn(params, b_k)_i)^2, given as {"fn",
    "batches", "targets", "weights"} -- e.g. job 2's liquid cells, fn their induced dipole and field response; always
    fitted (never held out), linearised with the rest, the ridge kept relative to the energy and force rows alone.
    .extra_history holds that part's (before, after) of each step.  v0.0.27: with w_energy = w_force = 0 the extra rows
    are the whole objective (data then only sizes the problem; the ridge is relative to the extra rows); extra["jac"] =
    "forward" takes their Jacobian by forward mode in chunks of extra.get("jac_chunk", 32) coefficients -- for blocks with
    many rows (every molecule's response), whose reverse-mode Jacobian holds one cotangent per row in memory.
    v0.0.29 (job 5's arm D', the third stage anchored): anchor -- the ridge pulls toward the starting coefficients
    instead of zero, (A + lam I) c = r + lam c_start, and the regularised objective's penalty is lam |c - c_start|^2;
    hold -- the energy and force rows' targets are the model's own values at the start, so those rows resist any change
    of what the starting read-out predicts for data's structures instead of pulling toward data's targets.
    .data_history holds the objective on data's own targets (with the extra rows) before and after each step; without
    hold it is .history."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and > 0")
    if any(not np.isfinite(value) or value < 0 for value in (w_energy, w_force, ridge)):
        raise ValueError("w_energy, w_force, and ridge must be finite and >= 0")
    if w_energy + w_force == 0 and extra is None:
        raise ValueError("w_energy and w_force cannot both be zero")
    rows = w_energy + w_force > 0                                    # the energy and force rows take part
    if isinstance(chunk, (bool, np.bool_)) or not isinstance(chunk, (int, np.integer)) or chunk < 1:
        raise ValueError("chunk must be an integer >= 1")
    if isinstance(iterations, (bool, np.bool_)) or not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("iterations must be an integer >= 1")
    if orders not in ("zero", "all", "higher"):
        raise ValueError("orders must be 'zero', 'all' or 'higher'")
    n = int(data["mask"].shape[0])
    held = _validation_split(groups, n) if ridge_grid is not None else np.zeros(n, dtype=bool)
    nr = rung.e0.get("n_rbf_readout", rung.e0["n_rbf"])
    n_all = int(params["readout"]["theta"][-1]["w"].shape[1])
    if orders == "higher" and n_all <= 2 * nr:
        raise ValueError("orders='higher' needs a read-out with orders >= 1")
    c, unflat = {"zero": lambda ro: _flat_last(ro, 2 * nr), "all": lambda ro: _flat_last(ro, n_all),
                 "higher": lambda ro: _flat_last(ro, n_all, start=2 * nr, node=False)}[orders](params["readout"])
    nc = len(c)
    # counts for the objective's normalisation (as in make_loss_sums over the whole set)
    wgt = np.array(data["weight"]); has_e = np.array(data["has_energy"]) * wgt * np.array(data.get("energy_weight", 1.0))
    fm_all = np.array(data["mask"])[:, :, None] * (np.array(data["has_forces"]) * wgt)[:, None, None] * np.ones((1, 1, 3))
    cnt_e = max(float(np.sum(has_e)), 1.0); cnt_f = max(float(np.sum(fm_all[..., 0])), 1.0)     # train counts atoms, not components
    natoms = np.maximum(np.array(data["mask"]).sum(1), 1.0)

    def ef_of_c(cc, b):
        return M.energy_forces(_with_last(params, cc, unflat), rung, b)
    val = jax.jit(ef_of_c)

    # the energy and force rows' targets: data's, or (hold, v0.0.29) the starting model's own values
    c_start = np.array(c)
    own_targets = (np.asarray(data["energy"], float), np.asarray(data["forces"], float))
    targets = own_targets
    if hold and rows:
        Es, Fs = [], []
        for i0 in range(0, n, chunk):
            idx = np.arange(i0, min(i0 + chunk, n)); b = {k: v[idx] for k, v in data.items()}
            E1, F1 = val(jnp.asarray(c_start), b)
            Es.append(np.array(E1)); Fs.append(np.array(F1))
        targets = (np.concatenate(Es), np.concatenate(Fs))

    # the Jacobian in c of one structure's energy and forces by reverse mode: 1 + 3N passes per structure (one per
    # output) instead of one forward pass per coefficient -- what makes the solve of every order affordable (~2000
    # coefficients at order 2 against ~45 force components)
    def jac_single(cc, bb):
        def ef(x):
            E, F = ef_of_c(x, jax.tree_util.tree_map(lambda v: v[None], bb))
            return E[0], F[0]
        if w_force == 0:                                                             # energies only: one reverse pass
            JE = jax.grad(lambda x: ef(x)[0])(cc)
            return JE, jnp.zeros(bb["forces"].shape + JE.shape, JE.dtype)
        return jax.jacrev(ef)(cc)                                                    # (nc,), (N, 3, nc)
    jac = jax.jit(lambda cc, b: jax.vmap(lambda bb: jac_single(cc, bb))(b))

    blocks, val_x, jac_x = [], None, None
    if extra is not None:
        fx = extra["fn"]
        val_x = jax.jit(lambda cc, bb: fx(_with_last(params, cc, unflat), bb))
        if extra.get("jac", "reverse") == "forward":
            k = int(extra.get("jac_chunk", 32))
            basis = np.eye(nc)
            cols = jax.jit(lambda cc, bb, T: jax.vmap(
                lambda t: jax.jvp(lambda x: fx(_with_last(params, x, unflat), bb), (cc,), (t,))[1])(T))

            def jac_x(cc, bb):
                parts = []
                for i0 in range(0, nc, k):
                    T = basis[i0:i0 + k]
                    pad = k - len(T)
                    if pad:                                            # one compiled shape for every chunk
                        T = np.concatenate([T, np.zeros((pad, nc))])
                    parts.append(np.asarray(cols(cc, bb, jnp.asarray(T)))[:k - pad])
                return np.concatenate(parts).T                         # (rows, nc)
        else:
            jac_x = jax.jit(jax.jacrev(lambda cc, bb: fx(_with_last(params, cc, unflat), bb)))
        if not len(extra["batches"]) == len(extra["targets"]) == len(extra["weights"]):
            raise ValueError("extra needs one target and one weight vector per batch")
        blocks = [(bb, np.asarray(t, float), np.asarray(w, float))
                  for bb, t, w in zip(extra["batches"], extra["targets"], extra["weights"])]

    def objective_extra(cc):
        return float(sum(np.sum(w * (t - np.array(val_x(jnp.asarray(cc), bb))) ** 2) for bb, t, w in blocks))

    def objective_rows(cc, tgt):
        out = 0.0
        for i0 in (range(0, n, chunk) if rows else ()):
            idx = np.arange(i0, min(i0 + chunk, n)); b = {k: v[idx] for k, v in data.items()}
            E1, F1 = val(jnp.asarray(cc), b); E1 = np.array(E1); F1 = np.array(F1)
            we = (w_energy * has_e[idx] / natoms[idx] ** 2 / cnt_e / scale ** 2)
            out += float(np.sum(we * (tgt[0][idx] - E1) ** 2))
            out += float(np.sum(fm_all[idx] * (w_force / cnt_f / scale ** 2) * (tgt[1][idx] - F1) ** 2))
        return out

    def objective(cc):
        return objective_rows(cc, targets) + (objective_extra(cc) if blocks else 0.0)

    def normal_equations(c0):
        """A, b of the model linearised at c0 (E = E0 + J (c - c0)), for the fitted and the held-out structures, and the
        held-out structures' constant term (their objective is c'A c - 2 c'b + k); and g, the rows' gradient at c0 (J'W
        times the residuals there: b - A c0, formed from the residuals, for the anchored step, v0.0.29)."""
        A = {False: np.zeros((nc, nc)), True: np.zeros((nc, nc))}; r = {False: np.zeros(nc), True: np.zeros(nc)}
        g = {False: np.zeros(nc), True: np.zeros(nc)}
        k_held = 0.0
        for i0 in (range(0, n, chunk) if rows else ()):
            idx = np.arange(i0, min(i0 + chunk, n))
            b = {k: v[idx] for k, v in data.items()}
            E0, F0 = val(c0, b); E0 = np.array(E0); F0 = np.array(F0)
            JE, JF = jac(c0, b); JE = np.array(JE); JF = np.array(JF)
            we = (w_energy * has_e[idx] / natoms[idx] ** 2 / cnt_e / scale ** 2)      # weights of the energy rows
            eE = targets[0][idx] - E0                                                   # the rows' residuals at c0
            rE = eE + JE @ np.array(c0)                                                 # residual target of the linear model
            wf = (fm_all[idx] * (w_force / cnt_f / scale ** 2)).reshape(len(idx), -1)
            Jf = JF.reshape(len(idx), -1, nc)
            eF = (targets[1][idx] - F0).reshape(len(idx), -1)
            rF = eF + np.einsum("bnk,k->bn", Jf, np.array(c0))
            for k in range(len(idx)):
                h = bool(held[idx[k]])
                A[h] += we[k] * np.outer(JE[k], JE[k]) + (Jf[k] * wf[k][:, None]).T @ Jf[k]
                r[h] += we[k] * JE[k] * rE[k] + Jf[k].T @ (wf[k] * rF[k])
                g[h] += we[k] * JE[k] * eE[k] + Jf[k].T @ (wf[k] * eF[k])
                if h:
                    k_held += we[k] * rE[k] ** 2 + float(np.sum(wf[k] * rF[k] ** 2))
        # the ridge's scales, from the energy and force rows alone (as before the extra rows existed)
        traces = (float(np.trace(A[False])), float(np.trace(A[False] + A[True])))
        for bb, t, w in blocks:                                     # the extra rows: always fitted
            y0 = np.array(val_x(c0, bb)); J = np.array(jac_x(c0, bb))
            A[False] += J.T @ (w[:, None] * J)
            r[False] += J.T @ (w * (t - y0 + J @ np.array(c0)))
            g[False] += J.T @ (w * (t - y0))
        if not rows:                                                # the extra rows alone: the ridge relative to them
            traces = (float(np.trace(A[False])), float(np.trace(A[False])))
        return A, r, g, k_held, traces

    def solve(A, r, g, c0, lam):
        """the regularised linearised problem's solution: (A + lam I) c = r; anchored (v0.0.29), the step from c0,
        (A + lam I)(c - c0) = g - lam (c0 - c_start), formed from the residuals (no cancellation of A c0 against r)."""
        if anchor:
            c0 = np.asarray(c0)
            if lam > 0:
                return c0 + np.linalg.solve(A + lam * np.eye(nc), g - lam * (c0 - c_start))
            return c0 + np.linalg.lstsq(A, g, rcond=None)[0]
        if lam > 0:
            return np.linalg.solve(A + lam * np.eye(nc), r)
        return np.linalg.lstsq(A, r, rcond=None)[0]

    def penalty(cc):
        d = np.asarray(cc) - c_start if anchor else np.asarray(cc)
        return lam * float(d @ d)

    history, ridges, history_x, history_d = [], [], [], []
    current = objective(c)
    current_x = objective_extra(c) if blocks else 0.0
    current_d = objective_rows(c, own_targets) + current_x if hold else current
    rel, lam = ridge, None
    for it in range(iterations):
        A, r, g, k_held, (trace_fit, trace_rows) = normal_equations(c)
        if it == 0:
            if ridge_grid is not None and held.any():
                scores = []
                for rg in ridge_grid:
                    cg = solve(A[False], r[False], g[False], c, rg * trace_fit / nc)
                    scores.append(float(cg @ A[True] @ cg - 2 * cg @ r[True] + k_held))
                rel = float(ridge_grid[int(np.argmin(scores))])
            # the ridge, chosen on the first (first-order) step, is kept in absolute terms: the Gauss-Newton steps then
            # minimise one regularised objective (re-choosing it at later linearisations drifts to vanishing ridges)
            lam = rel * trace_rows / nc
        c_new = jnp.asarray(solve(A[False] + A[True], r[False] + r[True], g[False] + g[True], c, lam))
        new = objective(c_new)
        if blocks:
            # with extra rows (a self-consistent response far from linear in c) a step can overshoot: halve it until the
            # regularised objective falls, at most five times
            step, frac = c_new - c, 1.0
            while not new + penalty(c_new) < current + penalty(c) and frac > 1 / 32:
                frac /= 2; c_new = c + frac * step; new = objective(c_new)
        new_x = objective_extra(c_new) if blocks else 0.0
        new_d = objective_rows(c_new, own_targets) + new_x if hold else new
        history.append((float(current), float(new))); ridges.append(rel); history_x.append((current_x, new_x))
        history_d.append((float(current_d), float(new_d)))
        if (it > 0 or blocks) and not new + penalty(c_new) < current + penalty(c):
            break                                              # a Gauss-Newton step that does not help: keep the last
        c, current, current_x, current_d = c_new, new, new_x, new_d
    report = SolveReport(history, ridges)
    report.extra_history = history_x
    report.data_history = history_d
    return _with_last(params, c, unflat), report
