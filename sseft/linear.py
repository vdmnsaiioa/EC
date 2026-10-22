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


def _flat_last(readout, n0):
    """the output-layer parameters of the read-out as one vector -- the first n0 output columns of the pair
    coefficients (order 0: the two order-0 invariants' radial columns; all columns: every order) and the node term --
    and the inverse.  At order 0 alone the orders >= 1 stay at zero and switch on in the L-BFGS polish: with ~2000
    output coefficients at order 2 against ~1700 equations (E3-R's 36 clusters) the plain solve is under-determined and
    over-fits the training set (126 % of the non-additive energy against 44-51 % at order 0)."""
    th = readout["theta"][-1]
    leaves = [th["w"][:, :n0], th["b"][:n0], readout["iso"][-1]["w"], readout["iso"][-1]["b"]]
    shapes = [l.shape for l in leaves]; sizes = [int(np.prod(s)) for s in shapes]
    def unflat(c):
        out = []; i = 0
        for sh, n in zip(shapes, sizes):
            out.append(c[i:i + n].reshape(sh)); i += n
        return out
    return jnp.concatenate([l.reshape(-1) for l in leaves]), unflat


def _with_last(params, c, unflat):
    w1, b1, w2, b2 = unflat(c)
    ro = dict(params["readout"]); th = ro["theta"][-1]; n0 = w1.shape[1]
    ro["theta"] = list(ro["theta"]); ro["theta"][-1] = {"w": th["w"].at[:, :n0].set(w1), "b": th["b"].at[:n0].set(b1)}
    ro["iso"] = list(ro["iso"]); ro["iso"][-1] = {"w": w2, "b": b2}
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
                  iterations=1, groups=None, ridge_grid=None):
    """set the read-out's output layers to the weighted least-squares optimum of the training objective (energies per
    atom and forces, scaled as in train.make_loss_sums, per-structure weights included), the rest of the model fixed.
    w_energy and w_force match train.make_loss; ridge is relative to the average diagonal of the normal matrix.
    orders: "zero" (the order-0 columns) or "all" (every order's).  iterations > 1: Gauss-Newton steps, each the
    regularised least-squares problem of the model linearised at the current coefficients (one step is exact for the
    first-order read-out), stopped when a step does not lower the objective.  ridge_grid with groups: the ridge is the
    grid value whose solution, fitted without every fifth group, has the lowest objective on those groups (chosen at
    the first step and kept through the Gauss-Newton steps); the final solve uses all structures.  Returns the new params and a SolveReport: (the unregularised
    objective before, after), with .history (before and after of each step) and .ridge (the relative ridge of each)."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and > 0")
    if any(not np.isfinite(value) or value < 0 for value in (w_energy, w_force, ridge)):
        raise ValueError("w_energy, w_force, and ridge must be finite and >= 0")
    if w_energy + w_force == 0:
        raise ValueError("w_energy and w_force cannot both be zero")
    if isinstance(chunk, (bool, np.bool_)) or not isinstance(chunk, (int, np.integer)) or chunk < 1:
        raise ValueError("chunk must be an integer >= 1")
    if isinstance(iterations, (bool, np.bool_)) or not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("iterations must be an integer >= 1")
    if orders not in ("zero", "all"):
        raise ValueError("orders must be 'zero' or 'all'")
    n = int(data["mask"].shape[0])
    held = _validation_split(groups, n) if ridge_grid is not None else np.zeros(n, dtype=bool)
    nr = rung.e0.get("n_rbf_readout", rung.e0["n_rbf"])
    n0 = 2 * nr if orders == "zero" else int(params["readout"]["theta"][-1]["w"].shape[1])
    c, unflat = _flat_last(params["readout"], n0)
    nc = len(c)
    # counts for the objective's normalisation (as in make_loss_sums over the whole set)
    wgt = np.array(data["weight"]); has_e = np.array(data["has_energy"]) * wgt * np.array(data.get("energy_weight", 1.0))
    fm_all = np.array(data["mask"])[:, :, None] * (np.array(data["has_forces"]) * wgt)[:, None, None] * np.ones((1, 1, 3))
    cnt_e = max(float(np.sum(has_e)), 1.0); cnt_f = max(float(np.sum(fm_all[..., 0])), 1.0)     # train counts atoms, not components
    natoms = np.maximum(np.array(data["mask"]).sum(1), 1.0)

    def ef_of_c(cc, b):
        return M.energy_forces(_with_last(params, cc, unflat), rung, b)
    val = jax.jit(ef_of_c)

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

    def objective(cc):
        out = 0.0
        for i0 in range(0, n, chunk):
            idx = np.arange(i0, min(i0 + chunk, n)); b = {k: v[idx] for k, v in data.items()}
            E1, F1 = val(jnp.asarray(cc), b); E1 = np.array(E1); F1 = np.array(F1)
            we = (w_energy * has_e[idx] / natoms[idx] ** 2 / cnt_e / scale ** 2)
            out += float(np.sum(we * (np.array(b["energy"]) - E1) ** 2))
            out += float(np.sum(fm_all[idx] * (w_force / cnt_f / scale ** 2) * (np.array(b["forces"]) - F1) ** 2))
        return out

    def normal_equations(c0):
        """A, b of the model linearised at c0 (E = E0 + J (c - c0)), for the fitted and the held-out structures, and the
        held-out structures' constant term (their objective is c'A c - 2 c'b + k)."""
        A = {False: np.zeros((nc, nc)), True: np.zeros((nc, nc))}; r = {False: np.zeros(nc), True: np.zeros(nc)}
        k_held = 0.0
        for i0 in range(0, n, chunk):
            idx = np.arange(i0, min(i0 + chunk, n))
            b = {k: v[idx] for k, v in data.items()}
            E0, F0 = val(c0, b); E0 = np.array(E0); F0 = np.array(F0)
            JE, JF = jac(c0, b); JE = np.array(JE); JF = np.array(JF)
            we = (w_energy * has_e[idx] / natoms[idx] ** 2 / cnt_e / scale ** 2)      # weights of the energy rows
            rE = np.array(b["energy"]) - E0 + JE @ np.array(c0)                       # residual target of the linear model
            wf = (fm_all[idx] * (w_force / cnt_f / scale ** 2)).reshape(len(idx), -1)
            Jf = JF.reshape(len(idx), -1, nc)
            rF = (np.array(b["forces"]) - F0).reshape(len(idx), -1) + np.einsum("bnk,k->bn", Jf, np.array(c0))
            for k in range(len(idx)):
                h = bool(held[idx[k]])
                A[h] += we[k] * np.outer(JE[k], JE[k]) + (Jf[k] * wf[k][:, None]).T @ Jf[k]
                r[h] += we[k] * JE[k] * rE[k] + Jf[k].T @ (wf[k] * rF[k])
                if h:
                    k_held += we[k] * rE[k] ** 2 + float(np.sum(wf[k] * rF[k] ** 2))
        return A, r, k_held

    def solve(A, r, lam):
        if lam > 0:
            return np.linalg.solve(A + lam * np.eye(nc), r)
        return np.linalg.lstsq(A, r, rcond=None)[0]

    history, ridges = [], []
    current = objective(c)
    rel, lam = ridge, None
    for it in range(iterations):
        A, r, k_held = normal_equations(c)
        if it == 0:
            if ridge_grid is not None and held.any():
                scores = []
                for g in ridge_grid:
                    cg = solve(A[False], r[False], g * np.trace(A[False]) / nc)
                    scores.append(float(cg @ A[True] @ cg - 2 * cg @ r[True] + k_held))
                rel = float(ridge_grid[int(np.argmin(scores))])
            # the ridge, chosen on the first (first-order) step, is kept in absolute terms: the Gauss-Newton steps then
            # minimise one regularised objective (re-choosing it at later linearisations drifts to vanishing ridges)
            lam = rel * np.trace(A[False] + A[True]) / nc
        c_new = jnp.asarray(solve(A[False] + A[True], r[False] + r[True], lam))
        new = objective(c_new)
        history.append((float(current), float(new))); ridges.append(rel)
        if it > 0 and not new + lam * float(c_new @ c_new) < current + lam * float(c @ c):
            break                                              # a Gauss-Newton step that does not help: keep the last
        c, current = c_new, new
    return _with_last(params, c, unflat), SolveReport(history, ridges)
