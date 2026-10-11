"""
The band-field read-out's output layers are linear in their coefficients, and so are the energies and forces of the
model in them (the rest of the model held fixed).  Given a model trained without the read-out, the optimal output
layers under the training objective are therefore a weighted linear least-squares problem -- solved here in one
step before the L-BFGS polish of everything.  This is the analogue of the column-normalised analytic family of M_A
(kernels.analytic_family_scales): a linear-in-coefficients problem inside a non-linear model, which gradient
training with a fixed step does not solve across the four orders of magnitude the band field spans
(e3r-bandfield-response-preregistration.md sections 6-9).
"""
import numpy as np
import jax
import jax.numpy as jnp
from . import model as M


def _flat_last(readout, n0):
    """the order-0 output-layer parameters of the read-out as one vector (the first n0 output columns of the pair
    coefficients and the node term), and the inverse.  The orders >= 1 stay at zero and switch on in the L-BFGS
    polish: with ~2000 output coefficients at order 2 against ~1700 equations the plain solve is under-determined
    and over-fits the training set (clusters: 126 % of the non-additive energy against 44-51 % at order 0)."""
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


def solve_readout(params, rung, data, scale, w_force=10.0, ridge=1e-8, chunk=12):
    """set the read-out's output layers to the weighted least-squares optimum of the training objective (energies per
    atom and forces, scaled as in train.make_loss_sums, per-structure weights included), the rest of the model fixed.
    Returns the new params and the (before, after) values of the objective."""
    nr = rung.e0.get("n_rbf_readout", rung.e0["n_rbf"])
    c0, unflat = _flat_last(params["readout"], 2 * nr)                  # the order-0 invariants' columns
    n = int(data["mask"].shape[0]); nc = len(c0)
    AtA = np.zeros((nc, nc)); Atr = np.zeros(nc); loss0 = 0.0
    # counts for the objective's normalisation (as in make_loss_sums over the whole set)
    wgt = np.array(data["weight"]); has_e = np.array(data["has_energy"]) * wgt
    fm_all = np.array(data["mask"])[:, :, None] * (np.array(data["has_forces"]) * wgt)[:, None, None] * np.ones((1, 1, 3))
    cnt_e = max(float(np.sum(has_e)), 1.0); cnt_f = max(float(np.sum(fm_all[..., 0])), 1.0)     # train counts atoms, not components
    natoms = np.maximum(np.array(data["mask"]).sum(1), 1.0)

    def ef_of_c(c, b):
        p = _with_last(params, c, unflat)
        E, F = M.energy_forces(p, rung, b)
        return E, F
    val = jax.jit(ef_of_c)
    # the Jacobian in c by forward mode, a slice of tangents at a time (memory: one slice of forward passes)
    nslice = 16
    def jac_slice(c, b, T):                                             # T: (nc, t) tangent basis slice
        return jax.vmap(lambda t: jax.jvp(lambda cc: ef_of_c(cc, b), (c,), (t,))[1], in_axes=1, out_axes=-1)(T)
    jac_slice = jax.jit(jac_slice)
    eye = np.eye(nc)
    for i0 in range(0, n, chunk):
        idx = np.arange(i0, min(i0 + chunk, n))
        b = {k: v[idx] for k, v in data.items()}
        E0, F0 = val(c0, b); E0 = np.array(E0); F0 = np.array(F0)
        JE = np.zeros((len(idx), nc)); JF = np.zeros(F0.shape + (nc,))
        for j0 in range(0, nc, nslice):
            T = jnp.asarray(eye[:, j0:j0 + nslice])
            jE, jF = jac_slice(c0, b, T)
            JE[:, j0:j0 + T.shape[1]] = np.array(jE); JF[..., j0:j0 + T.shape[1]] = np.array(jF)
        we = (has_e[idx] / natoms[idx] ** 2 / cnt_e / scale ** 2)                 # weights of the energy rows
        rE = np.array(b["energy"]) - E0 + JE @ np.array(c0)                       # residual target for the linear model E = E0 + JE (c - c0)
        for k in range(len(idx)):
            AtA += we[k] * np.outer(JE[k], JE[k]); Atr += we[k] * JE[k] * rE[k]
            loss0 += we[k] * (np.array(b["energy"])[k] - E0[k]) ** 2
        wf = fm_all[idx] * (w_force / cnt_f / scale ** 2)                          # (B,N,3)
        Jf = JF.reshape(len(idx), -1, nc); wf = wf.reshape(len(idx), -1); rF = (np.array(b["forces"]) - F0).reshape(len(idx), -1) + np.einsum("bnk,k->bn", Jf, np.array(c0))
        for k in range(len(idx)):
            AtA += (Jf[k] * wf[k][:, None]).T @ Jf[k]; Atr += Jf[k].T @ (wf[k] * rF[k])
            loss0 += float(np.sum(wf[k] * ((np.array(b["forces"]) - F0).reshape(len(idx), -1)[k]) ** 2))
    lam = ridge * np.trace(AtA) / nc
    c = np.linalg.solve(AtA + lam * np.eye(nc), Atr)
    new = _with_last(params, jnp.asarray(c), unflat)
    # the objective after, exactly (the model is linear in c)
    loss1 = 0.0
    for i0 in range(0, n, chunk):
        idx = np.arange(i0, min(i0 + chunk, n)); b = {k: v[idx] for k, v in data.items()}
        E1, F1 = val(jnp.asarray(c), b); E1 = np.array(E1); F1 = np.array(F1)
        we = (has_e[idx] / natoms[idx] ** 2 / cnt_e / scale ** 2)
        loss1 += float(np.sum(we * (np.array(b["energy"]) - E1) ** 2))
        wf = fm_all[idx] * (w_force / cnt_f / scale ** 2)
        loss1 += float(np.sum(wf * (np.array(b["forces"]) - F1) ** 2))
    return new, (float(loss0), float(loss1))
