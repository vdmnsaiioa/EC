"""
E_0: a compact equivariant message-passing network (PaiNN-style scalar + vector channels, plus a traceless
rank-2 tensor channel so that quadrupolar environment tensors are representable).  Pure functions with
explicit parameter pytrees; no framework.

    s_i in R^F, v_i in R^{3 x F}, t_i in R^{3 x 3 x F} (traceless symmetric),  T interaction blocks.

The read-outs (sources, energies, band-field response tensors) live in heads.py and bandfields.py.
"""
import math
import jax
import jax.numpy as jnp
from .structure import pair_geometry

ACT = jax.nn.silu


def _linear_init(key, n_in, n_out, scale=1.0):
    w = jax.random.normal(key, (n_in, n_out)) * scale / math.sqrt(n_in)
    return {"w": w, "b": jnp.zeros(n_out)}


def _linear(p, x):
    return x @ p["w"] + p["b"]


def _mlp_init(key, sizes, scale=1.0):
    keys = jax.random.split(key, len(sizes) - 1)
    return [_linear_init(k, a, b, scale) for k, a, b in zip(keys, sizes[:-1], sizes[1:])]


def mlp(p, x):
    for i, layer in enumerate(p):
        x = _linear(layer, x)
        if i < len(p) - 1:
            x = ACT(x)
    return x


def default_config(**kw):
    cfg = dict(F=32, T=2, n_rbf=20, r_cut=11.34, n_species=56)      # r_cut in bohr (6 A)
    cfg.update(kw)
    return cfg


def rbf(r, cfg):
    """Bessel radial basis sin(n pi r / r_c)/r times a cosine cutoff."""
    n = jnp.arange(1, cfg["n_rbf"] + 1)
    rc = cfg["r_cut"]
    basis = jnp.sin(n * math.pi * r[..., None] / rc) / r[..., None]
    fc = 0.5 * (1.0 + jnp.cos(math.pi * jnp.clip(r / rc, 0.0, 1.0)))
    return basis * fc[..., None], fc


def init_e0(key, cfg):
    F = cfg["F"]
    keys = jax.random.split(key, 2 + 4 * cfg["T"])
    params = {"embed": jax.random.normal(keys[0], (cfg["n_species"], F)) * 0.5, "blocks": []}
    for tblock in range(cfg["T"]):
        k = jax.random.split(keys[2 + tblock], 6)
        params["blocks"].append({
            "phi": _mlp_init(k[0], [F, F, 4 * F]),           # message filters from s_j
            "W": jax.random.normal(k[1], (cfg["n_rbf"], 4 * F)) / math.sqrt(cfg["n_rbf"]),   # radial weights, no bias
            "U": jax.random.normal(k[2], (F, F)) / math.sqrt(F),
            "V": jax.random.normal(k[3], (F, F)) / math.sqrt(F),
            "Tmap": jax.random.normal(k[4], (F, F)) / math.sqrt(F),
            "upd": _mlp_init(k[5], [3 * F, F, 4 * F]),       # a_ss, a_sv, a_vv, a_tt from [s, |Vv|, |t|]
        })
    return params


def apply_e0(params, cfg, positions, numbers, mask, cell=None, pbc=False):
    """returns s (N,F), v (N,3,F), t (N,3,3,F), and the pair geometry (D, r, pmask)."""
    F = cfg["F"]
    N = positions.shape[0]
    D, r, pmask = pair_geometry(positions, mask, cell, pbc)
    within = pmask & (r < cfg["r_cut"])
    rhat = D / r[..., None]
    basis, fc = rbf(r, cfg)
    basis = basis * within[..., None]
    outer = rhat[..., :, None] * rhat[..., None, :] - jnp.eye(3) / 3.0                # (N,N,3,3) traceless
    s = params["embed"][numbers] * mask[:, None]
    v = jnp.zeros((N, 3, F))
    t = jnp.zeros((N, 3, 3, F))
    for blk in params["blocks"]:
        # message
        phi = mlp(blk["phi"], s)                                  # (N,4F) of the sender j
        W = (basis @ blk["W"]) * within[..., None]                # (N,N,4F); exactly zero beyond the cutoff
        m = phi[None, :, :] * W                                   # (i, j, 4F)
        a, b, c, d = jnp.split(m, 4, axis=-1)
        ds = jnp.sum(a, axis=1)
        dv = jnp.einsum("ijf,jaf->iaf", b, v) + jnp.einsum("ijf,ija->iaf", c, rhat)
        dt = jnp.einsum("ijf,ijab->iabf", d, outer)
        s = s + ds * mask[:, None]; v = v + dv * mask[:, None, None]; t = t + dt * mask[:, None, None, None]
        # update
        Uv = jnp.einsum("iaf,fg->iag", v, blk["U"]); Vv = jnp.einsum("iaf,fg->iag", v, blk["V"])
        Tt = jnp.einsum("iabf,fg->iabg", t, blk["Tmap"])
        vnorm = jnp.sqrt(jnp.sum(Vv * Vv, axis=1) + 1e-12)
        tnorm = jnp.sqrt(jnp.sum(Tt * Tt, axis=(1, 2)) + 1e-12)
        aa = mlp(blk["upd"], jnp.concatenate([s, vnorm, tnorm], axis=-1))
        a_ss, a_sv, a_vv, a_tt = jnp.split(aa, 4, axis=-1)
        s = s + (a_ss + a_sv * jnp.sum(Uv * Vv, axis=1)) * mask[:, None]
        v = v + a_vv[:, None, :] * Uv * mask[:, None, None]
        t = t + a_tt[:, None, None, :] * Tt * mask[:, None, None, None]
    return s, v, t, (D, r, pmask)
