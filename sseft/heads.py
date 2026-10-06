"""
Per-atom read-outs of E_0: the site energy and the sources (the Wilson coefficients of the IR theory,
proposal section 1.1).

    eps_i        site energy (+ per-species reference e0(Z))
    q_i          charge: q0(Z) + dq_i, shifted uniformly so that sum_i q_i = Q exactly
    mu_i         dipole: gated combination of the vector channels
    alpha_i(iw)  dynamic polarisability at the K imaginary frequencies, positive (softplus), or the
                 one-oscillator form alpha0/(1 + w^2/w0^2) with alpha0, w0 > 0
Pinned mode: a source is taken from the structure (from an independent calculation) and not learned.
"""
import jax
import jax.numpy as jnp
from .e0 import _mlp_init, mlp


def init_heads(key, cfg, K=8, one_oscillator=False, energy_scale=1.0):
    F = cfg["F"]
    k = jax.random.split(key, 6)
    p = {
        "energy": _mlp_init(k[0], [F, F, 1]),
        "e_scale": jnp.asarray(energy_scale),                 # output scale of the site energies (data-derived)
        "e_ref": jnp.zeros(cfg["n_species"]),
        "charge": _mlp_init(k[1], [F, F, 1]),
        "q_ref": jnp.zeros(cfg["n_species"]),
        "dipole": _mlp_init(k[2], [F, F, F]),
        "alpha": _mlp_init(k[3], [F, F, 2 if one_oscillator else K]),
        "alpha_ref": jnp.zeros(cfg["n_species"]) + 1.0,      # species offset inside the softplus (log-scale start)
        "source_A": _mlp_init(k[4], [F, F, 1]),               # scalar source of the analytic-kernel rung
    }
    return p


def site_energy(p, s, numbers, mask):
    e = p["e_scale"] * mlp(p["energy"], s)[..., 0] + p["e_ref"][numbers]
    return e * mask


def charges(p, s, numbers, mask, total_charge):
    dq = mlp(p["charge"], s)[..., 0] + p["q_ref"][numbers]
    dq = dq * mask
    n = jnp.maximum(jnp.sum(mask), 1.0)
    shift = (jnp.sum(dq) - total_charge) / n
    return (dq - shift) * mask


def dipoles(p, s, v, mask):
    gate = mlp(p["dipole"], s)                                  # (N,F)
    mu = jnp.einsum("if,iaf->ia", gate, v)
    return mu * mask[:, None]


def polarisabilities(p, s, numbers, mask, omega, one_oscillator=False):
    """alpha_i(i omega_k), shape (N,K), positive."""
    raw = mlp(p["alpha"], s)
    if one_oscillator:
        a0 = jax.nn.softplus(raw[..., 0] + p["alpha_ref"][numbers])
        w0 = jax.nn.softplus(raw[..., 1]) + 1e-3
        alpha = a0[:, None] / (1.0 + (omega[None, :] / w0[:, None]) ** 2)
    else:
        alpha = jax.nn.softplus(raw + p["alpha_ref"][numbers][:, None])
    return alpha * mask[:, None]


def scalar_source(p, s, mask):
    return mlp(p["source_A"], s)[..., 0] * mask
