"""
Per-atom read-outs of E_0: the site energy and the sources (the Wilson coefficients of the IR theory,
proposal section 1.1).

    eps_i        site energy (+ per-species reference e0(Z))
    q_i          charge: q0(Z) + dq_i, shifted uniformly within each fragment so that the fragment's charges sum
                 exactly to its own total charge (a molecule stays neutral in any environment, an ion keeps its
                 charge); for structures without fragments one shift over the structure to Q
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
        "alpha2": _mlp_init(k[5], [F, F, 2 if one_oscillator else K]),   # quadrupole polarisability alpha_2(i w) (C8 channel)
        "alpha2_ref": jnp.zeros(cfg["n_species"]) + 2.0,
        "source_A": _mlp_init(k[4], [F, F, 1]),               # scalar source of the analytic-kernel rung
    }
    return p


def site_energy(p, s, numbers, mask):
    e = p["e_scale"] * mlp(p["energy"], s)[..., 0] + p["e_ref"][numbers]
    return e * mask


def charges(p, s, numbers, mask, total_charge, frag_mask=None, frag_charge=None):
    """q_i = q0(Z_i) + dq_i, shifted uniformly so that the charges sum exactly to their target.  With frag_mask (F,N)
    and frag_charge (F,) the shift is taken fragment by fragment -- each fragment's charges sum to its own total
    charge, whatever its environment, so that the charges of a fragment beyond r_c of everything else are exactly
    those of the fragment alone (the fragment reference E_int = E(all) - sum_f E(f) then cancels the intramolecular
    band energy exactly, and no fragment acquires a monopole from the others' environment-dependent dq).  Without
    fragments one shift over the structure to total_charge."""
    dq = mlp(p["charge"], s)[..., 0] + p["q_ref"][numbers]
    dq = dq * mask
    if frag_mask is None:
        n = jnp.maximum(jnp.sum(mask), 1.0)
        shift = (jnp.sum(dq) - total_charge) / n
        return (dq - shift) * mask
    fm = frag_mask & mask[None, :]                                          # (F,N): each fragment's atoms inside the mask
    n_f = jnp.maximum(jnp.sum(fm, axis=1), 1.0)
    shift_f = (jnp.sum(fm * dq[None, :], axis=1) - frag_charge) / n_f       # (F,) one shift per fragment
    shift = jnp.sum(fm * shift_f[:, None], axis=0)                          # (N,) the shift of each atom's own fragment
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


def quad_polarisabilities(p, s, numbers, mask, omega, one_oscillator=False):
    """alpha_2,i(i omega_k), shape (N,K), positive (the quadrupole polarisability; sources the C8 band)."""
    raw = mlp(p["alpha2"], s)
    if one_oscillator:
        a0 = jax.nn.softplus(raw[..., 0] + p["alpha2_ref"][numbers])
        w0 = jax.nn.softplus(raw[..., 1]) + 1e-3
        alpha = a0[:, None] / (1.0 + (omega[None, :] / w0[:, None]) ** 2)
    else:
        alpha = jax.nn.softplus(raw + p["alpha2_ref"][numbers][:, None])
    return alpha * mask[:, None]


def scalar_source(p, s, mask):
    return mlp(p["source_A"], s)[..., 0] * mask


# ---------------------------------------------------------------- E_0's field term (v0.0.14, Rung.e0_field) -----
def init_field_head(key, cfg):
    """the field term's parameters: two channel maps and an MLP whose output layer starts at zero, so that a rung with
    the field term starts as the rung without it."""
    F = cfg["F"]
    k = jax.random.split(key, 3)
    head = _mlp_init(k[2], [3 * F + 3, F, F, 1])
    head[-1] = {"w": jnp.zeros_like(head[-1]["w"]), "b": jnp.zeros_like(head[-1]["b"])}
    return {"W_v": jax.random.normal(k[0], (F, F)) / jnp.sqrt(F), "W_t": jax.random.normal(k[1], (F, F)) / jnp.sqrt(F),
            "mlp": head}


def field_energy(p, s, v, t, phi, E, G, mask, e_scale, unit):
    """per-atom energies (N,) of E_0's view of the pinned field: an MLP of
        [ s_i,  phi_i,  (W_v v_i) . E_i,  (W_t t_i) : G_i,  |E_i|^2,  |G_i|^2 ]      (3F + 3; phi, E, G in units of `unit`)
    -- the invariant features, the potential, E_0's vector and tensor channels contracted with the field and its
    gradient (MACE-POLAR's non-local read-out, Batatia et al. 2026, here local) and the two squares -- less the same MLP
    at zero field, so that the term vanishes identically where no source is in reach (an isolated molecule, a fragment
    beyond the taper of everything else)."""
    Eu, Gu = E / unit, G / unit
    d1 = jnp.einsum("iaf,fg,ia->ig", v, p["W_v"], Eu)
    d2 = jnp.einsum("iabf,fg,iab->ig", t, p["W_t"], Gu)
    z = jnp.concatenate([s, (phi / unit)[:, None], d1, d2, jnp.sum(Eu * Eu, axis=-1)[:, None],
                         jnp.sum(Gu * Gu, axis=(-2, -1))[:, None]], axis=-1)
    z0 = jnp.concatenate([s, jnp.zeros((s.shape[0], z.shape[1] - s.shape[1]), s.dtype)], axis=-1)
    return e_scale * (mlp(p["mlp"], z) - mlp(p["mlp"], z0))[..., 0] * mask
