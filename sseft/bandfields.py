"""
Band fields at the atoms (the v1.5 inputs, x2-protocol.md section 2) and the degree->=2 response read-out.

The band potential of the charge channel at a probe point x, excluding atom i,
    V_i(x) = sum_{j != i} q_j g(|x - r_j|; l),      g = erf(r/l)/r,
and its field tensors at x = r_i:  E = -grad V (3),  gE = grad E (3x3, symmetric),  ggE (3x3x3, symmetric),
all by nested forward-mode autodiff.  Read at the central atom only; the environment features (s, v, t)
come from positions and species alone -- the two rules of x2-protocol.md 2.2.

Response read-out, bilinear in the field (no terms linear in the band field: those are the sources'):
    order 0:  -1/2 [ a |E|^2 + sum_f b_f (E.v_f)^2 + sum_f b'_f E.t_f.E ]
    order 1:  sum_f [ c_f (E.v_f) tr gE + c'_f E.gE.v_f + d_f (E.v_f)(v_f.gE.v_f) + d'_f (E.v_f)(t_f : gE) ]
    order 2:  e E.lap E + sum_f [ g_f E_d v_b v_c ggE_bcd + g'_f (E.v_f) (lap E).v_f + g''_f E_d t_bc ggE_bcd ]
with every coefficient an output of an MLP of the invariant features s_i (environment-dependent).
"""
import jax
import jax.numpy as jnp
from . import kernels as kn
from .e0 import _mlp_init, mlp


def band_field_tensors(positions, q, pmask, l, order):
    """E (N,3), gE (N,3,3), ggE (N,3,3,3) of the charge-channel band potential at each atom (own charge excluded)."""
    def V(x, i):
        d = jnp.where(pmask[i][:, None], x[None, :] - positions, 1.0)    # excluded pairs get a finite dummy vector
        rr = jnp.sqrt(jnp.sum(d * d, axis=-1))                              # so that no derivative is taken at r = 0
        w = pmask[i].astype(x.dtype)
        return jnp.sum(w * q * kn.g_long(rr, l))
    def E(x, i):
        return -jax.grad(V)(x, i)
    idx = jnp.arange(positions.shape[0])
    Ef = jax.vmap(E)(positions, idx)
    gE = jax.vmap(jax.jacfwd(E))(positions, idx) if order >= 1 else None
    ggE = jax.vmap(jax.jacfwd(jax.jacfwd(E)))(positions, idx) if order >= 2 else None
    return Ef, gE, ggE


def init_readout(key, cfg, order):
    F = cfg["F"]
    n_out = 1 + 2 * F
    if order >= 1: n_out += 4 * F
    if order >= 2: n_out += 1 + 3 * F
    return {"coef": _mlp_init(key, [F, F, n_out], scale=0.1)}


def response_energy(p, cfg, order, s, v, t, mask, E, gE, ggE):
    """per-atom response energies (N,) from the band-field tensors, degree 2 in the field."""
    F = cfg["F"]
    c = mlp(p["coef"], s)
    k = 0
    a = c[:, k]; k += 1
    b = c[:, k:k + F]; k += F
    bp = c[:, k:k + F]; k += F
    Ev = jnp.einsum("ia,iaf->if", E, v)                            # (N,F)  E . v_f
    EtE = jnp.einsum("ia,iabf,ib->if", E, t, E)                    # E . t_f . E
    eps = -0.5 * (a * jnp.sum(E * E, axis=-1) + jnp.sum(b * Ev ** 2, axis=-1) + jnp.sum(bp * EtE, axis=-1))
    if order >= 1:
        cc = c[:, k:k + F]; cp = c[:, k + F:k + 2 * F]; d = c[:, k + 2 * F:k + 3 * F]; dp = c[:, k + 3 * F:k + 4 * F]; k += 4 * F
        trG = jnp.einsum("iaa->i", gE)
        EGv = jnp.einsum("ia,iab,ibf->if", E, gE, v)
        vGv = jnp.einsum("iaf,iab,ibf->if", v, gE, v)
        tG = jnp.einsum("iabf,iab->if", t, gE)
        eps = eps + jnp.sum(cc * Ev * trG[:, None] + cp * EGv + d * Ev * vGv + dp * Ev * tG, axis=-1)
    if order >= 2:
        e = c[:, k]; k += 1
        g = c[:, k:k + F]; gp = c[:, k + F:k + 2 * F]; gpp = c[:, k + 2 * F:k + 3 * F]; k += 3 * F
        lapE = jnp.einsum("ibbd->id", ggE)                          # (N,3)
        ElapE = jnp.sum(E * lapE, axis=-1)
        EvvH = jnp.einsum("id,ibf,icf,ibcd->if", E, v, v, ggE)
        lapEv = jnp.einsum("id,idf->if", lapE, v)
        EtH = jnp.einsum("id,ibcf,ibcd->if", E, t, ggE)
        eps = eps + e * ElapE + jnp.sum(g * EvvH + gp * Ev * lapEv + gpp * EtH, axis=-1)
    return eps * mask
