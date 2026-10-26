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


def field_energy(p, s, v, t, phi, E, G, mask, e_scale, unit, parts=None):
    """per-atom energies (N,) of E_0's view of the pinned field: an MLP of
        [ s_i,  phi_i,  (W_v v_i) . E_i,  (W_t t_i) : G_i,  |E_i|^2,  |G_i|^2 ]      (3F + 3; phi, E, G in units of `unit`)
    -- the invariant features, the potential, E_0's vector and tensor channels contracted with the field and its
    gradient (MACE-POLAR's non-local read-out, Batatia et al. 2026, here local) and the two squares -- less the same MLP
    at zero field, so that the term vanishes identically where no source is in reach (an isolated molecule, a fragment
    beyond the taper of everything else).

    parts (v0.0.16): (phi_f, E_f, G_f, reach), every source fragment's share apart (bandfields.local_field_tensors with
    per_frag), and the term less its value on each fragment alone,
        eps_i = h(s_i; x_i) - sum_B reach_iB h(s_i; x_i(B)),      h(s; x) = MLP(s, x) - MLP(s, 0),
    so that it gives nothing to any dimer -- the field's self terms, a 2-body, are gone -- and carries only what the
    fragments' fields do together: the non-additive energy.  The total and the parts go through one evaluation, so that
    where one fragment is in reach the two cancel to the last digit."""
    if parts is None:
        Eu, Gu = E / unit, G / unit
        d1 = jnp.einsum("iaf,fg,ia->ig", v, p["W_v"], Eu)
        d2 = jnp.einsum("iabf,fg,iab->ig", t, p["W_t"], Gu)
        z = jnp.concatenate([s, (phi / unit)[:, None], d1, d2, jnp.sum(Eu * Eu, axis=-1)[:, None],
                             jnp.sum(Gu * Gu, axis=(-2, -1))[:, None]], axis=-1)
        z0 = jnp.concatenate([s, jnp.zeros((s.shape[0], z.shape[1] - s.shape[1]), s.dtype)], axis=-1)
        return e_scale * (mlp(p["mlp"], z) - mlp(p["mlp"], z0))[..., 0] * mask
    phi_f, E_f, G_f, reach = parts
    N, K = phi_f.shape
    X = [jnp.concatenate([x[:, None], xf], axis=1) / unit for x, xf in ((phi, phi_f), (E, E_f), (G, G_f))]  # (N,K+1,...)
    Pu, Eu, Gu = X
    d1 = jnp.einsum("iaf,fg,ika->ikg", v, p["W_v"], Eu)
    d2 = jnp.einsum("iabf,fg,ikab->ikg", t, p["W_t"], Gu)
    z = jnp.concatenate([jnp.broadcast_to(s[:, None, :], (N, K + 1, s.shape[1])), Pu[..., None], d1, d2,
                         jnp.sum(Eu * Eu, axis=-1)[..., None], jnp.sum(Gu * Gu, axis=(-2, -1))[..., None]], axis=-1)
    z0 = jnp.concatenate([s, jnp.zeros((N, z.shape[2] - s.shape[1]), s.dtype)], axis=-1)
    h = mlp(p["mlp"], z.reshape(N * (K + 1), -1)).reshape(N, K + 1) - mlp(p["mlp"], z0)[:, 0][:, None]
    return e_scale * (h[:, 0] - jnp.sum(reach * h[:, 1:], axis=1)) * mask


# ------------------------------------------------- the field term at degree 2 (v0.0.15, Rung.e0_field_form) -----
def init_field_head2(key, cfg):
    """the degree-2 field term's coefficients: an MLP of E_0's invariants, 3F + 2 outputs, whose output layer starts at
    zero, so that a rung with the term starts as the rung without it."""
    F = cfg["F"]
    head = _mlp_init(key, [F, F, 3 * F + 2])
    head[-1] = {"w": jnp.zeros_like(head[-1]["w"]), "b": jnp.zeros_like(head[-1]["b"])}
    return {"coef": head}


def field_energy2(p, s, v, t, E, G, mask, e_scale, unit, parts=None):
    """per-atom energies (N,) of E_0's view of the pinned field at degree 2 in the field:
        eps_i = -1/2 e_scale [ E_i . A_i . E_i + G_i : B_i : G_i ]                        (E, G in units of `unit`)
        E.A.E = a |E|^2 + sum_f b_f (E . v_f)^2 + sum_f b'_f E . t_f . E,      G:B:G = c |G|^2 + sum_f d_f (t_f : G)^2,
    the coefficients (a, b, b', c, d) an MLP of E_0's invariants s_i: the band-field read-out's order-0 form
    (bandfields.response_energy) on the local field, and its square in the field gradient.  No potential, no term linear
    in the field, no function of the field beyond its square: zero at zero field, and exactly quadratic in the sources.

    parts (v0.0.16): (phi_f, E_f, G_f, reach), every source fragment's share apart, and the quadratic forms less their
    values on each fragment alone -- with E_i = sum_B E_i(B),
        eps_i = -1/2 e_scale sum_{B != C} [ E_i(B) . A_i . E_i(C) + G_i(B) : B_i : G_i(C) ],
    the cross terms only: the leading non-additive induction, with no self term -|E_i(B)|^2 (a 2-body) to cancel."""
    F = s.shape[-1]
    c = mlp(p["coef"], s)
    a, b, bp, cg, dg = c[:, 0], c[:, 1:F + 1], c[:, F + 1:2 * F + 1], c[:, 2 * F + 1], c[:, 2 * F + 2:]
    if parts is None:
        Eu, Gu = E / unit, G / unit
        Ev = jnp.einsum("ia,iaf->if", Eu, v)                              # E . v_f
        EtE = jnp.einsum("ia,iabf,ib->if", Eu, t, Eu)                     # E . t_f . E
        tG = jnp.einsum("iabf,iab->if", t, Gu)                            # t_f : G
        eAe = a * jnp.sum(Eu * Eu, axis=-1) + jnp.sum(b * Ev ** 2, axis=-1) + jnp.sum(bp * EtE, axis=-1)
        gBg = cg * jnp.sum(Gu * Gu, axis=(-2, -1)) + jnp.sum(dg * tG ** 2, axis=-1)
        return -0.5 * e_scale * (eAe + gBg) * mask
    _, E_f, G_f, reach = parts
    Eu = jnp.concatenate([E[:, None], E_f], axis=1) / unit               # (N, K+1, 3): the total, then every fragment
    Gu = jnp.concatenate([G[:, None], G_f], axis=1) / unit
    Ev = jnp.einsum("ika,iaf->ikf", Eu, v)
    EtE = jnp.einsum("ika,iabf,ikb->ikf", Eu, t, Eu)
    tG = jnp.einsum("iabf,ikab->ikf", t, Gu)
    eAe = a[:, None] * jnp.sum(Eu * Eu, axis=-1) + jnp.sum(b[:, None] * Ev ** 2, axis=-1) + jnp.sum(bp[:, None] * EtE, axis=-1)
    gBg = cg[:, None] * jnp.sum(Gu * Gu, axis=(-2, -1)) + jnp.sum(dg[:, None] * tG ** 2, axis=-1)
    qf = eAe + gBg                                                        # (N, K+1)
    return -0.5 * e_scale * (qf[:, 0] - jnp.sum(reach * qf[:, 1:], axis=1)) * mask


# ------------------------------- the self terms at their own reach, the cross terms by species (v0.0.17) -----
def field_energy_split(p, s, v, t, phi, E, G, mask, e_scale, unit, parts, mlp_cross=True):
    """per-atom energies (N,) of the MLP form (field_energy's h) with its self terms at their own reach (E3-QP8):
    parts = (phi_f, E_f, G_f, reach, w_self) from bandfields.local_field_tensors with per_frag and self_taper, and
        eps_i = h(s_i; x_i) - sum_B (reach_iB - w_iB) h(s_i; x_i(B))        (mlp_cross: the MLP's own cross terms kept)
        eps_i = sum_B w_iB h(s_i; x_i(B))                                    (not mlp_cross: its self terms alone),
    h(s; x) = MLP(s, x) - MLP(s, 0) -- the cross terms h(x_i) - sum_B h(x_i(B)) at the term's reach, the self terms
    h(x_i(B)) weighted by every molecule's taper over the shorter reach.  With w = reach it is field_energy's term (the
    self terms at the term's reach), with w = 0 its cross terms alone (field_energy with parts, v0.0.16).  The total and
    the parts go through one evaluation, as there."""
    phi_f, E_f, G_f, reach, w_self = parts
    N, K = phi_f.shape
    if mlp_cross:
        X = [jnp.concatenate([x[:, None], xf], axis=1) / unit for x, xf in ((phi, phi_f), (E, E_f), (G, G_f))]  # (N,K+1,...)
    else:
        X = [xf / unit for xf in (phi_f, E_f, G_f)]                                                      # (N,K,...)
    Pu, Eu, Gu = X
    M_ = Pu.shape[1]
    d1 = jnp.einsum("iaf,fg,ika->ikg", v, p["W_v"], Eu)
    d2 = jnp.einsum("iabf,fg,ikab->ikg", t, p["W_t"], Gu)
    z = jnp.concatenate([jnp.broadcast_to(s[:, None, :], (N, M_, s.shape[1])), Pu[..., None], d1, d2,
                         jnp.sum(Eu * Eu, axis=-1)[..., None], jnp.sum(Gu * Gu, axis=(-2, -1))[..., None]], axis=-1)
    z0 = jnp.concatenate([s, jnp.zeros((N, z.shape[2] - s.shape[1]), s.dtype)], axis=-1)
    h = mlp(p["mlp"], z.reshape(N * M_, -1)).reshape(N, M_) - mlp(p["mlp"], z0)[:, 0][:, None]
    if mlp_cross:
        e = h[:, 0] - jnp.sum((reach - w_self) * h[:, 1:], axis=1)
    else:
        e = jnp.sum(w_self * h, axis=1)
    return e_scale * e * mask


def init_field_cross_species(cfg):
    """the cross terms' coefficients by species: a_Z on the fields' and c_Z on the field gradients' products, starting at
    zero, so that a rung with them starts as the rung without them."""
    n = cfg["n_species"]
    return {"a": jnp.zeros(n), "c": jnp.zeros(n)}


def field_cross_species(p, numbers, E, G, mask, e_scale, unit, parts):
    """per-atom energies (N,) of the cross terms at degree 2 with coefficients by species alone (E3-QP8):
        eps_i = -1/2 e_scale sum_{B != C} [ a_Z(i) E_i(B) . E_i(C) + c_Z(i) G_i(B) : G_i(C) ]          (E, G in `unit`)
              = -1/2 e_scale [ a_Z (|E_i|^2 - sum_B reach_iB |E_i(B)|^2) + c_Z (|G_i|^2 - sum_B reach_iB |G_i(B)|^2) ],
    with E_i = sum_B E_i(B) (bandfields.local_field_tensors with per_frag).  No feature of E_0 in it: strictly the 3-body
    of the molecules' pinned fields at fixed coefficients -- nothing on a dimer, no 4-body, and no gradient into E_0."""
    _, E_f, G_f, reach = parts[:4]
    Eu, Gu, Efu, Gfu = E / unit, G / unit, E_f / unit, G_f / unit
    e2 = jnp.sum(Eu * Eu, axis=-1) - jnp.sum(reach * jnp.sum(Efu * Efu, axis=-1), axis=1)
    g2 = jnp.sum(Gu * Gu, axis=(-2, -1)) - jnp.sum(reach * jnp.sum(Gfu * Gfu, axis=(-2, -1)), axis=1)
    return -0.5 * e_scale * (p["a"][numbers] * e2 + p["c"][numbers] * g2) * mask
