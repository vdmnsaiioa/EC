"""
Band fields at the atoms (the v1.5 inputs, x2-protocol.md section 2) and the degree->=2 response read-out.

The band potential of the charge channel at a probe point x, excluding atom i,
    V_i(x) = sum_{j != i} q_j g(|x - r_j|; l),      g = erf(r/l)/r,
and its field tensors at x = r_i:  E = -grad V (3),  gE = grad E (3x3, symmetric),  ggE (3x3x3, symmetric),
all by nested forward-mode autodiff.  Read at the central atom only; the environment features (s, v, t)
come from positions and species alone -- the two rules of x2-protocol.md 2.2.

The derivative tensors enter the read-out in units of the band edge, G = l gE and H = l^2 ggE, so that
every invariant has the scale of |E|^2 (the band field above l is l-smooth: |grad^k E| ~ |E| / l^k).

Response read-out, bilinear in the field (no terms linear in the band field: those are the sources'):
    order 0:  -1/2 [ a |E|^2 + sum_f b_f (E.v_f)^2 + sum_f b'_f E.t_f.E ]
    order 1:  sum_f [ c_f (E.v_f) tr G + c'_f E.G.v_f + d_f (E.v_f)(v_f.G.v_f) + d'_f (E.v_f)(t_f : G) ]
    order 2:  e E.lap E + e' (tr G)^2 + e'' G:G
              + sum_f [ g_f E_d v_b v_c H_bcd + g'_f (E.v_f) (lap E).v_f + g''_f E_d t_bc H_bcd
                        + h_f v_f.G.G.v_f + h'_f t_f : (G.G) ]
with every coefficient an output of an MLP of the invariant features s_i (environment-dependent).  The
coefficients of the orders >= 1 start at zero, so that training begins from the order-0 model and the
higher orders switch on as the data ask for them.
"""
import jax
import jax.numpy as jnp
from . import kernels as kn
from .e0 import _mlp_init, mlp


def near_weight(r, rs):
    """the weight of a source at distance r in the band-field input: 1 - exp(-(r / r_s)^10), a smooth switch that
    removes the sources closer than r_s (the near field is E_0's) and is 1 beyond ~1.3 r_s; rs <= 0 switches it off.

    Why: a term quadratic in the band field, |E_near + E_far|^2, contains 2 E_near . E_far, a term LINEAR in the far
    field with an environment-dependent coefficient -- exactly the operator class the degree-2 rule excludes,
    because it is degenerate with a dipole source (x2-protocol.md 2.1).  For a molecule the near field of its own
    charges (smeared at l_1) is a fixed vector in the molecular frame, and the spurious term is -alpha E_intra . E_ext,
    of order 0.5 kcal/mol for water at 3 A: the near sources must not enter the band-field input."""
    if rs <= 0:
        return jnp.ones_like(r)
    return 1.0 - jnp.exp(-(r / rs) ** 10)


def band_field_tensors(positions, q, pmask, l, order, rs=0.0):
    """E (N,3), gE (N,3,3), ggE (N,3,3,3) of the charge-channel band potential at each atom (own charge excluded,
    sources closer than rs switched off smoothly)."""
    def V(x, i):
        d = jnp.where(pmask[i][:, None], x[None, :] - positions, 1.0)    # excluded pairs get a finite dummy vector
        rr = jnp.sqrt(jnp.sum(d * d, axis=-1))                              # so that no derivative is taken at r = 0
        w = pmask[i].astype(x.dtype) * near_weight(rr, rs)
        return jnp.sum(w * q * kn.g_long(rr, l))
    def E(x, i):
        return -jax.grad(V)(x, i)
    idx = jnp.arange(positions.shape[0])
    Ef = jax.vmap(E)(positions, idx)
    gE = jax.vmap(jax.jacfwd(E))(positions, idx) if order >= 1 else None
    ggE = jax.vmap(jax.jacfwd(jax.jacfwd(E)))(positions, idx) if order >= 2 else None
    return Ef, gE, ggE


# ---------------------------------------------------------------- pair read-out (default) --------------------
# The Taylor class on the physical structure (x2-protocol.md 1.3): the response tensors of atom i are sums over
# its neighbours of radial functions times powers of the unit vector n = r_ij / r, so the read-out is
#     eps_i = sum_{j != i} sum_m w_m(r_ij; s_i, s_j) I_m(n_ij; E_i, G_i, H_i),
# with I_m the degree-2 invariants of (E, G = l grad E, H = l^2 grad grad E) and n, and w_m = sum_n B_n(r) theta_mn(s_i, s_j)
# on the Bessel basis with the cosine cutoff of E_0 (the range of the medium's non-local response is learned
# inside r_c).  A node-level isotropic term a(s_i) |E_i|^2 completes the order-0 sector.  This represents the
# rank-4 environment tensors (sum_j w r^{(x)4}) that the order-2 operators need and that scalar / vector /
# rank-2 node channels cannot carry; the node read-out below is kept as the cheaper L <= 2 variant.

def pair_invariants(n, E, G, H, order):
    """(N,N,m) invariants of the pair direction n (N,N,3) and the field tensors at atom i (broadcast over j)."""
    En = jnp.einsum("ia,ija->ij", E, n)
    EE = jnp.sum(E * E, axis=-1)[:, None] * jnp.ones_like(En)
    inv = [EE, En ** 2]
    if order >= 1:
        trG = jnp.einsum("iaa->i", G)[:, None]
        nGn = jnp.einsum("ija,iab,ijb->ij", n, G, n)
        EGn = jnp.einsum("ia,iab,ijb->ij", E, G, n)
        inv += [En * trG, En * nGn, EGn]
    if order >= 2:
        lapE = jnp.einsum("ibbd->id", H)
        ElapE = jnp.sum(E * lapE, axis=-1)[:, None]
        lapEn = jnp.einsum("id,ijd->ij", lapE, n)
        EHnn = jnp.einsum("ib,ijc,ijd,ibcd->ij", E, n, n, H)
        nnnH = jnp.einsum("ijb,ijc,ijd,ibcd->ij", n, n, n, H)
        GG = jnp.einsum("iab,ibc->iac", G, G)
        trG2 = (trG ** 2) * jnp.ones_like(En)
        GdG = jnp.sum(G * G, axis=(1, 2))[:, None] * jnp.ones_like(En)
        nGGn = jnp.einsum("ija,iab,ijb->ij", n, GG, n)
        inv += [ElapE * jnp.ones_like(En), En * lapEn, EHnn, En * nnnH, trG2, GdG, nGn * trG, nGn ** 2, nGGn]
    return jnp.stack(inv, axis=-1)


def n_pair_invariants(order):
    return 2 + (3 if order >= 1 else 0) + (9 if order >= 2 else 0)


def init_readout_pair(key, cfg, order):
    F = cfg["F"]; m = n_pair_invariants(order)
    k = jax.random.split(key, 3)
    nr = cfg.get("n_rbf_readout", cfg["n_rbf"])
    theta = _mlp_init(k[0], [2 * F, F, m * nr], scale=0.1)                           # radial coefficients per invariant from (s_i, s_j)
    n0 = 2 * nr                                                                      # the order-0 invariants' columns
    last = theta[-1]
    theta[-1] = {"w": last["w"].at[:, n0:].set(0.0), "b": last["b"].at[n0:].set(0.0)}   # orders >= 1 start at zero
    return {"theta": theta,
            "iso": _mlp_init(k[1], [F, F, 1], scale=0.1)}                           # node-level a(s_i) |E_i|^2


def response_energy_pair(p, cfg, order, s, r, D, pmask, mask, E, gE, ggE, l):
    """per-atom response energies (N,) of the pair read-out."""
    from .e0 import rbf
    N, F = s.shape
    m = n_pair_invariants(order)
    within = pmask & (r < cfg["r_cut"])
    n = D / r[..., None]
    nr = cfg.get("n_rbf_readout", cfg["n_rbf"])
    basis, _ = rbf(r, {**cfg, "n_rbf": nr})
    basis = basis * within[..., None]                                                   # (N,N,nr)
    sij = jnp.concatenate([jnp.broadcast_to(s[:, None, :], (N, N, F)), jnp.broadcast_to(s[None, :, :], (N, N, F))], -1)
    theta = mlp(p["theta"], sij).reshape(N, N, m, nr)                                   # (N,N,m,nr)
    w = jnp.einsum("ijmn,ijn->ijm", theta, basis)                                       # (N,N,m) radial weights
    G = gE * l if order >= 1 else None
    H = ggE * l ** 2 if order >= 2 else None
    inv = pair_invariants(n, E, G, H, order)                                            # (N,N,m)
    eps = jnp.sum(w * inv * within[..., None], axis=(1, 2))
    eps = eps - 0.5 * mlp(p["iso"], s)[..., 0] * jnp.sum(E * E, axis=-1)
    return eps * mask


# ---------------------------------------------------------------- node read-out (L <= 2 variant) --------------
def band_field_tensors_pbc(positions, q, mask, cell, l, order, n_max=8, b_max=4.0, rs=0.0, D=None, pmask=None):
    """the same tensors for a periodic cell: the Ewald potential V(x) = (4 pi / V) sum_{k != 0} e^{-k^2 l^2/4} / k^2
    Re[S(k) e^{-i k.x}] (tin-foil, neutral), differentiated in x, with the own charge's contribution removed
    analytically (its field at its own centre is zero; its gradient is -q_i grad grad g(0) = q_i (4 / 3 sqrt(pi) l^3) I;
    its third derivative at the centre vanishes)."""
    import math
    from .ewald import k_vectors, _volume
    k = k_vectors(cell, n_max)
    k2 = jnp.sum(k * k, axis=-1)
    live = (k2 > 1e-12) & (k2 <= (2.0 * b_max / l) ** 2)
    k2s = jnp.where(live, k2, 1.0)
    qm = q * mask
    phase = k @ positions.T                                                   # (M,N)
    S_re = jnp.sum(qm[None, :] * jnp.cos(phase), axis=1); S_im = jnp.sum(qm[None, :] * jnp.sin(phase), axis=1)
    pref = jnp.where(live, 4.0 * math.pi / _volume(cell) * jnp.exp(-0.25 * k2s * l * l) / k2s, 0.0)
    def V(x):
        kx = k @ x
        return jnp.sum(pref * (S_re * jnp.cos(kx) + S_im * jnp.sin(kx)))     # Re[S e^{-ikx}] = S_re cos + S_im sin
    def E(x):
        return -jax.grad(V)(x)
    Ef = jax.vmap(E)(positions) * mask[:, None]
    gE = ggE = None
    if order >= 1:
        gE = jax.vmap(jax.jacfwd(E))(positions)
        gE = (gE - (qm * 4.0 / (3.0 * math.sqrt(math.pi) * l ** 3))[:, None, None] * jnp.eye(3)[None]) * mask[:, None, None]
    if order >= 2:
        ggE = jax.vmap(jax.jacfwd(jax.jacfwd(E)))(positions) * mask[:, None, None, None]
    if rs > 0:
        # remove the near sources (their minimum images) with the complementary weight exp(-(r/rs)^10), in real space
        assert D is not None and pmask is not None
        def Vn(x, i):
            img = positions[i] - D[i]                                            # nearest images of the sources
            d = jnp.where(pmask[i][:, None], x[None, :] - img, 1.0)
            rr = jnp.sqrt(jnp.sum(d * d, axis=-1))
            w = pmask[i].astype(x.dtype) * (1.0 - near_weight(rr, rs))
            return jnp.sum(w * q * kn.g_long(rr, l))
        En = lambda x, i: -jax.grad(Vn)(x, i)
        idx = jnp.arange(positions.shape[0])
        Ef = Ef - jax.vmap(En)(positions, idx) * mask[:, None]
        if order >= 1:
            gE = gE - jax.vmap(jax.jacfwd(En))(positions, idx) * mask[:, None, None]
        if order >= 2:
            ggE = ggE - jax.vmap(jax.jacfwd(jax.jacfwd(En)))(positions, idx) * mask[:, None, None, None]
    return Ef, gE, ggE


def _slices(F, order):
    """(name, size) of the coefficient blocks, in the order the read-out consumes them."""
    out = [("a", 1), ("b", F), ("bp", F)]
    if order >= 1: out += [("c", F), ("cp", F), ("d", F), ("dp", F)]
    if order >= 2: out += [("e", 1), ("ep", 1), ("epp", 1), ("g", F), ("gp", F), ("gpp", F), ("h", F), ("hp", F)]
    return out


def init_readout(key, cfg, order):
    F = cfg["F"]
    sl = _slices(F, order)
    n_out = sum(n for _, n in sl)
    p = _mlp_init(key, [F, F, n_out], scale=0.1)
    n0 = sum(n for _, n in sl[:3])                                           # the order-0 block
    last = p[-1]
    p[-1] = {"w": last["w"].at[:, n0:].set(0.0), "b": last["b"].at[n0:].set(0.0)}
    return {"coef": p}


def response_energy(p, cfg, order, s, v, t, mask, E, gE, ggE, l):
    """per-atom response energies (N,) from the band-field tensors, degree 2 in the field."""
    F = cfg["F"]
    c = mlp(p["coef"], s)
    blocks = {}
    k = 0
    for name, n in _slices(F, order):
        blocks[name] = c[:, k] if n == 1 else c[:, k:k + n]
        k += n
    Ev = jnp.einsum("ia,iaf->if", E, v)                            # (N,F)  E . v_f
    EtE = jnp.einsum("ia,iabf,ib->if", E, t, E)                    # E . t_f . E
    eps = -0.5 * (blocks["a"] * jnp.sum(E * E, axis=-1) + jnp.sum(blocks["b"] * Ev ** 2, axis=-1)
                  + jnp.sum(blocks["bp"] * EtE, axis=-1))
    if order >= 1:
        G = gE * l
        trG = jnp.einsum("iaa->i", G)
        EGv = jnp.einsum("ia,iab,ibf->if", E, G, v)
        vGv = jnp.einsum("iaf,iab,ibf->if", v, G, v)
        tG = jnp.einsum("iabf,iab->if", t, G)
        eps = eps + jnp.sum(blocks["c"] * Ev * trG[:, None] + blocks["cp"] * EGv + blocks["d"] * Ev * vGv
                            + blocks["dp"] * Ev * tG, axis=-1)
    if order >= 2:
        H = ggE * l ** 2
        lapE = jnp.einsum("ibbd->id", H)                            # (N,3)   l^2 lap E
        ElapE = jnp.sum(E * lapE, axis=-1)
        GG = jnp.einsum("iab,ibc->iac", G, G)
        EvvH = jnp.einsum("id,ibf,icf,ibcd->if", E, v, v, H)
        lapEv = jnp.einsum("id,idf->if", lapE, v)
        EtH = jnp.einsum("id,ibcf,ibcd->if", E, t, H)
        vGGv = jnp.einsum("iaf,iab,ibf->if", v, GG, v)
        tGG = jnp.einsum("iabf,iab->if", t, GG)
        eps = eps + blocks["e"] * ElapE + blocks["ep"] * trG ** 2 + blocks["epp"] * jnp.sum(G * G, axis=(1, 2))
        eps = eps + jnp.sum(blocks["g"] * EvvH + blocks["gp"] * Ev * lapEv + blocks["gpp"] * EtH
                            + blocks["h"] * vGGv + blocks["hp"] * tGG, axis=-1)
    return eps * mask
