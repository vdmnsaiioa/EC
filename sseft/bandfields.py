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
coefficients of the read-out start at zero (every order), so that training begins from the model without the
response and the read-out switches on as the data ask for it: with the field tensors in units of E_ref, a random
initialisation produced energies of the order of 10^2 e_scale on configurations with strong fields (an ion at 2 A)
and the fit collapsed (e3r-bandfield-response-preregistration.md section 7).
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


def band_field_tensors(positions, q, pmask, l, order, rs=0.0, src_mask=None):
    """E (N,3), gE (N,3,3), ggE (N,3,3,3) of the charge-channel band potential at each atom (own charge excluded,
    sources closer than rs switched off smoothly, and only the sources selected by src_mask (N,N) -- by default
    every other atom; for molecules the atoms of other fragments: the field that polarises a fragment is the
    field of the others, and the own fragment's near field is E_0's)."""
    src = pmask if src_mask is None else (pmask & src_mask)
    return band_field_tensors_from(positions, positions, q, src, l, order, rs)


def band_field_tensors_from(points, sources, q, src, l, order, rs=0.0):
    """the band field tensors at the points (N,3) of the charges q (S,) at the sources (S,3), source j acting on point
    i where src[i, j] (N,S): the atoms' fields of a source set that holds more than the atoms (a site model's virtual
    sites)."""
    def V(x, i):
        d = jnp.where(src[i][:, None], x[None, :] - sources, 1.0)        # excluded pairs get a finite dummy vector
        rr = jnp.sqrt(jnp.sum(d * d, axis=-1))                              # so that no derivative is taken at r = 0
        w = src[i].astype(x.dtype) * near_weight(rr, rs)
        return jnp.sum(w * q * kn.g_long(rr, l))
    def E(x, i):
        return -jax.grad(V)(x, i)
    idx = jnp.arange(points.shape[0])
    Ef = jax.vmap(E)(points, idx)
    gE = jax.vmap(jax.jacfwd(E))(points, idx) if order >= 1 else None
    ggE = jax.vmap(jax.jacfwd(jax.jacfwd(E)))(points, idx) if order >= 2 else None
    return Ef, gE, ggE


# ---------------------------------------------------------------- E_0's view of the pinned sources (v0.0.14) ----
# E3-QP5: E_0's site energies may read the potential, field and field gradient of the other fragments' pinned sources
# at the atom (Rung.e0_field, heads.field_energy) -- the field-like features E_0 otherwise has to build itself
# (e3qp4-results.md section 4).  Unlike the read-out's band field this input is local: every source's share is tapered
# to zero inside E_0's cutoff, so it never reaches farther than E_0's own graph and the far field is untouched; the
# degree-2 rule of the read-out (no terms linear in the field, which a far field would make degenerate with a dipole
# source) does not apply to it, as it does not to E_0.

def taper(r, r_on, r_off):
    """1 below r_on, 0 beyond r_off, the quintic smoothstep between: value, slope and curvature continuous at both
    ends (C2), so that the forces of a term built on tapered shares are continuous with their first derivatives."""
    x = jnp.clip((r - r_on) / (r_off - r_on), 0.0, 1.0)
    return 1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x ** 2)


def local_field_tensors(points, sources, q, src, l, r_on, r_off, frag_src=None, per_frag=False):
    """the potential (N,), field (N,3) and field gradient (N,3,3) at the points (N,3) of the charges q (S,) at the
    sources (S,3) through the band kernel g = erf(r/l)/r, source j acting on point i where src[i, j] (N,S), each source's
    share weighted by w_ij:
        phi_i = sum_j w_ij q_j g(r_ij),    E_i = -sum_j w_ij q_j g'(r_ij) n_ij,
        G_i = -sum_j w_ij q_j [g''(r_ij) n_ij n_ij + g'(r_ij) / r_ij (1 - n_ij n_ij)],    n_ij = (x_i - x_j) / r_ij,
    analytic.  w_ij = taper(r_ij, r_on, r_off), each source by its own distance (v0.0.14); or, with frag_src (F,S) -- the
    fragment of every source -- every fragment whole (v0.0.15): all of a fragment's sources share one weight at a point,
    the product of their tapers,
        w_ij = prod_{k in frag(j)} taper(r_ik, r_on, r_off),
    1 when every site of the fragment lies within r_on of the point, 0 when any lies beyond r_off -- so that a fragment
    straddling the taper enters with its own charges in their proportions (a neutral molecule neutral), never part
    charged.  The shares are weighted, not the potential: E is not -grad phi within the taper, and the forces need only
    the weights' first derivatives.  Nothing from a source beyond r_off.

    per_frag (v0.0.16, with frag_src): every fragment's share apart as well -- returns (phi, E, G, parts), parts =
    (phi_f (N,F), E_f (N,F,3), G_f (N,F,3,3), reach (N,F)): fragment f's potential, field and field gradient at every
    point, and 1 where f is in reach (a source of f acts on the point and the fragment's weight is not zero), else 0.
    The totals are then the sums of the parts over the fragments, so that where one fragment is in reach the total is its
    part to the last digit."""
    d = points[:, None, :] - sources[None, :, :]
    d = jnp.where(src[..., None], d, 1.0)                         # excluded pairs get a finite dummy vector, so that
    r = jnp.sqrt(jnp.sum(d * d, axis=-1))                         # no derivative is taken at r = 0
    n = d / r[..., None]
    tw = taper(r, r_on, r_off)                                    # (N,S): each source by its own distance
    if frag_src is not None:
        member = jnp.asarray(frag_src).astype(points.dtype) > 0.5  # (F,S)
        tf = jnp.prod(jnp.where(member[None, :, :] & src[:, None, :], tw[:, None, :], 1.0), axis=-1)   # (N,F)
        tw = tf @ member.astype(points.dtype)                     # (N,S): each source its fragment's weight
    elif per_frag:
        raise ValueError("per_frag needs the sources' fragments (frag_src)")
    w = src.astype(points.dtype) * tw * q[None, :]
    dg, d2g = kn.dg_long(r, l), kn.d2g_long(r, l)
    if per_frag:
        nn = n[..., :, None] * n[..., None, :]
        H = d2g[..., None, None] * nn + (dg / r)[..., None, None] * (jnp.eye(3) - nn)
        mf = member.astype(points.dtype)                                                   # (F,S)
        phi_f = jnp.einsum("ij,fj->if", w * kn.g_long(r, l), mf)
        E_f = -jnp.einsum("ija,fj->ifa", (w * dg)[..., None] * n, mf)
        G_f = -jnp.einsum("ijab,fj->ifab", w[..., None, None] * H, mf)
        reach = (((src.astype(points.dtype) @ mf.T) > 0.5) & (tf > 0.0)).astype(points.dtype)
        return jnp.sum(phi_f, axis=1), jnp.sum(E_f, axis=1), jnp.sum(G_f, axis=1), (phi_f, E_f, G_f, reach)
    phi = jnp.sum(w * kn.g_long(r, l), axis=1)
    E = -jnp.einsum("ij,ija->ia", w * dg, n)
    nn = n[..., :, None] * n[..., None, :]
    H = d2g[..., None, None] * nn + (dg / r)[..., None, None] * (jnp.eye(3) - nn)
    G = -jnp.einsum("ij,ijab->iab", w, H)
    return phi, E, G


# ---------------------------------------------------------------- the induced dipoles' mutual fields -----------
# The order-0 read-out is -1/2 E_i . C_i . E_i at every atom: a local polarisability tensor C_i (environment-dependent)
# responding to the band field of the permanent sources -- first order in the polarisabilities.  The induced dipoles
# mu_i = C_i E_i are themselves sources; their fields, through the band kernel's dipole tensor between atoms of different
# fragments, polarise the other fragments again:
#     mu = C (E + T mu)  ->  mu = (1 - C T)^-1 C E,      E_resp = -1/2 E . mu = -1/2 E.C.E - 1/2 E.C T C.E - ...
# the second-order term (alpha^2 T) and beyond; the exact solve is TTM3-F's self-consistency.  T is the Hessian of the
# band kernel erf(r/l)/r, finite at short range (the band edge does what Thole damping does in a polarisable model).

def band_dipole_tensor(positions, pmask, l):
    """T (N,N,3,3): the field at i of a unit dipole at j through the band kernel, grad grad g(r_ij), on the pairs of
    pmask (N,N) and zero elsewhere."""
    D = positions[:, None, :] - positions[None, :, :]
    r = jnp.sqrt(jnp.sum(D * D, axis=-1) + 1e-300)
    r = jnp.where(pmask, r, 1.0)
    n = D / r[..., None]
    nn = n[..., :, None] * n[..., None, :]
    T = kn.d2g_long(r, l)[..., None, None] * nn + (kn.dg_long(r, l) / r)[..., None, None] * (jnp.eye(3) - nn)
    return jnp.where(pmask[..., None, None], T, 0.0)


def mutual_response(C, T, E, order=-1):
    """the response energy -1/2 E . mu of local polarisabilities C (N,3,3) to the field E (N,3), with the induced
    dipoles' mutual fields through T (N,N,3,3): order -1 the self-consistent dipoles (1 - C T)^-1 C E, order k >= 1 the
    Neumann series through k-th order in C (1: first order, -1/2 E.C.E)."""
    N = E.shape[0]
    e = E.reshape(-1)
    c = jnp.einsum("ij,iab->iajb", jnp.eye(N, dtype=C.dtype), C).reshape(3 * N, 3 * N)
    t = T.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
    ce = c @ e
    if order == -1:
        mu = jnp.linalg.solve(jnp.eye(3 * N) - c @ t, ce)
    else:
        mu = ce; term = ce
        for _ in range(order - 1):
            term = c @ (t @ term)
            mu = mu + term
    return -0.5 * jnp.dot(e, mu), mu.reshape(N, 3)


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
    last = theta[-1]
    theta[-1] = {"w": jnp.zeros_like(last["w"]), "b": jnp.zeros_like(last["b"])}      # the read-out starts at zero (every order)
    iso = _mlp_init(k[1], [F, F, 1], scale=0.1)
    iso[-1] = {"w": jnp.zeros_like(iso[-1]["w"]), "b": jnp.zeros_like(iso[-1]["b"])}
    return {"theta": theta, "iso": iso}                                              # node-level a(s_i) |E_i|^2


SPECIES_KEY = 20261016                       # the species read-out's fixed initialisation (seed-independent)
SPECIES_UNIT = 1e-3                          # its output unit (hartree): independent of the site energies' e_scale


def init_readout_species(cfg, order):
    """the pair read-out with coefficients from the species alone: fixed random species features (a table, not E_0's
    features) through a hidden layer of the same width, so that the radial coefficients w_m(r; Z_i, Z_j) and the node
    term a(Z_i) are free functions of the species pair once the output layers are solved (the hidden features of the
    few species pairs are independent).  Deterministic: the response does not depend on the training seed or on E_0, so
    that it can be matched on the far field before E_0 is trained, and stays what was matched."""
    F = cfg["F"]
    k = jax.random.split(jax.random.PRNGKey(SPECIES_KEY), 3)
    p = init_readout_pair(k[0], cfg, order)
    m = n_pair_invariants(order); nr = cfg.get("n_rbf_readout", cfg["n_rbf"])
    theta = _mlp_init(k[1], [2 * F, F, m * nr], scale=1.0)
    theta[-1] = {"w": jnp.zeros_like(theta[-1]["w"]), "b": jnp.zeros_like(theta[-1]["b"])}
    iso = _mlp_init(k[2], [F, F, 1], scale=1.0)
    iso[-1] = {"w": jnp.zeros_like(iso[-1]["w"]), "b": jnp.zeros_like(iso[-1]["b"])}
    embed = jax.random.normal(jax.random.PRNGKey(SPECIES_KEY + 1), (cfg["n_species"], F))
    return {**p, "theta": theta, "iso": iso, "embed": embed}


def response_energy_pair(p, cfg, order, s, r, D, pmask, mask, E, gE, ggE, l, E_ref=1.0, e_scale=1.0, tensor=False):
    """per-atom response energies (N,) of the pair read-out.  The field tensors enter in units of E_ref and the
    output in units of e_scale (the data-derived site-energy scale), so that the learned coefficients are O(1):
    a physical polarisability alpha is the coefficient alpha E_ref^2 / e_scale -- with E_ref = 0.01 a.u. and
    e_scale ~ 5e-3 E_h that is 0.2 for alpha = 9.72 a.u., against 9.72 without the scales (which Adam does not reach
    from an O(0.1) initialisation in fields of 1e-3 a.u.: e3r-bandfield-response-preregistration.md section 6).
    tensor=True returns instead (the per-atom energies of the orders >= 1, the order-0 sector as the local
    polarisability tensor C (N,3,3) in bohr^3): the order-0 energy is -1/2 E_i.C_i.E_i, the same numbers, and the
    mutual response (mutual_response) builds on C."""
    from .e0 import rbf
    N, F = s.shape
    E = E / E_ref
    gE = gE / E_ref if gE is not None else None
    ggE = ggE / E_ref if ggE is not None else None
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
    iso = mlp(p["iso"], s)[..., 0]
    if tensor:
        # -1/2 E.C.E = sum_j (w_0 |E|^2 + w_1 (E.n)^2) - 1/2 iso |E|^2  ->  C = iso 1 - 2 sum_j (w_0 1 + w_1 n n)
        w0 = jnp.sum(w[..., 0] * within, axis=1); nn = n[..., :, None] * n[..., None, :]
        C = iso[:, None, None] * jnp.eye(3) - 2.0 * (w0[:, None, None] * jnp.eye(3)
                                                     + jnp.einsum("ij,ijab->iab", w[..., 1] * within, nn))
        eps_hi = jnp.sum(w[..., 2:] * inv[..., 2:] * within[..., None], axis=(1, 2))
        return eps_hi * mask * e_scale, C * (mask * e_scale / E_ref ** 2)[:, None, None]
    eps = jnp.sum(w * inv * within[..., None], axis=(1, 2))
    eps = eps - 0.5 * iso * jnp.sum(E * E, axis=-1)
    return eps * mask * e_scale


# ---------------------------------------------------------------- node read-out (L <= 2 variant) --------------
def band_field_tensors_pbc(positions, q, mask, cell, l, order, n_max=8, b_max=4.0, rs=0.0, D=None, pmask=None, excl_mask=None):
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
    if rs > 0 or excl_mask is not None:
        # remove the near sources (their minimum images) with the complementary weight exp(-(r/rs)^10), and the
        # excluded sources (the own fragment's atoms, excl_mask (N,N)) completely, in real space
        assert D is not None and pmask is not None
        def Vn(x, i):
            img = positions[i] - D[i]                                            # nearest images of the sources
            d = jnp.where(pmask[i][:, None], x[None, :] - img, 1.0)
            rr = jnp.sqrt(jnp.sum(d * d, axis=-1))
            w = 1.0 - near_weight(rr, rs)
            if excl_mask is not None:
                w = jnp.where(excl_mask[i], 1.0, w)
            w = pmask[i].astype(x.dtype) * w
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
