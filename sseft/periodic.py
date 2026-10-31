"""
The periodic path (E4, notes/e4-plan.md section 5): the site model, the band fields, the self-consistent response and
E_0's view of the pinned field in a cell.

The band kernel is erf(r/l)/r, smooth, so its periodic sum over every image lives entirely in reciprocal space (as in
ewald.py):
    phi(x) = (4 pi / V) sum_{k != 0} e^{-k^2 l^2/4} / k^2 Re[S(k) e^{-i k.x}],      S(k) = sum_j q_j e^{i k.r_j},
with tin-foil boundary conditions (the k = 0 term dropped; neutral cells).

How the sums are taken.  k = 2 pi n B^-T over the integers |n_a| <= n_max, masked by |k| <= k_max = 2 b_max / l, over a
half space (k and -k contribute equally), in chunks of fixed size: a lax.scan whose body is rematerialised in the backward
pass, so that memory is O(chunk x sources) at any cell size.  n_max must reach k_max along every lattice vector
(n_max_for); with a smaller n_max every sum here returns NaN rather than a truncated value.

What the periodic model computes (model.energy_single):
- The interaction energy against isolated fragments.  The periodic energy of the whole cell holds every pair over every
  image, each molecule's own pairs included (as the open model's E(all) does), and each fragment alone is subtracted,
  open.  So a molecule's interactions with its own images stay in -- the truth's (MBX) are in its labels -- and its own
  pairs at n = 0 cancel.
- The band field the response reads, and the response's dipole tensor, leave out the own fragment at n = 0 only: a
  molecule is polarised by everything else, its own images included (in the open model: by the other fragments).
- E_0's view of the pinned field (Rung.e0_field) is local: every other fragment whole at its image nearest the atom
  (the minimum image through the fragment's first atom), tapered to zero within E_0's cutoff.
- Molecules are first made whole (each fragment's atoms at the images nearest its first atom), so that virtual sites
  and a fragment's own terms are the geometry of the molecule wherever its atoms were wrapped.

The grouped layout.  A fragment's sources are its atoms (slots 0..A-1, from structure.pad_batch's frag_atoms) and then
its virtual sites (slots A.., frag_sites): src_X (F, P, 3), src_q (F, P), src_on (F, P).  Atom i is slot atom_slot[i]
of fragment atom_frag[i].
"""
import math
import numpy as np
import jax
import jax.numpy as jnp
from . import kernels as kn

SQRT_PI = math.sqrt(math.pi)


# ---------------------------------------------------------------- reciprocal space -------------------------------
def n_max_for(cell, l, b_max=4.0):
    """the smallest n_max whose integer grid holds every k with |k| <= 2 b_max / l, for a cell (3, 3) or a stack of
    cells (..., 3, 3), rows the lattice vectors: |n_a| = |a_a . k| / 2 pi <= |a_a| k_max / 2 pi."""
    cell = np.asarray(cell, float)
    kmax = 2.0 * b_max / l
    return int(np.ceil(np.max(np.linalg.norm(cell, axis=-1)) * kmax / (2.0 * math.pi) - 1e-9))


def covered(cell, l, b_max, n_max):
    """True where the grid |n_a| <= n_max holds the sphere |k| <= 2 b_max / l (traced)."""
    kmax = 2.0 * b_max / l
    return jnp.all(jnp.linalg.norm(cell, axis=-1) * kmax <= 2.0 * math.pi * n_max + 1e-9)


def cell_widths(cell):
    """the distances between opposite faces (traced): the minimum image holds within half the smallest."""
    V = jnp.abs(jnp.linalg.det(cell))
    return jnp.stack([V / jnp.linalg.norm(jnp.cross(cell[(a + 1) % 3], cell[(a + 2) % 3])) for a in range(3)])


def _chunk(n_src, chunk=0):
    """the number of k per chunk: about 2^21 phase entries per chunk (16 MB in float64)."""
    if chunk:
        return int(chunk)
    return int(max(64, min(4096, 2 ** 21 // max(int(n_src), 1))))


def _kgrid(n_max, chunk):
    """(number of chunks, chunk c -> (integer vectors (chunk, 3), half-space weights (chunk,))).  The weights are 2 on the
    half space n_x > 0, or n_x = 0 and n_y > 0, or n_x = n_y = 0 and n_z > 0, and 0 on the other half, at n = 0 and
    beyond the grid; the integers are decoded from a flat index, so that no grid is stored."""
    G1 = 2 * n_max + 1
    total = (n_max + 1) * G1 * G1
    n_chunks = -(-total // chunk)

    def ints(c):
        f = c * chunk + jnp.arange(chunk)
        nx = f // (G1 * G1); rem = f % (G1 * G1)
        ny = rem // G1 - n_max; nz = rem % G1 - n_max
        half = (nx > 0) | ((nx == 0) & ((ny > 0) | ((ny == 0) & (nz > 0))))
        return jnp.stack([nx, ny, nz], axis=-1), jnp.where((f < total) & half, 2.0, 0.0)
    return n_chunks, ints


def ksum(body, cell, l, n_max, b_max, chunk, init):
    """sum over the half space of body(k (C,3), k2 (C,), w (C,)), w the half-space weights times |k| <= k_max (zero
    elsewhere: a body must give nothing where w = 0).  init fixes the pytree of the accumulated value."""
    n_chunks, ints = _kgrid(n_max, chunk)
    recip = 2.0 * math.pi * jnp.linalg.inv(cell).T
    kmax2 = (2.0 * b_max / l) ** 2

    @jax.checkpoint
    def part(c):
        n, w = ints(c)
        k = n.astype(cell.dtype) @ recip
        k2 = jnp.sum(k * k, axis=-1)
        return body(k, k2, jnp.where(k2 <= kmax2, w, 0.0))

    def step(acc, c):
        return jax.tree_util.tree_map(jnp.add, acc, part(c)), None
    acc, _ = jax.lax.scan(step, init, jnp.arange(n_chunks))
    return acc


def _volume(cell):
    return jnp.abs(jnp.linalg.det(cell))


def _guard(x, cell, l, b_max, n_max):
    ok = covered(cell, l, b_max, n_max)
    return jax.tree_util.tree_map(lambda a: jnp.where(ok, a, jnp.nan), x)


# ---------------------------------------------------------------- the bands -------------------------------------
def coulomb_pbc(X, q, on, cell, l, n_max, b_max=4.0, mu=None, chunk=0):
    """the p = 1 band of charges q (S,) [and dipoles mu (S, 3)] at X (S, 3), the sources where on (S,):
        E = (2 pi / V) sum_{k != 0} e^{-k^2 l^2/4} / k^2 |S(k)|^2 - sum_i [q_i^2 / (sqrt(pi) l) + 2 |mu_i|^2 / (3 sqrt(pi) l^3)],
    S(k) = sum_j (q_j + i k.mu_j) e^{i k.r_j} -- ewald.coulomb_band_pbc, chunked over a half space."""
    qm = q * on
    mm = None if mu is None else mu * on[:, None]

    def body(k, k2, w):
        k2s = jnp.where(w > 0, k2, 1.0)
        ph = k @ X.T
        c, s = jnp.cos(ph), jnp.sin(ph)
        Sr, Si = c @ qm, s @ qm
        if mm is not None:
            km = k @ mm.T                                          # (C, S): k . mu_j
            Sr = Sr - jnp.sum(km * s, axis=1); Si = Si + jnp.sum(km * c, axis=1)
        return jnp.sum(w * jnp.exp(-0.25 * k2s * l * l) / k2s * (Sr * Sr + Si * Si))
    acc = ksum(body, cell, l, n_max, b_max, _chunk(X.shape[0], chunk), jnp.zeros((), X.dtype))
    E = (2.0 * math.pi / _volume(cell)) * acc - jnp.sum(qm * qm) / (SQRT_PI * l)
    if mm is not None:
        E = E - 2.0 * jnp.sum(mm * mm) / (3.0 * SQRT_PI * l ** 3)
    return _guard(E, cell, l, b_max, n_max)


def _ghat(p):
    from .ewald import ghat6, ghat8
    return ghat6 if p == 6 else ghat8


def dispersion_pbc(Bi, Bj, X, on, cell, l, p, n_max, b_max=4.0, chunk=0):
    """E = -(1/2V) sum_k ghat_p(k) sum_kappa Re[S_i^kappa(k) conj S_j^kappa(k)] + (1/2) sum_i C_ii g_p(0), the k = 0 term
    included -- ewald._dispersion_pbc, chunked over a half space; Bi, Bj (S, K)."""
    ghat = _ghat(p)
    g0 = (1.0 / l) ** 6 / 6.0 if p == 6 else (1.0 / l) ** 8 / 24.0
    Bi = Bi * on[:, None]; Bj = Bj * on[:, None]

    def body(k, k2, w):
        k2s = jnp.where(w > 0, k2, 1.0)
        ph = k @ X.T
        c, s = jnp.cos(ph), jnp.sin(ph)
        SS = jnp.sum((c @ Bi) * (c @ Bj) + (s @ Bi) * (s @ Bj), axis=-1)
        return jnp.sum(w * ghat(k2s, l) * SS)
    acc = ksum(body, cell, l, n_max, b_max, _chunk(X.shape[0], chunk), jnp.zeros((), X.dtype))
    acc = acc + ghat(jnp.zeros((), X.dtype), l) * jnp.sum(jnp.sum(Bi, axis=0) * jnp.sum(Bj, axis=0))   # k = 0
    E = -(0.5 / _volume(cell)) * acc + 0.5 * jnp.sum(Bi * Bj) * g0
    return _guard(E, cell, l, b_max, n_max)


def dispersion_band_pbc(alpha, w, X, on, cell, l, n_max, b_max=4.0, chunk=0):
    """the p = 6 band, C6_ij = (3/pi) sum_k w_k alpha_i alpha_j (ewald.dispersion_band_pbc)."""
    B = jnp.sqrt(3.0 * w / math.pi)[None, :] * alpha
    return dispersion_pbc(B, B, X, on, cell, l, 6, n_max, b_max, chunk)


def dispersion8_band_pbc(alpha1, alpha2, w, X, on, cell, l, n_max, b_max=4.0, chunk=0):
    """the p = 8 band (ewald.dispersion8_band_pbc)."""
    f = jnp.sqrt(15.0 * w / (2.0 * math.pi))[None, :]
    return 2.0 * dispersion_pbc(f * alpha1, f * alpha2, X, on, cell, l, 8, n_max, b_max, chunk)


# ---------------------------------------------------------------- band fields -----------------------------------
def field_tensors_pbc(points, X, q, on, cell, l, order, n_max, b_max=4.0, chunk=0):
    """E (N,3), grad E (N,3,3) and grad grad E (N,3,3,3) at the points of the charges q (S,) at X (S,3) and all their
    images, nothing left out (a source at a point included; the callers remove what they exclude):
        E = (4 pi/V) sum_k f(k) k [S_re sin(k.x) - S_im cos(k.x)],   grad E = (4 pi/V) sum_k f k k [S_re cos + S_im sin],
        grad grad E = -(4 pi/V) sum_k f k k k [S_re sin - S_im cos],    f = e^{-k^2 l^2/4} / k^2."""
    qm = q * on
    N = points.shape[0]

    def body(k, k2, w):
        k2s = jnp.where(w > 0, k2, 1.0)
        f = w * jnp.exp(-0.25 * k2s * l * l) / k2s
        ph = k @ X.T
        Sr, Si = jnp.cos(ph) @ qm, jnp.sin(ph) @ qm
        pp = k @ points.T                                           # (C, N)
        c, s = jnp.cos(pp), jnp.sin(pp)
        a = f[:, None] * (Sr[:, None] * s - Si[:, None] * c)
        out = {"E": jnp.einsum("ci,ca->ia", a, k)}
        if order >= 1:
            out["G"] = jnp.einsum("ci,ca,cb->iab", f[:, None] * (Sr[:, None] * c + Si[:, None] * s), k, k)
        if order >= 2:
            out["H"] = -jnp.einsum("ci,ca,cb,cd->iabd", a, k, k, k)
        return out
    init = {"E": jnp.zeros((N, 3), points.dtype)}
    if order >= 1:
        init["G"] = jnp.zeros((N, 3, 3), points.dtype)
    if order >= 2:
        init["H"] = jnp.zeros((N, 3, 3, 3), points.dtype)
    acc = ksum(body, cell, l, n_max, b_max, _chunk(max(X.shape[0], N), chunk), init)
    pref = 4.0 * math.pi / _volume(cell)
    acc = _guard(jax.tree_util.tree_map(lambda a: pref * a, acc), cell, l, b_max, n_max)
    return acc["E"], acc.get("G"), acc.get("H")


def near_tensors(points, Y, q, on, l, order):
    """the field tensors at every point i of a few sources of its own, Y (N, P, 3) with charges q (N, P) where on (N, P),
    directly (n = 0) by nested forward-mode autodiff (bandfields.band_field_tensors_from, per point)."""
    def V(x, y, qq, oo):
        d = jnp.where(oo[:, None], x[None, :] - y, 1.0)            # an excluded source gets a finite dummy vector
        rr = jnp.sqrt(jnp.sum(d * d, axis=-1))
        return jnp.sum(oo * qq * kn.g_long(rr, l))

    def E(x, y, qq, oo):
        return -jax.grad(V)(x, y, qq, oo)
    Ef = jax.vmap(E)(points, Y, q, on)
    gE = jax.vmap(jax.jacfwd(E))(points, Y, q, on) if order >= 1 else None
    ggE = jax.vmap(jax.jacfwd(jax.jacfwd(E)))(points, Y, q, on) if order >= 2 else None
    return Ef, gE, ggE


def site_band_field_pbc(points, mask, X, q, on, own_Y, own_q, own_on, q_self, cell, l, order, n_max, b_max=4.0, chunk=0):
    """the band field tensors at the atoms (points) of a site model's charges in a cell, every source and image except
    the atom's own fragment at n = 0: the reciprocal sums less the own fragment's other sources (own_Y, own_q, own_on
    (N, P), directly) and less the atom's own source analytically (its field at its centre is zero, its gradient
    q_self (4 / 3 sqrt(pi) l^3) 1, its second gradient zero)."""
    Ef, gE, ggE = field_tensors_pbc(points, X, q, on, cell, l, order, n_max, b_max, chunk)
    Eo, gEo, ggEo = near_tensors(points, own_Y, own_q, own_on, l, order)
    Ef = (Ef - Eo) * mask[:, None]
    if order >= 1:
        self_g = (q_self * 4.0 / (3.0 * SQRT_PI * l ** 3))[:, None, None] * jnp.eye(3)
        gE = (gE - gEo - self_g) * mask[:, None, None]
    if order >= 2:
        ggE = (ggE - ggEo) * mask[:, None, None, None]
    return Ef, gE, ggE


# ---------------------------------------------------------------- the response's dipole tensor -------------------
def _T_direct(d, l):
    """grad grad g(r) for displacements d (..., 3): the field of a unit dipole through the band kernel."""
    r = jnp.sqrt(jnp.sum(d * d, axis=-1))
    n = d / r[..., None]
    nn = n[..., :, None] * n[..., None, :]
    return kn.d2g_long(r, l)[..., None, None] * nn + (kn.dg_long(r, l) / r)[..., None, None] * (jnp.eye(3) - nn)


def own_dipole_field(mu, X, own_idx, own_on, l):
    """the field at every atom of its own fragment's other induced dipoles (n = 0): sum_a T(r_i - r_own(a)) mu_own(a)."""
    d = X[:, None, :] - X[own_idx]                                       # (N, A, 3)
    d = jnp.where(own_on[..., None], d, 1.0)
    T = jnp.where(own_on[..., None, None], _T_direct(d, l), 0.0)
    return jnp.einsum("iaxy,iay->ix", T, mu[own_idx])


def dipole_field_pbc(mu, X, mask, cell, l, n_max, b_max=4.0, chunk=0):
    """the field at every atom of every induced dipole and image through the band kernel, the own dipole at n = 0
    included:  E(x) = -(4 pi / V) sum_k f(k) k [cos(k.x) A_k + sin(k.x) B_k],  A_k + i B_k = sum_j (k.mu_j) e^{i k.r_j}."""
    mm = mu * mask[:, None]
    N = X.shape[0]

    def body(k, k2, w):
        k2s = jnp.where(w > 0, k2, 1.0)
        f = w * jnp.exp(-0.25 * k2s * l * l) / k2s
        ph = k @ X.T
        c, s = jnp.cos(ph), jnp.sin(ph)
        km = k @ mm.T                                                     # (C, N): k . mu_j
        A, B = jnp.sum(km * c, axis=1), jnp.sum(km * s, axis=1)
        return jnp.einsum("ci,ca->ia", f[:, None] * (c * A[:, None] + s * B[:, None]), k)
    acc = ksum(body, cell, l, n_max, b_max, _chunk(N, chunk), jnp.zeros((N, 3), X.dtype))
    return _guard(-(4.0 * math.pi / _volume(cell)) * acc, cell, l, b_max, n_max)


def dipole_matvec_pbc(mu, X, mask, own_idx, own_on, cell, l, n_max, b_max=4.0, chunk=0):
    """T mu in a cell: the field of every other induced dipole and every image, the own fragment at n = 0 left out (its
    own dipole: T(0) = -(4 / 3 sqrt(pi) l^3) 1, analytically; its other atoms directly)."""
    E = dipole_field_pbc(mu, X, mask, cell, l, n_max, b_max, chunk)
    E = E + (4.0 / (3.0 * SQRT_PI * l ** 3)) * mu * mask[:, None] - own_dipole_field(mu * mask[:, None], X, own_idx, own_on, l)
    return E * mask[:, None]


def dipole_tensor_pbc(X, mask, own_idx, own_on, cell, l, n_max, b_max=4.0, chunk=0):
    """T (N, N, 3, 3) explicitly, the same operator as dipole_matvec_pbc:
        T_ij = -(4 pi / V) sum_k f(k) k k^T cos(k.(r_i - r_j)),  less T(0) on the diagonal and the own fragment at n = 0.
    Cost O(N_k N^2): for the direct solve of small cells."""
    N = X.shape[0]

    def body(k, k2, w):
        k2s = jnp.where(w > 0, k2, 1.0)
        f = w * jnp.exp(-0.25 * k2s * l * l) / k2s
        ph = k @ X.T
        Ac = jnp.cos(ph)[:, :, None] * k[:, None, :]
        As = jnp.sin(ph)[:, :, None] * k[:, None, :]
        return jnp.einsum("c,cia,cjb->iajb", f, Ac, Ac) + jnp.einsum("c,cia,cjb->iajb", f, As, As)
    acc = ksum(body, cell, l, n_max, b_max, _chunk(N, chunk), jnp.zeros((N, 3, N, 3), X.dtype))
    T = -(4.0 * math.pi / _volume(cell)) * acc.transpose(0, 2, 1, 3)       # (N, N, 3, 3)
    T = T + (4.0 / (3.0 * SQRT_PI * l ** 3)) * jnp.eye(N, dtype=X.dtype)[:, :, None, None] * jnp.eye(3)
    d = jnp.where(own_on[..., None], X[:, None, :] - X[own_idx], 1.0)
    T_own = jnp.where(own_on[..., None, None], _T_direct(d, l), 0.0)          # (N, A, 3, 3)
    T = T.at[jnp.arange(N)[:, None], own_idx].add(-T_own)
    T = T * (mask[:, None] & mask[None, :])[..., None, None]
    return _guard(T, cell, l, b_max, n_max)


def mutual_response_pbc(C, E, X, mask, own_idx, own_on, cell, l, n_max, b_max=4.0, order=-1, solver="direct",
                        tol=1e-10, maxiter=50, restart=30, chunk=0):
    """the response energy -1/2 E . mu of local polarisabilities C (N,3,3) to the field E (N,3) in a cell, with the
    induced dipoles' mutual fields through the periodic dipole tensor (the own fragment at n = 0 left out):
    mu = (1 - C T)^-1 C E (order -1), or the series through order k (bandfields.mutual_response).
      solver "direct": T explicit (dipole_tensor_pbc) and a dense solve -- small cells.
      solver "gmres": matrix-free (dipole_matvec_pbc), restarted GMRES to the relative residual tol, differentiated
      implicitly (lax.custom_linear_solve: the derivatives solve the transposed system), order -1 only."""
    from . import bandfields as BF
    if solver == "direct":
        T = dipole_tensor_pbc(X, mask, own_idx, own_on, cell, l, n_max, b_max, chunk)
        return BF.mutual_response(C, T, E, order)
    if solver != "gmres" or order != -1:
        raise ValueError("the matrix-free solve (solver 'gmres') is the self-consistent response (order -1); "
                         "'direct' takes any order")
    from jax.scipy.sparse.linalg import gmres
    Tmv = lambda x: dipole_matvec_pbc(x, X, mask, own_idx, own_on, cell, l, n_max, b_max, chunk)
    Cx = lambda x: jnp.einsum("iab,ib->ia", C, x)
    rhs = Cx(E)

    def matvec(x):
        return x - Cx(Tmv(x))

    def solve(mv, b):
        return gmres(mv, b, tol=tol, atol=0.0, restart=restart, maxiter=maxiter)[0]
    mu = jax.lax.custom_linear_solve(matvec, rhs, solve, transpose_solve=solve)
    return -0.5 * jnp.sum(E * mu), mu


# ---------------------------------------------------------------- molecules whole, sources grouped ---------------
def make_whole(pos, atom_frag, frag_atoms, cell):
    """every fragment's atoms at the images nearest its first atom (a no-op for whole molecules)."""
    ref = pos[frag_atoms[:, 0]][atom_frag]
    n = jnp.round((pos - ref) @ jnp.linalg.inv(cell))
    return pos - jax.lax.stop_gradient(n) @ cell


def grouped_sources(pos, b, q_site=None):
    """(src_X (F, P, 3), src_q (F, P), src_on (F, P)) of every fragment: its atoms (charges q_site, b["pin_q_site"] by
    default) and then its virtual sites (b["vs_w"] @ pos, b["vs_q"]), from structure.pad_batch's grouping."""
    fa, fam = b["frag_atoms"], b["frag_atoms_mask"]
    q_site = b["pin_q_site"] if q_site is None else q_site
    X = [pos[fa]]; Q = [q_site[fa] * fam]; on = [fam & b["mask"][fa]]
    if "frag_sites" in b and "vs_w" in b:
        fs, fsm = b["frag_sites"], b["frag_sites_mask"] & b["vs_mask"][b["frag_sites"]]
        sites = b["vs_w"] @ pos
        X.append(sites[fs]); Q.append(b["vs_q"][fs] * fsm); on.append(fsm)
    return jnp.concatenate(X, axis=1), jnp.concatenate(Q, axis=1), jnp.concatenate(on, axis=1)


def own_fragment(b, P=None):
    """per atom: its fragment's other atoms (own_idx (N, A), own_on (N, A)) -- for the dipole tensor -- and, with P the
    number of source slots, the source slots of its fragment other than its own atom's (on (N, P))."""
    fa, fam, af, slot, mask = b["frag_atoms"], b["frag_atoms_mask"], b["atom_frag"], b["atom_slot"], b["mask"]
    A = fa.shape[1]
    own_idx = fa[af]
    own_on = fam[af] & (jnp.arange(A)[None, :] != slot[:, None]) & mask[:, None] & mask[own_idx]
    if P is None:
        return own_idx, own_on
    slots_on = (jnp.arange(P)[None, :] != slot[:, None]) & mask[:, None]
    return own_idx, own_on, slots_on


# ---------------------------------------------------------------- E_0's view of the pinned field, in a cell ------
def local_field_pbc(points, atom_frag, mask, src_X, src_q, src_on, cell, l, r_on, r_off, per_frag=False, self_taper=None):
    """bandfields.local_field_tensors with every fragment whole (frag_src) in a cell: at every atom the potential,
    field and field gradient of every other fragment's sources, the fragment at its image nearest the atom (the
    minimum image of its first source, an atom) and weighted as one, the product of its sites' tapers.  The same
    returns: (phi, E, G), or with per_frag (phi, E, G, parts), parts = (phi_f, E_f, G_f, reach[, w_self]) by fragment.
    Holds for cells at least 2 r_off wide (a fragment within reach through one image only)."""
    F = src_q.shape[0]
    ref = src_X[:, 0, :]
    n_img = jnp.round((points[:, None, :] - ref[None, :, :]) @ jnp.linalg.inv(cell))          # (N, F, 3)
    shift = jax.lax.stop_gradient(n_img) @ cell
    d = points[:, None, None, :] - src_X[None, :, :, :] - shift[:, :, None, :]                 # (N, F, P, 3)
    other = atom_frag[:, None] != jnp.arange(F)[None, :]
    act = mask[:, None, None] & other[:, :, None] & src_on[None, :, :]                       # (N, F, P)
    d = jnp.where(act[..., None], d, 1.0)
    r = jnp.sqrt(jnp.sum(d * d, axis=-1))
    n = d / r[..., None]
    from .bandfields import taper
    tf = jnp.prod(jnp.where(act, taper(r, r_on, r_off), 1.0), axis=-1)                        # (N, F)
    w = act * tf[..., None] * src_q[None, :, :]
    g, dg, d2g = kn.g_long(r, l), kn.dg_long(r, l), kn.d2g_long(r, l)
    phi_f = jnp.sum(w * g, axis=-1)
    E_f = -jnp.einsum("ifp,ifpa->ifa", w * dg, n)
    G_f = -(jnp.einsum("ifp,ifpa,ifpb->ifab", w * (d2g - dg / r), n, n)
            + jnp.sum(w * dg / r, axis=-1)[..., None, None] * jnp.eye(3))
    phi, E, G = jnp.sum(phi_f, axis=1), jnp.sum(E_f, axis=1), jnp.sum(G_f, axis=1)
    if self_taper is not None and not per_frag:
        raise ValueError("self_taper weights every fragment's share apart: it needs per_frag")
    if not per_frag:
        return phi, E, G
    reach = (jnp.any(act, axis=-1) & (tf > 0.0)).astype(points.dtype)
    parts = (phi_f, E_f, G_f, reach)
    if self_taper is not None:
        ts = jnp.prod(jnp.where(act, taper(r, self_taper[0], self_taper[1]), 1.0), axis=-1)
        parts = parts + (reach * ts,)
    return phi, E, G, parts
