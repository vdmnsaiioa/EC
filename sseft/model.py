"""
The model: E_0 + IR bands on learned (or pinned) sources [+ band-field inputs], assembled per rung.

    rung    bands                sources           band fields    stands for
    M_inf   none                 --                --             the incumbent (cutoff model)
    M_A     analytic pair kernel scalar s_i        --             learned kernel analytic in q^2: branch 3 (x1-prediction.md)
    M_1     Coulomb              q learned         --             LES / 4G-type: Coulomb added, no dispersion band
    M_6     dispersion           alpha learned     --             EFT leading order, neutral fragments
    M_16    Coulomb + dispersion q, alpha learned  --             both bands, charges free
    M_16p   Coulomb + dispersion pinned            --             the same architecture with sources from an independent calculation (dagger)
    M_S     as M_16p (or M_6)    pinned/learned    order k        v1.5: band fields read at the atom, degree 2
    M_G     global pair block    --                --             the null: an unconstrained function of r with no envelope
Pinned sources may be a site model (Rung.msites: charges on the atoms and on virtual sites placed from them, e.g. TTM3-F's
M-site), and the order-0 read-out's induced dipoles may act on each other (Rung.bf_mutual: the alpha^2 T terms and beyond).
E_0's site energies may read the pinned sources' local potential, field and field gradient (Rung.e0_field, v0.0.14:
bandfields.local_field_tensors, heads.field_energy), tapered to zero inside E_0's cutoff -- every fragment's sources with
one weight (Rung.e0_field_whole, v0.0.15), and through a term of degree 2 in the field (Rung.e0_field_form "quadratic",
heads.field_energy2) -- or the term less its value on each source fragment alone (Rung.e0_field_cross, v0.0.16): nothing
to any dimer, only what the fragments' fields do together; or (v0.0.17) with its self terms at a shorter reach than its
cross terms (Rung.e0_field_self_on / _off, heads.field_energy_split), and its cross terms at degree 2 with coefficients
by species alone beside the MLP's self terms (Rung.e0_field_cross_species, heads.field_cross_species).
In a cell (Rung.periodic, v0.0.20: sseft/periodic.py) the site model, the band fields, the self-consistent response and
the field term are periodic, and the interaction energy is taken against every fragment alone, open: a molecule keeps
its interactions with its own images.  A uniform external field b["E_ext"] adds to the band field the read-out
responds to (clamped nuclei: dielectric_inf).
Energies in hartree, positions in bohr; forces by autodiff.
"""
from dataclasses import dataclass, field, replace
import jax
import jax.numpy as jnp
from . import e0 as E0, heads as H, bands as B, bandfields as BF, kernels as kn, ewald as EW, periodic as PB
from .structure import pair_geometry
from .units import ang_to_bohr


@dataclass(frozen=True)
class Rung:
    name: str = "M_inf"
    charges: bool = False
    dipoles: bool = False
    dispersion: bool = False
    dispersion8: bool = False              # the C8 band on a learned (or pinned) quadrupole polarisability: L6b
    analytic: bool = False
    band_fields: int = -1                  # -1: off; k: field tensors through order k at the atom
    bf_rs: float = 0.0                     # near-source exclusion radius of the band-field input (bohr; 0 = off; molecules: ~l_1)
    bf_fragments: bool = True              # exclude the own fragment's sources from the band-field input (structures with fragments)
    bf_E_ref: float = 0.01                 # unit of the field tensors in the read-out (a.u.; ~ an intermolecular field), see bandfields
    bf_l: float = 0.0                      # the response's band edge (bohr): the band field read by the read-out and its dipole
    # tensor smeared at bf_l instead of l1 (0: l1) -- the regulator of the response, which does what Thole damping does
    readout: str = "pair"                  # band-field read-out: "pair" (Taylor class on the structure) or "node" (L <= 2)
    bf_mutual: int = 0                     # the order-0 read-out's induced dipoles: 0 first order (-1/2 E.C.E, as before);
    # -1 self-consistent through their mutual fields (the band kernel's dipole tensor between fragments), all orders;
    # k >= 2 the series through k-th order in the polarisabilities (2: + the alpha^2 T term)
    bf_species: bool = False               # the read-out's coefficients from the species alone (bandfields.init_readout_species:
    # fixed features, a fixed output unit), not from E_0's features: a response independent of E_0 and of the seed
    msites: bool = False                   # the pinned charges of a site model (b["pin_q_site"] on the atoms and the virtual
    # sites b["vs_w"], b["vs_q"], e.g. TTM3-F's H atoms and M-site) as the charge channel's sources, instead of b["pin_q"]
    e0_field: bool = False                 # E_0's view of the pinned sources (v0.0.14): its site energies read the potential,
    # field and field gradient of the other fragments' site-model charges at the atom (the band kernel at l1), every
    # source's share tapered to zero between e0_field_on and e0_field_off (inside r_cut: local, the far field untouched),
    # through a term that vanishes at zero field (heads.field_energy)
    e0_field_on: float = ang_to_bohr(4.5)
    e0_field_off: float = ang_to_bohr(5.5)
    e0_field_unit: float = 0.01            # a.u.: the unit of the potential, field and field gradient in the term's input
    e0_field_whole: bool = False           # v0.0.15: every fragment's sources share one weight at an atom, the product of their
    # tapers (bandfields.local_field_tensors with frag_src): a molecule enters whole or not at all, never part charged;
    # False: each source tapered by its own distance (v0.0.14)
    e0_field_form: str = "mlp"             # "mlp": an MLP of the features, the potential, the field's contractions and squares
    # (heads.field_energy, v0.0.14); "quadratic": degree 2 in the field, -1/2 (E.A.E + G:B:G) with A and B from E_0's
    # features (heads.field_energy2, v0.0.15) -- no potential, no term linear in the field
    e0_field_cross: bool = False           # v0.0.16: the term less its value on each source fragment alone (with
    # e0_field_whole), eps_i = h(x_i) - sum_B h(x_i(B)): no self terms -- nothing to any dimer, the 2-body left to E_0, the
    # bands and the read-out -- only the fragments' fields together; at degree 2 the cross terms -E_i(B).A_i.E_i(C), B != C
    e0_field_self_on: float = 0.0          # v0.0.17 (the MLP form, every fragment whole): the self terms h(x_i(B)) weighted by
    e0_field_self_off: float = 0.0         # every fragment's own whole-molecule taper over (self_on, self_off), within the
    # term's reach, the cross terms h(x_i) - sum_B h(x_i(B)) at the term's reach (heads.field_energy_split); 0: the self
    # terms at the term's reach (field_energy)
    e0_field_cross_species: bool = False   # v0.0.17: the MLP's cross terms replaced by degree-2 cross terms with coefficients
    # by species alone, -1/2 sum_{B != C} [a_Z E_i(B).E_i(C) + c_Z G_i(B):G_i(C)] (heads.field_cross_species), beside the
    # MLP's self terms: strictly 3-body, nothing to any dimer, no gradient into E_0 through them
    pin_q: bool = False
    pin_mu: bool = False
    pin_alpha: bool = False
    pin_alpha2: bool = False
    frag_neutral: bool = True              # learned charges: the neutrality shift per fragment (each molecule neutral, an ion
    # charged) on structures that carry fragments; False: one shift over the whole structure (the head before v0.0.8)
    l1: float = ang_to_bohr(1.5)           # band edge
    lA: float = ang_to_bohr(4.0)           # envelope of the analytic kernel (M_A)
    NA: int = 4                            # powers of q^2 in M_A
    K: int = 8                             # Casimir-Polder frequencies
    omega0: float = 0.3
    one_oscillator: bool = False
    site_energies: bool = True             # E_0's site energies (off only for toy regressions of the band-field path)
    fragments: bool = True                 # subtract the model's own fragment energies when the structures carry fragments
    coulomb_energy: bool = True            # the bare Coulomb band energy (off: charges only source the band fields)
    global_block: bool = False             # M_G: sum_{i != j} MLP([s_i, s_j, r_ij / r_G]) with no cutoff and no envelope
    periodic: bool = False                 # bands by Ewald over the structure's cell (E_0 by minimum image)
    ewald_n_max: int = 8                   # reciprocal grid |n_a| <= n_max (static; periodic.n_max_for gives the smallest that
    # holds |k| <= k_max for a cell -- the periodic sums return NaN when it does not)
    ewald_b_max: float = 4.0               # k_max = 2 b_max / l_1
    ewald_chunk: int = 0                   # k per chunk of the periodic sums (0: periodic._chunk's choice)
    bf_solver: str = "direct"              # the periodic mutual response (v0.0.20): "direct" (the dipole tensor explicit, a
    # dense solve; O(N_k N^2) to build) or "gmres" (matrix-free, implicit derivatives; for large cells)
    bf_tol: float = 1e-10                  # the GMRES solve's relative residual
    bf_maxiter: int = 50                   # its restarts, each of bf_restart iterations
    bf_restart: int = 30
    rG: float = 10.0                       # length unit of the global block's distance input (bohr)
    e0: dict = field(default_factory=E0.default_config)


RUNGS = {
    "M_inf": Rung("M_inf"),
    "M_G": Rung("M_G", global_block=True),
    "M_A": Rung("M_A", analytic=True),
    "M_1": Rung("M_1", charges=True),
    "M_6": Rung("M_6", dispersion=True),
    "M_68": Rung("M_68", dispersion=True, dispersion8=True),
    "M_16": Rung("M_16", charges=True, dispersion=True),
    "M_16p": Rung("M_16p", charges=True, dispersion=True, pin_q=True, pin_alpha=True),
    "M_S0": Rung("M_S0", charges=True, dispersion=True, pin_q=True, pin_alpha=True, band_fields=0),
    "M_S2": Rung("M_S2", charges=True, dispersion=True, pin_q=True, pin_alpha=True, band_fields=2),
}


def init_params(key, rung: Rung, energy_scale=1.0):
    if rung.bf_mutual != 0 and (rung.band_fields < 0 or rung.readout != "pair" or rung.bf_mutual < -1):
        raise ValueError("bf_mutual (-1, or a series order >= 1) needs the pair band-field read-out")
    k = jax.random.split(key, 4)
    p = {"e0": E0.init_e0(k[0], rung.e0), "heads": H.init_heads(k[1], rung.e0, rung.K, rung.one_oscillator, energy_scale)}
    if rung.analytic:
        p["A_coeffs"] = jnp.zeros(rung.NA + 1).at[0].set(1e-3)
    if rung.bf_species and (rung.band_fields < 0 or rung.readout != "pair"):
        raise ValueError("bf_species needs the pair band-field read-out")
    if rung.band_fields >= 0 and rung.bf_species:
        p["readout"] = BF.init_readout_species(rung.e0, rung.band_fields)
    elif rung.band_fields >= 0:
        p["readout"] = (BF.init_readout_pair if rung.readout == "pair" else BF.init_readout)(k[2], rung.e0, rung.band_fields)
    if rung.global_block:
        F = rung.e0["F"]
        p["global"] = E0._mlp_init(k[3], [2 * F + 1, F, F, 1])
    if rung.e0_field:
        if not rung.msites:
            raise ValueError("e0_field reads the pinned site model's charges (Rung.msites)")
        if not 0 < rung.e0_field_on < rung.e0_field_off <= rung.e0["r_cut"] or rung.e0_field_unit <= 0:
            raise ValueError("e0_field needs 0 < e0_field_on < e0_field_off <= E_0's r_cut and a positive unit")
        if rung.e0_field_form not in ("mlp", "quadratic"):
            raise ValueError("e0_field_form is 'mlp' or 'quadratic'")
        if rung.e0_field_cross and not rung.e0_field_whole:
            raise ValueError("e0_field_cross takes every fragment whole (e0_field_whole): a fragment's own value needs its "
                             "sources under one weight")
        if rung.e0_field_self_off > 0 or rung.e0_field_cross_species:
            if not rung.e0_field_whole or rung.e0_field_form != "mlp" or rung.e0_field_cross:
                raise ValueError("e0_field_self_on/_off and e0_field_cross_species split the MLP form's self and cross terms: "
                                 "they need e0_field_whole, the form 'mlp' and the self terms (not e0_field_cross)")
            if rung.e0_field_self_off > 0 and not 0 < rung.e0_field_self_on < rung.e0_field_self_off <= rung.e0_field_off:
                raise ValueError("the self terms' reach needs 0 < e0_field_self_on < e0_field_self_off <= e0_field_off")
        # its own key, so that the other parameters are the rung's without the field term, digit for digit
        init_field = H.init_field_head if rung.e0_field_form == "mlp" else H.init_field_head2
        p["field"] = init_field(jax.random.fold_in(key, 14), rung.e0)
        if rung.e0_field_cross_species:
            p["field_x"] = H.init_field_cross_species(rung.e0)
    return p


def global_pair_energy(p, s, r, pmask, rG, energy_scale):
    """the null: an unconstrained learned pair function of (s_i, s_j, r_ij) summed over all pairs, no cutoff."""
    N = s.shape[0]
    x = jnp.concatenate([jnp.broadcast_to(s[:, None, :], (N, N, s.shape[1])),
                         jnp.broadcast_to(s[None, :, :], (N, N, s.shape[1])),
                         (jnp.where(pmask, r, 1.0) / rG)[..., None]], axis=-1)
    e = E0.mlp(p, x)[..., 0] * pmask
    return 0.5 * energy_scale * jnp.sum(e)


def frequencies(rung: Rung):
    return kn.casimir_polder_grid(rung.K, rung.omega0)


def isolated_energy(params, rung: Rung, Z):
    """site energy of an isolated atom of species Z: the model's own reference, so that every energy it
    returns is an atomisation / interaction energy and vanishes identically for non-interacting fragments."""
    cfg = rung.e0
    pos = jnp.zeros((1, 3)); num = jnp.array([Z]); mask = jnp.array([True])
    s, _, _, _ = E0.apply_e0(params["e0"], cfg, pos, num, mask, jnp.eye(3) * 1e6, False)
    return H.site_energy(params["heads"], s, num, mask)[0]


def energy_single(params, rung: Rung, b):
    """b: one (unbatched) element of a padded batch.  Returns the interaction energy in hartree and an aux dict:
    the total energy minus the model's own energies of the non-interacting references -- the isolated atoms, or,
    when the structure carries fragments (b["frag_mask"], e.g. the molecules of a cluster), each fragment alone
    at its own geometry.  Either way the energy vanishes identically for non-interacting references, which is
    what the asymptotic measurements need (a learned reference would leave a constant that swamps any tail).
    In a cell (rung.periodic, with the fragment grouping of structure.pad_batch) every fragment alone is evaluated
    open (_periodic_interaction)."""
    if rung.periodic and rung.fragments and "frag_atoms" in b:
        return _periodic_interaction(params, rung, b)
    E, aux = _total_energy(params, rung, b, b["mask"], subtract_atoms=True)
    if "frag_mask" in b and rung.fragments:
        # E_int = E(all) - sum_f E(fragment f alone); the atom references cancel in the difference.  With subtraction
        # groups (b["sub_mask"], whole fragments each) the groups are subtracted instead: E(all) - E(core) - E(probe)
        # is a probe's interaction energy with the cluster
        groups = b.get("sub_mask", b["frag_mask"])
        E_all, aux = _total_energy(params, rung, b, b["mask"], subtract_atoms=False)
        E_frag = jax.vmap(lambda fm: _total_energy(params, rung, b, fm, subtract_atoms=False)[0])(groups)
        valid = jnp.any(groups, axis=1)
        E = E_all - jnp.sum(jnp.where(valid, E_frag, 0.0))
        aux["E_frag"] = E_frag
    if "pin_eself" in b:
        # a pinned per-atom interaction-energy term: the polarisation work 1/2 |mu_ind|^2 / alpha of embedded (induced)
        # pinned dipoles, which makes the Coulomb band with pinned self-consistent dipoles reproduce the induction energy
        # exactly (band(q-mu) + band(mu-mu) + self = -1/2 sum mu.E_q at the stationary point); zero unless pinned
        E_self = jnp.sum(b["pin_eself"] * b["mask"])
        E = E + E_self; aux["E_self"] = E_self
    return E, aux


def _periodic_interaction(params, rung: Rung, b):
    """the interaction energy of a cell: its periodic energy (every pair over every image, each molecule's own pairs
    included) less every fragment alone, open (_fragment_alone) -- so that a molecule's interactions with its own images
    stay in, as in a periodic truth, and its own pairs at n = 0 cancel.  The fragments' references are taken in zero
    external field: the response to b["E_ext"] is the cell's whole."""
    if "sub_mask" in b:
        raise ValueError("subtraction groups are not set up in cells")
    pos = PB.make_whole(b["positions"], b["atom_frag"], b["frag_atoms"], b["cell"])
    b = {**b, "positions": pos}
    E_all, aux = _total_energy(params, rung, b, b["mask"], subtract_atoms=False)
    open_rung = replace(rung, periodic=False)
    E_frag = jax.vmap(lambda f: _fragment_alone(params, open_rung, b, f))(jnp.arange(b["frag_atoms"].shape[0]))
    valid = jnp.any(b["frag_atoms_mask"] & b["mask"][b["frag_atoms"]], axis=1)
    E = E_all - jnp.sum(jnp.where(valid, E_frag, 0.0))
    aux["E_frag"] = E_frag
    if "pin_eself" in b:
        E_self = jnp.sum(b["pin_eself"] * b["mask"])
        E = E + E_self; aux["E_self"] = E_self
    return E, aux


def _fragment_alone(params, open_rung: Rung, b, f):
    """fragment f of a periodic structure alone and open: its atoms (and virtual sites) gathered into a small structure,
    evaluated by the open model."""
    idx = b["frag_atoms"][f]
    am = b["frag_atoms_mask"][f] & b["mask"][idx]
    zero = jnp.zeros((), b["positions"].dtype)
    fc = b["frag_charge"][f] if "frag_charge" in b else zero
    bf = {"positions": b["positions"][idx], "numbers": b["numbers"][idx] * am, "mask": am,
          "cell": jnp.eye(3, dtype=b["cell"].dtype) * 1e6, "pbc": jnp.zeros((), bool), "total_charge": fc,
          "frag_mask": am[None, :], "frag_charge": fc[None], "has_frag": jnp.ones((), bool),
          "E_ext": jnp.zeros(3, b["positions"].dtype)}
    for k in ("pin_q", "pin_mu", "pin_alpha", "pin_alpha2"):
        if k in b:
            bf[k] = b[k][idx]
    if "vs_w" in b:
        sidx = b["frag_sites"][f]
        sm = b["frag_sites_mask"][f] & b["vs_mask"][sidx]
        bf.update(pin_q_site=b["pin_q_site"][idx], vs_w=b["vs_w"][sidx][:, idx] * sm[:, None], vs_q=b["vs_q"][sidx] * sm,
                  vs_mask=sm, vs_frag_mask=sm[None, :])
    return _total_energy(params, open_rung, bf, am, subtract_atoms=False)[0]


def _total_energy(params, rung: Rung, b, mask, subtract_atoms):
    cfg = rung.e0
    pos, Z = b["positions"], b["numbers"]
    grouped = rung.periodic and "frag_atoms" in b                  # a cell with the fragment grouping (v0.0.20)
    if grouped:
        pos = PB.make_whole(pos, b["atom_frag"], b["frag_atoms"], b["cell"])
    s, v, t, (D, r, pmask) = E0.apply_e0(params["e0"], cfg, pos, Z, mask, b["cell"], b["pbc"])
    hp = params["heads"]
    if rung.site_energies:
        eps = H.site_energy(hp, s, Z, mask)
        if subtract_atoms:
            e_iso = jax.vmap(lambda z: isolated_energy(params, rung, z))(Z) * mask
            E = jnp.sum(eps - e_iso)
        else:
            E = jnp.sum(eps)
    else:
        E = jnp.zeros(())
    aux = {"E0": E}
    q = mu = alpha = None
    sites = None
    if rung.msites:
        assert "vs_w" in b, "a site-model rung needs the site model's pins (pin_q_site, vs_w, vs_q)"
        assert rung.charges and rung.pin_q and not rung.dipoles and rung.bf_fragments, \
            "the site model pins the charges (no learned charges or dipoles; band fields by fragment)"
        assert grouped or not rung.periodic, "a site model in a cell needs the fragment grouping (structure.pad_batch)"
        # the virtual sites of the fragments present in this evaluation (the whole structure, or one subtraction group)
        fm = b["frag_mask"].astype(pos.dtype)                                     # (F,N)
        present = (fm @ mask.astype(pos.dtype)) > 0.5                              # (F,) fragments with atoms here
        vs_on = b["vs_mask"] & ((present.astype(pos.dtype) @ b["vs_frag_mask"].astype(pos.dtype)) > 0.5)
        X_src = jnp.concatenate([pos, b["vs_w"] @ pos], axis=0)                   # atoms, then the sites
        q_src = jnp.concatenate([b["pin_q_site"], b["vs_q"]])
        on_src = jnp.concatenate([mask, vs_on])
        frag_src = jnp.concatenate([b["frag_mask"], b["vs_frag_mask"]], axis=1)   # (F, N+V)
        sites = (X_src, q_src, on_src, frag_src)
    n_max, b_max, chunk = rung.ewald_n_max, rung.ewald_b_max, rung.ewald_chunk
    src = PB.grouped_sources(pos, b) if (grouped and sites is not None) else None   # every fragment's atoms and sites
    if rung.charges:
        if rung.msites:
            q = b["pin_q_site"]
        elif rung.pin_q:
            q = b["pin_q"]
        else:
            q = H.charges(hp, s, Z, mask, b["total_charge"])
            if rung.frag_neutral and "frag_mask" in b:
                # structures that carry fragments: one neutrality shift per fragment, so that every molecule is neutral
                # and the ion carries its charge whatever the environment (a structure without fragments in the same
                # batch keeps the shift over the structure)
                q = jnp.where(b["has_frag"], H.charges(hp, s, Z, mask, b["total_charge"], b["frag_mask"], b["frag_charge"]), q)
        if rung.dipoles:
            mu = b["pin_mu"] if rung.pin_mu else H.dipoles(hp, s, v, mask)
        if not rung.coulomb_energy:
            Ec = jnp.zeros(())
        elif sites is not None and rung.periodic:
            X_src, q_src, on_src, _ = sites
            Ec = PB.coulomb_pbc(X_src, q_src, on_src, b["cell"], rung.l1, n_max, b_max, chunk=chunk)
        elif sites is not None:
            X_src, q_src, on_src, _ = sites
            Ds, rs_, pms = pair_geometry(X_src, on_src)
            Ec = B.coulomb_band(q_src, None, Ds, rs_, pms, rung.l1)
        elif rung.periodic:
            Ec = PB.coulomb_pbc(pos, q, mask, b["cell"], rung.l1, n_max, b_max, mu=mu, chunk=chunk)
        else:
            Ec = B.coulomb_band(q, mu, D, r, pmask, rung.l1)
        E = E + Ec; aux["E_coul"] = Ec; aux["q"] = q
        if mu is not None: aux["mu"] = mu
    if rung.dispersion:
        omega, w = frequencies(rung)
        alpha = b["pin_alpha"][:, :rung.K] if rung.pin_alpha else H.polarisabilities(hp, s, Z, mask, omega, rung.one_oscillator)
        if rung.periodic:
            Ed = PB.dispersion_band_pbc(alpha, w, pos, mask, b["cell"], rung.l1, n_max, b_max, chunk)
        else:
            Ed = B.dispersion_band(alpha, w, r, pmask, rung.l1)
        E = E + Ed; aux["E_disp"] = Ed; aux["alpha"] = alpha
        if rung.dispersion8:
            alpha2 = b["pin_alpha2"][:, :rung.K] if rung.pin_alpha2 else H.quad_polarisabilities(hp, s, Z, mask, omega, rung.one_oscillator)
            if rung.periodic:
                E8 = PB.dispersion8_band_pbc(alpha, alpha2, w, pos, mask, b["cell"], rung.l1, n_max, b_max, chunk)
            else:
                E8 = B.dispersion8_band(alpha, alpha2, w, r, pmask, rung.l1)
            E = E + E8; aux["E_disp8"] = E8; aux["alpha2"] = alpha2
    if rung.analytic:
        sA = H.scalar_source(hp, s, mask)
        if rung.periodic:
            Ea = EW.analytic_band_pbc(sA, params["A_coeffs"], pos, b["cell"], mask, rung.lA, rung.ewald_n_max, rung.ewald_b_max)
        else:
            Ea = B.analytic_band(sA, params["A_coeffs"], r, pmask, rung.lA)
        E = E + Ea; aux["E_A"] = Ea; aux["sA"] = sA
    if rung.global_block:
        Eg = global_pair_energy(params["global"], s, r, pmask, rung.rG, hp["e_scale"])
        E = E + Eg; aux["E_G"] = Eg
    if rung.band_fields >= 0:
        assert q is not None, "band-field inputs need the charge channel"
        l_bf = rung.bf_l if rung.bf_l > 0 else rung.l1                         # the response's band edge
        same_frag = None
        if rung.bf_fragments and "frag_mask" in b:
            fm = b["frag_mask"].astype(pos.dtype)                                 # (F,N): the own fragment's atoms are not sources
            same_frag = (fm.T @ fm) > 0.5
        if sites is not None and rung.periodic:
            # every source and image except the atom's own fragment at n = 0 (periodic.site_band_field_pbc)
            assert rung.bf_rs <= 0, "the near-source exclusion (bf_rs) is not set up in cells"
            X_src, q_src, on_src, _ = sites
            sX, sq, son = src
            _, _, slots_on = PB.own_fragment(b, sX.shape[1])
            af, sl = b["atom_frag"], b["atom_slot"]
            q_self = sq[af, sl] * son[af, sl] * mask
            Ef, gE, ggE = PB.site_band_field_pbc(pos, mask, X_src, q_src, on_src, sX[af], sq[af], son[af] & slots_on, q_self,
                                                 b["cell"], l_bf, rung.band_fields, n_max, b_max, chunk)
        elif sites is not None:
            X_src, q_src, on_src, frag_src = sites
            same_src = (b["frag_mask"].astype(pos.dtype).T @ frag_src.astype(pos.dtype)) > 0.5      # (N, N+V)
            src_m = mask[:, None] & on_src[None, :] & ~same_src
            Ef, gE, ggE = BF.band_field_tensors_from(pos, X_src, q_src, src_m, l_bf, rung.band_fields, rung.bf_rs)
        elif rung.periodic:
            Ef, gE, ggE = BF.band_field_tensors_pbc(pos, q, mask, b["cell"], l_bf, rung.band_fields, rung.ewald_n_max,
                                                    rung.ewald_b_max, rung.bf_rs, D, pmask, excl_mask=same_frag)
        else:
            Ef, gE, ggE = BF.band_field_tensors(pos, q, pmask, l_bf, rung.band_fields, rung.bf_rs,
                                                src_mask=None if same_frag is None else ~same_frag)
        if "E_ext" in b:
            Ef = Ef + b["E_ext"][None, :] * mask[:, None]                     # a uniform external field (clamped nuclei)
        s_ro, unit = s, hp["e_scale"]
        if rung.bf_species:
            s_ro, unit = params["readout"]["embed"][Z] * mask[:, None], BF.SPECIES_UNIT
        if rung.bf_mutual != 0:
            # the order-0 sector as local polarisabilities, their induced dipoles coupled through the band kernel's
            # dipole tensor between atoms of different fragments (none within a fragment, as in the band field)
            assert rung.readout == "pair", "the mutual response is set up for the pair read-out"
            assert grouped or not rung.periodic, "the mutual response in a cell needs the fragment grouping"
            eps_hi, C = BF.response_energy_pair(params["readout"], cfg, rung.band_fields, s_ro, r, D, pmask, mask, Ef, gE, ggE,
                                                l_bf, E_ref=rung.bf_E_ref, e_scale=unit, tensor=True)
            if rung.periodic:
                own_idx, own_on = PB.own_fragment(b)
                _, mu_ind = PB.mutual_response_pbc(C, Ef, pos, mask, own_idx, own_on, b["cell"], l_bf, n_max, b_max,
                                                   order=rung.bf_mutual, solver=rung.bf_solver, tol=rung.bf_tol,
                                                   maxiter=rung.bf_maxiter, restart=rung.bf_restart, chunk=chunk)
            else:
                Tmask = pmask if same_frag is None else (pmask & ~same_frag)
                _, mu_ind = BF.mutual_response(C, BF.band_dipole_tensor(pos, Tmask, l_bf), Ef, rung.bf_mutual)
            eps_bf = eps_hi - 0.5 * jnp.sum(Ef * mu_ind, axis=-1)                  # per atom: -1/2 E_i . mu_i + the orders >= 1
            aux["mu_ind"] = mu_ind
        elif rung.readout == "pair":
            eps_bf = BF.response_energy_pair(params["readout"], cfg, rung.band_fields, s_ro, r, D, pmask, mask, Ef, gE, ggE,
                                             l_bf, E_ref=rung.bf_E_ref, e_scale=unit)
        else:
            eps_bf = BF.response_energy(params["readout"], cfg, rung.band_fields, s, v, t, mask, Ef, gE, ggE, l_bf)
        E = E + jnp.sum(eps_bf); aux["E_bf"] = jnp.sum(eps_bf); aux["eps_bf"] = eps_bf; aux["Efield"] = Ef
    if rung.e0_field:
        # E_0's view of the pinned sources: the other fragments' site-model charges within the taper, at every atom
        assert sites is not None and "frag_mask" in b, "the field term reads a site model's charges by fragment"
        X_src, q_src, on_src, frag_src = sites
        split = rung.e0_field_self_off > 0 or rung.e0_field_cross_species         # v0.0.17
        self_taper = (rung.e0_field_self_on, rung.e0_field_self_off) if rung.e0_field_self_off > 0 else None
        if rung.periodic:
            # every other fragment whole at its image nearest the atom (periodic.local_field_pbc)
            assert rung.e0_field_whole, "the field term in a cell takes every fragment whole (e0_field_whole)"
            sX, sq, son = src
            tensors = PB.local_field_pbc(pos, b["atom_frag"], mask, sX, sq, son, b["cell"], rung.l1, rung.e0_field_on,
                                         rung.e0_field_off, per_frag=rung.e0_field_cross or split, self_taper=self_taper)
        else:
            same_src = (b["frag_mask"].astype(pos.dtype).T @ frag_src.astype(pos.dtype)) > 0.5      # (N, N+V)
            src_m = mask[:, None] & on_src[None, :] & ~same_src
            tensors = BF.local_field_tensors(pos, X_src, q_src, src_m, rung.l1, rung.e0_field_on, rung.e0_field_off,
                                             frag_src=frag_src if rung.e0_field_whole else None,
                                             per_frag=rung.e0_field_cross or split, self_taper=self_taper)
        phi, Eloc, Gloc = tensors[:3]
        parts = tensors[3] if (rung.e0_field_cross or split) else None      # every fragment's share apart (v0.0.16)
        if split:
            if rung.e0_field_self_off <= 0:
                parts = parts + (parts[3],)                                   # the self terms at the term's reach
            eps_f = H.field_energy_split(params["field"], s, v, t, phi, Eloc, Gloc, mask, hp["e_scale"], rung.e0_field_unit,
                                         parts, mlp_cross=not rung.e0_field_cross_species)
            if rung.e0_field_cross_species:
                eps_f = eps_f + H.field_cross_species(params["field_x"], Z, Eloc, Gloc, mask, hp["e_scale"],
                                                      rung.e0_field_unit, parts)
        elif rung.e0_field_form == "quadratic":
            eps_f = H.field_energy2(params["field"], s, v, t, Eloc, Gloc, mask, hp["e_scale"], rung.e0_field_unit,
                                    parts=parts)
        else:
            eps_f = H.field_energy(params["field"], s, v, t, phi, Eloc, Gloc, mask, hp["e_scale"], rung.e0_field_unit,
                                   parts=parts)
        E = E + jnp.sum(eps_f); aux["E_field"] = jnp.sum(eps_f)
    if rung.periodic:
        # E_0's minimum image (and the field term's, the read-out's pair geometry) holds in cells at least 2 r_c wide
        E = jnp.where(jnp.min(PB.cell_widths(b["cell"])) >= 2.0 * cfg["r_cut"], E, jnp.nan)
    return E, aux


def energy_forces_single(params, rung: Rung, b):
    def e_of_pos(pos):
        return energy_single(params, rung, {**b, "positions": pos})[0]
    E, g = jax.value_and_grad(e_of_pos)(b["positions"])
    return E, -g * b["mask"][:, None]


def energy_forces(params, rung: Rung, batch):
    """batched: energies (B,), forces (B,N,3)."""
    return jax.vmap(lambda bb: energy_forces_single(params, rung, bb))(batch)


def aux_batched(params, rung: Rung, batch):
    return jax.vmap(lambda bb: energy_single(params, rung, bb)[1])(batch)


def with_rung(rung: Rung, **kw):
    return replace(rung, **kw)


def induced_dipole(params, rung: Rung, b):
    """the cell's (or cluster's) induced dipole, -dE/dE_ext at b["E_ext"] (the read-out's response; clamped nuclei)."""
    return -jax.grad(lambda ext: energy_single(params, rung, {**b, "E_ext": ext})[0])(b["E_ext"])


def dielectric_inf(params, rung: Rung, b):
    """the clamped-nuclei dielectric tensor of a cell, eps_inf = 1 + 4 pi chi with chi = -(1/V) d^2E/dE_ext^2 at zero
    external field: with tin-foil boundary conditions the macroscopic field is E_ext, so chi is the polarisation per field
    of the read-out's induced dipoles (X4b, T4: 1 exactly without a response)."""
    V = jnp.abs(jnp.linalg.det(b["cell"]))
    H = jax.hessian(lambda ext: energy_single(params, rung, {**b, "E_ext": ext})[0])(jnp.zeros(3, b["positions"].dtype))
    return jnp.eye(3) - 4.0 * jnp.pi * H / V
