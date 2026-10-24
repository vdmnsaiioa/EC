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
heads.field_energy2).
Energies in hartree, positions in bohr; forces by autodiff.
"""
from dataclasses import dataclass, field, replace
import jax
import jax.numpy as jnp
from . import e0 as E0, heads as H, bands as B, bandfields as BF, kernels as kn, ewald as EW
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
    ewald_n_max: int = 8                   # reciprocal grid |n_a| <= n_max (static)
    ewald_b_max: float = 4.0               # k_max = 2 b_max / l_1
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
        # its own key, so that the other parameters are the rung's without the field term, digit for digit
        init_field = H.init_field_head if rung.e0_field_form == "mlp" else H.init_field_head2
        p["field"] = init_field(jax.random.fold_in(key, 14), rung.e0)
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
    what the asymptotic measurements need (a learned reference would leave a constant that swamps any tail)."""
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


def _total_energy(params, rung: Rung, b, mask, subtract_atoms):
    cfg = rung.e0
    pos, Z = b["positions"], b["numbers"]
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
        assert rung.charges and rung.pin_q and not rung.dipoles and not rung.periodic and rung.bf_fragments, \
            "the site model pins the charges of open structures (no learned charges or dipoles, no cell; band fields by fragment)"
        # the virtual sites of the fragments present in this evaluation (the whole structure, or one subtraction group)
        fm = b["frag_mask"].astype(pos.dtype)                                     # (F,N)
        present = (fm @ mask.astype(pos.dtype)) > 0.5                              # (F,) fragments with atoms here
        vs_on = b["vs_mask"] & ((present.astype(pos.dtype) @ b["vs_frag_mask"].astype(pos.dtype)) > 0.5)
        X_src = jnp.concatenate([pos, b["vs_w"] @ pos], axis=0)                   # atoms, then the sites
        q_src = jnp.concatenate([b["pin_q_site"], b["vs_q"]])
        on_src = jnp.concatenate([mask, vs_on])
        frag_src = jnp.concatenate([b["frag_mask"], b["vs_frag_mask"]], axis=1)   # (F, N+V)
        sites = (X_src, q_src, on_src, frag_src)
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
        elif sites is not None:
            X_src, q_src, on_src, _ = sites
            Ds, rs_, pms = pair_geometry(X_src, on_src)
            Ec = B.coulomb_band(q_src, None, Ds, rs_, pms, rung.l1)
        elif rung.periodic:
            Ec = EW.coulomb_band_pbc(q, mu, pos, b["cell"], mask, rung.l1, rung.ewald_n_max, rung.ewald_b_max)
        else:
            Ec = B.coulomb_band(q, mu, D, r, pmask, rung.l1)
        E = E + Ec; aux["E_coul"] = Ec; aux["q"] = q
        if mu is not None: aux["mu"] = mu
    if rung.dispersion:
        omega, w = frequencies(rung)
        alpha = b["pin_alpha"][:, :rung.K] if rung.pin_alpha else H.polarisabilities(hp, s, Z, mask, omega, rung.one_oscillator)
        if rung.periodic:
            Ed = EW.dispersion_band_pbc(alpha, w, pos, b["cell"], mask, rung.l1, rung.ewald_n_max, rung.ewald_b_max)
        else:
            Ed = B.dispersion_band(alpha, w, r, pmask, rung.l1)
        E = E + Ed; aux["E_disp"] = Ed; aux["alpha"] = alpha
        if rung.dispersion8:
            alpha2 = b["pin_alpha2"][:, :rung.K] if rung.pin_alpha2 else H.quad_polarisabilities(hp, s, Z, mask, omega, rung.one_oscillator)
            if rung.periodic:
                E8 = EW.dispersion8_band_pbc(alpha, alpha2, w, pos, b["cell"], mask, rung.l1, rung.ewald_n_max, rung.ewald_b_max)
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
        if sites is not None:
            X_src, q_src, on_src, frag_src = sites
            same_src = (b["frag_mask"].astype(pos.dtype).T @ frag_src.astype(pos.dtype)) > 0.5      # (N, N+V)
            src = mask[:, None] & on_src[None, :] & ~same_src
            Ef, gE, ggE = BF.band_field_tensors_from(pos, X_src, q_src, src, l_bf, rung.band_fields, rung.bf_rs)
        elif rung.periodic:
            Ef, gE, ggE = BF.band_field_tensors_pbc(pos, q, mask, b["cell"], l_bf, rung.band_fields, rung.ewald_n_max,
                                                    rung.ewald_b_max, rung.bf_rs, D, pmask, excl_mask=same_frag)
        else:
            Ef, gE, ggE = BF.band_field_tensors(pos, q, pmask, l_bf, rung.band_fields, rung.bf_rs,
                                                src_mask=None if same_frag is None else ~same_frag)
        s_ro, unit = s, hp["e_scale"]
        if rung.bf_species:
            s_ro, unit = params["readout"]["embed"][Z] * mask[:, None], BF.SPECIES_UNIT
        if rung.bf_mutual != 0:
            # the order-0 sector as local polarisabilities, their induced dipoles coupled through the band kernel's
            # dipole tensor between atoms of different fragments (none within a fragment, as in the band field)
            assert rung.readout == "pair" and not rung.periodic, "the mutual response is set up for the pair read-out, open"
            eps_hi, C = BF.response_energy_pair(params["readout"], cfg, rung.band_fields, s_ro, r, D, pmask, mask, Ef, gE, ggE,
                                                l_bf, E_ref=rung.bf_E_ref, e_scale=unit, tensor=True)
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
        same_src = (b["frag_mask"].astype(pos.dtype).T @ frag_src.astype(pos.dtype)) > 0.5      # (N, N+V)
        src = mask[:, None] & on_src[None, :] & ~same_src
        phi, Eloc, Gloc = BF.local_field_tensors(pos, X_src, q_src, src, rung.l1, rung.e0_field_on, rung.e0_field_off,
                                                 frag_src=frag_src if rung.e0_field_whole else None)
        if rung.e0_field_form == "quadratic":
            eps_f = H.field_energy2(params["field"], s, v, t, Eloc, Gloc, mask, hp["e_scale"], rung.e0_field_unit)
        else:
            eps_f = H.field_energy(params["field"], s, v, t, phi, Eloc, Gloc, mask, hp["e_scale"], rung.e0_field_unit)
        E = E + jnp.sum(eps_f); aux["E_field"] = jnp.sum(eps_f)
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
