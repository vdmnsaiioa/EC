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
Energies in hartree, positions in bohr; forces by autodiff.
"""
from dataclasses import dataclass, field, replace
import jax
import jax.numpy as jnp
from . import e0 as E0, heads as H, bands as B, bandfields as BF, kernels as kn, ewald as EW
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
    readout: str = "pair"                  # band-field read-out: "pair" (Taylor class on the structure) or "node" (L <= 2)
    pin_q: bool = False
    pin_mu: bool = False
    pin_alpha: bool = False
    pin_alpha2: bool = False
    l1: float = ang_to_bohr(1.5)           # band edge
    lA: float = ang_to_bohr(4.0)           # envelope of the analytic kernel (M_A)
    NA: int = 4                            # powers of q^2 in M_A
    K: int = 8                             # Casimir-Polder frequencies
    omega0: float = 0.3
    one_oscillator: bool = False
    site_energies: bool = True             # E_0's site energies (off only for toy regressions of the band-field path)
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
    k = jax.random.split(key, 4)
    p = {"e0": E0.init_e0(k[0], rung.e0), "heads": H.init_heads(k[1], rung.e0, rung.K, rung.one_oscillator, energy_scale)}
    if rung.analytic:
        p["A_coeffs"] = jnp.zeros(rung.NA + 1).at[0].set(1e-3)
    if rung.band_fields >= 0:
        p["readout"] = (BF.init_readout_pair if rung.readout == "pair" else BF.init_readout)(k[2], rung.e0, rung.band_fields)
    if rung.global_block:
        F = rung.e0["F"]
        p["global"] = E0._mlp_init(k[3], [2 * F + 1, F, F, 1])
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
    """b: one (unbatched) element of a padded batch.  Returns the interaction (atomisation) energy in hartree
    -- total minus the isolated-atom energies of the model -- and an aux dict."""
    cfg = rung.e0
    pos, Z, mask = b["positions"], b["numbers"], b["mask"]
    s, v, t, (D, r, pmask) = E0.apply_e0(params["e0"], cfg, pos, Z, mask, b["cell"], b["pbc"])
    hp = params["heads"]
    if rung.site_energies:
        eps = H.site_energy(hp, s, Z, mask)
        e_iso = jax.vmap(lambda z: isolated_energy(params, rung, z))(Z) * mask
        E = jnp.sum(eps - e_iso)
    else:
        E = jnp.zeros(())
    aux = {"E0": E}
    q = mu = alpha = None
    if rung.charges:
        q = b["pin_q"] if rung.pin_q else H.charges(hp, s, Z, mask, b["total_charge"])
        if rung.dipoles:
            mu = b["pin_mu"] if rung.pin_mu else H.dipoles(hp, s, v, mask)
        if not rung.coulomb_energy:
            Ec = jnp.zeros(())
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
        assert not rung.periodic, "M_A under periodic boundary conditions: not implemented yet (polynomial in q^2 in reciprocal space)"
        sA = H.scalar_source(hp, s, mask)
        Ea = B.analytic_band(sA, params["A_coeffs"], r, pmask, rung.lA)
        E = E + Ea; aux["E_A"] = Ea; aux["sA"] = sA
    if rung.global_block:
        Eg = global_pair_energy(params["global"], s, r, pmask, rung.rG, hp["e_scale"])
        E = E + Eg; aux["E_G"] = Eg
    if rung.band_fields >= 0:
        assert q is not None, "band-field inputs need the charge channel"
        assert not rung.periodic, "band-field inputs under periodic boundary conditions: not implemented yet (Ewald potential gradients)"
        Ef, gE, ggE = BF.band_field_tensors(pos, q, pmask, rung.l1, rung.band_fields)
        if rung.readout == "pair":
            eps_bf = BF.response_energy_pair(params["readout"], cfg, rung.band_fields, s, r, D, pmask, mask, Ef, gE, ggE, rung.l1)
        else:
            eps_bf = BF.response_energy(params["readout"], cfg, rung.band_fields, s, v, t, mask, Ef, gE, ggE, rung.l1)
        E = E + jnp.sum(eps_bf); aux["E_bf"] = jnp.sum(eps_bf); aux["eps_bf"] = eps_bf; aux["Efield"] = Ef
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
