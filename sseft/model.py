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
Energies in hartree, positions in bohr; forces by autodiff.
"""
from dataclasses import dataclass, field, replace
import jax
import jax.numpy as jnp
from . import e0 as E0, heads as H, bands as B, bandfields as BF, kernels as kn
from .units import ang_to_bohr


@dataclass(frozen=True)
class Rung:
    name: str = "M_inf"
    charges: bool = False
    dipoles: bool = False
    dispersion: bool = False
    analytic: bool = False
    band_fields: int = -1                  # -1: off; k: field tensors through order k at the atom
    pin_q: bool = False
    pin_mu: bool = False
    pin_alpha: bool = False
    l1: float = ang_to_bohr(1.5)           # band edge
    lA: float = ang_to_bohr(4.0)           # envelope of the analytic kernel (M_A)
    NA: int = 4                            # powers of q^2 in M_A
    K: int = 8                             # Casimir-Polder frequencies
    omega0: float = 0.3
    one_oscillator: bool = False
    site_energies: bool = True             # E_0's site energies (off only for toy regressions of the band-field path)
    coulomb_energy: bool = True            # the bare Coulomb band energy (off: charges only source the band fields)
    e0: dict = field(default_factory=E0.default_config)


RUNGS = {
    "M_inf": Rung("M_inf"),
    "M_A": Rung("M_A", analytic=True),
    "M_1": Rung("M_1", charges=True),
    "M_6": Rung("M_6", dispersion=True),
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
        p["readout"] = BF.init_readout(k[2], rung.e0, rung.band_fields)
    return p


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
        Ec = B.coulomb_band(q, mu, D, r, pmask, rung.l1) if rung.coulomb_energy else jnp.zeros(())
        E = E + Ec; aux["E_coul"] = Ec; aux["q"] = q
        if mu is not None: aux["mu"] = mu
    if rung.dispersion:
        omega, w = frequencies(rung)
        alpha = b["pin_alpha"][:, :rung.K] if rung.pin_alpha else H.polarisabilities(hp, s, Z, mask, omega, rung.one_oscillator)
        Ed = B.dispersion_band(alpha, w, r, pmask, rung.l1)
        E = E + Ed; aux["E_disp"] = Ed; aux["alpha"] = alpha
    if rung.analytic:
        sA = H.scalar_source(hp, s, mask)
        Ea = B.analytic_band(sA, params["A_coeffs"], r, pmask, rung.lA)
        E = E + Ea; aux["E_A"] = Ea; aux["sA"] = sA
    if rung.band_fields >= 0:
        assert q is not None, "band-field inputs need the charge channel"
        Ef, gE, ggE = BF.band_field_tensors(pos, q, pmask, rung.l1, rung.band_fields)
        eps_bf = BF.response_energy(params["readout"], cfg, rung.band_fields, s, v, t, mask, Ef, gE, ggE)
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
