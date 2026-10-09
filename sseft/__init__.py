"""
sseft -- the scale-separated EFT model (harness v0).

    E = E_0[local equivariant model] + sum_c (IR band of channel c with learned sources) [+ band-field inputs]

Everything is JAX, float64, with forces by autodiff.  Internal units are atomic (bohr, hartree); the IO
helpers convert from/to Angstrom and eV.  See README.md for the rung configurations and the project notes
(`claude/multiscale-eft-proposal.md`, `claude/phase-d-build-plan.md`) for what each piece is for.
"""
import jax

jax.config.update("jax_enable_x64", True)

from . import units, structure, kernels, e0, heads, bands, ewald, bandfields, model, train, measure, water  # noqa: E402,F401

__version__ = "0.0.5"
