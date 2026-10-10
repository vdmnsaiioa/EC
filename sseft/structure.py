"""
Structures and padded batches.

A `Structure` is positions (N,3) in bohr, atomic numbers (N,), an optional cell (3,3) in bohr with pbc,
and optional reference data (energy in hartree, forces in hartree/bohr, pinned sources).  Batches are
padded to a common N_max with a boolean mask so that everything jit-compiles on fixed shapes; padded
atoms have Z = 0 and sit at the origin, and every sum over atoms or pairs is masked.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np
import jax.numpy as jnp


@dataclass
class Structure:
    positions: np.ndarray                 # (N,3) bohr
    numbers: np.ndarray                   # (N,) int
    cell: Optional[np.ndarray] = None     # (3,3) bohr, rows = lattice vectors
    pbc: bool = False
    energy: Optional[float] = None        # hartree
    forces: Optional[np.ndarray] = None   # (N,3) hartree/bohr
    total_charge: float = 0.0
    pinned: Dict[str, np.ndarray] = field(default_factory=dict)   # e.g. {"q": (N,), "mu": (N,3), "alpha": (N,K), "eself": (N,)}
    info: Dict[str, Any] = field(default_factory=dict)
    frag: Optional[np.ndarray] = None     # (N,) fragment index per atom (the molecules); None: every atom its own fragment
    weight: float = 1.0                   # weight of this structure's energy and forces in the training loss

    @property
    def n_atoms(self):
        return len(self.numbers)


def pad_batch(structures: List[Structure], n_max: Optional[int] = None, n_freq: int = 8):
    """Stack structures into padded arrays.  Returns a dict of jnp arrays with a leading batch axis.  When any
    structure carries fragments, "frag_mask" (B, max_frag, n_max) holds one mask per fragment (atoms are their
    own fragments where no fragments are given)."""
    n_max = n_max or max(s.n_atoms for s in structures)
    B = len(structures)
    with_frag = any(s.frag is not None for s in structures)
    if with_frag:
        n_frag = [int(np.max(s.frag)) + 1 if s.frag is not None else s.n_atoms for s in structures]
        max_frag = max(n_frag)
        frag_mask = np.zeros((B, max_frag, n_max), dtype=bool)
        for b, s in enumerate(structures):
            fr = s.frag if s.frag is not None else np.arange(s.n_atoms)
            for i, f in enumerate(fr): frag_mask[b, int(f), i] = True
    out = {
        "positions": np.zeros((B, n_max, 3)), "numbers": np.zeros((B, n_max), dtype=np.int32),
        "mask": np.zeros((B, n_max), dtype=bool), "cell": np.tile(np.eye(3) * 1e6, (B, 1, 1)),
        "pbc": np.zeros(B, dtype=bool), "total_charge": np.zeros(B),
        "energy": np.zeros(B), "forces": np.zeros((B, n_max, 3)), "has_energy": np.zeros(B, dtype=bool),
        "has_forces": np.zeros(B, dtype=bool), "weight": np.ones(B), "pin_eself": np.zeros((B, n_max)),
        "pin_q": np.zeros((B, n_max)), "pin_mu": np.zeros((B, n_max, 3)), "pin_alpha": np.zeros((B, n_max, n_freq)),
        "pin_alpha2": np.zeros((B, n_max, n_freq)),
    }
    for b, s in enumerate(structures):
        n = s.n_atoms
        out["positions"][b, :n] = s.positions; out["numbers"][b, :n] = s.numbers; out["mask"][b, :n] = True
        if s.cell is not None: out["cell"][b] = s.cell
        out["pbc"][b] = s.pbc; out["total_charge"][b] = s.total_charge; out["weight"][b] = s.weight
        if s.energy is not None: out["energy"][b] = s.energy; out["has_energy"][b] = True
        if s.forces is not None: out["forces"][b, :n] = s.forces; out["has_forces"][b] = True
        if "q" in s.pinned: out["pin_q"][b, :n] = s.pinned["q"]
        if "mu" in s.pinned: out["pin_mu"][b, :n] = s.pinned["mu"]
        if "alpha" in s.pinned: out["pin_alpha"][b, :n] = s.pinned["alpha"]
        if "alpha2" in s.pinned: out["pin_alpha2"][b, :n] = s.pinned["alpha2"]
        if "eself" in s.pinned: out["pin_eself"][b, :n] = s.pinned["eself"]
    if with_frag:
        out["frag_mask"] = frag_mask
    return {k: jnp.asarray(v) for k, v in out.items()}


def pair_geometry(positions, mask, cell=None, pbc=False):
    """Dense pair vectors r_ij = r_i - r_j (N,N,3), distances (N,N) with the diagonal and padded pairs set
    to a large value, and the pair mask.  Minimum image for periodic cells (valid for cutoffs < L/2)."""
    D = positions[:, None, :] - positions[None, :, :]
    if cell is not None:                      # minimum image; a non-periodic structure carries a 1e6 cell, so this is a no-op there
        frac = D @ jnp.linalg.inv(cell)
        frac = frac - jnp.round(frac)
        D = frac @ cell
    N = positions.shape[0]
    pmask = mask[:, None] & mask[None, :] & ~jnp.eye(N, dtype=bool)
    r = jnp.sqrt(jnp.sum(D * D, axis=-1) + 1e-300)
    r = jnp.where(pmask, r, 1e6)
    return D, r, pmask
