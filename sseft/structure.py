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
    pinned: Dict[str, np.ndarray] = field(default_factory=dict)   # e.g. {"q": (N,), "mu": (N,3), "alpha": (N,K), "eself": (N,)};
    # a site model's charges: "q_site" (N,) on the atoms and virtual sites "vsite_w" (V,N) (each site at sum_a w_a r_a),
    # "vsite_q" (V,), "vsite_frag" (V,) (the fragment each site belongs to)
    info: Dict[str, Any] = field(default_factory=dict)
    frag: Optional[np.ndarray] = None     # (N,) fragment index per atom (the molecules); None: every atom its own fragment
    frag_charges: Optional[np.ndarray] = None   # (n_frag,) total charge of each fragment (the learned charges are shifted to it
    # fragment by fragment); None: every fragment neutral, which requires total_charge = 0 when fragments are given
    weight: float = 1.0                   # weight of this structure's energy and forces in the training loss
    sub: Optional[np.ndarray] = None      # (N,) subtraction group per atom: the model's energy is E(all) - sum_g E(group g
    # alone), each group whole fragments (e.g. a cluster's core and a probe molecule: the probe's interaction energy);
    # None: the fragments (or, without fragments, the isolated atoms) are the groups

    @property
    def n_atoms(self):
        return len(self.numbers)


def pad_batch(structures: List[Structure], n_max: Optional[int] = None, n_freq: int = 8):
    """Stack structures into padded arrays.  Returns a dict of jnp arrays with a leading batch axis.  When any
    structure carries fragments, "frag_mask" (B, max_frag, n_max) holds one mask per fragment (atoms are their
    own fragments where no fragments are given), "frag_charge" (B, max_frag) the total charge of each fragment
    and "has_frag" (B,) whether the structure gave fragments itself (the per-fragment neutrality shift of the
    learned charges applies to those; a structure without fragments keeps one shift to its total charge)."""
    n_max = n_max or max(s.n_atoms for s in structures)
    B = len(structures)
    with_frag = any(s.frag is not None for s in structures)
    if with_frag:
        n_frag = [int(np.max(s.frag)) + 1 if s.frag is not None else s.n_atoms for s in structures]
        max_frag = max(n_frag)
        frag_mask = np.zeros((B, max_frag, n_max), dtype=bool)
        frag_charge = np.zeros((B, max_frag)); has_frag = np.zeros(B, dtype=bool)
        for b, s in enumerate(structures):
            fr = s.frag if s.frag is not None else np.arange(s.n_atoms)
            for i, f in enumerate(fr): frag_mask[b, int(f), i] = True
            if s.frag is None:
                continue
            has_frag[b] = True
            if s.frag_charges is None:
                if abs(s.total_charge) > 1e-12:
                    raise ValueError("a charged structure with fragments needs frag_charges (the total charge of each fragment)")
                continue
            fc = np.asarray(s.frag_charges, float)
            if fc.shape != (n_frag[b],):
                raise ValueError(f"frag_charges must have one entry per fragment ({n_frag[b]}), got shape {fc.shape}")
            if abs(float(np.sum(fc)) - s.total_charge) > 1e-10:
                raise ValueError("frag_charges must sum to total_charge")
            frag_charge[b, :n_frag[b]] = fc
    out = {
        "positions": np.zeros((B, n_max, 3)), "numbers": np.zeros((B, n_max), dtype=np.int32),
        "mask": np.zeros((B, n_max), dtype=bool), "cell": np.tile(np.eye(3) * 1e6, (B, 1, 1)),
        "pbc": np.zeros(B, dtype=bool), "total_charge": np.zeros(B),
        "energy": np.zeros(B), "forces": np.zeros((B, n_max, 3)), "has_energy": np.zeros(B, dtype=bool),
        "has_forces": np.zeros(B, dtype=bool), "weight": np.ones(B), "pin_eself": np.zeros((B, n_max)),
        "pin_q": np.zeros((B, n_max)), "pin_mu": np.zeros((B, n_max, 3)), "pin_alpha": np.zeros((B, n_max, n_freq)),
        "pin_alpha2": np.zeros((B, n_max, n_freq)),
    }
    with_sites = any("vsite_w" in s.pinned for s in structures)
    if with_sites:
        # a site model (e.g. TTM3-F's charges on the H atoms and the M-site): the atoms' charges and the virtual sites
        if not with_frag or any("vsite_w" in s.pinned and s.frag is None for s in structures):
            raise ValueError("virtual sites belong to fragments: a structure with vsite_w needs frag")
        v_max = max(max(len(s.pinned.get("vsite_q", ())) for s in structures), 1)
        out.update(pin_q_site=np.zeros((B, n_max)), vs_w=np.zeros((B, v_max, n_max)), vs_q=np.zeros((B, v_max)),
                   vs_mask=np.zeros((B, v_max), dtype=bool), vs_frag_mask=np.zeros((B, max_frag, v_max), dtype=bool))
    with_sub = any(s.sub is not None for s in structures)
    if with_sub:
        groups = []
        for s in structures:
            if s.sub is not None:
                sub = np.asarray(s.sub, int)
                if sub.shape != (s.n_atoms,) or sub.min() < 0:
                    raise ValueError("sub must give a nonnegative group index per atom")
                if s.frag is not None and any(len(set(sub[np.asarray(s.frag) == f])) > 1 for f in np.unique(s.frag)):
                    raise ValueError("a subtraction group must hold whole fragments")
                groups.append(sub)
            else:
                groups.append(np.asarray(s.frag, int) if s.frag is not None else np.arange(s.n_atoms))
        sub_mask = np.zeros((B, max(int(g.max()) + 1 for g in groups), n_max), dtype=bool)
        for b, g in enumerate(groups):
            for i, k in enumerate(g): sub_mask[b, int(k), i] = True
        out["sub_mask"] = sub_mask
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
        if with_sites and "vsite_w" in s.pinned:
            w = np.asarray(s.pinned["vsite_w"], float); v = len(w)
            if w.shape != (v, n) or np.asarray(s.pinned["vsite_q"]).shape != (v,) or np.asarray(s.pinned["vsite_frag"]).shape != (v,):
                raise ValueError("vsite_w must be (V, N) with vsite_q and vsite_frag (V,)")
            out["pin_q_site"][b, :n] = s.pinned["q_site"]
            out["vs_w"][b, :v, :n] = w; out["vs_q"][b, :v] = s.pinned["vsite_q"]; out["vs_mask"][b, :v] = True
            for k, f in enumerate(s.pinned["vsite_frag"]): out["vs_frag_mask"][b, int(f), k] = True
    if with_frag:
        out["frag_mask"] = frag_mask; out["frag_charge"] = frag_charge; out["has_frag"] = has_frag
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
