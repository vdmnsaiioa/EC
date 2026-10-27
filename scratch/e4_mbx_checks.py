"""E4's checks of the truth before any data (notes/e4-plan.md section 2): MB-pol through MBX (sseft/mbpol.py) in
periodic cells. No model, no training; minutes on a few cores.

    python scratch/e4_mbx_checks.py external/mbx/build/MBX [--out scratch/e4_mbx_checks.json] [--quick]

1. sseft.mbpol.check: MBX's own MB-pol example (a trimer, open and in a 20 A box) to its printed digits, the
   interaction energy's zero for a lone monomer, forces against differences, the monomer's polarisability.
2. The boundary condition. One rigid molecule in cubic cells of side L: E(L) against tin-foil's dipole image term
   -(2 pi / 3) |mu|^2 / L^3 (size note, Lemma 1). The ratio tends to 1 with an O(L^-2) remainder (quadrupole
   images); under vacuum boundary conditions it would tend to 0.
3. The supercell identity, E(cell) = E(replica) / copies:
   (a) 32 molecules in a 13.5 A cell, every cutoff inside its half-width: every term;
   (b) 64 molecules in the liquid-density cell (12.43 A) as MBX takes it (replicate off, r_c 6.2 A for both): the
       2-body term, which MBX's minimum-image 2-body list gets wrong when the cell is narrower than twice the 6.5 A
       reach;
   (c) the same cell through its 2x2x2 replica (the driver's default) against its 3x3x3 replica.
4. PME convergence on the 32-molecule cell: the grid density, the spline order, the Ewald split.
5. Timings of energy and forces at 64 (through its replica), 128, 256 and 512 molecules (randomly packed rigid
   monomers at 0.997 g/cm^3), with the PME settings of MBX's example and with E4's (the driver's defaults).
7. The dispersion's real-space cutoff: the 64-molecule cell's dispersion through its replica at r_c 6.5 and 9 A
   (MBX's long-range O-H c6 product is 17 % below its pair C6, so r_c is part of the Hamiltonian).
6. The clamped-nuclei dielectric constant of a randomly packed 128-molecule cell through the external field: a test
   that the field works in a cell, not the liquid's value.

The configurations are random packings of rigid monomers, not liquids: the checks are of the code, not of water."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from sseft import mbpol as M  # noqa: E402

KCAL = M.KCALMOL_PER_HARTREE
COULOMB = 1e-7 * (299792458.0 * 1.602176565e-19) ** 2 / 1e-10 * 6.02214129e23 / 4184.0   # MBX's, kcal A/mol/e^2
ACCURATE = {}                         # the driver's defaults: E4's truth (r_c 6.5 A, alpha 0.7, grid 6 / A, order 8)
MBX_EXAMPLE = dict(alpha_ewald_elec=0.6, alpha_ewald_disp=0.6, grid_density_elec=2.5, grid_density_disp=2.5,
                   spline_order_elec=6, spline_order_disp=6)


def liquid_side(n_mol, rho=0.997):
    return (n_mol * 18.01528 / (rho * 0.6022140857)) ** (1 / 3)


def comps_kcal(t, pos, cell):
    return {k: v * KCAL for k, v in t.components(pos, cell).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=os.path.join(HERE, "e4_mbx_checks.json"))
    ap.add_argument("--quick", action="store_true", help="skip the 512-molecule timings and the 3x3x3 replica")
    a = ap.parse_args()
    out = {"threads": os.environ.get("OMP_NUM_THREADS")}

    print("1. sseft.mbpol.check")
    out["check"] = M.check(a.root)

    print("\n2. one molecule in cubic cells: E(L) against -(2 pi/3) |mu|^2 / L^3 (tin-foil)")
    m = M.monomer_ang() + np.array([1.0, 2.0, 3.0])
    t = M.MBPol(a.root, settings=ACCURATE)
    s = t.sites(m)
    mu = (s["charges"][:, None] * s["xyz"]).sum(0)
    print(f"   the monomer's dipole {np.linalg.norm(mu):.6f} e A = {np.linalg.norm(mu) / 0.20822678:.4f} D")
    rows = []
    for L in (20.0, 30.0, 40.0, 60.0):
        E = t.energy_kcal(m, np.eye(3) * L)
        pred = -(2 * math.pi / 3) * COULOMB * (mu @ mu) / L ** 3
        rows.append(dict(L=L, E=E, pred=pred, ratio=E / pred, remainder_L2=(1 - E / pred) * L * L))
        print(f"   L {L:5.1f} A: E {E:.6e}  tin-foil {pred:.6e}  ratio {E / pred:.6f}  (1 - ratio) L^2 "
              f"{(1 - E / pred) * L * L:.2f} A^2")
    out["boundary"] = rows

    print("\n3. the supercell identity")
    rng = np.random.default_rng(1)
    cell = np.eye(3) * 13.5
    pos = M.random_box(32, cell, rng)
    t = M.MBPol(a.root, settings=ACCURATE)
    c1 = comps_kcal(t, pos, cell)
    P, C = M.replicate(pos, cell)
    c8 = {k: v / 8 for k, v in comps_kcal(t, P, C).items()}
    out["supercell_a"] = {k: [c1[k], c8[k]] for k in c1}
    print("   (a) 32 molecules, 13.5 A, and 2x2x2 / 8:")
    for k in c1:
        print(f"       {k:15s} {c1[k]:16.10f} {c8[k]:16.10f}  {c8[k] - c1[k]:+.1e}")
    L = liquid_side(64)
    cell = np.eye(3) * L
    pos = M.random_box(64, cell, np.random.default_rng(2), min_oo=2.5)
    t = M.MBPol(a.root, settings=dict(ACCURATE, replicate=False, realspace_cutoff=6.2))   # r_c within the cell
    c1 = comps_kcal(t, pos, cell)
    P, C = M.replicate(pos, cell)
    c8 = {k: v / 8 for k, v in comps_kcal(t, P, C).items()}
    out["supercell_b"] = {k: [c1[k], c8[k]] for k in c1}
    print(f"   (b) 64 molecules, {L:.2f} A, as MBX takes the cell, and 2x2x2 / 8:")
    for k in ("two_body", "three_body", "dispersion", "electrostatics"):
        print(f"       {k:15s} {c1[k]:16.10f} {c8[k]:16.10f}  {c8[k] - c1[k]:+.1e}")
    if not a.quick:
        t = M.MBPol(a.root, settings=ACCURATE)
        e2 = t.energy_kcal(pos, cell)                               # the driver: through the 2x2x2 replica
        P, C = M.replicate(pos, cell, (3, 3, 3))
        e3 = t.energy_kcal(P, C) / 27
        out["supercell_c"] = [e2, e3]
        print(f"   (c) the same cell through its 2x2x2 replica {e2:.10f}, its 3x3x3 / 27 {e3:.10f}  ({e3 - e2:+.1e})")

    print("\n4. PME convergence, the 32-molecule cell (electrostatics; dispersion), kcal/mol")
    pos = M.random_box(32, np.eye(3) * 13.5, np.random.default_rng(1))
    rows = []
    for alpha in (0.5, 0.6, 0.7):
        for gd, so in ((2.5, 6), (4.0, 6), (6.0, 8), (8.0, 8)):
            t = M.MBPol(a.root, settings=dict(realspace_cutoff=6.5, alpha_ewald_elec=alpha, alpha_ewald_disp=alpha,
                                               grid_density_elec=gd, grid_density_disp=gd, spline_order_elec=so,
                                               spline_order_disp=so))
            c = comps_kcal(t, pos, np.eye(3) * 13.5)
            rows.append(dict(alpha=alpha, grid=gd, order=so, elec=c["electrostatics"], disp=c["dispersion"]))
            print(f"   alpha {alpha} grid {gd} order {so}: {c['electrostatics']:.10f}; {c['dispersion']:.10f}")
    out["pme"] = rows

    print("\n5. timings, energy and forces (s)")
    rows = []
    for n in ((64, 128, 256) if a.quick else (64, 128, 256, 512)):
        L = liquid_side(n)
        pos = M.random_box(n, np.eye(3) * L, np.random.default_rng(n), min_oo=2.5)
        for label, st in (("MBX's example PME", MBX_EXAMPLE), ("E4's truth", ACCURATE)):
            t = M.MBPol(a.root, settings=st)
            t.energy_forces(pos, np.eye(3) * L)                     # initialisation
            t0 = time.time()
            t.energy_forces(pos, np.eye(3) * L)
            dt = time.time() - t0
            rows.append(dict(n=n, L=L, settings=label, seconds=dt))
            print(f"   {n:4d} molecules, {L:.2f} A, {label}: {dt:.2f} s")
    out["timings"] = rows

    print("\n6. the clamped-nuclei dielectric constant of a random 128-molecule cell (a test of the field in a cell)")
    L = liquid_side(128)
    pos = M.random_box(128, np.eye(3) * L, np.random.default_rng(5), min_oo=2.5)
    t = M.MBPol(a.root, settings=ACCURATE)
    R = t.induced_response(pos, np.eye(3) * L)
    eps = 1 + 4 * math.pi / L ** 3 * R
    alpha_mol = out["check"]["alpha_monomer"]
    x = 4 * math.pi * (128 / L ** 3) * alpha_mol / 3
    out["epsilon"] = dict(tensor=eps.tolist(), isotropic=float(np.trace(eps) / 3),
                          first_order=1 + 3 * x, clausius_mossotti=(1 + 2 * x) / (1 - x))
    print(f"   eps_inf {np.trace(eps) / 3:.4f} (diagonal {np.round(np.diag(eps), 4)}); first order 1 + 4 pi n alpha "
          f"{1 + 3 * x:.4f}, Clausius-Mossotti {(1 + 2 * x) / (1 - x):.4f} with the monomer's {alpha_mol:.4f} A^3")

    print("\n7. the dispersion's real-space cutoff: the 64-molecule cell through its replica, kcal/mol")
    L = liquid_side(64)
    pos = M.random_box(64, np.eye(3) * L, np.random.default_rng(2), min_oo=2.5)
    rows = []
    for rc in (6.5, 9.0):
        t = M.MBPol(a.root, settings=dict(ACCURATE, realspace_cutoff=rc))
        c = comps_kcal(t, pos, np.eye(3) * L)
        rows.append(dict(rc=rc, dispersion=c["dispersion"], electrostatics=c["electrostatics"]))
        print(f"   r_c {rc}: dispersion {c['dispersion']:.6f}, electrostatics {c['electrostatics']:.6f}")
    out["dispersion_cutoff"] = rows

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
