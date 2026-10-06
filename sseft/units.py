"""Atomic units inside; Angstrom / eV at the interfaces."""
BOHR_PER_ANGSTROM = 1.8897259886
ANGSTROM_PER_BOHR = 1.0 / BOHR_PER_ANGSTROM
HARTREE_PER_EV = 1.0 / 27.211386245988
EV_PER_HARTREE = 27.211386245988
KCALMOL_PER_HARTREE = 627.5094740631
CM1_PER_HARTREE = 219474.6313632


def ang_to_bohr(x):
    return x * BOHR_PER_ANGSTROM


def bohr_to_ang(x):
    return x * ANGSTROM_PER_BOHR


def ev_to_hartree(x):
    return x * HARTREE_PER_EV


def hartree_to_ev(x):
    return x * EV_PER_HARTREE


def ev_per_ang_to_au(x):
    """force / gradient conversion: eV/A -> hartree/bohr"""
    return x * HARTREE_PER_EV * ANGSTROM_PER_BOHR


def au_to_ev_per_ang(x):
    return x * EV_PER_HARTREE * BOHR_PER_ANGSTROM
