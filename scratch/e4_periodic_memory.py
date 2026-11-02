"""
Where the periodic model's memory goes (v0.0.21; RESULTS.md): XLA's temporary buffers for the energy of E3's frozen
model M_16C2pm_us (width 32, GMRES) in a liquid-density water cell, compiled and never run, with parts switched off,
and the largest arrays in the compiled program.  A 64-molecule random packing (12.42 A) tiled 1 x 1 x 1 or 2 x 2 x 2.

    python scratch/e4_periodic_memory.py --rep 2 --parts full,no_field_term,bands_only,E0_only

On CPU (2 Nov, the container): at 512 molecules the energy's buffers are 8.52 GB in full, 8.28 without the field term,
7.52 with the bands alone and 7.52 with E_0 alone -- E_0's all-pairs arrays (1536 x 1536 x 128 and kin) are 90 % of
it, the chunked Ewald sums nothing measurable.  The GPU's absolute numbers differ (its energy-and-forces program asked
for 14.2 GiB at 512 molecules and 38.4 GiB at 1024 on the 24 GB card), the proportions not.
"""
import argparse
import os
import re
import sys
import time
from dataclasses import replace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))
import jax                                       # noqa: E402
import sseft                                     # noqa: E402,F401
from sseft import model as M, periodic as PB     # noqa: E402
from sseft.mbpol import random_box               # noqa: E402
from test_periodic import water_structure, one, frozen, switched_on  # noqa: E402

L64 = 12.42                                      # A, 64 molecules at 0.997 g/cm3


def variant(name, rung):
    if name == "full":
        return rung
    if name == "no_field_term":
        return replace(rung, e0_field=False)
    if name == "bands_only":
        return replace(rung, band_fields=-1, bf_mutual=0, e0_field=False, bf_species=False)
    if name == "E0_only":
        return replace(M.RUNGS["M_inf"], e0=rung.e0, periodic=True, ewald_n_max=rung.ewald_n_max)
    raise ValueError(name)


def largest(hlo, top=6):
    sizes = {}
    for m in re.finditer(r"(f64|f32|s32|pred)\[([0-9,]+)\]", hlo):
        dims = [int(x) for x in m.group(2).split(",") if x]
        sizes[f"{m.group(1)}[{m.group(2)}]"] = int(np.prod(dims)) * {"f64": 8, "f32": 4, "s32": 4, "pred": 1}[m.group(1)]
    return sorted(sizes.items(), key=lambda kv: -kv[1])[:top]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--rep", type=int, default=2, help="the 64-molecule cell tiled rep x rep x rep")
    ap.add_argument("--parts", default="full,no_field_term,bands_only,E0_only")
    a = ap.parse_args()
    base = random_box(64, np.eye(3) * L64, np.random.default_rng(0), min_oo=2.5, min_any=1.5)
    shifts = [np.array([i, j, k]) * L64 for i in range(a.rep) for j in range(a.rep) for k in range(a.rep)]
    b = one(water_structure(np.concatenate([base + s for s in shifts]), np.eye(3) * L64 * a.rep))
    r0 = frozen(F=32)
    r0 = replace(r0, periodic=True, ewald_n_max=PB.n_max_for(np.asarray(b["cell"]), r0.l1, 4.0), ewald_b_max=4.0,
                 bf_solver="gmres")
    print(f"{64 * a.rep ** 3} molecules in a {L64 * a.rep:.2f} A cell, n_max {r0.ewald_n_max}", flush=True)
    for name in a.parts.split(","):
        rung = variant(name, r0)
        try:
            p = switched_on(rung)
        except (KeyError, ValueError):
            p = M.init_params(jax.random.PRNGKey(0), rung)
        t0 = time.time()
        c = jax.jit(lambda pp, bb: M.energy_single(pp, rung, bb)[0]).lower(p, b).compile()
        print(f"  {name:14s} energy: temporary buffers {c.memory_analysis().temp_size_in_bytes / 1e9:7.2f} GB "
              f"(compiled in {time.time() - t0:.0f} s)", flush=True)
        for s, n in largest(c.as_text()):
            print(f"      {s:40s} {n / 1e9:8.3f} GB")


if __name__ == "__main__":
    main()
