"""A consistency check of the pinned rungs on the raw (near-rigid) frames.  In training every monomer is exactly rigid:
the pinned charges are the rigid monomer's, and the truth's forces carry the flux of its geometry-dependent charges,
which E_0 therefore learns to supply (the bands' forces hold the charges fixed).  The raw frames are read with the
charges recomputed at each frame's own monomer geometry (prepare_heldout; truth.site_model), so the bands already
carry the flux's energy -- and E_0 adds its learned flux response on top.  The amount at stake: the pinned
electrostatics with the frame's own charges less the same with the rigid monomer's charges, at the frame's positions.
It is pairwise (each molecule's charges follow its own geometry), so it cancels in the 3- and 4-body terms: a
double count shows in the 2-body reading only.

    python scratch/e3qp3_flux_double_count.py [external/qaqua/build_pol]   # ~1 min on a CPU
"""
import os, sys
import numpy as np
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "external", "qaqua", "build_pol")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from sseft import water as W, qaqua as QA
from sseft.units import ang_to_bohr
import e3_water as E3
K = 627.5094740631; AU = QA.AUANG; KC = QA.TTM3F_CHARGECON ** 2 / QA.AUKCAL
truth = W.qaqua_pol_truth(ROOT); qa = truth.qa
mono = W.monomer_positions()
q_site_rig = qa.ttm(mono / AU)["site_charges"][0]                  # H1, H2, M of the rigid monomer
q3_rig = truth.molecule_charges(mono)                              # O, H, H dipole-surface charges of the rigid monomer
rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
for key, n in (("2b", 2),):
    a = type("A", (), dict(heldout=key, heldout_oh_tol=0.03, heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02,
                           heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0, heldout_max=0))()
    d = E3.prepare_heldout(a, truth)[key]
    P = np.asarray(d["pos"]); roo = d["roo"]
    ttm_flux, q3_flux = [], []
    for p in P:
        t = qa.ttm(p / AU)
        sites = [np.array([p[3 * m + 1], p[3 * m + 2], t["msites"][m] * AU]) for m in range(n)]
        qa_ = t["site_charges"]; q3a = truth.molecule_charges(p)
        e_t = e_q = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                r = np.linalg.norm(sites[i][:, None] - sites[j][None], axis=-1)
                e_t += KC * np.sum((np.outer(qa_[i], qa_[j]) - np.outer(q_site_rig, q_site_rig)) / r)
                ra = np.linalg.norm(p[3 * i:3 * i + 3][:, None] - p[3 * j:3 * j + 3][None], axis=-1)
                e_q += KC * np.sum((np.outer(q3a[3 * i:3 * i + 3], q3a[3 * j:3 * j + 3]) - np.outer(q3_rig, q3_rig)) / ra)
        ttm_flux.append(e_t); q3_flux.append(e_q)
    ttm_flux, q3_flux = np.array(ttm_flux) * K, np.array(q3_flux) * K
    print(f"{key}: {len(P)} frames; pinned electrostatics, own charges less rigid-monomer charges: TTM3-F site charges "
          f"rms {rms(ttm_flux):.4f} kcal/mol (strict {rms(ttm_flux[d['strict']]):.4f}), q3 rms {rms(q3_flux):.4f}; "
          f"by largest O-O " + ", ".join(f"{lo}-{hi} A {rms(ttm_flux[(roo >= lo) & (roo < hi)]):.4f} / {rms(q3_flux[(roo >= lo) & (roo < hi)]):.4f}"
                                          for lo, hi in ((0, 3.2), (3.2, 5), (5, 7), (7, 99)) if ((roo >= lo) & (roo < hi)).any()), flush=True)
