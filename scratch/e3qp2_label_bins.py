"""The raw CCSD(T) sets' label rms by largest O-O bin, with the selection of configs/water_qaqua_pol2.yaml (E3-QP2's
held-out sets), so that the per-bin readings of the log can be read as shares of their signal.  The PES's own rmse per
bin is printed too: it must equal the log's (a check of the selection)."""
import os, sys, json, math
import numpy as np

REPO = sys.argv[1]
ROOT = sys.argv[2]
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from sseft import water as W
import e3_water as E3

K = 627.5094740631
truth = W.qaqua_pol_truth(ROOT)
a = type("A", (), dict(heldout="2b,3b,4b", heldout_oh_tol=0.03, heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02,
                       heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0, heldout_max=0))()
sets = E3.prepare_heldout(a, truth)
out = {}
rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")
for key, d in sets.items():
    lab, pes, roo = d["label"], d["pes"], d["roo"]
    row = {"n": len(lab), "label_rms": rms(lab) * K, "pes_rmse": rms(pes - lab) * K, "bins": {}}
    for lo, hi in ((0, 5), (5, 7), (7, 9), (9, math.inf)):
        sel = (roo >= lo) & (roo < hi)
        row["bins"][f"{lo}-{hi}"] = {"n": int(sel.sum()), "label_rms": rms(lab[sel]) * K, "pes_rmse": rms(pes[sel] - lab[sel]) * K}
    out[key] = row
    print(key, f"n {row['n']}, labels rms {row['label_rms']:.4f}, PES {row['pes_rmse']:.4f};",
          "; ".join(f"{b} n {v['n']} labels {v['label_rms']:.4f} PES {v['pes_rmse']:.4f}" for b, v in row["bins"].items()))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e3qp2_label_bins.json"), "w"), indent=1)
