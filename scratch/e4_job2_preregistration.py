"""
The numbers of E4 job 2's pre-registration (notes/e4-job2-preregistration.md), computed before training.  In this model
a uniform field reaches only the read-out, and the read-out is matched before training and held through it, so every
reading here is fixed before training: training cannot move it.

    python scratch/e4_job2_preregistration.py --config job2_local.yaml --arm A [--edges 0.75,1.5,2.0,2.5] \\
        [--names M_16C0pm_us,M_16C2pm_us] [--cache DIR] [--out scratch/e4_job2_preregistration.json]

arm A   the read-out matched on MB-pol's far probes alone (job 1's matching) with the mutual response's dipole tensor
        smeared at each edge (Rung.bf_l_mutual; 0.75 A is job 1's model): on MB-pol's held-out liquid frames, eps_inf,
        the cells' induced dipole and every molecule's induced dipole and field response against MB-pol's
        (E4.read_liquid), and the same on the calibration frames; eps_inf on the frames of the earlier scan that no
        calibration uses.
arm B   arm A's read-out calibrated on the calibration frames' cell sums (E4.match_readout's second stage,
        training.readout.liquid): the same readings, and the far probes and T5 on the calibrated read-out.

The read-outs are E4.matched's, cached in --cache under the keys of the earlier scan (scratch/e4_x3_preregistration.py
--sweep_mode mutual: {name}_l{edge}m; job 1's edge {name}_l0.75), so arm A reads the scan's own read-outs.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import jax                                        # noqa: E402
import sseft                                      # noqa: E402,F401
from sseft import model as M, water as W, train as T   # noqa: E402
import e4_water as E4                             # noqa: E402


def key_for(name, edge):
    return f"{name}_l0.75" if edge == 0.75 else f"{name}_l{edge:g}m"


def cfg_for(cfg, edge):
    return E4.merge(cfg, {"model": {"mutual_edge_angstrom": None if edge == 0.75 else edge}})


def eps_other(params, rung_cell, frames):
    """eps_inf (isotropic) of the model on frames given as structures."""
    r = E4.in_cell(rung_cell, frames[0].cell)
    f = jax.jit(lambda p, b: M.induced_response(p, r, b)[1])
    out = []
    for st in frames:
        b = {k: v[0] for k, v in E4.S.pad_batch([st], n_freq=r.K).items()}
        R = np.asarray(f(params, b))
        V = abs(np.linalg.det(np.asarray(st.cell)))
        out.append(float(1 + 4 * np.pi * np.trace(R) / 3 / V))
    return out


def short(v):
    return (f"eps {v['eps_inf_model']:.4f} (MB-pol {v['eps_inf_truth']:.4f}); cell M {100 * v['cell_dipole_rel_rmse']:.1f} %; "
            f"molecules: mu {100 * v['molecular_dipole_rel_rmse']:.1f} %, R {100 * v['molecular_response_rel_rmse']:.1f} % "
            f"(iso {100 * v['molecular_response_iso_rel_rmse']:.1f} %)"
            + (f"; iso R spread {v['molecular_response_iso_sd_ratio']:.2f} of MB-pol's (r {v['molecular_response_iso_corr']:.2f}), "
               f"|mu| {v['molecular_dipole_norm_ratio']:.3f} of MB-pol's (r {v['molecular_dipole_corr']:.3f})"
               if "molecular_response_iso_sd_ratio" in v else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", required=True, help="the local job-2 config (data, probes, response)")
    ap.add_argument("--arm", required=True, choices=("A", "B"))
    ap.add_argument("--edges", default="0.75,1.5,2.0,2.5")
    ap.add_argument("--names", default="M_16C0pm_us,M_16C2pm_us")
    ap.add_argument("--cache", default=None, help="the matched read-outs (default: the config's data directory)")
    ap.add_argument("--weight", type=float, default=0.1, help="arm B: training.readout.liquid's weight")
    ap.add_argument("--eps_ref", default=None, help="json with eps_truth.isotropic of the scan's frames (test, valid, "
                                                    "train order), for eps_inf on the frames no calibration uses")
    ap.add_argument("--splits", default="heldout,calibration", help="the liquid frames read")
    ap.add_argument("--other", type=int, default=1, help="0: skip eps_inf on the scan's other frames")
    ap.add_argument("--out", default=os.path.join(REPO, "scratch", "e4_job2_preregistration.json"))
    a = ap.parse_args()
    cfg = E4.load_config(a.config)
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    dcfg = cfg["data"]
    cache = a.cache or dcfg["dir"]
    probes = E4.load_probes(dcfg["probes"])
    train = W.e4_structures(os.path.join(dcfg["dir"], f"train_n{dcfg['train_size']}.npz"), pins=True)
    scale = T.energy_scale(train)
    response = E4.load_response(dcfg["response"], cfg)
    print(f"sseft {sseft.__version__}; probes {probes['hash']}; response {response['hash']}: "
          f"{len(response['calibration']['structures'])} calibration, {len(response['heldout']['structures'])} held-out "
          f"frames", flush=True)
    # the scan's other frames: those of the valid and test sets (no calibration frame among them), with MB-pol's eps_inf
    other, other_truth = [], []
    if a.eps_ref and a.other:
        ref = json.load(open(a.eps_ref))["eps_truth"]["isotropic"]
        k = 0
        for split in ("test", "valid"):
            st = W.e4_structures(os.path.join(dcfg["dir"], f"{split}_n{dcfg['train_size']}.npz"), pins=True)
            other += st; other_truth += ref[k:k + len(st)]; k += len(st)
    out.setdefault("meta", {}).update(probes=probes["hash"], response=response["hash"], energy_scale=scale,
                                      sseft=sseft.__version__)
    if other_truth:
        out["meta"]["other_frames_truth"] = other_truth
    lq = {"weight": a.weight}
    for edge in [float(x) for x in a.edges.split(",")]:
        cfg_e = cfg_for(cfg, edge)
        for name in a.names.split(","):
            t0 = time.time()
            key = key_for(name, edge)
            rung_cell, rung_open = E4.rungs_for(name, cfg_e, train[0].cell)
            p0 = M.init_params(jax.random.PRNGKey(0), rung_open, energy_scale=scale)
            liquid = E4.liquid_block(rung_cell, response, lq) if a.arm == "B" else None
            readout, match = E4.matched(key, rung_open, probes, scale, cfg_e, cache, liquid=liquid)
            params = E4.with_readout(p0, readout)
            rec = out.setdefault(f"arm{a.arm}", {}).get(f"{name}_l{edge:g}", {})
            rec.update(edge=edge, name=name, matching={k: v for k, v in match.items() if k != "probes_before_training"})
            rec["probes"] = match.get("probes_before_training") or E4.read_probes(params, rung_open, probes)
            rec["trimers"] = match.get("far_trimers_before_training") or E4.read_trimers(params, rung_open, probes)
            for split in a.splits.split(","):
                rec[split] = E4.read_liquid(params, rung_cell, response, split)
            if other:
                rec["eps_other"] = eps_other(params, rung_cell, other)
            out[f"arm{a.arm}"][f"{name}_l{edge:g}"] = rec
            json.dump(out, open(a.out, "w"), indent=1)
            t5, pr = rec["trimers"], rec["probes"]["by_distance_angstrom"]
            print(f"arm {a.arm} {name} at {edge:g} A [{time.time() - t0:.0f} s]\n"
                  + "".join(f"  {split + ':':12s} {short(rec[split])}\n" for split in ("heldout", "calibration") if split in rec)
                  + (f"  eps on the {len(rec['eps_other'])} frames no calibration uses: {np.mean(rec['eps_other']):.4f} "
                     f"(MB-pol {np.mean(out['meta']['other_frames_truth']):.4f})\n" if "eps_other" in rec else "")
                  + "  probes non-additive "
                  + " / ".join(f"{100 * v['nonadditive_rmse'] / v['nonadditive_rms']:.1f}" for v in pr.values())
                  + f" % at {', '.join(pr)} A; T5 slope {t5['slope']:.4f} ({100 * t5['three_body_rmse'] / t5['three_body_rms']:.1f} %)"
                  + (f"\n  matching objective {match['objective'][-1][1]:.3e}; liquid rows "
                     + "; ".join(f"{b:.3e} -> {c:.3e}" for b, c in match.get("liquid_objective", []))
                     if a.arm == "B" else ""), flush=True)
    print(f"written {a.out}", flush=True)


if __name__ == "__main__":
    main()
