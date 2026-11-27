"""A trained MACE / MACE-LES model's energies and latent charges on any extxyz files (E4 job 17, v0.0.42).

    python baselines/mace/energies.py <run>.model data/e4/job17_clusters.extxyz data/e4/job17_slabs.extxyz \
        data/e4/probes_mbpol_far.extxyz [--out <run>_job17.npz]

Runs in the mace-les env (baselines/mace/README.md). Frames may be open (pbc F F F: LES then sums its latent charges'
Coulomb interaction directly) or periodic. Writes one npz (default: the model's path with .model replaced by _job17.npz)
holding, for each file's stem: <stem>__energy (F,) the model's energy in eV, in the file's frame order; <stem>__q the
latent charges (e) of every frame concatenated, empty for a model without them; <stem>__offsets (F + 1,) where each
frame's charges start. scripts/e4_job17.py read reads them.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import predict  # noqa: E402  (baselines/mace/eval.py: one call of the model on the calculator's batch)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    a = ap.parse_args()

    import torch
    from ase.io import read
    from mace.calculators import MACECalculator

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    calc = MACECalculator(model_paths=a.model, device=device, default_dtype="float64", compute_stress=False)
    print(f"model {a.model} ({type(calc.models[0]).__name__}) on {device}", flush=True)
    out = {}
    for path in a.files:
        t0 = time.time()
        stem = os.path.splitext(os.path.basename(path))[0]
        frames = read(path, index=":")
        E, Q, off = [], [], [0]
        for at in frames:
            e, _, q = predict(calc, at)
            E.append(e)
            q = np.zeros(0) if q is None else np.asarray(q, float).reshape(-1)
            Q.append(q)
            off.append(off[-1] + len(q))
        out[f"{stem}__energy"] = np.array(E)
        out[f"{stem}__q"] = np.concatenate(Q) if Q else np.zeros(0)
        out[f"{stem}__offsets"] = np.array(off, int)
        print(f"  {path}: {len(frames)} frames  [{time.time() - t0:.0f} s]", flush=True)
    dest = a.out or os.path.splitext(a.model)[0] + "_job17.npz"
    np.savez(dest, model=np.array(os.path.abspath(a.model)), **out)
    print(f"written {dest}", flush=True)


if __name__ == "__main__":
    main()
