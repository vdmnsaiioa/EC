"""Evaluate a trained MACE / MACE-LES baseline on E4's sets, every size (Form II's reading).

    python baselines/mace/eval.py results/e4_baselines/<run>/<run>.model [--sets valid_n64 test_n512 ...]

Runs in the mace-les env (baselines/mace/README.md). For each set (default: every valid_/test_ extxyz in data/e4) it
prints and writes to <model>_eval.json:
  - the energy error per molecule: mean (signed: model - truth), MAE, RMSE, in kcal/mol and meV;
  - the force RMSE per component, kcal/mol/A and meV/A;
  - for MACE-LES, if the calculator returns the latent charges: the net latent charge per molecule (rms and largest),
    the sum rule that LES does not impose (molecules are O, H, H in order, scripts/e4_data.py's frame format).
Energies in the extxyz are eV (REF_energy / REF_forces), interaction energies with the isolated monomer at zero.
"""
import argparse
import glob
import json
import os
import time

import numpy as np

KCALMOL_PER_EV = 23.060547830619026
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def default_sets(data_dir):
    names = [os.path.basename(p)[:-len(".extxyz")] for p in glob.glob(os.path.join(data_dir, "*.extxyz"))]
    names = [n for n in names if n.split("_")[0] in ("valid", "test")]
    return sorted(names, key=lambda n: (int(n.split("_n")[1]), n))


def predict(calc, atoms):
    """Energy (eV), forces (eV/A) and, for MACE-LES, the latent charges, from one call of the model on the calculator's
    own batch. (MACECalculator computes the latent charges but does not keep them; one call instead of two also halves
    the memory, which matters for the 1024-molecule cells on a 24 GB GPU.)"""
    from mace.tools import torch_tools
    batch = calc._atoms_to_batch(atoms)
    with torch_tools.default_dtype("float64"):
        out = calc.models[0](batch.to_dict(), compute_force=True, compute_stress=False)
    q = out.get("latent_charges")
    return (float(out["energy"].detach().cpu().sum()), out["forces"].detach().cpu().numpy(),
            None if q is None else q.detach().cpu().numpy())


def evaluate(calc, frames, cpu_calc=None):
    """cpu_calc: a calculator on the CPU, for a frame that does not fit on the GPU."""
    import torch
    de, df, q_mol, on_cpu = [], [], [], 0
    for at in frames:
        e_ref, f_ref = at.info["REF_energy"], at.arrays["REF_forces"]
        try:
            e, f, q = predict(calc, at)
        except torch.OutOfMemoryError:
            if cpu_calc is None:
                raise
            torch.cuda.empty_cache()
            e, f, q = predict(cpu_calc(), at)
            on_cpu += 1
        n_mol = len(at) // 3
        de.append((e - e_ref) / n_mol)
        df.append((f - f_ref).ravel())
        if q is not None:
            q_mol.append(q.reshape(n_mol, 3).sum(axis=1))
    de = np.array(de) * KCALMOL_PER_EV
    df = np.concatenate(df) * KCALMOL_PER_EV
    out = dict(frames=len(frames), molecules=len(frames[0]) // 3,
               e_mean_kcal_per_mol=float(de.mean()), e_mae_kcal_per_mol=float(np.abs(de).mean()),
               e_rmse_kcal_per_mol=float(np.sqrt((de ** 2).mean())),
               f_rmse_kcal_per_mol_A=float(np.sqrt((df ** 2).mean())), frames_on_cpu=on_cpu)
    for k in list(out):
        if k.endswith("kcal_per_mol") or k.endswith("kcal_per_mol_A"):
            out[k.replace("kcal_per_mol", "meV")] = out[k] / KCALMOL_PER_EV * 1000
    if q_mol:
        q_mol = np.concatenate(q_mol)
        out["latent_q_per_molecule_rms"] = float(np.sqrt((q_mol ** 2).mean()))
        out["latent_q_per_molecule_max"] = float(np.abs(q_mol).max())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--data_dir", default=os.path.join(REPO, "data", "e4"))
    ap.add_argument("--sets", nargs="*", default=None, help="set names, e.g. test_n512 (default: every valid_/test_)")
    ap.add_argument("--max_frames", type=int, default=None, help="evaluate only the first frames of each set")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--out", default=None, help="json path (default: <model>_eval.json)")
    a = ap.parse_args()

    import torch
    from ase.io import read
    from mace.calculators import MACECalculator

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    calc = MACECalculator(model_paths=a.model, device=device, default_dtype="float64", compute_stress=False)
    model_class = type(calc.models[0]).__name__
    print(f"model {a.model} ({model_class}) on {device}")
    cpu = {}

    def cpu_calc():          # built on first need
        if "calc" not in cpu:
            print("  (out of GPU memory: this set's remaining large frames go to the CPU)", flush=True)
            cpu["calc"] = MACECalculator(model_paths=a.model, device="cpu", default_dtype="float64",
                                         compute_stress=False)
        return cpu["calc"]

    results = dict(model=os.path.abspath(a.model), model_class=model_class, sets={})
    for name in a.sets or default_sets(a.data_dir):
        frames = read(os.path.join(a.data_dir, name + ".extxyz"), index=":")
        if a.max_frames:
            frames = frames[:a.max_frames]
        t = time.time()
        r = evaluate(calc, frames, cpu_calc=cpu_calc if device == "cuda" else None)
        r["seconds"] = time.time() - t
        results["sets"][name] = r
        line = (f"{name:12s} {r['frames']:3d} frames of {r['molecules']:4d} molecules: E per molecule mean "
                f"{r['e_mean_kcal_per_mol']:+.4f}, MAE {r['e_mae_kcal_per_mol']:.4f}, RMSE {r['e_rmse_kcal_per_mol']:.4f}"
                f" kcal/mol; F RMSE {r['f_rmse_kcal_per_mol_A']:.3f} kcal/mol/A")
        if "latent_q_per_molecule_rms" in r:
            line += f"; net latent q per molecule rms {r['latent_q_per_molecule_rms']:.3g}"
        if r["frames_on_cpu"]:
            line += f"; {r['frames_on_cpu']} frames on the CPU"
        print(line + f" ({r['seconds']:.0f} s)", flush=True)

    out = a.out or os.path.splitext(a.model)[0] + "_eval.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"written {out}")


if __name__ == "__main__":
    main()
