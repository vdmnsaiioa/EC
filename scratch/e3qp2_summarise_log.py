"""E3-QP2's cluster readings from the job log: seed means, seed ranges and shares of the signal, per rung.

    python scratch/e3qp2_summarise_log.py logs/e3qp2_<job>.out      # needs scratch/e3qp2_label_bins.json
"""
import re, json, os, sys
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
lines = open(sys.argv[1]).read().splitlines()
bins = json.load(open(os.path.join(HERE, "e3qp2_label_bins.json")))
HEX_NA, HEX_IND = 0.997, 0.772

R = defaultdict(lambda: defaultdict(dict))      # R[rung][seed][key] = value
pes_seen = defaultdict(set)
num = r"([-+0-9.e]+)"
for ln in lines:
    m = re.match(r"\s*(\S+) seed (\d): (.*)$", ln)
    if not m:
        continue
    rung, seed, rest = m.group(1), int(m.group(2)), m.group(3)
    d = R[rung][seed]
    if rest.startswith("size 6"):
        e, na = re.findall(r"rmse ([0-9.]+)", rest)[:2]
        d["hexE"], d["hexNA"] = float(e), float(na)
    elif rest.startswith("size 2"):
        d["dimE"] = float(re.findall(r"rmse ([0-9.]+)", rest)[0])
    elif rest.startswith("probes ("):
        kind = "train" if "(train)" in rest else "test"
        if "through order 1" in rest:
            kind = "test_o1"
        elif "through order 2" in rest:
            kind = "test_o2"
        it, ip, na, nap = re.search(r"interaction ([0-9.]+) kcal/mol \((\d+) %\), non-additive ([0-9.]+) \((\d+) %\)", rest).groups()
        d[kind] = {"int": float(it), "int%": int(ip), "na": float(na), "na%": int(nap),
                   "by_d": {int(dd): (int(a), int(b)) for dd, a, b in re.findall(r"(\d+) A (\d+) / (\d+) %", rest)}}
    elif rest.startswith("held-out"):
        key = re.match(r"held-out (\db)", rest).group(1)
        tot = float(re.search(r"rmse ([0-9.]+) kcal/mol", rest).group(1))
        strict = float(re.search(r"strict ([0-9.]+)", rest).group(1))
        bb = {}
        for lo, hi, v, p in re.findall(r"(\d+)-(\d+|inf) A ([0-9.]+) / PES ([0-9.]+)", rest):
            bb[f"{lo}-{hi}"] = float(v); pes_seen[(key, f"{lo}-{hi}")].add(p)
        d[key] = {"tot": tot, "strict": strict, "bins": bb}

R = {r: v for r, v in R.items() if all("hexE" in d for d in v.values())}     # the clusters part's rungs only
assert all(len(v) == 1 for v in pes_seen.values()), {k: v for k, v in pes_seen.items() if len(v) > 1}
rungs = list(R)
assert all(sorted(R[r]) == [0, 1, 2, 3] for r in rungs), {r: sorted(R[r]) for r in rungs}

def stat(vals):
    v = np.array(vals, float)
    return v.mean(), v.min(), v.max()

summary = {}
print(f"{'rung':10s} {'hexE':>17s} {'hexNA (kcal, %)':>24s} {'dimE':>7s} | {'2b':>14s} {'3b':>14s} {'4b':>14s}")
for r in rungs:
    S = R[r]; s = {}
    for k in ("hexE", "hexNA", "dimE"):
        s[k] = stat([S[i][k] for i in range(4)])
    for key in ("2b", "3b", "4b"):
        lab = bins[key]["label_rms"]
        s[key] = {"tot": stat([S[i][key]["tot"] for i in range(4)]),
                  "strict": stat([S[i][key]["strict"] for i in range(4)]),
                  "bins": {b: stat([S[i][key]["bins"][b] for i in range(4)]) for b in S[0][key]["bins"]}}
    s["seeds"] = {k: [S[i][k] for i in range(4)] for k in ("hexE", "hexNA", "dimE")}
    s["seeds"].update({f"{key}_{b}": [S[i][key]["bins"][b] for i in range(4)] for key in ("2b", "3b", "4b") for b in S[0][key]["bins"]})
    s["seeds"].update({f"{key}_tot": [S[i][key]["tot"] for i in range(4)] for key in ("2b", "3b", "4b")})
    s["seeds"].update({f"{key}_strict": [S[i][key]["strict"] for i in range(4)] for key in ("2b", "3b", "4b")})
    # probes: are the 10-15 A readings the same on every seed?
    for kind in ("test", "train", "test_o1", "test_o2"):
        if kind in S[0]:
            byd = [S[i][kind]["by_d"] for i in range(4)]
            s[f"probes_{kind}"] = {dd: [b[dd] for b in byd] for dd in (8, 10, 12, 15)}
            s[f"probes_{kind}_overall"] = [(S[i][kind]["int%"], S[i][kind]["na%"]) for i in range(4)]
    summary[r] = s
    hn = s["hexNA"]
    print(f"{r:10s} {s['hexE'][0]:.3f} [{s['hexE'][1]:.3f},{s['hexE'][2]:.3f}] "
          f"{hn[0]:.3f} ({100 * hn[0] / HEX_NA:4.1f} %) [{hn[1]:.3f},{hn[2]:.3f}] {s['dimE'][0]:.3f} | "
          + " ".join(f"{s[k]['tot'][0]:.4f} ({100 * s[k]['tot'][0] / bins[k]['label_rms']:3.0f}%)" for k in ("2b", "3b", "4b")))

print("\nraw sets by largest O-O bin: seed mean (share of the bin's label rms); PES's own share")
for key in ("2b", "3b", "4b"):
    print(f"  {key}: " + "; ".join(f"{b} n {v['n']} labels {v['label_rms']:.4f} PES {100 * v['pes_rmse'] / v['label_rms']:.0f}%"
                                 for b, v in bins[key]["bins"].items() if v["n"]))
    for r in rungs:
        bb = summary[r][key]["bins"]
        print(f"    {r:10s} " + "  ".join(f"{b}: {bb[b][0]:.4f} ({100 * bb[b][0] / bins[key]['bins'][b]['label_rms']:3.0f}%)"
                                         for b in bb if bins[key]["bins"][b]["n"])
              + f"  strict {summary[r][key]['strict'][0]:.4f}")

print("\nprobes (test), non-additive % by d (seed 0..3), and whether the 10-15 A readings agree across seeds")
for r in rungs:
    for kind in ("test", "test_o1", "test_o2"):
        if f"probes_{kind}" in summary[r]:
            p = summary[r][f"probes_{kind}"]
            same = all(len(set(p[dd])) == 1 for dd in (10, 12, 15))
            print(f"  {r:10s} {kind:8s} " + "  ".join(f"{dd} A " + "/".join(str(x[1]) for x in p[dd]) for dd in (8, 10, 12, 15))
                  + f"   interaction 10-15 A: " + " ".join("/".join(str(x[0]) for x in p[dd]) for dd in (10, 12, 15))
                  + ("   seed-invariant at 10-15 A" if same else "   SEEDS DIFFER at 10-15 A"))

print("\nper-seed hexamer non-additive (kcal/mol) and 4b totals")
for r in rungs:
    print(f"  {r:10s} hexNA " + " ".join(f"{x:.3f}" for x in summary[r]["seeds"]["hexNA"])
          + "   hexE " + " ".join(f"{x:.3f}" for x in summary[r]["seeds"]["hexE"])
          + "   4b " + " ".join(f"{x:.4f}" for x in summary[r]["seeds"]["4b_tot"]))

json.dump({r: {k: v for k, v in s.items()} for r, s in summary.items()}, open(os.path.join(HERE, "e3qp2_summary.json"), "w"),
          indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
