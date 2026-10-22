"""E3-QP3's readings from the job log (the clusters part and the short-range diagnostic after it): seed means and
ranges per rung, and every registered prediction of notes/e3qp3-preregistration.md section 4 marked against them.

    python scratch/e3qp3_summarise_log.py logs/e3qp3_<job>.out     # needs scratch/e3qp2_label_bins.json, e3qp2_summary.json
"""
import re, json, os, sys
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
lines = open(sys.argv[1]).read().splitlines()
bins = json.load(open(os.path.join(HERE, "e3qp2_label_bins.json")))
qp2 = json.load(open(os.path.join(HERE, "e3qp2_summary.json")))
HEX_NA = float(re.search(r"size 6 rms ([0-9.]+) kcal/mol", "\n".join(lines)).group(1))      # the test hexamers' non-additive rms

R = defaultdict(lambda: defaultdict(dict))      # R[rung][seed][key]
pes_seen = defaultdict(set)
diag_cols = None
for ln in lines:
    m = re.match(r"\s*(\S+) seed (\d): (.*)$", ln)
    if m:
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
            far, near = rest.split("by smallest O-O")
            bl, bs = {}, {}
            for lo, hi, v, p in re.findall(r"([0-9.]+)-([0-9.]+|inf) A ([0-9.]+) / PES ([0-9.]+)", far):
                bl[f"{lo}-{hi}"] = float(v); pes_seen[(key, "largest", f"{lo}-{hi}")].add(p)
            for lo, hi, v, p in re.findall(r"([0-9.]+)-([0-9.]+|inf) A ([0-9.]+) / PES ([0-9.]+)", near):
                bs[f"{lo}-{hi}"] = float(v); pes_seen[(key, "smallest", f"{lo}-{hi}")].add(p)
            d[key] = {"tot": tot, "strict": strict, "largest": bl, "smallest": bs}
        elif rest.startswith("training clusters"):
            num = r"([0-9]+\.[0-9]+)"
            f = re.match(rf"training clusters E {num} kcal/mol, F rigid {num} / internal {num} kcal/mol/A; contact configurations "
                         rf"E {num} kcal/mol, F rigid {num} / internal {num} kcal/mol/A; hexamers {num} \(2-body {num}, rest {num}\)", rest)
            f = [float(x) for x in f.groups()]
            d["fit"] = dict(train_E=f[0], train_Frig=f[1], train_Fint=f[2], contact_E=f[3], contact_Frig=f[4],
                            contact_Fint=f[5], hex_E=f[6], hex_2b=f[7], hex_rest=f[8])
            assert abs(f[6] - d["hexE"]) < 1e-9, (rung, seed, f[6], d["hexE"])       # the diagnostic re-reads the run's hexamers
        continue
    m = re.match(r"\s*(\db) \(([^)]*), kcal/mol\): (.*)$", ln)
    if m:                                        # a diagnostic line, for the last rung/seed seen
        key, cols, rest = m.group(1), [c.strip() for c in m.group(2).split("/")], m.group(3)
        out = {}
        for part in rest.split("; "):
            mm = re.match(r"(.+?) n (\d+) (.*)$", part.strip())
            name, n, vals = mm.group(1), int(mm.group(2)), [float(x) for x in mm.group(3).split(" / ")]
            out[name] = dict(n=n, **dict(zip(cols, vals)))
        d.setdefault("diag", {})[key] = out

assert all(len(v) == 1 for v in pes_seen.values()), {k: v for k, v in pes_seen.items() if len(v) > 1}
rungs = [r for r, v in R.items() if all("hexE" in d for d in v.values())]
assert all(sorted(R[r]) == [0, 1, 2, 3] for r in rungs), {r: sorted(R[r]) for r in rungs}
pinned = [r for r in rungs if r != "M_inf"]
S = {r: [R[r][i] for i in range(4)] for r in rungs}


def mean(v): return float(np.mean(v))
def seeds(r, f): return [f(S[r][i]) for i in range(4)]
def comb(rows):                                   # rms over bins: [(n, rms), ...]
    n = sum(a for a, _ in rows); return float(np.sqrt(sum(a * b * b for a, b in rows) / n))


n_small = {}                                       # counts of the smallest-O-O bins, from the held-out summary lines
for ln in lines:
    m = re.match(r"\s*held-out (\db): .*by smallest O-O: (.*)$", ln)
    if m:
        n_small[m.group(1)] = {f"{lo}-{hi}": int(n) for lo, hi, n in re.findall(r"([0-9.]+)-([0-9.]+|inf) A n = (\d+)", m.group(2))}
lab = {k: bins[k]["label_rms"] for k in ("2b", "3b", "4b")}

T = {}                                             # the table of seed means
for r in rungs:
    t = {}
    t["train_E"] = seeds(r, lambda d: d["fit"]["train_E"]); t["contact_E"] = seeds(r, lambda d: d["fit"]["contact_E"])
    t["train_Frig"] = seeds(r, lambda d: d["fit"]["train_Frig"]); t["train_Fint"] = seeds(r, lambda d: d["fit"]["train_Fint"])
    t["hex_2b"] = seeds(r, lambda d: d["fit"]["hex_2b"]); t["hex_rest"] = seeds(r, lambda d: d["fit"]["hex_rest"])
    t["hexE"] = seeds(r, lambda d: d["hexE"]); t["hexNA"] = seeds(r, lambda d: d["hexNA"]); t["dimE"] = seeds(r, lambda d: d["dimE"])
    for k in ("2b", "3b", "4b"):
        t[f"{k}"] = seeds(r, lambda d: d[k]["tot"])
        for b in S[r][0][k]["largest"]:
            t[f"{k}_L{b}"] = seeds(r, lambda d: d[k]["largest"][b])
        for b in S[r][0][k]["smallest"]:
            t[f"{k}_S{b}"] = seeds(r, lambda d: d[k]["smallest"][b])
    ns = n_small["2b"]
    t["2b_from2.7"] = seeds(r, lambda d: comb([(ns["2.7-3"], d["2b"]["smallest"]["2.7-3"]), (ns["3-inf"], d["2b"]["smallest"]["3-inf"])]))
    # the diagnostic's 2.3-2.7 bin read with the pins as trained (fixed_labels for the pinned rungs; labels for M_inf)
    col = "fixed_labels" if r != "M_inf" else "labels"
    t["2b_2.3-2.7"] = seeds(r, lambda d: d["diag"]["2b"]["2.3-2.7"][col])
    t["3b_rigid_0-5"] = seeds(r, lambda d: d["diag"]["3b"]["0-5"]["rigid"])
    t["3b_dist_0-5"] = seeds(r, lambda d: d["diag"]["3b"]["0-5"]["fixed_distortion" if r != "M_inf" else "distortion"])
    t["2b_covered"] = seeds(r, lambda d: d["diag"]["2b"]["covered"][col])
    t["2b_notcovered"] = seeds(r, lambda d: d["diag"]["2b"]["not covered"][col])
    for kind in ("test", "test_o1", "test_o2"):
        if kind in S[r][0]:
            t[f"probes_{kind}"] = {dd: [S[r][i][kind]["by_d"][dd] for i in range(4)] for dd in (8, 10, 12, 15)}
    T[r] = t

fmt = lambda v: f"{mean(v):.3f} [{min(v):.3f}-{max(v):.3f}]"
print(f"test hexamers' non-additive rms {HEX_NA} kcal/mol; label rms 2b/3b/4b {lab['2b']:.4f}/{lab['3b']:.4f}/{lab['4b']:.4f}")
for k in ("train_E", "contact_E", "train_Frig", "train_Fint", "hexE", "hex_2b", "hex_rest", "hexNA", "dimE", "2b", "2b_S0-2.7",
          "2b_2.3-2.7", "2b_from2.7", "2b_S3-inf", "2b_covered", "2b_notcovered", "3b", "3b_L0-5", "3b_S0-2.7", "3b_S2.7-3",
          "3b_S3-inf", "3b_rigid_0-5", "3b_dist_0-5", "3b_L7-9", "3b_L9-inf", "4b", "4b_L0-5", "4b_S0-2.7"):
    print(f"{k:14s} " + "  ".join(f"{r}: {fmt(T[r][k])}" for r in rungs))
print(f"{'hexNA %':14s} " + "  ".join(f"{r}: {100 * mean(T[r]['hexNA']) / HEX_NA:.1f}" for r in rungs))
print(f"{'4b %':14s} " + "  ".join(f"{r}: {100 * mean(T[r]['4b']) / lab['4b']:.0f}" for r in rungs))
lab9 = bins["3b"]["bins"]["9-inf"]["label_rms"]; lab79 = bins["3b"]["bins"]["7-9"]["label_rms"]
print(f"{'3b >=9 %':14s} " + "  ".join(f"{r}: {100 * mean(T[r]['3b_L9-inf']) / lab9:.1f}" for r in rungs) + f"   (labels {lab9:.4f})")
print(f"{'3b 7-9 %':14s} " + "  ".join(f"{r}: {100 * mean(T[r]['3b_L7-9']) / lab79:.1f}" for r in rungs))

# E3-QP2 alongside (its four seeds; the 2- and 3-body read with the pins at the frames' geometry)
print("\nE3-QP2 (seed means): " + "; ".join(
    f"{r}: hexE {qp2[r]['hexE'][0]:.3f}, hexNA {100 * qp2[r]['hexNA'][0] / HEX_NA:.1f} %, dimers {qp2[r]['dimE'][0]:.3f}, "
    f"2b {qp2[r]['2b']['tot'][0]:.3f}, 3b {qp2[r]['3b']['tot'][0]:.3f}, 3b 0-5 {qp2[r]['3b']['bins']['0-5'][0]:.3f}, "
    f"4b {100 * qp2[r]['4b']['tot'][0] / lab['4b']:.0f} %" for r in ("M_inf", "M_16pm", "M_16C2pm")))

# ---------------------------------------------------------------------------------------------- the predictions
V = []
def check(text, ok, detail):
    V.append((text, bool(ok), detail)); print(f"  [{'hit ' if ok else 'MISS'}] {text}: {detail}")

print("\nComputed (to the printed digit, every seed, _e1 = base):")
for r in ("M_16C2pm", "M_16C2pm_e1"):
    p, o1, o2 = T[r]["probes_test"], T[r]["probes_test_o1"], T[r]["probes_test_o2"]
    check(f"{r} test probes 10/12/15 A non-additive 4/5/3 %, interaction 0/0/0 %",
          all(p[dd] == [(0, x)] * 4 for dd, x in ((10, 4), (12, 5), (15, 3))), {dd: p[dd] for dd in (10, 12, 15)})
    check(f"{r} truncations at 10/12/15 A: order 1 15/15/14 %, order 2 5/5/4 %",
          all(o1[dd][i][1] == a and o2[dd][i][1] == b for dd, a, b in ((10, 15, 5), (12, 15, 5), (15, 14, 4)) for i in range(4)),
          {dd: (o1[dd], o2[dd]) for dd in (10, 12, 15)})
for r in ("M_16pm", "M_16pm_e1"):
    p = T[r]["probes_test"]
    check(f"{r} test probes 10/12/15 A non-additive 100 %, interaction 10/9/15 %",
          all(p[dd] == [(a, 100)] * 4 for dd, a in ((10, 10), (12, 9), (15, 15))), {dd: p[dd] for dd in (10, 12, 15)})

print("\nExpected (seed means unless said):")
for r in pinned:
    check(f"{r} training clusters' energy <= 0.10 on every seed", max(T[r]["train_E"]) <= 0.10, fmt(T[r]["train_E"]))
check("M_inf training clusters' energy <= 0.15 on every seed", max(T["M_inf"]["train_E"]) <= 0.15, fmt(T["M_inf"]["train_E"]))
check("contact configurations' energy <= 0.15 (every rung)", all(mean(T[r]["contact_E"]) <= 0.15 for r in rungs),
      {r: round(mean(T[r]["contact_E"]), 3) for r in rungs})
check("training forces' rigid-body part <= 1.0 kcal/mol/A per molecule", all(max(T[r]["train_Frig"]) <= 1.0 for r in rungs),
      {r: round(max(T[r]["train_Frig"]), 3) for r in rungs})
for b in ("2b_2.3-2.7", "2b_S0-2.7"):
    check(f"compressed dimers ({b}) <= 0.30 for every rung", all(mean(T[r][b]) <= 0.30 for r in rungs),
          {r: round(mean(T[r][b]), 3) for r in rungs})
    check(f"compressed dimers ({b}): pinned rungs within 0.05 of M_inf or below",
          all(mean(T[r][b]) <= mean(T["M_inf"][b]) + 0.05 for r in pinned), {r: round(mean(T[r][b]), 3) for r in rungs})
check("dimers from 2.7 A: pinned <= 0.10", all(mean(T[r]["2b_from2.7"]) <= 0.10 for r in pinned),
      {r: round(mean(T[r]["2b_from2.7"]), 4) for r in rungs})
check("dimers from 2.7 A: M_inf <= 0.13", mean(T["M_inf"]["2b_from2.7"]) <= 0.13, round(mean(T["M_inf"]["2b_from2.7"]), 4))
check("dimers from 2.7 A: pinned below M_inf on every seed (paired)",
      all(T[r]["2b_from2.7"][i] < T["M_inf"]["2b_from2.7"][i] for r in pinned for i in range(4)),
      {r: [round(x, 4) for x in T[r]["2b_from2.7"]] for r in rungs})
check("raw 2-body, all: every rung <= 0.15", all(mean(T[r]["2b"]) <= 0.15 for r in rungs), {r: round(mean(T[r]["2b"]), 4) for r in rungs})
check("raw 2-body, all: pinned below M_inf on >= 3 of 4 seeds (paired)",
      all(sum(T[r]["2b"][i] < T["M_inf"]["2b"][i] for i in range(4)) >= 3 for r in pinned),
      {r: sum(T[r]["2b"][i] < T["M_inf"]["2b"][i] for i in range(4)) for r in pinned})
check("raw 3-body: every rung <= 0.12 overall", all(mean(T[r]["3b"]) <= 0.12 for r in rungs), {r: round(mean(T[r]["3b"]), 4) for r in rungs})
check("raw 3-body: every rung <= 0.14 on the compact trimers (largest O-O < 5 A)", all(mean(T[r]["3b_L0-5"]) <= 0.14 for r in rungs),
      {r: round(mean(T[r]["3b_L0-5"]), 4) for r in rungs})
check("(open) compact trimers: pinned at or below M_inf", all(mean(T[r]["3b_L0-5"]) <= mean(T["M_inf"]["3b_L0-5"]) for r in pinned),
      {r: round(mean(T[r]["3b_L0-5"]), 4) for r in rungs})
check("hexamers: pinned energy <= 0.15", all(mean(T[r]["hexE"]) <= 0.15 for r in pinned), {r: round(mean(T[r]["hexE"]), 3) for r in rungs})
check("hexamers: M_16C2pm energy <= 0.12", mean(T["M_16C2pm"]["hexE"]) <= 0.12, round(mean(T["M_16C2pm"]["hexE"]), 3))
check("hexamers: M_inf energy 0.35-0.55", 0.35 <= mean(T["M_inf"]["hexE"]) <= 0.55, round(mean(T["M_inf"]["hexE"]), 3))
pct = lambda r: 100 * mean(T[r]["hexNA"]) / HEX_NA
check("hexamers' non-additive: M_16C2pm <= 16 %", pct("M_16C2pm") <= 16, round(pct("M_16C2pm"), 2))
check("hexamers' non-additive: M_16pm <= 40 %", pct("M_16pm") <= 40, round(pct("M_16pm"), 1))
check("hexamers' non-additive: M_inf <= 25 %", pct("M_inf") <= 25, round(pct("M_inf"), 1))
check("test dimers: pinned <= 0.05", all(mean(T[r]["dimE"]) <= 0.05 for r in pinned), {r: round(mean(T[r]["dimE"]), 4) for r in rungs})
check("test dimers: M_inf <= 0.08", mean(T["M_inf"]["dimE"]) <= 0.08, round(mean(T["M_inf"]["dimE"]), 4))
check("raw 4-body: M_16C2pm <= 63 %", 100 * mean(T["M_16C2pm"]["4b"]) / lab["4b"] <= 63, round(100 * mean(T["M_16C2pm"]["4b"]) / lab["4b"], 1))
for r in rungs:
    if r in qp2:
        now = [x[1] for x in T[r]["probes_test"][8]]; then = [x[1] for x in qp2[r]["probes_test"]["8"]]
        check(f"{r} test probes at 8 A within +-6 points of E3-QP2's (seed means)", abs(mean(now) - mean(then)) <= 6,
              f"{mean(now):.1f} against {mean(then):.1f} %")
for r in rungs:
    v = 100 * mean(T[r]["3b_L9-inf"]) / lab9
    ok = (22 <= v <= 28) if "C2" in r else (95 <= v <= 105)
    check(f"{r} CCSD(T) trimers beyond 9 A {'25 +- 3' if 'C2' in r else '95-105'} %", ok, f"{v:.1f} %")
for base in ("M_16pm", "M_16C2pm"):
    e1 = base + "_e1"
    for b in ("2b_2.3-2.7", "2b_S0-2.7"):
        k = sum(T[e1][b][i] <= T[base][b][i] for i in range(4))
        check(f"(open) {e1} at or below {base} on the compressed dimers ({b}) on >= 3 of 4 seeds", k >= 3,
              f"{k} of 4: {[round(x, 4) for x in T[e1][b]]} against {[round(x, 4) for x in T[base][b]]}")
    check(f"(open) {e1} within 0.01 of {base} from O-O 3.0 A (seed means)", abs(mean(T[e1]["2b_S3-inf"]) - mean(T[base]["2b_S3-inf"])) <= 0.01,
          f"{mean(T[e1]['2b_S3-inf']):.4f} against {mean(T[base]['2b_S3-inf']):.4f}")
print(f"\n{sum(ok for _, ok, _ in V)} of {len(V)} readings as registered")
json.dump({"rungs": T, "verdicts": V, "hex_na_rms": HEX_NA, "n_smallest": n_small}, open(os.path.join(HERE, "e3qp3_summary.json"), "w"),
          indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
