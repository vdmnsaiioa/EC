"""E3-QP5's readings from its two job logs (each the clusters part and the short-range diagnostic after it), against
E3-QP4's job B (its log), and the registered predictions of notes/e3qp5-preregistration.md section 4 marked: seed means
and ranges per rung, the field rungs paired seed by seed with their bases, and the run part read against the diagnostic
(the same numbers twice, a check of the logs as transcribed).

    python scratch/e3qp5_summarise_log.py logs/e3qp5a_<job>.out logs/e3qp5b_<job>.out logs/e3qp4b_<job>.out
"""
import re, json, os, sys
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
bins = json.load(open(os.path.join(HERE, "e3qp2_label_bins.json")))
LAB = {k: bins[k]["label_rms"] for k in ("2b", "3b", "4b")}
LAB9 = bins["3b"]["bins"]["9-inf"]["label_rms"]
NUM = r"([0-9]+\.[0-9]+)"


def parse(path):
    """R[rung][seed] = readings of one job's log, the read-out solves' lines, the test hexamers' non-additive rms."""
    lines = open(path).read().splitlines()
    R = defaultdict(lambda: defaultdict(dict))
    solves, d = defaultdict(list), None
    hex_na = float(re.search(r"size 6 rms ([0-9.]+) kcal/mol", "\n".join(lines)).group(1))
    current = None
    for ln in lines:
        m = re.match(r"\s*(\S+): the read-out matched on", ln)
        if m:
            current = m.group(1); continue
        m = re.match(r"\s*read-out linear solve on \d+ structures: (.*?)\s+\[\d+ s\]$", ln)
        if m and current:
            solves[current].append(m.group(1)); continue
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
                d[kind] = {int(dd): (int(a), int(b)) for dd, a, b in re.findall(r"(\d+) A (\d+) / (\d+) %", rest)}
            elif rest.startswith("held-out"):
                key = re.match(r"held-out (\db)", rest).group(1)
                far, near = rest.split("by smallest O-O")
                d[key] = {"tot": float(re.search(r"rmse ([0-9.]+) kcal/mol", rest).group(1)),
                          "strict": float(re.search(r"strict ([0-9.]+)", rest).group(1)),
                          "largest": {f"{lo}-{hi}": float(v) for lo, hi, v, _ in
                                      re.findall(r"([0-9.]+)-([0-9.]+|inf) A ([0-9.]+) / PES ([0-9.]+)", far)},
                          "smallest": {f"{lo}-{hi}": float(v) for lo, hi, v, _ in
                                       re.findall(r"([0-9.]+)-([0-9.]+|inf) A ([0-9.]+) / PES ([0-9.]+)", near)}}
            elif rest.startswith("training clusters"):
                f = re.match(rf"training clusters E {NUM} kcal/mol, F rigid {NUM} / internal {NUM} kcal/mol/A; contact configurations "
                             rf"E {NUM} kcal/mol, F rigid {NUM} / internal {NUM} kcal/mol/A; hexamers {NUM} \(2-body {NUM}, rest {NUM}\)", rest)
                f = [float(x) for x in f.groups()]
                assert abs(f[6] - d["hexE"]) < 1e-9, (path, rung, seed)
                d["fit"] = dict(train_E=f[0], train_Frig=f[1], train_Fint=f[2], contact_E=f[3], contact_Frig=f[4], contact_Fint=f[5],
                                hex_2b=f[7], hex_rest=f[8])
            continue
        m = re.match(r"\s*(\db) \(([^)]*), kcal/mol\): (.*)$", ln)
        if m and d is not None:
            cols = [c.strip() for c in m.group(2).split("/")]
            out = {}
            for part in m.group(3).split("; "):
                mm = re.match(r"(.+?) n (\d+) (.*)$", part.strip())
                out[mm.group(1)] = dict(n=int(mm.group(2)), **dict(zip(cols, [float(x) for x in mm.group(3).split(" / ")])))
            d.setdefault("diag", {})[m.group(1)] = out
    rungs = [r for r, v in R.items() if all("hexE" in x for x in v.values())]
    assert all(sorted(R[r]) == [0, 1, 2, 3] for r in rungs), {r: sorted(R[r]) for r in rungs}
    return {r: [R[r][i] for i in range(4)] for r in rungs}, dict(solves), hex_na


def check_transcription(S, label):
    """the run part's held-out readings are the diagnostic's with the pins as trained (or, without pins, its labels):
    every number read twice must agree to its printed digits."""
    bad = 0
    for r, seeds in S.items():
        for i, x in enumerate(seeds):
            col = "fixed_labels" if "fixed_labels" in x["diag"]["3b"]["all"] else "labels"
            for k in ("2b", "3b", "4b"):
                dg = x["diag"][k]
                pairs = [(x[k]["tot"], dg["all"][col]), (x[k]["strict"], dg["strict"][col])]
                lmap = {"0-5": "0-5", "5-7": "5-7", "7-9": "7-9", "9-inf": "9-99"} if k == "3b" else \
                       ({"0-5": "0-5", "5-7": "5-7", "7-9": "7-99"} if k == "4b" else {})
                for b, db in lmap.items():
                    if b in x[k]["largest"] and db in dg:
                        pairs.append((x[k]["largest"][b], dg[db][col]))
                if k != "2b":
                    for b, db in (("0-2.7", "smallest 0-2.7"), ("2.7-3", "smallest 2.7-3"), ("3-inf", "smallest 3-99")):
                        pairs.append((x[k]["smallest"][b], dg[db][col]))
                for a, b in pairs:
                    if abs(a - b) > 1.5e-4:
                        bad += 1; print(f"  [transcription?] {label} {r} seed {i} {k}: run {a} diagnostic {b}")
    print(f"{label}: the run part against the diagnostic, {'every number agrees' if not bad else f'{bad} disagree'}")


def table(S):
    """seed lists of every reading, per rung."""
    T = {}
    for r, seeds in S.items():
        col = "fixed_labels" if "fixed_labels" in seeds[0]["diag"]["3b"]["all"] else "labels"
        dis = "fixed_distortion" if col == "fixed_labels" else "distortion"
        g = lambda f: [f(x) for x in seeds]
        t = {k: g(lambda x, k=k: x["fit"][k]) for k in ("train_E", "contact_E", "train_Frig", "train_Fint", "hex_2b", "hex_rest")}
        t.update(hexE=g(lambda x: x["hexE"]), hexNA=g(lambda x: x["hexNA"]), dimE=g(lambda x: x["dimE"]))
        for k in ("2b", "3b", "4b"):
            t[k] = g(lambda x, k=k: x[k]["tot"])
            t[f"{k}_strict"] = g(lambda x, k=k: x[k]["strict"])
            t[f"{k}_rigid"] = g(lambda x, k=k: x["diag"][k]["all"]["rigid"])
            t[f"{k}_dist"] = g(lambda x, k=k: x["diag"][k]["all"][dis])
            for b in seeds[0][k]["largest"]:
                t[f"{k}_L{b}"] = g(lambda x, k=k, b=b: x[k]["largest"][b])
            for b in seeds[0][k]["smallest"]:
                t[f"{k}_S{b}"] = g(lambda x, k=k, b=b: x[k]["smallest"][b])
        for b in ("2.3-2.7", "2.7-2.9", "2.9-3.2", "3.2-5", "5-99"):
            t[f"2b_{b}"] = g(lambda x, b=b: x["diag"]["2b"][b][col])
        t["3b_strict_rigid"] = g(lambda x: x["diag"]["3b"]["strict"]["rigid"])
        t["2b_covered"] = g(lambda x: x["diag"]["2b"]["covered"][col])
        t["2b_uncovered"] = g(lambda x: x["diag"]["2b"]["not covered"][col])
        # the 2.9-3.2 A bin split at 3.0 A, the upper edge of the contact configurations' O-O range: from the bins by the
        # smallest O-O (3-inf: 3.0-3.2 + 3.2-5 + 5-99) and the diagnostic's 2.9-3.2, 3.2-5, 5-99, each an rms over its frames
        def split(x, n=(3002, 638, 88, 1487)):
            dg = x["diag"]["2b"]; sq = lambda b_, n_: n_ * dg[b_][col] ** 2
            hi = (n[3] * x["2b"]["smallest"]["3-inf"] ** 2 - sq("3.2-5", n[1]) - sq("5-99", n[2])) / (n[3] - n[1] - n[2])
            lo = (sq("2.9-3.2", n[0]) - (n[3] - n[1] - n[2]) * hi) / (n[0] - (n[3] - n[1] - n[2]))
            return float(np.sqrt(max(lo, 0.0))), float(np.sqrt(max(hi, 0.0)))
        t["2b_2.9-3.0"] = g(lambda x: split(x)[0]); t["2b_3.0-3.2"] = g(lambda x: split(x)[1])
        for kind in ("train", "test", "test_o1", "test_o2"):
            if kind in seeds[0]:
                t[f"probes_{kind}"] = {dd: [x[kind][dd] for x in seeds] for dd in (8, 10, 12, 15)}
        T[r] = t
    return T


mean = lambda v: float(np.mean(v))
fmt = lambda v: f"{mean(v):.4f} [{min(v):.4f}-{max(v):.4f}]"
Sa, solves_a, HEX_NA = parse(sys.argv[1])
Sb, solves_b, _ = parse(sys.argv[2])
SB, solves_B, _ = parse(sys.argv[3])
check_transcription(Sa, "job a"); check_transcription(Sb, "job b"); check_transcription(SB, "E3-QP4 job B")
Q = {**table(Sa), **table(Sb)}
B = table(SB)
order = [r for r in ("M_16pm", "M_16pm_f", "M_16C2pm", "M_16C2pm_f", "M_inf_f") if r in Q]
KEYS = ("train_E", "train_Frig", "train_Fint", "contact_E", "hexE", "hex_2b", "hex_rest", "dimE",
        "2b", "2b_2.3-2.7", "2b_2.7-2.9", "2b_2.9-3.2", "2b_2.9-3.0", "2b_3.0-3.2", "2b_3.2-5", "2b_5-99", "2b_S3-inf", "2b_L5-7",
        "2b_covered", "2b_uncovered",
        "3b", "3b_strict", "3b_rigid", "3b_dist", "3b_S0-2.7", "3b_S2.7-3", "3b_S3-inf", "3b_L0-5", "3b_L5-7", "3b_L7-9", "3b_L9-inf",
        "4b", "4b_strict", "4b_S0-2.7")
print(f"\nE3-QP5 (seed means [range]); E3-QP4 job B alongside where the rung ran there; the hexamers' non-additive rms {HEX_NA}")
for k in KEYS:
    print(f"{k:13s} " + "  ".join(f"{r}: {fmt(Q[r][k])}" + (f" ({mean(B[r][k]):.4f})" if r in B else "") for r in order)
          + f"  | job B M_inf {mean(B['M_inf'][k]):.4f}")
print("hexNA %       " + "  ".join(f"{r}: {100 * mean(Q[r]['hexNA']) / HEX_NA:.1f}" for r in order) + f"  | job B M_inf {100 * mean(B['M_inf']['hexNA']) / HEX_NA:.1f}")
print("4b %          " + "  ".join(f"{r}: {100 * mean(Q[r]['4b']) / LAB['4b']:.0f}" for r in order) + f"  | job B M_inf {100 * mean(B['M_inf']['4b']) / LAB['4b']:.0f}")
print("3b >=9 %      " + "  ".join(f"{r}: {100 * mean(Q[r]['3b_L9-inf']) / LAB9:.1f}" for r in order) + f"  | job B M_inf {100 * mean(B['M_inf']['3b_L9-inf']) / LAB9:.1f}")

PAIRS = (("M_16pm_f", "M_16pm"), ("M_16C2pm_f", "M_16C2pm"))
print("\nthe field rungs against their bases, paired seed by seed: change in % (seeds 0-3; mean of the seeds' changes)")
for f, b in PAIRS:
    for k in KEYS + ("hexNA",):
        ch = [100 * (x / y - 1) for x, y in zip(Q[f][k], Q[b][k])]
        print(f"  {f:11s} {k:13s} " + " ".join(f"{c:+5.0f}" for c in ch) + f"   {np.mean(ch):+5.0f}   ({mean(Q[f][k]):.4f} against {mean(Q[b][k]):.4f})")
print("\nM_inf_f against job B's M_inf (seed means): change in %")
print("  " + ", ".join(f"{k} {100 * (mean(Q['M_inf_f'][k]) / mean(B['M_inf'][k]) - 1):+.0f}" for k in KEYS + ("hexNA",)))
print("\nthe controls against job B: seed by seed, and the seed mean inside job B's range")
for r in ("M_16pm", "M_16C2pm"):
    outside = [k for k in KEYS + ("hexNA",) if not min(B[r][k]) <= mean(Q[r][k]) <= max(B[r][k])]
    same = sum(np.allclose(Q[r][k], B[r][k], rtol=0, atol=1e-12) for k in KEYS)
    print(f"  {r}: {same} of {len(KEYS)} readings identical to job B's on every seed; seed means outside job B's range: "
          + (", ".join(f"{k} {mean(Q[r][k]):.4f} [{min(B[r][k]):.4f}-{max(B[r][k]):.4f}]" for k in outside) or "none"))
print("\nthe 8 A probes (test; non-additive %, per seed): " + "; ".join(
    f"{r} {[x[1] for x in Q[r]['probes_test'][8]]}" for r in order) + "  | train: " + "; ".join(
    f"{r} {[x[1] for x in Q[r]['probes_train'][8]]}" for r in order))

V = []
def check(text, ok, detail):
    V.append((text, bool(ok), detail))
    print(f"  [{'hit ' if ok else 'MISS'}] {text}: {detail}")

print("\nComputed (every seed):")
for r in ("M_16C2pm_f",):
    t = Q[r]
    ok = all(t["probes_test"][dd][i] == (0, na) for dd, na in ((10, 4), (12, 5), (15, 3)) for i in range(4)) \
        and all(t["probes_test_o1"][dd][i][1] == na for dd, na in ((10, 15), (12, 15), (15, 14)) for i in range(4)) \
        and all(t["probes_test_o2"][dd][i][1] == na for dd, na in ((10, 5), (12, 5), (15, 4)) for i in range(4))
    check(f"{r}: test probes' non-additive 4 / 5 / 3 % at 10 / 12 / 15 A, interaction 0; order 1 15 / 15 / 14, order 2 5 / 5 / 4",
          ok, {dd: (t["probes_test"][dd][0], t["probes_test_o1"][dd][0][1], t["probes_test_o2"][dd][0][1]) for dd in (10, 12, 15)})
check("M_16pm_f: interaction 10 / 9 / 15 %, non-additive 100 % at 10 / 12 / 15 A",
      all(Q["M_16pm_f"]["probes_test"][dd][i] == (ia, 100) for dd, ia in ((10, 10), (12, 9), (15, 15)) for i in range(4)),
      {dd: Q["M_16pm_f"]["probes_test"][dd] for dd in (10, 12, 15)})
check("M_inf_f: 100 % at 10 / 12 / 15 A", all(Q["M_inf_f"]["probes_test"][dd][i] == (100, 100) for dd in (10, 12, 15) for i in range(4)),
      {dd: Q["M_inf_f"]["probes_test"][dd][0] for dd in (10, 12, 15)})
check("M_16C2pm_f's read-out solve M_16C2pm's, step for step", solves_b.get("M_16C2pm_f") == solves_b.get("M_16C2pm"),
      f"{len(solves_b.get('M_16C2pm_f', []))} solves")

print("\nExpected:")
f, b = Q["M_16pm_f"], Q["M_16pm"]
k = sum(f["3b_strict"][i] <= 0.85 * b["3b_strict"][i] for i in range(4))
check("(b): M_16pm_f's strict-subset 3-body >= 15 % below M_16pm's on >= 3 of 4 seeds, paired", k >= 3,
      f"{k} of 4: {f['3b_strict']} against {b['3b_strict']}")
check("(b): its mean <= 0.09", mean(f["3b_strict"]) <= 0.09, f"{mean(f['3b_strict']):.4f}")
check("(b): the rigid part <= 0.09", mean(f["3b_rigid"]) <= 0.09, f"{mean(f['3b_rigid']):.4f}")
check("(b): all raw trimers <= 0.10", mean(f["3b"]) <= 0.10, f"{mean(f['3b']):.4f}")
check("no price: raw 2-body at most 0.005 above M_16pm's", mean(f["2b"]) <= mean(b["2b"]) + 0.005, f"{mean(f['2b']):.4f} against {mean(b['2b']):.4f}")
check("no price: hexamers at or below M_16pm's", mean(f["hexE"]) <= mean(b["hexE"]), f"{mean(f['hexE']):.4f} against {mean(b['hexE']):.4f}")
check("no price: training clusters' energy at or below M_16pm's", mean(f["train_E"]) <= mean(b["train_E"]),
      f"{mean(f['train_E']):.4f} against {mean(b['train_E']):.4f}")
check("no price: raw 4-body not above M_16pm's", mean(f["4b"]) <= mean(b["4b"]), f"{mean(f['4b']):.4f} against {mean(b['4b']):.4f}")
g_f = 1 - mean(f["3b_strict"]) / mean(b["3b_strict"])
g_inf = 1 - mean(Q["M_inf_f"]["3b_strict"]) / mean(B["M_inf"]["3b_strict"])
check("the pinned rungs' piece: M_inf_f's strict-subset gain (against job B's M_inf) less than half M_16pm_f's", g_inf < 0.5 * g_f,
      f"M_inf_f {100 * g_inf:+.0f} % ({mean(Q['M_inf_f']['3b_strict']):.4f} against {mean(B['M_inf']['3b_strict']):.4f}), "
      f"M_16pm_f {100 * g_f:+.0f} %")
fc, bc = Q["M_16C2pm_f"], Q["M_16C2pm"]
k = sum(fc["3b_S0-2.7"][i] <= 0.85 * bc["3b_S0-2.7"][i] for i in range(4))
check("the C rung: M_16C2pm_f's raw 3-body below 2.7 A >= 15 % below M_16C2pm's on >= 3 of 4 seeds, paired", k >= 3,
      f"{k} of 4: {fc['3b_S0-2.7']} against {bc['3b_S0-2.7']}")
check("the C rung: hexamers at or below M_16C2pm's", mean(fc["hexE"]) <= mean(bc["hexE"]), f"{mean(fc['hexE']):.4f} against {mean(bc['hexE']):.4f}")
check("the C rung: hexamers' non-additive at or below M_16C2pm's", mean(fc["hexNA"]) <= mean(bc["hexNA"]),
      f"{100 * mean(fc['hexNA']) / HEX_NA:.1f} against {100 * mean(bc['hexNA']) / HEX_NA:.1f} %")
print("\nKept:")
check("hexamers' non-additive M_16C2pm <= 17 %", 100 * mean(bc["hexNA"]) / HEX_NA <= 17, f"{100 * mean(bc['hexNA']) / HEX_NA:.1f} %")
for r in order:
    v = 100 * mean(Q[r]["3b_L9-inf"]) / LAB9
    lo, hi = (24, 28) if "C2" in r else (95, 105)
    check(f"{r}: trimers beyond 9 A {lo}-{hi} %", lo <= v <= hi, f"{v:.1f} %")
for r in ("M_16pm", "M_16C2pm"):
    outside = [k for k in KEYS + ("hexNA",) if not min(B[r][k]) <= mean(Q[r][k]) <= max(B[r][k])]
    check(f"the control {r}: every seed mean within job B's range", not outside, ", ".join(outside) or "all inside")
print(f"\n{sum(ok for _, ok, _ in V)} of {len(V)} checks as registered")
json.dump({"Q": Q, "B": B, "verdicts": V, "hex_na_rms": HEX_NA}, open(os.path.join(HERE, "e3qp5_summary.json"), "w"), indent=1,
          default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
