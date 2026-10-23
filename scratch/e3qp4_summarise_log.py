"""E3-QP4's readings from its job logs (the clusters part and the short-range diagnostic after it): seed means and ranges per
rung, set against E3-QP3's (scratch/e3qp3_summary.json), and the registered predictions of notes/e3qp4-preregistration.md
section 4 marked against them -- job A's alone when job B's log is not given.

    python scratch/e3qp4_summarise_log.py logs/e3qp4a_<job>.out [logs/e3qp4b_<job>.out]
"""
import re, json, os, sys
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
bins = json.load(open(os.path.join(HERE, "e3qp2_label_bins.json")))
qp3 = json.load(open(os.path.join(HERE, "e3qp3_summary.json")))["rungs"]
LAB = {k: bins[k]["label_rms"] for k in ("2b", "3b", "4b")}
LAB9 = bins["3b"]["bins"]["9-inf"]["label_rms"]
NUM = r"([0-9]+\.[0-9]+)"


def parse(path):
    """R[rung][seed] = readings of one job's log; the smallest-O-O bin counts; the test hexamers' non-additive rms."""
    lines = open(path).read().splitlines()
    R = defaultdict(lambda: defaultdict(dict))
    n_small, d = {}, None
    hex_na = float(re.search(r"size 6 rms ([0-9.]+) kcal/mol", "\n".join(lines)).group(1))
    for ln in lines:
        m = re.match(r"\s*held-out (\db): .*by smallest O-O: (.*)$", ln)
        if m:
            n_small[m.group(1)] = {f"{lo}-{hi}": int(n) for lo, hi, n in re.findall(r"([0-9.]+)-([0-9.]+|inf) A n = (\d+)", m.group(2))}
            continue
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
                          "largest": {f"{lo}-{hi}": float(v) for lo, hi, v, _ in
                                      re.findall(r"([0-9.]+)-([0-9.]+|inf) A ([0-9.]+) / PES ([0-9.]+)", far)},
                          "smallest": {f"{lo}-{hi}": float(v) for lo, hi, v, _ in
                                       re.findall(r"([0-9.]+)-([0-9.]+|inf) A ([0-9.]+) / PES ([0-9.]+)", near)}}
            elif rest.startswith("training clusters"):
                f = re.match(rf"training clusters E {NUM} kcal/mol, F rigid {NUM} / internal {NUM} kcal/mol/A; contact configurations "
                             rf"E {NUM} kcal/mol, F rigid {NUM} / internal {NUM} kcal/mol/A; hexamers {NUM} \(2-body {NUM}, rest {NUM}\)", rest)
                f = [float(x) for x in f.groups()]
                assert abs(f[6] - d["hexE"]) < 1e-9, (rung, seed)
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
    return {r: [R[r][i] for i in range(4)] for r in rungs}, n_small, hex_na


def table(S, n_small):
    """seed lists of every reading, per rung (the same keys as scratch/e3qp3_summary.json)."""
    comb = lambda rows: float(np.sqrt(sum(a * b * b for a, b in rows) / sum(a for a, _ in rows)))
    T = {}
    for r, seeds in S.items():
        col = "labels" if r == "M_inf" else "fixed_labels"
        g = lambda f: [f(x) for x in seeds]
        t = {k: g(lambda x, k=k: x["fit"][k]) for k in ("train_E", "contact_E", "train_Frig", "train_Fint", "hex_2b", "hex_rest")}
        t.update(hexE=g(lambda x: x["hexE"]), hexNA=g(lambda x: x["hexNA"]), dimE=g(lambda x: x["dimE"]))
        for k in ("2b", "3b", "4b"):
            t[k] = g(lambda x, k=k: x[k]["tot"])
            for b in seeds[0][k]["largest"]:
                t[f"{k}_L{b}"] = g(lambda x, k=k, b=b: x[k]["largest"][b])
            for b in seeds[0][k]["smallest"]:
                t[f"{k}_S{b}"] = g(lambda x, k=k, b=b: x[k]["smallest"][b])
        ns = n_small["2b"]
        t["2b_from2.7"] = g(lambda x: comb([(ns["2.7-3"], x["2b"]["smallest"]["2.7-3"]), (ns["3-inf"], x["2b"]["smallest"]["3-inf"])]))
        t["2b_2.3-2.7"] = g(lambda x: x["diag"]["2b"]["2.3-2.7"][col])
        t["3b_rigid_0-5"] = g(lambda x: x["diag"]["3b"]["0-5"]["rigid"])
        for kind in ("test", "test_o1", "test_o2"):
            if kind in seeds[0]:
                t[f"probes_{kind}"] = {dd: [x[kind][dd] for x in seeds] for dd in (8, 10, 12, 15)}
        T[r] = t
    return T


mean = lambda v: float(np.mean(v))
fmt = lambda v: f"{mean(v):.3f} [{min(v):.3f}-{max(v):.3f}]"
SA, nsA, HEX_NA = parse(sys.argv[1])
A = table(SA, nsA)
B = None
if len(sys.argv) > 2:
    SB, nsB, _ = parse(sys.argv[2])
    B = table(SB, nsB)
rungs = list(A)
print(f"job A rungs {rungs}; the test hexamers' non-additive rms {HEX_NA} kcal/mol")
KEYS = ("train_E", "train_Frig", "train_Fint", "contact_E", "hexE", "hex_2b", "hex_rest", "dimE", "2b", "2b_2.3-2.7", "2b_from2.7",
        "3b", "3b_L0-5", "3b_S0-2.7", "3b_S2.7-3", "3b_S3-inf", "3b_rigid_0-5", "4b", "4b_S0-2.7")
for label, T in (("job A", A),) + ((("job B", B),) if B else ()):
    print(f"\n{label} (seed means [range]); E3-QP3 alongside where the rung ran there")
    for k in KEYS:
        print(f"{k:13s} " + "  ".join(f"{r}: {fmt(T[r][k])}" + (f" ({mean(qp3[r][k]):.3f})" if r in qp3 and k in qp3[r] else "")
                                      for r in rungs))
    print(f"{'hexNA %':13s} " + "  ".join(f"{r}: {100 * mean(T[r]['hexNA']) / HEX_NA:.1f}" for r in rungs))
    print(f"{'4b %':13s} " + "  ".join(f"{r}: {100 * mean(T[r]['4b']) / LAB['4b']:.0f}" for r in rungs))
    print(f"{'3b >=9 %':13s} " + "  ".join(f"{r}: {100 * mean(T[r]['3b_L9-inf']) / LAB9:.1f}" for r in rungs))
print("\nthe width (job A against E3-QP3, seed means): change in %")
for r in ("M_inf", "M_16pm", "M_16C2pm"):
    print(f"  {r:9s} " + ", ".join(f"{k} {100 * (mean(A[r][k]) / mean(qp3[r][k]) - 1):+.0f}" for k in
                                  ("train_E", "train_Frig", "train_Fint", "contact_E", "2b", "2b_2.3-2.7", "3b", "3b_S0-2.7", "3b_S2.7-3",
                                   "hexE", "hexNA", "dimE", "4b")))

V = []
def check(text, ok, detail):
    V.append((text, bool(ok), detail)); print(f"  [{'hit ' if ok else 'MISS'}] {text}: {detail}")

print("\nComputed (every seed, both jobs):")
for job, T in (("A", A),) + ((("B", B),) if B else ()):
    for r, nonadd, o1, o2 in (("M_16C2pm", (4, 5, 3), (15, 15, 14), (5, 5, 4)), ("M_16C2pm_r1", (3, 3, 2), (14, 15, 13), (3, 3, 3))):
        p, q1, q2 = T[r]["probes_test"], T[r]["probes_test_o1"], T[r]["probes_test_o2"]
        check(f"job {job}, {r}: test probes' non-additive at 10/12/15 A {nonadd}, interaction 0; order 1 {o1}; order 2 {o2}",
              all(p[dd] == [(0, x)] * 4 for dd, x in zip((10, 12, 15), nonadd))
              and all(q1[dd][i][1] == a and q2[dd][i][1] == b for dd, a, b in zip((10, 12, 15), o1, o2) for i in range(4)),
              {dd: (p[dd][0], q1[dd][0], q2[dd][0]) for dd in (10, 12, 15)})
    check(f"job {job}, M_16pm: non-additive 100 %, interaction 10/9/15 % at 10/12/15 A",
          all(T["M_16pm"]["probes_test"][dd] == [(a, 100)] * 4 for dd, a in ((10, 10), (12, 9), (15, 15))),
          {dd: T["M_16pm"]["probes_test"][dd][0] for dd in (10, 12, 15)})

print("\nExpected, job A:")
for r, cap in (("M_inf", 0.17), ("M_16pm", 0.13)):
    e = mean(A[r]["train_E"]); e0 = mean(qp3[r]["train_E"])
    check(f"{r}: training clusters' energy >= 15 % below E3-QP3's (<= {cap})", e <= 0.85 * e0, f"{e:.3f} against {e0:.3f} ({100 * (e / e0 - 1):+.1f} %)")
    check(f"{r}: training forces (rigid, internal) at or below E3-QP3's",
          mean(A[r]["train_Frig"]) <= mean(qp3[r]["train_Frig"]) and mean(A[r]["train_Fint"]) <= mean(qp3[r]["train_Fint"]),
          f"{mean(A[r]['train_Frig']):.3f} / {mean(A[r]['train_Fint']):.3f} against {mean(qp3[r]['train_Frig']):.3f} / {mean(qp3[r]['train_Fint']):.3f}")
check("M_16C2pm: training clusters' energy <= 0.11", mean(A["M_16C2pm"]["train_E"]) <= 0.11, f"{mean(A['M_16C2pm']['train_E']):.4f}")
for r, cap in (("M_16pm", 0.24), ("M_16C2pm", 0.17)):
    check(f"{r}: hexamers <= {cap}", mean(A[r]["hexE"]) <= cap, f"{mean(A[r]['hexE']):.3f}")
check("M_inf: hexamers 0.35-0.50", 0.35 <= mean(A["M_inf"]["hexE"]) <= 0.50, f"{mean(A['M_inf']['hexE']):.3f}")
pinned = [r for r in rungs if r != "M_inf"]
check("test dimers: pinned <= 0.05", all(mean(A[r]["dimE"]) <= 0.05 for r in pinned), {r: round(mean(A[r]["dimE"]), 4) for r in pinned})
check("test dimers: M_inf <= 0.07", mean(A["M_inf"]["dimE"]) <= 0.07, round(mean(A["M_inf"]["dimE"]), 4))
check("raw dimers: pinned <= 0.085", all(mean(A[r]["2b"]) <= 0.085 for r in pinned), {r: round(mean(A[r]["2b"]), 4) for r in pinned})
check("raw dimers: pinned below M_inf on every seed (paired)", all(A[r]["2b"][i] < A["M_inf"]["2b"][i] for r in pinned for i in range(4)),
      {r: [round(x, 4) for x in A[r]["2b"]] for r in rungs})
check("raw dimers: M_inf <= 0.105", mean(A["M_inf"]["2b"]) <= 0.105, round(mean(A["M_inf"]["2b"]), 4))
check("compressed dimers (2.3-2.7 A) <= 0.12 for every rung", all(mean(A[r]["2b_2.3-2.7"]) <= 0.12 for r in rungs),
      {r: round(mean(A[r]["2b_2.3-2.7"]), 4) for r in rungs})
for r, cap in (("M_inf", 0.095), ("M_16pm", 0.13), ("M_16C2pm", 0.145)):
    check(f"{r}: raw 3-body <= {cap}", mean(A[r]["3b"]) <= cap, f"{mean(A[r]['3b']):.4f}")
print("\nThe read-out's near field (M_16C2pm_r1 against M_16C2pm), job A:")
k = sum(A["M_16C2pm_r1"]["3b_S0-2.7"][i] < A["M_16C2pm"]["3b_S0-2.7"][i] for i in range(4))
check("raw 3-body below O-O 2.7 A lower on >= 3 of 4 seeds", k >= 3, f"{k} of 4: {A['M_16C2pm_r1']['3b_S0-2.7']} against {A['M_16C2pm']['3b_S0-2.7']}")
check("hexamers' non-additive at or below", mean(A["M_16C2pm_r1"]["hexNA"]) <= mean(A["M_16C2pm"]["hexNA"]),
      f"{100 * mean(A['M_16C2pm_r1']['hexNA']) / HEX_NA:.1f} against {100 * mean(A['M_16C2pm']['hexNA']) / HEX_NA:.1f} %")
print("\nKept:")
check("raw 4-body M_16C2pm <= 55 %", 100 * mean(A["M_16C2pm"]["4b"]) / LAB["4b"] <= 55, f"{100 * mean(A['M_16C2pm']['4b']) / LAB['4b']:.0f} %")
check("hexamers' non-additive M_16C2pm <= 17 %", 100 * mean(A["M_16C2pm"]["hexNA"]) / HEX_NA <= 17, f"{100 * mean(A['M_16C2pm']['hexNA']) / HEX_NA:.1f} %")
for r in rungs:
    v = 100 * mean(A[r]["3b_L9-inf"]) / LAB9
    check(f"{r}: trimers beyond 9 A {'24-28' if 'C2' in r else '95-105'} %", (24 <= v <= 28) if "C2" in r else (95 <= v <= 105), f"{v:.1f} %")
if B:
    print("\nExpected, job B against job A (every rung):")
    for r in rungs:
        check(f"{r}: relaxed clusters' energy >= 10 % lower", mean(B[r]["train_E"]) <= 0.9 * mean(A[r]["train_E"]),
              f"{mean(B[r]['train_E']):.3f} against {mean(A[r]['train_E']):.3f}")
        check(f"{r}: contacts' energy higher, <= 0.15", mean(A[r]["contact_E"]) < mean(B[r]["contact_E"]) <= 0.15,
              f"{mean(B[r]['contact_E']):.3f} against {mean(A[r]['contact_E']):.3f}")
        check(f"{r}: compressed dimers within 0.02 of A's", mean(B[r]["2b_2.3-2.7"]) <= mean(A[r]["2b_2.3-2.7"]) + 0.02,
              f"{mean(B[r]['2b_2.3-2.7']):.3f} against {mean(A[r]['2b_2.3-2.7']):.3f}")
        check(f"{r}: hexamers lower", mean(B[r]["hexE"]) < mean(A[r]["hexE"]), f"{mean(B[r]['hexE']):.3f} against {mean(A[r]['hexE']):.3f}")
    k = sum(B["M_16C2pm_r1"]["3b_S0-2.7"][i] < B["M_16C2pm"]["3b_S0-2.7"][i] for i in range(4))
    check("job B: M_16C2pm_r1's raw 3-body below 2.7 A lower on >= 3 of 4 seeds", k >= 3, f"{k} of 4")
    check("job B: M_16C2pm_r1's hexamers' non-additive at or below M_16C2pm's", mean(B["M_16C2pm_r1"]["hexNA"]) <= mean(B["M_16C2pm"]["hexNA"]),
          f"{100 * mean(B['M_16C2pm_r1']['hexNA']) / HEX_NA:.1f} against {100 * mean(B['M_16C2pm']['hexNA']) / HEX_NA:.1f} %")
    check("job B, kept: raw 4-body M_16C2pm <= 55 %", 100 * mean(B["M_16C2pm"]["4b"]) / LAB["4b"] <= 55,
          f"{100 * mean(B['M_16C2pm']['4b']) / LAB['4b']:.0f} %")
    check("job B, kept: hexamers' non-additive M_16C2pm <= 17 %", 100 * mean(B["M_16C2pm"]["hexNA"]) / HEX_NA <= 17,
          f"{100 * mean(B['M_16C2pm']['hexNA']) / HEX_NA:.1f} %")
    for r in rungs:
        v = 100 * mean(B[r]["3b_L9-inf"]) / LAB9
        check(f"job B, {r}: trimers beyond 9 A {'24-28' if 'C2' in r else '95-105'} %", (24 <= v <= 28) if "C2" in r else (95 <= v <= 105), f"{v:.1f} %")
    print("\nthe balance (job B against job A, seed means): change in %")
    for r in rungs:
        print(f"  {r:11s} " + ", ".join(f"{k} {100 * (mean(B[r][k]) / mean(A[r][k]) - 1):+.0f}" for k in
                                       ("train_E", "train_Frig", "train_Fint", "contact_E", "2b", "2b_2.3-2.7", "3b", "3b_S0-2.7",
                                        "3b_S2.7-3", "hexE", "hexNA", "dimE", "4b")))
for label, T in (("job A", A),) + ((("job B", B),) if B else ()):           # not registered: read in the results note
    print(f"\n{label}: the raw 3-body over M_inf's: " + ", ".join(f"{r} {mean(T[r]['3b']) / mean(T['M_inf']['3b']):.2f}"
                                                                for r in rungs if r != "M_inf"))
    print(f"{label}: M_16C2pm_r1 - M_16C2pm per seed: " + "; ".join(
        f"{k} [" + ", ".join(f"{T['M_16C2pm_r1'][k][i] - T['M_16C2pm'][k][i]:+.3f}" for i in range(4)) + "]"
        for k in ("hexE", "hexNA", "hex_2b", "hex_rest", "3b", "3b_S0-2.7", "3b_S2.7-3", "4b")))
print(f"\n{sum(ok for _, ok, _ in V)} of {len(V)} readings as registered")
json.dump({"A": A, "B": B, "verdicts": V, "hex_na_rms": HEX_NA}, open(os.path.join(HERE, "e3qp4_summary.json"), "w"), indent=1,
          default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
