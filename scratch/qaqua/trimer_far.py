"""the q-AQUA 3-body set as a CCSD(T)-F12 measurement of the far-field three-body energy: for trimers with one
molecule far from a bound pair, |E_3b| against the distance d of the far O from the pair's O-O midpoint."""
import sys, numpy as np
path = sys.argv[1]
lines = open(path).read().split("\n"); E, X = [], []; i = 0; nat = 9
while i + nat + 1 < len(lines):
    if lines[i].strip() == "": i += 1; continue
    E.append(float(lines[i + 1].split()[0])); X.append([[float(v) for v in lines[i + 2 + k].split()[1:4]] for k in range(nat)]); i += nat + 2
E = np.array(E) * 627.509; X = np.array(X); O = X[:, 6:9]
d01 = np.linalg.norm(O[:, 0] - O[:, 1], axis=1); d02 = np.linalg.norm(O[:, 0] - O[:, 2], axis=1); d12 = np.linalg.norm(O[:, 1] - O[:, 2], axis=1)
D = np.stack([d01, d02, d12], 1); order = np.argsort(D, axis=1)
dmin = D[np.arange(len(D)), order[:, 0]]; dmid = D[np.arange(len(D)), order[:, 1]]; dmax = D[np.arange(len(D)), order[:, 2]]
# the far molecule: the one not in the closest pair; its distance to the pair midpoint
pairs = [(0, 1), (0, 2), (1, 2)]; far_idx = np.array([3 - sum(pairs[k]) for k in order[:, 0]])
near = np.array([pairs[k] for k in order[:, 0]])
mid = (O[np.arange(len(O)), near[:, 0]] + O[np.arange(len(O)), near[:, 1]]) / 2
dfar = np.linalg.norm(O[np.arange(len(O)), far_idx] - mid, axis=1)
bound = dmin < 3.2                                     # a hydrogen-bonded pair
print("trimers with a bound pair (closest O-O < 3.2 A) and the third molecule at distance d from the pair's midpoint:")
print("   d (A)      n    rms |E3b|   median |E3b|   frac |E3b| > 0.01   mean E3b   [kcal/mol]")
for lo, hi in [(3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 12)]:
    s = bound & (dfar >= lo) & (dfar < hi)
    if s.sum() > 5:
        e = E[s]; print(f"  [{lo:2d},{hi:2d})  {s.sum():6d}   {np.sqrt(np.mean(e**2)):9.4f}   {np.median(np.abs(e)):9.4f}      {np.mean(np.abs(e) > 0.01):5.2f}        {e.mean():+8.4f}")
# power law of the rms beyond 6 A
sel = bound & (dfar >= 6) & (dfar < 12)
lo_bins = np.array([6, 7, 8, 9, 10]); rms = []
for a, b in zip(lo_bins, [7, 8, 9, 10, 12]):
    s = bound & (dfar >= a) & (dfar < b); rms.append(np.sqrt(np.mean(E[s] ** 2)))
rms = np.array(rms); mids = np.array([6.5, 7.5, 8.5, 9.5, 11])
p = np.polyfit(np.log(mids), np.log(rms), 1)[0]
print(f"  rms |E3b| vs d on [6, 12) A: local power law d^({p:.2f})  (pre-existing induced dipole x far field: -3; mutual response: -6)")
# the energy resolution: smallest nonzero |E| and the digit structure
print("  smallest nonzero |E3b| in the set:", np.min(np.abs(E[E != 0])), "kcal/mol; number of exact zeros:", int(np.sum(E == 0)))
