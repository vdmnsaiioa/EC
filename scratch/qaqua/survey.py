"""survey of the q-AQUA n-body data sets: counts, energy ranges, O-O distance coverage, monomer flexibility."""
import sys, numpy as np
path, nat = sys.argv[1], int(sys.argv[2])
E, X = [], []
with open(path) as fh:
    lines = fh.read().split("\n")
i = 0; n = len(lines)
while i + nat + 1 < n:
    if lines[i].strip() == "": i += 1; continue
    assert int(lines[i].split()[0]) == nat, (i, lines[i])
    E.append(float(lines[i + 1].split()[0]))
    X.append([[float(v) for v in lines[i + 2 + k].split()[1:4]] for k in range(nat)])
    i += nat + 2
E = np.array(E); X = np.array(X)
nmol = nat // 3; nH = 2 * nmol
O = X[:, nH:nH + nmol]; H = X[:, :nH]
print(f"{path}: {len(E)} configurations, {nat} atoms; energy range {E.min()*627.509:.3f} .. {E.max()*627.509:.3f} kcal/mol, "
      f"rms {np.sqrt(np.mean(E**2))*627.509:.4f}, median |E| {np.median(np.abs(E))*627.509:.4f} kcal/mol")
# O-O distances
dOO = []
for a in range(nmol):
    for b in range(a + 1, nmol):
        dOO.append(np.linalg.norm(O[:, a] - O[:, b], axis=1))
dOO = np.array(dOO)             # (pairs, N)
dmin = dOO.min(axis=0); dmax = dOO.max(axis=0)
qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
print("  O-O (all pairs) percentiles [A]:", np.round(np.percentile(dOO.ravel(), qs), 2).tolist(), "for", qs)
print("  largest O-O per configuration percentiles [A]:", np.round(np.percentile(dmax, qs), 2).tolist())
# monomer geometry: O-H bond lengths and HOH angle; assume H pairs (2m, 2m+1) belong to O m
rOH = []; ang = []
for m in range(nmol):
    h1 = H[:, 2 * m] - O[:, m]; h2 = H[:, 2 * m + 1] - O[:, m]
    r1 = np.linalg.norm(h1, axis=1); r2 = np.linalg.norm(h2, axis=1)
    rOH += [r1, r2]
    ang.append(np.degrees(np.arccos(np.sum(h1 * h2, axis=1) / (r1 * r2))))
rOH = np.concatenate(rOH); ang = np.concatenate(ang)
print(f"  r_OH percentiles [A]: {np.round(np.percentile(rOH, qs), 3).tolist()}; HOH angle percentiles [deg]: {np.round(np.percentile(ang, qs), 1).tolist()}")
# energies vs the largest O-O distance: tail statistics for the 2-b set
if nmol == 2:
    for lo, hi in [(2.5, 3.5), (3.5, 4.5), (4.5, 6), (6, 7), (7, 8), (8, 10), (10, 100)]:
        sel = (dmin >= lo) & (dmin < hi)
        if sel.sum():
            print(f"  O-O in [{lo}, {hi}) A: {sel.sum():6d} configs, |E| median {np.median(np.abs(E[sel]))*627.509:.4f}, max {np.abs(E[sel]).max()*627.509:.3f} kcal/mol")
    # the smallest |E| and the finest energy resolution (number of digits)
    txt = open(path).read().split("\n")[1]; print("  energy string example:", repr(txt))
else:
    for lo, hi in [(0, 3.5), (3.5, 4.5), (4.5, 6), (6, 8), (8, 100)]:
        sel = (dmax >= lo) & (dmax < hi)
        if sel.sum():
            print(f"  largest O-O in [{lo}, {hi}) A: {sel.sum():6d} configs, |E_nb| median {np.median(np.abs(E[sel]))*627.509:.4f}, max {np.abs(E[sel]).max()*627.509:.3f} kcal/mol")
np.save(path.replace(".xyz", "_E.npy"), E)
