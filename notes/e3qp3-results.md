# E3-QP3 — the short range on q-AQUA-pol: results

*22 Oct 2026. The run of 22 Oct (s002, one RTX 3090, commit b2106f5 = v0.0.12 as applied; clusters part 10.0 h, the
diagnostic 26 min) read against `e3qp3-preregistration.md`. Seed means over the 4 seeds unless said, kcal/mol;
`scratch/e3qp3_summarise_log.py` on the job log. §3.3–3.4 are computed from the PES and the matched read-out alone, no
trained model (`scratch/e3qp3_response_contact.py`, `scratch/e3qp2_matched_response.py --edge=`). The local master of
this note is `notes/` in the repository.*

## 1. Verdict

- **The 2-body short range is fixed.** On the compressed raw dimers (O–O 2.3–2.7 Å) every rung reads 0.11–0.13 (E3-QP2
  read with the pins as trained: 0.41–0.95; registered ≤ 0.30), and every pinned rung beats M_∞ on the raw dimers on
  every seed: overall 0.090–0.098 against 0.113, from 2.7 Å 0.089–0.096 against 0.113. Coverage no longer matters —
  the raw dimers beyond the training pairs read as the covered ones.
- **The far field is as computed**: the probes at 10–15 Å and the C rungs' truncations on every seed, the _e1 rungs
  equal to their base rungs; the CCSD(T) trimers beyond 9 Å at 27 % for the C rungs and 100 % for the others.
- **E₀'s fit did not reach its registered level, nor what rests on it**: the training clusters' energy error is
  0.13–0.16 for the pinned rungs and 0.20 for M_∞ (registered ≤ 0.10 / ≤ 0.15); the hexamers 0.18–0.27 for the pinned
  rungs (≤ 0.15; M_16C2†ᵐ 0.189 against ≤ 0.12, E3-QP2 0.201); the test dimers 0.055–0.059 (≤ 0.05).
- **The 3-body at contact is now the short range's deficit.** The pinned rungs trail M_∞ on the raw 3-body (0.139–0.154
  against 0.100; compact trimers 0.160–0.194 against 0.120), most at the closest contacts (below O–O 2.7 Å: the
  daggers 0.25, the C rungs 0.32–0.33, M_∞ 0.16). M_16†ᵐ's 3-body improved on E3-QP2's (0.196 → 0.140); M_16C2†ᵐ's did
  not move (0.155 → 0.153).
- **The pinned bands' edge 1.0 Å is no lever**: on the raw dimers below O–O 2.7 Å M_16†ᵐ_e1 a little better than
  M_16†ᵐ (0.109 against 0.118, 4 of 4 seeds), M_16C2†ᵐ_e1 a little worse than M_16C2†ᵐ (0.130 against 0.119, 0 of 4);
  every other reading within the seeds' spread.
- **M_16C2†ᵐ stays the best rung overall**: hexamers 0.189 (M_∞ 0.443), their non-additive energy 16.0 % (24.1 %),
  the raw 4-body 49 % (59 %), the raw 2-body 0.093 (0.113), every reading beyond 3 Å. M_∞ wins only the raw 3-body
  below 3 Å, and ties the compressed dimers (0.114 against 0.119).

## 2. The predictions

| registered (§4) | E3-QP3 (M_∞ / M_16†ᵐ / M_16†ᵐ_e1 / M_16C2†ᵐ / M_16C2†ᵐ_e1) | |
|---|---|---|
| far field at 10–15 Å on every seed, _e1 = base | C rungs 4 / 5 / 3 %, truncations 15 / 15 / 14 and 5 / 5 / 4 %; daggers 100 %, interaction 10 / 9 / 15 % | ✓ exact |
| training clusters ≤ 0.10 pinned (every seed), ≤ 0.15 M_∞ | 0.198 / 0.155 / 0.153 / 0.129 / 0.131 | ✗ |
| contact configurations ≤ 0.15 | 0.082 / 0.079 / 0.075 / 0.080 / 0.084 | ✓ |
| training forces' rigid-body part ≤ 1.0 kcal/mol/Å | 0.59 / 0.50 / 0.48 / 0.51 / 0.53 | ✓ |
| compressed dimers (O–O 2.3–2.7 Å) ≤ 0.30; pinned within 0.05 of M_∞ | 0.114 / 0.117 / 0.108 / 0.119 / 0.128 | ✓ |
| dimers from 2.7 Å: pinned ≤ 0.10, M_∞ ≤ 0.13, pinned below M_∞ on every seed | 0.113 / 0.091 / 0.089 / 0.092 / 0.095; 16 of 16 | ✓ |
| raw 2-body ≤ 0.15; pinned below M_∞ on ≥ 3 of 4 seeds | 0.113 / 0.093 / 0.090 / 0.093 / 0.098; 4 of 4 each | ✓ |
| raw 3-body ≤ 0.12; compact trimers ≤ 0.14 | 0.100 / 0.140 / 0.139 / 0.153 / 0.154; 0.120 / 0.162 / 0.160 / 0.192 / 0.194 | ✗ (M_∞ ✓) |
| (open) compact trimers: pinned at or below M_∞ | as above | ✗ |
| hexamers: pinned ≤ 0.15, M_16C2†ᵐ ≤ 0.12, M_∞ 0.35–0.55 | 0.443 / 0.267 / 0.269 / 0.189 / 0.182 | ✗ (M_∞ ✓) |
| hexamers' non-additive: M_16C2†ᵐ ≤ 16 %, M_16†ᵐ ≤ 40 %, M_∞ ≤ 25 % | 24.1 / 31.2 / 32.3 / 16.0 / 16.6 % | ✓ (M_16C2†ᵐ at the bound) |
| test dimers: pinned ≤ 0.05, M_∞ ≤ 0.08 | 0.074 / 0.058 / 0.055 / 0.057 / 0.057 | ✗ (M_∞ ✓) |
| raw 4-body: M_16C2†ᵐ ≤ 63 % | 59 / 104 / 109 / 49 / 47 % | ✓ |
| test probes at 8 Å within ± 6 points of E3-QP2's | M_∞ 96 (92), M_16†ᵐ 100 (100), M_16C2†ᵐ 6 (5) % | ✓ |
| trimers beyond 9 Å: C rungs 25 ± 3 %, the others 95–105 % | 100 / 100 / 100 / 26.9 / 26.7 % | ✓ |
| (open) _e1 at or below its base on the compressed dimers on ≥ 3 of 4 seeds; within 0.01 from 3.0 Å | M_16†ᵐ_e1 4 of 4, 0.003; M_16C2†ᵐ_e1 0 of 4, 0.000 | ✓ / ✗ |

No registered surprise occurred: the far field exact, the compressed dimers far below 0.4, no pinned rung behind M_∞
from 2.7 Å, the C rungs' training clusters below 0.15.

## 3. What limits the short range now

**3.1 E₀ trades its energies against its forces.** The energy weight 100 and twice the training lowered the training
clusters' energy error by about a quarter (E3-QP2 seed 0 → E3-QP3: M_∞ 0.258 → 0.198, M_16†ᵐ 0.204 → 0.155, M_16C2†ᵐ
0.175 → 0.129) and raised their force errors (rigid-body 0.46–0.55 → 0.50–0.59, internal 0.52–0.58 → 0.66–0.72
kcal/mol/Å): on its own training data E₀ moved along a front instead of improving both — a fit limited by E₀'s width
(16 features, 2 layers) more than by its training. The test sets bought little of it: the hexamers 0.451 / 0.305 /
0.201 → 0.443 / 0.267 / 0.189 (M_∞ / M_16†ᵐ / M_16C2†ᵐ), the test dimers 0.096 / 0.076 / 0.056 → 0.074 / 0.058 /
0.057. Across rungs the same front: M_16†ᵐ fits its 2-body better than M_∞ and its 3-body worse, on the same 3-body
target (the pinned bands are pairwise).

**3.2 The contacts dominate the energy term.** The loss averages (ΔE / n_atoms)² over structures, so a structure
weighs 1 / n_atoms²: a 6-atom contact dimer six times a 15-atom pentamer. At E3-QP3's errors the 800 contact
configurations carry 58 % (M_∞) to 76–77 % (the C rungs) of the energy term shared with the 300 relaxed clusters — the
clusters the test hexamers resemble.

**3.3 The read-out at contact** (computed; no trained model). The far-field-matched read-out of M_16C2†ᵐ (frozen in
training, the same on every seed) read alone on the raw trimers, against TTM3-F's 3-body induction — the physics it
stands for (TTM3-F's other terms are pairwise):

| closest O–O | trimers | labels | TTM3-F's 3-body induction | read-out | read-out less induction (slope) | first order only: slope |
|---|---|---|---|---|---|---|
| below 2.7 Å | 371 | 1.268 | 0.659 | 0.546 | 0.375 (0.68) | 0.17 |
| 2.7–3.0 Å | 3361 | 0.596 | 0.352 | 0.318 | 0.141 (0.83) | 0.56 |
| from 3.0 Å | 941 | 0.145 | 0.111 | 0.114 | 0.022 (1.01) | 0.94 |

Beyond 3 Å the read-out is TTM3-F's induction (slope 1.01); at the hydrogen-bond contact it is 0.83 of it and at
compressed contacts 0.68, its first-order part far too weak there (0.17) and made up in part by its mutual part:
far-field matching does not fix the read-out's near field. It still leaves E₀ less 3-body to learn — the labels less
the read-out, 0.41 against the labels' 0.62 overall, 0.93 against 1.27 below 2.7 Å — yet the trained C rungs' 3-body
is worse than M_16†ᵐ's, which learns the labels whole, in both bins below 3 Å (0.316 against 0.246 below 2.7 Å,
0.146 against 0.138 at 2.7–3.0 Å) and better beyond (0.036 against 0.069). E₀ does not absorb the read-out's contact
error. Below 3 Å lies 99 % of M_16C2†ᵐ's squared 3-body error (34 % below 2.7 Å, in 8 % of the trimers).

**3.4 The read-out's own band edge** (computed): M_16C2†ᵐ's read-out matched again at other edges, the pinned bands at
0.75 Å; the far field stays within a point at 10–15 Å.

| response edge | 0.5 Å | 0.75 Å (E3-QP3) | 1.0 Å | 1.5 Å |
|---|---|---|---|---|
| test probes' non-additive energy at 10 / 12 / 15 Å | 5 / 5 / 3 % | 4 / 5 / 3 % | 3 / 4 / 3 % | 4 / 5 / 4 % |
| hexamers: the response's many-body energy against TTM3-F's | 15 % | 21 % | 12 % | 64 % |
| hexamers: its whole energy against TTM3-F's induction, rmse | 0.63 | 0.55 | 0.32 | 0.47 |
| test dimers: the same, rmse | 0.143 | 0.146 | 0.093 | 0.031 |
| raw trimers: read-out less induction, all / below 2.7 / 2.7–3.0 Å | 0.171 / 0.408 / 0.148 | 0.160 / 0.375 / 0.141 | 0.140 / 0.295 / 0.133 | 0.182 / 0.305 / 0.186 |
| E₀'s 3-body target (labels less read-out), all / below 2.7 Å | 0.425 / 0.963 | 0.409 / 0.925 | 0.394 / 0.873 | 0.411 / 0.893 |

At 1.0 Å the response is closest to TTM3-F's induction wherever the far field does not fix it — the hexamers'
many-body 12 % against 21 %, its whole energy on the hexamers and on the dimers about 40 % closer, the trimers' 3-body
at contact 0.30 against 0.38 — with the far field unchanged; at the contact it is still only 0.69–0.78 of TTM3-F's.
E3-QP3's _e1 rungs moved the pinned bands' edge, not this one.

## 4. Next

- **E3-QP4, set up 22 Oct** after Vadic's "yes" (`e3qp4-preregistration.md`, `configs/water_qaqua_pol4a.yaml` and
  `pol4b.yaml`, harness v0.0.13): E3-QP3's data, matching and readings, rungs M_∞, M_16†ᵐ, M_16C2†ᵐ
  and M_16C2†ᵐ with the read-out's band edge at 1.0 Å (§3.4; the _e1 rungs dropped), 4 seeds, as two jobs that run in
  parallel. Job A: E₀'s width 16 → 32 (§3.1; the read-out's species features share the width, so its far field is
  recomputed before training). Job B: the same with the contact configurations weighted down so that the relaxed
  clusters get back their share of the energy term (§3.2). A against E3-QP3 reads the width, B against A the weighting,
  and the response edge is a factor within both (§3.3–3.4). About 14 h per job on a 3090 (estimate).
- The ion part on MB-pol/MB-nrg (MBX), the K = 16 batch of E1/E2, decision 4 and the periodic path, the MACE baseline:
  as before.
