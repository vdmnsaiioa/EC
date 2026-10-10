# E3-R — the band-field response, tested explicitly: results

*10 Oct 2026. Pre-registration: `e3r-bandfield-response-preregistration.md` (§3 has the ceilings computed
from the truth, §4 the predictions). Harness v0.0.6 with the own-fragment exclusion and the ion probe;
every number is in `results/e3r_*` with the command that produced it. K = 2 throughout (the convergence
rule did not exclude any seed).*

## 1. R1 — clusters at the band edge ℓ₁ = 0.75 Å (`results/e3r_clusters_l1_0.75_K2.log`)

    python scripts/e3_water.py --part clusters --rungs M_16p,M_16S0p,M_16S2p --seeds 2 --steps 1000 --lbfgs 150 \
        --batch 12 --lbfgs_chunk 6 --l1 0.75 --rs_over_l1 0 --cache results/cache_clusters.pkl

Permanent sources (the isolated monomer's charges) pinned, the dispersion band on the pinned α_O, the band
field at an atom being the field of the *other* molecules' charges; 36 training clusters (n = 3–5), 8 hexamers
and 8 dimers held out; the hexamers' non-additive energy has rms 1.287 kcal/mol, of which the first-order
induction E_ind1 is 71 % (the order-0 read-out's computed ceiling) and the self-consistent correction 30 %.

| rung | hexamer rmse (kcal/mol), seeds | dimer rmse | non-additive rmse, % of 1.287 | predicted → verdict |
|---|---|---|---|---|
| M_16† (no read-out) | 0.659, 0.588 | 0.178, 0.176 | **106, 90 %** | 80–95 %, hexamer 0.5–0.7 → ✓ (one seed at 106) |
| **M_16S0† (order 0)** | **0.206, 0.159** | 0.064, 0.015 | **36, 15 %** | ≤ 45 %, hexamer ≤ 0.35 → **✓** |
| M_16S2† (order 2) | 0.148, 0.144 | 0.020, 0.026 | 22, 19 % | within 1.3× of order 0 → ✓ (means 0.265 against 0.327 kcal/mol, ratio 0.81; no gain above 1.5×) |

The read-out carries the many-body induction of the hexamers: the non-additive error falls from 90–106 %
of the non-additive energy without it to 15–36 % with it, the hexamer error from 0.59–0.66 to 0.16–0.21
kcal/mol, with the same 36 training clusters that left every rung of the rehearsal at 73–120 %. One seed
(15 %) sits *below* the 30 % "floor" of §3: the floor assumed the read-out carries E_ind1 with a fixed
coefficient α, but its coefficients are environment-dependent (a(s_i), and the pair terms w(r_ij; s_i, s_j)
(n̂·E)²), so the local self-consistent enhancement of the response — the effective polarisability of an
oxygen in its hydrogen-bond environment — is partly representable too. The §3 ceiling was a ceiling for a
fixed coefficient, not for the class; recorded as a prediction wrong in the conservative direction.
Order 2 is not worse than order 0 (the rehearsal's loss of a factor 1.5 came from unidentified
coefficients, now identified) and slightly better and more even across seeds (0.265 against 0.327).

R2_PLACEHOLDER

R3_PLACEHOLDER

## 4. Wrong predictions

1. M_16†'s non-additive error 80–95 % → 90 and 106 % (one seed above): immaterial; the baseline.
2. The §3 "floor" of 30 % for order 0 on the hexamers → 15 % for one seed: the floor was for a fixed
   coefficient, the class has environment-dependent ones.
WRONG_PLACEHOLDER

## 5. What this settles

SETTLES_PLACEHOLDER
