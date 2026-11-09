# E4, job 3: the results

*8 Nov 2026. Job 3 ran as pre-registered (`e4-job3-preregistration.md`) on Niflheim at a7cf750, the v0.0.27 patch's
tree. First MB-pol's finite-k response in two CPU tasks (`run_e4_response_k.sbatch`), then `kread` on one RTX 3090
(`run_e4_job3.sbatch`): 17:08–17:24 CET, 16 minutes against about an hour estimated.*

*Read from its log and `results/e4_job3/kread.json`, the bands by `scripts/e4_report.py --job3`.*

- *§§1–5 are the registered readings and the readings registered beside them.*
- *§6 was not registered. It was written after these numbers: the response's k-dependence order by order, from
  `kread.json`, and a local diagnostic over the mutual edge.*

## 1. The bands

| | reading (held-out cells unless said) | registered | Niflheim | |
|---|---|---|---|---|
| K0 | MB-pol's ε_L(k₁), ε_L(k₂) on the 40 cells; χ(k)/χ(0⁺) − 1 | 1.772 ± 0.010, 2.088 ± 0.020; 5–9 %, 24–32 % | 1.7734, 2.0916; 7.0 %, 28.1 % | holds |
| K1 | arm C's χ(k₁); χ(k₂) against MB-pol's | −3.5 to −0.5 %; −15 to −8 % | −1.84 %; **−10.96 %** | holds |
| K2 | arm C's molecules at k₁ / k₂, below C0's | 2–6 % / 3–7 % | 3.8 / 5.5 % (C0 5.8 / 11.3 %) | holds |
| K3 | C0's χ(k₂), closer to MB-pol's than arm C's | +1.5 to +5.5 % | **+3.34 %** | holds |
| K4 | arm A: χ(k₁); χ(k₂) | +1.5 to +4.5 %; −8.5 to −2 % | +3.23 %; −4.69 % | holds |
| | arm B: χ(k₁); χ(k₂) | +1.5 to +4.5 %; +1.5 to +7.5 % | **+5.16 %; +12.55 %** | **fails** |
| K5 | L in state: the molecules' dipoles; response; iso r | ≤ 3 %; ≤ 2 %; ≥ 0.9, each within 1 point of the calibration cells | 2.02 %; 1.07 %; 0.968 (calibration 1.96 %, 1.09 %, 0.967) | holds |
| K6 | L out of state: far probes at 10 / 12 / 15 Å; T5 | ≤ 10.5 / 6.0 / 6.8 %; 0.95–1.10 | 6.7 / 5.4 / 5.2 %; 1.0247 | holds |

Six of the seven hold, and K4 fails on arm B alone.

- **The main prediction holds (K1, K3).** Arm C's orders ≥ 1 mispredict MB-pol's finite-k response at k₂; its order-0
  sector does not.
- **The locality test holds,** in state (K5) and out of state (K6).
- **The calibration cells** read within 0.4 points of the held-out ones on every χ.

## 2. Arm C's prediction (K1–K3)

Held-out cells. The local column is the pre-registration's: 13 cells and the local read-outs.

| read-out | χ(k₁) / χ(k₂), local | χ(k₁) / χ(k₂), Niflheim | molecules k₁ / k₂ | their size k₁ / k₂ |
|---|---|---|---|---|
| A (far probes) | +3.1 / −5.3 % | +3.2 / −4.7 % | 7.8 / 8.4 % | 1.063 / 1.062 |
| B (liquid, every order) | +2.8 / +4.6 % | **+5.2 / +12.6 %** | 4.3 / 8.2 % | 1.029 / 1.053 |
| **C (E4's model)** | −1.9 / −11.5 % | **−1.8 / −11.0 %** | 3.8 / 5.5 % | 1.011 / 0.993 |
| C0 (C's order-0 sector) | +2.6 / +3.5 % | +2.4 / +3.3 % | 5.8 / 11.3 % | 1.040 / 1.094 |
| L (the locality test) | +2.4 / +3.2 % | +2.4 / +3.2 % | 4.8 / 11.1 % | 1.042 / 1.104 |

- **What reproduces.** Every read-out whose orders ≥ 1 come from the far probes (A, C), or are off (C0, L), reproduces
  the local reading to 0.7 points, from 13 local cells to 40 Niflheim cells. C0 and L reproduce it to 0.15 points.
- **Arm C.**
  - Its χ is 11.0 % low at k₂, where its order-0 sector is 3.3 % high.
  - Its orders ≥ 1 take 14.3 points at k₂ and 4.3 at k₁. In χ itself their shift at k₂ is exactly four times that at
    k₁: a k² term (§6b).
  - The frame-to-frame sd of these differences is 0.15 points at k₁ and 0.5 at k₂.
- **Per molecule the same orders ≥ 1 help:** 5.5 % against the order-0 sector's 11.3 % at k₂. The dipoles' size is 0.993
  against C0's 1.094.

## 3. Arm B (K4): its orders ≥ 1 do not reproduce

**At k = 0 arm B reads as it did locally:**

- ε₀ 1.6887 against MB-pol's 1.6885;
- the molecules' dipoles 4.4 % and response 2.0 %, against 4.6 % and 2.2 % locally.

**At finite k it does not:** +5.2 / +12.6 % against +2.8 / +4.6 % locally, 2.4 and 8 points apart.

- Arm B is the one read-out whose orders ≥ 1 the liquid's k = 0 data moved.
- Its calibration here had 30 cells (job 2's), against 10 locally, at the same total weight.
- Its k² coefficient (§6) is 0.51 Å² here and 0.41 Å² locally, while arm A's and C's reproduce.

**Reading.**

- The same k = 0 fit leaves different finite-k responses: the cells' sums at k = 0 leave the orders ≥ 1's finite-k
  response free. This is the derivative counting's statement, seen directly.
- The local agreement that the pre-registration noted ("arm B's orders ≥ 1, moved by k = 0 data, happen to keep both")
  was an accident of its 10 cells.
- The failure is arm B's, not MB-pol's or the data's: K0 holds, and arms A and C reproduce.
- The band, written from one calibration set, treated an unidentified quantity as a prediction.
- Arm B is not a candidate for the model.

## 4. The locality test (K5, K6)

L is the order-0 coefficient solved from zero on every molecule of the 30 calibration cells, with no probe rows.

- **Its fit.** It converged in three Gauss–Newton steps (the liquid rows 0.924 → 0.0168 → 2.38×10⁻⁴ → 2.350×10⁻⁴). It
  took 422 s on the RTX 3090, against 2486 s for 10 cells on two CPU cores locally.
- **In state:**

| | L, held out | L, calibration | arm C, held out |
|---|---|---|---|
| ε₀ (MB-pol's 1.6885 held out, 1.6876 calibration) | 1.6884 | 1.6877 | 1.6893 |
| the cells' induced dipole | 0.75 % | 0.73 % | 1.5 % |
| the molecules' induced dipole | 2.02 % | 1.96 % | 5.1 % |
| their field response (iso) | 1.07 (0.39) % | 1.09 (0.38) % | 2.9 (1.5) % |
| the iso response's spread ÷ MB-pol's (r) | 0.95 (0.968) | 0.95 (0.967) | 0.84 (0.42) |
| the dipoles' size | 1.000 | 1.000 | 0.991 |

- **Out of state:**
  - The far probes' non-additive energy is 5.8 / 6.7 / 5.4 / 5.2 % at 8 / 10 / 12 / 15 Å, and the interaction 2.15 %.
    T5's slope is 1.0247 (3.4 %).
  - Against the order-0 read-out matched on the probes alone (4.9 / 7.0 / 4.0 / 4.5 %), that is 1.2×, 1.0×, 1.35× and
    1.2×.
  - C0, arm C's order-0 sector matched jointly on the probes and the cells' sums, reads 7.4 / 7.4 / 5.7 / 6.5 %. The
    liquid alone predicts the vacuum cores' far field at least as well.
- **At finite k, L reads as C0.** Its χ is +2.35 / +3.22 % against C0's +2.43 / +3.34 %, and its molecules 4.8 / 11.1 %
  against 5.8 / 11.3 %. The order-0 coefficient's two calibrations give the same k-dependence.

**Reading.**

- The order-0 coefficient is local at the 6 Å cutoff. The far probes and the liquid ask the same of it, so arm C's joint
  matching is consistent.
- L's per-molecule calibration fits the liquid's molecules two to four times better than arm C's cell sums (dipoles
  2.0 % against 5.1 %, response 1.1 % against 2.9 %), with the pattern (spread 0.95, r 0.97) that arm C misses. It is a
  candidate for the model's order-0 sector.

## 5. The readings registered beside them

- **The calibration cells against the held-out ones.**
  - MB-pol's ε_L is 1.7730 / 2.0908 against 1.7745 / 2.0938.
  - Arm C reads −1.92 / −11.33 % against −1.84 / −10.96 %.
  - Every read-out's χ differences agree within 0.4 points.
- **The χ differences' frame-to-frame sd:** 0.05–0.3 points at k₁ and 0.15–0.7 at k₂. Arm B's are the largest at k₂.
- **ε_L against each read-out's own ε₀:** §6.
- **L's finite-k response:** §4.
- **The cells' induced dipole at k = 0:** A 7.0 %, B 0.9 %, C 1.5 %, C0 2.4 %, L 0.75 %.
- **The cost:**
  - L, 422 s;
  - the whole reading, 16 minutes;
  - each read-out's readings, 81–156 s; arms A, B and C came from job 2's caches.

## 6. Not registered: the k-dependence, order by order

This was written after the Niflheim numbers. Items (a)–(d) are from `kread.json`. Item (e) is a local diagnostic
(`scratch/e4_job3_kdep.py`) on one local held-out frame, with the matched read-outs of the mutual-edge sweep
(`e4-response-regulator.md` §2).

Write q(k) = χ(k)/χ(0⁺) − 1, each against its own screened limit χ(0⁺) = (1 − 1/ε₀)/4π, and fit q = c₂k² + c₄k⁴
through k₁ and k₂ (k₂ = 2k₁).

**(a) MB-pol's χ(k) is k² up to k₂.**

- Over the 40 cells q is 7.01 % at k₁ and 28.06 % at k₂: a ratio of 4.001, against (k₂/k₁)² = 4.
- So c₂ is 0.274 Å², and the k⁴ term at k₂ is within ±0.1 % of χ(0⁺) on either split.
- On one local frame, at k₃ = 1.52 and k₄ = 2.02 Å⁻¹, q is +61.0 % and +94.1 %, against the k² term's +62.6 % and
  +111 %.
- MB-pol's higher orders set in between 1.5 and 2 Å⁻¹: wavelengths of 3–4 Å, the first shell's scale.

**(b) The read-out's orders ≥ 1 add exactly a k² term.**

- χ_C(k) − χ_C0(k) is −0.001486 at k₁ and −0.005943 at k₂: a ratio of 4.0000, on both splits.
- The orders ≥ 1 read the mode's field with its first and second derivatives, outside the mutual solve. In the sum of
  the two phases only their k² terms survive.
- So χ(k) = χ_order-0(k) + γk². Arm C's γ is −0.179 Å² in units of its χ(0⁺), as c₂ is.

**(c) Every read-out's k⁴ term is its order-0 sector's, and the same.** Relative to each read-out's own χ(0⁺), held-out
cells:

| | c₂ | the k⁴ term at k₂ |
|---|---|---|
| MB-pol | 0.275 Å² | −0.1 % |
| A | 0.241 Å² | −7.9 % |
| B | 0.510 Å² | −8.0 % |
| C | 0.215 Å² | −8.0 % |
| C0 | 0.394 Å² | −8.0 % |
| L | 0.394 Å² | −8.0 % |

- c₄ is −0.075 to −0.077 Å⁴ for all five read-outs, against MB-pol's ≈ 0. The read-outs differ in c₂ alone.
- Arm C's far-probe orders ≥ 1 take its order-0 sector's c₂ of 0.394 Å² down to 0.215; MB-pol's is 0.275.

**(d) The ceiling with γ alone.** Arm C's order-0 sector, with any γ:

- the best joint reading is ±1.1 %: χ(k₁) +1.1 % and χ(k₂) −1.1 %, at γ = −0.056 Å². Arm C's far-probe γ is three
  times that, −0.179;
- exact at k₁, χ(k₂) is −4.8 % (γ = −0.102);
- exact at k₂, χ(k₁) is +1.4 % (γ = −0.042).

**(e) The k⁴ term is the mutual edge's.** One local held-out frame. Each edge's sweep read-out (the frozen form matched
on the probes at that edge) is read with its orders ≥ 1 off, for its order-0 sector, and on:

| mutual edge | order-0 sector: c₂; c₄ | the probe-matched orders ≥ 1: Δc₂ (Δc₄) | γ alone: best joint χ (γ needed) | exact at k₁: χ(k₂) |
|---|---|---|---|---|
| 0.75 Å (job 1's, the response edge) | 0.135 Å²; −0.005 Å⁴ | −0.379 (+0.0003) | ±0.14 % (+0.146) | −0.6 % |
| 1.5 Å | 0.222; −0.017 | −0.167 (−0.0003) | ±0.30 % (+0.068) | −1.3 % |
| 2.0 Å (E4's) | 0.407; −0.072 | −0.178 (−0.0001) | ±1.07 % (−0.071) | −4.7 % |
| 2.5 Å | 0.657; −0.187 | −0.212 (0.0000) | ±2.70 % (−0.223) | −11.7 % |
| MB-pol, the same frame | 0.272; +0.005 | | | |

- **The order-0 class matched on its own** at each edge gives the same c₂ and c₄, to 0.02 Å² and 0.003 Å⁴.
- **c₄ grows about as the edge's fifth power** from 1.5 to 2.0 Å, and its fourth from 2.0 to 2.5 Å. c₂ grows with the
  edge too, and MB-pol's lies between the 1.5 and 2.0 Å sectors'.
- **The orders ≥ 1 change c₄ by at most 0.0003 Å⁴ at every edge**, as (b) says.
- **The far probes' orders ≥ 1 shift c₂ by −0.17 to −0.38 Å² at every edge**, whatever the order-0 sector needs (+0.15
  to −0.22). The far probes do not identify them at any edge.
- *Added 9 Nov (`e4-job4-preregistration.md` §3): these read-outs matched every order on the probes at once, beside an
  order-0 sector that the liquid had not calibrated.*
  - *Job 4's control Dx re-solves the orders ≥ 1 on the probes alone, after arm C's calibration of order 0. On the
    local cells χ(k₂) is then +0.7 % (held out) and +2.2 % (calibration), with a frame-to-frame sd of 0.8–1.4 points.*
  - *So most of arm C's −11 % came from orders ≥ 1 matched beside the uncalibrated order 0. Given the calibrated one,
    the probes put the mean near MB-pol's but do not pin it from frame to frame.*

**Reading.**

- The order-0 sector's k-dependence beyond k² is a regulator artefact. It grows steeply with the mutual edge and is −8 %
  of χ(0⁺) at k₂ at E4's 2.0 Å.
- The read-out's order-2 counterterm, the single number γ in the cells' χ, can absorb the edge's k² dependence but not
  its k⁴.
- At 2.0 Å the model's finite-k response can therefore be matched to ±1.1 % at k₁ and k₂ together, no better: the
  frozen form's truncation error at the cutoff's scale.
- The k⁴ term is 0.5 % at k₁, where the EFT's expansion holds. At k₂ ≈ 2π/r_c the expansion is at its limit.
- These numbers are what H3 reads on the finite-k response (`e4-x3-preregistration.md` §6). The edge dependence of
  χ(k₂) before γ is calibrated is about 27 points of χ(0⁺) from 1.5 to 2.5 Å; after it, the residuals left are
  0.3–2.7 %. That fall is partly by construction, since γ would be fitted on the same data. H3's reading is a
  prediction at another k, or on another observable.

## 7. Reading

1. **Arm C's orders ≥ 1, from the far probes, are wrong at the cutoff's scale (K1–K3).** They put the k² coefficient
   0.18 Å² below the order-0 sector's, three times the shift MB-pol asks. The order-0 sector is right to 3 %. The
   derivative counting's second step is needed: the orders ≥ 1 calibrated on the finite-k response (arm D).
2. **k = 0 data do not identify the orders ≥ 1 (K4).** Arm B moved between 10 and 30 calibration cells by 8 points at k₂,
   with the same k = 0 fit.
3. **The order-0 coefficient is local at this cutoff (K5, K6).** Arm C's joint matching is consistent, and L's
   per-molecule calibration is a candidate for its order-0 sector.
4. **In the cells' χ(k) the orders ≥ 1 are one number, γ, and the regulator's k⁴ is the frozen form's (§6).** At the
   2.0 Å edge, arm D can reach ±1.1 % at k₁ and k₂ together.

## 8. Next (proposed, not registered)

**Job 4: arm D at matching level, with H3 beside it.**

- **Arm D.**
  - Arm C's read-out, then its orders ≥ 1 columns calibrated on the calibration cells' finite-k response.
  - The rows: each cell's χ_mode, 12 a cell. The probes' rows and arm C's liquid rows are kept.
  - It is the derivative counting's second step. The data fix γ, and the probes decide the rest of the orders ≥ 1.
- **Read at matching level:**
  - the cells' χ at k₁ and k₂ (§6d: ±1.1 % if the ceiling binds);
  - the molecules, as a prediction;
  - what the probes and T5 pay;
  - ε₀, which the orders ≥ 1 do not enter.
- **H3 beside it, at 1.5 and 2.5 Å.**
  - At each edge, arms A → C → D at matching level, from the sweep's matching and the two calibrations.
  - Read the residual edge dependence of χ(k), the probes and T5.
  - §6e predicts ±0.3 and ±2.7 % on χ.
  - The edge stays the a-priori 2.0 Å (`e4-response-regulator.md` §3: not chosen by an observable).
- **Then training,** if arm D holds at matching level: its two C rungs, 4 seeds, about 26 GPU hours.
- **Options, not proposed now:**
  - the molecules' finite-k rows (12 × 64 × 3 a cell): a forward-mode Jacobian over the ~3200 order ≥ 1 coefficients,
    several GPU hours at matching level;
  - L's per-molecule order 0 under arm D's orders ≥ 1, as a second arm.
