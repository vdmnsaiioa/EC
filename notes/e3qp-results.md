# E3-QP results — E3 on q-AQUA-pol, read against `e3qp-preregistration.md`

*16 Oct 2026. One run of `configs/water_qaqua_pol.yaml` (both parts), 15 Oct, Niflheim node sd653, one A100, harness
v0.0.10 (commit 0d74e91). Dimer part 3.0 h (7 rungs × 8 seeds), clusters 2.35 h (7 rungs × 4 seeds); the authors'
benchmark passed in the job. The dimer's 2×-best rule excluded 5 of 8 seeds of M_G, 1 of M_A, 4 of M_1μ; none on the
clusters. Percentages are rmse as a share of the signal rms, means over seeds. Section 3 holds what was found after
the run, by computation on the truth (`scratch/e3qp_decomposition.py`, `scratch/e3qp_response_floor.py`). The local
master of this note is `notes/` in the repository.*

## 1. The readings

| reading | registered | result |
|---|---|---|
| M_∞ / M_G / M_A tails | branches 3 / 1 / 3 | ✓ M_∞ 1.000 at every row (window 4.5e-4 E_h); M_G grows without bound (slope +4.2, q → −1); M_A 0.60 at 7.8 Å, → 1.00 from 19.5 Å |
| M_1†, M_16† tails | the computed values to 1e-6 | ✓ every row to the printed digits (M_1† 0.162 … 0.00376, M_16† 0.144 … 0.00376) |
| M_1 tail form | q = 3.00 ± 0.05, flat | ✓ q 3.04 → 3.00 beyond 30 Å, slope +0.01 |
| M_1 bias | class minimum ± 0.03 | ~ +0.027 to +0.030 at 7.8–14.7 Å, +0.031 to +0.033 beyond (0.269 → 0.131 against 0.242 → 0.098) |
| M_1 dipole | 1.72–1.82 D, spread ≤ 2 % | range ✓ 1.739 ± 0.054 D; spread ✗ 3.1 %. Four of eight seeds sit at the sign-flipped charges (q_O ≈ +0.6): the Coulomb energy is even in q, the dipole's sign is not identified by energies |
| M_1μ | tail below M_1's by ≥ 2× at 10–30 Å; 1.79–1.94 D, spread ≤ 1 % | ✓ 14× at 10 Å, 4.6× at 30 Å (0.017 … 0.042 over the rows); 1.902 ± 0.021 D ✓, spread 1.1 % (~) |
| hexamer energy, dimers | every rung < 0.4, dimers ≤ 0.2 kcal/mol | ✗ M_∞ 0.57; the rest 0.31–0.37 ✓; dimers 0.08–0.11 ✓ |
| hexamer non-additive (0.997 kcal/mol) | S0†, 16S0† 20–45 %; S2† ≤ 1.3 × S0†; M_∞ 30–70; M_1†, M_16† 40–90; M_1μ 30–80 | S0† ✓ 23 %, 16S0† ✓ 27, S2† ✓ 22; M_∞ 23 (better than registered), M_1† 36, M_16† 37 (better), M_1μ ✓ 32 |
| open: S0†, 16S0† ≤ M_∞ on ≥ 3/4 seeds | stated expectation | ✗ 1–2 of 4 each (seeds 19/20/26/28 and 22/20/32/32 against M_∞'s 29/20/19/25) |
| probes, non-additive, M_∞, M_1†, M_16† | 100 % at 10–15 Å; 70–105 % at 8 Å | ✓ 100 / 100 / 100 %; 8 Å 94 / 100 / 99 % |
| probes, non-additive, read-out rungs | below 100 % everywhere; 25–65 % at 10–15 Å | ✓ S0† 50–52 %, 16S0† 55–58 %, S2† 55–61 %; 8 Å 56–63 %. M_1μ 46–56 % (registered 50–100) |
| probes, interaction, daggers | the pinned charges' 13–20 % ± 5 | ✓ 13 / 13 / 20 % (M_1†), 13 / 13 / 19 % (M_16†); read-out rungs 10–14 % |
| raw 2-body | every rung 0.05–0.4 kcal/mol | ✓ 0.17–0.25, except M_16†, M_16S0† 1.13 — a reading error, §3.1 |
| raw 3-body, all 4673 | S0†, 16S0† 20–50 %; M_∞ 30–70 % | ✓ 21, 25 %; M_∞ 17 % (better than registered) |
| raw 3-body, 7–9 Å (311) | S0†, 16S0† ≤ 60 % (PES 22) | ✓ 58, 60 % (M_1μ 52, M_∞ 85, M_1† 96, M_16† 100) |
| raw 3-body, ≥ 9 Å (108) | S0†, 16S0† ≤ 60 % on ≥ 3/4 seeds; M_∞, M_1†, M_16† 95–105 % | S0† ✓ 57 % (3/4); 16S0† ✗ 59 % (2/4: 63, 72, 44, 55); M_1μ 47 %; M_∞, M_1†, M_16† ✓ 100 % |
| raw 4-body (267) | every rung 50–130 % | ✓ 71–109 % (M_1μ 71, M_∞ 77, M_1† 94, M_16† 94, S0† 101, 16S0† 107, S2† 109) |

No §5 surprise: the daggers are their computed values; M_∞ and the daggers are at 100 % on the probes at 12 and 15 Å;
no read-out rung reached 100 % at every distance; M_S0† is below M_1† on the hexamers and the trimers; M_1's dipole is
inside 1.58–2.04 D.

## 2. What it says

- **The read-out carries far-field many-body induction, at half strength.** On the probes beyond r_c, where nothing
  but the bands and the read-out acts, every read-out rung removes 40–50 % of the non-additive energy that M_∞ and the
  daggers cannot touch, on every seed; on the CCSD(T) trimers beyond 9 Å M_S0† cuts the error from 100 % to 57 %. This
  is the reading E3-Q could not have (its truth had nothing there).
- **But it learned the response from the clusters, not from the far field.** The read-outs (51–62 % at 10–15 Å) are
  worse than the fixed first-order form with the hexamers' coefficients (41–45 %), the training probes' residual equals
  the test probes' (56 against 55 %), and the order-2 read-out is no better than order 0. The coefficients are set by
  the clusters' short-range residual: the probes entered the loss with their total energies, whose common core energy
  is a hundred times their interaction.
- **Inside the clusters E₀ does the work.** M_∞ ties the read-outs on the hexamers (23 %) and beats them on all trimers
  (17 against 21 %); the read-out helps where E₀ is blind (trimers beyond 7 Å) and nowhere else.
- **Matching.** M_1μ's dipole and tail (1.90 D; 1–4 % from 10 Å) are the dimer's best learned-source result so far:
  atomic dipoles carry the M-site's quadrupole that atom-centred charges cannot (M_1 stays ~0.03 above its class
  minimum).

## 3. Found after the run

**3.1 The M_16 rungs' 2-body readings are a reading error.** `model_nbody` built the held-out frames with the pinned
charges only; the M_16 rungs' dispersion band reads the pinned α, absent there, so their 2-body energies lacked the
band they were trained with (−C₆/R⁶ = −1.2 kcal/mol at the hydrogen bond). Their 3- and 4-body readings are exact
(the band is pairwise and cancels in the subset expansion). Fixed in v0.0.11 (every reading applies the rung's pins).

**3.2 q-AQUA-pol's energy-only routine is off in the 3-body blend region.** `pot3b` (called by `getpot`) mixes the
short- and long-range 3-body fits for trimers whose largest O–O is 5.0–5.5 Å with a switch value it never computes
(q-AQUA v1's `pot3b` calls `f_switch` there; q-AQUA-pol's dropped the call); `pot_gd`, the gradient routine, computes
it, and its forces integrate to its energy. The differences: up to 0.04 kcal/mol per trimer, 0.09 kcal/mol on relaxed
hexamers. Every label of E3-QP came from `pot_gd` (energies and forces together), and the probes' cores were outside
the band (largest O–O 2.96–5.96 Å); what used `getpot` was the Metropolis relaxation (harmless: a sampling
distribution) and the PES's own error on the raw sets: in the 3-body 5–7 Å bin, 629 frames are in the band, where the
PES's rmse is 0.0095 kcal/mol by `pot_gd` against 0.0122 by `getpot`. From v0.0.11 the truth's energy is `pot_gd`'s;
the Metropolis relaxation still uses `getpot` (cheaper, and the data sets reproduce bit for bit).

**3.3 Where the far-field residual comes from** (`scratch/e3qp_decomposition.py`, exact on the truth). On the test
probes the non-additive energy is TTM3-F's induction to 0.0 % at every distance; its first-order part (the cross terms
−α E_j·E_k of the permanent fields at the M-sites) leaves 9–11 %, through second order (α²T) 1.0–1.3 %. Piece (2) of
the rehearsal — the core's own self-consistent dipoles in the probe's permanent field — is 96–99.6 % of it at 10–15 Å
(with first-order core dipoles, 89 %). On the test hexamers the many-body induction (0.772 kcal/mol) is 0.585 first
order and 0.192 mutual (25 %); the CCSD(T) corrections add 0.274 non-additive. So the far field is first order to 90 %,
and the mutual remainder — the "second order and beyond" — is 10 % of the probes' signal and 25 % of the hexamers'.

**3.4 The response field.** The read-out answers the band field of the pinned charges, and the dipole-surface charges
q3 are not TTM3-F's permanent charges: same dipole, half the quadrupole (§1 of the pre-registration). Pinned at their
true sites (TTM3-F's H charges and its M-site), the probes' interaction beyond r_c is the electrostatics plus the
induction exactly; with q3 the electrostatics alone misses by 13–24 % of the interaction — as much as the non-additive
signal. A first-order response with one polarisability per species fitted on the training probes' non-additive energy
leaves, on the test probes, 13–28 % (q3) and 9–15 % (TTM3-F's charges) at 8–15 Å (`scratch/e3qp_response_floor.py`).
And a structural point found while matching (E3-QP2): the truth's atomic forces on a rigid monomer carry the flux of
its geometry-dependent charges (the dipole surface), which pinned charges cannot represent — fitting them corrupts a
far-field response; its energies carry no such term.

## 4. Next

E3-QP2 (`e3qp2-preregistration.md`): the truth's permanent charges exactly (a virtual M-site), the response matched on
the far field — the training probes' interaction energies, before E₀ is trained, with coefficients that do not depend on
E₀ — and the induced dipoles' mutual fields to all orders (self-consistent), against the first-order read-out.
