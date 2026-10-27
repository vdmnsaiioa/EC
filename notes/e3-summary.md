# E3 — water: what it established, and the model it leaves

*27 Oct 2026. Closes E3 on Vadic's decision of 27 Oct to move on to E4 (`e4-plan.md`) and freezes its model. Nothing
new was run: every number is a seed mean over 4 seeds, in kcal/mol, from the results note cited. The local master of
this note is `notes/` in the repository.*

## 1. The question, and the truths

E3 asked whether the scale-separated model carries the long range of a molecular truth with many-body induction. The
model is E₀ inside r_c, bands on pinned or learned sources beyond it, and a read-out of the band fields. "Carries" meant
two things: the far field as computed before training, and the contacts no worse than a short-range model (M_∞).

There were three truths in turn:

- **The synthetic PW model**: the rehearsal and E3-R, 10–14 Oct.
- **q-AQUA v1** (E3-Q). It has no induction or dispersion beyond 7.8 Å and no many-body energy beyond 9 Å, so there was
  nothing long-range to learn.
- **q-AQUA-pol** (E3-QP to E3-QP8, 15–27 Oct): TTM3-F's polarisable long range, with CCSD(T) corrections inside 7 Å.

q-AQUA's raw CCSD(T) 2-, 3- and 4-body sets were held out throughout.

## 2. What it established

- **The far field is set before training, and it holds.** The self-consistent read-out is matched to the truth's far
  field. It leaves 3–5 % of the far probes' non-additive energy at 10–15 Å (M_16C2†ᵐ: 4 / 5 / 3 %). This came out to the
  printed digit on every seed of every run since E3-QP2. On the CCSD(T) trimers beyond 9 Å, which it was never fitted
  to, it leaves 26.7 % (the PES itself leaves 21 %). The frozen-source rungs read 100 % on both: these are M_∞ and the
  daggers, with pinned charges and no response, which is the structure Theorem 2 gives MACE-LES beyond r_c
  (`e3qp2-results.md`; `e3qp8-results.md` §2).
- **The many-body energy needs mutual induction.** The self-consistent C rungs beat the first-order S rungs on the
  hexamers' non-additive energy on every seed (15.8 % against 23.1 %, E3-QP2). They also beat them on the raw 4-body
  (63 % against 110–139 %).
- **The pins cost at contact until E₀ saw the pinned field.**
  - In E3-QP3 and E3-QP4 the pinned rungs trailed M_∞ by 1.4–1.5× on the raw 3-body. Two deficits caused it: E₀'s own,
    and the read-out's near field below O–O 2.7 Å.
  - The field term closed the gap and reversed it. In the field term, E₀'s site energies read the pinned field of the
    other molecules, each source molecule taken whole.
  - From E3-QP5 to E3-QP8:
    - raw 3-body 0.125 → 0.077–0.081 (M_∞ 0.084);
    - raw 2-body 0.081 → 0.051–0.061 (M_∞ 0.113);
    - the hexamers' non-additive energy 12.6 → 10.9–11.3 %.
- **The E₀–self-term split of the 2-body is a gauge** (the split check, `e3qp8-results.md` §5). The data fix the sum,
  and the models keep it off the data. No penalty is needed.
- **Training lessons that carry over:**
  - Far-field identification data must be in training (E3-R: 0.6 of α without them, 1.0 with them).
  - Hundreds of clusters are needed, not 36 (E3-Q).
  - The contacts' energy weight should be set for equal errors (E3-QP4 job B).
  - Field inputs go in reference units (E3-R).
  - Source molecules must be weighted whole (E3-QP6: part charges were a fault).

## 3. The frozen model: M_16C2†ᵐ_us

The rung is `M_16C2pm_us` in `configs/water_qaqua_pol8.yaml`; the trained models are in `results/e3qp8/`. It has four
parts:

- **E₀**: 32 features, 2 layers, r_c 6 Å.
- **Two pinned bands**, both at edge ℓ₁ = 0.75 Å: the Coulomb band on TTM3-F's charges with its M-sites, and the
  dispersion band on its α.
- **The read-out** of the band field to order 2, its order-0 part self-consistent (C2), matched to the truth's far
  field before training. †ᵐ marks the pins: the truth's site model, M-sites included.
- **The field term**: each molecule's view of the other molecules' pinned field.
  - Its self terms are an MLP of E₀'s features, tapered over 4–5 Å.
  - Its cross terms are of degree 2, with four coefficients by species (a_O, a_H, c_O, c_H). They are strictly 3-body.

| E3-QP8, seed means | M_16C2†ᵐ (base) | M_16C2†ᵐ_w | M_16C2†ᵐ_us |
|---|---|---|---|
| raw 2-body (4937 dimers; the PES 0.011) | 0.081 | 0.051 | 0.061 |
| raw 3-body (4673 trimers; labels' rms 0.623) | 0.125 | 0.077 | 0.081 |
| … smallest O–O below 2.7 Å (371) | 0.264 | 0.144 | 0.172 |
| raw 4-body (267 tetramers), % of the labels' rms | 60.4 | 75.7 | **51.7** |
| hexamers: energy; non-additive part | 0.162; 12.6 % | 0.156; 12.0 % | 0.156; 11.3 % |
| hexamers' 4-body; 5-body | 0.059; 0.0054 | 0.054; 0.0084 | **0.042; 0.0029** |
| far probes at 10 / 12 / 15 Å, non-additive | 4 / 5 / 3 % | 4 / 5 / 3 % | 4 / 5 / 3 % |
| CCSD(T) trimers beyond 9 Å | 26.7 % | 26.7 % | 26.7 % |

Why this model was chosen:

- It has the lowest many-body tail of any E3 model.
- Its cross term is four numbers with a physical reading: polarisability-like, in bohr³ and bohr⁵.
- Its E₀–self-term split is a gauge.
- Its seeds agree best off the data: 0.0185 on the scan below 5 Å, against 0.0197–0.0251 for the other rungs.

What it costs against M_16C2†ᵐ_w (paired): +21 % on the 2-body, most of it at 3.0–3.2 Å and off the training pairs,
and +22 % on the compact trimers.

## 4. What stays open

- **The read-out's near field at the closest contacts.** On the 15 compact tetramers its 4-body is 0.57, against the
  labels' 0.34. That is the largest 4-body error of any C rung there. The compact trimers show the same (0.172). Every
  molecule in a liquid has such contacts, so E4 registers this as a prediction (`e4-plan.md` §6).
- **The self terms' reach.** It trades the 5–7 Å tail against the 3.2–5 Å gain and the training forces (E3-QP8
  §3.1–3.2). E3-QP9 was proposed and not run. Its two rungs were the species cross terms with the self terms at full
  reach (_u), and self terms with inputs even in the sources (_ue).
- **The coarse-grained density** beside the pinned charges (Vadic's question of 25 Oct). This is a bulk question and is
  carried to E4.
- **Owed from E3-QP8's follow-up job**, once the read patches are applied:
  - the species numbers as trained (`scratch/e3qp8_species_numbers.py --trimers`);
  - E3-QP6's self-term check on E3-QP8's frames.
- **Not tested in E3:**
  - flexible monomers beyond the near-rigid raw frames;
  - ions;
  - the field of the induced dipoles as E₀'s input;
  - E₀'s backbone;
  - any periodic cell.

## 5. What E4 takes from E3

- **The model form, not its parameters.** E4 retrains M_16C2†ᵐ_us's rung on the new truth, with that truth's pins. The
  ladder around it (M_∞, the learned-source rungs, the daggers) carries the size predictions.
- **The method:**
  - numbers computed from the truth before training;
  - far-field probes and daggers as controls that must come out as computed;
  - the decomposition by body order;
  - the coverage and split checks.
- **The near-field prediction** of §4.
