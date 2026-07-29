# Interpretation of Results — Parameterized Spectrum (aperiodic + alpha peak)

**Stage 5 of 5.** Source: `results/metric_eval_5subj.csv` (5 subjects / 21 subject-days;
methods run: linear, spline). ZUNA column **pending** — see §5.

## 1. What it indexes
Five posterior parameters from one average-referenced occipito-parietal spectrum: the 1/f
`aperiodic_exponent` and `aperiodic_offset`, plus the alpha peak's center frequency
(`alpha_cf`/IAF), power-above-background (`alpha_pw`), and bandwidth (`alpha_bw`). Drop set is the
**entire** posterior neighborhood (`O1 O2 OZ POZ PO3 PO4`) — the hardest possible test, since every
reconstructed value comes from surviving frontal/central/temporal channels with **no real posterior
data left**.

## 2. Same-day reliability floors
Floors are tight where the literature predicts: `aperiodic_exponent` 0.164 and `alpha_pw` 0.183 are
demanding; `aperiodic_offset` 0.199 is moderate; `alpha_cf` 0.381 and `alpha_bw` 0.512 are looser
(the two grid-quantized, day-to-day-variable ones). Four of five parameters are stable enough
day-to-day to give reconstruction a genuinely tight target.

## 3. Linear vs spline, per submetric — and why
**Alpha peak trio is preserved by both methods.** `alpha_cf` (linear 0.185, spline 0.065), `alpha_pw`
(0.135 / 0.155), `alpha_bw` (0.256 / 0.196) all sit comfortably under floor. Peak *frequency/shape*
is a relative spectral feature that interpolation reproduces even from remote channels; spline's cf
error (0.065) is near one Welch bin.

**Aperiodic offset fails for both** (linear 0.349, spline 0.245; floor 0.199). Offset is *absolute
broadband power* at f=1 Hz — the one magnitude quantity here — and it cannot be recovered when the
whole posterior neighborhood is gone. Linear **collapses** it (mean truth 1.43 → recon 1.16, with
per-recording near-zero blow-ups of 0.00 and 0.38): spatial averaging bleeds away local posterior
power. Spline **overshoots** (→1.63). **Exponent** is a knife-edge: linear just clears (0.153),
spline just misses (0.183).

## 4. Easy or hard vs FAA
**Mixed, and easier than FAA overall.** The alpha peak is effectively *solved* by classical methods;
the aperiodic *magnitude* is the open problem. Linear passes 4/5 here (fails only offset); FAA's
linear fails both submetrics (0.311/0.321 over 0.208/0.301). Spline passes 3/5 (over on offset and
exponent).

## 5. The bar ZUNA must clear
Per submetric, error **< floor** and ideally **< best classical**:
- `aperiodic_offset` **< 0.199** — the prize: **no classical method clears it** (best spline 0.245).
  Recovering posterior broadband power is where ZUNA can win outright.
- `aperiodic_exponent` **< 0.153** (beat linear).
- `alpha_pw` **< 0.135**; `alpha_cf` **< 0.065**; `alpha_bw` **< 0.196** — must not regress the trio
  classical already preserves (don't over-index on noisy `alpha_bw`).

**ZUNA numbers are not yet computed** — the GPU reconstruction pass has not been run, so no ZUNA
column exists in `metric_eval_5subj.csv`. This verdict is provisional until that pass lands.

## ZUNA (G001 sample)
G001 5-day floor / ZUNA error: aperiodic_exponent 0.264 / **0.807**, aperiodic_offset 0.359 / 0.382,
alpha_cf 0.800 / **1.900 Hz**, alpha_pw 0.187 / **0.448**, alpha_bw 0.950 / 1.150 — ZUNA is OVER on ALL
five and worse than both classical methods. Most telling: the posterior alpha peak frequency is off by
~1.9 Hz and the 1/f exponent by 0.8 — ZUNA reconstructs the posterior spectral SHAPE poorly. The peak
parameters that spline/linear preserved best are exactly where ZUNA fails hardest.
