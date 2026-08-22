# Interpretation of Results — Sensorimotor Mu Asymmetry (`mu_asymmetry`)

Stage 5 of 5 (INTERPRETATION). Numbers are from `results/metric_eval_5subj.csv` (5 subjects /
21 subject-days; methods run: `linear`, `spline`). ZUNA is not yet computed.

## 1. What it indexes

`mu_asym = ln(bp C4) − ln(bp C3)`, the right−left log-power asymmetry of the 8–13 Hz sensorimotor
mu rhythm on CSD, plus the two single-channel log band powers `mu_c3` / `mu_c4`. The drop set is
`C1+C2+C3+C4` — the two recording sites *and* their medial in-row neighbors — so reconstruction
must infer central-hand mu from distant surviving rows (FC/CP, Cz, T7/T8), not from a trivial
adjacent-electrode interpolation.

## 2. Same-day reliability floor

Floors: `mu_asym` 0.305, `mu_c3` 0.267, `mu_c4` 0.277 nepers. These are moderate and tightly
clustered (~0.27–0.31), so the metric is reasonably stable Rest1-vs-Rest2 and gives a real,
non-trivial target. The two absolute single-channel powers are *slightly tighter* than the
difference score, as expected (a difference compounds both channels' error). All three sit near
FAA's lateral floor (0.301) and above FAA mid-frontal (0.208).

## 3. Linear vs spline

**Spline preserves all three** (0.184 / 0.193 / 0.180, every one under floor). It rebuilds the
central-hand cluster with tight amplitude fidelity, so both single channels *and* their ratio hold.

**Linear fails all three** (0.326 / 0.652 / 0.780, all OVER). The single-channel powers fail
hardest — ~2.4× and ~2.8× their floors — because interpolating C3/C4 from distant rows
systematically attenuates the reconstructed central amplitude (per-recording, linear pushes
`mu_c3`/`mu_c4` roughly a full neper more negative; e.g. G001Day2Rest2 `mu_c4` −13.91 → −15.30).
The asymmetry only *just* fails (0.326 vs 0.305): scale-invariance lets the two channels' shared-sign
biases partly cancel in the difference, protecting the ratio far better than the absolute powers.

## 4. Easy or hard? vs FAA

Mixed, and the same shape as FAA. The **absolute single-channel powers are the hard part** — a naive
interpolator misses them by ~2.5–3×, worse than linear misses FAA (`faa` 0.311). The scale-invariant
`mu_asym` is the most forgiving of the three under linear. So `mu_c3`/`mu_c4` are among the more
demanding targets in the suite for any method lacking per-channel amplitude fidelity.

## 5. The bar ZUNA must clear

Per submetric, mean recon error **< floor** (`mu_asym` < 0.305, `mu_c3` < 0.267, `mu_c4` < 0.277)
and, to matter, **beat the best classical method** — spline's 0.184 / 0.193 / 0.180. The single-channel
powers are the crux: this is exactly the absolute per-channel amplitude fidelity ZUNA missed on FAA's
F4/F8. **ZUNA is not yet computed here** — the column is pending and requires the GPU pass; no mu
verdict can be claimed until then.

## ZUNA (5 subjects)
Floors: mu_asym 0.305, mu_c3 0.267, mu_c4 0.277. ZUNA: mu_asym **0.296** (ok — the scale-invariant ratio,
just under floor) but mu_c3 **0.566** and mu_c4 **0.579** (both OVER, vs spline 0.193 / 0.180). Confirms the
pattern: ZUNA preserves the C4/C3 ratio but not each channel's absolute mu power.
