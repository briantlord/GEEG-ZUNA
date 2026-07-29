# Interpretation of Results — Theta/Beta Ratio (`theta_beta`)

Stage 5 of 5. Aggregate source: `results/metric_eval_5subj.csv` (5 subjects, 21 subject-days,
42 recordings; methods run: linear, spline). ZUNA not yet computed.

**1. What it indexes.** TBR = `ln(theta 4–8 / beta 13–30)` bandpower at **Cz** (`tbr_cz`) and
**Fz** (`tbr_fz`) on the scalp average-referenced PSD (no CSD) — the classic frontocentral index
of cortical arousal/attention. The drop set removes the whole midline cluster
**CZ+FCZ+FZ+CPZ**, so both read sites *and* their nearest anterior/posterior neighbours are gone;
reconstruction must bridge a genuine spatial gap rather than copy an adjacent channel.

**2. Reliability floor.** Same-day Rest1-vs-Rest2 floor is **0.235 nats (Cz)** and **0.228 nats
(Fz)** — near-identical, moderate targets. Against a between-day truth spread of ~0.40 nats SD
(Cz mean 0.65, Fz 0.44), this is consistent with the published ICC ≈ 0.7–0.9: TBR is stable
enough day-to-day that ~0.23 nats is a real, but not razor-tight, bar to beat.

**3. Linear vs spline (why).** **Spline preserves both** — Cz mean err **0.123** (median 0.05),
Fz **0.154** (median 0.06), both roughly half the floor. Spherical-spline reconstruction
recovers the smooth midline potential field, keeping the frontocentral theta concentration
intact. **Linear fails both** — Cz **0.270** (just over 0.235), Fz **0.386** (well over 0.228,
max 2.98). Linear interpolation across the widened gap averages distant lateral channels, smearing
out the focal midline theta while leaving the broader beta band; the ratio collapses and often
**flips sign** (e.g. Fz truth +1.39 → linear −1.57). Fz is worst because it sits at the anterior
edge of the gap with FCz also dropped — linear must extrapolate.

**4. Easy or hard.** **Moderately hard, but cleanly solved by spline** — unlike FAA, where even
spline sits right at the floor (0.185 vs 0.208). Here spline clears the floor with margin; linear
does not. The metric discriminates methods sharply.

**5. The bar ZUNA must clear.** Error **< 0.235 (Cz)** and **< 0.228 (Fz)**, and to matter it must
**beat spline's 0.123 / 0.154**. **ZUNA is not yet computed — pending the GPU pass.**

## ZUNA (G001 sample)
G001 5-day floor: tbr_cz 0.501, tbr_fz 0.462. ZUNA: tbr_cz **2.255**, tbr_fz **2.004** — catastrophic
(~4-5x the floor), far worse than linear (0.437 / 1.030) and spline (0.043 / 0.348). TBR is a ratio of
two bands at ONE channel, so ZUNA's poor spectral-shape reconstruction of the dropped midline channel
wrecks it. Clear fail; spline is decisively the method to use.
