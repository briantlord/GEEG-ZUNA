# Interpretation of Results — Frontal Midline Theta (`frontal_midline_theta`)

**Stage 5 of 5 (Interpretation).** Aggregate from `results/metric_eval_5subj.csv` (5 subjects, 21
subject-days, 42 recon recordings; methods: linear, spline). ZUNA column **not yet computed**.

**1. What it indexes / drop set.** `fmt_fz` = ln theta(4–8 Hz) power at Fz (the ACC/medial-frontal
generator); `fmt_rel` = that same log-power minus ln mean posterior theta (a topographic-specificity
contrast). Drop set `F1+F2+FCZ+FZ` removes Fz and its three nearest neighbors, forcing reconstruction
from distant channels rather than a copy of an adjacent electrode.

**2. Reliability floor.** Same-day Rest1-vs-Rest2 floor is **0.315** ln-units for `fmt_fz` and a much
tighter **0.132** for `fmt_rel`. The tighter `fmt_rel` floor is expected: the frontal-vs-posterior
contrast cancels global amplitude drift (arousal, impedance), so the topography is more reproducible
day-to-day than the absolute level. Both are stable enough to give reconstruction a real target.

**3. Linear vs spline, and why.** Note the reconstruction abs_err is **identical across submetrics**
(linear 1.615 / 1.615; spline 0.332 / 0.332): the posterior denominator is *not* in the drop set, so
that term is bit-identical in recon and truth and cancels in recon−truth — the whole error is the
reconstructed Fz numerator. **Both classical methods exceed both floors.** Linear is catastrophic (mean
1.615, median 0.798, max 8.55): averaging distant, phase-misaligned neighbors cancels theta, collapsing
Fz power to near-zero and driving ln power sharply negative (recon values of −2 to −6). **Spline** is far
better on typical recordings (median **0.087**, below even the 0.132 floor; 33/42 recordings clear 0.315,
29/42 clear 0.132) but its **mean 0.332** is dragged over both floors by ~6 high-theta Day2/Day3
recordings (G003/G004/G005, truth ln-power 3–4.8) where the spherical surface *overshoots* the focal peak
(errors 0.8–2.6).

**4. Easy or hard vs FAA.** **Hard.** Despite a *wider* floor than FAA (0.315 vs 0.208), FMt defeats
both classical methods, whereas spline passes FAA (0.185 < 0.208). FAA is a left–right contrast whose
reconstruction bias partly cancels; `fmt_fz` is an absolute focal level where the full amplitude error
lands directly, and `fmt_rel` inherits it undiluted.

**5. The bar ZUNA must clear.** Because the recon error is shared, the binding target is the **tighter
`fmt_rel` floor: mean recon abs_err < 0.132 ln-units** (which also clears `fmt_fz` at 0.315), and ideally
below spline's 0.332 to be the first method that matters here. Spline's clean median (0.087) shows this
is reachable *if* the high-amplitude-day overshoots are tamed. **The ZUNA column is not yet computed and
requires the GPU reconstruction pass before any verdict.**

## ZUNA (G001 sample)
G001 5-day floor: fmt_fz 0.543 (fmt_rel 0.160). ZUNA: **1.239** — OVER; better than linear (2.260) but
far worse than spline (0.103). ZUNA does not rescue this hard metric — spline reconstructs Fz theta
amplitude cleanly while ZUNA and linear both smear it.
