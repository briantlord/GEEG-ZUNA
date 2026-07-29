# Interpretation of Results — Frontal Alpha Asymmetry (`faa`)

Stage 5 of 5 (INTERPRETATION). Source: `results/metric_eval_5subj.csv`, 5 subjects / 21 subject-days,
methods `linear` and `spline`. FAA is the reference metric of the panel.

**1. What it indexes / drop set.** FAA is `ln(alpha F4) − ln(alpha F3)` (mid-frontal, submetric
`faa`) and `ln(alpha F8) − ln(alpha F7)` (lateral, `faa_lat`), 8–13 Hz, on reference-free CSD
(surface Laplacian). Positive ⇒ greater right-frontal alpha ⇒ *less* right-frontal activation
(inverse-power caveat). Both are **scale-invariant log-difference scores**. The drop set removes the
four channels the metric reads — **F3, F4, F7, F8** — so reconstruction must fabricate the very
electrodes it is scored on, not interpolate a surviving neighbor.

**2. Same-day reliability floor.** Mid-frontal `faa` floors at **0.208 nepers**, lateral `faa_lat`
at **0.301** (mean |Rest1−Rest2| over 21 subject-days). The mid-frontal pair is the tighter, more
stable day-to-day target; the lateral pair is noisier (edge electrodes; a difference score compounds
both channels' error), hence the looser floor. Truth values span widely (`faa` sd ≈1.15, min −4.25 /
max +2.53), so ~0.2 nepers is a real but not razor-thin target to beat.

**3. Linear vs spline, and why.** Spline preserves **both** — `faa` **0.185 < 0.208**, `faa_lat`
**0.260 < 0.301** — while linear misses **both** (`faa` **0.311 OVER**, `faa_lat` **0.321 OVER**).
The gap is geometric: spherical-spline reconstruction respects head curvature and rebuilds F3/F4/F7/F8
with little *relative* left/right power bias, so the log-difference survives; linear interpolation
from surviving neighbors injects an asymmetric bias, worst on the lateral edge electrodes (linear
`faa_lat` per-recording errors reach 1.38, `faa` reach 1.51). Because the metric is a log ratio,
global amplitude / z-score error cancels exactly — only the relative F4-vs-F3 (F8-vs-F7) bias
survives, which is precisely what linear gets wrong and spline gets right.

**4. Easy or hard.** FAA sits on the **easy end** of the panel: scale-invariance makes it immune to
the global-amplitude failures that sink absolute-power metrics (frontal-midline-theta, where both
methods overshoot ~5×). A geometry-aware method clears it. Its spatial analog `mu_asymmetry` (C3/C4)
behaves identically — spline ok, linear OVER — confirming the asymmetry construction, not the region,
drives the easiness.

**5. The bar ZUNA must clear.** Error **< floor** and, ideally, **< best classical (spline)**. FAA's
ZUNA column is **already computed** (prior 5-subject run — no GPU pass pending for these two
submetrics): `faa` **0.228 > 0.208 → OVER**, missing the primary mid-frontal floor and losing to
spline's 0.185; `faa_lat` **0.221 < 0.301 → ok**, beating both spline (0.260) and linear (0.321).
So FAA is a **split verdict**: ZUNA preserves and wins on the lateral pair but misses the headline
mid-frontal floor. The miss traces to per-channel amplitude fidelity of reconstructed **F4/F8** (a
residual relative bias), not to z-scoring — which cancels in this scale-invariant metric. No
both-worlds win on FAA.

## ZUNA (G001 sample)
G001 5-day floor: faa 0.357, faa_lat 0.235. ZUNA: faa **0.324** (ok vs G001's loose floor), faa_lat
**0.226** (ok). Consistent with the full 5-subject verdict: ZUNA passes the lateral pair but is
marginal mid-frontal (it exceeds the tighter 5-subject faa floor of 0.208), and it does not beat
spline (0.131 / 0.156) on either. FAA is a scale-invariant log-ratio, which is why ZUNA — whose
per-channel amplitude fidelity is poor — still lands near the floor here while failing the
absolute-power metrics below.
