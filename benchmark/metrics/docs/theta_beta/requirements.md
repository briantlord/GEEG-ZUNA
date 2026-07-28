# Requirements — Theta/Beta Ratio (`theta_beta`)

Stage 1 of 5 (Requirements) for the `m_theta_beta.py` metric plug-in in the modular
biomarker-preservation harness. This document is the contract the implementation must satisfy.
It touches **docs only** — no code is written or modified at this stage.

| Field | Value |
|---|---|
| `key` | `theta_beta` |
| `name` | Theta/Beta Ratio |
| Module | `benchmark/metrics/m_theta_beta.py` |
| `submetrics` | `["tbr_cz", "tbr_fz"]` |
| `drop_channels` | `["CZ", "FCZ", "FZ", "CPZ"]` |
| Reference frame | Scalp (surviving-channel average reference) PSD via `C.mean_psd` — **not** CSD |
| Bands | theta 4–8 Hz, beta 13–30 Hz |

---

## 1. Scientific motivation & meaning

The **Theta/Beta Ratio (TBR)** is the ratio of low-frequency theta-band power (4–8 Hz) to
higher-frequency beta-band power (13–30 Hz), measured over frontocentral cortex. It is the
single most-studied quantitative-EEG (qEEG) spectral index in psychiatry and is the classic
electrophysiological correlate of **attention and cortical arousal**.

Psychophysiologically, theta reflects lower-arousal / drowsy / idling cortical states, while beta
reflects active, alert cortical engagement. A **high TBR therefore indexes lower cortical arousal
and is associated with inattention**; it is elevated (on average) in ADHD populations relative to
controls, and TBR at Cz was the basis of the first FDA-cleared qEEG aid for ADHD assessment
(the "NEBA" theta/beta measure). It is also used as a trait marker of trait anxiety / attentional
control and as a neurofeedback training target.

**Important caveats** (to be reflected in interpretation, not in the computation):
the ADHD group-level effect has weakened in later cohorts and the ratio is **not** a stand-alone
diagnostic; it is heterogeneous across labs, sensitive to eyes-open vs eyes-closed state and to
band definitions. For this harness the point is not diagnosis but **preservation**: whether a
reconstruction method reproduces the TBR that the full-montage recording would have yielded.

---

## 2. Exact definition & frequency bands

Per channel, on the scalp (average-referenced) power spectral density (PSD):

```
theta = bandpower(f, psd_ch, 4, 8)     # µV²/Hz integrated over [4, 8) Hz
beta  = bandpower(f, psd_ch, 13, 30)   # µV²/Hz integrated over [13, 30) Hz
tbr   = ln(theta + eps) - ln(beta + eps)     # == ln(theta / beta), eps-guarded
```

- **Log ratio, natural log (ln).** The metric is stored as `ln(theta/beta)`, not the raw ratio.
  Logging stabilizes the heavy-tailed distribution of a power ratio, symmetrizes it around 0,
  makes the reconstruction error additive, and matches the framework convention already used by
  `common.log_asymmetry`. (Base of the log only rescales; ln is chosen for consistency.)
- **Bands.** theta = 4–8 Hz, beta = 13–30 Hz, exactly as fixed by the harness (`common`).
  `C.bandpower` integrates with the trapezoid rule over the **half-open** interval `[lo, hi)`
  and is guaranteed non-negative (PSD ≥ 0).
- **PSD source.** `f, psd = C.mean_psd(data_uV)` — Welch PSD (sfreq 256 Hz, `nperseg =
  min(1024, n_times)`, ≈0.25 Hz resolution) averaged over epochs, giving `psd[n_ch, n_f]`.
  The PSD is computed on the **referenced scalp signal directly**, with **no CSD / surface
  Laplacian** transform (see §5).
- `psd_ch` is the row of `psd` for the target channel, selected by name via `C.ix`.

`eps = 1e-20` is a numerical floor that only matters in the degenerate zero-power case (§7);
for real EEG it is negligible and `tbr ≈ ln(theta/beta)`.

**Sign note.** Because the beta integration window (17 Hz wide) is far wider than theta
(4 Hz wide), integrated beta power can exceed integrated theta power at rest, so `tbr` may be
**negative**. That is valid and expected — the implementation must **not** clamp or rectify the
sign. Only reproducibility of the value matters.

---

## 3. Electrodes used & the drop set that tests it

**Recording sites (outputs are read here):**
- **Cz** — primary frontocentral-midline site; the canonical TBR electrode (Monastra/NEBA).
- **Fz** — secondary frontal-midline site, reported alongside Cz.

**Drop set (channels removed to create the preservation test):**
`["CZ", "FCZ", "FZ", "CPZ"]` — the midline cluster **Cz, FCz, Fz, CPz**.

Rationale:
- The two output electrodes (Cz, Fz) are in the drop set, so after dropping them the metric can
  only be recovered from a **reconstruction** of those channels — which is exactly what the
  benchmark scores.
- **FCz and CPz** (Cz's immediate anterior/posterior midline neighbors) are also dropped. If only
  Cz and Fz were removed, their nearest neighbors would remain and spline/linear interpolation
  could recover them almost trivially, making the test uninformative. Removing the contiguous
  midline strip forces the reconstruction to bridge a genuine spatial gap and makes preservation
  of frontocentral spectral shape a real challenge — the intended stress test for ZUNA vs
  classical interpolation.
- The drop set is matched **case-insensitively** by the harness (`drop_indices`), and only names
  actually present in a given montage are dropped; missing names are silently ignored.

---

## 4. Submetric outputs & units

| Submetric | Channel | Definition | Units |
|---|---|---|---|
| `tbr_cz` | Cz | `ln(theta/beta)` at Cz | dimensionless (natural-log power ratio, i.e. nats) |
| `tbr_fz` | Fz | `ln(theta/beta)` at Fz | dimensionless (natural-log power ratio, i.e. nats) |

- Both values are **dimensionless**: theta and beta are both in µV²/Hz, so their ratio is
  unitless and its logarithm is in log-ratio units (nats). A change of +ln(2) ≈ 0.69 means the
  raw theta/beta ratio doubled.
- The returned dict keys **must exactly equal** `submetrics = ["tbr_cz", "tbr_fz"]`.
- Each value is a **finite Python float**, or `float('nan')` when not computable (§7). No `inf`.

---

## 5. Reference frame: scalp vs CSD, and why

**Use the scalp average-referenced PSD (`C.mean_psd`). Do NOT apply `C.csd`.**

- TBR is **defined and clinically validated on scalp-referenced power.** The thresholds,
  effect sizes, and normative values in the literature (Monastra et al.; Snyder & Hall; NEBA)
  are all derived from referenced scalp montages. Computing it on CSD would produce a number that
  is not comparable to any published TBR and would change what the metric *means*.
- **CSD (surface Laplacian) is a spatial high-pass filter.** It sharpens topography by
  subtracting a weighted neighbor average, which reweights the spectrum toward focal/high spatial-
  frequency sources and alters the relative theta-vs-beta power balance. That is desirable for a
  spatial-specificity metric like FAA (which is why `m_faa` uses CSD) but it would **distort the
  band-power ratio** that TBR is built on.
- The midline sites of interest are also poorly served by a Laplacian: at Cz the Laplacian is
  dominated by the very neighbors (FCz, CPz, C1/C2) that this benchmark drops, so a CSD-based TBR
  would be unstable precisely in the drop condition.

The harness already delivers `data_uV` in the **surviving-channel average-reference frame**, so
the metric must **not** re-reference — it computes the PSD on the data as received.

---

## 6. Test–retest reliability & the floor it is judged against

- **Published reliability.** Eyes-closed resting TBR has **good short-term test–retest
  reliability, ICC typically ≈ 0.7–0.9** for frontocentral sites (better eyes-closed than
  eyes-open; better within-session/within-day than across weeks). This makes TBR a sensible
  preservation target: it is stable enough that a reconstruction error should be judged against a
  meaningful, small change.
- **What "the floor" means in this harness.** For every recording the runner first computes the
  metric on the **full montage** (`kind='truth'`) as the reliability reference, then drops the
  channel cluster, reconstructs each method, and logs `abs_err = |value_recon − value_truthframe|`
  per submetric. A reconstruction method is judged by how small that error is **relative to the
  metric's own between-session variability** — i.e. reconstruction must not perturb TBR by more
  than the natural test–retest noise.
- **Concrete acceptance guidance for later stages.** With ICC in the 0.7–0.9 range and a typical
  between-session `ln(theta/beta)` spread on the order of a few tenths of a nat, a reconstruction
  that preserves TBR should keep `abs_err` **well below ~0.1–0.2 nats** (comfortably inside the
  session-to-session change band). This is guidance for the evaluation/analysis stages, not a
  branch in `compute` — the metric itself only produces the value.

---

## 7. Edge cases & robustness requirements

The `compute(data_uV, ch_names)` implementation must be defensive and always return finite floats
or explicit NaN — never raise, never return `inf`.

1. **Missing target channel.** Guard each output with `C.has(ch_names, 'CZ')` / `C.has(ch_names,
   'FZ')` (or `C.ix` returning `None`). If Cz is absent, `tbr_cz` must be `float('nan')`; if Fz is
   absent, `tbr_fz` must be `float('nan')`. A missing channel for one output must not prevent the
   other from being computed. (Per the base contract, a submetric key may also simply be omitted;
   emitting NaN is preferred so the row is explicit.)
2. **Tiny / zero band power → log instability.** `bandpower` is non-negative and can be exactly 0
   (e.g. a flat/interpolated dead channel) or extremely small. Compute the log ratio as the
   **difference of eps-guarded logs**: `ln(theta + eps) - ln(beta + eps)` with `eps = 1e-20`.
   This prevents `log(0) = -inf` and division-by-zero. Do not compute `ln(theta/beta)` directly on
   a raw quotient that could be `0/0` or `x/0`.
3. **Degenerate / non-finite result.** After computing, coerce to `float` and verify
   `np.isfinite`. If the value is not finite (NaN or ±inf leaking from an all-NaN PSD, empty band,
   or upstream artifact), return `float('nan')` for that submetric.
4. **Empty or malformed band selection.** If the frequency vector yields no bins in a band (should
   not happen at 256 Hz / ≥0.25 Hz resolution, but must be safe), treat the band power as
   effectively zero and fall through to the eps-guarded / NaN handling above rather than erroring.
5. **Do not re-reference or transform.** No average-reference, no CSD, no filtering inside
   `compute` — the data arrives already in the correct frame (§5). The metric is PSD → bandpower →
   log ratio only.
6. **Channel lookup is case-insensitive** via the `common` helpers (`up`, `has`, `ix`); do not
   assume a particular case for `ch_names`.
7. **Determinism.** Given the same `data_uV`/`ch_names`, the output must be deterministic (Welch
   PSD is deterministic); no randomness or global state.

---

## 8. Literature references

1. **Monastra, V. J., Lubar, J. F., & Linden, M. (2001/1999).** *Assessing attention-deficit
   hyperactivity disorder via quantitative electroencephalography: An initial validation study.*
   Neuropsychology, 13(3), 424–433. — Establishes the theta/beta ratio at **Cz** as the primary
   frontocentral qEEG index for ADHD/attention; basis of subsequent clinical use.
2. **Snyder, S. M., & Hall, J. R. (2006).** *A meta-analysis of quantitative EEG power associated
   with attention-deficit hyperactivity disorder.* Journal of Clinical Neurophysiology, 23(5),
   440–455. — Meta-analytic support for elevated theta/beta and the band conventions used here.
3. **Arns, M., Conners, C. K., & Kraemer, H. C. (2013).** *A decade of EEG theta/beta ratio
   research in ADHD: A meta-analysis.* Journal of Attention Disorders, 17(5), 374–383. —
   Documents the **declining/heterogeneous** effect size; source for the interpretive caveats in
   §1 (TBR is a robust, reliable spectral index but not a stand-alone diagnostic).

---

## Design-decision summary (drives implementation)

- Compute on **scalp average-referenced PSD** via `C.mean_psd(data_uV)` — never CSD.
- Bands **theta 4–8**, **beta 13–30** via `C.bandpower` (half-open, trapezoid, non-negative).
- Output **`ln(theta/beta)`** per channel as `ln(theta+eps) - ln(beta+eps)`, `eps = 1e-20`.
- Read **Cz → `tbr_cz`**, **Fz → `tbr_fz`**; guard each with `C.has`/`C.ix`, NaN if missing or
  non-finite; sign is meaningful and must not be clamped.
- Drop set **`["CZ","FCZ","FZ","CPZ"]`** removes the midline cluster so reconstruction faces a real
  spatial gap around the recording sites.
- Judged by `abs_err` of the reconstructed value against the full-montage truth, targeting
  preservation within TBR's ICC ≈ 0.7–0.9 test–retest band (≲0.1–0.2 nats).
