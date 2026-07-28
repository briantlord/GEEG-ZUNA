# Requirements — Frontal Midline Theta (`frontal_midline_theta`)

**Stage 1 of 5 (Requirements).** This document specifies the metric so the implementation
(`benchmark/metrics/m_frontal_midline_theta.py`) can be written against a fixed contract. It does not
change any shared framework file (`base.py`, `common.py`, `run.py`, `aggregate.py`).

| Field | Value |
|---|---|
| `key` | `frontal_midline_theta` |
| `name` | Frontal midline theta |
| Module | `benchmark/metrics/m_frontal_midline_theta.py` |
| `submetrics` | `["fmt_fz", "fmt_rel"]` |
| `drop_channels` | `["FZ", "FCZ", "F1", "F2"]` |
| Reference frame | Scalp, average-reference (the frame the runner passes in) — **not** CSD |
| Bands | theta **4–8 Hz** (project convention; alpha 8–13, beta 13–30 for reference) |

---

## 1. Scientific motivation — what the metric means

**Frontal midline theta (FMt)** is a 4–8 Hz oscillation that appears over the frontal midline of the
scalp, maximal at **Fz / FCz**. Source-localization, MEG, and intracranial work converge on a generator
in the **medial prefrontal cortex / anterior cingulate cortex (ACC)**. It is one of the most robust
electrophysiological correlates of **cognitive control, working-memory load, and sustained attention**:

- FMt power increases monotonically with **working-memory load** (e.g., n-back, Sternberg) and with task
  difficulty and time-on-task during sustained attention (Onton et al. 2005; Gevins et al. 1997).
- FMt is the leading candidate "mechanism of cognitive control" — it phase-organizes distant regions when
  the system must exert control after conflict, errors, or negative feedback (Cavanagh & Frank 2014).
- Because the generator is medial-frontal and midline, FMt has a characteristic **focal frontal-midline
  scalp topography** that falls off toward lateral and posterior sites. This topographic specificity is
  what distinguishes true FMt from generic broadband theta, drowsiness theta, or posterior/temporal theta.

**Clinical / applied relevance.** FMt indexes prefrontal engagement and is studied in ADHD, aging and
cognitive decline, meditation and flow states, mental-workload monitoring, and as a neurofeedback target.
For this harness the point is narrower: FMt is a **generator-specific, topographically focal biomarker**,
so it is a good stress test of whether a channel-reconstruction method preserves a localized frontal-midline
source rather than smearing or borrowing power from neighbors.

---

## 2. Exact definition & frequency bands

**Theta band: 4–8 Hz.** Consistent with the project band convention (theta 4–8, alpha 8–13, beta 13–30).
Band power is the trapezoid integral of the PSD over the half-open interval `[4, 8)` Hz via
`C.bandpower(f, psd, 4, 8)`.

The PSD is the epoch-averaged Welch periodogram of the **scalp (average-reference) signal**, obtained from
`C.mean_psd(data)` → `(f, psd[n_ch, n_f])` at `sfreq = 256`. No CSD transform is applied (see §5).

### Submetrics

Let `P_theta(ch) = C.bandpower(f, psd[ch], 4, 8)` be the theta band power (µV²) of channel `ch`.
Let `POST = {O1, O2, OZ, PZ, POZ} ∩ present-channels`.

1. **`fmt_fz`** — log theta band power at Fz:
   ```
   fmt_fz = ln( P_theta(Fz) )
   ```
   A guarded log is used to avoid `-inf`/`nan` on tiny/zero power (see §7):
   `ln(P_theta(Fz) + eps)` with `eps = 1e-20` (matching the `common.py` convention in `log_asymmetry`).

2. **`fmt_rel`** — frontal-midline theta relative to posterior theta (topographic-specificity index):
   ```
   fmt_rel = fmt_fz - ln( mean_{ch in POST} P_theta(ch) )
           = ln( P_theta(Fz) ) - ln( mean posterior theta power )
   ```
   i.e. the **natural-log ratio** of Fz theta power to the mean theta power across the posterior channels
   that are present. Same guarded log (`+ eps`) on the posterior mean.

`fmt_rel` is deliberately a *contrast*, not an absolute level: it isolates the degree to which theta is
**concentrated at the frontal midline** versus distributed posteriorly. This is the quantity that separates
genuine ACC-driven FMt from global/posterior theta, and it is largely invariant to any global amplitude
scaling (an overall gain multiplies both Fz and posterior power and cancels in the log-difference), which
also makes it robust to the harness's global z-score normalization.

---

## 3. Electrodes used & the drop set

### Electrodes read by `compute`
- **Primary site:** `FZ` (frontal midline; the FMt maximum).
- **Posterior reference set for `fmt_rel`:** `O1, O2, OZ, PZ, POZ` — whichever are present.

`FCZ`, `F1`, `F2` are *not* read by `compute`; they appear only in the drop set (below).

### Drop set — `["FZ", "FCZ", "F1", "F2"]` and why
The drop set is the group of channels removed by the benchmark before reconstruction; the metric is then
recomputed on the reconstructed data and compared to truth. These four are the **tight electrode
neighborhood of the frontal-midline generator**:

- **FZ** — the metric's own recording site; must be reconstructed for the test to mean anything.
- **FCZ** — the adjacent midline site (the other canonical FMt maximum); carries nearly the same signal as
  Fz, so leaving it in would let an interpolator copy it verbatim.
- **F1, F2** — the immediate lateral neighbors of Fz; they share the most mutual information with Fz and
  would make interpolation trivially easy.

Dropping all four forces the reconstruction to recover the frontal-midline theta topography from **more
distant** channels (Fpz/AFz anteriorly, FC1/FC2 and Cz posteriorly, F3/F4 laterally) rather than by
copying a near neighbor. That is the real test: does the method preserve the **ACC/medial-frontal
generator**, or does it merely exploit local spatial redundancy? A method that only interpolates from the
nearest electrode will fail this drop set on `fmt_fz`. Note the posterior set (`O1,O2,OZ,PZ,POZ`) is
intentionally **outside** the drop set, so `fmt_rel`'s denominator is measured on untouched channels and
the ratio's error is attributable to the reconstructed numerator (Fz).

---

## 4. Submetric outputs & units

| Submetric | Definition | Units |
|---|---|---|
| `fmt_fz` | `ln(P_theta(Fz))` | natural-log of power, i.e. `ln(µV²)` (nats, relative to a 1 µV² implicit reference). Dimensioned quantity under log; used consistently, so comparisons/differences are meaningful. |
| `fmt_rel` | `ln(P_theta(Fz)) - ln(mean posterior P_theta)` | **dimensionless** log-ratio (nats). Positive ⇒ theta more concentrated at Fz than posteriorly. |

Both are single finite floats per recording. `fmt_rel` is the difference of two logs, so any consistent
power unit and any global amplitude scale cancels; `fmt_fz` retains the absolute (log) power level and is
therefore the more scale-sensitive of the two.

---

## 5. Reference frame — scalp (average-reference), not CSD

`compute` runs on the **scalp average-reference** data exactly as the runner supplies it, via
`C.mean_psd(data)`. It does **not** apply `C.csd(...)`. Rationale:

- **The metric is a power/level metric, not an asymmetry.** FMt is defined in the literature as theta
  *power* over the frontal midline on referenced montages. CSD (surface Laplacian) is a spatial
  high-pass that sharpens focal sources and removes the shared reference — appropriate for the FAA
  asymmetry metric (`m_faa.py`), where reference-independence is the whole point — but for an absolute
  power level it would change the quantity being measured and discard the broad frontal-midline
  distribution that defines FMt.
- **`fmt_rel` depends on the scalp topographic distribution.** The frontal-vs-posterior theta ratio is a
  statement about the referenced scalp field. Laplacian-transforming both terms would redefine the
  contrast in terms of local curvature rather than field amplitude.
- **Consistency with the benchmark frame.** The harness delivers data already in the surviving-channel
  average-reference frame; computing FMt in that same frame keeps the truth-vs-reconstruction comparison
  in one well-defined space and avoids the extra montage/interpolation assumptions the CSD step introduces.

(The one deliberate divergence from `m_faa.py`, which does use CSD, is therefore intentional and specific
to this metric's meaning.)

---

## 6. Expected test-retest reliability / the judging floor

The benchmark judges reconstruction error `|recon − truth|` against a **same-day test–retest reliability
floor** computed for each submetric (the natural scatter of the metric when nothing is reconstructed). A
reconstruction "passes" when its mean absolute error does not exceed that floor.

Expected reliability for FMt:

- **Moderate-to-good.** FMt is a well-established, reproducible signal. Reliability is **highest during
  cognitive task** (elevated, sustained ACC theta) and **lower but still measurable at rest**, where
  frontal theta is smaller and closer to noise.
- **`fmt_fz`** is an absolute log-power level, so it inherits session-to-session variance in overall theta
  amplitude (electrode impedance, arousal, drowsiness). Expect a **moderate** floor — wider than tightly
  reproducible spectral landmarks like IAF, comparable to other single-site band-power measures.
- **`fmt_rel`** cancels global amplitude and isolates topography, which *can* tighten its floor; but its
  denominator (posterior theta) is small at rest, so on resting data the ratio can be noisier. Expect a
  **moderate** floor as well; do not assume it is dramatically tighter than `fmt_fz` on resting recordings.

The implementation is not required to hit any particular number — the floor is derived empirically by the
aggregator. The requirement here is that both submetrics be **deterministic and stable** given identical
input, so the floor reflects physiology and reconstruction, not compute-side jitter.

---

## 7. Edge cases & robustness requirements

`compute(data, ch_names)` must always return finite floats or explicit `float('nan')`; it must never raise.
Keys returned must be a subset of `submetrics` (missing keys are skipped per-recording by the runner).

1. **Missing Fz.** Guard with `C.has(ch_names, 'FZ')`. If Fz is absent, **both** `fmt_fz` and `fmt_rel`
   are `float('nan')` (neither is computable without the numerator). Use `C.ix(ch_names, 'FZ')` to locate
   the channel; do not assume ordering.
2. **No posterior channels present.** If `POST = {O1,O2,OZ,PZ,POZ} ∩ present` is empty, `fmt_rel` is
   `float('nan')` while `fmt_fz` is still returned normally. Build the posterior set by testing each label
   with `C.has`/`C.ix`; use only the channels that are actually present (partial sets, e.g. only `PZ,POZ`,
   are valid — average over whatever is present).
3. **Tiny / zero power → log instability.** Theta band power is non-negative (`C.bandpower` integrates a
   non-negative PSD). Zero or near-zero power would make `ln(·)` return `-inf` or error. Guard every log
   with an additive floor: `ln(x + 1e-20)` (the established `common.py` convention). This yields a large
   but finite negative number rather than `-inf`, keeping the output finite.
4. **NaN / non-finite input.** If the input data or resulting PSD contains `nan`/`inf`, band power will be
   non-finite; the corresponding submetric must be returned as `float('nan')` rather than propagated as a
   raw non-finite float. Apply a final `np.isfinite` check on each computed value and coerce non-finite
   results to `float('nan')`.
5. **Case-insensitive channel matching.** All channel lookups go through `common.py` helpers
   (`C.has`, `C.ix`, `C.up`), which upper-case labels; never compare raw strings.
6. **Determinism.** No randomness, no reliance on channel order, no global state. Same input ⇒ same output.
7. **Single-epoch / short data.** `C.mean_psd` averages over epochs and `C.welch` sets
   `nperseg = min(1024, n_times)`, so a single epoch or short segment is handled without special-casing;
   the 4–8 Hz band still integrates correctly at 256 Hz. No additional guard required, but the code must
   not assume `n_epochs > 1`.

---

## 8. Literature references

1. **Cavanagh, J. F., & Frank, M. J. (2014).** Frontal theta as a mechanism for cognitive control.
   *Trends in Cognitive Sciences, 18*(8), 414–421. — FMt as the core signature of medial-frontal
   cognitive control.
2. **Onton, J., Delorme, A., & Makeig, S. (2005).** Frontal midline EEG dynamics during working memory.
   *NeuroImage, 27*(2), 341–356. — Load-dependent frontal-midline theta and its ACC/medial-frontal source.
3. **Gevins, A., Smith, M. E., McEvoy, L., & Yu, D. (1997).** High-resolution EEG mapping of cortical
   activation related to working memory. *Cerebral Cortex, 7*(4), 374–385. — Frontal-midline theta scales
   with working-memory load and task difficulty (background/topography).

---

### Implementation summary (non-binding, for stage 2)
- `f, psd = C.mean_psd(data)` on the average-reference `data` (no CSD).
- `fmt_fz = ln(bandpower(f, psd[Fz], 4, 8) + 1e-20)`, guarded on missing Fz → nan.
- `fmt_rel = fmt_fz - ln(mean over present {O1,O2,OZ,PZ,POZ} of bandpower(4,8) + 1e-20)`, nan if none present.
- Register `Metric(key='frontal_midline_theta', name='Frontal midline theta',
  drop_channels=['FZ','FCZ','F1','F2'], submetrics=['fmt_fz','fmt_rel'], compute=compute, ...)`.
- Coerce any non-finite result to `float('nan')`.
