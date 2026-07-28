# Requirements — Parameterized Spectrum (aperiodic + alpha peak)

**Metric key:** `specparam_peaks`
**Module (stage 2):** `benchmark/metrics/m_specparam_peaks.py`
**Human name:** Parameterized spectrum (aperiodic + alpha peak)
**Stage:** 1 of 5 — Requirements (this document is READ/WRITE docs only; no code yet)

| Field | Value |
|---|---|
| `key` | `specparam_peaks` |
| `submetrics` | `["aperiodic_exponent", "aperiodic_offset", "alpha_cf", "alpha_pw", "alpha_bw"]` |
| `drop_channels` | `["O1", "O2", "OZ", "POZ", "PO3", "PO4"]` |
| Recording site | Posterior (occipito-parietal), averaged over the present channels of the drop set |
| Reference frame | **Scalp average reference** (the frame the runner supplies) — **not** CSD |
| Shared reconstruction | Same drop set as IAF → both metrics reuse one reconstruction per method in `run.py` |
| Core helper | `common.aperiodic_and_peak(f, psd, fit_range=(2,40), peak_band=(7,14), peak_label="alpha")` |

---

## 1. Scientific motivation and meaning

A raw EEG power spectrum is the **sum of two physiologically distinct processes** that are
conventionally reported as one number (e.g. "alpha power") and therefore routinely conflated:

1. **An aperiodic ("1/f") background** — power that falls off smoothly and monotonically with
   frequency, with no preferred rhythm. Its **slope/exponent** tracks the balance of excitatory vs
   inhibitory synaptic currents (E:I balance) and cortical arousal state; steeper (larger exponent)
   spectra correspond to more inhibition / lower arousal, flatter spectra to more excitation / higher
   arousal, and the exponent flattens with healthy aging. Its **offset** indexes broadband power
   (overall neuronal population firing / recording gain).

2. **True narrowband oscillations** — genuine rhythmic peaks that rise *above* the aperiodic
   background. Posteriorly the dominant peak is the **alpha rhythm (~8–13 Hz)**, the signature of the
   idling/relaxed-wakefulness occipito-parietal cortex, strongly modulated by eyes-open vs
   eyes-closed and by visual attention.

The clinical/psychophysiological payoff of separating them is that a change in "alpha power"
measured the classic way can be caused by (a) a genuine change in the oscillation, or (b) a change
in the 1/f background that merely shifts the whole spectrum up/down, or (c) a shift of the peak
**frequency** into or out of a fixed band. Parameterizing the spectrum resolves the ambiguity:

- **`aperiodic_exponent`** — E:I balance / arousal proxy; excellent test–retest; sensitive to
  aging, anesthesia, sleep stage, and neuropsychiatric conditions.
- **`aperiodic_offset`** — broadband power / population activity level.
- **`alpha_cf`** (center frequency) — individual alpha frequency (IAF); a trait-like marker tied to
  cognitive/processing speed and memory; excellent test–retest.
- **`alpha_pw`** (peak power above background) — the *genuine* oscillatory power once the 1/f floor
  is removed, i.e. what "alpha power" was always meant to mean.
- **`alpha_bw`** (peak bandwidth) — the frequency spread / sharpness of the alpha rhythm; the
  noisiest of the five.

This metric therefore delivers, from a single posterior spectrum, the two most reliable trait
markers in resting EEG (aperiodic exponent and IAF) plus a background-corrected alpha-power estimate.

---

## 2. Exact definition and frequency bands

Let `psd_post(f)` be the epoch-averaged, posterior-averaged scalp power spectral density
(units µV²/Hz), formed as:

1. `f, psd = C.mean_psd(data_uV)` → `psd` has shape `(n_channels, n_freq)`, epoch-averaged Welch PSD
   at `sfreq = 256 Hz` (`nperseg = min(1024, n_times)`).
2. Select the rows for the **posterior set present in `ch_names`** — the intersection of
   `{O1, O2, OZ, POZ, PO3, PO4}` with the montage — and average them across channels:
   `psd_post = psd[posterior_rows].mean(axis=0)` → shape `(n_freq,)`.
3. Parameterize: `out = C.aperiodic_and_peak(f, psd_post, fit_range=(2,40), peak_band=(7,14),
   peak_label="alpha")` and return `out` directly (its keys already equal `submetrics`).

**Frequency windows (fixed, do not change without re-baselining):**

| Purpose | Window | Notes |
|---|---|---|
| Aperiodic fit range | **2–40 Hz** | Log-log least-squares fit of `log10(psd) ~ offset + slope·log10(f)`. |
| Peak-exclusion band (removed from the aperiodic fit) | **7–14 Hz** | The alpha bump is masked out so it does **not** bias the 1/f slope. |
| Peak search band | **7–14 Hz** | Center frequency is the argmax of the log residual within this band. |

The peak search band (**7–14 Hz**) is deliberately *wider* than the canonical alpha band
(8–13 Hz) so it captures slowed alpha (e.g. aging, drowsiness → 7–8 Hz) and fast alpha
(→ 13–14 Hz) without clipping the true individual peak.

**Definitional details enforced by `aperiodic_and_peak` (do not re-implement):**

- The exponent is the **negated slope** of the log-log fit (`exponent = -slope`), so a normal
  decaying spectrum yields a **positive** exponent.
- The offset is the fitted **intercept at `log10(f)=0`, i.e. f = 1 Hz** (an extrapolation below the
  2 Hz fit floor, by construction).
- `alpha_pw` and `alpha_cf` are read off the **log10 residual** (`log10(psd)` minus the aperiodic
  fit line), so `alpha_pw` is power *above the 1/f background*, not raw band power.
- `alpha_bw` is the **full width at half maximum (FWHM)** of that residual peak, walked outward from
  the argmax bin until the residual drops below `alpha_pw/2`.
- `alpha_cf` and `alpha_bw` are **quantized to the Welch frequency grid** (bin spacing
  `df = sfreq / nperseg`, e.g. 0.25 Hz at nperseg 1024, 0.5 Hz at nperseg 512). See §7.

---

## 3. Electrodes used and the drop set that tests it

**Electrodes read (posterior / occipito-parietal):** `O1, O2, Oz, POz, PO3, PO4`.
These six carry the dominant posterior alpha rhythm and the cleanest posterior 1/f background;
averaging over them gives a stable midline-symmetric occipito-parietal spectrum and suppresses
per-channel noise before parameterization.

**Drop set (channels removed to test preservation):** `["O1","O2","OZ","POZ","PO3","PO4"]` — i.e.
**exactly the channels the metric reads.**

Rationale for this drop set:

- **It is the hardest possible preservation test for a posterior spectral metric.** Because the
  measurement neighborhood is dropped *in its entirety*, every value the metric consumes in the
  reconstructed frame is an *interpolated/reconstructed* spectrum — no real posterior data survives.
  Preservation therefore means the reconstruction reproduced both the 1/f background **and** the
  alpha peak at sites where nothing real remains, from the surviving (frontal/central/temporal)
  montage alone.
- **It is the correct null for "did we recover the posterior spectrum?"** A metric that only dropped
  half its sites could pass by leaning on the surviving half; dropping all six removes that escape.
- **It matches the IAF drop set exactly**, so `run.py` groups the two metrics by their shared
  `frozenset` drop set and computes each method's reconstruction **once** per drop set (important
  because the ZUNA pass is expensive). It also makes `specparam_peaks` and `iaf` **directly
  comparable** — same reconstructed data, different read-outs.

`drop_channels` must be written in **upper case** (`OZ`, `POZ`, not `Oz`, `POz`); the runner
upper-cases for matching, and channel selection inside `compute` must be case-insensitive
(`C.up` / `C.has` / uppercase index lookup).

---

## 4. Submetric outputs and units

`compute` returns a dict with exactly these five keys (all finite floats, or `nan` when not
computable — see §7):

| Submetric | Meaning | Unit | Typical range (posterior rest) |
|---|---|---|---|
| `aperiodic_exponent` | Steepness of the 1/f background (`-slope` of log10 PSD vs log10 f) | **dimensionless** | ~0.7 – 2.5 |
| `aperiodic_offset` | Log broadband power; fitted intercept at f = 1 Hz | **log10(µV²/Hz)** | order ~0 – 2 |
| `alpha_cf` | Alpha peak center frequency (IAF) | **Hz** | ~8 – 12 (search 7–14) |
| `alpha_pw` | Alpha power **above** the aperiodic background (log10 residual) | **log10 power ratio (dimensionless)** | ~0.1 – 1.0 |
| `alpha_bw` | Alpha peak FWHM bandwidth | **Hz** | ~1 – 4 (grid-quantized) |

Notes:
- Units follow from `C.welch`/`C.mean_psd` using scipy's default `scaling='density'`, so PSD is in
  µV²/Hz (input is microvolts). The offset is a log10 PSD value; `alpha_pw` is a log10 *ratio*
  (residual above the fit) and is therefore unitless.
- `aperiodic_exponent` corresponds to the conventional "1/f slope"; a value of 1.0 is pink noise,
  2.0 is Brownian-like. Values near 0 indicate a (near) flat spectrum (see degenerate case in §7).

---

## 5. Reference frame: scalp average reference (not CSD) and why

**Requirement: compute on the scalp average-referenced PSD that the runner supplies — do NOT apply
`C.csd`.** This is the opposite choice from the FAA reference metric, and the difference is
deliberate:

- **The three most valuable submetrics here are magnitude/shape quantities of the local spectrum**
  (offset = broadband power, exponent = spectral slope, alpha_pw = oscillatory power). CSD is a
  spatial second derivative (surface Laplacian): it re-weights power in a topography- and
  frequency-dependent way and strongly attenuates spatially broad sources. That would make the
  `aperiodic_offset` reflect the Laplacian transform rather than posterior broadband power, and can
  systematically steepen/flatten the exponent depending on spatial–spectral coupling. FAA can afford
  CSD because it takes a **left–right ratio** in which the reference and much of the spatial gain
  cancel; a single-site spectral parameterization has no such cancellation.
- **The benchmark's "truth" and every reconstruction are defined in the surviving-channel
  average-reference frame** (`pilot.surviving_average_reference`). Computing the metric in that same
  frame keeps the read-out and the reconstruction in one coordinate system, so the reported error is
  a clean statement about the reconstructed spectrum, not about a CSD re-projection layered on top.
- Averaging over the six posterior channels already provides spatial stabilization and midline
  symmetry, which is the practical benefit CSD would otherwise offer here.

Average reference is a known caveat for absolute-power interpretation (the common reference is
shared across channels), but it is the frame the harness evaluates in, and it is standard for
posterior IAF/alpha work; the metric is defined and judged consistently within it.

---

## 6. Test–retest reliability and the floor it is judged against

**How the harness judges preservation (`aggregate.py`):** for every `(metric, submetric)` it computes
a **same-day test–retest floor** = the mean over `(subject, day)` of
`|value(Rest1) − value(Rest2)|` on full-montage **truth** rows, and compares it to the mean
`abs_err = |reconstruction − truth|` on **recon** rows. A method **preserves** a submetric when its
reconstruction error is **below that submetric's own floor** — i.e. the reconstruction perturbs the
value *less than simply re-recording the same subject the same day does*. Each of the five submetrics
gets its **own** floor and its **own** pass/fail.

**Expected relative floors (from the literature and this metric's design):**

| Submetric | Expected test–retest | Consequence for its floor |
|---|---|---|
| `aperiodic_exponent` | **Excellent** (among the most reliable resting-EEG measures) | Tight floor → demanding target; a method must reproduce the slope closely. |
| `alpha_cf` | **Excellent** (IAF is trait-like) | Tight floor, but **bounded below by `df`** (grid quantization). |
| `aperiodic_offset` | Good/moderate | Moderate floor. |
| `alpha_pw` | Moderate | Moderate floor. |
| `alpha_bw` | **Noisier** — the least reliable of the five | Wider (more permissive) floor, but also the most variable recon error; interpret cautiously. |

Design implications:
- Because `alpha_cf` and `alpha_bw` are quantized to the frequency grid, their same-day floor cannot
  be smaller than one bin (`df = sfreq/nperseg`). A method that reproduces the peak to within one bin
  is effectively at the reliability floor for `alpha_cf`.
- `alpha_bw` should be reported but treated as the weakest signal; do not over-index on it when
  judging overall preservation.

---

## 7. Edge cases and robustness requirements

The implementation (stage 2) must handle all of the following and always return **finite floats or
`float('nan')`** — never raise, never return non-finite `inf`.

1. **Missing posterior channels.** Average only over the posterior channels **present** in
   `ch_names` (case-insensitive). Guard with `C.has`/`C.up`. If the montage has, say, only `O1` and
   `POz`, average those two rows.
2. **All posterior channels absent.** If **none** of `{O1,O2,OZ,POZ,PO3,PO4}` are present, the metric
   is not computable → return **all five keys set to `nan`** (do not average an empty set, which
   would produce `nan`/warnings implicitly — return the NaN dict explicitly).
3. **Tiny / zero power → log instability.** `C.aperiodic_and_peak` already guards logs with `+1e-30`
   inside the fit and the PSD is non-negative by construction, so a zero/near-zero spectrum will not
   raise. The implementation must **not** subtract, divide, or take logs of the PSD itself before
   handing it to the helper — feed the raw non-negative posterior-mean PSD.
4. **Degenerate / flat spectrum.** If the posterior spectrum is (near) flat, the helper returns
   `exponent ≈ 0` and `offset ≈ mean(log10 psd)`; these are finite and should be passed through, not
   special-cased. If there is no bump above background in 7–14 Hz the helper still returns the argmax
   bin as `alpha_cf` with a small/near-zero `alpha_pw` — acceptable.
5. **No peak / empty peak band.** If the peak band contains no bins or the residual is non-finite
   there, `aperiodic_and_peak` returns `nan` for `alpha_cf/pw/bw` (already handled) — pass through.
6. **NaN propagation from the PSD.** If a posterior row contains non-finite values (defensive; Welch
   should not produce them), the channel mean would poison the fit. Requirement: before calling the
   helper, verify the posterior-mean PSD is finite; if not, drop the offending rows, and if nothing
   finite remains, return the all-`nan` dict.
7. **Bandwidth at the search-band edge.** The FWHM walk can hit the 7–14 Hz array edge, giving a
   truncated `alpha_bw`. This is acceptable (the helper handles it), but note that edge-truncated
   bandwidths are a known contributor to `alpha_bw` noise (§6).
8. **Frequency-grid dependence.** `alpha_cf`/`alpha_bw` resolution equals `df = sfreq/nperseg`.
   `nperseg = min(1024, n_times)`; if epochs are short, `df` grows and these two submetrics coarsen.
   The metric must not assume a fixed `df`; it must read `f` from the helper's inputs (it already
   does, since `f` comes from `C.mean_psd`).
9. **Return-key exactness.** The returned dict keys must be **exactly** the five `submetrics`. The
   helper's output keys already match when `peak_label="alpha"`; do not rename or add keys (the
   runner only writes keys listed in `submetrics`, but extra/renamed keys silently drop data).
10. **No mutation of inputs.** `compute` must not modify `data_uV` or `ch_names` in place.

---

## 8. Literature references

1. **Donoghue, T., Haller, M., Peterson, E. J., Varma, P., Sebastian, P., Gao, R., Noto, T.,
   Lara, A. H., Wallis, J. D., Knight, R. T., Shestyuk, A., & Voytek, B. (2020).** Parameterizing
   neural power spectra into periodic and aperiodic components. *Nature Neuroscience, 23*(12),
   1655–1665. — Defines the aperiodic-plus-peaks decomposition this metric approximates (specparam /
   FOOOF); motivates separating the 1/f exponent and offset from true oscillatory peaks.
2. **Gao, R., Peterson, E. J., & Voytek, B. (2017).** Inferring synaptic excitation/inhibition
   balance from field potentials. *NeuroImage, 158*, 70–78. — Links the aperiodic **exponent** to
   E:I balance, grounding the psychophysiological interpretation of `aperiodic_exponent`.
3. **Klimesch, W. (1999).** EEG alpha and theta oscillations reflect cognitive and memory
   performance: a review and analysis. *Brain Research Reviews, 29*(2–3), 169–195. — Establishes the
   individual alpha frequency (posterior `alpha_cf`/IAF) as a trait-like, functionally meaningful
   marker, and the posterior origin of the dominant alpha rhythm.

*(Supporting reliability context for §6: aperiodic exponent and IAF are consistently reported among
the most stable resting-EEG measures across sessions, while peak bandwidth is the least reliable of
the parameterized outputs.)*

---

## Acceptance checklist (drives stage 2 implementation)

- [ ] Module `m_specparam_peaks.py`, flat imports (`from base import Metric, register`,
      `import common as C`), self-registers at import.
- [ ] `Metric(key='specparam_peaks', name=…, drop_channels=['O1','O2','OZ','POZ','PO3','PO4'],
      submetrics=['aperiodic_exponent','aperiodic_offset','alpha_cf','alpha_pw','alpha_bw'],
      compute=…, reference=…, notes=…)`.
- [ ] `compute(data_uV, ch_names)`: `C.mean_psd` → average present posterior rows →
      `C.aperiodic_and_peak(f, psd_post, fit_range=(2,40), peak_band=(7,14), peak_label='alpha')` →
      return dict unchanged.
- [ ] Scalp average-reference frame (no `C.csd`).
- [ ] All §7 edge cases handled; returns finite floats or explicit `nan`; never raises.
- [ ] Returned keys exactly equal `submetrics`.
