# GEEG-ZUNA — Findings Report

**Exploring use cases for ZUNA (Zyphra) on resting-state EEG — waveform reconstruction and,
centrally, the preservation of EEG-derived psychological/clinical measures (frontal alpha
asymmetry).** Prepared to share privately with Zyphra to seek counsel on correct use of the model.

> **Version note.** All results below are for **ZUNA 1.0** (`Zyphra/ZUNA`, arXiv:2602.18478). Zyphra
> released **ZUNA 1.1** (`Zyphra/ZUNA1.1`, arXiv:2607.27308, July 2026) after this evaluation — same
> 380M architecture, 4× more training, larger corpus, variable-length inputs, per-channel quality
> scoring, and two preprocessing variants (0.1–45 Hz band-pass; 0.01 Hz HP + notch). Its headline
> gains are on **waveform NMSE / regional reconstruction** (where it reports beating spline); it does
> **not** evaluate biomarker/spectral-parameter preservation — the axis this report is about. A 1.1
> re-run of the metric battery is **built and prepared for the HPC** (`benchmark/zuna_method_v11.py`
> via `reconstruct_fif`; `benchmark/slurm_zuna11_metrics.sh` + `benchmark/HPC_RUNBOOK_zuna11.md`) — the
> harness is held constant and only the model is swapped. It runs on Linux where 1.1's `torch.compile`
> works; on the local Windows box it falls back to eager (~8× slower) and is impractical at scale.
> §6.4 will get the 1.1 column once the HPC pass completes.

---

## 1. Executive summary

We are **exploring use cases** for ZUNA (the ~382M-parameter masked-diffusion EEG foundation
model, `Zyphra/ZUNA`) — in particular, whether its learned prior adds value for
**psychological/clinical measures derived from EEG**, using missing-channel reconstruction as the
pretext task. We compare it to classical interpolation on a resting-state cohort with a nested
test–retest design (subject → day → Rest1/Rest2). The measure we care about most is **frontal
alpha asymmetry (FAA)**, a widely used affective/clinical index.

Two findings, and one request:

1. **Classical interpolation is a strong baseline for waveform recovery — at least in the regime we
   tested.** On our dense 62-channel montage, scored in-band (≤45 Hz) in a common-mode-free frame,
   **K=8 nearest-neighbour linear interpolation is hard to beat** (r ≈ 0.955, RMSE ≈ 12 µV,
   SDR ≈ 20 dB) and degrades only modestly even when many channels are dropped. ZUNA does not beat
   it here. We read this as: *raw-waveform reconstruction on a dense montage is a poor place to look
   for a learned prior's advantage* — **not** that interpolation is universally sufficient (it
   depends on montage density, dropout pattern, frequency band, and reference frame, none of which
   we have swept exhaustively). It is a workable baseline, not a solved problem.

2. **On the one biomarker where classical methods struggle — frontal alpha asymmetry (FAA) —
   ZUNA helps but does not clear the bar.** Judged against each biomarker's own same-day
   test–retest floor across 5 subjects, ZUNA **misses the primary (mid-frontal F3/F4) FAA floor**
   (error 0.228 vs floor 0.208), though it **beats linear** and **passes the lateral (F7/F8) floor**
   (0.221 < 0.301). It is the **worst of the three methods** on the posterior biomarkers
   (IAF, posterior-α, 1/f slope).

3. **The request:** our two preprocessing pipelines reach *opposite* conclusions (see §4/§6), and
   we are not certain we are feeding ZUNA data in the form it expects. Before we conclude anything
   about the model, we would value Zyphra's guidance on preprocessing, input normalization, and
   whether dense-montage inpainting + biomarker preservation is a use case ZUNA is suited to at all
   (§8).

> **Bottom line:** used the way we currently use it, ZUNA does not do what neither classical method
> manages (preserve FAA within its reliability floor *and* match linear's fidelity). We want to know
> whether that is a property of the model or of how we are driving it.

---

## 2. What we are evaluating, and why this framing

This is a **use-case exploration**: our interest is less "can ZUNA reconstruct a waveform" and more
"does ZUNA help for the EEG-derived psychological/clinical variables people actually report" — FAA
first among them. "Reconstruct a missing channel" has no natural error scale in µV. Following our
protocol
(`BENCHMARK_PROTOCOL.md`), we grade reconstruction two ways that a scientist/clinician can act on:

- **Fidelity** — temporal r, RMSE, SDR, spectral r, band-power error on the dropped channels.
- **Biomarker preservation (the central test, §7.2 of the protocol)** — drop the channels a
  biomarker is computed from, reconstruct, recompute the biomarker, and compare the induced error
  to that biomarker's **natural test–retest variability**. A method "preserves" a biomarker if it
  perturbs it *less than a real re-recording of the same person does*. The **same-day
  (Rest1 vs Rest2) floor** is the primary margin.

Biomarkers: **FAA** (Allen-style, on current-source-density / surface-Laplacian data:
`ln(α F4) − ln(α F3)`, plus lateral `F7/F8`), **IAF** (posterior alpha centre-of-gravity),
posterior α log-power, and aperiodic **1/f slope**.

---

## 3. How we drive ZUNA (please sanity-check this)

For each case we build the `.pt` the dataloader expects and call `zuna.inference(...)`. The exact
recipe (`benchmark/zuna_method.py`, mirroring the reference in `pipeline/load_data.py`):

1. **Mask** dropped channels to 0.
2. **Z-score** the preserved channels with a **single global mean/std** pooled over all non-zero
   samples (not per-channel), storing `zscore_mean/std` in metadata. Dropped channels stay 0. This
   matches the paper's stated convention — *"z-score normalization based on the global mean and
   standard deviation computed across all EEG channels within each recording"* (arXiv:2602.18478) —
   and (as we confirmed) `data_norm=10.0` then divides this unit-variance input to the model's
   expected std ≈ 0.1.
3. **Scale electrode positions** into ZUNA's ±0.12 bounding box (`0.119/max` if `max > 0.119`).
4. Write `data`, `channel_positions`, `metadata` to `ds..._{n_ep:05d}_{n_ch}_{n_time}.pt`.
5. `zuna.inference(data_norm=10.0, diffusion_sample_steps=50, tokens_per_batch=…)`. We cap
   `target_packed_seqlen` (~8000 tokens) so the `flex_attention` mask fits a 12 GB GPU; epochs are
   independent documents so packing is throughput-only.
6. **Map the output back to microvolts by self-calibration:** we fit `truth ≈ a·Z + b` on the
   *observed* channels (whose true µV we know) and apply `a·Z + b` to the dropped channels. We use
   this instead of the stored z-score reversal because we could not verify ZUNA's internal
   normalization offline. It is a clean fit (R² ≈ 0.955; §5), but **we would like to know whether
   we should instead be reading ZUNA's native de-normalized output directly** (`data_norm` reversal),
   and whether self-calibration biases anything.

**Why the normalization is safe for FAA, and where the real dependency lies.** Because the z-score
is *global* (one scale for all channels), it cancels exactly in FAA's log-ratio —
`ln(a²·P₄) − ln(a²·P₃) = ln(P₄) − ln(P₃)` — and the mean-subtraction only affects 0 Hz, outside the
alpha band; both verified numerically. (A *per-channel* z-score would instead inject a bias of
`2·ln(std₃/std₄)` and corrupt FAA; ZUNA and our wrapper both avoid this by using global stats.) So
FAA does not depend on the normalization arithmetic, and alpha power — a Welch-PSD band integral, non-
negative through the linear CSD transform — cannot go negative. The real dependency is **per-channel
amplitude fidelity**: our self-calibration is a single global `(a,b)`, so a global amplitude error
cancels in the ratio but a *channel-specific* one (e.g. systematically under-powered F4) passes
straight through into FAA. That, not z-scoring, is the sensitive point, and it is why the F4/F8
reconstruction weakness (§5) maps onto the FAA miss.

**Reference frame.** In Method B all methods reconstruct and are scored in a **surviving-channel average reference** (mean over surviving channels, computed after dropout). This was a deliberate fix: a
full average reference makes a dropped channel the exact negative sum of the survivors, so linear
regression recovers it trivially (r ≈ 0.99) — an artifact, not skill.

---

## 4. Two preprocessing methods (we are unsure which is right for ZUNA)

We include **both** pipelines because they embody different guesses about ZUNA's training
distribution, and they are **not directly comparable**.

| | **Method A** (`pipeline/`) | **Method B** (`benchmark/`) |
|---|---|---|
| Filter | **1–100 Hz band-pass** | **0.5 Hz high-pass, no low-pass** (broadband) |
| Reference | **Average reference**, before dropout | **Average over survivors only**, after dropout |
| Scope | 1 subject / 1 session, 19→62 ch upsample | 5 subjects / 42 recordings, light-mask biomarker drop |
| Baseline | spherical spline | **K=8 linear** (spline underperforms it here) |
| Conclusion | *ZUNA beats spline* (fidelity) | *linear near-ceiling here; ZUNA misses primary FAA floor* |

The tension is real and central to our request. ZUNA's docs say it was trained on
**average-referenced 256 Hz** data with input normalized to **std ≈ 0.1** (hence `data_norm=10`
after z-scoring). Method A matches "average reference"; Method B matches "broadband, feed the model
what it saw" but uses the surviving-channel average reference for *scoring* fairness. **We do not know which the
model actually prefers, nor whether the low-pass at 100 Hz (A) vs none (B) matters to ZUNA.**

---

## 5. Wrapper validation (the wiring is correct)

Before trusting any number we validated the wrapper on `G001Day1Rest1` (drop F3/F4/F7/F8):

| Check | Result | Meaning |
|---|---|---|
| Observed-channel alignment (raw ZUNA vs truth) | mean r **+0.923**, median +1.000, frac>0 **0.96** | no channel-permutation / flip / scale bug |
| Self-calibration fit (good channels) | **R² = 0.955** (a≈294, b≈0.24) | µV mapping is a clean linear fit |
| Good-channel hard-inpaint | exact | observed channels preserved |

Per-dropped-channel on that recording: F3/F7 reconstruct well (α-band r +0.72 / +0.77), F4/F8
poorly (−0.07 / −0.61) — a genuine per-channel property of the reconstruction, not a bug.

> Note on amplitudes: Method-B preprocessing (no low-pass, no EMG removal) leaves the **truth** at a
> median channel std ~327 µV (frontotemporal muscle + near-Nyquist content dominate broadband
> variance). ZUNA's reconstruction tracks that scale; FAA is computed on **CSD alpha-band power**,
> insulated from the broadband inflation, and truth biomarkers are physiological (FAA −0.16,
> IAF 9.9 Hz).

---

## 6. Results

### 6.1 Method A / phase 1 — single subject (`archive/Project_Overview_phase1.docx`)

With 1–100 Hz + average reference on G001, ZUNA **out-performed spherical spline** on fidelity across
random N-channel dropout (8 trials each):

| Metric | N=2 | N=4 | N=8 |
|---|---|---|---|
| Temporal r — spline / **ZUNA** | 0.674 / **0.759** | 0.653 / **0.732** | 0.685 / **0.733** |
| RMSE µV — spline / **ZUNA** | 3.13 / **2.69** | 4.16 / **3.17** | 4.29 / **4.23** |
| SDR dB — spline / **ZUNA** | 3.40 / **4.04** | 1.79 / **3.27** | **4.29** / 3.42 |

IAF was preserved by both (truth 10.379 → spline 10.369 → ZUNA 10.363 Hz). **Caveat:** single
subject/session, and the comparator is **spline**, not the stronger K=8 linear baseline. This is the
result that made ZUNA look promising — and that the 5-subject work below reframes.

### 6.2 Method B / phase 2 — 5 subjects, 42 recordings

**Waveform fidelity — linear-baseline study** (`results/linear_ceiling_confirmation.csv`,
`results/cluster_sweep.csv`):
linear-K8 r = **0.955 ± 0.053**, RMSE ≈ 11.6 µV, SDR ≈ 20 dB; consistent per-subject
(0.942–0.964); linear beats spline in **96%** of condition-cells; and linear **does not degrade even
at N=48/62 dropped** (r ≈ 0.95). In this regime, then, there is little bulk-waveform headroom for a
learned prior to capture — though we have not swept montage density, band, or dropout pattern
exhaustively, so this is a bounded claim about the conditions tested, not a general one.

**Biomarker preservation vs same-day floor** (`results/zuna_eval_5subj.csv`, via
`benchmark/aggregate.py`; 21 subject-days for the floor, 42 recon rows/method):

| biomarker | same-day floor | linear | spline | **ZUNA** |
|---|---|---|---|---|
| **FAA (F3/F4)** | 0.208 | 0.311 · OVER | **0.185 · ok** | **0.228 · OVER** |
| **FAA-lat (F7/F8)** | 0.301 | 0.321 · OVER | **0.260 · ok** | **0.221 · ok** |
| IAF (Hz) | 0.138 | 0.073 · ok | 0.041 · ok | **0.383 · OVER** |
| posterior-α | 0.459 | 0.313 · ok | 0.145 · ok | **0.468 · OVER** |
| 1/f slope | 0.140 | 0.063 · ok | 0.071 · ok | **0.258 · OVER** |

*ok = error below floor (preserved within test–retest reliability); OVER = exceeds it.*
The linear/spline columns reproduce our pre-ZUNA baseline exactly, which validates the run.

### 6.3 FAA per subject (why the mid-frontal miss is marginal)

The mean-of-days floor mixes subjects with very different intrinsic FAA stability, so the aggregate
FAA verdict is close:

| subject | FAA floor | linear | spline | ZUNA |
|---|---|---|---|---|
| G001 | 0.357 | 0.276 | 0.131 | 0.324 |
| G002 | 0.115 | 0.406 | 0.103 | 0.225 |
| G003 | 0.099 | 0.237 | 0.340 | 0.108 |
| G004 | 0.041 | 0.517 | 0.263 | 0.392 |
| G005 | 0.390 | 0.126 | 0.102 | 0.066 |

Where a subject's FAA is extremely stable day-to-day (G003/G004, floor ≤ 0.10), **even spline fails**
— no reconstruction can match a biomarker that reproducible. On pooled **median** |error|, ZUNA
(0.168) ≈ linear (0.154), with spline best (0.115).

### 6.4 Beyond FAA — a modular metric battery

FAA is one example of a class. We built a **modular metric-testing framework**
(`benchmark/metrics/`) in which each psychophysiological metric is a self-registering plug-in scored
by the same reliability-floor logic, so new metrics need no new script. Four more are implemented,
each with its own 5-part development record (`requirements → plan → code → output → interpretation`
under `benchmark/metrics/docs/<key>/`). All three methods below are the **full 5 subjects / 21 days**
(`results/metric_eval_5subj_zuna.csv`).

| metric / submetric | floor | linear | spline | ZUNA |
|---|---|---|---|---|
| theta/beta ratio — Cz | 0.235 | 0.270 OVER | **0.123 ok** | 1.311 OVER |
| theta/beta ratio — Fz | 0.228 | 0.386 OVER | **0.154 ok** | 1.256 OVER |
| frontal-midline θ — Fz | 0.315 | 1.615 OVER | 0.332 OVER | 0.896 OVER |
| frontal-midline θ — relative | 0.132 | 1.615 OVER | 0.332 OVER | 0.896 OVER |
| mu asymmetry (C3/C4 ratio) | 0.305 | 0.326 OVER | **0.184 ok** | **0.296 ok** |
| mu power — C3 / C4 | 0.267 / 0.277 | 0.65 / 0.78 OVER | **0.19 / 0.18 ok** | 0.57 / 0.58 OVER |
| specparam — aperiodic exponent | 0.164 | **0.153 ok** | 0.183 OVER | 0.583 OVER |
| specparam — aperiodic offset | 0.199 | 0.349 OVER | 0.245 OVER | 0.319 OVER |
| specparam — alpha cf | 0.381 | **0.185 ok** | **0.065 ok** | 1.321 OVER |
| specparam — alpha pw / bw | 0.183 / 0.512 | **ok / ok** | **ok / ok** | 0.302 / 1.006 OVER |
| FAA — F3/F4 · F7/F8 | 0.208 / 0.301 | OVER · OVER | **ok · ok** | 0.228 OVER · **0.221 ok** |

**The verdict (5 subjects, confirmed).** Counting passes against each submetric's floor: **spline
preserves 10 of 14, linear 4, ZUNA just 2** — and ZUNA's two (lateral FAA, mu-asymmetry) are the
**scale-invariant asymmetry ratios**. ZUNA is worse than spline on 13 of 14 submetrics (it edges spline
only on lateral FAA, 0.221 vs 0.260). Everything that depends on *absolute* band power or spectral
*shape* fails, often grossly: theta/beta ≈ **1.3 ln-units** (≈ 5–6× the floor; spline 0.12–0.15), the
posterior **alpha peak frequency off ≈ 1.3** and **bandwidth off ≈ 1.0** (spline 0.07 / 0.20), the
**aperiodic exponent off 0.58**, mu channel power ≈ 0.57 (spline ≈ 0.19), frontal-midline θ 0.90
(spline 0.33). This is the sharpest corroboration of §3: ZUNA's **global** self-calibration preserves
*ratios between homologous channels* (where its per-channel amplitude error cancels) but not *absolute
power or spectral shape at a channel* — which is most of what these clinical metrics measure. Two
secondary notes: **frontal-midline theta is hard for every method** (none reaches its floor — a genuine
open target), and **the aperiodic offset is preserved by none**.

---

## 7. Interpretation

- **Fidelity:** classical local interpolation is a strong, hard-to-beat baseline on this dense
  montage in the regime we tested (in-band, common-mode-free) — so raw-waveform recovery here is a
  poor showcase for a learned prior. ZUNA's potential value is more likely in *derived structure*
  (biomarkers, individuality) than in waveform RMSE. This is a statement about this regime, not a
  claim that interpolation is universally optimal.
- **FAA:** ZUNA clearly **beats linear** on both FAA measures and **passes the lateral floor**, but
  **misses the primary mid-frontal floor** and does not match spline's clean preservation. So it does
  not achieve the hoped "spline-like preservation *and* linear-like fidelity" both-worlds result.
- **Posterior biomarkers:** ZUNA is the worst method — it perturbs IAF / posterior-α / 1/f slope
  more than a re-recording does. It is **not** a general-purpose biomarker-preserving reconstructor
  as currently driven.
- **Preprocessing dependence:** the phase-1 (Method A) "ZUNA wins" and phase-2 (Method B) "ZUNA
  misses" results differ in preprocessing, reference frame, subject count, *and* comparator. We
  cannot cleanly attribute the flip, which is exactly why we want Zyphra's read on §8.

---

## 8. Open questions for Zyphra (the counsel we seek)

Since drafting this we located the paper (Warner, Mago, Huml, Osman, Millidge, *"ZUNA: Flexible EEG
Superresolution with Position-Aware Diffusion Autoencoders,"* arXiv:2602.18478, 9 Feb 2026), which
resolves several preprocessing questions and sharpens others. The items below reflect that.

1. **Reference frame (preprocessing largely resolved).** The paper specifies 0.5 Hz high-pass +
   **common-average reference** + an adaptive 45 Hz–Nyquist notch, resampled to 256 Hz, with **no
   low-pass** — which matches our **Method B** filtering, not Method A's 1–100 Hz band-pass. Our one
   remaining deviation is the reference: we score in an **average over the surviving channels only** (so the
   dropped channel does not leak as the negative sum of the others), whereas ZUNA trained on the full
   common-average. Does reconstructing/scoring in the surviving-channel reference frame put inputs meaningfully outside
   the training distribution, and how would you recommend handling missing channels at inference?
2. **Per-channel amplitude fidelity (the FAA-sensitive point).** Your global-per-recording z-score is
   confirmed, so FAA's log-ratio is scale-invariant and robust to the normalization — good. But that
   means FAA reconstruction rests entirely on **per-channel absolute alpha power**, which our single
   global self-calibration (`a·Z + b`) cannot correct if the model has a channel-specific amplitude
   bias (we see F4/F8 systematically under-reconstructed). Is there a recommended way to recover
   per-channel absolute power from ZUNA's output (the intended de-normalization vs. a global rescale),
   or is channel-wise amplitude simply not something the model is expected to preserve?
3. **High-frequency consistency & anti-aliasing (underspecified in the paper).** With a 0.5 Hz
   high-pass, no low-pass, geometry-only conditioning, and a 208-dataset corpus spanning many sample
   rates and amplifiers, "high-frequency activity" is not a comparably-defined quantity across sources.
   The paper describes *up*sampling low-rate recordings to 256 Hz but not how recordings acquired
   **above** 256 Hz (e.g. TUH, high-density systems) were **decimated / anti-alias filtered**. (a) How
   was that anti-aliasing done? (b) Given no acquisition-metadata conditioning, how consistent is the
   learned representation above ~40 Hz, and can broadband HF input bleed into lower bands during
   diffusion sampling? (c) Should we band-limit inputs to the range where the prior is well-defined?
4. **Geometry.** We scale electrode positions into a ±0.12 box (`0.119/max` if `max > 0.119`). Are these
   the correct units/coordinate frame, and does non-uniform scaling of a real 10–20 montage distort the
   4D-RoPE geometry?
5. **Masking convention.** We zero dropped channels and rely on the channel-position set to signal which
   are present. Is zeroing the intended mask, or should missing channels be *omitted* rather than zeroed?
6. **Sampling budget.** We use 50 diffusion steps and `diffusion_cfg=1.0`. Would more steps / CFG > 1
   improve biomarker fidelity, particularly for the right-frontal channels (F4/F8) we reconstruct worst?
7. **Intended use.** The paper frames ZUNA as EEG **super-resolution** (sparse → dense). Is
   dense-montage single-channel inpainting (a few channels missing from 62) a regime it is designed for,
   or is its value in sparse-montage upsampling / representation extraction? And is "preserve a
   CSD-derived biomarker within same-day test–retest reliability" a reasonable success criterion?

---

## 9. Reproducibility

- **Environment:** Python 3.10, torch 2.6.0+cu124, MNE ≥ 1.6, on an RTX 3080 Ti (12 GB). The 382M
  model fits in 12 GB at ~3 GB peak with our token cap. `requirements.txt` lists the Python deps;
  ZUNA itself (code + `Zyphra/ZUNA` weights) is an external dependency (§requirements).
- **Headline commands:**
  ```bash
  # Method B, 5-subject biomarker evaluation (GPU for 'zuna'; ~6.5 min/inference)
  python benchmark/biomarker_eval.py --subjects G001 G002 G003 G004 G005 \
         --methods linear spline zuna --out results/zuna_eval_5subj.csv
  python benchmark/aggregate.py --csv results/zuna_eval_5subj.csv
  # Wrapper validation / diagnostics
  python benchmark/_validate_zuna.py
  python benchmark/_diag_zuna.py
  ```
  `biomarker_eval.py` is resumable by recording. Each ZUNA inference runs as an isolated subprocess,
  so GPU memory is released between calls (no accumulation across the ~84 inferences).
- **Data availability:** raw `.cnt` recordings are human EEG and are **not** included in this repo
  (see `.gitignore`); they can be shared separately under agreement. The GEEG resting-state corpus is
  a nested design (subject → 4+ days → Rest1/Rest2); this evaluation used subjects G001–G005.
- **Results included:** `results/zuna_eval_5subj.csv` (+ `zuna_eval_G001.csv`) — Evaluation A rows
  (truth + linear/spline/zuna recon, per biomarker); `results/linear_ceiling_confirmation.csv` and
  `results/cluster_sweep.csv` — the linear-baseline fidelity studies; `results/figures/` — the PSD/alpha and
  dropout-sweep summary plots.

---

## Appendix — repository map

```
README.md                 how to read/run this repo
REPORT.md                 this document
BENCHMARK_PROTOCOL.md     the frozen experimental protocol (design of record)
requirements.txt
pipeline/                 Method A — 1–100 Hz + average reference (proof-of-concept)
benchmark/                Method B — 0.5 Hz HPF + surviving-channel average reference (current evaluation)
results/                  result CSVs + key figures (no raw data / weights)
archive/                  superseded single-subject scripts + phase-1 write-up
```
