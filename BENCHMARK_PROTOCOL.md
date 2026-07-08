# GEEG-ZUNA Benchmark Protocol

**Evaluating an EEG foundation model (ZUNA) against classical methods for resting-state channel reconstruction**

| | |
|---|---|
| **Version** | 0.6 (evaluation frame = surviving-channel average reference; linear-K8 is the baseline to beat) |
| **Date** | 2026-06-20 |
| **Status** | Pre-analysis protocol — to be frozen before any benchmark inference is run |
| **Model under test** | ZUNA (Zyphra, arXiv 2602.18478), 380M-param masked diffusion autoencoder |
| **Primary comparator** | MNE spherical spline interpolation (`interpolate_bads`) |
| **Dataset** | GEEG resting-state corpus — up to 323 subjects × 4 days × 2 sessions |

---

## 0. How to use this document

This is a **pre-analysis plan**. Its purpose is to fix every analytic decision — hypotheses, conditions, metrics, statistical models, and the equivalence margins — *before* benchmark inference is run, so that results are confirmatory rather than exploratory. Sections marked **[FREEZE]** contain decisions that should be locked and version-controlled before the first benchmark run; changes after freeze are logged as amendments with dates.

The document is organized as: rationale and hypotheses (§1–2), data and design (§3–5), outcome measures (§6), the analysis plan including the three signature analyses — reliability-grounded equivalence, fingerprinting, and regression-to-the-mean (§7), the statistical and power/sampling plan (§8–9), and the practical compute and pipeline-scaling plan (§10). Deliverables, validity threats, and appendices follow.

**Decision log (v0.3).** Locked with the project lead: (1) input preprocessing **matches ZUNA exactly** — 0.5 Hz highpass, auto-notch, average reference, 256 Hz, **no lowpass** — replacing the wave-1 1–40 Hz band; (2) **input** is broadband (match ZUNA training) with EMG cleaning, but **scoring** is capped at **≤30 Hz primary (δ/θ/α/β) and 30–45 Hz caveated secondary** — HF characterization (§0.2-D) showed 45–128 Hz is EMG/noise-dominated, spatially unreconstructable, and a subject-confound, so high-γ is exploratory-only, not a neural claim; (3) **64 epochs** (≈ full ~6 min recording) per session; (4) the equivalence margin uses the **same-day Rest1↔Rest2** noise floor as primary, across-day as secondary; (5) the comparator ladder is **classical-only for wave 1**, with a peer foundation model deferred to wave 2; (6) production compute runs on the **University of Arizona HPC** (SLURM, Linux) — the full grid runs without trimming, the Windows box is debug/pilot only, and the run is staged (discovery ~160 → replication + expansion to the full 323).

### 0.2 Pilot findings (v0.4) — required amendments

The wave-0 pilot (`benchmark/pilot.py`, validated on real G001 data) surfaced three issues that amend the sections below:

- **A. Average-reference leakage [corrects §4].** After average referencing, Σ(channels) ≈ 0, so a dropped channel is the exact negative sum of the survivors. Linear neighbour regression then recovers it trivially (pilot: r ≈ 0.99, 30 dB) and spline is inflated too. **Average referencing is therefore applied *after* channel dropout, over surviving channels** (the realistic missing-channel scenario) — not in Stage 0.
- **B. Non-cortical channels [amends §3.3, §5, App. A].** Spline reconstructs cortical FP1 at r = +0.995 but mastoid M2 at r = −0.983; pooling them is misleading. Exclude M1/M2 and any non-cortical/reference-like electrodes from the eligible-to-drop and scored sets; report metrics per channel/region, not only pooled.
- **C. Raw-data reality [amends §3–§4].** The `.cnt` files are **1000 Hz, ~11 min, 64 EEG + 3 aux** (marker codes 1–8) — not the doc's 256 Hz / 6 min / 16 epochs; preprocessing resamples 1000→256 and yields 62 channels. **Loader = Neuroscan `read_raw_cnt`, `data_format='int32'`** (confirmed by byte layout and by the project lead; `int16`/`auto` misread it). **Scaling verified OK:** clean marker-locked epochs are physiological in-band (1–40 Hz ≈ 4 µV RMS/ch; α ≈ 1.7, δ ≈ 2.1 µV). The alarmingly large values seen first were raw-*continuous* artifacts (removed by epoching/BAD-rejection) plus a large **near-Nyquist 80–128 Hz component** that dominates broadband variance — almost certainly HF noise/EMG/resample residue, not cortex. **That HF component is the thing to watch for the broadband high-γ scoring decision** (handle via the §4 EMG-control step; verify the resample anti-alias).

- **D. High-frequency content is EMG/noise [revises the broadband-scoring decision].** Empirical characterization (G001–G004): the spectrum follows a clean neural 1/f only to ~30 Hz (posterior alpha peak), then flattens into an EMG/noise shelf (raw 1000 Hz: 0.8 dB drop 30→90 Hz vs 12 dB 10→30). The 45–80 and 80–124 Hz topographies are **frontotemporal** (frontalis/temporalis muscle), not cortical; spatial **effective-rank rises with frequency** (alpha→high-γ), i.e. HF is progressively less reconstructable from neighbours; and EMG load **varies strongly across subjects** (a confound). The 80–128 Hz is genuinely in the recording (not a resample artifact); 60 Hz line noise is present and notched. **Consequence:** feed ZUNA broadband (matching training) with EMG cleaning, but **score ≤30 Hz primary, 30–45 Hz caveated secondary, and treat >45 Hz as exploratory-only (EMG), never a neural claim** — walking back the earlier "score to ~80–100 Hz" choice.

- **E. Evaluation frame = surviving-channel average reference; linear-K8 is the bar [v0.6 decision].** These are single-electrode-reference recordings, so a large common-mode is shared across channels and inflates every method's correlation. Locked decision: score in the **surviving-channel average reference** (mean over *surviving* channels) — it removes the common-mode without leaking the dropped channel (which is not in the average); all methods, including ZUNA, reconstruct and are scored in this frame. Pilot baseline (G001, in-band ≤45 Hz, common-mode-free): **linear-K8 neighbour regression is near-ceiling — RMSE ≈ 3.9 µV, SDR ≈ 22 dB, r ≈ 0.94** — so classical local interpolation already reconstructs missing channels very well, and ZUNA's headroom is small (mainly contiguous / high-N). Notably **spherical spline (the nominal clinical standard) underperforms linear here** (per-channel amplitude errors + posterior sign-inversions; RMSE ≈ 58 µV), so the **meaningful comparator is linear-K8, not spline** — H1 should treat linear-K8 as the baseline to beat. Strategic implication: ZUNA's value, if any, likely lies in **biomarker / individual-structure preservation (fingerprinting, §7.3)**, not bulk waveform fidelity. **CONFIRMED across 5 subjects / 42 recordings (all sessions): linear-K8 r = 0.955 ± 0.053, RMSE ≈ 11.6 µV, SDR ≈ 20 dB, barely degrading even at N=16 contiguous (r = 0.94); consistent per-subject (0.942–0.964); linear beats spline in 96% of condition-cells.** So bulk-fidelity headroom for ZUNA is effectively nil on this dense 62-channel montage, even in the hardest regime tested — the project's value proposition must rest on the biomarker / fingerprint / reliability-equivalence analyses (and possibly N≥32 or whole-region removal), not waveform reconstruction. Results: `benchmark/linear_ceiling_confirmation.csv`. **Large-cluster sweep (G001, 10 sessions, contiguous N = 8→48):** linear-K8 does **not** degrade even when **48 of 62 channels are dropped** (only 14 left) — r stays ≈ 0.95, SDR ≈ 22 dB (RMSE rises only modestly, 5→15 µV). EEG's scalp field is low-rank / heavily oversampled, so local linear interpolation is near-ceiling in *every* dropout regime tested. Channel reconstruction is therefore a poor showcase for ZUNA in any regime; its plausible value is representation / biomarker-plausibility, not waveform recovery. Results: `benchmark/cluster_sweep.csv`.

- **F. Evaluation A — biomarker preservation (5 subjects, 42 sessions, same-day floor) [implemented, `benchmark/biomarker_eval.py`].** Reconstruction error vs each biomarker's **same-day (Rest1↔Rest2) test-retest floor** (protocol §7.2 primary), light-mask of the biomarker's own channels. **IAF, posterior α, and 1/f slope are robustly preserved by every method** (error well below floor) — no opportunity there. **FAA is the exception and the target:** linear-K8 **exceeds** the floor (0.31 vs 0.21 mid-frontal; 0.32 vs 0.30 lateral — i.e. it perturbs FAA *more than a same-day re-recording does*), while **spline stays within** it; spline beats linear on FAA in **4/5 subjects**. So the bulk-fidelity ranking (§0.2-E: linear ≫ spline) **reverses** for FAA — a clean demonstration that fidelity ≠ biomarker preservation. **The sharp ZUNA test:** can ZUNA keep FAA inside its reliability floor (like spline) *while* staying accurate (like linear)? A model that does both is a genuine contribution. Results: `benchmark/biomarker_eval_5subj.csv`.

---

## 1. Background and rationale

ZUNA is a masked **diffusion autoencoder** trained on large-scale EEG. At its core it is a *learned generative prior* over scalp EEG: it tokenizes signal into coarse spatio-temporal blocks, encodes electrode geometry and time with 4D rotary positional encoding, and reconstructs masked channels by diffusion sampling. "Channel reconstruction" is therefore a pretext task that exposes the quality of that prior.

The clinical and research standard for recovering missing/bad channels is **spherical spline interpolation**, a purely geometric method that estimates a missing electrode as a smooth function of its neighbours. The open question this benchmark answers is narrow and falsifiable:

> When one or more channels are missing from resting-state EEG, does ZUNA's learned prior reconstruct them **more faithfully and more *usefully*** than spline interpolation and other classical baselines — and does it do so without **erasing the individual structure** that makes EEG scientifically and clinically informative?

"Usefully" is the operative word. Prior imputation benchmarks report waveform error (RMSE, correlation) in absolute units that have no natural reference. This protocol instead grades reconstruction against **the measurement's own test-retest reliability** and against **preservation of individual identity**, using the dataset's nested repeat structure. That reframing is the methodological contribution.

### 1.1 What a positive result would and would not establish

A win on this benchmark establishes that ZUNA is a better *imputation* operator for resting-state, average-referenced, 62-channel EEG in a healthy cohort. It does **not** by itself establish clinical safety (see the regression-to-the-mean analysis, §7.4), generalization to task/event-related paradigms, or value as a representation-learning backbone — those are separate studies (§12).

---

## 2. Objectives and hypotheses [FREEZE]

### 2.1 Primary objective

Quantify, across subjects and sessions, whether ZUNA outperforms spline interpolation on **temporal fidelity of reconstructed channels**, with **per-subject pairing** so that the comparison reflects population-level rather than channel-selection variance.

- **H1 (superiority, directional):** ZUNA mean per-recording temporal correlation (Pearson *r*) on dropped channels exceeds spline, tested as a method main effect in a linear mixed-effects model. Pre-registered as one-sided at α = 0.025.
- **H1b:** ZUNA RMSE (µV) on dropped channels is lower than spline (same model family).

### 2.2 Secondary objectives

- **H2 (spectral & band power):** ZUNA preserves the power spectrum (spectral *r*) and band-power (δ θ α β, plus caveated low-γ) better than spline, with attention to β where spline over-smooths. (High-γ is excluded as EMG-dominated, §0.2-D; any apparent gamma "win" likely reflects EMG reconstruction, not neural signal.)
- **H3 (reliability-grounded equivalence):** For each clinical biomarker (§6.2), ZUNA-reconstructed values are **statistically equivalent** to ground truth within a margin defined by that biomarker's own test-retest reliability (TOST; §7.2). This is the central practical claim: *reconstruction perturbs the biomarker less than a real re-recording does.*
- **H4 (individuality preservation):** ZUNA retains subject **identifiability** after reconstruction better than spline (fingerprinting; §7.3).
- **H5 (no mean-reversion):** ZUNA does not regress individual biomarker deviations toward the cohort mean more than spline does; the shrinkage slope is ≥ a pre-specified threshold (§7.4).

### 2.3 Stress-test objective (exploratory → confirmatory)

- **H6:** ZUNA's advantage over spline **widens** under spatially contiguous/regional dropout and at high dropout counts (N ≥ 16), where spline loses its local anchors. Exploratory in the first wave; promoted to confirmatory in a held-out replication sample.

### 2.4 Direction of expected effects

| Hypothesis | Metric | Expected winner | Form of test |
|---|---|---|---|
| H1/H1b | temporal *r*, RMSE | ZUNA | superiority (MLM) |
| H2 | spectral *r*, band-power error | ZUNA (esp. β/γ) | superiority (MLM) |
| H3 | IAF, FAA, 1/f slope, connectivity | equivalence to truth | TOST vs reliability margin |
| H4 | identification accuracy / *I*_diff | ZUNA | superiority (paired) |
| H5 | shrinkage slope, variance ratio | ZUNA ≥ threshold | estimation + threshold |
| H6 | all fidelity metrics | ZUNA (widening) | interaction (MLM) |

---

## 3. Dataset [FREEZE]

### 3.1 Structure

The GEEG resting-state corpus contains up to **323 subjects**, each recorded over **4 days** with **2 resting sessions per day** (Rest1, Rest2); some subjects have additional days (e.g., 5–6) usable as extra within-subject repeats. Recordings are 62-channel Neuroscan `.cnt` (10–20 system, `standard_1005` montage), acquired and resampled to 256 Hz.

The design is deliberately **nested**, and this nesting is the analytic engine of the protocol:

```
subject (between-subject variance)
└── day            (between-day / long-term repeat)
    └── session    (Rest1, Rest2 — within-day / short-term repeat)
        └── 5 s epochs (within-session repeats)
```

- **Within-day repeats (Rest1 vs Rest2)** → short-term test-retest noise floor.
- **Between-day repeats (Day_i vs Day_j)** → long-term test-retest noise floor.
- **Between-subject contrast** → identifiability ceiling and population variance.

### 3.2 Training-corpus leakage [RESOLVED]

The GEEG corpus is a **private dataset collected by John Allen's laboratory** and was never publicly released; it therefore **cannot be in ZUNA's training data**. The no-leakage precondition is satisfied, and the benchmark is a clean held-out test of ZUNA's generalization. (Provenance is recorded here as the basis for the claim; the only way this would need revisiting is if a subset were ever shared into a public corpus ZUNA may have trained on.) As a bonus, the Allen lab's published test-retest reliability values for resting EEG and FAA provide an **external sanity check** for the noise-floor estimates computed in §7.2.

### 3.3 Inclusion / exclusion and quality control [FREEZE]

Per-recording QC, applied identically to all methods and decided **before** dropout:

1. Successful load and montage match (62 channels present, positions resolved in `standard_1005`).
2. Minimum retained clean epochs after filtering and artifact rejection (threshold: ≥ 48 of 64 epochs; recordings below are excluded and logged).
3. No channel flat/railed across the whole session (such channels are "naturally bad"). The *eligible-to-drop* and *scored* pools also **exclude non-cortical electrodes** (M1/M2 mastoids and any reference-like channels), which interpolate pathologically (Pilot Finding B, §0.2).
4. Reference and sampling-rate integrity (average reference applied; sfreq = 256 Hz confirmed — guards against the historical `sfreq=250` bug).

QC outcomes (pass/fail + reason) are written to a manifest so the analyzed sample is fully reproducible.

---

## 4. Preprocessing [FREEZE]

All methods receive **identically preprocessed** input; the only difference between conditions is which channels are dropped and which reconstruction operator is applied. Preprocessing matches ZUNA's training distribution:

1. Load `.cnt`; set `standard_1005` montage.
2. **Highpass 0.5 Hz, no lowpass** (matches ZUNA: `hpf_freq = 0.5`, `h_freq = None`). Broadband content is retained to the 128 Hz Nyquist. *(The wave-1 1–40 Hz band is removed: it mismatched ZUNA's training distribution — narrower on both ends — and discarded the gamma range entirely.)*
3. **Auto-notch** line noise and harmonics (ZUNA's detector scans from 45 Hz upward; reproduce via `zuna.preprocessing`).
4. Resample / confirm **256 Hz** (the resample applies an implicit anti-alias at the 128 Hz Nyquist).
5. **Average reference** — applied **after** channel dropout, over the surviving channels (Pilot Finding A, §0.2), *not* here in Stage 0; referencing over all 62 first leaks the dropped channel as the exact negative sum of the rest. (Required because ZUNA trained on average-referenced data.)
6. **High-frequency / EMG control** *(required because scoring now extends to ~45–80 Hz)*: remove muscle components via ICA + ICLabel (or CCA-based EMG removal), applied **identically to ground truth and every reconstruction**; exclude notch-affected line-noise bins from all spectral integration. Without this the high-γ band reflects EMG, not cortex.
7. Edge-artifact crop.
8. Epoch into **64 × 5.0 s** non-overlapping segments (≈ 5.3 min of the ~6 min recording), marker-locked to resting-state event codes (every 0.5 s) to avoid artifact-contaminated periods.
9. Export ground-truth tensors (shape **64 × 62 × 1280**, float32).
10. **ZUNA normalization** at inference only: per-channel z-score, then divide by `data_norm = 10.0` to reach the model's expected std ≈ 0.1, with `data_clip = 1.0` as in ZUNA's config. (Omission of `data_norm` was a corrected 10× scale bug; it is asserted explicitly and unit-tested in the pipeline.)

**Recommended:** run input preprocessing through `zuna.preprocessing` directly, so the benchmark's input distribution is identical to ZUNA's training pipeline by construction rather than by reimplementation.

**Hard inpainting [FREEZE].** After any reconstruction, observed (non-dropped) channels are overwritten with their ground-truth values, so that *only the missing channels are ever scored*. This is applied identically to ZUNA and to every baseline.

---

## 5. Experimental design [FREEZE]

### 5.1 Channel-dropout conditions

Reconstruction quality is probed along three factors crossed with the method ladder:

**(a) Dropout count N** — `N ∈ {2, 4, 8, 16, 32}`. The first wave (doc) covered 2/4/8; 16/32 are explicitly added here to map where the foundation-model advantage saturates or reverses (the N=8 SDR reversal hints at this).

**(b) Spatial pattern** — the key stress-test factor:
- **Scattered** — N channels drawn at random across the cap (spline's easy regime; dense neighbours remain).
- **Contiguous / regional** — N channels forming a spatial cluster (e.g., a whole occipital, frontal, or temporal patch). This removes spline's local anchors and is where a learned long-range prior should win. Regions are defined from the montage by nearest-neighbour adjacency (Appendix A).

**(c) Region identity** (for fixed, interpretable drops) — targeted single-region drops over functionally meaningful areas: occipital (alpha generators), frontal (FAA electrodes F3/F4), central (sensorimotor), temporal (edge electrodes with sparse neighbours).

### 5.2 Replication within recording

For each (recording × N × pattern) cell, draw **8 trials** with distinct, logged random seeds selecting the dropped set (reproducing the existing 8-trial scheme). Seeds are derived deterministically from `(subject, day, session, N, pattern, trial)` so the entire mask set is regenerable. The eligible-to-drop pool excludes QC-failed channels (§3.3) and, for FAA-specific analyses, is constrained to include the relevant electrodes in a dedicated condition.

### 5.3 Counterbalancing and identical masks across methods [FREEZE]

**Every reconstruction method sees exactly the same dropped-channel masks** for a given (recording, N, pattern, trial). This makes the ZUNA-vs-baseline comparison fully **paired** at the finest grain, which is what gives the design its statistical power (§9).

---

## 6. Reconstruction methods and outcome measures

### 6.1 The baseline ladder [FREEZE]

A single comparator (spline) cannot tell you whether ZUNA's gains justify a 380M-parameter diffusion model. The ladder below spans trivial floors to a peer foundation model, so each rung answers a specific question. **The decisive comparator is the linear neighbour model:** if it ties ZUNA, the learned prior is not buying anything.

| Rung | Method | Question it answers |
|---|---|---|
| 0a | Zero-fill (dropped → 0 after avg-ref) | Absolute floor; sanity bound for metrics |
| 0b | Channel-mean / regional-mean fill | Floor that any method must beat |
| 0c | Nearest-neighbour electrode copy | Is geometry alone enough? |
| 1 | **Spherical spline** (MNE `interpolate_bads`) | The clinical/research standard (primary comparator) |
| 2 | Surface Laplacian / Hjorth reconstruction | Alternative geometric estimator |
| 3 | **Linear neighbour regression** (ridge from K nearest channels, fit per recording or cross-validated) | *The value test* — does a cheap linear model match ZUNA? |
| 4 | Gaussian-process / kriging over scalp coordinates | Best classical statistical estimator |
| 5 | ICA-based reconstruction (project out, back-reconstruct) | Source-informed classical method |
| 6 | **ZUNA** (50 diffusion steps, `data_norm=10.0`) | Model under test |
| 7 | *Wave 2 (decide after wave 1):* a second EEG foundation model (e.g., LaBraM / EEGPT / CBraMod) | Apples-to-apples FM comparison |

All rungs receive identical masks and identical hard-inpainting post-processing. **Rung 7 is deferred to wave 2** and run only if wave-1 results warrant it; it is what licenses a claim about "foundation models vs classical" rather than merely "ZUNA vs spline."

### 6.2 Outcome measures

Two tiers: **signal-fidelity** metrics (waveform-level, fast, for H1–H2) and **neural biomarkers** (scientifically meaningful quantities, for H3–H5). All are computed **on dropped channels only**, per epoch then aggregated, with method applied to identical masks.

**Tier 1 — signal fidelity (per dropped channel, per epoch):**

- Temporal correlation (Pearson *r*) — primary.
- RMSE (µV) and relative RMSE (normalized by truth SD).
- Signal-to-Distortion Ratio, SDR (dB).
- Spectral correlation (Pearson *r* on the PSD).
- Band-power error: mean absolute log-ratio across **primary** bands δ (1–4), θ (4–8), α (8–13), β (13–30); **low-γ (30–45) as caveated secondary**. High-γ (45–80) is EMG-dominated (§0.2-D) and excluded from primary/secondary scoring — exploratory appendix only. Line-noise bins (notch frequencies and harmonics) are excluded from every band integral.

**Tier 2 — neural biomarkers (per recording, computed on the full reconstructed montage):**

- **IAF** (individual alpha frequency) — center-of-gravity over posterior channels (O1, O2, Oz, P3, P4, Pz, P7, P8, PO3, PO4, POz) at 0.125 Hz resolution. *(Posterior-COG definition fixes the historical all-channel/argmax bug.)*
- **FAA** (frontal alpha asymmetry) — computed **as Allen's group does** (Smith, Reznik, Stewart & Allen 2017; Coan & Allen 2004): **ln(α power, right) − ln(α power, left)** at **mid-frontal F3/F4** (primary) and **lateral-frontal F7/F8** (secondary), on **current-source-density / surface-Laplacian** transformed data. CSD is *reference-free* — which is why Allen prefers it for asymmetry (FAA is otherwise strongly reference-dependent), and it conveniently makes this biomarker independent of our surviving-channel-reference choice (§0.2-E). α power via Hamming-windowed FFT (~2 s segments, 50% overlap); band **8–13 Hz**, with an **IAF-individualized band (IAF ± 2 Hz)** reported as Allen's recommended refinement. Higher FAA = greater relative *left*-frontal activity (α is inverse to cortical activity).
- **Band powers** (absolute and relative) in δ, θ, α, β (primary) and low-γ 30–45 (caveated secondary); high-γ exploratory only — posterior and global.
- **Aperiodic 1/f slope and offset** (spectral parameterization / FOOOF-style fit over **1–30 Hz**, below the EMG shelf; an exploratory 1–45 Hz fit reported separately) — fitting through the muscle band would bias the slope, so the primary fit stops at 30 Hz (§0.2-D).
- **Connectivity matrices** — PLV, wPLI, and imaginary coherence in α and β bands (all channel pairs).
- **Graph metrics** derived from connectivity — global efficiency, clustering, modularity.
- **Microstate** parameters (optional, wave 2) — mean duration, coverage, transition probabilities.

Each biomarker carries a **clinically/scientifically interpretable unit**, which is what makes the reliability-grounded equivalence test (§7.2) meaningful.

---

## 7. Analysis plan

### 7.1 Primary fidelity comparison

For each Tier-1 metric, fit a **linear mixed-effects model** on per-(recording, N, pattern, trial, channel) observations:

```
metric ~ method * N * pattern + (1 + method | subject) + (1 | subject:day) + (1 | subject:day:session)
```

- **Fixed effects:** `method` (ladder rung), `N`, `pattern`, and their interactions (the `method:pattern` and `method:N` terms test H6).
- **Random effects:** by-subject random intercept and random *slope of method* (subjects may differ in how much they benefit), plus nested day and session intercepts to absorb the repeated-measures structure.
- **Contrasts:** primary is ZUNA − spline (H1); secondary contrasts are ZUNA − {linear regression, GP, ICA}. Report estimate, 95% CI, and standardized effect size.
- Correlations are Fisher *z*-transformed before modelling; SDR and band-power log-ratios are already on appropriate scales.

This replaces the wave-1 "8 random seeds on 1 subject" approach (which estimated channel-selection noise, not population variance) with a model whose uncertainty reflects **between-subject** generalization.

### 7.2 Reliability-grounded equivalence (the central analysis)

**Idea.** A reconstruction is *good enough* for a biomarker if the error it introduces is **smaller than that biomarker's natural test-retest variability**. If reconstruction perturbs IAF less than IAF moves between two real recordings of the same person, the reconstruction is, for that biomarker, statistically interchangeable with a re-recording — the strongest practical claim available.

**Step 1 — Measure the noise floor (truth only).** For each biomarker B and each subject, compute B on two real ground-truth recordings at two levels:
- *short-term:* Rest1 vs Rest2 (same day),
- *long-term:* Day_i vs Day_j.

Across subjects, summarize reliability with:
- **ICC(2,1)** (absolute-agreement intraclass correlation),
- **SEM** = SD_pooled · √(1 − ICC) (standard error of measurement),
- **SDC** = 1.96 · √2 · SEM (smallest detectable change — the change that exceeds measurement noise).

The SDC (in the biomarker's own units) is the **pre-registered equivalence margin** for that biomarker. **[LOCKED: same-day Rest1↔Rest2 is the primary margin — "reconstruction perturbs the biomarker less than re-recording an hour later"; across-day Day_i↔Day_j is reported as a secondary, more lenient bound.]**

**Step 2 — Measure reconstruction error.** Drop channels, reconstruct with each method, recompute B on the reconstruction, and form Δ_recon = |B_truth − B_recon| in the same units.

**Step 3 — Equivalence test (TOST).** For each method × biomarker, test whether the reconstruction-induced difference falls inside ±SDC using **two one-sided tests** (Schuirmann TOST), α = 0.05. A method **passes** for a biomarker if its 90% CI for the mean difference lies entirely within ±SDC. Report per-biomarker pass/fail plus the ratio Δ_recon / SDC (an interpretable "fraction of the noise floor consumed").

**Why it matters.** This converts "ZUNA has lower RMSE" into statements a clinician can act on: *"ZUNA-reconstructed IAF is equivalent to truth within test-retest reliability; spline-reconstructed FAA is not — its error exceeds the SDC and could flip an asymmetry reading."* Per-biomarker, because a method can pass for IAF and fail for connectivity.

### 7.3 Fingerprinting (individuality preservation)

**Idea.** Resting EEG identifies individuals (connectivity/spectral "brainprints"; cf. Finn et al. 2015 for fMRI connectomes, with EEG analogues since). A good reconstruction must preserve what makes a subject identifiable; if subjects become interchangeable after reconstruction, individual structure has been smeared away — regardless of RMSE.

**Procedure (identification accuracy):**

1. **Fingerprint feature [FREEZE]:** vectorized upper-triangle of the α+β connectivity matrix (wPLI primary; PLV as sensitivity), optionally concatenated with per-channel log-PSD. Computed on the full montage after reconstruction.
2. **Target (gallery) set:** each subject's Day1 truth. **Query (probe) set:** each subject's Day2 (and, separately, Rest1↔Rest2 within day).
3. For each probe, rank gallery fingerprints by similarity (Pearson/cosine); predicted identity = top match (nearest neighbour).
4. **Identification accuracy** = fraction of probes matched to the correct subject. Chance = 1/N_gallery (≈ 0.3% at N = 323 — a stringent test that *requires* a large gallery).

**Reconstruction experiment.** Compute the **ceiling** on truth↔truth; then drop N channels in the *probe* recordings, reconstruct per method, recompute fingerprints, and re-identify against the truth gallery.
- **Identifiability retention** = accuracy(reconstructed) / accuracy(truth) — primary, intuitive.
- **Differential identifiability** *I*_diff = mean(within-subject similarity) − mean(between-subject similarity) (Amico & Goñi 2018) — a continuous score that degrades smoothly with dropout and reveals remaining margin. Also test whether reconstruction *raises* between-subject similarity (a direct fingerprint-space signature of mean-reversion, §7.4).

### 7.4 Regression-to-the-mean (mean-reversion / hallucination risk)

**Idea.** A generative model trained on a corpus learns the *population* distribution; its lowest-expected-error fill for a masked channel is near the conditional population average given neighbours. This Bayesian shrinkage pulls each individual's reconstruction toward cohort-typical, compressing real deviations. In a healthy cohort this is lost individuality; clinically it is the dangerous failure — a focal abnormality is *by definition* a deviation from population-typical EEG, exactly what a population prior tends to erase ("hallucinating health").

**Test A — shrinkage slope.** For biomarker B, per subject:
- Δ_true = B_truth(subj) − mean_subjects(B_truth)
- Δ_recon = B_recon(subj) − mean_subjects(B_truth)

Regress Δ_recon on Δ_true across subjects. **Slope = 1 → deviations preserved; slope < 1 → mean-reversion, and the slope is literally the fraction of individual signal retained** (e.g., 0.6 ⇒ 40% erased). Report slope with CI per method; also the variance-compression ratio Var(Δ_recon)/Var(Δ_true). Pre-register a minimum acceptable slope (e.g., ≥ 0.8) for H5. *(Spline smooths too, so this is a comparison, not a ZUNA-only indictment.)*

**Test B — synthetic-abnormality injection (clinical worst case).** Since the cohort is presumably healthy, inject a controlled deviation into a clean recording: focal posterior δ excess, an imposed inter-hemispheric asymmetry, or a localized spectral peak. Drop that channel/region, reconstruct, and measure **recovery ratio** = (abnormality magnitude in reconstruction) / (abnormality injected). A mean-reverting model under-recovers (ratio < 1). Sweep abnormality magnitude and location to map where each method hides pathology. This quantifies clinical risk **without** needing patient data.

### 7.5 Stress / failure atlas

Aggregate §7.1 across the `pattern` and high-N conditions into a "where does each method break" map: metric vs N for scattered vs contiguous, per region. Identify the crossover N at which any ZUNA advantage saturates or reverses (the wave-1 N=8 SDR reversal is the seed observation), and whether contiguous-dropout widens ZUNA's margin as hypothesized (H6).

---

## 8. Statistical analysis plan [FREEZE]

- **Models:** linear mixed-effects (§7.1) for fidelity; TOST (§7.2) for equivalence; nearest-neighbour identification with permutation CIs (§7.3); OLS/robust regression with bootstrap CIs for shrinkage slopes (§7.4). Software: Python (`statsmodels`, `pingouin`) and/or R (`lme4`, `emmeans`, `TOSTER`); both produce auditable model objects.
- **Inference:** primary hypotheses one-sided where directional (α = 0.025), equivalence at α = 0.05 (TOST). Effect sizes and 95% CIs reported for *everything*; CIs, not just p-values, are the headline.
- **Multiple comparisons:** the metric × N × pattern × biomarker grid is large. Control the false discovery rate (Benjamini–Hochberg) **within each pre-declared hypothesis family** (H1, H2, H3, H4, H5 are separate families). The single confirmatory primary (H1, ZUNA−spline temporal *r*) is not penalized.
- **Aggregation order:** epoch → channel → recording → subject, using model random effects rather than naive averaging, to avoid pseudo-replication.
- **Missing data:** QC-excluded recordings are missing-by-design; report the analyzed N and reasons. No imputation of *outcomes*.
- **Reproducibility:** all masks, seeds, model code, and library versions are version-controlled; every reported number traces to a script + a results manifest (§10.5).

---

## 9. Power and sampling plan [FREEZE]

You do **not** need all 323 subjects for the confirmatory tests; you need enough for population-level power, and the surplus is best spent on a held-out replication and a large fingerprinting gallery. All figures below assume **per-subject pairing** (identical masks across methods), which is the design's main power source. Calculations: `statsmodels`, power = 0.80, α = 0.05 unless noted; reproduced in `power_analysis.py`.

**(1) Primary superiority (ZUNA − spline), paired across subjects** — required N by true paired effect size (Cohen's *d_z*):

| *d_z* | 0.15 | 0.20 | 0.25 | 0.30 | 0.40 | 0.50 |
|---|---|---|---|---|---|---|
| **N subjects** | 351 | 199 | 128 | 90 | 52 | 34 |

The wave-1 differences (temporal *r* 0.759 vs 0.674; RMSE 2.69 vs 3.13) are **large** effects; if they hold across subjects (*d_z* ≳ 0.5) even ~35 subjects suffice. A sample of **~100–120** powers detection down to *d_z* ≈ 0.25–0.28, a conservative buffer if the true effect is smaller than wave-1 suggests. Repeated measures (4 days × 2 sessions × trials) further tighten within-subject SE.

**(2) Reliability equivalence (TOST)** — N for 80% power to declare equivalence when the true difference is ~0, by margin (in SD-of-biomarker units):

| margin | 0.20 SD | 0.30 SD | 0.40 SD | 0.50 SD | 0.70 SD |
|---|---|---|---|---|---|
| **N subjects** | 156 | 71 | 41 | 27 | 15 |

Equivalence is the more sample-hungry claim when the margin (SDC/SD) is small. **~120–150** subjects covers margins down to ~0.3 SD; biomarkers with poor reliability (small SDC relative to SD) may need the full cohort — decided per-biomarker once the §7.2 Step-1 reliabilities are measured.

**(3) Shrinkage-slope precision** — 95% CI half-width on the slope vs N (by fit R²):

| | N=50 | N=100 | N=200 | N=323 |
|---|---|---|---|---|
| R²=0.5 | ±0.28 | ±0.20 | ±0.14 | ±0.11 |
| R²=0.7 | ±0.19 | ±0.13 | ±0.09 | ±0.07 |
| R²=0.9 | ±0.09 | ±0.07 | ±0.05 | ±0.04 |

To distinguish a slope of 0.8 from 1.0 you need ≈ ±0.1 resolution → **~150–200** subjects at moderate fit.

**(4) Fingerprinting precision** — 95% CI half-width on identifiability retention (binomial), and chance level, vs gallery size:

| accuracy | N=50 | N=100 | N=200 | N=323 |
|---|---|---|---|---|
| p=0.90 | ±8.3 pp | ±5.9 pp | ±4.2 pp | ±3.3 pp |
| chance (1/N) | 2.0% | 1.0% | 0.5% | 0.31% |

Bigger is strictly better here (lower chance, tighter CI), so the **full cohort is used as the gallery**.

### 9.1 Recommended sampling design [FREEZE]

| Sample | Size | Purpose |
|---|---|---|
| **Discovery / confirmatory** | **~160 subjects**, randomly selected (seed logged), **full grid** (all N, both patterns, 8 trials) | Powers H1–H6 with margin: superiority at *d_z* ≥ 0.22, equivalence at margin ≥ 0.3 SD, slope to ±0.1 |
| **Held-out replication** | a second **~160 subjects**, untouched until discovery analyses are frozen | Replicates H1–H6; guards against overfitting the design |
| **Expansion (wave 2+)** | remaining subjects + pooled re-analysis of all 323 | Final full-cohort confirmation; tightens all estimates |
| **Fingerprinting gallery** | **all 323 subjects** | Maximizes stringency (chance ≈ 0.3%) and CI precision |

Wave 1 runs the discovery sample over the **full grid** — N ∈ {2,4,8,16,32} × {scattered, contiguous} × 8 trials — since HPC removes compute as the binding constraint (§10). Recommended discovery size is **~160 subjects** (powers *d_z* ≥ 0.22 and equivalence margins ≥ ~0.3 SD with room to spare); the held-out replication and expansion to the full 323-subject cohort follow in wave 2. The full cohort is always the fingerprinting gallery. Between-subject variance, not within-recording noise, remains the limiting factor — so subjects are the last thing to cut if the allocation is ever constrained.

---

## 10. Compute and pipeline-scaling plan

The wave-1 pipeline was built for a single recording. Scaling to a ~160-subject full grid changes the problem to "run a fault-tolerant batch of ~10⁵ independent GPU jobs on an HPC cluster." Production runs on the **University of Arizona HPC** (SLURM, Linux) — which the code was originally written for, before being moved to a Windows machine for debugging — with the Windows box retained only for debugging/pilot. This section is the practical plan.

### 10.1 Work volume and a realistic budget

One **unit of work** = one (recording × N × pattern × trial): drop the masked channels, run every ladder method, score, emit metric rows. ZUNA inference dominates cost; baselines are negligible.

Full discovery grid (~160 subjects × 8 recordings × N∈{2,4,8,16,32} × {scattered, contiguous} × 8 trials):

```
160 × 8 × 5 × 2 × 8  =  102,400 ZUNA inference runs
```

At **64 epochs and 50 diffusion steps** budget **~2–4 min per run** (calibrate on the pilot). That is **≈ 3,400–6,800 GPU-hours** of *sequential* work — but every unit is independent, so on HPC it is embarrassingly parallel and wall-clock = GPU-hours ÷ concurrent GPUs:

| Concurrent GPUs | Wall-clock, full discovery grid |
|---|---|
| 25 | ~6–11 days |
| 50 | ~3–6 days |
| 100 | ~1.5–3 days |

So feasibility is no longer the issue — **the full grid runs without trimming**; throughput/allocation is the only lever. If the allocation is capped, trim in cheapest-information-loss order — trials 8→4, then defer N=32, then reduce subjects (between-subject variance is the limiting factor, §9) — otherwise run everything.

### 10.2 Stage separation and caching (the biggest single speedup)

Restructure the 4-stage pipeline so expensive-but-reusable work happens **once per recording**, not once per unit:

- **Stage 0 (once per recording, CPU):** load `.cnt` → filter → avg-ref → epoch → export the ground-truth tensor (64×62×1280, float32, ~20 MB) and precompute truth biomarkers + truth fingerprints. Cache to a content-addressed store keyed by `(subject, day, session, preprocessing-hash)`. ~960 recordings in the discovery sample → trivial CPU cost, computed **once**, reused by every condition and every method.
- **Stage 1–2 (per unit, GPU):** apply mask → ZUNA inference. This is the only step that must run per unit.
- **Stage 3 (per unit, CPU):** z-score reversal, hard inpainting, score Tier-1 metrics and Tier-2 biomarkers on dropped channels; append rows to the metrics store.
- **Stage 4 (once, CPU):** aggregate, fit models, render figures.

Caching Stage 0 removes redundant load/filter/epoch work that the wave-1 script repeats; for a 80-unit-per-recording grid that is an ~80× reduction in preprocessing.

### 10.3 Storage budget

| Item | Per unit | Discovery total | Policy |
|---|---|---|---|
On HPC, raw data is staged once to parallel scratch; per-unit intermediates live on node-local scratch, and only metrics (plus a sampled subset of reconstructions) are written back to project storage.

| Item | Per unit | Discovery total (~160 subj) | Policy |
|---|---|---|---|
| Raw `.cnt` (staged to scratch) | ~170 MB / recording | ~210 GB (discovery) / ~430 GB (full cohort) | stage once; read-only |
| Stage-0 truth tensors (64×62×1280 f32) | ~20 MB / recording | ~24 GB | keep (cached, reused) |
| Reconstruction tensors | ~20 MB / unit | **~1.9 TB if all kept** | **discard after scoring**; retain a logged ~2% sample for figures |
| Metric rows | ~KB / unit | < 1 GB | keep (tidy Parquet) — the real deliverable |

Net durable footprint: staged raw + truth tensors + metrics (~250 GB), well within HPC project storage; the ~1.9 TB of reconstructions is written to node-local scratch and discarded after scoring rather than persisted.

### 10.4 Execution model and fault tolerance (SLURM)

Production runs as a **SLURM job array** over an idempotent unit queue, which is preemption- and failure-tolerant by construction:

1. **Job array = unit queue.** Each array task claims a batch of `pending` units from a manifest (SQLite/Parquet or marker files), runs them, and marks them `done`. Re-submitting the array after any failure reprocesses only `pending`/`failed` units — no redo of completed work.
2. **Preemptible-friendly.** Because units are idempotent and checkpointed, jobs can run on UA's **windfall / preemptible** partition: a preempted task simply leaves its units `pending` for the next array, unlocking large opportunistic throughput at no allocation cost.
3. **One model load per task, many units.** Load ZUNA once per array task and stream a batch of units through it to amortize model-load cost; `fsync` metric rows after each unit.
4. **Node-local scratch.** Do per-unit I/O on node-local SSD; sync only metrics and sampled tensors back to project storage to avoid hammering the parallel filesystem.
5. **Wall-time & resource hygiene.** Size each array task to fit the partition wall-time with margin; one GPU per task; `torch.cuda.empty_cache()` between units.
6. **Containerized environment.** Run inside Apptainer/Singularity (or a pinned module + conda stack) so execution is reproducible across heterogeneous GPU nodes (e.g., P100/V100/A100).

*(The Windows/ReFS instability that forced reboots — ZUNA subprocess GPU crashes corrupting kernel objects — is irrelevant on HPC: Linux, no ReFS. The Windows box is retained only as a debug/pilot environment. Verified UA HPC specs (June 2026): **Puma** 9 GPU nodes × 4 V100S (32 GB) + 8 A100 (40 GB MIG); **Ocelote** 36 × 2 P100 (16 GB, 10-GPU/group cap); **2026 "new cat"** 5 × 8 H200 (141 GB); SLURM, 240 h max wall-time, Rocky Linux 9, Apptainer; free monthly 100k CPU-h on Puma plus unlimited preemptible windfall. ZUNA (380M) fits a V100S or one A100 MIG slice. **[Still to confirm: your PI group/account name and GPU-SU budget.]**)*

### 10.5 Reproducibility and provenance

- **Seeds:** one master seed; per-unit mask seed derived deterministically from `(subject, day, session, N, pattern, trial)`. Store the actual dropped-channel lists, not just seeds.
- **Manifest:** every emitted metric row carries its full provenance key + method + library versions + preprocessing hash, so any number traces back to an exact unit and code state.
- **Environment pinning:** freeze `mne`, `torch==2.5.1`, `numpy`, `scipy` versions (already in `requirements.txt`); record the ZUNA model commit/weights hash and `data_norm`, diffusion-step count, and reference settings in the run config. On HPC, pin the stack in an Apptainer/Singularity image (or a versioned module + conda env) and log the SLURM job/array IDs alongside each unit's provenance.
- **Unit tests for the known bugs:** assertions in CI that `data_norm=10.0` is passed, average reference is set, `sfreq==256`, and the coarse-step count (10 = 1280/128) — the four corrected wave-1 bugs become regression tests.

### 10.6 Phased execution

| Wave | Sample | Grid | Compute | Output |
|---|---|---|---|---|
| **0 — pilot** | 5 subjects | reduced grid | a few GPU-h (Windows box or a few HPC GPUs) | validate ZUNA-matched preprocessing, Stage-0 cache, SLURM job array, metric schema; **calibrate per-run GPU time** |
| **1 — discovery** | **~160 subjects** | **full grid** (N∈{2,4,8,16,32} × 2 patterns × 8 trials) | ~3,400–6,800 GPU-h; days of wall-clock on HPC | confirmatory H1–H6 |
| **2 — replication + expansion** | held-out **~160** + remainder of 323 | full grid + synthetic-injection tests; full-cohort fingerprint gallery | scales with allocation | replicate, expand to full cohort, clinical-risk atlas |

---

## 11. Deliverables and outputs

- **Frozen protocol** (this document, version-controlled) + `power_analysis.py`.
- **Metrics store** — tidy Parquet/CSV: one row per (unit × method × metric), with full provenance.
- **Reliability tables** — ICC / SEM / SDC per biomarker at short- and long-term horizons (the equivalence margins).
- **Primary results** — mixed-effects model objects + forest plots of ZUNA−baseline contrasts per metric, with CIs.
- **Equivalence panel** — per-biomarker TOST outcomes (Δ_recon / SDC) per method.
- **Fingerprinting** — identifiability retention and *I*_diff curves vs N, per method.
- **Mean-reversion** — shrinkage-slope plots and synthetic-injection recovery-ratio maps.
- **Failure atlas** — metric-vs-N surfaces for scattered vs contiguous, per region.
- **Reproducibility bundle** — masks, seeds, configs, environment lockfile, run manifest.

---

## 12. Threats to validity and limitations

- **Training leakage** (§3.2) — **resolved**: the data is a private, unreleased dataset (John Allen lab), so it cannot appear in ZUNA's training corpus.
- **Single paradigm** — resting state only; conclusions do not transfer to task/ERP data, where non-stationary transients dominate and a resting-trained prior may behave differently.
- **Healthy cohort** — clinical risk is assessed only via synthetic injection (§7.4B); real pathological validation is out of scope and is the natural follow-up study.
- **Montage / hardware** — one 62-channel Neuroscan montage; cross-montage and cross-site generalization untested.
- **Spline is not the only "standard"** — the ladder mitigates this, but the optional peer-FM rung (6.1 rung 7) is what licenses claims about *foundation models* generally rather than ZUNA specifically.
- **Metric-prior circularity** — biomarkers computed on reconstructions partly reflect the prior that generated them; the reliability-grounded and fingerprint analyses are designed precisely to expose that, but interpretation should stay alert to it.
- **Diffusion stochasticity** — ZUNA sampling is stochastic; fix/seed the sampler or report across sampling seeds so reconstruction variance is characterized, not hidden.

---

## Appendix A — Channel sets and regions [FREEZE]

- **Posterior / IAF set:** O1, O2, Oz, P3, P4, Pz, P7, P8, PO3, PO4, POz.
- **FAA electrodes:** F3, F4.
- **Regional clusters for contiguous dropout** (defined from `standard_1005` nearest-neighbour adjacency; exact lists fixed at freeze): occipital, parietal, central/sensorimotor, frontal, left-temporal, right-temporal.
- **Edge electrodes** (sparse-neighbour, hardest for spline): peripheral ring of the montage — enumerated at freeze.

## Appendix B — Biomarker definitions (summary)

| Biomarker | Definition | Units |
|---|---|---|
| IAF | center-of-gravity of PSD over posterior set, 8–13 Hz, 0.125 Hz resolution | Hz |
| FAA | ln(α power R) − ln(α power L) at F3/F4 (+F7/F8), on **CSD** (surface-Laplacian, reference-free) data; 8–13 Hz (or IAF±2); Hamming-windowed FFT — per Allen (Smith et al. 2017; Coan & Allen 2004) | log-ratio |
| Band power | integrated PSD in δ/θ/α/β/low-γ/high-γ (line-noise bins excluded); absolute and relative | µV² / fraction |
| 1/f slope | exponent of aperiodic fit over 1–40 Hz (spectral parameterization) | unitless |
| Connectivity | PLV, wPLI, imaginary coherence, α & β, all pairs | 0–1 |
| Graph metrics | global efficiency, clustering, modularity on thresholded connectivity | unitless |

## Appendix C — Glossary

- **ICC** intraclass correlation; **SEM** standard error of measurement; **SDC** smallest detectable change.
- **TOST** two one-sided tests (equivalence testing).
- ***I*_diff** differential identifiability = within- minus between-subject fingerprint similarity.
- ***d_z*** Cohen's effect size for paired/within-subject designs.
- **MLM** (linear) mixed-effects model.

---

*End of protocol v0.1. Freeze all [FREEZE] sections before the first benchmark inference run; record subsequent changes as dated amendments.*



