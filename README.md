# GEEG-ZUNA

**Exploring use cases for ZUNA** (Zyphra's masked-diffusion EEG foundation model, `Zyphra/ZUNA`) on
resting-state EEG. We use missing-channel reconstruction as a pretext task and compare against
classical interpolation, but our real interest is whether the model helps for the
**psychological/clinical measures derived from EEG** — chiefly **frontal alpha asymmetry (FAA)** —
judged against those measures' natural test–retest reliability.

> This repository is shared privately with **Zyphra** to seek counsel on correct use of the model.
> **Start with [`REPORT.md`](REPORT.md)** — it summarizes the findings and lists the specific
> questions we have for you (see §8). The open questions center on preprocessing, input
> normalization, and whether this use case suits ZUNA.
>
> **Version:** results here are for **ZUNA 1.0** (`Zyphra/ZUNA`). **ZUNA 1.1** (`Zyphra/ZUNA1.1`,
> arXiv:2607.27308) was released after this work; a 1.1 re-run of the metric battery is **built and
> prepared for the HPC** (`benchmark/zuna_method_v11.py`, `slurm_zuna11_metrics.sh`,
> `HPC_RUNBOOK_zuna11.md`) — it runs on Linux where 1.1's `torch.compile` works. See the version note
> atop [`REPORT.md`](REPORT.md).

## TL;DR of findings
- For raw **waveform fidelity** on our dense 62-channel montage, classical **K=8 linear
  interpolation is a very strong baseline** (r ≈ 0.955, and hard to beat even at high dropout in the
  regime we tested); ZUNA does not beat it there — so waveform recovery looks like a poor place to
  find the model's advantage (a bounded claim about this regime, not that interpolation is universally
  sufficient).
- On **frontal alpha asymmetry** — the one biomarker classical methods struggle with — ZUNA **beats
  linear** and **passes the lateral (F7/F8) test–retest floor**, but **misses the primary
  (mid-frontal F3/F4) floor**, and is worst-of-three on posterior biomarkers.
- Our two preprocessing pipelines reach opposite conclusions; we are unsure which matches ZUNA's
  training distribution. Details and the ask in [`REPORT.md`](REPORT.md).

## What we are really testing (the vision)

The test is **not** "does ZUNA reconstruct a waveform with low error." Classical interpolation already
does that on a dense montage, and RMSE / temporal-correlation / SDR are **signal-processing
abstractions with no inherent meaning** — a reconstruction can score well on them while smearing away
the asymmetries, peak frequencies, and spatial structure that make EEG a window into affect, arousal,
cognition, and pathology.

What we care about is whether ZUNA, when it **"scales up" or repairs a recording, preserves — or even
sharpens — the psychophysiologically/biologically meaningful quantities** that researchers and
clinicians actually interpret. **Frontal alpha asymmetry (FAA)** is one sterling, easy-to-test example
(a single, well-validated, spatially specific number with published test–retest reliability, trivially
broken by dropping F3/F4/F7/F8). It is one member of a large family. For **every** candidate metric we
ask two questions:

- **Preservation** — drop the channels the metric is computed from, reconstruct, and check the value
  stays inside its own **test–retest reliability floor** (the FAA protocol).
- **Super-resolution / enhancement** — reconstruct a *sparse* montage up to dense and check whether the
  metric lands *closer to the true dense-montage value* than geometry-only interpolation does. This is
  the more ambitious "increase the resolution of a meaningful measure" claim.

### Candidate metrics (catalog)

Grouped by the spatial structure each one stresses. ✅ = implemented in the modular framework.

| Group | Metric | Indexes | Channels it needs |
|---|---|---|---|
| Hemispheric asymmetry | **Frontal alpha asymmetry** ✅ | approach/withdrawal affect, depression risk | F3/F4, F7/F8 |
| | Sensorimotor **mu asymmetry** ✅ | motor/somatosensory lateralization | C3/C4 |
| | Posterior alpha asymmetry | spatial-attention bias | P3/P4, O1/O2 |
| Spectral landmarks | Individual alpha frequency (IAF) | cognitive speed, aging, memory | posterior |
| | **specparam peak parameters** ✅ (α center-freq / power / bandwidth, aperiodic exponent+offset) | true-oscillation vs 1/f background; E:I balance, arousal | posterior |
| | θ–α transition frequency | memory/attention | frontal+posterior |
| Regional power & ratios | **Theta/beta ratio** ✅ | attention/arousal; classic ADHD index | Cz/Fz |
| | **Frontal midline theta** ✅ | cognitive control, working-memory load | Fz/FCz |
| | Delta/alpha ratio (DAR) | clinical slowing (stroke, encephalopathy) | whole-scalp |
| Connectivity / network | α/β phase connectivity (wPLI, PLV); graph metrics; **brainprint** identifiability | functional coupling, individual identity | multi-channel |
| Whole-scalp / topographic | EEG **microstates** (A–D duration/coverage/transitions); source-space ROI power | large-scale dynamics, localization | full montage |

Out of scope for this **resting-state** corpus (named so the boundary is explicit): ERP components
(P300, MMN, ERN — need task data) and sleep spindles / slow oscillations (need sleep data).

### Modular metric-testing framework

Each metric is a **self-registering plug-in** (`benchmark/metrics/m_<key>.py`) implementing one common
contract — a name, the channels to drop, and a `compute(data, ch_names) → {submetric: value}` function.
A single generalized runner (`benchmark/metrics/run.py`) and aggregator (`aggregate.py`) then evaluate
**every** registered metric against the same reliability-floor logic, grouping metrics that share a
drop set so each reconstruction (including the expensive ZUNA pass) runs once. **Adding a new metric is
a new module, not a new script.** Each metric ships a 5-part development record under
`benchmark/metrics/docs/<key>/`: `requirements.md` → `plan.md` → code → `output` → `interpretation.md`.

## Layout
| Path | What it is |
|---|---|
| [`REPORT.md`](REPORT.md) | **Full findings report + questions for Zyphra** (read this first) |
| [`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) | Frozen experimental protocol (design of record) |
| [`pipeline/`](pipeline/) | **Preprocessing Method A** — 1–100 Hz + average reference (proof-of-concept) |
| [`benchmark/`](benchmark/) | **Preprocessing Method B** — 0.5 Hz HPF + surviving-channel average reference (current 5-subject evaluation) |
| [`results/`](results/) | Result CSVs + key figures (no raw data or model weights) |
| [`archive/`](archive/) | Superseded single-subject exploration scripts + the phase-1 write-up |

Both preprocessing methods are included deliberately — see [`REPORT.md`](REPORT.md) §4.

## Setup
```bash
pip install -r requirements.txt          # Python 3.10; GPU run used torch 2.6.0+cu124
# ZUNA is a separate dependency: the `zuna` package + `Zyphra/ZUNA` weights (auto-download from HF).
# Put the zuna package on PYTHONPATH; optionally set HF_HOME to a local weights snapshot.
```

## Reproduce the headline result
```bash
python benchmark/biomarker_eval.py --subjects G001 G002 G003 G004 G005 \
       --methods linear spline zuna --out results/zuna_eval_5subj.csv
python benchmark/aggregate.py --csv results/zuna_eval_5subj.csv
```
The `zuna` method needs a GPU + weights; `linear`/`spline` run on CPU. `biomarker_eval.py` is
resumable by recording.

## Data
Raw `.cnt` recordings are human EEG and are **not** included (see `.gitignore`); they can be shared
separately under agreement. The evaluation used subjects **G001–G005** from the GEEG resting-state
corpus (nested subject → day → Rest1/Rest2 design).

## A note on `zuna/`
The vendored ZUNA package from the working tree is intentionally **omitted** here — it is Zyphra's
own code. The benchmark imports `from zuna import inference`; the model wiring we use (input `.pt`
construction, `data_norm=10`, self-calibration) is documented in [`REPORT.md`](REPORT.md) §3 so it
can be reviewed without it.
