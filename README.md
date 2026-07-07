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

## Layout
| Path | What it is |
|---|---|
| [`REPORT.md`](REPORT.md) | **Full findings report + questions for Zyphra** (read this first) |
| [`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) | Frozen experimental protocol (design of record) |
| [`pipeline/`](pipeline/) | **Preprocessing Method A** — 1–100 Hz + average reference (proof-of-concept) |
| [`benchmark/`](benchmark/) | **Preprocessing Method B** — 0.5 Hz HPF + bad-aware reference (current 5-subject evaluation) |
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
