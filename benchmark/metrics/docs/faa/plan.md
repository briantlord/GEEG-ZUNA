# FAA — implementation plan (reference metric)

Module: `benchmark/metrics/m_faa.py`. Implemented and validated (reproduces the original
`biomarker_eval.py` FAA numbers: faa floor 0.208 → linear 0.311 / spline 0.185; faa_lat 0.301 →
0.321 / 0.260).

**compute(data, ch_names):**
1. `csd = C.csd(data, ch_names)` — surface Laplacian (n_ep, n_ch, n_t).
2. `f, pc = C.mean_psd(csd)` — Welch PSD of the CSD, averaged over epochs → `pc[n_ch, n_f]`.
3. For each pair (L, R, key) in [('F3','F4','faa'), ('F7','F8','faa_lat')]: if both present,
   `out[key] = C.log_asymmetry(pc[R], pc[L], f, 8, 13)` = `ln(bandpower_R) − ln(bandpower_L)`.
4. Return `out`.

**Registration:** `Metric(key='faa', name='Frontal alpha asymmetry', drop_channels=['F3','F4','F7','F8'],
submetrics=['faa','faa_lat'], compute=compute, reference='Allen et al. 2004; Smith et al. 2017')`.

**Robustness / validation:** framework run on G001 (linear+spline) reproduces the published FAA floor
and per-method errors exactly — the correctness anchor for every other plug-in.
