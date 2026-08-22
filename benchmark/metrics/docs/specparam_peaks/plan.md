# Implementation Plan — `specparam_peaks`

> **Superseded 2026-08-21:** this pre-Phase 3 plan implemented a hand-written
> approximation under the specparam name. It is retained as history only. See
> `PHASE3_METRIC_CORRECTNESS.md`; active runs require official pinned
> `specparam==2.0.0rc7` and the v3 result schema.

**Stage 2 of 5 — Implementation plan (docs only; no code yet).**
Target module: `benchmark/metrics/m_specparam_peaks.py`
Handed to the coding step verbatim. Do **not** edit any shared file (`base.py`, `common.py`,
`run.py`, `aggregate.py`).

This plan is exhaustive and unambiguous: the coding step should be able to type the module directly
from §2–§4 and run the self-test in §6 without further decisions.

---

## 1. Module skeleton and imports

Flat imports (the module lives in `benchmark/metrics/` and is imported by name, so imports are
top-level, not package-relative):

```python
from base import Metric, register
import common as C
import numpy as np
```

`numpy` is required for the per-row finiteness guard (§7 edge cases 2/6) and row selection. It is a
hard dependency of `common` already, so importing it adds nothing new.

Module-level constants (define once, above `compute`):

```python
POSTERIOR = ['O1', 'O2', 'OZ', 'POZ', 'PO3', 'PO4']   # UPPER-CASE read set
SUBMETRICS = ['aperiodic_exponent', 'aperiodic_offset', 'alpha_cf', 'alpha_pw', 'alpha_bw']
```

`SUBMETRICS` is the single source of truth: pass the **same list object** to `Metric(submetrics=…)`
and build the all-NaN fallback from it, so the two can never drift.

---

## 2. `compute(data, ch_names)` — exact algorithm

Signature: `def compute(data, ch_names):` — `data` is `(n_epochs, n_channels, n_times)` in µV, already
in the surviving-channel **average-reference** frame supplied by the runner. **Do not call `C.csd`**
(§5 of requirements). Do not mutate `data` or `ch_names`.

Step by step:

1. **Select present posterior rows, case-insensitively.**
   ```python
   u = C.up(ch_names)                                   # upper-cased labels aligned to axis 1
   rows = [u.index(p) for p in POSTERIOR if p in u]     # indices of present posterior channels
   ```
   This is the intersection of `POSTERIOR` with the montage, preserving `POSTERIOR` order (order is
   irrelevant to the later mean). Uses `u.index`, which is exactly the case-insensitive lookup the
   contract mandates.

2. **Edge case — no posterior channel present (§7.2).** If `rows` is empty, the metric is not
   computable:
   ```python
   if not rows:
       return {k: float('nan') for k in SUBMETRICS}
   ```
   Return an **explicit** all-NaN dict (never average an empty selection).

3. **Welch PSD, epoch-averaged, in the average-reference frame.**
   ```python
   f, psd = C.mean_psd(data)          # f:(n_f,)  psd:(n_channels, n_f)
   ```
   `C.mean_psd` uses `sfreq=256`, `nperseg=min(1024, n_times)`, scipy default `scaling='density'`
   → PSD in µV²/Hz, non-negative. Frequency grid spacing `df = sfreq/nperseg` is data-driven; the
   metric reads `f` from the helper and never assumes a fixed `df` (§7.8).

4. **Extract posterior rows and drop any non-finite row (§7.6, defensive).**
   ```python
   post = psd[rows]                              # (k, n_f)
   finite = np.isfinite(post).all(axis=1)        # rows with no NaN/inf
   post = post[finite]
   if post.shape[0] == 0:
       return {k: float('nan') for k in SUBMETRICS}
   ```
   Welch should never emit non-finite values on real data, but a poisoned row must not contaminate
   the channel mean. If **nothing finite** survives, return the all-NaN dict.

5. **Posterior-average spectrum.**
   ```python
   psd_post = post.mean(axis=0)                  # (n_f,)  µV²/Hz
   ```
   Mean over the surviving finite posterior rows → the midline-symmetric occipito-parietal spectrum.
   **Do not** log, subtract, or normalise the PSD before the helper (§7.3) — feed the raw
   non-negative mean.

6. **Parameterize (aperiodic + alpha peak) via the shared helper.**
   ```python
   out = C.aperiodic_and_peak(f, psd_post,
                              fit_range=(2, 40), peak_band=(7, 14), peak_label='alpha')
   return out
   ```
   With `peak_label='alpha'` the helper's keys are exactly
   `aperiodic_exponent, aperiodic_offset, alpha_cf, alpha_pw, alpha_bw` = `SUBMETRICS`. Return the
   dict **unchanged** — do not rename, add, or drop keys (§7.9). The helper already guards logs with
   `+1e-30`, masks 7–14 Hz out of the 1/f fit, and returns NaN for `alpha_cf/pw/bw` when the peak
   band is empty/non-finite (§7.4/§7.5) — all passed through untouched.

**Frequency windows (fixed constants passed to the helper, never recomputed here):**

| Purpose | Window | Where |
|---|---|---|
| Aperiodic log-log fit range | 2–40 Hz | `fit_range=(2,40)` |
| Peak-exclusion band (masked from 1/f fit) | 7–14 Hz | `peak_band=(7,14)` inside helper |
| Peak search band (argmax of log residual) | 7–14 Hz | `peak_band=(7,14)` |

No band-power (`C.bandpower`), no asymmetry (`C.log_asymmetry`), no CSD — this metric uses **only**
`C.up`, `C.mean_psd`, and `C.aperiodic_and_peak`.

---

## 3. `Metric(...)` registration

Build at import time and `register` it (self-registering plug-in):

```python
register(Metric(
    key='specparam_peaks',
    name='Parameterized spectrum (aperiodic + alpha peak)',
    drop_channels=['O1', 'O2', 'OZ', 'POZ', 'PO3', 'PO4'],
    submetrics=SUBMETRICS,
    compute=compute,
    reference='Donoghue et al. 2020 (specparam/FOOOF); Gao, Peterson & Voytek 2017 (E:I balance); '
              'Klimesch 1999 (individual alpha frequency).',
    notes='Posterior aperiodic exponent+offset and alpha peak (cf/pw/bw) from a dependency-free '
          'specparam fit on the average-referenced PSD averaged over O1/O2/Oz/POz/PO3/PO4 '
          '(fit 2-40 Hz, peak 7-14 Hz). No CSD. Shares the IAF drop set.',
))
```

Field notes:
- `drop_channels` is **UPPER-CASE** (`OZ`, `POZ`, not `Oz`, `POz`) per requirements §3 — the runner
  upper-cases for matching (`drop_indices`) and groups by `frozenset` of upper labels, so this drop
  set collides deliberately with the future `iaf` metric and reconstructs **once** per method.
- `submetrics` is the shared `SUBMETRICS` list object (order matters only for the aggregate table;
  the five keys are what the runner writes — see `run.py` line 139, it iterates `m.submetrics`).
- `key='specparam_peaks'` ⇒ file must be `m_specparam_peaks.py` (the runner maps `key → m_<key>.py`
  in `discover`).

---

## 4. Reference frame confirmation

- **Average reference, not CSD.** `compute` consumes `data` directly from `C.mean_psd`; it never
  calls `C.csd`. This is the deliberate opposite of `m_faa.py` (which does `C.mean_psd(C.csd(...))`)
  because offset/exponent/`alpha_pw` are magnitude/shape quantities that CSD would distort, and there
  is no left–right ratio to cancel the transform (requirements §5).
- The runner passes `pilot.surviving_average_reference(...)` frames for both truth and recon, so
  read-out and reconstruction share one coordinate system.

---

## 5. Edge-case handling map (requirements §7 → this module)

| # | Case | Handling in `compute` |
|---|---|---|
| 1 | Some posterior channels missing | `rows` = intersection via `u.index(p) for p in POSTERIOR if p in u`; average only present rows. |
| 2 | **All** posterior channels absent | `if not rows: return {k: nan}` — explicit all-NaN dict, no empty-mean. |
| 3 | Tiny/zero power → log instability | Not our concern; feed **raw** non-negative PSD; helper adds `+1e-30`. No pre-log/subtract/divide. |
| 4 | Flat/degenerate spectrum | Passed through: helper returns `exponent≈0`, finite `offset`, argmax-bin `alpha_cf` with small `alpha_pw`. |
| 5 | No peak / empty peak band | Helper returns NaN for `alpha_cf/pw/bw`; returned unchanged. |
| 6 | Non-finite PSD row (defensive) | `finite = np.isfinite(post).all(axis=1)`; drop bad rows; if none remain → all-NaN dict. |
| 7 | FWHM hits 7–14 Hz edge | Accepted; helper truncates `alpha_bw`. No action. |
| 8 | Frequency-grid dependence | `f` read from `C.mean_psd`; no hard-coded `df`. |
| 9 | Return-key exactness | `peak_label='alpha'` makes helper keys == `SUBMETRICS`; return dict unchanged. |
| 10 | No input mutation | Only reads `data`/`ch_names`; `psd[rows]` and `.mean` create new arrays; NaN dict is freshly built each call. |

Guarantee: `compute` **never raises and never returns `inf`** — every return path yields the five
keys as finite floats or `float('nan')`. (The helper's only non-finite output is `nan`, never `inf`.)

---

## 6. Robustness self-test (run after coding)

**Goal:** invoke the real runner on subject G001 with method `linear` and confirm finite truth **and**
recon rows for all five submetrics.

**Interpreter (per project memory):** `zuna_env/Scripts/python.exe`. Run from the repo root
`<user-home>/Projects/GEEG-ZUNA`.

**Command** (fresh `--out` so the resumable runner does not skip already-logged recordings; `--metrics
specparam_peaks` makes `discover` import **only** `m_specparam_peaks.py`, isolating the run from
half-written sibling metrics):

```
zuna_env/Scripts/python.exe benchmark/metrics/run.py \
  --subjects G001 --methods linear --metrics specparam_peaks \
  --out results/_selftest_specparam.csv
```

- G001 has 10 recordings (`G001Day{1..5}Rest{1,2}.cnt`), all with the truth cache already populated
  in `%TEMP%/tcache`, so no `.cnt` re-preprocessing — the run is fast.
- All six posterior channels are present in the 62-channel montage (verified), so both the truth read
  and the drop-and-reconstruct path exercise real posterior data.

**Expected CSV shape** (columns: `recording,subject,kind,drop_set,method,metric,submetric,truth,value,abs_err`):
- `kind='truth'` rows: 10 recordings × 5 submetrics = **50 rows**, `drop_set='-'`, `method='-'`,
  `abs_err=0`, `value` finite.
- `kind='recon'` rows: 10 recordings × 5 submetrics × 1 method = **50 rows**,
  `drop_set='O1+O2+OZ+PO3+PO4+POZ'` (`'+'.join(sorted(frozenset))`, so alphabetical), `method='linear'`,
  `truth`, `value`, and `abs_err` all finite.

**Pass criteria (the check script must assert all):**
1. Metric `specparam_peaks` appears with exactly the five submetrics
   `aperiodic_exponent, aperiodic_offset, alpha_cf, alpha_pw, alpha_bw`.
2. Every `truth` `value` parses as a finite float (`math.isfinite`), 50 rows total.
3. Every `recon` `value` **and** `abs_err` parse as finite floats, 50 rows total.
4. No traceback / no `[truth:…]` or `[linear:…]` error line printed by the runner.
5. Sanity ranges on truth rows (soft check, warn-only — real posterior rest data):
   `aperiodic_exponent` ∈ ~[0.5, 3], `alpha_cf` ∈ [7, 14] (search band), `alpha_pw` > 0. These are
   informational; hard pass is criteria 1–4.

**Verification snippet** (run after the runner; pure-stdlib `csv`, no pandas needed):

```
zuna_env/Scripts/python.exe - <<'PY'
import csv, math
rows = list(csv.DictReader(open('results/_selftest_specparam.csv')))
sp = [r for r in rows if r['metric'] == 'specparam_peaks']
subs = {'aperiodic_exponent','aperiodic_offset','alpha_cf','alpha_pw','alpha_bw'}
def fin(x):
    try: return math.isfinite(float(x))
    except: return False
truth = [r for r in sp if r['kind']=='truth']
recon = [r for r in sp if r['kind']=='recon' and r['method']=='linear']
assert {r['submetric'] for r in sp} == subs, {r['submetric'] for r in sp}
assert len(truth)==50 and all(fin(r['value']) for r in truth), 'truth rows'
assert len(recon)==50 and all(fin(r['value']) and fin(r['abs_err']) for r in recon), 'recon rows'
print('OK truth=%d recon=%d drop_set=%s' % (len(truth), len(recon), recon[0]['drop_set']))
PY
```

Expected stdout: `OK truth=50 recon=50 drop_set=O1+O2+OZ+PO3+PO4+POZ`.

**Optional** (not required for pass): run
`zuna_env/Scripts/python.exe benchmark/metrics/aggregate.py --csv results/_selftest_specparam.csv`
to eyeball the same-day floor vs linear error per submetric. (Floors need Rest1/Rest2 pairs, which
G001 provides across 5 days.)

**Cleanup:** delete `results/_selftest_specparam.csv` after the check so it is not confused with real
result files.

---

## 7. Acceptance checklist mapping (requirements §Acceptance)

- [x] Flat imports, self-registers at import → §1, §3.
- [x] `Metric(key='specparam_peaks', name=…, drop_channels=[6 UPPER], submetrics=[5], compute, reference, notes)` → §3.
- [x] `C.mean_psd` → average present posterior rows → `C.aperiodic_and_peak(...(2,40),(7,14),'alpha')` → return unchanged → §2.
- [x] Average-reference frame, no `C.csd` → §2 step 6, §4.
- [x] All §7 edge cases handled; finite floats or explicit `nan`; never raises / never `inf` → §5.
- [x] Returned keys exactly equal `submetrics` → §2 step 6 (`peak_label='alpha'`).
- [x] Robustness self-test defined and reproducible → §6.
