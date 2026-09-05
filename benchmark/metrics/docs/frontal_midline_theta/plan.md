# Implementation Plan — Frontal Midline Theta (`frontal_midline_theta`)

**Stage 2 of 5 (Implementation Plan).** Target file: `benchmark/metrics/m_frontal_midline_theta.py`.
This plan is the exact, unambiguous spec handed to the coding step. It changes **no shared file**
(`base.py`, `common.py`, `run.py`, `aggregate.py`) — only a new `m_*.py` plug-in is added.

Derived from `requirements.md` (stage 1) and cross-checked against the framework contract
(`base.py`), the helper library (`common.py`), the worked example (`m_faa.py`), and the runner
(`run.py`). All channels the metric touches — `FZ`, drop set `{FZ,FCZ,F1,F2}`, and the posterior
set `{O1,O2,OZ,PZ,POZ}` — were verified present in the G001 62-channel `standard_1005` montage.

---

## 1. Module skeleton & imports

FLAT import style, exactly as `m_faa.py` (the metrics dir is on `sys.path`; do **not** use package
paths):

```python
from base import Metric, register
import common as C
import numpy as np
```

`numpy` is needed for `np.log`, `np.mean`, `np.isfinite`. The module builds a `Metric(...)` and calls
`register(...)` at import time (self-registering plug-in contract).

---

## 2. `compute(data, ch_names)` — exact algorithm

Signature and contract per `base.py`: `compute(data_uV, ch_names) -> {submetric: float}`.
`data` is `(n_epochs, n_channels, n_times)` in **microvolts**, already in the surviving-channel
**average-reference** frame supplied by the runner. `ch_names` are `standard_1005` labels aligned to
axis 1. Must return finite floats or explicit `float('nan')`; **must never raise**.

Constants: `EPS = 1e-20` (matches the guarded-log convention in `common.log_asymmetry`),
`POST_LABELS = ('O1', 'O2', 'OZ', 'PZ', 'POZ')`, theta band `= (4, 8)`.

**Step-by-step:**

1. **PSD (no CSD).** Compute the epoch-averaged Welch PSD directly on the average-reference scalp
   data — do **not** call `C.csd`. This is the one deliberate divergence from `m_faa.py` (§5 of
   requirements: FMt is an absolute power/level metric, not an asymmetry, so the surface Laplacian is
   inappropriate).
   ```python
   f, psd = C.mean_psd(data)          # f:(n_f,), psd:(n_ch, n_f); sfreq=256 default
   ```

2. **Locate Fz (guard first).** Use `C.ix` (case-insensitive; never compare raw strings).
   ```python
   fz = C.ix(ch_names, 'FZ')
   if fz is None:                      # edge case 1: no numerator -> both submetrics nan
       return {'fmt_fz': float('nan'), 'fmt_rel': float('nan')}
   ```
   (Equivalent guard: `if not C.has(ch_names, 'FZ')`. Using `C.ix` also yields the index in one call.)

3. **`fmt_fz` = guarded log theta power at Fz.**
   ```python
   p_fz   = C.bandpower(f, psd[fz], 4, 8)      # µV², trapezoid over [4,8) Hz, non-negative
   fmt_fz = float(np.log(p_fz + EPS))          # ln(µV²); +EPS avoids -inf on zero power
   ```
   `psd[fz]` is the 1-D PSD of a single channel; `C.bandpower` returns a scalar `np.float64`.

4. **Posterior theta mean (build set from present channels only).**
   ```python
   post_ix = [C.ix(ch_names, n) for n in POST_LABELS]
   post_ix = [i for i in post_ix if i is not None]     # keep only present sites (partial sets OK)
   ```

5. **`fmt_rel` = Fz-vs-posterior theta log-ratio (topographic-specificity index).**
   The denominator is the **arithmetic mean of the posterior band powers** (not a mean of logs),
   then a single guarded log; `fmt_rel` reuses the already-computed `fmt_fz`, so it is exactly
   `ln(P_theta(Fz) + EPS) − ln(mean posterior P_theta + EPS)`.
   ```python
   if post_ix:
       p_post  = float(np.mean([C.bandpower(f, psd[i], 4, 8) for i in post_ix]))
       fmt_rel = float(fmt_fz - np.log(p_post + EPS))
   else:                               # edge case 2: no posterior channels present
       fmt_rel = float('nan')          # fmt_fz still returned normally
   ```

6. **Final finiteness coercion (edge case 4).** Any non-finite result (e.g. `nan`/`inf` propagated
   from bad input PSD) is coerced to explicit `float('nan')`:
   ```python
   out = {'fmt_fz': fmt_fz, 'fmt_rel': fmt_rel}
   return {k: (v if np.isfinite(v) else float('nan')) for k, v in out.items()}
   ```
   (The already-`nan` `fmt_rel` from the no-posterior branch passes through unchanged.)

Returned keys are exactly a subset of `submetrics` (`fmt_fz`, `fmt_rel`), satisfying the contract.

---

## 3. Registration fields

```python
register(Metric(
    key='frontal_midline_theta',
    name='Frontal midline theta',
    drop_channels=['FZ', 'FCZ', 'F1', 'F2'],
    submetrics=['fmt_fz', 'fmt_rel'],
    compute=compute,
    reference='Cavanagh & Frank 2014; Onton, Delorme & Makeig 2005; Gevins et al. 1997',
    notes='ln theta(4-8 Hz) power at Fz (fmt_fz) and Fz-vs-posterior '
          'theta log-ratio (fmt_rel), on average-ref scalp PSD (no CSD).',
))
```

- `key` MUST be `frontal_midline_theta` — the runner maps `key -> m_<key>.py` (`discover()` in
  `run.py`) and the module filename MUST be `m_frontal_midline_theta.py`.
- `submetrics` order `['fmt_fz', 'fmt_rel']` matches the returned dict keys.
- `drop_channels` `['FZ','FCZ','F1','F2']` is the tight frontal-midline neighborhood (§3 of
  requirements). The runner upper-cases and frozenset-groups these; the resulting `drop_set` label in
  the CSV will be the sorted join **`F1+F2+FCZ+FZ`**.

---

## 4. Edge-case handling summary

| # | Condition | Behavior | Mechanism |
|---|---|---|---|
| 1 | `FZ` absent | **both** `fmt_fz` and `fmt_rel` = `nan` | early `return` on `C.ix(...,'FZ') is None` |
| 2 | No posterior channel present | `fmt_rel` = `nan`; `fmt_fz` normal | empty `post_ix` → `nan` branch |
| 2b | Partial posterior set (e.g. only `PZ,POZ`) | valid; average over present sites | list-comp keeps present indices only |
| 3 | Zero / tiny theta power | large finite negative, never `-inf` | `np.log(x + 1e-20)` on every log |
| 4 | `nan`/`inf` in input → PSD | offending submetric coerced to `nan` | final `np.isfinite` gate |
| 5 | Case / channel ordering | robust | all lookups via `C.ix`/`C.has`/`C.up` |
| 6 | Determinism | same input ⇒ same output | no RNG, no global state, no order reliance |
| 7 | Single epoch / short data | handled, no special-case | `C.mean_psd` averages axis 0; `nperseg=min(1024,n_t)` |

`compute` never raises: the only operations are indexing (guarded), `C.bandpower` (pure), and
scalar `np.log`/`np.mean` on non-negative inputs. No try/except is required inside `compute`; the
runner already wraps calls in try/except, but the function is written so that path is never taken on
valid montages.

---

## 5. Robustness self-test (run after coding, before sign-off)

Interpreter: **`zuna_env/Scripts/python.exe`** (relative to the repository root; run all commands
from the repository root so `run.py`'s `GEEG_Raw/G001Day*.cnt` glob resolves). Truth
epochs are cached in `%TEMP%/tcache`, so preprocessing is fast.

### 5.1 Primary: real runner on G001, method `linear`

```
zuna_env/Scripts/python.exe benchmark/metrics/run.py \
  --subjects G001 --methods linear --metrics frontal_midline_theta \
  --out "<SCRATCH>/_selftest_fmt.csv"
```
Use a throwaway absolute `--out` under the session scratchpad (a fresh file so the runner's resume
logic treats no recording as `done` and processes all 10 `G001DayNRestM` recordings). The runner
writes `kind=truth` rows (full-montage floor) and, for the `F1+F2+FCZ+FZ` drop set, `kind=recon`
rows reconstructed with `linear`.

**Pass criteria (verify by parsing the output CSV):**

1. **No error lines.** The runner prints `[truth:...]` / `[linear:...]` only on exception —
   stdout must contain **none** of these for `frontal_midline_theta`; it must print `done <rec>` for
   each recording.
2. **Truth rows finite.** Rows with `metric==frontal_midline_theta` and `kind==truth`: both
   `fmt_fz` and `fmt_rel` appear, and every `value` parses as a **finite** float (expect 10
   recordings × 2 submetrics = 20 truth rows; `fmt_fz` a moderate negative-to-small `ln µV²`,
   `fmt_rel` typically near 0 / mildly signed at rest).
3. **Recon rows finite.** Rows with `kind==recon`, `method==linear`, `drop_set==F1+F2+FCZ+FZ`:
   both submetrics present, and `truth`, `value`, `abs_err` all **finite** (expect 20 recon rows).
   `abs_err == round(abs(value − truth), 6)` and is non-negative.

A tiny checker script (written to scratchpad, run with the same interpreter) that loads the CSV with
`csv.DictReader`, filters `metric=='frontal_midline_theta'`, and asserts the three conditions above
via `math.isfinite` is sufficient. Report counts of finite truth rows and finite recon rows.

### 5.2 Secondary: direct unit smoke on a cached truth (edge cases + determinism)

Import the module directly against one cached truth epoch set and assert the guarded paths, since the
runner alone never exercises the missing-channel branches (the montage is complete):

```python
import numpy as np, sys; sys.path.insert(0, 'benchmark/metrics')
import m_frontal_midline_theta as M
z = np.load(r'<TEMP>/tcache/G001Day1Rest1.cnt.npz', allow_pickle=True)
data = z['data']; ch = [str(c) for c in z['ch_names']]
a = M.compute(data, ch)                      # both finite
b = M.compute(data, ch)                      # determinism: a == b exactly
# drop FZ -> both nan:
nofz = [c for c in ch if c.upper() != 'FZ']; iF = [i for i,c in enumerate(ch) if c.upper()!='FZ']
r1 = M.compute(data[:, iF, :], nofz)         # {'fmt_fz': nan, 'fmt_rel': nan}
# drop all posterior -> fmt_rel nan, fmt_fz finite:
POST = {'O1','O2','OZ','PZ','POZ'}
ip = [i for i,c in enumerate(ch) if c.upper() not in POST]; chp = [ch[i] for i in ip]
r2 = M.compute(data[:, ip, :], chp)          # fmt_fz finite, fmt_rel nan
```

**Assert:** `a` has finite `fmt_fz` and `fmt_rel`; `a == b` (bit-identical, determinism); `r1`'s
both values are `nan`; `r2['fmt_fz']` finite and `r2['fmt_rel']` is `nan`. This confirms edge cases
1, 2, and 6 that the complete-montage runner cannot reach.

Clean up the scratchpad CSV/checker after the self-test; leave `results/` untouched.

---

## 6. Explicit non-goals / guardrails

- **No CSD.** Do not call `C.csd`. FMt is a level metric (requirements §5). This is intentional and
  is the sole behavioral divergence from `m_faa.py`.
- **Do not modify shared files** (`base.py`, `common.py`, `run.py`, `aggregate.py`) or any sibling
  `m_*.py`. The only new artifact is `m_frontal_midline_theta.py`.
- **No new dependencies.** Only `base`, `common`, and `numpy` (already used across the harness).
- **Guarded logs everywhere** (`+1e-20`); **coerce non-finite to `float('nan')`**; **never raise**.
```
