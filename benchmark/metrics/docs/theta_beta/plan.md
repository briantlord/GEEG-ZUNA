# Implementation Plan — Theta/Beta Ratio (`theta_beta`)

Stage 2 of 5 (Implementation Plan) for the `m_theta_beta.py` metric plug-in. This is the exact,
unambiguous spec handed to the coding step. It implements the Stage-1 requirements
(`docs/theta_beta/requirements.md`) against the frozen framework contract (`base.py`, `common.py`)
and mirrors the structure of the worked reference `m_faa.py`. **Docs only — no code is written at
this stage; no shared file is ever modified.**

Deliverable file: `benchmark/metrics/m_theta_beta.py`

---

## 1. Module skeleton & imports

Flat import style, exactly as `m_faa.py` (the module is imported from inside `metrics/` with that
dir on `sys.path`):

```python
from base import Metric, register
import common as C
import numpy as np          # needed for np.log / np.isfinite
```

`numpy` is required (FAA did not import it because it delegated to `C.log_asymmetry`; we compute the
log ratio inline here, so we need `np` directly). No other imports. No scipy/mne — all spectral work
goes through `C.*`.

Module docstring: one short paragraph stating the metric is `ln(theta 4–8 / beta 13–30)` bandpower
at Cz (primary) and Fz (secondary), computed on the **scalp average-referenced PSD** (`C.mean_psd`),
**not CSD** — contrast with FAA, and cite that this is the clinically-validated reference frame.

---

## 2. `compute(data, ch_names)` — exact algorithm

Signature and contract (from `base.py`): `compute(data_uV, ch_names) -> dict{submetric: float}`.
`data` is `(n_epochs, n_channels, n_times)` in microvolts, already in the surviving-channel
average-reference frame; `ch_names` are `standard_1005` labels aligned to axis 1.

### Step-by-step

1. **Initialize** `out = {}`.

2. **One PSD for the whole montage, no spatial transform.**
   ```python
   f, psd = C.mean_psd(data)      # (f[n_f], psd[n_ch, n_f]); Welch sfreq=256, nperseg=min(1024,n_t)
   ```
   - Pass `data` **directly** — do **NOT** wrap it in `C.csd(...)`. This is the single deliberate
     divergence from `m_faa.py` (§5 of requirements): TBR is defined on scalp-referenced power; CSD
     is a spatial high-pass that would distort the theta-vs-beta balance and is unstable at the
     midline sites under the drop condition.
   - No re-referencing, filtering, or detrending — the data arrives in the correct frame.

3. **Fixed constants.** `eps = 1e-20` (numerical log floor). Bands are literals in the calls:
   theta `(4, 8)`, beta `(13, 30)`.

4. **Loop over the two output sites**, each guarded independently so a missing channel for one does
   not block the other:
   ```python
   for name, key in (('CZ', 'tbr_cz'), ('FZ', 'tbr_fz')):
       i = C.ix(ch_names, name)               # case-insensitive index, or None if absent
       if i is None:                          # equivalently: not C.has(ch_names, name)
           out[key] = float('nan')            # missing target channel -> explicit NaN (§7.1)
           continue
       theta = C.bandpower(f, psd[i], 4, 8)   # trapezoid over [4, 8),  µV²/Hz, non-negative
       beta  = C.bandpower(f, psd[i], 13, 30) # trapezoid over [13, 30), µV²/Hz, non-negative
       tbr   = float(np.log(theta + eps) - np.log(beta + eps))   # == ln(theta/beta), eps-guarded
       out[key] = tbr if np.isfinite(tbr) else float('nan')      # never emit ±inf (§7.3)
   ```
   - `psd[i]` is the 1-D PSD row for that channel; `C.bandpower` accepts it via its `[..., m]`
     indexing and returns a NumPy scalar. Wrapping the final result in `float(...)` yields a plain
     Python float.
   - **Log ratio is computed as a difference of eps-guarded logs**, never as `np.log(theta/beta)` —
     this avoids `0/0`, `x/0`, and `log(0) = -inf` on a flat/dead channel (§7.2).
   - **Sign is meaningful and must NOT be clamped/rectified.** Because the beta window (17 Hz) is
     far wider than theta (4 Hz), integrated beta often exceeds theta at rest, so `tbr` is commonly
     **negative**. Store it as-is (§2 sign note).

5. **Return** `out`. Keys are exactly `{"tbr_cz", "tbr_fz"}`, matching `submetrics`.

### Channel mapping (fixed)

| Submetric | Channel label (matched case-insensitively via `C.ix`) | Band pair |
|---|---|---|
| `tbr_cz` | `CZ` | theta (4,8) / beta (13,30) |
| `tbr_fz` | `FZ` | theta (4,8) / beta (13,30) |

No other channels are read. (The drop set includes `FCZ`/`CPZ`, but those are never *read* by
`compute`; they exist only to widen the spatial gap for the reconstruction test.)

---

## 3. `Metric(...)` registration

At import time, build and register exactly:

```python
register(Metric(
    key='theta_beta',
    name='Theta/beta ratio',
    drop_channels=['CZ', 'FCZ', 'FZ', 'CPZ'],
    submetrics=['tbr_cz', 'tbr_fz'],
    compute=compute,
    reference='Monastra et al. 2001; Snyder & Hall 2006; Arns et al. 2013',
    notes='ln(theta 4-8 / beta 13-30) bandpower at Cz and Fz on scalp average-referenced PSD (no CSD).',
))
```

Field rationale:
- `key='theta_beta'` — must equal the filename stem after `m_` so the runner's
  `importlib.import_module('m_theta_beta')` resolves it (`run.py:discover`).
- `drop_channels=['CZ','FCZ','FZ','CPZ']` — midline cluster. Cz & Fz are the read sites (forces
  recovery from reconstruction); FCz & CPz are Cz's nearest anterior/posterior midline neighbours,
  dropped so linear/spline interpolation cannot recover Cz trivially. Matched case-insensitively;
  names absent from a montage are silently ignored by `run.py:drop_indices`.
- `submetrics=['tbr_cz','tbr_fz']` — order and spelling must equal the dict keys `compute` returns
  and the aggregator's per-submetric grouping.
- `reference` / `notes` — provenance only; not used in computation.

---

## 4. Edge-case handling checklist (must all hold)

Sourced from requirements §7; each maps to a line above.

1. **Missing target channel** → `C.ix` returns `None` → emit `float('nan')` for that key and
   `continue`; the other submetric is still computed. (`C.has(ch_names, name)` is the equivalent
   guard.)
2. **Tiny / zero band power** → eps-guarded difference-of-logs (`eps=1e-20`) prevents `-inf` and
   division-by-zero.
3. **Non-finite result** → `np.isfinite(tbr)` check coerces any leaked NaN/±inf to `float('nan')`.
4. **Empty band selection** → `C.bandpower` returns `0.0` when the mask is empty (cannot happen at
   256 Hz / ~0.25 Hz resolution for 4–8 & 13–30, but is safe); flows into the eps/NaN handling
   rather than raising.
5. **No re-reference / no CSD / no filtering** inside `compute` — PSD → bandpower → log ratio only.
6. **Case-insensitive channel lookup** via `C.ix` / `C.has` — never assume a case for `ch_names`.
7. **Determinism** — Welch PSD is deterministic; no RNG, no global/module state, no mutation of
   inputs (`C.mean_psd` does not mutate `data`).
8. **Never raise; never return `inf`.** Every path returns a finite float or explicit `float('nan')`.

`compute` must **not** need a try/except: with the `C.ix is None` guard and the eps + `isfinite`
guards, no line can raise on well-formed `(n_ep, n_ch, n_t)` input. (The runner also wraps
`m.compute` in its own try/except, so a raise would merely skip the row — but the requirement is
that `compute` itself is total.)

---

## 5. Robustness self-test (run after coding, before sign-off)

Goal: prove the plug-in registers, computes, and is exercised end-to-end by the **real runner** on
real data, producing **finite** truth and recon rows. The gate is *finiteness and presence of the
expected rows*, not an error-magnitude threshold.

### Interpreter & working directory
- Interpreter: `zuna_env/Scripts/python.exe` (relative to the repository root;
  do not use a bare `python`).
- **Run from the repository root** — `run.py` globs
  `GEEG_Raw/G001Day*.cnt` relative to the current directory. G001 has 10 recordings
  (`Day1..5 × Rest1..2`), all present.

### Command (scoped: one metric, one subject, one method)
Write to a **fresh scratch CSV** (the runner resumes by skipping recordings already present in
`--out`, so a stale file would suppress new rows):

```
zuna_env/Scripts/python.exe benchmark/metrics/run.py \
    --subjects G001 --methods linear --metrics theta_beta \
    --out <SCRATCH>/theta_beta_selftest.csv
```

`<SCRATCH>` = the session scratchpad directory under the system temporary directory.
`--metrics theta_beta` makes `discover` import **only** `m_theta_beta.py`, isolating the test from
any half-written sibling plug-in. `--methods linear` skips the GPU/HF-weights ZUNA path and the
slower spline path.

### Expected CSV shape (columns from `run.py`)
`recording, subject, kind, drop_set, method, metric, submetric, truth, value, abs_err`

Assertions (script the checks; do not eyeball only):
1. **File exists and is non-empty** with a header + data rows.
2. **Truth rows** — for at least one recording, rows with `kind=='truth'`, `metric=='theta_beta'`,
   `submetric ∈ {tbr_cz, tbr_fz}` exist, and their `value` parses to a **finite** float
   (`math.isfinite`). These are the full-montage reliability references (`abs_err==0`).
3. **Recon rows** — rows with `kind=='recon'`, `method=='linear'`, `metric=='theta_beta'`,
   `submetric ∈ {tbr_cz, tbr_fz}` exist, and **both** `truth` and `value` are finite, and `abs_err`
   is finite and `>= 0`. (Both are finite by construction: `surviving_average_reference` keeps the
   dropped-channel rows — re-referenced original data — so the truth-frame `bt` value is real, and
   `reconstruct('linear', …)` fills those rows with a ridge estimate for the `value`.)
4. **Drop-set label** on recon rows equals `CPZ+CZ+FCZ+FZ` (the runner builds it as
   `'+'.join(sorted(frozenset(upper(drop_channels))))`).
5. **No `NaN`/`inf` strings** in the `value`/`truth` columns for G001 (all four midline channels are
   present in this montage, so neither submetric should degenerate). Encountering `nan` here would
   signal a channel-lookup or PSD bug, not an expected missing-channel case.

### Observational sanity (report, do not gate on)
- Sign: `tbr_*` values are frequently **negative** at rest — expected, not a bug (§2).
- Magnitude: per requirements §6, a good reconstruction keeps `abs_err ≲ 0.1–0.2` nats; `linear`
  across the widened midline gap may land near or above that. Record the observed `abs_err` for
  `tbr_cz`/`tbr_fz` as context, but the self-test **passes on finiteness**, not on beating the
  floor (that judgment belongs to the Stage-4 aggregation via `aggregate.py`).

### Optional micro-check (fast, no runner)
Before the full runner pass, a quick import/shape smoke test may be run to fail fast:
- `import m_theta_beta` (from `benchmark/metrics/`) and assert `base.REGISTRY['theta_beta']` exists
  with `submetrics == ['tbr_cz','tbr_fz']` and `drop_channels == ['CZ','FCZ','FZ','CPZ']`.
- Call `compute` on a small synthetic array (e.g. `np.random.randn(2, n_ch, 512).astype(float)` with
  `ch_names` including `CZ`,`FZ`) and assert the returned dict has exactly the two keys, both
  finite floats.
- Call `compute` with a `ch_names` list that omits `FZ` and assert `tbr_fz` is `nan` while `tbr_cz`
  is finite (missing-channel path). This synthetic check is a convenience; the **authoritative**
  self-test is the real-runner pass on G001 above.

---

## 6. Divergences from the `m_faa.py` reference (quick diff for the coder)

| Aspect | `m_faa.py` (reference) | `m_theta_beta.py` (this plug-in) |
|---|---|---|
| Spatial transform | `C.mean_psd(C.csd(data, ch_names))` | `C.mean_psd(data)` — **no CSD** |
| Quantity | `C.log_asymmetry` (R vs L, alpha) | inline `ln(theta+eps) − ln(beta+eps)` per channel |
| Bands | alpha 8–13 | theta 4–8 **and** beta 13–30 |
| Channels | pairs F3/F4, F7/F8 | single sites Cz, Fz |
| Missing-channel policy | key omitted (`if L in u and R in u`) | explicit `float('nan')` per key |
| Extra import | none | `import numpy as np` |

Everything else (flat imports, `register(Metric(...))` at import time, returning a plain dict of
Python floats, no shared-file edits) follows the reference unchanged.
