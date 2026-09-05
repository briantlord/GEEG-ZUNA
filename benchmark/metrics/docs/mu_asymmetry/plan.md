# Implementation Plan — Sensorimotor Mu Asymmetry (`mu_asymmetry`)

Stage 2 of 5 (IMPLEMENTATION PLAN). Target file: `benchmark/metrics/m_mu_asymmetry.py`.
This plan is handed verbatim to the coding step; it fully specifies the module. No shared framework
file (`base.py`, `common.py`, `run.py`, `aggregate.py`, `pilot.py`) is touched.

The metric is the **central-row spatial analog of the reference `m_faa.py`**: identical asymmetry
machinery and reference frame (CSD → mean PSD → `log_asymmetry`), relocated from mid-frontal F3/F4 to
sensorimotor C3/C4, plus the two single-channel log-power submetrics. When in doubt, mirror
`m_faa.py` exactly.

---

## 1. Module skeleton (exact structure to produce)

```
"""Mu asymmetry — Sensorimotor Mu (Rolandic) Asymmetry (metric plug-in).

ln(mu power, right C4) - ln(mu power, left C3) at central sensorimotor sites, plus the two
single-channel log band powers, computed on current-source-density (surface-Laplacian) data,
8-13 Hz. Spatial analog of FAA (m_faa.py) relocated F3/F4 -> C3/C4.
"""
from base import Metric, register
import common as C


def compute(data, ch_names):
    ...
    return out


register(Metric(
    key='mu_asymmetry',
    name='Sensorimotor mu asymmetry',
    drop_channels=['C3', 'C4', 'C1', 'C2'],
    submetrics=['mu_asym', 'mu_c3', 'mu_c4'],
    compute=compute,
    reference='Pfurtscheller & Lopes da Silva 1999; Pineda 2005; Allen et al. 2004 (CSD variant)',
    notes='ln(mu C4)-ln(mu C3) plus ln bandpower C3/C4 on CSD, 8-13 Hz; spatial analog of FAA.',
))
```

- Import style is **FLAT** (`from base import ...`, `import common as C`) — never package-qualified.
  The runner inserts the metrics dir on `sys.path` and imports `m_mu_asymmetry` by bare module name.
- Registration happens **at import time** as a side effect of `register(...)`. The module body has no
  other top-level statements.
- `key` must be exactly `mu_asymmetry` so the runner's convention `key -> m_<key>.py` resolves this
  file, and so `--metrics mu_asymmetry` imports it.

---

## 2. `compute(data, ch_names)` — exact algorithm, step by step

Signature and contract (from `base.py`): `data` is `np.ndarray (n_epochs, n_channels, n_times)` in
**microvolts**, already in the surviving-channel average-reference frame; `ch_names` are
`standard_1005` labels aligned to axis 1. Return `dict{submetric: float}` with finite floats; keys
must be a subset of `submetrics`. Pure and deterministic — no randomness, no global state, no I/O.

**Constants used:** mu band `lo=8, hi=13` (numerically identical to alpha; half-open `[8, 13)` as
`C.bandpower` integrates). Epsilon floor `1e-20` for the two single-channel logs — the **same**
constant `C.log_asymmetry` applies internally, so all three submetrics share one convention (§4/§7 of
requirements).

Steps:

1. `out = {}` — start empty. A partial dict is valid; the runner skips any `submetric` key it does
   not receive (`run.py` guards `if sk in bt.get(m.key, {}) and sk in br`).

2. **Presence gate before any heavy compute.** If neither C3 nor C4 is present there is nothing to
   emit — return early to avoid running CSD on a montage with no target channel:
   ```
   if not (C.has(ch_names, 'C3') or C.has(ch_names, 'C4')):
       return out
   ```
   (Uses `C.has`, which is case-insensitive.)

3. **Reference frame — CSD then mean PSD**, exactly the `m_faa.py` line:
   ```
   f, pc = C.mean_psd(C.csd(data, ch_names))     # PSD of CSD, mean over epochs -> pc[n_ch, n_f]
   ```
   - `C.csd` builds a `standard_1005` montage, converts µV→V, computes the surface Laplacian, and
     returns channels in the **same order** as `ch_names` (index alignment preserved;
     `on_missing='ignore'`). `C.mean_psd` returns `f` (freq vector) and `pc` (per-channel PSD averaged
     over epochs).
   - Do **not** call `C.welch`/`C.bandpower` directly on the raw potential and do **not** average-
     reference — the reference frame is CSD by requirement (§5). Reuse the shared helpers; do not
     re-implement Welch/CSD/band integration.

4. **Channel indexing** via the upper-cased name list, mirroring `m_faa.py`:
   ```
   u = C.up(ch_names)
   ```
   Then index with `u.index('C3')` / `u.index('C4')` **only after** confirming presence (step 5),
   never blindly (avoids `ValueError`). `pc[u.index('C3')]` is the C3 PSD row; `pc[u.index('C4')]` the
   C4 row.

5. **Submetric `mu_asym`** — right−left log asymmetry (headline construct). Requires **both** C3 and
   C4:
   ```
   if C.has(ch_names, 'C3', 'C4'):
       v = C.log_asymmetry(pc[u.index('C4')], pc[u.index('C3')], f, 8, 13)
       if math.isfinite(v):
           out['mu_asym'] = float(v)
       else:
           out['mu_asym'] = float('nan')
   ```
   - Argument order is `(psd_right, psd_left, f, lo, hi)` = `(C4, C3, f, 8, 13)` → returns
     `ln(bp C4) − ln(bp C3)`. This matches FAA's `ln(R)−ln(L)` sign convention exactly (positive ⇒
     greater right-central mu ⇒ *less* right sensorimotor activation; report with the inverse-power
     caveat downstream).
   - `C.log_asymmetry` already returns a Python `float` and already adds the `1e-20` floor to each
     band power before the log, so a silent channel yields a large finite value, not `-inf`.
   - Scale-invariant: any global multiplicative scaling and the CSD normalization constant cancel in
     this difference of logs.

6. **Submetric `mu_c3`** — left-hemisphere absolute log band power. Requires C3 only:
   ```
   if C.has(ch_names, 'C3'):
       bp = C.bandpower(f, pc[u.index('C3')], 8, 13)
       v = float(np.log(bp + 1e-20))
       out['mu_c3'] = v if math.isfinite(v) else float('nan')
   ```
   - `C.bandpower(f, psd_row, 8, 13)` is trapezoid-integrated and non-negative; adding `1e-20` before
     `np.log` keeps a near-zero (flat/near-silent reconstructed) channel finite and large-negative
     rather than `-inf`. Do **not** clamp to a physiological range — a large negative log power is a
     legitimate result of a nearly flat channel and must surface as-is, finite.
   - This is an **absolute** log power (units: ln of (V/m²)² band power, with an arbitrary additive
     offset from the CSD normalization). It does **not** cancel a global scale — this is deliberate:
     it is the demanding, amplitude-fidelity-sensitive target (the FAA absolute-amplitude failure
     mode).

7. **Submetric `mu_c4`** — right-hemisphere absolute log band power. Requires C4 only; identical to
   step 6 with `'C4'`:
   ```
   if C.has(ch_names, 'C4'):
       bp = C.bandpower(f, pc[u.index('C4')], 8, 13)
       v = float(np.log(bp + 1e-20))
       out['mu_c4'] = v if math.isfinite(v) else float('nan')
   ```

8. `return out`.

**Imports the algorithm needs at module top:** `from base import Metric, register`,
`import common as C`, plus `import numpy as np` (for `np.log`) and `import math` (for
`math.isfinite`). `common` already imports numpy, but the module must import its own `np`/`math`
rather than reaching through `C`.

**Why the epsilon is duplicated as a literal `1e-20`:** requirements §2/§7 mandate that `mu_c3`/`mu_c4`
use the *same* `1e-20` `C.log_asymmetry` uses internally, so all three submetrics share one floor
convention. `common.py` does not export the constant, so the literal is written inline in steps 6–7.

---

## 3. Metric registration fields (exact values)

| field | value |
|-------|-------|
| `key` | `'mu_asymmetry'` |
| `name` | `'Sensorimotor mu asymmetry'` |
| `drop_channels` | `['C3', 'C4', 'C1', 'C2']` |
| `submetrics` | `['mu_asym', 'mu_c3', 'mu_c4']` |
| `compute` | the `compute` function above |
| `reference` | `'Pfurtscheller & Lopes da Silva 1999; Pineda 2005; Allen et al. 2004 (CSD variant)'` |
| `notes` | one line, ≤ ~100 chars, per skeleton |

- `drop_channels` = **recording sites C3/C4 + medial in-row neighbors C1/C2** (requirements §3):
  dropping the whole central-hand cluster forces reconstruction to infer mu from more distant
  surviving rows (FC/CP, Cz, T7/T8) instead of trivially interpolating an adjacent same-row electrode.
  Matching is case-insensitive in `run.py` (`drop_indices` upper-cases).
- `submetrics` order is fixed as `mu_asym, mu_c4?`→ **`['mu_asym', 'mu_c3', 'mu_c4']`** exactly; the
  aggregator preserves first-seen order for its table.

---

## 4. Edge-case handling (all must return finite float or explicit NaN — never crash)

Cross-referenced to requirements §7:

1. **Missing channels — guard each submetric independently.** Every submetric is behind its own
   `C.has(...)` gate (steps 5–7): `mu_asym` needs `C.has(ch_names, 'C3', 'C4')`; `mu_c3` needs
   `C.has(ch_names, 'C3')`; `mu_c4` needs `C.has(ch_names, 'C4')`. Absent → the key is simply omitted
   from `out` (the `m_faa.py` omit-key pattern; the runner skips absent keys). Valid partial results
   (e.g. only C4 survives ⇒ emit `mu_c4`, omit `mu_asym` and `mu_c3`) must not crash. `u.index(...)`
   is **only** called inside a confirmed-present branch — never blindly (no `ValueError`).
2. **Tiny/zero power → log instability.** The `+ 1e-20` floor in steps 5–7 keeps `ln` finite for a
   flat/near-silent channel; result is large-negative but finite. No clamping.
3. **NaN / non-finite propagation.** Each computed value is checked with `math.isfinite(v)`; if CSD or
   PSD produced non-finite output for a needed channel, that submetric is written as `float('nan')`
   rather than a poisoned number. Every emitted value is cast with `float(...)` — no numpy scalars or
   arrays escape.
4. **CSD prerequisites.** `C.csd` ignores position-less channels and preserves order, so C3/C4 index
   alignment after CSD holds whenever they are present (contract guarantees `standard_1005` labels).
   The finiteness guards in (2)–(3) cover a degenerate/failed Laplacian defensively.
5. **Few epochs.** `C.welch` uses `nperseg=min(1024, n_times)`; small `n_epochs` is noisier but
   computable — no special-casing beyond the finiteness guards.
6. **Degenerate equality.** If a reconstruction sets C3 == C4, `mu_asym == 0` exactly — correct, not
   an error.

Determinism: `compute` is pure given `(data, ch_names)`.

---

## 5. Robustness self-test (run after coding, before declaring done)

**Interpreter:** `zuna_env/Scripts/python.exe` (relative to the repository root; weights and the
environment are local). Run all commands from the repository root.

**Step A — import/registration smoke test** (no data; fast fail on syntax/contract errors):
```
zuna_env/Scripts/python.exe -c "import sys; sys.path.insert(0,'benchmark/metrics'); import m_mu_asymmetry, base; m=base.REGISTRY['mu_asymmetry']; print(m.key, m.drop_channels, m.submetrics)"
```
Expect: `mu_asymmetry ['C3', 'C4', 'C1', 'C2'] ['mu_asym', 'mu_c3', 'mu_c4']` and no exception.

**Step B — real runner on G001, method `linear`** (the required self-test). Invoke the actual
`run.py` so the full pipeline (preprocess → surviving-average-reference truth → drop C3/C4/C1/C2 →
linear reconstruct → recompute) exercises `compute` on both truth and recon frames:
```
zuna_env/Scripts/python.exe benchmark/metrics/run.py --subjects G001 --methods linear --metrics mu_asymmetry --out results/mu_asymmetry_selftest.csv
```
- `--metrics mu_asymmetry` isolates this plug-in (the runner imports only `m_mu_asymmetry.py`, so a
  half-written sibling cannot interfere).
- Runs over all `G001Day*Rest*.cnt` (10 recordings: Days 1–5 × Rest1/Rest2) — enough for both `truth`
  rows and, since Rest1+Rest2 exist per day, a valid same-day floor should be computable by
  `aggregate.py` later. The self-test itself only needs finite rows.
- **Delete any stale `results/mu_asymmetry_selftest.csv` first** — the runner is resumable and will
  skip recordings already present, so a re-run against a stale file would not re-exercise `compute`.

**Step C — assert finite truth + recon rows.** Parse the output CSV and check:
```
zuna_env/Scripts/python.exe -c "
import csv, math
rows=list(csv.DictReader(open('results/mu_asymmetry_selftest.csv')))
subs={'mu_asym','mu_c3','mu_c4'}
truth=[r for r in rows if r['metric']=='mu_asymmetry' and r['kind']=='truth']
recon=[r for r in rows if r['metric']=='mu_asymmetry' and r['kind']=='recon' and r['method']=='linear']
assert truth and recon, ('missing rows', len(truth), len(recon))
def fin(rs,col): return all(math.isfinite(float(r[col])) for r in rs)
assert set(r['submetric'] for r in truth)==subs, set(r['submetric'] for r in truth)
assert set(r['submetric'] for r in recon)==subs, set(r['submetric'] for r in recon)
assert fin(truth,'value') and fin(recon,'value') and fin(recon,'truth') and fin(recon,'abs_err')
print('OK truth_rows=%d recon_rows=%d'%(len(truth),len(recon)))
print('drop_set=', recon[0]['drop_set'])
"
```
Pass criteria:
- Both `truth` and `recon` rows exist for `mu_asymmetry`.
- All three submetrics (`mu_asym`, `mu_c3`, `mu_c4`) appear in each kind.
- Every `value`, and every recon `truth` and `abs_err`, is **finite** (no NaN/inf).
- `recon` rows carry `drop_set == 'C1+C2+C3+C4'` (the runner labels the group by sorted upper-cased
  drop channels).

**Sanity anchors (not hard assertions):** `mu_asym` truth values should land roughly in the
±0.2–0.35 neper order of magnitude (requirements §6, FAA cohort anchor ≈0.21–0.30); `mu_c3`/`mu_c4`
are absolute CSD log powers with an arbitrary offset so only their finiteness and cross-recording
consistency matter. If `mu_asym` is systematically an order of magnitude off, re-check the
`log_asymmetry` argument order (must be C4 then C3).

**Optional aggregate cross-check** (confirms the floor pipeline consumes the rows):
```
zuna_env/Scripts/python.exe benchmark/metrics/aggregate.py --csv results/mu_asymmetry_selftest.csv
```
Expect a table row per submetric with a finite floor and a finite `linear` error column. This is
informational for the plan stage; the authoritative pass/fail is Step C.

---

## 6. Definition of done (implementation stage)

- `benchmark/metrics/m_mu_asymmetry.py` exists, flat-imports `base`/`common`, and self-registers
  `key='mu_asymmetry'` at import.
- `compute` returns `{mu_asym, mu_c3, mu_c4}` (subset when channels are missing), all finite floats or
  explicit `float('nan')`, on CSD 8–13 Hz, using `C.mean_psd(C.csd(...))`, `C.log_asymmetry`, and
  `C.bandpower` — no re-implemented spectral/spatial math.
- Steps A–C of the self-test pass on G001/linear with all-finite truth and recon rows.
- No shared framework file modified.
