# GEEG-ZUNA benchmark

This repository is being remediated after the 2026-08-21 scratch code audit.
The only authoritative implementation is `benchmark/`. The directory
`GEEG-ZUNA-share/` is generated deployment output and must not be hand-edited.

No current result is a complete corrected ZUNA 1.1 benchmark. Full model or HPC
execution remains blocked until the validation ladder in
`CODE_REMEDIATION_PLAN_2026-08-21.md` passes.

The primary Stage-0 contract is `PHASE1_CORRECTED_PROTOCOL.md`: minimally
processed EEG with no ICA, no component subtraction, and no amplitude-based
epoch selection. Stage-0 cache v3 is obsolete; current code writes v4 so an
ICA-cleaned tensor cannot be resumed accidentally.

## Active boundaries

- `benchmark/`: authoritative benchmark source.
- `scripts/`: repository checks and deterministic release builder.
- `config/`: versioned scientific and execution contracts.
- `archive/`: locally preserved legacy/invalid material; never imported or shipped.
- `GEEG_Raw/`, `HF_cache/`, `results/`: data/model/generated state; never source.
- `GEEG-ZUNA-share/`: generated HPC upload bundle; never source.

Do not invoke root-level historical scripts or import a project-local `zuna`
package. Active inference must resolve the pinned installed ZUNA distribution
from the declared environment.
