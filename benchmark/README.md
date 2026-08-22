# Authoritative benchmark implementation

This is the only active implementation. Historical root scripts, ZUNA 1.0 code,
old validators, and the former `pipeline/` tree are excluded from generated
releases.

The scientific contract is `../config/scientific_contract_v1.json`. Its current
status is `development_blocked`; therefore no production ZUNA or full HPC run is
authorized. In particular, the pinned ZUNA inference coordinate range conflicts
with nine unmodified `standard_1005` positions, and the adapter now fails closed
instead of allowing ZUNA to clamp them.

## Active commands

1. Check the active tree:

   ```bash
   python scripts/check_active_tree.py
   ```

2. Create an immutable run manifest after all inputs are present:

   ```bash
   python benchmark/run_manifest.py --data-dir /path/to/GEEG_Raw \
     --expected-recordings 42 --methods spline zuna \
     --out /path/to/zuna11_out/run_manifest.json
   ```

3. Validate an exact completed bundle:

   ```bash
   python benchmark/validate_bundle.py --run-manifest RUN.json \
     --results RESULTS.csv --qc QC.jsonl --require-all-success
   ```

4. Build the uploadable source-only release:

   ```bash
   python scripts/build_hpc_share.py dist/GEEG-ZUNA-share
   ```

See `HPC_RUNBOOK_zuna11.md` for the gated HPC workflow. The old one-step model
artifact remains useful only for saved-artifact replay and must fail experiment
completeness validation.

