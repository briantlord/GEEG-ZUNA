# Runbook — ZUNA 1.1 metric battery on the UA HPC

Runs the modular metric harness with the **ZUNA 1.1** reconstructor (`Zyphra/ZUNA1.1`, `.fif` /
`reconstruct_fif` path) across all 5 subjects, to fill the ZUNA-1.1 column of REPORT §6.4.

**Why the HPC:** ZUNA 1.1's inference uses `torch.compile`, which needs Triton — available on Linux
but not on the Windows box (there it falls back to eager and is ~8× slower, making the pass
infeasible). On Linux it runs at full speed. Everything below assumes the **UA HPC** (verified against
<https://hpcdocs.hpc.arizona.edu/>, 2026).

## 0. What the code does (already built, cross-platform)
- `benchmark/metrics/run.py --zuna-version 1.1` selects `benchmark/zuna_method_v11.py`, which lays the
  epochs into one continuous `.fif`, runs ZUNA 1.1 via the standalone `benchmark/_recon11.py` helper,
  reads the model output, and **self-calibrates** it back to our µV frame (same logic as the 1.0
  wrapper). The harness is held constant; only the model changes.
- The Windows-only workarounds (SIGUSR2 guard, `gloo` backend, `USE_LIBUV=0`, `TORCHDYNAMO_DISABLE=1`)
  are gated to `os.name == "nt"` and are **NOT needed on Linux** — a fresh `pip install zuna` runs
  natively (NCCL + `torch.compile`). No package patching required on the HPC.

## 1. One-time setup
```bash
# scratch is the right home for data + I/O (large, high-throughput)
GRP=<your_PI_group>
mkdir -p /xdisk/$GRP/geeg && cd /xdisk/$GRP/geeg
git clone <your GEEG-ZUNA repo> GEEG-ZUNA        # or rsync the repo here
mkdir -p GEEG-ZUNA/GEEG_Raw                        # then stage the raw .cnt files into it
export HF_HOME=/xdisk/$GRP/geeg/HF_cache           # ZUNA1.1 weights auto-download here on first run

# environment (Anaconda is deprecated on UA HPC -> use mamba; conda shown for familiarity)
module load anaconda/2022.05        # or the current mamba module
source ~/.bashrc
conda create -y -n zuna11 python=3.10
conda activate zuna11
pip install "torch>=2.5" --index-url https://download.pytorch.org/whl/cu124   # CUDA build (brings triton)
pip install mne scipy numpy
pip install zuna                     # on Linux this pulls ALL its deps (transformers, lm-eval, etc.) — fine
python -c "import zuna, torch; print('zuna', zuna.__version__, '| torch', torch.__version__, torch.cuda.is_available())"
```

## 2. Validate ONE recording first (the gate we could not finish on Windows)
The wrapper's `.fif` build → model-output parse → self-calibration was confirmed to run end-to-end, but
the local eager run was too slow to finish, so **confirm sane output on one recording before the full
array**:
```bash
cd /xdisk/$GRP/geeg/GEEG-ZUNA
python benchmark/metrics/run.py --subjects G001 --methods linear spline zuna --zuna-version 1.1 \
       --shard-index 0 --shard-count 10 --out /tmp/zuna11_smoke.csv     # 1 recording, 5 drop sets
python - <<'PY'
import csv,math
rows=[r for r in csv.DictReader(open('/tmp/zuna11_smoke.csv'))]
z=[r for r in rows if r['kind']=='recon' and r['method']=='zuna']
bad=[r for r in z if not math.isfinite(float(r['value']))]
print(f"zuna recon rows={len(z)}  nonfinite={len(bad)}")
print("sample:", [(r['metric'],r['submetric'],round(float(r['value']),3)) for r in z[:6]])
PY
```
Expect ~ (5 drop sets × their submetrics) finite zuna rows with physiological magnitudes. If good, proceed.

## 3. Submit the full array
Edit the three placeholders in `benchmark/slurm_zuna11_metrics.sh` (`--account`, paths `REPO`/`OUT`/`HF_HOME`),
then:
```bash
cd /xdisk/$GRP/geeg/GEEG-ZUNA
sbatch benchmark/slurm_zuna11_metrics.sh          # 42 tasks (one recording each), writes per-shard CSVs
squeue -u $USER                                   # monitor;  tail -f logs/zuna11_*_*.out
```
Per-shard CSVs mean **no cross-task races** and easy resume: re-`sbatch` the same array and finished
recordings are skipped (run.py resumes by recording within each shard file).

## 4. Collect + aggregate
```bash
OUT=/xdisk/$GRP/geeg/zuna11_out
awk 'FNR==1 && NR!=1 {next} {print}' "$OUT"/shard_*.csv > "$OUT/metric_eval_5subj_zuna11.csv"
python benchmark/metrics/aggregate.py --csv "$OUT/metric_eval_5subj_zuna11.csv"
```
That prints the **floor / linear / spline / ZUNA-1.1** table (14 submetrics). Copy
`metric_eval_5subj_zuna11.csv` back into `results/` in the repo, and update REPORT §6.4 with the 1.1
column alongside the 1.0 numbers.

## Notes / gotchas
- **Partition:** GPU jobs need `gpu_standard` (consumable) or `gpu_windfall` (free, preemptible, no
  `--account`). Plain `standard` is CPU-only. Buy-in `gpu_high_priority` also needs
  `--qos=user_qos_<group>`.
- **GPU:** `--gres=gpu:1` works anywhere; `--gres=gpu:volta:1` pins a Puma V100S (32 GB, recommended;
  the 380M model + activations peaked ~12 GB at seqlen 8000). P100 (16 GB) also fits.
- **seqlen:** the script leaves `--seqlen` at the wrapper default (8000). On a 32 GB card you can raise
  it (edit `zuna_method_v11.py` / pass through) for better GPU utilization.
- **Preprocessing:** each task preprocesses its own recording from `GEEG_Raw/*.cnt` (MNE). No shared
  cache needed; `$TMPDIR` (node-local NVMe) is used for the transient `.fif` I/O automatically.
