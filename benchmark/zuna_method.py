"""
zuna_method.py — ZUNA reconstruction for the benchmark.  *** REQUIRES A GPU ***

Cannot run in a CPU-only environment (no torch/GPU). Run on the Windows GPU box where
main_pipeline.py already works, or on the HPC. It reproduces the proven load_data.py
.pt export (z-score preserved channels; scale electrode positions into ZUNA's +/-0.12
box; the filename the dataloader parses), calls zuna.inference (data_norm=10, 50 steps),
then maps ZUNA's output back to microvolts by a SELF-CALIBRATION against the observed
channels (whose true uV are known) — so it is robust to ZUNA's internal normalization,
which we could not verify offline. **Sanity-check the first reconstruction** against a
known-good main_pipeline.py output before trusting the numbers.
"""
import os, sys, glob, tempfile, shutil
import numpy as np

# zuna/pipeline.py prints a "✓" after inference; on a Windows cp1252 console that raises
# UnicodeEncodeError and kills the run *after* the output .pt is already written. main_pipeline.py
# dodges this by reconfiguring stdout to utf-8 — do the same for any process importing this wrapper.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Make the project root importable so `from zuna import inference` resolves no matter how
# this module is launched. When run as `python benchmark/biomarker_eval.py`, sys.path[0] is
# benchmark/, not the project root where the `zuna` package lives — without this the GPU rung
# fails at import (the linear/spline rungs never import zuna, so this path was never exercised).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# Point HF at the project-local cache the proven main_pipeline.py uses (only if unset), so the
# weights resolve to the known-good location instead of re-downloading to the default cache.
os.environ.setdefault("HF_HOME", os.path.join(_ROOT, "HF_cache"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_ROOT, "HF_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ZUNA chops each 5 s / 1280-sample epoch into coarse-time steps of num_fine_time_pts=32 samples
# (config_infer.yaml), so one epoch = n_channels x (n_time // 32) tokens. flex_attention builds an
# O(L^2) document mask over the *packed* batch, so the YAML default target_packed_seqlen=100000
# packs ~41 dense-montage epochs into one batch and OOMs a 12 GB GPU (mask alone ~38 GiB). We cap
# the packed length here instead; epochs are independent documents, so packing never changes the
# result — only throughput. ~8000 tokens keeps the mask under ~1 GB with headroom for the model.
NUM_FINE_TIME_PTS = 32
SAFE_PACKED_TOKENS = 8000


def _stack(x):
    return np.asarray(x) if not isinstance(x, list) else np.stack([np.asarray(s) for s in x])


def zuna_reconstruct(data, ch_names, pos, dropped, gpu_device=0,
                     diffusion_sample_steps=50, data_norm=10.0, tokens_per_batch=None,
                     debug=None, inference_fn=None):
    """data: (n_ep, n_ch, n_time) uV in the surviving-channel reference frame. Returns same shape, dropped filled.

    If `debug` is a dict, it is populated with the raw ZUNA output Z (model units) and the
    self-calibration coefficients (a, b) — for diagnostics only; the return value is unchanged.
    """
    import torch
    # Default: the vendored ZUNA 1.0 inference(). Pass inference_fn=... to run a different version
    # (e.g. zuna_method_v11._inference_11 for ZUNA 1.1) with the harness otherwise held constant.
    _inference = inference_fn
    if _inference is None:
        from zuna import inference as _inference
    tin, tout = tempfile.mkdtemp(prefix="zuna_in_"), tempfile.mkdtemp(prefix="zuna_out_")
    try:
        ne, nc, nt = data.shape
        good = [i for i in range(nc) if i not in dropped]
        # Cap packed batch length so the flex_attention mask fits in GPU memory (see module note).
        if tokens_per_batch is None:
            tok_per_epoch = nc * (nt // NUM_FINE_TIME_PTS)
            epochs_per_batch = max(1, SAFE_PACKED_TOKENS // tok_per_epoch)
            tokens_per_batch = tok_per_epoch * epochs_per_batch
        # 1. mask dropped -> 0 ; z-score preserved channels (load_data.export_zuna_tensors logic)
        X = data.astype(np.float32).copy(); X[:, dropped, :] = 0.0
        nz = np.any(X != 0.0, axis=-1)
        zmean, zstd = float(X[nz].mean()), float(X[nz].std())
        Xn = X.copy()
        for ep in range(ne):
            for c in range(nc):
                if nz[ep, c]:
                    Xn[ep, c, :] = (Xn[ep, c, :] - zmean) / zstd
        # 2. positions into ZUNA's +/-0.12 box
        P = pos.astype(np.float32).copy(); mx = float(np.max(np.abs(P)))
        if mx > 0.119: P *= (0.119 / mx)
        # 3. write the .pt the dataloader expects (prefix_epochs_chans_time)
        pt = {'data': torch.tensor(Xn, dtype=torch.float32),
              'channel_positions': torch.tensor(P, dtype=torch.float32),
              'metadata': {'sfreq': 256, 'ch_names': list(ch_names),
                           'zscore_mean': zmean, 'zscore_std': zstd}}
        torch.save(pt, os.path.join(tin, f"ds000000_000000_000000_d00_{ne:05d}_{nc}_{nt}.pt"))
        # 4. ZUNA inference (writes a .pt with the reconstruction under key 'data')
        _inference(input_dir=tin, output_dir=tout, gpu_device=gpu_device,
                   data_norm=data_norm, diffusion_sample_steps=diffusion_sample_steps,
                   tokens_per_batch=tokens_per_batch)
        # weights_only=False: the output .pt holds numpy arrays (the reconstruction), which
        # torch 2.6's default weights_only=True refuses to unpickle. We wrote this file ourselves
        # this run, so it is trusted — same as main_pipeline.py's loads.
        out = torch.load(glob.glob(os.path.join(tout, "*.pt"))[0], map_location="cpu", weights_only=False)
        Z = _stack(out['data']).reshape(ne, nc, nt).astype(np.float32)   # ZUNA output (model units)
        # 5. self-calibrate model units -> uV using OBSERVED channels (true uV known)
        zo = Z[:, good, :].ravel(); to = data[:, good, :].ravel()
        a, b = np.polyfit(zo, to, 1)                                     # to ~= a*zo + b
        rec = data.astype(np.float32).copy()
        rec[:, dropped, :] = a * Z[:, dropped, :] + b                    # ZUNA fill, in uV
        rec[:, good, :] = data[:, good, :]                              # hard-inpaint observed
        if debug is not None:
            debug.update(Z=Z, a=float(a), b=float(b), good=good, dropped=list(dropped))
        return rec
    finally:
        shutil.rmtree(tin, ignore_errors=True); shutil.rmtree(tout, ignore_errors=True)
