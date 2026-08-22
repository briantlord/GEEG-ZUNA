"""_recon11.py — standalone ZUNA 1.1 (.fif / v4) reconstruction helper.

Run as a SUBPROCESS (`python benchmark/_recon11.py ...`) so that sys.path[0] is benchmark/ and the
vendored ZUNA 1.0 at the project root is NOT importable — `import zuna` then resolves to the
pip-installed 1.1 package. Replicates zuna.reconstruct_fif's eeg_eval invocation but (a) caps
target_packed_seqlen for a 12 GB GPU and (b) skips the per-file overlay figures (batch throughput).
Loads Zyphra/ZUNA1.1 weights (hardcoded in the 1.1 eeg_eval.py).
"""
import os, sys, argparse, subprocess, hashlib, json
import importlib.metadata
from pathlib import Path

if os.name == "nt":
    _mne_home = Path(__file__).resolve().parents[1] / ".mne_local"
    _mne_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(_mne_home))

import zuna  # must be the pip 1.1 package (has reconstruct_fif); see module docstring
assert hasattr(zuna, "reconstruct_fif"), f"imported non-1.1 zuna from {getattr(zuna,'__file__','?')}"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_cache(revision, expected_weight, expected_config):
    hub = Path(os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME", ""))
    cache = hub / "models--Zyphra--ZUNA1.1"
    ref = cache / "refs" / "main"
    if not ref.is_file() or ref.read_text(encoding="utf-8").strip() != revision:
        raise RuntimeError("Hugging Face main ref does not equal the pinned ZUNA revision")
    snapshot = cache / "snapshots" / revision
    config = snapshot / "config.json"
    weights = sorted(snapshot.glob("*.safetensors"))
    if not weights:
        weights = sorted(
            path for path in (cache / "blobs").glob("*")
            if path.is_file() and path.stat().st_size > 100_000_000
        )
    if len(weights) != 1 or not config.is_file():
        raise RuntimeError("Pinned ZUNA model cache is incomplete or ambiguous")
    if sha256_file(weights[0]) != expected_weight or sha256_file(config) != expected_config:
        raise RuntimeError("Pinned ZUNA model weight/config hash mismatch")
    return snapshot, weights[0], config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--tmp_dir', required=True)
    ap.add_argument('--repair', required=True, help='comma-separated channel names to reconstruct')
    ap.add_argument('--montage', default='standard_1005')
    ap.add_argument('--highpass', default='none',
                    help="'none' for prefiltered corrected-v2 input, or a numeric Hz value")
    ap.add_argument('--segment_sec', type=float, default=5.0)
    ap.add_argument('--sample_steps', type=int, default=50)
    ap.add_argument('--seqlen', type=int, default=8000)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--seed', type=int, default=333)
    ap.add_argument('--model_revision', required=True)
    ap.add_argument('--weight_sha256', required=True)
    ap.add_argument('--config_sha256', required=True)
    a = ap.parse_args()

    if importlib.metadata.version("zuna") != "1.1.3":
        raise RuntimeError("Expected installed zuna distribution 1.1.3")
    snapshot, weight_path, model_config = verify_model_cache(
        a.model_revision, a.weight_sha256, a.config_sha256
    )

    pkg = Path(zuna.__file__).parent
    app = pkg / "inference" / "AY2l" / "lingua" / "apps" / "AY2latent_bci"
    eeg_eval = app / "eeg_eval.py"
    config = app / "configs" / "config_infer_fif.yaml"
    lingua_root = app.parent.parent
    for d in (a.output_dir, a.tmp_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    highpass = None if str(a.highpass).lower() in {'none', 'null', 'off'} else float(a.highpass)
    ov = {
        "config": str(config),
        "seed": a.seed,
        "dump_dir": str(Path(a.tmp_dir).absolute()),
        "inference_figures_dir": str(Path(a.tmp_dir).absolute()),
        "data.data_dir": str(Path(a.input_dir).absolute()),
        "data.v4_recon_save_fif": "true",
        "data.v4_recon_out_dir": str(Path(a.output_dir).absolute()),
        "data.v4_segment_sec": a.segment_sec,
        "data.v4_montage": a.montage,
        "data.v4_use_fif_annotations": "false",
        # Corrected-v2 input is already band-limited and referenced over surviving
        # channels. A second filter/reference pass would reintroduce the audited bugs.
        "data.v4_highpass_hz": "null" if highpass is None else highpass,
        "data.v4_lowpass_hz": "null",
        "data.v4_notch_hz": "null",
        "data.do_avg_ref": "false",
        "data.z_score_type": "across_channel",
        "data.v4_recon_unmasked_from_original": "true",
        "data.v4_recon_seam_correct": "false",
        "data.shuffle": "false",
        "data.v4_drop_channels": "[" + ",".join(s.strip() for s in a.repair.split(",")) + "]",
        "data.target_packed_seqlen": a.seqlen,
        "diffusion_cfg": 1.0,
        "diffusion_sample_steps": a.sample_steps,
    }
    cmd = [sys.executable, str(eeg_eval)] + [f"{k}={v}" for k, v in ov.items()]

    env = os.environ.copy()
    scheduler_gpu_mask = env.get("CUDA_VISIBLE_DEVICES")
    if scheduler_gpu_mask:
        effective_gpu = "0"
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
        effective_gpu = str(a.gpu)
    env["WANDB_MODE"] = "disabled"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    if os.name == "nt":
        # Windows-only crutches: this torch build has no libuv and torch.compile/Triton is unreliable.
        # On Linux (HPC) we deliberately DO NOT set these, so torch.compile runs — ~8x faster.
        env["USE_LIBUV"] = "0"
        env["TORCHDYNAMO_DISABLE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(lingua_root), str(app), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    port_seed = f"{env.get('SLURM_JOB_ID', '')}:{env.get('SLURM_ARRAY_TASK_ID', '')}:{os.getpid()}"
    master_port = 15000 + int(hashlib.sha256(port_seed.encode()).hexdigest()[:8], 16) % 40000
    env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(master_port), WORLD_SIZE="1",
               RANK="0", LOCAL_RANK="0", NCCL_IB_DISABLE="1")
    for s in (sys.stdout, sys.stderr):
        if s is not None and hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8")
            except Exception:
                pass
    subprocess.run(cmd, env=env, check=True)
    verify_model_cache(a.model_revision, a.weight_sha256, a.config_sha256)
    provenance = {
        "zuna_distribution": importlib.metadata.version("zuna"),
        "zuna_module": str(Path(zuna.__file__).resolve()),
        "python": sys.version,
        "model_revision": a.model_revision,
        "model_snapshot": str(snapshot.resolve()),
        "weight_path": str(weight_path.resolve()),
        "weight_sha256": a.weight_sha256,
        "config_path": str(model_config.resolve()),
        "config_sha256": a.config_sha256,
        "scheduler_cuda_visible_devices": scheduler_gpu_mask,
        "effective_local_gpu": effective_gpu,
        "master_port": master_port,
        "slurm": {key: value for key, value in os.environ.items() if key.startswith("SLURM_")},
    }
    (Path(a.output_dir) / "helper_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("recon11 helper done")


if __name__ == "__main__":
    main()
