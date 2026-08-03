"""_recon11.py — standalone ZUNA 1.1 (.fif / v4) reconstruction helper.

Run as a SUBPROCESS (`python benchmark/_recon11.py ...`) so that sys.path[0] is benchmark/ and the
vendored ZUNA 1.0 at the project root is NOT importable — `import zuna` then resolves to the
pip-installed 1.1 package. Replicates zuna.reconstruct_fif's eeg_eval invocation but (a) caps
target_packed_seqlen for a 12 GB GPU and (b) skips the per-file overlay figures (batch throughput).
Loads Zyphra/ZUNA1.1 weights (hardcoded in the 1.1 eeg_eval.py).
"""
import os, sys, argparse, subprocess
from pathlib import Path

import zuna  # must be the pip 1.1 package (has reconstruct_fif); see module docstring
assert hasattr(zuna, "reconstruct_fif"), f"imported non-1.1 zuna from {getattr(zuna,'__file__','?')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--tmp_dir', required=True)
    ap.add_argument('--repair', required=True, help='comma-separated channel names to reconstruct')
    ap.add_argument('--montage', default='standard_1005')
    ap.add_argument('--highpass', type=float, default=0.5)
    ap.add_argument('--segment_sec', type=float, default=5.0)
    ap.add_argument('--sample_steps', type=int, default=50)
    ap.add_argument('--seqlen', type=int, default=8000)
    ap.add_argument('--gpu', default='0')
    a = ap.parse_args()

    pkg = Path(zuna.__file__).parent
    app = pkg / "inference" / "AY2l" / "lingua" / "apps" / "AY2latent_bci"
    eeg_eval = app / "eeg_eval.py"
    config = app / "configs" / "config_infer_fif.yaml"
    lingua_root = app.parent.parent
    for d in (a.output_dir, a.tmp_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    ov = {
        "config": str(config),
        "dump_dir": str(Path(a.tmp_dir).absolute()),
        "inference_figures_dir": str(Path(a.tmp_dir).absolute()),
        "data.data_dir": str(Path(a.input_dir).absolute()),
        "data.v4_recon_save_fif": "true",
        "data.v4_recon_out_dir": str(Path(a.output_dir).absolute()),
        "data.v4_segment_sec": a.segment_sec,
        "data.v4_montage": a.montage,
        "data.v4_use_fif_annotations": "true",
        "data.v4_highpass_hz": a.highpass,
        "data.v4_drop_channels": "[" + ",".join(s.strip() for s in a.repair.split(",")) + "]",
        "data.target_packed_seqlen": a.seqlen,
        "diffusion_cfg": 1.0,
        "diffusion_sample_steps": a.sample_steps,
    }
    cmd = [sys.executable, str(eeg_eval)] + [f"{k}={v}" for k, v in ov.items()]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    env["WANDB_MODE"] = "disabled"
    if os.name == "nt":
        # Windows-only crutches: this torch build has no libuv and torch.compile/Triton is unreliable.
        # On Linux (HPC) we deliberately DO NOT set these, so torch.compile runs — ~8x faster.
        env["USE_LIBUV"] = "0"
        env["TORCHDYNAMO_DISABLE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(lingua_root), str(app), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    for k in [k for k in env if k.startswith("SLURM_")]:
        del env[k]
    env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29500", WORLD_SIZE="1",
               RANK="0", LOCAL_RANK="0", NCCL_IB_DISABLE="1")
    for s in (sys.stdout, sys.stderr):
        if s is not None and hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8")
            except Exception:
                pass
    subprocess.run(cmd, env=env, check=True)
    print("recon11 helper done")


if __name__ == "__main__":
    main()
