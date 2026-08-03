"""zuna_method_v11.py — reconstruct dropped channels with ZUNA 1.1 (Zyphra/ZUNA1.1), .fif path.

ZUNA 1.1 removed the .pt reconstruction output the 1.0 wrapper used; its supported inference for
arbitrary data is the v4 `.fif` path (`zuna.reconstruct_fif`). This module drives that path while
keeping the harness comparison intact:

  1. take the (n_ep, n_ch, n_time) data the runner passes (already in the surviving-channel
     average-reference frame, microvolts) and lay the epochs end-to-end into ONE continuous .fif
     (256 Hz, standard_1005 montage);
  2. run ZUNA 1.1 v4 reconstruction on it via the standalone `_recon11.py` helper (a subprocess so
     `import zuna` resolves to the pip-installed 1.1 package, not the vendored 1.0 at the repo root),
     repairing exactly the dropped channels;
  3. read the FULL model-output .fif (model on every channel), re-epoch it, and — exactly like the
     1.0 wrapper — SELF-CALIBRATE model units back to our microvolt frame with a single linear fit on
     the observed channels, then hard-inpaint (dropped = calibrated model, observed = truth).

The one deliberate difference from a pure model swap: ZUNA 1.1's .fif path applies its own light
preprocessing (0.5 Hz highpass; no low-pass; 256 Hz — matches ours, so effectively a no-op) — see
REPORT version note. Biomarkers are all <= 30 Hz, so this does not affect the metrics; self-calibration
absorbs any residual global scale/offset.
"""
import os, sys, glob, shutil, tempfile, subprocess
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_HELPER = os.path.join(_HERE, "_recon11.py")
# 1.1 weights (Zyphra/ZUNA1.1) download to the project-local HF cache, same as 1.0.
os.environ.setdefault("HF_HOME", os.path.join(_ROOT, "HF_cache"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_ROOT, "HF_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

SFREQ = 256


def zuna_reconstruct(data, ch_names, pos, dropped, gpu_device=0,
                     sample_steps=50, segment_sec=5.0, highpass_hz=0.5, seqlen=8000,
                     debug=None):
    """data: (n_ep, n_ch, n_time) uV in the surviving-channel reference frame. Returns same shape,
    dropped channels filled by ZUNA 1.1 (self-calibrated to uV). Signature matches zuna_method."""
    import mne
    mne.set_log_level("ERROR")
    ne, nc, nt = data.shape
    good = [i for i in range(nc) if i not in dropped]
    tin = tempfile.mkdtemp(prefix="z11_in_")
    tout = tempfile.mkdtemp(prefix="z11_out_")
    ttmp = tempfile.mkdtemp(prefix="z11_tmp_")
    try:
        # 1. epochs -> one continuous .fif (uV -> V), standard_1005 montage
        cont = np.asarray(data, np.float64).transpose(1, 0, 2).reshape(nc, ne * nt)  # (nc, ne*nt)
        info = mne.create_info(list(ch_names), float(SFREQ), ch_types="eeg")
        raw = mne.io.RawArray(cont * 1e-6, info, verbose="ERROR")
        raw.set_montage(mne.channels.make_standard_montage("standard_1005"),
                        match_case=False, on_missing="ignore")
        raw.save(os.path.join(tin, "sample_raw.fif"), overwrite=True, verbose="ERROR")

        # 2. ZUNA 1.1 v4 reconstruction via the standalone helper (imports the pip 1.1 package)
        repair = ",".join(ch_names[i] for i in dropped)
        cmd = [sys.executable, _HELPER,
               "--input_dir", tin, "--output_dir", tout, "--tmp_dir", ttmp,
               "--repair", repair, "--montage", "standard_1005",
               "--highpass", str(highpass_hz), "--segment_sec", str(segment_sec),
               "--sample_steps", str(sample_steps), "--seqlen", str(seqlen),
               "--gpu", str(gpu_device)]
        subprocess.run(cmd, check=True)

        # 3. read the FULL model-output .fif (model everywhere) -> Z, re-epoch
        outs = glob.glob(os.path.join(tout, "full_reconstruction", "*.fif"))
        if not outs:
            raise RuntimeError(f"ZUNA 1.1 produced no full_reconstruction .fif in {tout}")
        rr = mne.io.read_raw_fif(outs[0], preload=True, verbose="ERROR")
        name2i = {n.upper(): k for k, n in enumerate(rr.ch_names)}
        picks = [name2i[n.upper()] for n in ch_names]           # our channel order
        Zc = rr.get_data()[picks] * 1e6                         # (nc, n_samp) uV
        if Zc.shape[1] != ne * nt:
            # be robust to any segment padding/cropping: trim or pad to ne*nt
            if Zc.shape[1] > ne * nt:
                Zc = Zc[:, :ne * nt]
            else:
                Zc = np.pad(Zc, ((0, 0), (0, ne * nt - Zc.shape[1])))
        Z = Zc.reshape(nc, ne, nt).transpose(1, 0, 2).astype(np.float32)   # (ne, nc, nt)

        # 4. self-calibrate model units -> uV on OBSERVED channels; hard-inpaint
        zo = Z[:, good, :].ravel(); to = data[:, good, :].ravel()
        a, b = np.polyfit(zo, to, 1)
        rec = np.asarray(data, np.float32).copy()
        rec[:, dropped, :] = a * Z[:, dropped, :] + b
        rec[:, good, :] = data[:, good, :]
        if debug is not None:
            debug.update(Z=Z, a=float(a), b=float(b), good=good, dropped=list(dropped))
        return rec
    finally:
        for d in (tin, tout, ttmp):
            shutil.rmtree(d, ignore_errors=True)
