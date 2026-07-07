import os, sys
# ZUNA weights load from HuggingFace (Zyphra/ZUNA) and auto-download to the default HF cache.
# To use a pre-downloaded snapshot, set HF_HOME before running (e.g. export HF_HOME=/path/to/HF_cache).
if os.environ.get("HF_HOME"):
    os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"])
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from main_pipeline import get_alpha_metrics, SUBJECT_ID, SESSION_NAME
import torch, glob

truth     = np.load(f"{SUBJECT_ID}_{SESSION_NAME}_y_truth.npy")[:16]
spline    = np.load(f"{SUBJECT_ID}_{SESSION_NAME}_X_spline.npy")[:16]
zuna_data = np.load(f"{SUBJECT_ID}_{SESSION_NAME}_X_zuna_test.npy")[:16]

pt_file  = sorted(glob.glob("test_pt_out/*.pt"))[0]
ch_names = sorted(torch.load(pt_file, weights_only=False)["metadata"]["ch_names"])
post = [c for c in ch_names if c.upper() in {"O1","O2","OZ","P3","P4","PZ","P7","P8","PO3","PO4","POZ"}]
print("Posterior channels used for IAF:", post)
print()

f, psd_t, iaf_t, faa_t = get_alpha_metrics(truth,     sfreq=256, ch_names=ch_names)
f, psd_s, iaf_s, faa_s = get_alpha_metrics(spline,    sfreq=256, ch_names=ch_names)
f, psd_z, iaf_z, faa_z = get_alpha_metrics(zuna_data, sfreq=256, ch_names=ch_names)

print("Condition        IAF (Hz)      FAA")
print("-" * 40)
print(f"TRUTH            {iaf_t:.3f}      {faa_t:.4f}")
print(f"SPLINE           {iaf_s:.3f}      {faa_s:.4f}")
print(f"ZUNA             {iaf_z:.3f}      {faa_z:.4f}")
