"""Validate the ZUNA 1.1 (.fif) wrapper on one case (G001Day1Rest1, FAA drop).
Triggers the Zyphra/ZUNA1.1 weights download on first run; confirms the output parses and the
reconstruction is physiological before any full pass."""
import sys, os, numpy as np
sys.path.insert(0, 'benchmark'); sys.path.insert(0, os.path.join('benchmark', 'metrics'))
import pilot, biomarker_eval, zuna_method_v11

F = 'GEEG_Raw/G001Day1Rest1.cnt'
NAMES = ['F3', 'F4', 'F7', 'F8']
data, ch, pos = biomarker_eval.load_truth(F)
up = [c.upper() for c in ch]; dd = [up.index(n) for n in NAMES]
ref = pilot.surviving_average_reference(data, dd)
print(f"loaded {F} {data.shape}; dropping {NAMES}", flush=True)

dbg = {}
rec = zuna_method_v11.zuna_reconstruct(ref, ch, pos, dd, debug=dbg)
Z, a, b, good = dbg['Z'], dbg['a'], dbg['b'], dbg['good']


def tcorr(t, r):
    tc = t - t.mean(-1, keepdims=True); rc = r - r.mean(-1, keepdims=True)
    return ((tc * rc).sum(-1)) / (np.sqrt((tc**2).sum(-1) * (rc**2).sum(-1)) + 1e-20)


gc = tcorr(ref[:, good, :], Z[:, good, :])
print(f"\n[align] observed-channel raw ZUNA-1.1 vs truth: mean r {gc.mean():+.3f} "
      f"median {np.median(gc):+.3f} frac>0 {(gc > 0).mean():.2f}")
zo = Z[:, good, :].ravel(); to = ref[:, good, :].ravel(); pred = a * zo + b
r2 = 1 - ((to - pred)**2).sum() / (((to - to.mean())**2).sum() + 1e-20)
print(f"[calib] a={a:.3f} b={b:.3f} R2={r2:.4f}")
print("\nchan | recon std | truth std | temporal r")
for n, i in zip(NAMES, dd):
    r = tcorr(ref[:, i:i + 1, :], rec[:, i:i + 1, :]).mean()
    print(f"  {n:4s} | {rec[:, i, :].std():8.2f} | {ref[:, i, :].std():8.2f} | {r:+.3f}")
print(f"\nALL dropped: recon std {rec[:, dd, :].std():.2f}  truth std {ref[:, dd, :].std():.2f}")
print("VALIDATION OK" if np.isfinite(rec).all() else "VALIDATION: non-finite output!")
