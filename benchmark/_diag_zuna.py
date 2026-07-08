"""Comprehensive single-recording ZUNA diagnostic (G001Day1Rest1, FAA drop set).

Answers, in one GPU run:
  1. CHANNEL ALIGNMENT — does ZUNA's *raw* output for the observed (good) channels correlate
     positively with truth? (rules out a channel-permutation bug; good chans are inpainted in the
     final rec, so this is the only place the raw model output for known channels is visible)
  2. CALIBRATION — self-cal a,b and the R^2 of the good-channel fit.
  3. DROPPED CHANNELS — amplitude + broadband AND alpha-band (8-13 Hz) temporal correlation.
  4. FAA BIOMARKER (the real question) — truth vs linear vs spline vs zuna, |err| vs floor.
Saves the reconstruction + raw output to _diag_*.npy for reuse without re-running.
"""
import sys, os, numpy as np
sys.path.insert(0, 'benchmark')
import pilot, zuna_method, biomarker_eval
from scipy.signal import butter, filtfilt

F = 'GEEG_Raw/G001Day1Rest1.cnt'
NAMES = ['F3', 'F4', 'F7', 'F8']
data, ch, pos = biomarker_eval.load_truth(F)
up = [c.upper() for c in ch]
dd = [up.index(n) for n in NAMES]
ref = pilot.surviving_average_reference(data, dd)

dbg = {}
print("running ZUNA (FAA drop set) ...", flush=True)
rec = zuna_method.zuna_reconstruct(ref, ch, pos, dd, debug=dbg)
Z, a, b, good = dbg['Z'], dbg['a'], dbg['b'], dbg['good']
np.save('benchmark/_diag_zuna_rec.npy', rec)
np.save('benchmark/_diag_zuna_Z.npy', Z)


def tcorr(t, r):                       # mean temporal corr along time, per (epoch, channel)
    tc = t - t.mean(-1, keepdims=True); rc = r - r.mean(-1, keepdims=True)
    num = (tc * rc).sum(-1); den = np.sqrt((tc**2).sum(-1) * (rc**2).sum(-1)) + 1e-20
    return num / den


# 1. CHANNEL ALIGNMENT — raw ZUNA good-channel output vs truth (corr is scale-invariant)
gc = tcorr(ref[:, good, :], Z[:, good, :])
gcm = gc.mean(0)
print(f"\n[1] ALIGNMENT — raw ZUNA vs truth on the {len(good)} OBSERVED channels:")
print(f"    temporal r: mean {gc.mean():+.3f}, median {np.median(gc):+.3f}, frac>0 {(gc>0).mean():.2f}")
print(f"    worst observed: {[(ch[good[i]], round(float(gcm[i]),2)) for i in np.argsort(gcm)[:6]]}")
print(f"    best  observed: {[(ch[good[i]], round(float(gcm[i]),2)) for i in np.argsort(gcm)[-6:]]}")

# 2. CALIBRATION fit quality
zo = Z[:, good, :].ravel(); to = ref[:, good, :].ravel()
pred = a * zo + b
r2 = 1 - ((to - pred)**2).sum() / (((to - to.mean())**2).sum() + 1e-20)
print(f"\n[2] CALIBRATION  a={a:.4f}  b={b:.4f}  good-channel fit R^2={r2:.4f}")

# 3. DROPPED channels: amplitude + broadband and alpha-band correlation
bb, aa = butter(4, [8/128., 13/128.], btype='band')
alpha = lambda x: filtfilt(bb, aa, x, axis=-1)
print("\n[3] DROPPED channels:")
print("    chan | recon std | truth std | r_broadband | r_alpha(8-13Hz)")
for n, i in zip(NAMES, dd):
    rb = tcorr(ref[:, i:i+1, :], rec[:, i:i+1, :]).mean()
    ra = tcorr(alpha(ref[:, i, :]), alpha(rec[:, i, :])).mean()
    print(f"    {n:<4} | {rec[:,i,:].std():9.1f} | {ref[:,i,:].std():9.1f} | {rb:+11.3f} | {ra:+.3f}")

# 4. FAA biomarker — the real question
bt = pilot.biomarkers(ref, ch)
bz = pilot.biomarkers(rec, ch)
bl = pilot.biomarkers(pilot.reconstruct('linear', ref, ch, pos, dd), ch)
bs = pilot.biomarkers(pilot.reconstruct('spline', ref, ch, pos, dd), ch)
print("\n[4] FAA biomarker (single recording; floor is the 5-subj same-day reference):")
print("    metric    truth   linear   spline    zuna  |  |lin|  |spl|  |zuna|  floor")
for key, fl in [('faa', 0.208), ('faa_lat', 0.301)]:
    el, es, ez = abs(bl[key]-bt[key]), abs(bs[key]-bt[key]), abs(bz[key]-bt[key])
    print(f"    {key:<8} {bt[key]:+.3f}  {bl[key]:+.3f}   {bs[key]:+.3f}   {bz[key]:+.3f}  "
          f"| {el:.3f}  {es:.3f}  {ez:.3f}   {fl:.3f}")
