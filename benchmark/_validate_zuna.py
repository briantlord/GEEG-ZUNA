"""Scratch validation of the ZUNA wrapper on a single case (Step 2 of the run).

Mirrors biomarker_eval exactly: preprocess G001Day1Rest1, drop the FAA channels
(F3/F4/F7/F8), reconstruct with ZUNA, and report whether the reconstructed dropped
channels have physiological amplitude (~5-40 uV) and positive temporal correlation
to the truth channels (in the bad-aware reference frame the benchmark scores in).
"""
import sys, os, numpy as np
# Point HF at the project-local cache (this project lives on C:, not the stale D:\ path the
# reference scripts hardcode). zuna_method also setdefaults these, but the reference weights
# live here, so set them explicitly before importing the wrapper.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = os.path.join(_ROOT, "HF_cache")
os.environ["HF_HUB_CACHE"] = os.path.join(_ROOT, "HF_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, 'benchmark')
import pilot, zuna_method, biomarker_eval

F = 'GEEG_Raw/G001Day1Rest1.cnt'
NAMES = ['F3', 'F4', 'F7', 'F8']

# load_truth caches to tcache, so the real biomarker_eval run reuses this preprocess
data, ch, pos = biomarker_eval.load_truth(F)
up = [c.upper() for c in ch]
dd = [up.index(n) for n in NAMES]
print(f"preprocessed {F}")
print(f"  data shape (n_ep, n_ch, n_time) = {data.shape}")
print(f"  dropping {list(zip(NAMES, dd))}")

ref = pilot.bad_aware_reference(data, dd)            # benchmark scoring frame
print("  calling zuna_method.zuna_reconstruct ...", flush=True)
rc = zuna_method.zuna_reconstruct(ref, ch, pos, dd)
print(f"  recon shape = {rc.shape}")

# good-channel passthrough sanity: should be identical to ref (hard inpainting)
good = [i for i in range(data.shape[1]) if i not in dd]
passthrough_ok = np.allclose(rc[:, good, :], ref[:, good, :], atol=1e-4)
print(f"  good-channel hard-inpaint preserved: {passthrough_ok}")

print("\n  channel |  recon std |  truth std |  temporal r")
print("  --------|------------|------------|------------")
for n, i in zip(NAMES, dd):
    tch, rch = ref[:, i, :], rc[:, i, :]
    tc = tch - tch.mean(-1, keepdims=True)
    rcc = rch - rch.mean(-1, keepdims=True)
    num = (tc * rcc).sum(-1)
    den = np.sqrt((tc ** 2).sum(-1) * (rcc ** 2).sum(-1)) + 1e-20
    r = float(np.mean(num / den))
    print(f"  {n:<7} | {rch.std():9.3f}  | {tch.std():9.3f}  | {r:+.3f}")

print(f"\n  ALL dropped: recon std = {rc[:, dd, :].std():.3f} uV | "
      f"truth std = {ref[:, dd, :].std():.3f} uV")
print(f"  ALL dropped: mean temporal r = "
      f"{np.mean([np.corrcoef(ref[e, i], rc[e, i])[0,1] for e in range(data.shape[0]) for i in dd]):+.3f}")
