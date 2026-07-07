"""
biomarker_eval.py  —  Evaluation A: biomarker & fingerprint preservation (light-mask regime)
============================================================================================
The scientifically meaningful test for ZUNA (per BENCHMARK_PROTOCOL.md §7.2-7.3): not waveform
fidelity (linear interpolation already wins, §0.2-E), but whether a reconstruction keeps the
biomarkers INSIDE their natural test-retest noise floor and preserves individual identity.

  reliability floor : within-subject SD of each biomarker across that subject's sessions (truth)
  reconstruction err: drop the biomarker-relevant channels (light mask), reconstruct, recompute
                      the biomarker, |recon - truth|.  A method "preserves" the biomarker if its
                      error < the floor (it perturbs it less than a real re-recording does).

Runs linear & spline on CPU now (baseline ZUNA must beat). ZUNA is method 'zuna' on the HPC GPU.
Usage:  python benchmark/biomarker_eval.py --subjects G001 G002 ...
"""
import sys, glob, os, csv, argparse, warnings, tempfile, numpy as np
warnings.simplefilter("ignore")
sys.path.insert(0, 'benchmark'); import pilot
try:
    import zuna_method
except Exception:
    zuna_method = None
from scipy.signal import welch

CACHE = os.path.join(tempfile.gettempdir(), 'tcache'); os.makedirs(CACHE, exist_ok=True)

# biomarker-relevant channels to drop (light mask = a handful from the dense cap)
DROP_SETS = {'FAA': ['F3', 'F4', 'F7', 'F8'],
             'IAF': ['O1', 'O2', 'OZ', 'POZ', 'PO3', 'PO4']}

def _slope(f, p):                                  # aperiodic 1/f slope, 2-30 Hz (below EMG)
    m = (f >= 2) & (f <= 30)
    return float(np.polyfit(np.log(f[m]), np.log(p[m] + 1e-20), 1)[0])

def biomarkers(data, ch):
    """IAF + FAA (CSD, Allen) from pilot, plus posterior log-alpha-power and 1/f slope."""
    bm = pilot.biomarkers(data, ch)                # iaf_hz, faa, faa_lat (FAA via CSD)
    up = [c.upper() for c in ch]
    f, p = welch(data, fs=256, nperseg=512, axis=-1); pm = p.mean(0)
    post = [up.index(c) for c in pilot.POSTERIOR if c in up]
    if post:
        m = (f >= 8) & (f < 13)
        bm['post_alpha'] = float(np.log(np.trapz(pm[post][:, m].mean(0), f[m]) + 1e-20))
        bm['slope_1f'] = float(np.mean([_slope(f, pm[i]) for i in post]))
    return bm

def load_truth(f):
    rec = os.path.basename(f); cf = f"{CACHE}/{rec}.npz"
    if os.path.exists(cf):
        z = np.load(cf, allow_pickle=True)
        return z['data'], [str(c) for c in z['ch_names']], z['pos']
    t = pilot.preprocess(f, n_epochs=64, emg=False)
    np.savez(cf, data=t['data'], ch_names=np.array(t['ch_names']), pos=t['pos'])
    return t['data'], t['ch_names'], t['pos']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', nargs='+', default=['G001'])
    ap.add_argument('--methods', nargs='+', default=['linear', 'spline'])   # + 'zuna' on GPU
    ap.add_argument('--out', default='/tmp/bioeval.csv')
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)): done.add(r['recording'])
    files = []
    for s in a.subjects: files += sorted(glob.glob(f'GEEG_Raw/{s}Day*.cnt'))
    cols = ['recording', 'subject', 'kind', 'drop_set', 'method', 'biomarker', 'truth', 'value', 'abs_err']
    new = not os.path.exists(a.out); fh = open(a.out, 'a', newline=''); w = csv.DictWriter(fh, fieldnames=cols)
    if new: w.writeheader()
    for f in files:
        rec = os.path.basename(f)
        if rec in done: continue
        meta = pilot.parse_meta(f); truth, ch, pos = load_truth(f)
        # (1) truth biomarkers on full montage (for the reliability floor)
        bt_full = biomarkers(pilot.bad_aware_reference(truth, []), ch)
        for bm, v in bt_full.items():
            w.writerow(dict(recording=rec, subject=meta['subject'], kind='truth', drop_set='-', method='-', biomarker=bm, truth=round(v, 5), value=round(v, 5), abs_err=0))
        # (2) reconstruction-induced biomarker error (light mask of biomarker-relevant channels)
        for dset, names in DROP_SETS.items():
            dd = pilot.idx_of(ch, names) if hasattr(pilot, 'idx_of') else [ [c.upper() for c in ch].index(n) for n in names if n in [c.upper() for c in ch] ]
            ref = pilot.bad_aware_reference(truth, dd); bt = biomarkers(ref, ch)
            for m in a.methods:
                try:
                    rc = (zuna_method.zuna_reconstruct(ref, ch, pos, dd) if m == 'zuna'
                          else pilot.reconstruct(m, ref, ch, pos, dd)); br = biomarkers(rc, ch)
                except Exception as e:
                    print(f"  [{m}] {e}"); continue
                for bm in bt:
                    if bm in br:
                        w.writerow(dict(recording=rec, subject=meta['subject'], kind='recon', drop_set=dset, method=m, biomarker=bm, truth=round(bt[bm], 5), value=round(br[bm], 5), abs_err=round(abs(br[bm] - bt[bm]), 5)))
        fh.flush(); print(f"done {rec}", flush=True)
    fh.close()

if __name__ == '__main__':
    main()
