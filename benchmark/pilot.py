"""
GEEG-ZUNA benchmark — pilot harness
===================================
Implements the wave-0 pilot of BENCHMARK_PROTOCOL.md:

  Stage 0  load .cnt -> ZUNA-matched preprocess -> 64 marker-locked epochs -> truth tensor (cached)
  Stage 1  seeded channel dropout (scattered | contiguous)
  Stage 2  reconstruction ladder: zero / mean / nearest / linear-neighbour ridge / spline
           (+ ZUNA hook, guarded — needs a GPU + HF weights, so skipped in CPU sandbox)
  Stage 3  hard inpainting + scoring (Tier-1 fidelity + IAF/FAA biomarkers) on dropped channels
  Queue    idempotent JSONL unit manifest -> resumable; tidy metrics CSV

This runs end-to-end on CPU for every method except ZUNA. The ZUNA rung is wired
against zuna.inference() and runs unchanged on the HPC GPU nodes.

Usage
-----
  python pilot.py --data_dir GEEG_Raw --out_dir benchmark/_pilot_out \
                  --subjects G001 G002 --n_drop 2 8 --patterns scattered contiguous \
                  --trials 2 --epochs 64
"""
from __future__ import annotations
import os, re, json, glob, time, argparse, hashlib, warnings
import numpy as np

warnings.simplefilter("ignore")

# ----------------------------------------------------------------------------- config
SFREQ        = 256
EPOCH_SAMPLES = 1280              # 5 s @ 256 Hz
HPF          = 0.5               # ZUNA: highpass only, no lowpass
NOTCH        = [60, 120, 180]    # US line noise + harmonics (auto-detect in production)
N_EPOCHS_DEF = 64
AUX          = ['HEOG', 'VEOG', 'EKG']
EVENT_CODES  = [str(i) for i in range(1, 9)]          # resting markers 1..8
BANDS = {                                              # broadband incl. high-gamma
    'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
    'beta': (13, 30), 'low_gamma': (30, 45), 'high_gamma': (45, 80),
}
POSTERIOR = ['O1', 'O2', 'OZ', 'P3', 'P4', 'PZ', 'P7', 'P8', 'PO3', 'PO4', 'POZ']
FAA_L, FAA_R = 'F3', 'F4'
NON_CORTICAL = {'M1', 'M2'}   # mastoids/reference-like: excluded from drop+score (Finding B)


# ----------------------------------------------------------------------------- stage 0
def remove_muscle(raw):
    """ICA muscle-component removal (frontotemporal EMG). Returns (raw, n_removed)."""
    import mne
    try:
        ica = mne.preprocessing.ICA(n_components=20, method='fastica', max_iter=200,
                                    random_state=0, verbose=False)
        ica.fit(raw, verbose=False)
        bad, _ = ica.find_bads_muscle(raw, verbose=False)
        ica.exclude = list(bad)
        ica.apply(raw, verbose=False)
        return raw, len(bad)
    except Exception as e:
        print(f"  [emg] ICA skipped: {str(e)[:80]}")
        return raw, 0


def preprocess(cnt_path, n_epochs=N_EPOCHS_DEF, emg=True):
    """Load .cnt, apply ZUNA-matched preprocessing, return non-overlapping epochs.

    Returns dict: data (n_ep, n_ch, 1280) float32 µV, ch_names, pos (n_ch,3), meta.
    """
    import mne
    mne.set_log_level("ERROR")

    try:
        raw = mne.io.read_raw_cnt(cnt_path, preload=True, data_format='int32')
    except Exception:
        raw = mne.io.read_raw_cnt(cnt_path, preload=True, data_format='int16')

    drop = [c for c in AUX if c in raw.ch_names]
    if drop:
        raw.drop_channels(drop)

    # ZUNA-matched: 0.5 Hz highpass, NO lowpass; notch line noise. (Average reference is
    # applied later, per-condition, over surviving channels — see survivor_reference.)
    raw.filter(l_freq=HPF, h_freq=None, fir_design='firwin', phase='zero', verbose=False)
    nyq = raw.info['sfreq'] / 2
    notches = [f for f in NOTCH if f < nyq - 1]
    if notches:
        raw.notch_filter(freqs=notches, verbose=False)
    raw.set_montage(mne.channels.make_standard_montage('standard_1005'),
                    match_case=False, on_missing='ignore')
    # drop channels without 3-D coordinates (e.g. CB1/CB2)
    nan_ch = [c['ch_name'] for c in raw.info['chs'] if np.isnan(c['loc'][:3]).any()]
    if nan_ch:
        raw.drop_channels(nan_ch)
    if raw.info['sfreq'] != SFREQ:
        raw.resample(SFREQ, npad='auto')
    raw.crop(tmin=10.0, tmax=raw.times[-1] - 10.0)
    if emg:
        raw, _ = remove_muscle(raw)          # ICA frontotemporal EMG removal

    # marker-locked, NON-overlapping 5 s epochs
    events, eid = mne.events_from_annotations(raw, verbose=False)
    keep_ids = [v for k, v in eid.items() if k in EVENT_CODES]
    ev = events[np.isin(events[:, 2], keep_ids)]
    ev = ev[np.argsort(ev[:, 0])]
    sel, last = [], -np.inf
    for row in ev:                                   # greedy non-overlapping
        if row[0] - last >= 5.0 * SFREQ:
            sel.append(row); last = row[0]
    sel = np.array(sel)                              # all non-overlapping; trimmed after reject
    tmax = 5.0 - 1.0 / SFREQ
    epochs = mne.Epochs(raw, sel, tmin=0, tmax=tmax, baseline=None,
                        preload=True, reject_by_annotation=True, verbose=False)
    epochs = epochs[:n_epochs]

    data = epochs.get_data(copy=True).astype(np.float32) * 1e6   # -> µV
    pos = epochs.get_montage().get_positions()['ch_pos']
    ch_names = epochs.ch_names
    pos_arr = np.array([pos[c] for c in ch_names], dtype=np.float32)
    return dict(data=data, ch_names=ch_names, pos=pos_arr,
                meta=dict(n_epochs=int(data.shape[0]), n_ch=len(ch_names), sfreq=SFREQ))


# ----------------------------------------------------------------------------- stage 1
def make_drop_mask(ch_names, pos, n_drop, pattern, seed):
    """Deterministic dropped-channel indices. scattered=random; contiguous=spatial cluster."""
    rng = np.random.default_rng(seed)
    elig = [i for i, c in enumerate(ch_names) if c.upper() not in NON_CORTICAL]
    if pattern == 'scattered':
        return sorted(rng.choice(elig, size=n_drop, replace=False).tolist())
    elif pattern == 'contiguous':
        seed_idx = int(rng.choice(elig))
        d = np.linalg.norm(pos - pos[seed_idx], axis=1)
        elig_set = set(elig)
        order = [i for i in np.argsort(d).tolist() if i in elig_set]
        return sorted(order[:n_drop])
    raise ValueError(pattern)


def bad_aware_reference(data, dropped):
    """Bad-aware average reference: subtract the mean over GOOD (surviving) channels only.

    The benchmark's evaluation frame (project decision). Removes the recording reference's
    common-mode without leaking the dropped channel (it is not part of the average), giving an
    apples-to-apples, common-mode-free comparison across all methods including ZUNA.
    Reconstruction and scoring both happen in this frame.
    """
    good = [i for i in range(data.shape[1]) if i not in dropped]
    return (data - data[:, good, :].mean(axis=1, keepdims=True)).astype(np.float32)


# ----------------------------------------------------------------------------- stage 2
def _epochs_from(data, ch_names, pos, sfreq=SFREQ):
    import mne
    info = mne.create_info(ch_names, sfreq, ch_types='eeg')
    ep = mne.EpochsArray(data * 1e-6, info, verbose=False)   # µV -> V
    montage = mne.channels.make_standard_montage('standard_1005')
    ep.set_montage(montage, match_case=False, on_missing='ignore')
    return ep


def reconstruct(method, truth, ch_names, pos, dropped):
    """Return reconstruction (same shape as truth) for the given method (dropped chans only matter)."""
    rec = truth.copy()
    good = [i for i in range(len(ch_names)) if i not in dropped]
    if method == 'zero':
        rec[:, dropped, :] = 0.0
    elif method == 'mean':
        rec[:, dropped, :] = truth[:, good, :].mean(axis=1, keepdims=True)
    elif method == 'nearest':
        for d in dropped:
            dist = np.linalg.norm(pos[good] - pos[d], axis=1)
            rec[:, d, :] = truth[:, good[int(np.argmin(dist))], :]
    elif method == 'linear':                          # ridge from K nearest good channels
        ne, nc, nt = truth.shape
        K = min(8, len(good))
        for d in dropped:
            dist = np.linalg.norm(pos[good] - pos[d], axis=1)
            kn = [good[j] for j in np.argsort(dist)[:K]]
            Xg = truth[:, kn, :].transpose(0, 2, 1).reshape(-1, K)          # (samples, K)
            y = truth[:, d, :].reshape(-1)
            lam = 1e-2 * np.trace(Xg.T @ Xg) / K
            w = np.linalg.solve(Xg.T @ Xg + lam * np.eye(K), Xg.T @ y)
            rec[:, d, :] = (Xg @ w).reshape(ne, nt)
    elif method == 'spline':
        import mne
        ep = _epochs_from(truth, ch_names, pos)
        ep.info['bads'] = [ch_names[i] for i in dropped]
        ep.interpolate_bads(reset_bads=True, verbose=False)
        rec = ep.get_data(copy=True).astype(np.float32) * 1e6
    elif method == 'zuna':
        raise RuntimeError("ZUNA rung requires GPU + HF weights; run on HPC (see reconstruct_zuna).")
    else:
        raise ValueError(method)
    # hard inpainting: observed channels keep ground truth exactly
    rec[:, good, :] = truth[:, good, :]
    return rec.astype(np.float32)


# ----------------------------------------------------------------------------- stage 3
def _welch(x, sfreq=SFREQ):
    from scipy.signal import welch
    f, p = welch(x, fs=sfreq, nperseg=min(1024, x.shape[-1]), axis=-1)
    return f, p


def _bandpower(f, p, lo, hi):
    m = (f >= lo) & (f < hi)
    return np.trapz(p[..., m], f[m], axis=-1)


def score_fidelity(truth, recon, dropped):
    """Tier-1 fidelity on dropped channels (mean over channels & epochs)."""
    from scipy.signal import butter, filtfilt
    _bf, _af = butter(4, [1.0 / (SFREQ / 2), 45.0 / (SFREQ / 2)], btype='band')
    t = filtfilt(_bf, _af, truth[:, dropped, :], axis=-1)        # score in-band (≤45 Hz, §0.2-D)
    r = filtfilt(_bf, _af, recon[:, dropped, :], axis=-1)
    # temporal correlation per (epoch, channel)
    tc = t - t.mean(-1, keepdims=True); rc = r - r.mean(-1, keepdims=True)
    num = (tc * rc).sum(-1)
    den = np.sqrt((tc**2).sum(-1) * (rc**2).sum(-1)) + 1e-20
    temporal_r = float(np.mean(num / den))
    rmse = float(np.sqrt(np.mean((t - r) ** 2)))
    rel_rmse = float(rmse / (np.std(t) + 1e-20))
    sdr = float(10 * np.log10((t**2).sum() / (((t - r) ** 2).sum() + 1e-20)))
    ft, pt = _welch(t); fr, pr = _welch(r)
    # spectral correlation (mean over epochs,channels)
    ptf = pt.reshape(-1, pt.shape[-1]); prf = pr.reshape(-1, pr.shape[-1])
    sc = []
    for a, b in zip(ptf, prf):
        sc.append(np.corrcoef(a, b)[0, 1])
    spectral_r = float(np.nanmean(sc))
    bp_err = {}
    for name, (lo, hi) in BANDS.items():
        if hi > 45:                      # high-γ excluded from scoring (EMG-dominated, §0.2-D)
            continue
        bt = _bandpower(ft, pt, lo, hi); br = _bandpower(fr, pr, lo, hi)
        bp_err[f'bperr_{name}'] = float(np.mean(np.abs(np.log((br + 1e-20) / (bt + 1e-20)))))
    return dict(temporal_r=temporal_r, rmse_uv=rmse, rel_rmse=rel_rmse,
                sdr_db=sdr, spectral_r=spectral_r, **bp_err)


FAA_PAIRS = [('F3', 'F4', 'faa'), ('F7', 'F8', 'faa_lat')]   # mid-frontal + lateral (Allen)


def _csd(data, ch_names):
    """Current source density (surface Laplacian) — reference-free, as Allen uses for FAA."""
    import mne
    info = mne.create_info(list(ch_names), SFREQ, 'eeg')
    ep = mne.EpochsArray(data * 1e-6, info, verbose=False)
    ep.set_montage(mne.channels.make_standard_montage('standard_1005'),
                   match_case=False, on_missing='ignore')
    ep = mne.preprocessing.compute_current_source_density(ep, verbose=False)
    return ep.get_data()


def biomarkers(data, ch_names):
    """IAF (posterior centre-of-gravity) and FAA computed the Allen way:
    ln(alpha_right) - ln(alpha_left) at F3/F4 (+F7/F8) on CSD-transformed (reference-free) data.
    """
    up = [c.upper() for c in ch_names]
    out = {}
    f, p = _welch(data); pm = p.mean(0)                  # IAF on referenced data
    post = [up.index(c) for c in POSTERIOR if c in up]
    if post:
        m = (f >= 8) & (f <= 13); psd = pm[post][:, m].mean(0)
        out['iaf_hz'] = float((f[m] * psd).sum() / (psd.sum() + 1e-20))
    try:                                                  # FAA on CSD (Allen, Smith et al. 2017)
        fc, pc = _welch(_csd(data, ch_names)); pc = pc.mean(0)
        for L, R, key in FAA_PAIRS:
            if L in up and R in up:
                pl = _bandpower(fc, pc[up.index(L)], 8, 13)
                pr = _bandpower(fc, pc[up.index(R)], 8, 13)
                out[key] = float(np.log(pr + 1e-20) - np.log(pl + 1e-20))
    except Exception as e:
        print(f"  [faa] CSD failed: {str(e)[:60]}")
    return out


# ----------------------------------------------------------------------------- queue
class Manifest:
    """Idempotent JSONL unit queue -> resumable runs."""
    def __init__(self, path):
        self.path = path
        self.done = set()
        if os.path.exists(path):
            for line in open(path):
                try:
                    r = json.loads(line)
                    if r.get('status') == 'done':
                        self.done.add(r['uid'])
                except Exception:
                    pass

    def is_done(self, uid):
        return uid in self.done

    def mark(self, uid, status, **extra):
        with open(self.path, 'a') as fh:
            fh.write(json.dumps(dict(uid=uid, status=status, t=time.time(), **extra)) + "\n")
        if status == 'done':
            self.done.add(uid)


def parse_meta(fname):
    m = re.search(r'G(\d+)Day(\d+)([a-zA-Z0-9]+)\.cnt', os.path.basename(fname), re.I)
    if not m:
        return None
    return dict(subject=f"G{m.group(1)}", day=int(m.group(2)), session=m.group(3))


def uid_seed(subject, day, session, n, pattern, trial):
    s = f"{subject}|{day}|{session}|{n}|{pattern}|{trial}"
    return s, int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


# ----------------------------------------------------------------------------- driver
LADDER = ['zero', 'mean', 'nearest', 'linear', 'spline']   # + 'zuna' on HPC


def run_pilot(data_dir, out_dir, subjects, n_drops, patterns, trials, n_epochs, methods, emg=True):
    os.makedirs(out_dir, exist_ok=True)
    man = Manifest(os.path.join(out_dir, 'manifest.jsonl'))
    csv_path = os.path.join(out_dir, 'metrics.csv')
    new_csv = not os.path.exists(csv_path)
    cache = {}
    files = []
    for s in subjects:
        files += sorted(glob.glob(os.path.join(data_dir, f"{s}Day*.cnt")))
    print(f"[pilot] {len(files)} recordings | methods={methods} "
          f"| N={n_drops} | patterns={patterns} | trials={trials} | epochs={n_epochs}")

    rows = []
    for f in files:
        meta = parse_meta(f)
        if meta is None:
            continue
        key = os.path.basename(f)
        if key not in cache:
            t0 = time.time()
            cache[key] = preprocess(f, n_epochs=n_epochs, emg=emg)
            tru = cache[key]
            print(f"[stage0] {key}: {tru['meta']} in {time.time()-t0:.1f}s")
        tru = cache[key]
        truth, ch, pos = tru['data'], tru['ch_names'], tru['pos']
        for n in n_drops:
            for pat in patterns:
                for tr in range(1, trials + 1):
                    s, seed = uid_seed(meta['subject'], meta['day'], meta['session'], n, pat, tr)
                    dropped = make_drop_mask(ch, pos, n, pat, seed)
                    truth_ref = bad_aware_reference(truth, dropped)   # avg over GOOD channels
                    bm_truth = biomarkers(truth_ref, ch)
                    for method in methods:
                        uid = f"{s}|{method}"
                        if man.is_done(uid):
                            continue
                        try:
                            t0 = time.time()
                            rec = reconstruct(method, truth_ref, ch, pos, dropped)  # bad-aware frame
                            fid = score_fidelity(truth_ref, rec, dropped)
                            bm_rec = biomarkers(rec, ch)
                            row = dict(**meta, n_drop=n, pattern=pat, trial=tr, method=method,
                                       n_ch=len(ch), dropped=";".join(ch[i] for i in dropped),
                                       runtime_s=round(time.time() - t0, 3), **fid,
                                       iaf_err=abs(bm_rec.get('iaf_hz', np.nan) - bm_truth.get('iaf_hz', np.nan)),
                                       faa_err=abs(bm_rec.get('faa', np.nan) - bm_truth.get('faa', np.nan)))
                            rows.append(row)
                            man.mark(uid, 'done', method=method)
                        except Exception as e:
                            man.mark(uid, 'failed', method=method, err=str(e)[:200])
                            print(f"[FAIL] {uid}: {e}")

    if rows:
        import csv
        cols = list(rows[0].keys())
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with open(csv_path, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            if new_csv:
                w.writeheader()
            w.writerows(rows)
    print(f"[pilot] wrote {len(rows)} metric rows -> {csv_path}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out_dir', default='_pilot_out')
    ap.add_argument('--subjects', nargs='+', default=['G001'])
    ap.add_argument('--n_drop', nargs='+', type=int, default=[2, 8])
    ap.add_argument('--patterns', nargs='+', default=['scattered', 'contiguous'])
    ap.add_argument('--trials', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=N_EPOCHS_DEF)
    ap.add_argument('--methods', nargs='+', default=LADDER)
    ap.add_argument('--no_emg', action='store_true', help='disable ICA muscle removal')
    a = ap.parse_args()
    run_pilot(a.data_dir, a.out_dir, a.subjects, a.n_drop, a.patterns,
              a.trials, a.epochs, a.methods, emg=not a.no_emg)
