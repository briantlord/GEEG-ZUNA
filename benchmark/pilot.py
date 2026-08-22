"""
GEEG-ZUNA benchmark — pilot harness
===================================
Implements the wave-0 pilot of BENCHMARK_PROTOCOL.md:

  Stage 0  load .cnt -> ZUNA-matched preprocess -> 64 marker-locked epochs -> truth tensor (cached)
  Stage 1  seeded channel dropout (scattered | contiguous)
  Stage 2  reconstruction ladder: zero / mean / nearest / linear-neighbour ridge / spline /
           mne (minimum-norm source-space round trip, see mne_source_method.py)
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

try:
    from .protocol_v2 import PREPROCESSING_SHA256, PREPROCESSING_SPEC, PROTOCOL_ID
except ImportError:  # Script execution from benchmark/
    from protocol_v2 import PREPROCESSING_SHA256, PREPROCESSING_SPEC, PROTOCOL_ID

warnings.simplefilter("ignore")

# ----------------------------------------------------------------------------- config
SFREQ         = PREPROCESSING_SPEC['target_sfreq_hz']
EPOCH_SAMPLES = int(PREPROCESSING_SPEC['epoch_seconds'] * SFREQ)
HPF           = PREPROCESSING_SPEC['bandpass_hz'][0]
LPF           = PREPROCESSING_SPEC['bandpass_hz'][1]
N_EPOCHS_DEF  = PREPROCESSING_SPEC['target_epochs']
MIN_CLEAN_EPOCHS = PREPROCESSING_SPEC['minimum_clean_epochs']
AUX           = list(PREPROCESSING_SPEC['aux_channels'])
EVENT_CODES   = list(PREPROCESSING_SPEC['event_codes'])
BANDS = {                                              # broadband incl. high-gamma
    'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
    'beta': (13, 30), 'low_gamma': (30, 45), 'high_gamma': (45, 80),
}
POSTERIOR = ['O1', 'O2', 'OZ', 'P3', 'P4', 'PZ', 'P7', 'P8', 'PO3', 'PO4', 'POZ']
FAA_L, FAA_R = 'F3', 'F4'
NON_CORTICAL = {'M1', 'M2'}   # mastoids/reference-like: excluded from drop+score (Finding B)


# ----------------------------------------------------------------------------- stage 0
def remove_artifacts(raw, strict=True):
    """Deterministic ocular+muscle ICA; retain EOG channels until scoring."""
    import mne
    try:
        ica = mne.preprocessing.ICA(
            n_components=PREPROCESSING_SPEC['ica_components'], method='fastica',
            max_iter=200, random_state=PREPROCESSING_SPEC['ica_random_state'], verbose=False)
        ica.fit(raw, picks="eeg", verbose=False)
        ocular = set()
        ocular_scores = {}
        for channel in (name for name in ("HEOG", "VEOG") if name in raw.ch_names):
            bad, scores = ica.find_bads_eog(
                raw, ch_name=channel,
                threshold=PREPROCESSING_SPEC["ocular_threshold"],
                measure=PREPROCESSING_SPEC["ocular_measure"], verbose=False,
            )
            ocular.update(int(index) for index in bad)
            ocular_scores[channel] = np.asarray(scores, dtype=float).tolist()
        muscle, muscle_scores = ica.find_bads_muscle(
            raw, threshold=PREPROCESSING_SPEC["muscle_threshold"],
            l_freq=PREPROCESSING_SPEC["muscle_band_hz"][0],
            h_freq=PREPROCESSING_SPEC["muscle_band_hz"][1], verbose=False,
        )
        ica.exclude = sorted(ocular | {int(index) for index in muscle})
        component_topographies = np.asarray(ica.get_components(), dtype=float).T
        component_variance = np.asarray(ica.pca_explained_variance_, dtype=float)
        ica.apply(raw, verbose=False)
        return raw, {
            "ocular_components": sorted(ocular),
            "muscle_components": sorted(int(index) for index in muscle),
            "ocular_scores": ocular_scores,
            "muscle_scores": np.asarray(muscle_scores, dtype=float).tolist(),
            "excluded_components": list(ica.exclude),
            "excluded_fraction": float(len(ica.exclude) / ica.n_components_),
            "n_components": int(ica.n_components_),
            "ica_channel_names": list(ica.ch_names),
            "component_topographies": component_topographies.tolist(),
            "pca_explained_variance": component_variance.tolist(),
            "muscle_threshold": PREPROCESSING_SPEC["muscle_threshold"],
            "muscle_band_hz": PREPROCESSING_SPEC["muscle_band_hz"],
            "ocular_threshold": PREPROCESSING_SPEC["ocular_threshold"],
            "ocular_measure": PREPROCESSING_SPEC["ocular_measure"],
        }
    except Exception as e:
        if strict:
            raise RuntimeError(f"Required ocular/muscle ICA cleaning failed: {e}") from e
        print(f"  [ica] artifact cleaning explicitly disabled after failure: {str(e)[:80]}")
        return raw, {
            "ocular_components": [], "muscle_components": [], "excluded_components": [],
            "ocular_scores": {}, "muscle_scores": [], "excluded_fraction": 0.0,
            "n_components": 0, "ica_channel_names": [],
            "component_topographies": [], "pca_explained_variance": [],
            "muscle_threshold": PREPROCESSING_SPEC["muscle_threshold"],
            "muscle_band_hz": PREPROCESSING_SPEC["muscle_band_hz"],
            "ocular_threshold": PREPROCESSING_SPEC["ocular_threshold"],
            "ocular_measure": PREPROCESSING_SPEC["ocular_measure"],
        }


def _session_channel_qc(raw):
    """QC the cropped analysis interval, not excluded raw-file edges."""
    rows, failures = [], []
    for index, name in enumerate(raw.ch_names):
        values_uv = raw.get_data(picks=[index])[0] * 1e6
        finite = bool(np.isfinite(values_uv).all())
        if finite:
            std_uv = float(np.std(values_uv))
            minimum, maximum = float(np.min(values_uv)), float(np.max(values_uv))
            rail_fraction = float(max(np.mean(values_uv == minimum), np.mean(values_uv == maximum)))
        else:
            std_uv, rail_fraction = float('nan'), 1.0
        reasons = []
        if not finite:
            reasons.append('nonfinite')
        if finite and std_uv < PREPROCESSING_SPEC['session_flat_std_uv']:
            reasons.append('flat')
        if rail_fraction > PREPROCESSING_SPEC['session_rail_fraction_max']:
            reasons.append('railed')
        max_abs_uv = float(np.max(np.abs(values_uv))) if finite else float('nan')
        max_jump_uv = float(np.max(np.abs(np.diff(values_uv)))) if finite and values_uv.size > 1 else 0.0
        if finite and max_abs_uv > PREPROCESSING_SPEC['analysis_abs_max_uv']:
            reasons.append('implausible_absolute_scale')
        if finite and max_jump_uv > PREPROCESSING_SPEC['analysis_max_sample_jump_uv']:
            reasons.append('discontinuity')
        rows.append(dict(channel=name, std_uv=std_uv, rail_fraction=rail_fraction,
                         max_abs_uv=max_abs_uv, max_sample_jump_uv=max_jump_uv,
                         passed=not reasons, reasons=reasons))
        if reasons:
            failures.append(f"{name} ({'+'.join(reasons)})")
    return rows, failures


def _raw_tail_qc(raw, eeg_names, seconds):
    """Describe the raw tail separately; it is excluded before analysis."""
    n_tail = max(1, int(round(seconds * raw.info['sfreq'])))
    start = max(0, raw.n_times - n_tail)
    values_uv = raw.get_data(picks=eeg_names, start=start, stop=raw.n_times) * 1e6
    maximum = float(np.max(np.abs(values_uv)))
    return {
        "seconds": float(seconds),
        "start_sample": int(start),
        "stop_sample": int(raw.n_times),
        "max_abs_uv": maximum,
        "warning_threshold_uv": float(PREPROCESSING_SPEC['raw_tail_abs_warn_uv']),
        "warning": bool(maximum > PREPROCESSING_SPEC['raw_tail_abs_warn_uv']),
    }


def preprocess(cnt_path, n_epochs=N_EPOCHS_DEF, emg=True,
               minimum_clean_epochs=MIN_CLEAN_EPOCHS):
    """Load CNT and enforce corrected-v2 preprocessing and recording QC.

    Returns dict: data (n_ep, n_ch, 1280) float32 µV, ch_names, pos (n_ch,3), meta.
    """
    import mne
    mne.set_log_level("ERROR")

    raw = mne.io.read_raw_cnt(
        cnt_path, preload=True, data_format=PREPROCESSING_SPEC['cnt_data_format'])
    original_sfreq = float(raw.info['sfreq'])

    auxiliary_types = {}
    for channel in ("HEOG", "VEOG"):
        if channel in raw.ch_names:
            auxiliary_types[channel] = "eog"
    if "EKG" in raw.ch_names:
        auxiliary_types["EKG"] = "ecg"
    if auxiliary_types:
        raw.set_channel_types(auxiliary_types, verbose=False)
    non_montage = [c for c in PREPROCESSING_SPEC['non_montage_channels'] if c in raw.ch_names]
    if non_montage:
        raw.drop_channels(non_montage)

    eeg_names = [raw.ch_names[index] for index in mne.pick_types(raw.info, eeg=True, exclude=[])]
    raw_tail_qc = _raw_tail_qc(raw, eeg_names, PREPROCESSING_SPEC['edge_crop_seconds'])

    raw.set_montage(mne.channels.make_standard_montage('standard_1005'),
                    match_case=False, on_missing='ignore')
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    unresolved = [raw.info['chs'][index]['ch_name'] for index in eeg_picks
                  if np.isnan(raw.info['chs'][index]['loc'][:3]).any()]
    expected_channels = PREPROCESSING_SPEC['expected_eeg_channels']
    if unresolved or len(eeg_picks) != expected_channels:
        raise RuntimeError(
            f"Montage/QC failure: expected {expected_channels} positioned EEG channels; "
            f"found {len(eeg_picks)}, unresolved={unresolved}")

    # One and only one bandpass. Average reference happens later, after dropout.
    raw.filter(l_freq=HPF, h_freq=LPF, fir_design='firwin', phase='zero', verbose=False)
    if raw.info['sfreq'] != SFREQ:
        raw.resample(SFREQ, npad='auto')
    edge_crop = PREPROCESSING_SPEC['edge_crop_seconds']
    raw.crop(tmin=edge_crop, tmax=raw.times[-1] - edge_crop)
    analysis_eeg = raw.copy().pick(picks="eeg")
    channel_qc, channel_failures = _session_channel_qc(analysis_eeg)
    if channel_failures:
        raise RuntimeError("Cropped analysis-interval channel QC failed: " + ", ".join(channel_failures))

    artifact_components = {
        "ocular_components": [], "muscle_components": [], "excluded_components": [],
        "ocular_scores": {}, "muscle_scores": [], "excluded_fraction": 0.0,
        "n_components": 0, "ica_channel_names": [],
        "component_topographies": [], "pca_explained_variance": [],
        "muscle_threshold": PREPROCESSING_SPEC["muscle_threshold"],
        "muscle_band_hz": PREPROCESSING_SPEC["muscle_band_hz"],
        "ocular_threshold": PREPROCESSING_SPEC["ocular_threshold"],
        "ocular_measure": PREPROCESSING_SPEC["ocular_measure"],
    }
    if emg:
        raw, artifact_components = remove_artifacts(raw, strict=True)
    elif PREPROCESSING_SPEC['emg_required']:
        warnings.warn(
            "EMG cleaning explicitly disabled: output is development-only and cannot enter "
            "the corrected benchmark.", RuntimeWarning)

    drop_aux = [channel for channel in AUX if channel in raw.ch_names]
    if drop_aux:
        raw.drop_channels(drop_aux)

    # marker-locked, NON-overlapping 5 s epochs
    events, eid = mne.events_from_annotations(raw, verbose=False)
    keep_ids = [v for k, v in eid.items() if k in EVENT_CODES]
    ev = events[np.isin(events[:, 2], keep_ids)]
    ev = ev[np.argsort(ev[:, 0])]
    sel, selected_event_indices, last = [], [], -np.inf
    for event_index, row in enumerate(ev):            # greedy non-overlapping
        if row[0] - last >= 5.0 * SFREQ:
            sel.append(row); selected_event_indices.append(event_index); last = row[0]
    sel = np.asarray(sel, dtype=int).reshape(-1, 3)
    if not len(sel):
        raise RuntimeError("No marker-locked 5-second epoch candidates were found")
    tmax = 5.0 - 1.0 / SFREQ
    epochs = mne.Epochs(raw, sel, tmin=0, tmax=tmax, baseline=None,
                        preload=True, reject_by_annotation=True, verbose=False)
    candidate_data = epochs.get_data(copy=True).astype(np.float32) * 1e6
    peak_to_peak = np.ptp(candidate_data, axis=-1)
    clean = np.all(
        (peak_to_peak >= PREPROCESSING_SPEC['epoch_peak_to_peak_flat_uv'])
        & (peak_to_peak <= PREPROCESSING_SPEC['epoch_peak_to_peak_max_uv']),
        axis=1,
    )
    clean_indices = np.flatnonzero(clean)
    data = candidate_data[clean_indices[:n_epochs]]
    if data.shape[0] < minimum_clean_epochs:
        raise RuntimeError(
            f"Recording QC failed: retained {data.shape[0]} clean epochs; "
            f"minimum is {minimum_clean_epochs}")
    if data.shape[2] != EPOCH_SAMPLES or not np.isfinite(data).all():
        raise RuntimeError(f"Invalid Stage-0 tensor shape/values: {data.shape}")
    pos = epochs.get_montage().get_positions()['ch_pos']
    ch_names = epochs.ch_names
    pos_arr = np.array([pos[c] for c in ch_names], dtype=np.float32)

    accepted_selected = {int(selected_index): accepted_index
                         for accepted_index, selected_index in enumerate(epochs.selection)}
    final_accepted = {int(index): order for order, index in enumerate(clean_indices[:n_epochs])}
    selected_lookup = {event_index: selected_index
                       for selected_index, event_index in enumerate(selected_event_indices)}
    event_descriptions = {value: key for key, value in eid.items()}
    event_qc = []
    for event_index, row in enumerate(ev):
        selected_index = selected_lookup.get(event_index)
        accepted_index = None if selected_index is None else accepted_selected.get(selected_index)
        final_order = None if accepted_index is None else final_accepted.get(accepted_index)
        record = {
            "event_index": int(event_index),
            "raw_sample_estimate": int(round(float(row[0]) * original_sfreq / SFREQ)),
            "resampled_sample": int(row[0]),
            "onset_seconds": float((row[0] - raw.first_samp) / SFREQ),
            "event_code": int(row[2]),
            "annotation_description": event_descriptions.get(int(row[2]), "UNKNOWN"),
            "nonoverlap_selected": selected_index is not None,
            "annotation_accepted": accepted_index is not None,
            "amplitude_flat_accepted": None if accepted_index is None else bool(clean[accepted_index]),
            "peak_to_peak_min_uv": None if accepted_index is None else float(np.min(peak_to_peak[accepted_index])),
            "peak_to_peak_max_uv": None if accepted_index is None else float(np.max(peak_to_peak[accepted_index])),
            "final_selected_order": None if final_order is None else int(final_order),
        }
        event_qc.append(record)
    return dict(data=data, ch_names=ch_names, pos=pos_arr,
                meta=dict(
                    protocol_id=PROTOCOL_ID,
                    preprocessing_sha256=PREPROCESSING_SHA256,
                    n_epochs=int(data.shape[0]), n_ch=len(ch_names), sfreq=SFREQ,
                    bandpass_hz=[HPF, LPF], emg_cleaning=bool(emg),
                    artifact_components=artifact_components,
                    epoch_candidates=int(len(sel)),
                    epochs_after_annotations=int(len(candidate_data)),
                    epochs_rejected_amplitude_or_flat=int(np.sum(~clean)),
                    minimum_clean_epochs=int(minimum_clean_epochs),
                    analysis_interval_channel_qc=channel_qc,
                    raw_tail_qc=raw_tail_qc,
                    event_qc=event_qc,
                ))


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


def surviving_average_reference(data, dropped, ch_names=None):
    """Average reference over surviving, contract-authorized contributors only.

    The benchmark's evaluation frame (project decision). Subtracting the mean over only the
    surviving channels removes the recording reference's common-mode without letting the dropped
    channel leak into its own reference (it is excluded from the average), giving an
    apples-to-apples, common-mode-free comparison across all methods including ZUNA.
    Reconstruction and scoring both happen in this frame.
    """
    if ch_names is not None and len(ch_names) != data.shape[1]:
        raise ValueError("ch_names must align with the channel axis")
    excluded = {name.upper() for name in PREPROCESSING_SPEC['reference_excluded_channels']}
    good = [
        i for i in range(data.shape[1])
        if i not in dropped and (ch_names is None or ch_names[i].upper() not in excluded)
    ]
    if not good:
        raise ValueError("drop set leaves no authorized average-reference contributors")
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
    elif method == 'mne':                             # minimum-norm source-space round trip
        # Lazy + path-guarded so this resolves whether pilot is launched from the project root
        # or from benchmark/ (same reason zuna_method fixes sys.path at import).
        import sys as _sys
        _d = os.path.dirname(os.path.abspath(__file__))
        if _d not in _sys.path:
            _sys.path.insert(0, _d)
        from mne_source_method import mne_source_reconstruct
        rec = mne_source_reconstruct(truth, ch_names, pos, dropped)
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
    integrate = getattr(np, 'trapezoid', None)
    if integrate is None:  # NumPy 1.x
        integrate = np.trapz
    return integrate(p[..., m], f[m], axis=-1)


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
LADDER = ['zero', 'mean', 'nearest', 'linear', 'spline', 'mne']   # + 'zuna' on HPC


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
                    truth_ref = surviving_average_reference(truth, dropped, ch_names)
                    bm_truth = biomarkers(truth_ref, ch)
                    for method in methods:
                        uid = f"{s}|{method}"
                        if man.is_done(uid):
                            continue
                        try:
                            t0 = time.time()
                            rec = reconstruct(method, truth_ref, ch, pos, dropped)  # surviving-channel reference frame
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
    raise SystemExit(
        "BLOCKED LEGACY CLI: use metrics/run.py with an immutable --run-manifest"
    )
