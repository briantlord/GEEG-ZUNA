"""Shared spectral / spatial helpers for metric plug-ins.

Centralizes the primitives metrics are built from so each `m_<key>.py` stays tiny:
Welch PSD, band power (trapezoid), current-source-density (surface Laplacian), and channel picking.
"""
import numpy as np

SFREQ = 256


def trapezoid(y, x, axis=-1):
    """Integrate with the NumPy 2.x name while remaining compatible with NumPy 1.x."""
    integrate = getattr(np, 'trapezoid', None)
    if integrate is None:
        integrate = np.trapz
    return integrate(y, x, axis=axis)


def up(ch_names):
    return [c.upper() for c in ch_names]


def has(ch_names, *names):
    u = up(ch_names)
    return all(n.upper() in u for n in names)


def ix(ch_names, name):
    u = up(ch_names)
    return u.index(name.upper()) if name.upper() in u else None


def welch(data, sfreq=SFREQ, nperseg=None):
    """Welch PSD along the last axis. Returns (f, psd) with psd shaped like data on all but last axis."""
    from scipy.signal import welch as _w
    nt = data.shape[-1]
    nperseg = nperseg or min(1024, nt)
    return _w(data, fs=sfreq, nperseg=nperseg, axis=-1)


def mean_psd(data, sfreq=SFREQ, nperseg=None):
    """PSD averaged over epochs. data (n_ep, n_ch, n_t) -> (f, psd[n_ch, n_f])."""
    f, p = welch(data, sfreq, nperseg)
    return f, p.mean(0)


def bandpower(f, psd, lo, hi):
    """Trapezoid-integrated power in [lo, hi). Non-negative (psd >= 0)."""
    m = (f >= lo) & (f < hi)
    return trapezoid(psd[..., m], f[m], axis=-1)


def csd(data_uV, ch_names, sfreq=SFREQ):
    """Current source density (surface Laplacian) via MNE — reference-free.
    data_uV (n_ep, n_ch, n_t) microvolts -> csd (n_ep, n_ch, n_t)."""
    import mne
    info = mne.create_info(list(ch_names), sfreq, 'eeg')
    ep = mne.EpochsArray(np.asarray(data_uV) * 1e-6, info, verbose=False)
    ep.set_montage(mne.channels.make_standard_montage('standard_1005'),
                   match_case=False, on_missing='ignore')
    ep = mne.preprocessing.compute_current_source_density(ep, verbose=False)
    return ep.get_data()


def log_asymmetry(psd_right, psd_left, f, lo=8, hi=13):
    """Allen-style asymmetry: ln(bandpower right) - ln(bandpower left)."""
    pr = bandpower(f, psd_right, lo, hi)
    pl = bandpower(f, psd_left, lo, hi)
    return float(np.log(pr + 1e-20) - np.log(pl + 1e-20))

