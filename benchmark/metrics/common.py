"""Shared spectral / spatial helpers for metric plug-ins.

Centralizes the primitives metrics are built from so each `m_<key>.py` stays tiny:
Welch PSD, band power (trapezoid), current-source-density (surface Laplacian), channel picking,
and a small dependency-free specparam-style aperiodic+peak fit.
"""
import numpy as np

SFREQ = 256


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
    return np.trapz(psd[..., m], f[m], axis=-1)


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


def aperiodic_and_peak(f, psd, fit_range=(2, 40), peak_band=(7, 14), peak_label='alpha'):
    """Dependency-free specparam-style fit (fooof/specparam not installed).

    1. Aperiodic: robust least-squares fit of log10(psd) ~ offset - exponent*log10(f) over
       `fit_range`, EXCLUDING `peak_band` so an oscillatory bump does not bias the 1/f slope.
    2. Peak: on the log10 residual (psd minus aperiodic fit), find the max within `peak_band`
       -> center frequency (cf), power above aperiodic (pw, log10), and FWHM bandwidth (bw).

    Returns dict: aperiodic_exponent, aperiodic_offset, <label>_cf, <label>_pw, <label>_bw.
    """
    f = np.asarray(f, float); psd = np.asarray(psd, float)
    m = (f >= fit_range[0]) & (f <= fit_range[1]) & (f > 0)
    lf = np.log10(f[m]); lp = np.log10(psd[m] + 1e-30)
    fitmask = ~((f[m] >= peak_band[0]) & (f[m] <= peak_band[1]))
    if fitmask.sum() >= 2:
        A = np.vstack([np.ones(fitmask.sum()), lf[fitmask]]).T
        (offset, slope), *_ = np.linalg.lstsq(A, lp[fitmask], rcond=None)
    else:
        offset, slope = float(lp.mean()), 0.0
    exponent = -float(slope)
    resid = lp - (offset + slope * lf)
    out = {'aperiodic_exponent': exponent, 'aperiodic_offset': float(offset)}
    pk = (f[m] >= peak_band[0]) & (f[m] <= peak_band[1])
    if pk.any() and np.isfinite(resid).any():
        rr = np.where(pk, resid, -np.inf)
        i = int(np.argmax(rr))
        fr = f[m]
        cf, pw = float(fr[i]), float(resid[i])
        half = pw / 2.0
        li = i
        while li > 0 and resid[li] > half:
            li -= 1
        ri = i
        while ri < len(resid) - 1 and resid[ri] > half:
            ri += 1
        bw = float(fr[ri] - fr[li])
        out.update({f'{peak_label}_cf': cf, f'{peak_label}_pw': pw, f'{peak_label}_bw': bw})
    else:
        out.update({f'{peak_label}_cf': float('nan'), f'{peak_label}_pw': float('nan'),
                    f'{peak_label}_bw': float('nan')})
    return out
