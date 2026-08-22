"""specparam_peaks — Parameterized posterior spectrum (aperiodic + alpha peak).

Fits the official ``specparam`` SpectralModel to the average-referenced power spectrum
averaged over the posterior channels O1/O2/Oz/POz/PO3/PO4. Reports the 1/f aperiodic
component (exponent, offset) and the alpha oscillatory peak (center frequency, power,
bandwidth). The aperiodic slope indexes cortical excitation/inhibition balance and the
alpha peak the individual alpha frequency; both are read from the raw non-negative PSD
in the average-reference frame (NO CSD — offset/power are magnitude quantities a surface
Laplacian would distort, and there is no left/right ratio to cancel the transform).

Fit range 2-40 Hz; the strongest model-detected peak in 7-14 Hz is reported as alpha.
"""
from base import Metric, register
import common as C
import numpy as np
from importlib.metadata import version
from specparam import SpectralModel

POSTERIOR = ['O1', 'O2', 'OZ', 'POZ', 'PO3', 'PO4']   # UPPER-CASE read set
SUBMETRICS = ['aperiodic_exponent', 'aperiodic_offset', 'alpha_cf', 'alpha_pw', 'alpha_bw']
SPECPARAM_VERSION = version('specparam')
if SPECPARAM_VERSION != '2.0.0rc7':
    raise RuntimeError(
        f"Phase 3 requires pinned specparam==2.0.0rc7, found {SPECPARAM_VERSION}"
    )


def evaluate(data, ch_names):
    # 1. Present posterior rows, case-insensitively (intersection with the montage).
    u = C.up(ch_names)
    rows = [u.index(p) for p in POSTERIOR if p in u]

    # 2. No posterior channel present -> not computable.
    if not rows:
        return {k: float('nan') for k in SUBMETRICS}, {
            'fit_status': 'missing_posterior_channels', 'posterior_channel_count': 0
        }

    # 3. Welch PSD, epoch-averaged, average-reference frame (uV^2/Hz, non-negative).
    f, psd = C.mean_psd(data)

    # 4. Extract posterior rows; drop any non-finite row (defensive).
    post = psd[rows]
    finite = np.isfinite(post).all(axis=1)
    post = post[finite]
    if post.shape[0] == 0:
        return {k: float('nan') for k in SUBMETRICS}, {
            'fit_status': 'no_finite_posterior_spectra', 'posterior_channel_count': 0
        }

    # 5. Posterior-average spectrum (raw, non-negative — no log/subtract/normalise here).
    psd_post = post.mean(axis=0)
    if not np.isfinite(psd_post).all() or np.any(psd_post <= 0):
        return {key: float('nan') for key in SUBMETRICS}, {
            'fit_status': 'invalid_posterior_psd',
            'posterior_channel_count': int(post.shape[0]),
        }

    # 6. Official specparam fit. Inputs must remain in linear frequency/power space.
    model = SpectralModel(
        aperiodic_mode='fixed', periodic_mode='gaussian',
        peak_width_limits=(0.5, 12.0), max_n_peaks=6,
        min_peak_height=0.0, peak_threshold=2.0, verbose=False,
    )
    model.fit(f, psd_post, (2, 40))
    aperiodic = np.asarray(model.get_params('aperiodic'), dtype=float).reshape(-1)
    if aperiodic.size != 2:
        raise RuntimeError(f"unexpected fixed-aperiodic parameters: {aperiodic!r}")

    peaks = np.asarray(model.get_params('peak'), dtype=float)
    if peaks.ndim == 1:
        peaks = peaks.reshape(1, -1)
    alpha = peaks[
        (peaks.shape[1] == 3)
        & np.isfinite(peaks).all(axis=1)
        & (peaks[:, 0] >= 7.0)
        & (peaks[:, 0] <= 14.0)
    ] if peaks.size else np.empty((0, 3), dtype=float)
    if not len(alpha):
        peak_values = (float('nan'),) * 3
    else:
        peak_values = tuple(float(value) for value in alpha[np.argmax(alpha[:, 1])])
    values = {
        'aperiodic_exponent': float(aperiodic[1]),
        'aperiodic_offset': float(aperiodic[0]),
        'alpha_cf': peak_values[0],
        'alpha_pw': peak_values[1],
        'alpha_bw': peak_values[2],
    }
    diagnostics = {
        'fit_status': 'success' if len(alpha) else 'missing_alpha_peak',
        'posterior_channel_count': int(post.shape[0]),
        'r_squared': float(model.get_metrics('gof', 'rsquared')),
        'mean_absolute_error': float(model.get_metrics('error', 'mae')),
        'detected_peak_count': int(peaks.shape[0]) if peaks.size else 0,
        'alpha_peak_count': int(len(alpha)),
    }
    return values, diagnostics


def compute(data, ch_names):
    return evaluate(data, ch_names)[0]


METRIC = register(Metric(
    key='specparam_peaks',
    name='Parameterized spectrum (aperiodic + alpha peak)',
    drop_channels=['O1', 'O2', 'OZ', 'POZ', 'PO3', 'PO4'],
    submetrics=SUBMETRICS,
    compute=compute,
    reference='Donoghue et al. 2020 (specparam/FOOOF); Gao, Peterson & Voytek 2017 (E:I balance); '
              'Klimesch 1999 (individual alpha frequency).',
    notes='Posterior aperiodic exponent+offset and strongest detected alpha peak (cf/pw/bw) from '
          'the official specparam SpectralModel on the average-referenced PSD averaged over O1/O2/Oz/POz/PO3/PO4 '
          '(fit 2-40 Hz, peak 7-14 Hz). No CSD. Shares the IAF drop set.',
    implementation=f'specparam=={SPECPARAM_VERSION}; SpectralModel fixed/gaussian',
    evaluate=evaluate,
))
