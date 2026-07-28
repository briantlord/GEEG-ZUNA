"""specparam_peaks — Parameterized posterior spectrum (aperiodic + alpha peak).

Fits a dependency-free specparam-style model to the average-referenced power spectrum
averaged over the posterior channels O1/O2/Oz/POz/PO3/PO4. Reports the 1/f aperiodic
component (exponent, offset) and the alpha oscillatory peak (center frequency, power,
bandwidth). The aperiodic slope indexes cortical excitation/inhibition balance and the
alpha peak the individual alpha frequency; both are read from the raw non-negative PSD
in the average-reference frame (NO CSD — offset/power are magnitude quantities a surface
Laplacian would distort, and there is no left/right ratio to cancel the transform).

Fit range 2-40 Hz, peak search/exclusion band 7-14 Hz.
"""
from base import Metric, register
import common as C
import numpy as np

POSTERIOR = ['O1', 'O2', 'OZ', 'POZ', 'PO3', 'PO4']   # UPPER-CASE read set
SUBMETRICS = ['aperiodic_exponent', 'aperiodic_offset', 'alpha_cf', 'alpha_pw', 'alpha_bw']


def compute(data, ch_names):
    # 1. Present posterior rows, case-insensitively (intersection with the montage).
    u = C.up(ch_names)
    rows = [u.index(p) for p in POSTERIOR if p in u]

    # 2. No posterior channel present -> not computable.
    if not rows:
        return {k: float('nan') for k in SUBMETRICS}

    # 3. Welch PSD, epoch-averaged, average-reference frame (uV^2/Hz, non-negative).
    f, psd = C.mean_psd(data)

    # 4. Extract posterior rows; drop any non-finite row (defensive).
    post = psd[rows]
    finite = np.isfinite(post).all(axis=1)
    post = post[finite]
    if post.shape[0] == 0:
        return {k: float('nan') for k in SUBMETRICS}

    # 5. Posterior-average spectrum (raw, non-negative — no log/subtract/normalise here).
    psd_post = post.mean(axis=0)

    # 6. Parameterize aperiodic + alpha peak via the shared helper; return unchanged.
    return C.aperiodic_and_peak(f, psd_post,
                                fit_range=(2, 40), peak_band=(7, 14), peak_label='alpha')


register(Metric(
    key='specparam_peaks',
    name='Parameterized spectrum (aperiodic + alpha peak)',
    drop_channels=['O1', 'O2', 'OZ', 'POZ', 'PO3', 'PO4'],
    submetrics=SUBMETRICS,
    compute=compute,
    reference='Donoghue et al. 2020 (specparam/FOOOF); Gao, Peterson & Voytek 2017 (E:I balance); '
              'Klimesch 1999 (individual alpha frequency).',
    notes='Posterior aperiodic exponent+offset and alpha peak (cf/pw/bw) from a dependency-free '
          'specparam fit on the average-referenced PSD averaged over O1/O2/Oz/POz/PO3/PO4 '
          '(fit 2-40 Hz, peak 7-14 Hz). No CSD. Shares the IAF drop set.',
))
