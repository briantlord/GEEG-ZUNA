"""Theta/Beta Ratio (TBR) — metric plug-in.

ln(theta 4-8 Hz / beta 13-30 Hz) bandpower at Cz (primary) and Fz (secondary),
computed on the scalp average-referenced PSD (`C.mean_psd`) directly — NOT on CSD.
Unlike FAA (a reference-free surface-Laplacian asymmetry), TBR is clinically defined
and validated on scalp-referenced absolute power (Monastra 2001; Snyder & Hall 2006;
Arns 2013), so applying a surface Laplacian would distort the theta-vs-beta balance
and is unstable at the midline sites under the drop condition. The log ratio is stored
signed: because the beta window (17 Hz) is far wider than theta (4 Hz), integrated beta
often exceeds theta at rest, so values are commonly negative.
"""
from base import Metric, register
import common as C
import numpy as np


def compute(data, ch_names):
    out = {}
    f, psd = C.mean_psd(data)          # PSD, mean over epochs -> (n_ch, n_f); NO CSD (scalp-ref)
    eps = 1e-20                        # numerical log floor
    for name, key in (('CZ', 'tbr_cz'), ('FZ', 'tbr_fz')):
        i = C.ix(ch_names, name)       # case-insensitive index, or None if absent
        if i is None:
            out[key] = float('nan')    # missing target channel -> explicit NaN
            continue
        theta = C.bandpower(f, psd[i], 4, 8)    # trapezoid over [4, 8), uV^2/Hz, >= 0
        beta = C.bandpower(f, psd[i], 13, 30)   # trapezoid over [13, 30), uV^2/Hz, >= 0
        tbr = float(np.log(theta + eps) - np.log(beta + eps))   # == ln(theta/beta), eps-guarded
        out[key] = tbr if np.isfinite(tbr) else float('nan')    # never emit +/-inf
    return out


register(Metric(
    key='theta_beta',
    name='Theta/beta ratio',
    drop_channels=['CZ', 'FCZ', 'FZ', 'CPZ'],
    submetrics=['tbr_cz', 'tbr_fz'],
    compute=compute,
    reference='Monastra et al. 2001; Snyder & Hall 2006; Arns et al. 2013',
    notes='ln(theta 4-8 / beta 13-30) bandpower at Cz and Fz on scalp average-referenced PSD (no CSD).',
))
