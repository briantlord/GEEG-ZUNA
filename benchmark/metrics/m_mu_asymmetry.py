"""Mu asymmetry — Sensorimotor Mu (Rolandic) Asymmetry (metric plug-in).

ln(mu power, right C4) - ln(mu power, left C3) at central sensorimotor sites, plus the two
single-channel log band powers, computed on current-source-density (surface-Laplacian) data,
8-13 Hz. Spatial analog of FAA (m_faa.py) relocated F3/F4 -> C3/C4.
"""
import math

import numpy as np

from base import Metric, register
import common as C


def compute(data, ch_names):
    out = {}

    # Presence gate before any heavy compute (CSD): nothing to emit without a target channel.
    if not (C.has(ch_names, 'C3') or C.has(ch_names, 'C4')):
        return out

    # Reference frame — CSD then mean PSD (mirrors m_faa.py exactly).
    f, pc = C.mean_psd(C.csd(data, ch_names))          # PSD of CSD, mean over epochs -> pc[n_ch, n_f]
    u = C.up(ch_names)

    # mu_asym — right-left log asymmetry; requires both C3 and C4.
    if C.has(ch_names, 'C3', 'C4'):
        v = C.log_asymmetry(pc[u.index('C4')], pc[u.index('C3')], f, 8, 13)
        out['mu_asym'] = float(v) if math.isfinite(v) else float('nan')

    # mu_c3 — left-hemisphere absolute log band power; requires C3 only.
    if C.has(ch_names, 'C3'):
        bp = C.bandpower(f, pc[u.index('C3')], 8, 13)
        v = float(np.log(bp + 1e-20))
        out['mu_c3'] = v if math.isfinite(v) else float('nan')

    # mu_c4 — right-hemisphere absolute log band power; requires C4 only.
    if C.has(ch_names, 'C4'):
        bp = C.bandpower(f, pc[u.index('C4')], 8, 13)
        v = float(np.log(bp + 1e-20))
        out['mu_c4'] = v if math.isfinite(v) else float('nan')

    return out


register(Metric(
    key='mu_asymmetry',
    name='Sensorimotor mu asymmetry',
    drop_channels=['C3', 'C4', 'C1', 'C2'],
    submetrics=['mu_asym', 'mu_c3', 'mu_c4'],
    compute=compute,
    reference='Pfurtscheller & Lopes da Silva 1999; Pineda 2005; Allen et al. 2004 (CSD variant)',
    notes='ln(mu C4)-ln(mu C3) plus ln bandpower C3/C4 on CSD, 8-13 Hz; spatial analog of FAA.',
))
