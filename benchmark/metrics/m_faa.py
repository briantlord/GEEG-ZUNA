"""FAA — Frontal Alpha Asymmetry (reference metric plug-in).

ln(alpha power, right) - ln(alpha power, left) at mid-frontal F3/F4 (primary) and lateral F7/F8,
computed on current-source-density (surface-Laplacian) data, 8-13 Hz. Matches the original
benchmark's FAA so the modular framework reproduces the published numbers.
"""
from base import Metric, register
import common as C


def compute(data, ch_names):
    out = {}
    f, pc = C.mean_psd(C.csd(data, ch_names))          # PSD of CSD, mean over epochs -> (n_ch, n_f)
    u = C.up(ch_names)
    for L, R, key in [('F3', 'F4', 'faa'), ('F7', 'F8', 'faa_lat')]:
        if L in u and R in u:
            out[key] = C.log_asymmetry(pc[u.index(R)], pc[u.index(L)], f, 8, 13)
    return out


register(Metric(
    key='faa',
    name='Frontal alpha asymmetry',
    drop_channels=['F3', 'F4', 'F7', 'F8'],
    submetrics=['faa', 'faa_lat'],
    compute=compute,
    reference='Allen et al. 2004; Smith, Reznik, Stewart & Allen 2017',
    notes='ln(alpha F4)-ln(alpha F3) and F7/F8 on CSD (reference-free surface Laplacian).',
))
