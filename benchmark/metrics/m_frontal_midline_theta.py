"""FMt — Frontal Midline Theta (absolute frontal-midline theta level).

Two submetrics on the epoch-averaged, average-reference scalp PSD (NO current-source-density —
FMt is a power/level metric, not an asymmetry, so the surface Laplacian is inappropriate here):

  fmt_fz  : ln theta(4-8 Hz) power at Fz  = ln(P_theta(Fz) + eps)          [ln µV²]
  fmt_rel : Fz-vs-posterior theta log-ratio (topographic-specificity index)
            = ln(P_theta(Fz) + eps) - ln(mean posterior P_theta + eps)     [dimensionless]

Posterior reference set = {O1, O2, OZ, PZ, POZ}, averaged over whichever of those sites are
present. Guarded logs (+1e-20) avoid -inf on zero power; any non-finite result is coerced to
float('nan'). compute() never raises.
"""
from base import Metric, register
import common as C
import numpy as np

EPS = 1e-20
POST_LABELS = ('O1', 'O2', 'OZ', 'PZ', 'POZ')
THETA_LO, THETA_HI = 4, 8


def compute(data, ch_names):
    # 1. Epoch-averaged Welch PSD directly on average-reference scalp data (no CSD).
    f, psd = C.mean_psd(data)                 # f:(n_f,), psd:(n_ch, n_f)

    # 2. Locate Fz (guard first); no numerator -> both submetrics nan.
    fz = C.ix(ch_names, 'FZ')
    if fz is None:
        return {'fmt_fz': float('nan'), 'fmt_rel': float('nan')}

    # 3. fmt_fz = guarded log theta power at Fz.
    p_fz = C.bandpower(f, psd[fz], THETA_LO, THETA_HI)
    fmt_fz = float(np.log(p_fz + EPS))

    # 4. Posterior theta mean over present sites only (partial sets OK).
    post_ix = [C.ix(ch_names, n) for n in POST_LABELS]
    post_ix = [i for i in post_ix if i is not None]

    # 5. fmt_rel = Fz-vs-posterior theta log-ratio (mean of powers, single guarded log).
    if post_ix:
        p_post = float(np.mean([C.bandpower(f, psd[i], THETA_LO, THETA_HI) for i in post_ix]))
        fmt_rel = float(fmt_fz - np.log(p_post + EPS))
    else:
        fmt_rel = float('nan')

    # 6. Coerce any non-finite result to explicit float('nan').
    out = {'fmt_fz': fmt_fz, 'fmt_rel': fmt_rel}
    return {k: (v if np.isfinite(v) else float('nan')) for k, v in out.items()}


register(Metric(
    key='frontal_midline_theta',
    name='Frontal midline theta',
    drop_channels=['FZ', 'FCZ', 'F1', 'F2'],
    submetrics=['fmt_fz', 'fmt_rel'],
    compute=compute,
    reference='Cavanagh & Frank 2014; Onton, Delorme & Makeig 2005; Gevins et al. 1997',
    notes='ln theta(4-8 Hz) power at Fz (fmt_fz) and Fz-vs-posterior '
          'theta log-ratio (fmt_rel), on average-ref scalp PSD (no CSD).',
))
