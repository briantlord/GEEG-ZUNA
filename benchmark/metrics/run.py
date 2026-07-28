"""Generalized biomarker-preservation runner over ALL registered metric plug-ins.

Drops each metric's own channels, reconstructs (linear / spline / zuna), recomputes the metric,
and logs |recon - truth|. Metrics that share a drop set reuse a single reconstruction per method
(so the expensive ZUNA pass runs once per drop set, not once per metric). Resumable by recording.
Same reliability-floor design as the original `biomarker_eval.py`, but metric-agnostic.

Usage:
  python benchmark/metrics/run.py --subjects G001 G002 G003 G004 G005 \
         --methods linear spline zuna --metrics faa theta_beta ... --out results/metric_eval.csv
  (default --metrics = every registered plug-in; --methods default = linear spline)
"""
import sys, os, glob, csv, argparse, warnings, tempfile, importlib
import numpy as np
warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
for p in (HERE, BENCH):
    if p not in sys.path:
        sys.path.insert(0, p)

import pilot
try:
    import zuna_method
except Exception:
    zuna_method = None
import base

CACHE = os.path.join(tempfile.gettempdir(), 'tcache')
os.makedirs(CACHE, exist_ok=True)


def discover(keys=None):
    # When specific keys are requested, import only their modules (convention: key -> m_<key>.py).
    # This keeps a run isolated from sibling plug-ins that may be half-written (parallel development).
    if keys:
        for k in keys:
            mod = os.path.join(HERE, f"m_{k}.py")
            if os.path.exists(mod):
                importlib.import_module(f"m_{k}")
    else:
        for f in sorted(glob.glob(os.path.join(HERE, "m_*.py"))):
            importlib.import_module(os.path.splitext(os.path.basename(f))[0])
    return {k: v for k, v in base.REGISTRY.items() if (keys is None or k in keys)}


def load_truth(f):
    rec = os.path.basename(f)
    cf = f"{CACHE}/{rec}.npz"
    if os.path.exists(cf):
        z = np.load(cf, allow_pickle=True)
        return z['data'], [str(c) for c in z['ch_names']], z['pos']
    t = pilot.preprocess(f, n_epochs=64, emg=False)
    np.savez(cf, data=t['data'], ch_names=np.array(t['ch_names']), pos=t['pos'])
    return t['data'], t['ch_names'], t['pos']


def drop_indices(ch, names):
    u = [c.upper() for c in ch]
    return sorted(u.index(n.upper()) for n in names if n.upper() in u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', nargs='+', default=['G001'])
    ap.add_argument('--methods', nargs='+', default=['linear', 'spline'])
    ap.add_argument('--metrics', nargs='+', default=None, help='metric keys (default: all registered)')
    ap.add_argument('--out', default='results/metric_eval.csv')
    a = ap.parse_args()

    reg = discover(a.metrics)
    if not reg:
        print("no metrics registered/selected"); return
    print(f"[metrics] {list(reg)} | methods={a.methods} | subjects={a.subjects}")

    # group metrics by drop set so each reconstruction is computed once
    groups = {}   # frozenset(upper channels) -> (label, [Metric])
    for m in reg.values():
        fs = frozenset(c.upper() for c in m.drop_channels)
        groups.setdefault(fs, ('+'.join(sorted(fs)), []))[1].append(m)

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add(r['recording'])
    cols = ['recording', 'subject', 'kind', 'drop_set', 'method', 'metric', 'submetric',
            'truth', 'value', 'abs_err']
    new = not os.path.exists(a.out)
    fh = open(a.out, 'a', newline='')
    w = csv.DictWriter(fh, fieldnames=cols)
    if new:
        w.writeheader()

    files = []
    for s in a.subjects:
        files += sorted(glob.glob(f'GEEG_Raw/{s}Day*.cnt'))
    for f in files:
        rec = os.path.basename(f)
        if rec in done:
            continue
        meta = pilot.parse_meta(f)
        truth, ch, pos = load_truth(f)
        # (1) full-montage truth -> reliability floor
        ref0 = pilot.surviving_average_reference(truth, [])
        for m in reg.values():
            try:
                vals = m.compute(ref0, ch)
            except Exception as e:
                print(f"  [truth:{m.key}] {rec}: {e}"); continue
            for sk, v in vals.items():
                w.writerow(dict(recording=rec, subject=meta['subject'], kind='truth', drop_set='-',
                                method='-', metric=m.key, submetric=sk,
                                truth=round(float(v), 6), value=round(float(v), 6), abs_err=0))
        # (2) drop each group's channels, reconstruct once per method, recompute its metrics
        for fs, (label, metrics) in groups.items():
            dd = drop_indices(ch, list(fs))
            if not dd:
                continue
            ref = pilot.surviving_average_reference(truth, dd)
            bt = {}
            for m in metrics:
                try:
                    bt[m.key] = m.compute(ref, ch)
                except Exception as e:
                    print(f"  [truthframe:{m.key}] {rec}: {e}"); bt[m.key] = {}
            for method in a.methods:
                try:
                    rc = (zuna_method.zuna_reconstruct(ref, ch, pos, dd) if method == 'zuna'
                          else pilot.reconstruct(method, ref, ch, pos, dd))
                except Exception as e:
                    print(f"  [{method}] {rec} {label}: {e}"); continue
                for m in metrics:
                    try:
                        br = m.compute(rc, ch)
                    except Exception as e:
                        print(f"  [{method}:{m.key}] {rec}: {e}"); continue
                    for sk in m.submetrics:
                        if sk in bt.get(m.key, {}) and sk in br:
                            w.writerow(dict(recording=rec, subject=meta['subject'], kind='recon',
                                            drop_set=label, method=method, metric=m.key, submetric=sk,
                                            truth=round(float(bt[m.key][sk]), 6),
                                            value=round(float(br[sk]), 6),
                                            abs_err=round(abs(float(br[sk]) - float(bt[m.key][sk])), 6)))
        fh.flush()
        print(f"done {rec}", flush=True)
    fh.close()


if __name__ == '__main__':
    main()
