"""
aggregate.py — Evaluation A analysis: biomarker preservation vs the same-day test-retest floor.

The scientific question (BENCHMARK_PROTOCOL §7.2-7.3): a reconstruction "preserves" a biomarker
if dropping that biomarker's own channels and reconstructing perturbs it LESS than a real same-day
re-recording does. So we compare, per biomarker:

    same-day floor[bm]   = mean over (subject, day) of |B(Rest1) - B(Rest2)|   on kind=='truth' rows
                           (within-day test-retest reliability of the biomarker itself)
    method error[bm, m]  = mean abs_err on kind=='recon' rows, for that biomarker's OWN drop set
                           (faa/faa_lat <- drop_set 'FAA' ; iaf_hz/post_alpha/slope_1f <- 'IAF')

A method passes (biomarker preserved) when method error < floor.

Usage:  python benchmark/aggregate.py --csv zuna_eval_5subj.csv
"""
import argparse, csv, re, math
from collections import defaultdict

# Each biomarker is scored after dropping ITS OWN channels (light mask); the recon rows for the
# other drop set carry abs_err=0 for it (untouched) and must NOT be averaged in.
BM_DROP = {'faa': 'FAA', 'faa_lat': 'FAA', 'iaf_hz': 'IAF', 'post_alpha': 'IAF', 'slope_1f': 'IAF'}
BM_ORDER = ['faa', 'faa_lat', 'iaf_hz', 'post_alpha', 'slope_1f']
BM_LABEL = {'faa': 'FAA (F3/F4)', 'faa_lat': 'FAA-lat (F7/F8)', 'iaf_hz': 'IAF (Hz)',
            'post_alpha': 'post-alpha', 'slope_1f': '1/f slope'}
METHODS = ['linear', 'spline', 'zuna']
REC_RE = re.compile(r'G(\d+)Day(\d+)Rest(\d+)', re.I)


def parse_rec(rec):
    m = REC_RE.search(rec)
    return None if not m else dict(subject='G' + m.group(1), day=int(m.group(2)), rest=int(m.group(3)))


def fnum(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='zuna_eval_5subj.csv')
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))

    # ---- same-day floor: |Rest1 - Rest2| of the full-montage truth biomarker, per (subject, day) ----
    truth = defaultdict(dict)               # (subject, day, biomarker) -> {rest: value}
    for r in rows:
        if r['kind'] != 'truth':
            continue
        info = parse_rec(r['recording'])
        if info:
            truth[(info['subject'], info['day'], r['biomarker'])][info['rest']] = fnum(r['value'])

    floor, floor_n = {}, {}
    for bm in BM_ORDER:
        diffs = [abs(v[1] - v[2]) for (s, d, b), v in truth.items()
                 if b == bm and 1 in v and 2 in v and not math.isnan(v[1]) and not math.isnan(v[2])]
        floor[bm] = (sum(diffs) / len(diffs)) if diffs else float('nan')
        floor_n[bm] = len(diffs)

    # ---- method error: mean abs_err on recon rows for the biomarker's own drop set ----
    err = defaultdict(list)                 # (biomarker, method) -> [abs_err, ...]
    for r in rows:
        if r['kind'] != 'recon':
            continue
        bm = r['biomarker']
        if bm in BM_DROP and r['drop_set'] == BM_DROP[bm]:
            err[(bm, r['method'])].append(fnum(r['abs_err']))

    def mean_err(bm, m):
        v = [x for x in err[(bm, m)] if not math.isnan(x)]
        return (sum(v) / len(v)) if v else float('nan')

    methods = [m for m in METHODS if any((bm, m) in err for bm in BM_ORDER)]
    n_rec = {(bm, m): len(err[(bm, m)]) for bm in BM_ORDER for m in methods}

    # ---- table ----
    w = 11
    print(f"\nEvaluation A - biomarker preservation vs same-day test-retest floor  ({a.csv})")
    print(f"  recon-row counts per method/biomarker: {dict(n_rec)}\n")
    head = f"{'biomarker':<17}{'floor':>9}  " + "".join(f"{m:>{w}}" for m in methods)
    print(head)
    print('-' * len(head))
    for bm in BM_ORDER:
        fl = floor[bm]
        line = f"{BM_LABEL[bm]:<17}{fl:>9.3f}  "
        for m in methods:
            e = mean_err(bm, m)
            mark = ''
            if not math.isnan(e) and not math.isnan(fl):
                mark = '  ok' if e < fl else ' OVER'
            line += f"{e:>{w-5}.3f}{mark}"
        print(line + f"   (n={floor_n[bm]} days)")
    print('-' * len(head))
    print("ok = error below floor (preserved within test-retest) ; OVER = exceeds floor (not preserved)\n")

    # ---- headline verdict on FAA (the biomarker that matters) ----
    for bm in ['faa', 'faa_lat']:
        fl = floor[bm]
        print(f"{BM_LABEL[bm]}: floor={fl:.3f}")
        for m in methods:
            e = mean_err(bm, m)
            verdict = 'PRESERVED' if (not math.isnan(e) and e < fl) else 'NOT preserved'
            print(f"    {m:<8} error={e:.3f}  -> {verdict}")
        print()


if __name__ == '__main__':
    main()
