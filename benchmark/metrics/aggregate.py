"""Aggregate the modular metric run: same-day test-retest floor vs per-method error.

For every (metric, submetric):
  same-day floor = mean over (subject, day) of |value(Rest1) - value(Rest2)| on kind=='truth' rows
  method error   = mean abs_err on kind=='recon' rows for that submetric
A method PRESERVES a submetric when its error < floor (perturbs it less than a re-recording does).

Usage:  python benchmark/metrics/aggregate.py --csv results/metric_eval.csv
"""
import argparse, csv, re, math
from collections import defaultdict

REC = re.compile(r'G(\d+)Day(\d+)Rest(\d+)', re.I)


def parse(rec):
    m = REC.search(rec)
    return None if not m else ('G' + m.group(1), int(m.group(2)), int(m.group(3)))


def fnum(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results/metric_eval.csv')
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))

    # same-day floor
    truth = defaultdict(dict)   # (metric, submetric, subject, day) -> {rest: value}
    for r in rows:
        if r['kind'] != 'truth':
            continue
        i = parse(r['recording'])
        if i:
            truth[(r['metric'], r['submetric'], i[0], i[1])][i[2]] = fnum(r['value'])
    floor = defaultdict(list)
    for (met, sub, s, d), v in truth.items():
        if 1 in v and 2 in v and not math.isnan(v[1]) and not math.isnan(v[2]):
            floor[(met, sub)].append(abs(v[1] - v[2]))

    # per-method reconstruction error
    err = defaultdict(list)     # (metric, submetric, method) -> [abs_err]
    for r in rows:
        if r['kind'] != 'recon':
            continue
        err[(r['metric'], r['submetric'], r['method'])].append(fnum(r['abs_err']))

    methods = sorted({r['method'] for r in rows if r['kind'] == 'recon'})
    order, seen = [], set()
    for r in rows:
        k = (r['metric'], r['submetric'])
        if k not in seen:
            seen.add(k); order.append(k)

    w = 12
    print(f"\nMetric preservation vs same-day test-retest floor  ({a.csv})\n")
    head = f"{'metric / submetric':<30}{'floor':>9}  " + "".join(f"{m:>{w}}" for m in methods)
    print(head)
    print('-' * len(head))
    cur = None
    for (met, sub) in order:
        if met != cur:
            cur = met
        fl = mean(floor[(met, sub)])
        line = f"{met + ' / ' + sub:<30}{fl:>9.3f}  "
        for m in methods:
            e = mean(err[(met, sub, m)])
            mark = '' if (math.isnan(e) or math.isnan(fl)) else ('  ok' if e < fl else ' OVER')
            line += f"{e:>{w - 5}.3f}{mark}"
        n = len(floor[(met, sub)])
        print(line + f"   (n={n} days)")
    print('-' * len(head))
    print("ok = error below floor (preserved within test-retest) ; OVER = exceeds floor\n")


if __name__ == '__main__':
    main()
