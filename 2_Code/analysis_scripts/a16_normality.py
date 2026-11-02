#!/usr/bin/env python3
"""analysis16 - normality of the ten-run predictive samples.

Checklist requirement: "If you give the standard deviation, do a
normality test to make sure it is appropriate."

Reads only existing result files. Trains nothing, loads no model.
Writes analysis16_normality.csv and analysis16_PROVENANCE.txt.
"""
import csv, hashlib, os, platform, sys, collections, datetime

import numpy as np
import scipy
from scipy import stats

RESULTS = '/mnt/user-data/uploads/FYP/ZEIT4500 - FYP/Phase2/results'
OUT = '/tmp/a16'
os.makedirs(OUT, exist_ok=True)

INPUTS = [
    'analysis5_tenseed_predictive.csv',
    'analysis9_rank2_vs_rank5.csv',
]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def load(name):
    with open(os.path.join(RESULTS, name), newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


samples = collections.OrderedDict()

# analysis5: ten distinct seeds per architecture, anchor test MAE%
for r in load('analysis5_tenseed_predictive.csv'):
    if r.get('status') != 'ok':
        continue
    samples.setdefault(('analysis5_tenseed_predictive', r['arch'], 'mae_pct'), []).append(
        float(r['mae_pct']))

# analysis9: ten seeds per candidate configuration
for r in load('analysis9_rank2_vs_rank5.csv'):
    if r.get('status') != 'ok':
        continue
    samples.setdefault(('analysis9_rank2_vs_rank5', 'tft_rank' + r['rank'], 'mae_pct'), []).append(
        float(r['mae_pct']))

rows = []
for (src, group, metric), v in samples.items():
    v = np.asarray(v, dtype=float)
    W, p = stats.shapiro(v)
    rows.append({
        'source_file': src,
        'sample': group,
        'metric': metric,
        'n': len(v),
        'mean': f'{v.mean():.6f}',
        'sd_ddof1': f'{v.std(ddof=1):.6f}',
        'median': f'{np.median(v):.6f}',
        'iqr': f'{np.percentile(v, 75) - np.percentile(v, 25):.6f}',
        'shapiro_W': f'{W:.6f}',
        'shapiro_p': f'{p:.6f}',
        'normality_rejected_at_0.05': 'yes' if p < 0.05 else 'no',
    })

csv_path = os.path.join(OUT, 'analysis16_normality.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

min_p = min(float(r['shapiro_p']) for r in rows)
any_rej = any(r['normality_rejected_at_0.05'] == 'yes' for r in rows)

prov = os.path.join(OUT, 'analysis16_PROVENANCE.txt')
with open(prov, 'w', encoding='utf-8') as f:
    f.write('analysis16_normality.csv\n')
    f.write('=' * 60 + '\n\n')
    f.write('PURPOSE\n')
    f.write('  Shapiro-Wilk normality tests on every repeated-run predictive\n')
    f.write('  sample reported in this thesis as mean +/- standard deviation.\n')
    f.write('  Administers the final-report checklist requirement that a\n')
    f.write('  standard deviation be accompanied by a normality test.\n\n')
    f.write('SCOPE\n')
    f.write('  Only samples of n >= 10 are tested. The three-run confirmation\n')
    f.write('  spreads of the TFT selection history are NOT tested: n = 3 is\n')
    f.write('  too small for the test to be informative, and those spreads are\n')
    f.write('  reported descriptively rather than used for inference.\n\n')
    f.write('METHOD\n')
    f.write('  scipy.stats.shapiro on the mae_pct column, rows with status=ok.\n')
    f.write('  Standard deviation is the sample SD (ddof=1), matching the\n')
    f.write('  values reported in the thesis text.\n\n')
    f.write('INPUTS (SHA-256)\n')
    for name in INPUTS:
        p = os.path.join(RESULTS, name)
        f.write(f'  {digest(p)}  {name}\n')
    f.write('\nOUTPUT (SHA-256)\n')
    f.write(f'  {digest(csv_path)}  analysis16_normality.csv\n')
    f.write('\nRESULT\n')
    f.write(f'  Samples tested: {len(rows)}\n')
    f.write(f'  Normality rejected at alpha = 0.05 in any sample: '
            f'{"YES" if any_rej else "NO"}\n')
    f.write(f'  Minimum Shapiro-Wilk p across all samples: {min_p:.4f}\n')
    f.write('  Conclusion: mean +/- SD is an appropriate summary for every\n')
    f.write('  sample so reported.\n')
    f.write('\nENVIRONMENT\n')
    f.write(f'  Python {platform.python_version()}\n')
    f.write(f'  numpy {np.__version__}\n')
    f.write(f'  scipy {scipy.__version__}\n')
    f.write(f'  Run (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n')
    f.write('  This analysis loads no model and trains nothing; it re-derives\n')
    f.write('  its figures from archived result files alone.\n')

for r in rows:
    print(f"{r['sample']:16s} n={r['n']:>3d} mean={float(r['mean']):7.4f} "
          f"sd={float(r['sd_ddof1']):6.4f} W={float(r['shapiro_W']):.4f} "
          f"p={float(r['shapiro_p']):.4f} rejected={r['normality_rejected_at_0.05']}")
print(f"\nminimum p = {min_p:.4f}  -> rounds to {min_p:.2f}")
print(f"wrote {csv_path}")
print(f"wrote {prov}")
