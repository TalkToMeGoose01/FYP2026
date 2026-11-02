#!/usr/bin/env python3
"""Authoritative word count for ZEIT4500 main.tex.

Rule (fixed 2026-09-11, used for every count from here on):
  COUNTED   body text from \section{Introduction} to \appendix,
            plus body floats at AIAA equivalence
            (single-column figure/table 200, two-column 450,
             large two-column table 700).
  EXCLUDED  title/author block, abstract, index terms, reference list
            (checklist), and everything from \appendix onward
            (declared on the front page).
Annex content is measured and reported separately, never added in.
"""
import re, sys

TEX = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/outputs/culled/main.tex'
raw = open(TEX, encoding='utf-8').read()
raw = re.sub(r'(?<!\\)%.*', '', raw)

# ---- split body / annex ------------------------------------------------
# NOTE: "\appendix" is NOT a prefix of "\appendices" (...ndix vs ...ndices),
# so match either spelling explicitly or the annex is silently missed.
_m = re.search(r'\\appendices\b|\\appendix\b', raw)
ai = _m.start() if _m else -1
if ai < 0:
    print("WARNING: no \\appendix or \\appendices found -- annex NOT separated")
body, annex = (raw[:ai], raw[ai:]) if ai > 0 else (raw, '')

# ---- body: drop everything before \section{Introduction} --------------
bi = body.find('\\section{Introduction}')
front, body = body[:bi], body[bi:]

# ---- body: drop the reference list -------------------------------------
for mark in ('\\begin{thebibliography}', '\\bibliography{'):
    k = body.find(mark)
    if k > 0:
        body = body[:k]
        break

def eat(t, env):
    out = []
    while True:
        m = re.search(r'\\begin\{' + env + r'\*?\}', t)
        if not m:
            return t, out
        star = t[m.start():m.end()].endswith('*}')
        e = re.search(r'\\end\{' + env + r'\*?\}', t[m.end():])
        out.append((star, t[m.end():m.end() + e.start()]))
        t = t[:m.start()] + t[m.end() + e.end():]

def words(t):
    t = re.sub(r'\\begin\{(equation|align|split)\*?\}.*?\\end\{\1\*?\}', ' X ', t, flags=re.S)
    t = re.sub(r'\\(label|ref|eqref|cite|includegraphics|begin|end)\*?(\[[^\]]*\])?\{[^}]*\}', ' ', t)
    t = re.sub(r'\$[^$]*\$', ' X ', t)
    t = re.sub(r'\\[a-zA-Z]+\*?', ' ', t)
    t = re.sub(r'[{}\\&~^_]', ' ', t)
    return len([w for w in t.split() if re.search(r'[A-Za-z0-9]', w)])

def floatcost(figs, tabs):
    tot, det = 0, []
    for star, b in figs:
        v = 450 if star else 200
        g = re.search(r'figures/([^},]*)', b)
        det.append(('figure*' if star else 'figure', g.group(1) if g else '?', v)); tot += v
    for star, b in tabs:
        rows = b.count('\\\\')
        v = (700 if rows > 12 else 450) if star else 200
        g = re.search(r'\\label\{([^}]*)\}', b)
        det.append(('table*' if star else 'table', g.group(1) if g else '?', v)); tot += v
    return tot, det

btxt, bfigs = eat(body, 'figure')
btxt, btabs = eat(btxt, 'table')
bfloat, bdet = floatcost(bfigs, btabs)

atxt, afigs = eat(annex, 'figure')
atxt, atabs = eat(atxt, 'table')
afloat, adet = floatcost(afigs, atabs)

secs = [(m.start(), m.group(1)) for m in re.finditer(r'\\section\*?\{([^}]*)\}', btxt)]
rows = []
for k, (pos, name) in enumerate(secs):
    end = secs[k + 1][0] if k + 1 < len(secs) else len(btxt)
    rows.append((name, words(btxt[pos:end])))
text = sum(w for _, w in rows)

CEIL = 16500
print(f"{'section':34s} {'words':>6s}")
for n, w in rows:
    print(f"  {n[:32]:32s} {w:6d}")
print(f"  {'-'*32} {'-'*6}")
print(f"  {'BODY TEXT':32s} {text:6d}\n")
for k, n, v in bdet:
    print(f"  {k:8s} {n[:40]:40s} {v:4d}")
print(f"  {'BODY FLOATS':32s} {bfloat:6d}   "
      f"({len(bfigs)} figures, {len(btabs)} tables)")
print(f"  {'='*32} {'='*6}")
print(f"  {'COUNTED TOTAL':32s} {text + bfloat:6d}")
print(f"  {'ceiling':32s} {CEIL:6d}")
print(f"  {'OVER BY':32s} {text + bfloat - CEIL:+6d}\n")
print(f"  not counted: annex text {words(atxt)}, annex floats {afloat} "
      f"({len(afigs)} fig, {len(atabs)} tab); front matter {words(front)}")
