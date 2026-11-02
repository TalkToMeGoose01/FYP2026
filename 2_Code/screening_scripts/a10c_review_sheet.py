#!/usr/bin/env python3
"""
a10c_review_sheet.py -- build the author's review sheet for the flagged
Search A and B records (the energy-domain searches the gap rests on).
Abstracts are joined from the original exports by DOI, falling back to
normalised title. Writes screening/REVIEW_AB_flagged.xlsx. Read-only on
every input.
"""
import sys, re, csv, hashlib, datetime, pathlib
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import a10_screening_triage as a10

EXP = pathlib.Path(sys.argv[1])
HERE = pathlib.Path(__file__).resolve().parent / "screening"
OUT = HERE / "REVIEW_AB_flagged.xlsx"

def nd(s): return a10.norm_doi(s) if isinstance(s, str) else ""
def nt(s): return a10.norm_title(s) if isinstance(s, str) else ""

abs_by_doi, abs_by_title = {}, {}
for s in ("A", "B"):
    for fname, db in zip(a10.EXPORTS[s], ("Scopus", "WoS")):
        d = a10.read_export(EXP / fname, db)
        for _, r in d.iterrows():
            a = r["abstract"] if isinstance(r["abstract"], str) else ""
            kw = "; ".join(str(r[k]) for k in ("kw1", "kw2") if isinstance(r[k], str))
            rec = (a.strip(), kw.strip())
            if nd(r["doi"]) and len(a) > len(abs_by_doi.get(nd(r["doi"]), ("",))[0]):
                abs_by_doi[nd(r["doi"])] = rec
            t = nt(r["title"])
            if t and len(a) > len(abs_by_title.get(t, ("",))[0]):
                abs_by_title[t] = rec

rows, seen = [], {}
for s in ("A", "B"):
    for r in csv.DictReader(open(HERE / f"screening_{s}.csv", encoding="utf-8")):
        if r["triage_flag"] != "True":
            continue
        key = nd(r["doi"]) or nt(r["title"])
        if key in seen:
            seen[key]["search"] += "+" + s
            continue
        a, kw = abs_by_doi.get(nd(r["doi"])) or abs_by_title.get(nt(r["title"])) or ("", "")
        rec = dict(search=s, tier=r["tier"].replace(".0", ""), title=r["title"],
                   year=r["year"].replace(".0", ""), source=r["source"], doi=r["doi"],
                   code=r["decision"], reason=r["reason"], abstract=a or "(no abstract in export)",
                   keywords=kw, bib=r["bib_key"])
        seen[key] = rec; rows.append(rec)

order = {"IN": 0, "E1": 1, "E2": 1, "E3": 1, "E4": 1, "E5": 1, "E6": 1, "E7": 1, "E8": 1, "E9-block": 2}
rows.sort(key=lambda x: (x["tier"] != "1", order.get(x["code"], 1), x["search"], x["title"].lower()))

F = "Arial"; thin = Side(style="thin", color="BFBFBF")
hdr = Font(name=F, bold=True, color="FFFFFF"); hfill = PatternFill("solid", fgColor="404040")
yel = PatternFill("solid", fgColor="FFFF00"); body = Font(name=F, size=10)
wrap = Alignment(wrap_text=True, vertical="top")

wb = Workbook()
lg = wb.active; lg.title = "Legend"
legend = [
 ("Review sheet: flagged records from Searches A and B", None),
 ("", None),
 ("What this is", "The 48 distinct records the triage flagged in Searches A and B (67 flags, 19 records retrieved by both; Search C flagged none). A wrong exclusion here is the only way the research gap could fail, so these are the records to check yourself."),
 ("What to do", "Read each abstract on the Review sheet. Fill the two yellow columns only: 'Your decision' (pick a code from the dropdown) and 'Your note' (optional). Leave the other columns alone."),
 ("Agreement column", "Fills itself: 'agree' when your code matches the first-pass code, 'CHANGED' when it differs. Any CHANGED row in Search A or B coded IN would need the gap re-examined."),
 ("Order", "Tier 1 first (strong vocabulary match), then tier 2. Within a tier, IN first, then individually coded exclusions, then the E9-block group."),
 ("Source of first-pass codes", "Drafted by Claude on 3 September 2026 from title, abstract and keywords (screening/CODING_OUTCOME.md). Not yet reviewed by the author."),
 ("Source of abstracts", "Original Scopus and Web of Science exports of 11 August 2026, matched by DOI, then by normalised title. SHA-256 of each export matches screening/PROVENANCE.txt."),
 ("", None),
 ("Codes", None),
 ("IN", "Meets both halves of the criterion: same specification trained more than once under different random initialisations AND attribution output compared across those runs"),
 ("E1", "Seed mentioned only as a fixed-value reproducibility statement"),
 ("E2", "Reproducibility refers to data or code availability"),
 ("E3", "Retraining refers to model updating or concept drift"),
 ("E4", "Robustness tested as input perturbation, distribution shift or ablation fidelity"),
 ("E5", "Seed variance reported for predictive accuracy only"),
 ("E6", "Not an attribution study"),
 ("E7", "Secondary study (review or survey); trains no model"),
 ("E8", "Attributions compared across methods, models or architectures, not across re-initialisations of one specification"),
 ("E9-block", "Tier-2 record matched only weak vocabulary (retrain, refit, initialisation, seeds, variance); coded as a group"),
 ("", None),
 ("Example of a filled row", "Your decision: E4    Your note: perturbs inputs on one trained model, no retraining"),
]
for i, (a, b) in enumerate(legend, 1):
    lg.cell(i, 1, a).font = Font(name=F, bold=True, size=12 if i == 1 else 10)
    if b: c = lg.cell(i, 2, b); c.font = body; c.alignment = wrap
lg.column_dimensions["A"].width = 26; lg.column_dimensions["B"].width = 110

ws = wb.create_sheet("Review")
cols = [("#", 5), ("Search", 8), ("Tier", 6), ("Title", 45), ("Year", 7), ("Source", 22),
        ("DOI", 22), ("First-pass code", 11), ("First-pass reason", 40), ("Abstract", 90),
        ("Keywords", 30), ("Your decision", 12), ("Your note", 30), ("Agreement", 11)]
for j, (h, w) in enumerate(cols, 1):
    c = ws.cell(1, j, h); c.font = hdr; c.fill = hfill; c.alignment = Alignment(wrap_text=True, vertical="center")
    ws.column_dimensions[c.column_letter].width = w
n = len(rows)
for i, r in enumerate(rows, 2):
    vals = [i - 1, r["search"], int(r["tier"]), r["title"], int(r["year"]) if r["year"].isdigit() else r["year"],
            r["source"], r["doi"], r["code"], r["reason"], r["abstract"], r["keywords"], None, None,
            f'=IF(L{i}="","",IF(L{i}=H{i},"agree","CHANGED"))']
    for j, v in enumerate(vals, 1):
        c = ws.cell(i, j, v); c.font = body; c.alignment = wrap
        c.border = Border(top=thin, bottom=thin, left=thin, right=thin)
    ws.cell(i, 12).fill = yel; ws.cell(i, 13).fill = yel
    ws.row_dimensions[i].height = 150
dv = DataValidation(type="list", formula1='"IN,E1,E2,E3,E4,E5,E6,E7,E8,E9-block"', allow_blank=True)
ws.add_data_validation(dv); dv.add(f"L2:L{n+1}")
ws.freeze_panes = "E2"; ws.auto_filter.ref = f"A1:N{n+1}"

sm = wb.create_sheet("Progress")
items = [("Records to review", f"=COUNTA(Review!D2:D{n+1})"),
         ("Reviewed", f"=COUNTA(Review!L2:L{n+1})"),
         ("Remaining", "=B1-B2"),
         ("Agree with first pass", f'=COUNTIF(Review!N2:N{n+1},"agree")'),
         ("Changed from first pass", f'=COUNTIF(Review!N2:N{n+1},"CHANGED")'),
         ("Your IN codes", f'=COUNTIF(Review!L2:L{n+1},"IN")')]
for i, (a, f) in enumerate(items, 1):
    sm.cell(i, 1, a).font = Font(name=F, bold=True); sm.cell(i, 2, f).font = Font(name=F)
sm.cell(8, 1, "If 'Your IN codes' is above zero, the gap claim that Searches A and B returned no included record no longer holds.").font = Font(name=F, italic=True)
sm.column_dimensions["A"].width = 28; sm.column_dimensions["B"].width = 12
wb.save(OUT)

from collections import Counter
print("rows", n, "| by search", Counter(r["search"] for r in rows), "| tier", Counter(r["tier"] for r in rows))
print("codes", Counter(r["code"] for r in rows))
print("missing abstracts", sum(r["abstract"].startswith("(no abstract") for r in rows))
print("wrote", OUT, hashlib.sha256(OUT.read_bytes()).hexdigest()[:16])
