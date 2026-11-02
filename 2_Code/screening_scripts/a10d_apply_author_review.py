#!/usr/bin/env python3
"""Analysis 10d: apply the author's title/abstract review of the 48 flagged
Search A and B records (screening/REVIEW_AB_flagged.xlsx, column L/M) to the
screening CSVs. Every copy of a reviewed record (A, B, D and ALL) takes the
author's code. Three columns are appended to every CSV:
  first_pass_decision, first_pass_reason, review_status
Byte copies of the CSVs are taken before writing. Usage:
  python3 a10d_apply_author_review.py <screening dir>
"""
import csv, hashlib, re, shutil, sys, datetime, pathlib, openpyxl

D = pathlib.Path(sys.argv[1])
TAG = "BACKUP-20261019-preA10D_"
FULLTEXT = {
    3: "Full text checked 19 Oct 2026 (Sections 3.7, 4.5, 4.6).",
    5: "Full text checked 19 Oct 2026: 20 initialisations for R2; SHAP applied to the best-performing run only.",
    8: "Full text checked 19 Oct 2026: seeds used for RMSE reproducibility only; LIME and SHAP from a single model.",
    15: "Full text checked 19 Oct 2026: ten seeds for error metrics; SHAP computed once on the final model.",
    43: "Proceedings volume. T-REX (Zakrzewski et al., LNAI 15166, pp. 216-227) read in full: compares computer-vision explanation methods and architectures against human saliency annotations; no repeated initialisation.",
}
STD = {
    "E1": "seed stated only as a fixed value; attribution not compared across runs.",
    "E3": "retraining refers to model updating or drift, not re-initialisation.",
    "E5": "seed variance reported for predictive accuracy only; attribution not compared across runs.",
    "E6": "not an attribution stability study; any attribution is computed on a single trained model.",
    "E7": "secondary study (review or survey); trains no model.",
    "E8": "attribution compared across methods, models or architectures, not across re-initialisations.",
    "IN": "meets both halves of the criterion.",
}
TYPOS = {"bet-performing": "best-performing"}
norm = lambda t: re.sub(r"[^a-z0-9]", "", (t or "").lower())

ws = openpyxl.load_workbook(D / "REVIEW_AB_flagged.xlsx", data_only=True)["Review"]
rev_doi, rev_title = {}, {}
for r in ws.iter_rows(min_row=2, values_only=True):
    if r[0] is None:
        continue
    n, title, doi, code, note = int(r[0]), r[3], r[6], (r[11] or "").strip(), (r[12] or "").strip()
    if not code:
        sys.exit(f"record {n} has no author decision")
    for a, b in TYPOS.items():
        note = note.replace(a, b)
    reason = "Author review: " + (note.rstrip(".") + "." if note else STD[code])
    if n in FULLTEXT:
        reason += " " + FULLTEXT[n]
    status = "author reviewed 19 Oct 2026" + (" (title, abstract and full text)" if n in FULLTEXT else " (title and abstract)")
    rec = (n, code, reason, status)
    if doi and str(doi) != "None":
        rev_doi[str(doi).lower()] = rec
    else:
        rev_title[norm(title)] = rec

log = [f"ANALYSIS 10d -- apply author review of REVIEW_AB_flagged.xlsx",
       f"run (UTC) : {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
       f"script    : a10d_apply_author_review.py  sha256 {hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}",
       f"review    : REVIEW_AB_flagged.xlsx  sha256 {hashlib.sha256((D/'REVIEW_AB_flagged.xlsx').read_bytes()).hexdigest()}",
       f"backups   : {TAG}*.csv (byte copies taken before this run)"]
seen = set()
for name in ["screening_A.csv", "screening_B.csv", "screening_C.csv", "screening_D.csv", "screening_ALL.csv"]:
    p = D / name
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    shutil.copy2(p, D / (TAG + name))
    with open(p, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f); fields = rd.fieldnames; rows = list(rd)
    if "review_status" in fields:
        sys.exit(f"{name} already processed")
    out_fields = fields + ["first_pass_decision", "first_pass_reason", "review_status"]
    changed = matched = 0
    for row in rows:
        row["first_pass_decision"], row["first_pass_reason"] = row["decision"], row["reason"]
        rec = rev_doi.get(row["doi"].lower()) if row["doi"] else None
        rec = rec or rev_title.get(norm(row["title"]))
        if rec:
            n, code, reason, status = rec
            matched += 1; seen.add(n)
            changed += row["decision"] != code
            row["decision"], row["reason"], row["review_status"] = code, reason, status
        elif row["decision"] == "TRIAGE-OUT":
            row["review_status"] = "not flagged by triage script; not read"
        else:
            row["review_status"] = "first pass by Claude 3 Sep 2026 on title, abstract and keywords; not author reviewed"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields); w.writeheader(); w.writerows(rows)
    log.append(f"  {name:18s} rows {len(rows):4d}  reviewed {matched:3d}  decision changed {changed:3d}  sha256 before {before}  after {hashlib.sha256(p.read_bytes()).hexdigest()}")
missing = sorted(set(range(1, 49)) - seen)
log.append(f"reviewed records matched: {len(seen)}/48" + (f"  MISSING {missing}" if missing else ""))
(D / "PROVENANCE_a10d.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
print("\n".join(log))
