#!/usr/bin/env python3
"""
a10b_fill_triage_codes.py
=========================
ANALYSIS 10b -- COMPLETE THE SCREENING LOG FOR RECORDS THE TRIAGE DID NOT FLAG

a10_screening_triage.py leaves `decision` and `reason` blank for every
record it does not flag. Those records were never read: the triage itself
excluded them, deterministically, because the title, abstract and keywords
lack at least one of the three required vocabularies (SEED, ATTR, STAB).

This script writes that outcome into the blank rows so the log has no
empty decisions. It uses only the triage's own columns (hit_seed,
hit_attr, hit_stab) and changes no row that already carries a decision.
It makes no judgement about any record.

    decision  TRIAGE-OUT
    reason    "Not flagged by triage: no <VOCAB> term in title, abstract
               or keywords" (all missing vocabularies are named)

Records without a DOI keep their own row in the per-search files and are
filled by the same rule.
"""
import csv, hashlib, datetime, pathlib

HERE = pathlib.Path(__file__).resolve().parent / "screening"
FILES = ["screening_A.csv", "screening_B.csv", "screening_C.csv",
         "screening_D.csv", "screening_ALL.csv"]
VOCAB = [("hit_seed", "SEED (re-initialisation)"),
         ("hit_attr", "ATTR (attribution)"),
         ("hit_stab", "STAB (stability)")]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

log = []
for name in FILES:
    p = HERE / name
    before = sha(p)
    rows = list(csv.DictReader(open(p, encoding="utf-8", newline="")))
    fields = list(rows[0].keys())
    filled = 0
    for r in rows:
        if r["decision"].strip():
            continue
        if r["triage_flag"] == "True":
            raise SystemExit(f"{name}: flagged record with no decision -- "
                             f"needs reading, not filling: {r['title'][:60]}")
        missing = [label for col, label in VOCAB if r[col] != "True"]
        if not missing:
            raise SystemExit(f"{name}: unflagged record with all vocab hits: {r['title'][:60]}")
        r["decision"] = "TRIAGE-OUT"
        r["reason"] = ("Not flagged by triage: no " + " and no ".join(missing)
                       + " term in title, abstract or keywords")
        filled += 1
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    log.append(f"  {name:20s} rows {len(rows):4d}  filled {filled:4d}  "
               f"sha256 before {before}  after {sha(p)}")

prov = HERE / "PROVENANCE_a10b.txt"
prov.write_text(
    "ANALYSIS 10b -- fill TRIAGE-OUT codes for unflagged records\n"
    f"run (UTC) : {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n"
    f"script    : a10b_fill_triage_codes.py  sha256 {sha(pathlib.Path(__file__).resolve())}\n"
    "originals : ORIGINAL_pre_a10b_*.csv (byte copies taken before this run)\n"
    + "\n".join(log) + "\n")
print(prov.read_text())
