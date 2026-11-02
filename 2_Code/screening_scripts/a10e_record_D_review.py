#!/usr/bin/env python3
"""Analysis 10e: record the author's review of the 126 flagged records
retrieved only by Search D.

The first pass (3 Sep 2026) coded these records on title, abstract and
keywords. Analysis 10d (19 Oct 2026) applied the author's review of the 48
flagged Search A and B records and left the 126 Search D records marked
"not author reviewed". The author has since reviewed all 126 and agreed with
every first-pass code. This script records that outcome. It changes only the
review_status column, and only on rows that a10d left unreviewed. It refuses
to run if any such row has a final decision that differs from its first-pass
decision, since "agrees with first-pass code" would then be false.

Inputs must be the a10d outputs (their SHA-256 digests are checked).
Byte copies are taken before writing. Usage:
  python3 a10e_record_D_review.py <screening dir>
"""
import csv, hashlib, shutil, sys, datetime, pathlib

D = pathlib.Path(sys.argv[1])
TAG = "BACKUP-20261026-preA10E_"
OLD = "first pass by Claude 3 Sep 2026 on title, abstract and keywords; not author reviewed"
NEW = ("first pass by Claude 3 Sep 2026 on title, abstract and keywords; "
       "author reviewed, agrees with first-pass code")
A10D_AFTER = {  # from PROVENANCE_a10d.txt
    "screening_D.csv":   "e549af55bae7e5284c458d2adf7ef0393be6427f3d90c43088c050f0f0a4e031",
    "screening_ALL.csv": "81c9e57fff1ffee631bc80b10bdcdff3561c13c58b9e83b673100842866a22b3",
}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

log = ["ANALYSIS 10e -- record author review of the Search D records",
       f"run (UTC) : {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
       f"script    : a10e_record_D_review.py  sha256 {sha(pathlib.Path(__file__))}",
       "basis     : author reviewed all 126 flagged records retrieved only by Search D",
       "            and agreed with every first-pass code (author statement, 26 Oct 2026)",
       f"backups   : {TAG}*.csv (byte copies taken before this run)"]
total = 0
for name, expect in A10D_AFTER.items():
    p = D / name
    before = sha(p)
    if before != expect:
        sys.exit(f"{name}: digest {before[:16]} is not the a10d output {expect[:16]}")
    with open(p, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f); fields = rd.fieldnames; rows = list(rd)
    n = 0
    for row in rows:
        if row["review_status"] == OLD and row["search"] == "D":
            if row["decision"] != row["first_pass_decision"]:
                sys.exit(f"{name}: decision differs from first pass for {row['title'][:60]}")
            row["review_status"] = NEW
            n += 1
    if n != 126:
        sys.exit(f"{name}: expected 126 rows to update, found {n}")
    shutil.copy2(p, D / (TAG + name))
    with open(p, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, lineterminator="\r\n")
        wr.writeheader(); wr.writerows(rows)
    total += n
    log.append(f"  {name:<18} rows {len(rows):>4}  status updated {n:>3}  "
               f"sha256 before {before}  after {sha(p)}")
log.append(f"decisions changed: 0 (review_status column only)")
(D / "PROVENANCE_a10e.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
print("\n".join(log))
