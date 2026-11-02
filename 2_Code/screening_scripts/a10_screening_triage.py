#!/usr/bin/env python3
"""
a10_screening_triage.py
=======================
ANALYSIS 10 -- REPRODUCIBLE SCREENING LOG FOR THE STRUCTURED SEARCH

WHY THIS EXISTS
---------------
Section~\\ref{sec:search} of main.tex rests on a negative claim: no
record in either database measured attribution reproducibility under
re-initialisation for a sequential model on time-series data. It reports
that ten records met the inclusion criterion and that six of the ten did
not appear in the Web of Science results.

The screening that produced those statements was performed in
conversation and written to no file. It is the largest unverified
load-bearing claim in the document: the funnel counts reproduce exactly
from the exports, but nothing records which records were flagged, which
were excluded, or why. A negative claim with no screening log is not
auditable, and this project has already learned twice what happens to
figures that exist only as recollection.

This script rebuilds the deterministic half of that screening and leaves
the judgement half to the author.

WHAT IS DETERMINISTIC AND WHAT IS NOT
-------------------------------------
Deterministic, and done here: retrieval, deduplication, and a vocabulary
triage over title, abstract and keywords. Given the same exports this
produces the same flags every time, and a marker can rerun it.

Judgement, and NOT done here: reading each flagged record and coding it
against the written inclusion criterion. The script emits empty
`decision` and `reason` columns for that. Coding must be the author's
own; inheriting a coding that exists only in a conversation would
reproduce the problem this script exists to fix.

THE CRITERION, QUOTED FROM main.tex
-----------------------------------
    "A study was retained if it both trained the same model
     specification more than once under different random initialisation
     and compared the resulting attribution output across those runs.
     Records failing either half were excluded."

THE EXCLUSION CODES, QUOTED FROM main.tex
-----------------------------------------
    E1  seed mentioned only as a fixed-value reproducibility statement
    E2  reproducibility referring to data or code availability rather
        than explanation variance
    E3  retraining referring to model updating or concept drift
    E4  robustness tested as input perturbation, distribution shift or
        ablation fidelity
    E5  seed variance reported for predictive accuracy only
    E6  not an attribution study
    IN  meets both halves of the criterion -- included

Use exactly these codes in the `decision` column. `reason` is free text
naming the sentence or table that decided it.

THE TRIAGE
----------
A record is flagged when its title, abstract or keywords contain at
least one term from EACH of three vocabularies:

    SEED    re-initialisation / random seed / retraining vocabulary
    ATTR    attribution / explanation vocabulary
    STAB    stability / reproducibility / variability vocabulary

This is deliberately broad. It is a recall device, not a decision: its
job is to ensure no record that could meet the criterion goes unread,
and it is expected to flag many records that will be excluded. The
matched terms are recorded per record so the flag can be audited.

RECALL CHECK -- READ THE OUTPUT
-------------------------------
The script checks its own triage against the set of papers main.tex
currently cites as the included ten, and prints any it fails to flag.
A miss is a finding, not a nuisance: either the triage vocabulary is too
narrow, or that paper's abstract does not support its inclusion, and
both matter. The vocabularies below must NOT be widened to chase a
specific paper after seeing the result. If a miss occurs, record it.

DEDUPLICATION
-------------
Matches the stated rule: on DOI, within and across databases. Records
carrying no DOI cannot be deduplicated that way and are therefore absent
from the union counts in Table~\\ref{tab:funnel}; they are nonetheless
retained here and screened, because exclusion from a count is not a
reason to leave a record unread. Hase et al., one of the currently
included ten, is such a record. Every row carries `has_doi` and
`in_union_count` so the two populations stay distinguishable.

OUTPUTS
-------
    screening/screening_A.csv .. screening_D.csv   every record, per search
    screening/screening_ALL.csv                    the deduplicated union
    screening/screening_SUMMARY.txt                counts + the recall check
    screening/PROVENANCE.txt                       inputs and their digests

Columns: search, database(s), doi, title, year, source, document_type,
has_doi, in_union_count, in_thesis_bib, bib_key, hit_seed, hit_attr,
hit_stab, triage_flag, matched_terms, decision, reason.

`in_thesis_bib` marks records already carried into the thesis, so the
records that must be coded first are identifiable at a glance.

Re-running preserves any `decision` / `reason` already entered: existing
output is read back and merged on DOI (or on normalised title where a
record has no DOI), so coding is never overwritten by a rerun.

USAGE
-----
    python a10_screening_triage.py \\
        --exports-dir "/mnt/e/Claude/FYP" \\
        --bib Thesis.bib \\
        --out-dir screening
"""

import argparse
import datetime
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------
# Export files, as retrieved 11 August 2026. Scopus first, then WoS.
# ---------------------------------------------------------------------
EXPORTS = {
    "A": ("export_be8c7f03-71fa-438d-b993-16333d2af457_"
          "2026-08-11T020136.106608641.csv", "Search_A_savedrecs.xls"),
    "B": ("SearchBCorrected_export_478a912e-c6d9-4758-9737-44a9864f7863_"
          "2026-08-11T021229.989950577.csv", "Search_B_savedrecs.xls"),
    "C": ("SearchCCorrected_export_609d2f88-b93e-4818-9c73-145d9ef4facf_"
          "2026-08-11T022323.76599963.csv", "Search_C_savedrecs.xls"),
    "D": ("SearchD_export_0f056ddf-c588-4817-bbe3-31170e845486_"
          "2026-08-11T022955.517678818.csv", "Search_D_savedrecs.xls"),
}

SEARCH_LABEL = {
    "A": "Re-initialisation, energy domain",
    "B": "Cross-method attribution agreement",
    "C": "Building-class stratification",
    "D": "Re-initialisation, domain unrestricted",
}

# ---------------------------------------------------------------------
# Triage vocabularies. Word-boundary regex, case-insensitive.
# Broad by design. Do not narrow or widen these to chase a result.
# ---------------------------------------------------------------------
VOCAB = {
    "seed": [
        r"random seed", r"\bseeds?\b", r"seeded", r"re-?initiali[sz]",
        r"weight initiali[sz]", r"random initiali[sz]", r"initiali[sz]ation",
        r"retrain", r"re-?train", r"refit", r"re-?fit",
        r"multiple runs", r"repeated runs", r"independent runs",
        r"training runs", r"across runs", r"run-to-run",
    ],
    "attr": [
        r"\bshap\b", r"shapley", r"\bxai\b", r"explainab", r"interpretab",
        r"feature importance", r"feature attribution", r"attribution",
        r"integrated gradients", r"saliency", r"\blime\b",
        r"permutation importance", r"feature ranking", r"variable importance",
        r"explanation", r"relevance propagation", r"gnnexplainer",
    ],
    "stab": [
        r"stabilit", r"stable", r"unstable", r"instabilit",
        r"reproducib", r"irreproducib", r"replicab",
        r"consisten", r"inconsisten", r"variabilit", r"variance",
        r"robust", r"disagree", r"agreement", r"sensitivit",
        r"trustworth", r"reliab",
    ],
}
COMPILED = {k: [(p, re.compile(p, re.I)) for p in v] for k, v in VOCAB.items()}

# Search D's Boolean query already requires the same conjunction this
# triage tests, so every record it retrieves is flagged by construction
# and the flag does no ordering work there. `tier` orders the reading
# queue without changing what must be read. It is descriptive, not a
# decision: tier 1 states the criterion's first half explicitly, tier 2
# reaches the vocabulary only through weaker terms.
STRONG_SEED = re.compile(
    r"random seed|re-?initiali[sz]|weight initiali[sz]|random initiali[sz]"
    r"|multiple runs|repeated runs|independent runs|training runs"
    r"|across runs|run-to-run|across seeds|per-?seed|different seeds", re.I)

EXCLUSION_CODES = """E1  seed mentioned only as a fixed-value reproducibility statement
E2  reproducibility referring to data or code availability
E3  retraining referring to model updating or concept drift
E4  robustness tested as input perturbation, distribution shift or ablation fidelity
E5  seed variance reported for predictive accuracy only
E6  not an attribution study
IN  meets both halves of the criterion -- included"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def norm_title(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def norm_doi(s):
    s = str(s).strip().lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    return "" if s in ("", "nan", "none") else s


def read_export(path, database):
    """Normalise a Scopus CSV or a WoS xls into one schema."""
    if str(path).lower().endswith((".xls", ".xlsx")):
        d = pd.read_excel(path)
        cols = dict(title="Article Title", abstract="Abstract",
                    year="Publication Year", source="Source Title",
                    doctype="Document Type", doi="DOI",
                    kw1="Author Keywords", kw2="Keywords Plus")
    else:
        d = pd.read_csv(path, low_memory=False)
        cols = dict(title="Title", abstract="Abstract", year="Year",
                    source="Source title", doctype="Document Type",
                    doi="DOI", kw1="Author Keywords", kw2="Index Keywords")
    out = pd.DataFrame()
    for k, c in cols.items():
        out[k] = d[c] if c in d.columns else ""
    out["database"] = database
    return out


def triage(row):
    """Flag when at least one term from each vocabulary appears."""
    text = " ".join(str(row.get(f, "")) for f in
                    ("title", "abstract", "kw1", "kw2"))
    hits, matched = {}, []
    for group, pats in COMPILED.items():
        found = [p for p, rx in pats if rx.search(text)]
        hits[group] = bool(found)
        matched += [f"{group}:{p}" for p in found[:4]]
    return hits["seed"], hits["attr"], hits["stab"], "; ".join(matched)


def bib_index(bib_path):
    """DOIs and normalised titles of everything in Thesis.bib, by key."""
    txt = Path(bib_path).read_text(encoding="utf-8")
    by_doi, by_title = {}, {}
    for m in re.finditer(r"@\w+\{\s*([^,]+),(.*?)\n\}", txt, re.S):
        key, body = m.group(1).strip(), m.group(2)
        d = re.search(r"doi\s*=\s*\{([^}]*)\}", body, re.I)
        t = re.search(r"title\s*=\s*\{(.*?)\},\s*\n", body, re.S | re.I)
        if d:
            by_doi[norm_doi(d.group(1))] = key
        if t:
            by_title[norm_title(re.sub(r"[{}]", "", t.group(1)))] = key
    return by_doi, by_title


def main(args):
    ex = Path(args.exports_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_doi, by_title = bib_index(args.bib)

    # previously entered coding, so a rerun never overwrites judgement
    prior = {}
    prev = out / "screening_ALL.csv"
    if prev.exists():
        p = pd.read_csv(prev)
        for _, r in p.iterrows():
            if str(r.get("decision", "")).strip() not in ("", "nan"):
                k = r["doi"] if r["doi"] else "T:" + norm_title(r["title"])
                prior[k] = (r.get("decision", ""), r.get("reason", ""))
        print(f"carrying forward {len(prior)} previously coded records")

    frames, prov = [], []
    for s, (sc, wo) in EXPORTS.items():
        parts = []
        for path, db in ((ex / sc, "Scopus"), (ex / wo, "WoS")):
            if not path.is_file():
                sys.exit(f"missing export: {path}")
            prov.append((path.name, sha256(path)))
            parts.append(read_export(path, db))
        d = pd.concat(parts, ignore_index=True)
        d["search"] = s
        d["doi"] = d["doi"].map(norm_doi)
        d["has_doi"] = d["doi"] != ""

        # dedup on DOI within and across databases, per the stated rule;
        # no-DOI records are kept individually and marked
        d["_dup"] = d["has_doi"] & d.duplicated(subset="doi", keep=False)
        # merge database provenance on DOI where present and on normalised
        # title where absent -- otherwise a record carried by both databases
        # without a DOI (Hase, one of the included set) reads as Scopus only
        d["_mk"] = d.apply(
            lambda r: r["doi"] if r["has_doi"] else "T:" + norm_title(r["title"]),
            axis=1)
        agg = d.groupby("_mk")["database"].agg(
            lambda v: "+".join(sorted(set(v))))
        d["databases"] = d["_mk"].map(agg)
        d = d[~d.duplicated(subset="_mk", keep="first")].copy()
        d = d.drop(columns="_mk")
        d["in_union_count"] = d["has_doi"]

        tri = d.apply(triage, axis=1, result_type="expand")
        d[["hit_seed", "hit_attr", "hit_stab", "matched_terms"]] = tri
        d["triage_flag"] = d.hit_seed & d.hit_attr & d.hit_stab
        _txt = d.apply(lambda r: " ".join(str(r.get(f, "")) for f in
                       ("title", "abstract", "kw1", "kw2")), axis=1)
        d["tier"] = [(1 if STRONG_SEED.search(x) else 2) if fl else ""
                     for fl, x in zip(d.triage_flag, _txt)]

        d["bib_key"] = d.apply(
            lambda r: by_doi.get(r["doi"]) or by_title.get(norm_title(r["title"])),
            axis=1)
        d["in_thesis_bib"] = d["bib_key"].notna()

        d["decision"] = ""
        d["reason"] = ""
        for i, r in d.iterrows():
            k = r["doi"] if r["doi"] else "T:" + norm_title(r["title"])
            if k in prior:
                d.at[i, "decision"], d.at[i, "reason"] = prior[k]

        cols = ["search", "databases", "doi", "title", "year", "source",
                "doctype", "has_doi", "in_union_count", "in_thesis_bib",
                "bib_key", "hit_seed", "hit_attr", "hit_stab", "triage_flag",
                "tier", "matched_terms", "decision", "reason"]
        d = d[cols].sort_values(
            ["triage_flag", "in_thesis_bib", "tier", "title"],
            ascending=[False, False, True, True])
        d.to_csv(out / f"screening_{s}.csv", index=False)
        frames.append(d)
        print(f"  Search {s}: {len(d):4d} records after dedup, "
              f"{int(d.triage_flag.sum()):3d} flagged, "
              f"{int(d.in_thesis_bib.sum()):2d} already in Thesis.bib")

    allrec = pd.concat(frames, ignore_index=True)
    # Union view: collapse on DOI where present, and on normalised title
    # where absent, so a record retrieved by two searches without a DOI
    # (Hase and Rong are both such cases) appears once rather than twice.
    allrec["_k"] = allrec.apply(
        lambda r: r["doi"] if r["has_doi"] else "T:" + norm_title(r["title"]),
        axis=1)
    union = allrec[~allrec.duplicated(subset="_k", keep="first")].copy()
    union = union.drop(columns="_k")
    union.to_csv(out / "screening_ALL.csv", index=False)

    # ---- recall check against the set main.tex currently includes ----
    tex = Path(args.main_tex)
    included = []
    if tex.is_file():
        m = re.search(r"outside this domain, on tabular.*?\\cite\{([^}]*)\}",
                      tex.read_text(encoding="utf-8"), re.S)
        if m:
            included = [k.strip() for k in m.group(1).split(",")]

    lines = []
    lines.append("SCREENING TRIAGE SUMMARY")
    lines.append(f"generated (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    lines.append("")
    lines.append("per search (after DOI dedup within and across databases):")
    for s, f in zip(EXPORTS, frames):
        lines.append(f"  {s}  {SEARCH_LABEL[s]:42s} "
                     f"records {len(f):4d}   in union count {int(f.in_union_count.sum()):4d}   "
                     f"no DOI {int((~f.has_doi).sum()):3d}   flagged {int(f.triage_flag.sum()):3d}")
    lines.append("")
    lines.append(f"deduplicated union across all four searches: {len(union)}")
    lines.append(f"  flagged for reading: {int(union.triage_flag.sum())}")
    lines.append(f"  already in Thesis.bib: {int(union.in_thesis_bib.sum())}")
    lines.append(f"  flagged AND in Thesis.bib: "
                 f"{int((union.triage_flag & union.in_thesis_bib).sum())}")
    fl = union[union.triage_flag]
    lines.append(f"  reading queue by tier: "
                 f"tier 1 = {int((fl.tier == 1).sum())}, "
                 f"tier 2 = {int((fl.tier == 2).sum())}")
    lines.append("")
    lines.append("NOTE ON SEARCH D. Its Boolean query already requires the")
    lines.append("same conjunction this triage tests, so every record it")
    lines.append("retrieved is flagged by construction. The screening burden")
    lines.append("for D is therefore its whole yield, not a subset of it.")
    lines.append("")
    lines.append("RECORDS ALREADY IN Thesis.bib (code these first):")
    for _, r in union[union.in_thesis_bib].sort_values("bib_key").iterrows():
        lines.append(f"  {str(r.bib_key):16s} search {r.search} "
                     f"tier {str(r.tier) or chr(45):1s}  "
                     f"{'flagged    ' if r.triage_flag else 'NOT flagged'}  "
                     f"{str(r.title)[:56]}")
    lines.append("")
    if included:
        lines.append(f"RECALL CHECK against the {len(included)} papers main.tex "
                     f"currently cites as included:")
        got = set(union[union.triage_flag].bib_key.dropna())
        present = set(union.bib_key.dropna())
        for k in included:
            if k in got:
                st = "flagged"
            elif k in present:
                st = "PRESENT BUT NOT FLAGGED  <- triage miss, record it"
            else:
                st = "NOT FOUND IN ANY EXPORT  <- retrieved by chaining?"
            lines.append(f"  {k:16s} {st}")
        miss = [k for k in included if k not in got]
        lines.append("")
        lines.append(f"  recall: {len(included)-len(miss)}/{len(included)}")
        if miss:
            lines.append("  Do NOT widen the vocabularies to capture these. "
                         "A miss is a finding.")
    lines.append("")
    lines.append("EXCLUSION CODES for the `decision` column:")
    lines.append(EXCLUSION_CODES)
    txt = "\n".join(lines)
    (out / "screening_SUMMARY.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)

    with open(out / "PROVENANCE.txt", "w", encoding="utf-8") as fh:
        fh.write("ANALYSIS 10 -- screening triage\n")
        fh.write(f"run (UTC) : {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
        fh.write(f"script    : {Path(__file__).name}  sha256 {sha256(__file__)}\n")
        fh.write(f"bib       : {args.bib}  sha256 {sha256(args.bib)}\n")
        fh.write("exports read:\n")
        for n, h in prov:
            fh.write(f"  {n[:60]:62s} sha256 {h}\n")
    print(f"\nwrote {out}/  (screening_A..D.csv, screening_ALL.csv, "
          f"screening_SUMMARY.txt, PROVENANCE.txt)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports-dir", default="/mnt/e/Claude/FYP",
                    help="directory holding the eight export files")
    ap.add_argument("--bib", default="Thesis.bib")
    ap.add_argument("--main-tex", default="main.tex")
    ap.add_argument("--out-dir", default="screening")
    sys.exit(main(ap.parse_args()))
