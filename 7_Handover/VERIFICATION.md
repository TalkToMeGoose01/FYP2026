# Verification guide

The checks behind Table IV of the thesis (Appendix D), with the file or script that carries out each one. A successor can run the first three in under an hour with no GPU. None of them retrains a model.

## Checks recorded in the thesis

| Check | What it establishes | How it was done | Where to look |
|---|---|---|---|
| Artefact provenance | Every reported number traces to a named result file | Each analysis script writes a `PROVENANCE` file with the SHA-256 digest of every input it read and every output it wrote | `3_Results/*PROVENANCE*.txt`, Table VI of the thesis |
| Archived weights | The models behind the reported numbers still exist after GPU nondeterminism | Weights sealed in three dated archives with checksums | `4_Archives/CHECKSUMS.TXT`, `5_Models/` |
| Reconstruction | Results can be rebuilt from the artefacts, not only traced to them | 5 August 2026 confirmation run on a forced-CPU path with a different Keras version. Five checks, including the anchor MAE and the Phase 1 SHAP magnitudes to four significant figures | `2_Code/analysis_scripts/a7_rank2_draw_check.py`, `3_Results/a7_rank2_draw_check.csv`, Appendix D |
| Gate registration | Acceptance criteria were fixed before results were seen | Gate 1 registered 14 July 2026 and Gate 2 on 17 July 2026 in the analysis notebook | `FYP_Phase2_TFT_FINAL.ipynb` cells 2 and 28, also in the notebook copy inside `analysis1_2026-07-21.zip` |
| Analysis controls | Absent-meter handling and reference windows behave as intended | Zero-fill audit and reference-window diagnostic. The gradient-method artefact it found led to the masked convention | `a12_zerofill_audit.py`, `analysis12_baseline_diagnostic.csv` |
| Self-audit | The work's own claims and registered predictions were tested | Predictions registered before runs are recorded in the result files. Failed predictions were kept | `analysis18_summary.txt`, `analysis19_gas_summary.txt`, `analysis21_PREDICTIONS.txt`, `7_Handover/DECISION_LOG.md` |

## Checks a successor can run

### 1. Archive integrity

Windows PowerShell, in `4_Archives/`

```
Get-FileHash -Algorithm SHA256 *.zip
```

Linux or WSL

```
sha256sum 4_Archives/*.zip
```

Compare with `4_Archives/CHECKSUMS.TXT`. The leading 16 characters also appear in Table V of the thesis.

### 2. A result file against its record

Pick any result file, compute its SHA-256 and compare it with the digest in the matching `PROVENANCE` file. The inputs listed in that file can be checked the same way, which walks the chain back to the archives.

### 3. A deterministic rerun

Scripts that read only result files and write CSV give byte-identical output on any machine. The two below were each run twice during packaging with identical results. `a10e` gave the same digests in two different environments.

| Script | Command from the working folder | Expected output digest (first 16) |
|---|---|---|
| `a16b_normality_gas.py` | `python a16b_normality_gas.py results` | `analysis16b_normality.csv` `889287b92f1b3224` |
| `a10e_record_D_review.py` | Run on the 19 October screening files as its docstring describes | `screening_D.csv` `268f871b05aa23ec` |


### 4. A model reload

Load an archived model and reproduce its recorded test error. The 10-seed scripts (`a17`, `a18`) refuse to continue unless every reloaded model matches its Analysis 5 test error to within 0.05 percentage points, so running either one is itself a reload check. See the README for how to load a TFT.

## What these checks do not cover

- GPU training cannot be repeated bit for bit. Retraining gives models from the same distribution, not the same weights. Section III-I of the thesis explains the three sources of variation this creates.
- There is no automated unit test suite. Correctness rests on the checks above, the assertions inside the figure scripts and the 10-seed gates.
- The BDGP2 source data is not included. Its download is fixed by Miller et al. (2020). The raw file digests read by Analysis 14 are in `analysis14_PROVENANCE.txt`.
