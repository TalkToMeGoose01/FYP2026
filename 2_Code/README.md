# 2_Code

All code behind the thesis. The top-level README gives the working-folder layout these scripts expect. `7_Handover/WORKFLOW.pdf` shows how they connect.

Run `check_versions.py` first to print the library versions in use. Every analysis script writes a `PROVENANCE` file into `results/` recording the SHA-256 digest of each input it read.

## Shared code

| File | Purpose |
|---|---|
| `fyp_pipeline.py` | Shared preprocessing for Phase 1 and Phase 2. Screening, merging, `align_to_schema`, scaling and 48-hour windows |
| `check_versions.py` | Prints Python, TensorFlow, SHAP and other versions. Run before any rerun |
| `wordcount.py` | The word-count rule used for the thesis declaration |

## notebooks/

| File | Purpose | Thesis |
|---|---|---|
| `FYP_MultiSite_BDGP2_SHAP_CPU.ipynb` | Phase 1. Anchor LSTM, zero-shot transfer and SHAP | Section I, Phase 1 |
| `FYP_Phase2_TFT_FINAL.ipynb` (`.py`, `.html`) | Phase 2 main notebook. LSTM refit, TFT grid, Gate 1 and 2 registrations, the 1,294-building sweep and Analyses 1 to 4. The `.html` keeps the executed outputs. Read README hazard 1 before running | Sections III and IV |
| `Analysis5_TenSeed_Confirmation.ipynb` | Ten seeds per architecture at the anchor | Sections IV-A, V-C2 |
| `rerun_tft_confirmation.py` | Reruns of the TFT top-5 confirmation on 2 September 2026 | Appendix A |

The same-seed retraining rung (Analysis 6) has its scripts inside `4_Archives/analysis6_2026-08-17.zip` (`a6_sameseed_rung.py`, `a6b_fixedlength_rung.py`).

## analysis_scripts/

| Script | What it does | Main outputs | Thesis |
|---|---|---|---|
| `a7_rank2_draw_check.py` | Confirmation run on a forced-CPU path | `a7_rank2_draw_check.csv` | Appendix D |
| `a8_shap_ceiling.py` | SHAP estimation ceiling | `analysis8_shap_ceiling.csv` | III-G |
| `a9_rank2_vs_rank5_tenseed.py` | Selected TFT configuration against the one the rule returns, 10 seeds each | `analysis9_*` | Appendix A |
| `a11_stratification_eps2.py` | Building-type stratification and η²_H effect sizes | `analysis11_*` | III-J, IV-G |
| `a12_zerofill_audit.py` | Zero-fill audit, feature mask and the masked agreement figures | `analysis12_*` | IV-C, V-C1 |
| `a13_input_identity.py` | Digests of every building's input tensor | `analysis13_*` | IV-G |
| `a14_target_correlation.py` | Lag-1 and meter correlations with each building's electricity | `analysis14_*` | V-D |
| `a15_amendments.py` | Between-architecture tests, de-duplicated stratification, cluster noise and concentration | `analysis15_*` | IV-E, IV-G, V-C2 |
| `a16_normality.py` | Shapiro-Wilk on the four electricity 10-run samples. Uses fixed paths, see the top README | `analysis16_*` | Appendix B |
| `a16b_normality_gas.py` | The same test on the two gas-target samples | `analysis16b_*` | Appendix B |
| `a17_tft_shap_attention.py` | SHAP on the 20 Analysis 5 models, TFT selection weights and attention | `analysis17_*` | IV-E, Appendix B |
| `a18_tft_ig.py` | Integrated Gradients on the same 20 models | `analysis18_*` | IV-E |
| `a19_target_anchor.py` | Gas as the target at the anchor, 10 seeds per architecture | `analysis19_*` | IV-F |
| `a20_ig_baseline_factorial.py` | IG baseline against step count, 2 by 2 | `analysis20_*` | V-B |
| `a21_ceiling_across_types.py` | Ceilings and instability on 20 buildings outside education | `analysis21_*` | III-G |

## screening_scripts/

Run in order. Each writes a provenance file into `6_Literature_Search/`.

| Script | What it does |
|---|---|
| `a10_screening_triage.py` | Merges the database exports, removes duplicates and flags records for reading |
| `a10b_fill_triage_codes.py` | Fills first-pass codes for records the triage did not flag |
| `a10c_review_sheet.py` | Builds the author's review sheet for flagged Search A and B records |
| `a10d_apply_author_review.py` | Applies the author's codes from the review sheet |
| `a10e_record_D_review.py` | Records the author's review of the Search D records |

## figure_scripts/

| Script | Figure in the thesis | Reads |
|---|---|---|
| `make_figures.py` | Instability by method, input degeneracy, type discrimination, zero-fill artefact (files `f01`, `f03`, `f04`, `f07`). It also wrote 12 figures that were not used | See `make_figures_PROVENANCE.txt` |
| `f02_ig_baseline_steps.py` | IG baseline and step count | `analysis20_ig_factorial.csv` |
| `f06_search_flow.py` | Search and screening flow | `screening/` |
| `f08_controllable.py` | Leading controllable meter | `analysis12_controllable.csv` |
| `f09_gate1_tenseed.py` | Gate 1 at the anchor | `analysis5_tenseed_predictive.csv` |
| `f10_gate2_ratio.py` | Gate 2 parity ratio | `gate2_parity.csv` |
| `f11_tft_selection.py` | TFT configuration selection | TFT selection logs and `analysis9_rank2_vs_rank5.csv` |

The `f` scripts share `figstyle.py` and check each number quoted in the thesis against their input before drawing.
