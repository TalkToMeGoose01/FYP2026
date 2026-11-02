# Decision log

The research decisions that shaped this thesis, in date order, with the reason for each and where the evidence sits in this package. Writing and formatting changes to the report are not listed. Section numbers refer to the final thesis in `1_Thesis/`.

Earlier entries come from the Phase 2 notebook, the sealed archives and the September 2026 audit of the draft against the artefacts. Later entries come from the script provenance files written at the time.

## Phase 1 and project setup (before July 2026)

| Decision | Reason | Evidence |
|---|---|---|
| Use the BDGP2 dataset, 2016 only | Open benchmark with 1,636 non-residential buildings across 19 sites. 2016 gives a full leap year of hourly data | Section III-A, `0_Phase1_and_earlier/` |
| Anchor building `Bobcat_education_Alissa` | Carries the most meter types of any screened building (six of eight) with under 1% of its electricity readings missing | Section III-A, Appendix A |
| 48-hour lookback window | Lower validation RMSE than 24 hours in the initial runs reported in the Interim Report. 72 and 168 hours were not trialled because of training cost | Section III-A |
| Leave the building's own electricity history out of the inputs | It would rank first under every method and hide which other inputs drive consumption. This departs from 16 of the 20 load forecasting studies tabulated by Arabzadeh and Frank and limits what can be said about building type | Sections III-A and V-D |
| Insert a constant zero column for absent meters (`align_to_schema`) | Every building must present the same 15-column tensor to the anchor model | Section III-A, `fyp_pipeline.py` |
| Scale each building with its own scaler fitted on its own training partition | Keeps error in each building's own units with no data passed between buildings | Section III-A1 |

## July 2026, registered gates and the first portfolio runs

| Date | Decision | Reason | Evidence |
|---|---|---|---|
| 14 Jul | Gate 1 registered. The TFT is a valid comparator only if its anchor test MAE% is at most 26% (1.5 times the LSTM's 17.3%) | Fix the acceptance test before any TFT result is seen | `FYP_Phase2_TFT_FINAL.ipynb` cell 2, also inside `4_Archives/analysis1_2026-07-21.zip` |
| 14 Jul | Full 72-configuration TFT grid registered | Search the space before choosing a configuration | Notebook cell 18, `3_Results/tft_selection_logs/` |
| 17 Jul | Gate 2 registered. Per-building parity band 0.80 to 1.25, sensitivity band 0.667 to 1.50 | Bian et al. report architecture rankings that change with the target site. Registered before any ratio was computed | Notebook cell 28, `3_Results/gate2_parity.csv` |
| 21 Jul | Six portfolio draws and the permutation ceilings sealed | Freeze the weights behind every portfolio figure, since GPU training cannot be repeated bit for bit | `4_Archives/analysis1_2026-07-21.zip`, `CHECKSUMS.TXT` |
| 24 Jul | Integrated Gradients run at two baselines (training mean and zero) and the stratification sealed. The training mean is the reported baseline | A zero vector is the minimum of every feature at once under MinMax scaling, an impossible operating point | `4_Archives/analysis4_2026-07-24.zip`, Section III-E3 |

## August 2026

| Date | Decision | Reason | Evidence |
|---|---|---|---|
| 11 Aug | Structured search run in Scopus and Web of Science with four Boolean strings | The research gap is a negative claim, so the search had to be reproducible | Appendix F, `6_Literature_Search/` |
| 17 Aug | Same-seed retraining rung run and sealed | Separate retraining noise from initialisation noise | `4_Archives/analysis6_2026-08-17.zip`, Appendix C |

## September 2026, audit and corrections

| Date | Decision | Reason | Evidence |
|---|---|---|---|
| 2 Sep | TFT confirmation figures in the draft replaced by three real confirmation sessions, all reported | The earlier figures traced to a hand-typed string that no script produced | `3_Results/tft_selection_logs/`, Appendix A |
| 3 Sep | Ten seeds per architecture at the anchor (Analysis 5) | Three seeds cannot support a stability claim. The three-seed result that the TFT was four times more stable did not survive | `Analysis5_TenSeed_Confirmation.ipynb`, `analysis5_tenseed_predictive.csv`, Section V-C2 |
| 3 Sep | Selected TFT configuration tested against the one the stated rule returns, 10 seeds each (Analysis 9) | The configuration used was not the one the selection rule returned. The two cannot be told apart (Welch p = 0.33) | `a9_rank2_vs_rank5_tenseed.py`, Appendix A |
| 3 Sep | First-pass screening of all 817 records by script and Claude, with every reason recorded | Make each screening call checkable without re-reading the paper | `a10_screening_triage.py`, `6_Literature_Search/PROVENANCE.txt` |
| 4 Sep | Masked convention adopted for every reported rank correlation | Gradient methods give non-zero attribution to zero-filled absent meters. Ranking those columns inflated agreement | `a12_zerofill_audit.py`, `analysis12_feature_mask.csv`, Section V-C1 |
| 4 Sep | Controllable-meter result corrected from gas to chilled water | The gas result was an artefact of ranking meters a building does not carry | Section IV-B1, `analysis12_controllable.csv` |
| 4 Sep | Building-type contribution reframed around input distinctiveness | The input tensor comparison showed most stratified buildings share an input with a building of another type. The full-year figure reported is 650 of 1,157 | `a13_input_identity.py`, `analysis13_input_identity.csv`, Section IV-G |
| 4 Sep | Clipping, ceiling sample and statistic names disclosed. The ceiling buildings are all education buildings, every reported correlation is unclipped and the effect size is η²_H | An audit before the results were written up found these conventions in use but undisclosed | Sections III-G, III-J and IV-C, Appendix B |
| 8 Sep | Target-correlation check (Analysis 14) | Bound what a lagged target term would do without retraining | `a14_target_correlation.py`, Section V-D |
| 9 Sep | Between-architecture tests, de-duplicated stratification, cluster noise and concentration given their own result files (Analysis 15) | Archive quantities had been reported without a result file | `a15_amendments.py`, `analysis15_PROVENANCE.txt` |
| 10 Sep | Shapiro-Wilk test on every 10-run sample reported as mean ± SD (Analysis 16) | Report checklist requirement | `a16_normality.py`, `analysis16_PROVENANCE.txt` |

## October 2026, extension analyses

| Date | Decision | Reason | Evidence |
|---|---|---|---|
| 12 Oct | SHAP and TFT attention on 10 seeds per architecture at the anchor (Analysis 17) | Extend the method comparison to the second architecture | `a17_tft_shap_attention.py`, Section IV-E |
| 12 Oct | Integrated Gradients on the same 20 models (Analysis 18). Four predictions registered before the run, all held | Complete the three-method comparison on both architectures at the anchor | `a18_tft_ig.py`, `analysis18_summary.txt` |
| 12 Oct | Revised research question stated in two parts, reproducibility and building specificity | The agreed extension deliverables asked about architecture, method and target variable | Section I |
| 13 Oct | Gas used as the second target at the anchor (Analysis 19). Portfolio-wide alternative target deferred | The planned chilled-water target is zero through the anchor's test partition. Deferral agreed with the supervisor | `a19_target_anchor.py`, Section IV-F |
| 19 Oct | Author review of all 48 flagged Search A and B records. Ghannam (2026) recoded from E1 to IN after a full-text read | The first pass missed a seed-variance study whose abstract did not mention it | `a10d_apply_author_review.py`, `CODING_OUTCOME.md` |
| 20 Oct | IG baseline and step count tested as a 2 by 2 design (Analysis 20) | The archived zero-baseline run used 50 steps against 200, so two settings had changed at once | `a20_ig_baseline_factorial.py`, Section V-B |
| 21 Oct | Appendix figures drawn by scripts that check every quoted number against its result file before drawing | Stop a figure drifting from the number in the text | `2_Code/figure_scripts/` |
| 24 Oct | Ceilings and instability repeated on 20 buildings outside education (Analysis 21) | All 25 calibration buildings were education buildings | `a21_ceiling_across_types.py`, `analysis21_PREDICTIONS.txt` |
| 26 Oct | Author review of the 126 records retrieved only by Search D recorded. All first-pass codes agreed | Complete the author review of all 174 flagged records | `a10e_record_D_review.py`, `CODING_OUTCOME.md` |
| 26 Oct | Normality test extended to the two gas-target samples (Analysis 16b) | The gas models were trained after Analysis 16 ran | `a16b_normality_gas.py`, `analysis16b_PROVENANCE.txt` |
