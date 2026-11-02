# Risk register and next steps

What a successor should know before building on this work. Each risk names what could make a reported result wrong or narrower than it looks, how the thesis handles it now and the step that would close it. Section numbers refer to the final thesis in `1_Thesis/`.

Impact is rated against the main findings. **High** means the finding could change. **Medium** means its scope could change. **Low** means its precision could change.

## Risks to the findings

| # | Risk | Impact | Current handling | Next step |
|---|---|---|---|---|
| R1 | The building's own load history is not an input. Hour of Day may lead mostly because the daily cycle has nowhere else to go | High | Disclosed in Sections III-A and V-D. Lag-1 autocorrelation is 0.944 against 0.401 for gas, the strongest meter (`analysis14_target_correlation.csv`) | Add a lagged-load feature, retrain both architectures and repeat the method comparison and the type test |
| R2 | One anchor building and one year (2016) | High | Disclosed in Section V-D | Repeat with anchors from other types and sites and on 2017 |
| R3 | The building-type null is set by the input representation. 650 of 1,157 stratified buildings share an input with a building of another type | High | Reported as a finding in Section IV-G and repeated on one building per distinct input | Add inputs that identify the building, such as floor area, occupancy or lagged load, before testing type again |
| R4 | SHAP background is the first 200 training sequences, about eight days of January. This gives Month an exact zero | Medium | Explained in Section V-B. The background was never varied | Draw the background across the whole training partition and repeat the sweep |
| R5 | Estimation ceilings rest on 25 education buildings plus 5 buildings from each of four other types | Medium | Analysis 21 found no clear difference in instability. Five buildings per type cannot rule out a difference in one type | Calibrate at least 20 buildings per type |
| R6 | The portfolio architecture comparison uses permutation importance only. SHAP and IG reach the TFT only at the anchor | Medium | Figures reported as lower bounds (Section IV-E) | Run SHAP and IG on the TFT across the portfolio |
| R7 | Three seeds behind the portfolio draws and three runs per condition behind the retraining rung | Medium | Disclosed in Section V-D. Ten seeds were used at the anchor | Rerun the portfolio decomposition with 10 seeds per architecture |
| R8 | The architectures differ in batch size (LSTM 32, TFT 64) and the TFT grid never tried 32 | Low | Disclosed in Section III-B1 | Retrain the TFT at batch 32 |
| R9 | The TFT configuration used is not the one the stated selection rule returns | Low | The two are indistinguishable over 10 seeds each (Appendix A) | None needed unless the configuration matters to a new question |
| R10 | Under the gas target, hot water tracks gas at a correlation of 0.98 and acts as a proxy for the target | Medium | Disclosed in Sections IV-F and V-D | Repeat with a target that has no near-copy among the inputs. Run the deferred portfolio-wide alternative target |
| R11 | The retraining rung read the outlier-screened meter files while every other analysis read the unscreened ones | Low | Disclosed and left uncorrected in Appendix C | Rerun the rung on the unscreened files |
| R12 | Zero-shot accuracy is modest. Median MAE% across the portfolio is 40.6% for the LSTM | Medium | Reported in Section IV-A. Attribution results depend on accuracy only through Gate 2 parity | Test whether instability or type signal changes with per-building accuracy |
| R13 | The Phase 1 top-1 agreement rate of 50.8% was never reconciled with the 12-feature collapse | Low | Disclosed in Section V-C2 with the Phase 2 figure of 44.8% | Recompute from the Phase 1 attribution tensors |

## Risks to reproducing the work

| # | Risk | Current handling | Next step |
|---|---|---|---|
| P1 | GPU training is not bit-identical even under a fixed seed | Weights archived and fingerprinted (`4_Archives/`, `5_Models/`). README hazard 1 | Load the archived weights and never retrain to reproduce a reported number |
| P2 | One seed value gives different models under different scripts | Documented in Appendix B and `analysis9_rank2_vs_rank5_PROVENANCE.txt` | Compare distributions across scripts, never individual seeds |
| P3 | The Conda environment is not pinned in the package | `check_versions.py` prints versions. Appendix A lists the main ones | Add `environment.yml` and `requirements.txt` to `2_Code/` |
| P4 | `a16_normality.py` reads fixed paths set at the top of the script | Noted in the README | Point `RESULTS` and `OUT` at `results/` before running |
| P5 | TFT files need the custom layers defined before loading | README hazard 2 | Use the `custom_objects` load or `build_tft` plus `load_weights` |

## Suggested order for a successor

1. Rebuild the environment from `2_Code/` and check one archived result against its provenance file (`7_Handover/VERIFICATION.md` lists the checks).
2. Close R1 and R3 together. A lagged-load feature both tests the Hour of Day result and gives each building a distinct input.
3. Close R4, then R6. Both are reruns of existing scripts with one setting changed.
4. Extend to more anchors and years (R2) once the single-anchor results are settled.
5. Check the watch list in `READING_ROADMAP.md` for new work on attribution stability under retraining before writing up.
