# Reading roadmap

The literature behind this thesis, grouped by the decision each source supports. Keys match `1_Thesis/Thesis.bib`. The section column says where the thesis uses each one. A successor new to the area should read the groups in order.

How the literature was found is recorded in `6_Literature_Search/` (search strings, screening codes and reasons for all 817 records plus the author review). Appendix F of the thesis gives the strings.

## 1. Start here, the gap

| Key | Source | Why it matters here | Thesis section |
|---|---|---|---|
| `arabzadeh2025` | Arabzadeh and Frank (2025), systematic review of XAI in energy forecasting | Fewer than 15% of 50 reviewed studies test explanation robustness. Also the source for how often load forecasting studies include the target's own history | I, II-B, II-C, II-G, III-A |
| `iumanova2026` | Iumanova et al. (2026), review of XAI for trustworthy energy forecasting | Names the many-values problem of attribution on time series, which drives the feature collapse of Section III-F | II-B, II-C, II-G, III-F |
| `thinn2026` | Thinn and Saeed (2026), agreement between attribution methods for solar forecasting | Treats method agreement as evidence that a feature is physical, with no calibration for estimator noise. The thesis's main point of contrast | II-C, II-G, V-A |
| `liao2024` | Liao et al. (2024), trust in XAI for wind power | Fidelity against a retraining reference. The distinction between fidelity and reproducibility starts here | II-D, II-G, V-A |
| `bian2026` | Bian et al. (2026), cross-climate robustness of building energy forecasting | Architecture rankings change with the target site. The reason for Gate 2 | II-A, II-G, III-D |

## 2. Models

| Key | Source | Why it matters here | Thesis section |
|---|---|---|---|
| `hochreiter1997` | Hochreiter and Schmidhuber (1997) | The LSTM | II-A |
| `lim2021` | Lim et al. (2021) | The Temporal Fusion Transformer, built from this description in Keras | II-A, III-C |
| `kong2019`, `surmon2024comparison`, `guo2024stacking`, `amasyali2018` | LSTM and load forecasting studies and a data-driven prediction review | Context for choosing the LSTM as the base model | II-A |
| `kingma2014` | Kingma and Ba (2015) | The Adam optimiser | III-B |

## 3. Attribution methods

| Key | Source | Why it matters here | Thesis section |
|---|---|---|---|
| `lundberg2017` | Lundberg and Lee (2017) | SHAP | II-B, III-E2 |
| `breiman2001` | Breiman (2001) | Permutation importance | II-B, III-E1 |
| `sundararajan2017` | Sundararajan et al. (2017) | Integrated Gradients and the reason for a non-zero baseline | II-B, III-E3 |
| `ribeiro2016` | Ribeiro et al. (2016) | LIME, considered and excluded | II-B, III-E4 |
| `machlev2022`, `maarif2023`, `li2022` | XAI applied to energy and LSTM models | Shows SHAP is the default in this field and that no prior study tested its reproducibility | II-B |
| `dean2025xai`, `rong2022`, `liao2025` | Method agreement, efficient fidelity testing and a glass-box wind model | Neighbouring work that does not test retraining | II-C, II-D |
| `bhatt2020`, `hendro2025` | Sensitivity measures for explanations | Robustness to input changes, which is a different property from robustness to retraining | II-E |

## 4. Instability under retraining, the watch list

These are the 12 records that met the inclusion criterion of the structured search. They are the closest work to this thesis and are the papers to watch for follow-ups. Eleven come from outside energy forecasting.

| Key | Source | Setting |
|---|---|---|
| `ghannam2026` | Ghannam (2026) | Random Forest and XGBoost on Australian CO2 emissions, 30 seeds. The only energy-domain inclusion |
| `yasodhara2021` | Yasodhara et al. (2021) | Tree ensembles |
| `lin2025` | Lin and Wang (2025) | SHAP stability across 100 gradient boosting models, credit data |
| `rahman2026` | Rahman (2026) | SHAP stability across algorithms and seeds, materials data |
| `vos2025` | Vos et al. (2025) | Stabilising explanations by repeated validation, clinical data |
| `cvetkovic2026` | Cvetković and Mitrović Dankulov (2026) | Graph neural networks |
| `yudistira2027` | Yudistira et al. (2027) | TabNet, tabular deep learning |
| `parajuli2025` | Parajuli and Racharak (2025) | Underspecification in image classification |
| `hase2021` | Hase et al. (2021) | Text, feature importance search methods |
| `rivero2026` | Rivero (2026) | Loss landscape flatness and explanation stability |
| `liang2026` | Liang et al. (2026) | Local feature importance for tree models |
| `osojnik2025` | Osojnik et al. (2025) | Feature ranking for online learners |

To update the watch list, rerun Search D (Appendix F) with a date limit after 11 August 2026 and screen the new records with `2_Code/screening_scripts/a10_screening_triage.py`.

## 5. Data and statistics

| Key | Source | Why it matters here | Thesis section |
|---|---|---|---|
| `miller2020` | Miller et al. (2020) | The BDGP2 dataset | III-A |
| `kruskal1952` | Kruskal and Wallis (1952) | Test for differences between building types | III-J |
| `holm1979` | Holm (1979) | Correction for testing many features | III-J |
| `tomczak2014` | Tomczak and Tomczak (2014) | The η²_H effect size | III-J |
| `iea2022` | International Energy Agency (2022) | Share of emissions from building operation | I |

## 6. Software

`abadi2016` (TensorFlow), `mckinney2010` (pandas), `harris2020` (NumPy), `pedregosa2011` (scikit-learn), `virtanen2020` (SciPy), `hunter2007` (Matplotlib) and `anthropic_claude` (Claude, see Section III-K for how it was used).

## 7. Further reading not cited in the thesis

These eight entries are in `Thesis.bib` but were not needed in the final text. They suit an extension of the work.

| Key | Source | Possible use |
|---|---|---|
| `fan2020` | Fan et al. (2020), transfer learning for short-term building load prediction | Starting point for adding target-building data instead of pure zero-shot transfer |
| `fan2021` | Fan et al. (2021), data analytics for building performance | Background on data-driven building analytics |
| `michailidis2026` | Michailidis et al. (2026), review of machine learning for building energy management | Recent review to update the literature map |
| `chiang2026` | Chiang et al. (2026), campus building energy forecasting | Campus buildings like most of the education set here |
| `piscitelli2021` | Piscitelli et al. (2021), anomaly detection in daily energy profiles | Possible use of per-building explanations once they are reliable |
| `granderson2016` | Granderson et al. (2016), automated measurement and verification | Accuracy standards for building energy baselines |
| `coraci2025`, `silvestri2025` | Deep reinforcement learning for building control | Where trusted per-building explanations could be applied |
