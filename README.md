# Same Model, Same Data, Different Answer

**Two Failure Modes of Per-Building Explanation in Cross-Building Energy Models: Attribution Reproducibility and Input Distinctiveness**

Final Year Project (ZEIT4500/4297). Paul Turner, School of Engineering and Technology, UNSW Canberra (ADFA). Supervisors Runpu Wang (Phase 2), Huadong Mo (co-supervisor) and Zongjun Li (Phase 1).

## Status

The thesis has been submitted and is awaiting grading. The code and handover documents are here now. The report itself will be added once grading is complete.

## What this project is about

Post-hoc attribution methods such as SHAP, permutation importance and Integrated Gradients are widely used to explain building energy prediction models. This work asks whether those explanations survive retraining. If the same model is trained again on the same data with a different random seed, does the same method give the same answer?

The answer is measured against an estimation ceiling, which separates real model instability from noise in the explanation method itself.

- **Attribution method dominates reproducibility.** On the LSTM, permutation importance is about five times as unstable as SHAP or Integrated Gradients, and its instability varies widely from building to building.
- **A reproducible explanation is not necessarily a meaningful one.** The 1,157 buildings tested present only 519 distinct inputs to the model. 56% share their input with a building of a different type, so no attribution method can tell those buildings apart.

## What is in this repository

| Folder | Contents |
|---|---|
| `2_Code/` | The preprocessing pipeline, notebooks, analysis scripts (a7 to a21), screening scripts and figure scripts, with `environment.yml` (WSL2) and `requirements.txt` (Windows). `2_Code/README.md` says what each script does |
| `7_Handover/` | Start here. The workflow flowchart, verification guide, decision log, risk register and reading roadmap |

The result files, sealed archives and trained models were submitted with the thesis as a separate package. They are not in this repository because of their size. Documents here that refer to `3_Results/`, `4_Archives/` or `5_Models/` mean that package.

The Building Data Genome Project 2 dataset is not included. It is openly available from its project repository (Miller et al., 2020).

## Contact

Paul Turner, z5462414@ad.unsw.edu.au
