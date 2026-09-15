# Same Model, Same Data, Different Answer

**Reproducibility and Meaning in Per-Building Explanations of LSTM Building Energy Models: Attribution Method Governs the First, Input Representation the Second**

Final Year Project (ZEIT4500/4297) · Paul Turner · School of Engineering and Technology, UNSW Canberra (ADFA) · Supervisor: Runpu Wang

## Status

This thesis has been submitted and is currently awaiting supervisor grading. **The full report and code will be uploaded to this repository once grading is complete.** This page exists so the QR code on the accompanying exam poster and seminar slides has somewhere real to point.

## What this project is about

Post-hoc attribution methods (SHAP, permutation importance, Integrated Gradients) are widely used to explain what a building energy prediction model is doing — which conditions it thinks drive consumption, and which measurements matter. This work asks a question that's rarely tested: if you retrain the exact same model on the exact same data with a different random seed, and re-run the same explanation method, do you get the same answer?

Two findings, calibrated against a purpose-built "estimation ceiling" that separates genuine model instability from noise in the explanation method itself:

- **Attribution method dominates reproducibility.** Permutation importance is roughly five times less reproducible than SHAP or Integrated Gradients under re-initialisation, and how unreproducible it is varies unpredictably from building to building.
- **A reproducible explanation isn't necessarily a meaningful one.** Across the 1,157-building stratified population, the model sees only 519 distinct input patterns — 56% of buildings share their exact input with a building of a different type. No attribution method can tell those buildings apart, because the model can't either.

## Coming when grading is complete

- Final report (PDF)
- Analysis code and notebooks
- Result artefacts and provenance records

## Contact

Paul Turner — z5462414@ad.unsw.edu.au
