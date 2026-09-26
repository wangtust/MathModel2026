# Q3 journal figure layer

`question3/visualization/main_text.py` reorganizes the existing Q3 experiment outputs into a compact main-text figure set. It does **not** retrain the model, rerun model selection, or recompute Integrated Gradients.

## Generate

From the project root:

```bash
python question3/run.py paper-figures
```

To rebuild both the legacy audit package and the compact journal layer:

```bash
python run.py q3 --force-figures
```

The compact outputs are written to:

```text
outputs/q3/paper_figures/
├── fig01_q3_explainability_framework.pdf/.png
├── fig02_q3_fidelity.pdf/.png
├── fig03_q3_modality_effects.pdf/.png
├── fig04_q3_representative_cases.pdf/.png
├── fig05_q3_localization_and_audit.pdf/.png
├── table01_q3_frozen_model_performance.csv
├── table01_q3_frozen_model_performance.md
├── figure_manifest.json
└── supplementary/
```

## Main-text semantics

### Fig Q3-1 — explainability and evidence-audit framework

The frozen Q2 predictor is not redrawn in detail. The figure emphasizes the Q3 chain:

```text
prediction
→ Integrated Gradients
→ continuous top-20% candidate window
→ independent deletion validation
→ token/word/source-time mapping
→ alignment audit
→ explanation card
```

The Q3 experiment uses **48-step Integrated Gradients with the fixed 0.1× input reference**, not a zero baseline. Therefore IG is interpreted relative to the 10% attenuated reference input.

### Fig Q3-2 — validation fidelity

The main text uses `top − matched random` as the primary fidelity comparison. The random windows are matched for window length and native valid-position count. Panel (b) zooms Audio/Vision so the near-zero effects and the Vision CI crossing zero remain visible.

`top − low IG` remains in the legacy/supplementary fidelity analysis.

### Fig Q3-3 — modality-level effects

The heatmap is the **signed deletion response** of each selected modality window. The adjacent main-modality strip uses the perturbation-derived `dominant_modality` field.

Internal gate weights are deliberately not shown as contribution percentages. The gate-vs-deletion diagnostic is supplementary because the top gate modality is Text for every sample and does not outperform its marginal baseline as a modality-importance classifier.

### Fig Q3-4 — representative explanation cases

The three main cases are:

- `01`: typical Text-dominant sample;
- `19`: Vision-dominant sample with trusted localization;
- `15`: ASR mismatch; the system keeps token/text evidence but refuses second-level localization.

Each modality IG track is normalized only within that modality (`IG / max|IG|`) to answer **Where?**. Cross-modality effect size is shown separately with signed deletion response to answer **How much?**.

### Fig Q3-5 — localization and quality control

The figure combines:

1. WhisperX word coverage × independent ASR agreement;
2. per-sample/per-modality localization state;
3. Sample 19 source-time tracks;
4. the corresponding source video frame when available.

A video frame only verifies temporal correspondence. It must not be described as an object-level visual attribution or as “what the model looked at”.

## Interpretation rules

Keep these concepts separate in paper text and captions:

- **IG attribution / local importance** → where a candidate local feature window lies;
- **signed deletion response** → perturbation-derived effect on the model output;
- **dominant reference modality** → selected from positive deletion response under the predefined rule;
- **audited time localization** → token → word → forced-alignment mapping that passed quality checks;
- **internal gate weight** → model-state diagnostic, not a causal contribution percentage.

Negative or inconclusive results are retained rather than hidden: Vision fidelity is not stably positive on the validation set, Sample 15 rejects source-time localization, and Sample 16 can remain `unclear` instead of being forced into Text/Audio/Vision.
