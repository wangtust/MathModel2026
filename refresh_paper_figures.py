#!/usr/bin/env python3
"""Refresh the publication-facing figures for Q1/Q2/Q3 without retraining models.

This script is intended for the final local rendering pass before paper writing:

1. Rebuild Q1 reports, the explicit 100×3 modality summary table and the four
   compact main-text figures under outputs/q1/paper_figures.
2. Recompute the Q2 position×rate validation grid with the validation-selected
   MRRF variant, then build the six compact Q2 main-text figures + supplement.
3. Build the five compact Q3 main-text figures + supplement from saved Q3 tables.
4. Verify that every publication-facing PDF/PNG pair and manifest exists.

No model is trained here. Q2 performs inference on the validation set for the
position×rate grid and error-attribution figure; Q1/Q3 reuse saved experiment
outputs.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
Q1 = ROOT / "outputs" / "q1"
Q2 = ROOT / "outputs" / "q2"
Q3 = ROOT / "outputs" / "q3"


def run(args: list[str], title: str) -> None:
    print("\n" + "=" * 78, flush=True)
    print(title, flush=True)
    print("=" * 78, flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def selected_q2_model_dir() -> Path:
    path = Q2 / "analysis" / "selected_model.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Finish Q2 model selection before refreshing paper figures."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = data.get("selected_variant")
    seed = int(data.get("prediction_checkpoint_seed", 2026))
    mapping = {
        "mrrf_no_recon": "no_recon",
        "mrrf_full": "full",
        "mrrf_dual_recon": "dual_recon",
    }
    if selected not in mapping:
        raise RuntimeError(f"Unsupported selected Q2 variant: {selected!r}")
    model_dir = Q2 / "runs" / "mrrf" / mapping[selected] / f"seed_{seed}"
    if not (model_dir / "checkpoint.pt").is_file():
        raise FileNotFoundError(f"Selected Q2 checkpoint is missing: {model_dir}")
    print(f"Q2 selected model: {selected}, seed={seed}", flush=True)
    return model_dir


def expected_outputs() -> list[Path]:
    q1_names = [
        "fig01_q1_framework",
        "fig02_q1_quality_overview",
        "fig03_q1_representative_alignment",
        "fig04_q1_quality_control",
    ]
    q2_names = [
        "fig01_q2_framework",
        "fig02_q2_missing_type_rate",
        "fig03_q2_position_rate_interaction",
        "fig04_q2_models_and_ablation",
        "fig05_q2_error_attribution",
        "fig06_q2_attachment3",
    ]
    q3_names = [
        "fig01_q3_explainability_framework",
        "fig02_q3_fidelity",
        "fig03_q3_modality_effects",
        "fig04_q3_representative_cases",
        "fig05_q3_localization_and_audit",
    ]

    outputs: list[Path] = []
    for name in q1_names:
        outputs += [Q1 / "paper_figures" / f"{name}.pdf", Q1 / "paper_figures" / f"{name}.png"]
    for name in q2_names:
        outputs += [Q2 / "paper_figures" / f"{name}.pdf", Q2 / "paper_figures" / f"{name}.png"]
    for name in q3_names:
        outputs += [Q3 / "paper_figures" / f"{name}.pdf", Q3 / "paper_figures" / f"{name}.png"]
    outputs += [
        Q1 / "tables" / "10_journal_case_manifest.csv",
        Q1 / "tables" / "11_paper_feature_summary.csv",
        Q1 / "paper_figures" / "figure_manifest.json",
        Q2 / "paper_figures" / "figure_manifest.json",
        Q2 / "paper_figures" / "fig05_error_attribution_samples.csv",
        Q2 / "paper_figures" / "supplementary" / "figure_manifest.json",
        Q3 / "paper_figures" / "figure_manifest.json",
        Q3 / "paper_figures" / "table01_q3_frozen_model_performance.csv",
        Q3 / "paper_figures" / "supplementary" / "supplement_manifest.json",
    ]
    return outputs


def audit() -> None:
    missing = [p for p in expected_outputs() if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError(
            "Paper-figure refresh finished but required outputs are missing/empty:\n"
            + "\n".join(f"  - {p.relative_to(ROOT)}" for p in missing)
        )

    # Q1 competition table must contain exactly 100 samples × 3 modalities and
    # the fields explicitly requested in the statement.
    import csv

    q1_table = Q1 / "tables" / "11_paper_feature_summary.csv"
    with q1_table.open("r", encoding="utf-8-sig", newline="") as f:
        q1_rows = list(csv.DictReader(f))
    q1_required = {
        "sample_id", "modality", "source_duration_sec", "feature_dim",
        "alignment_granularity", "aligned_sequence_length", "valid_fraction",
    }
    if len(q1_rows) != 300:
        raise RuntimeError(f"Q1 paper feature summary should contain 300 rows, found {len(q1_rows)}")
    if not q1_rows or not q1_required.issubset(q1_rows[0]):
        raise RuntimeError("Q1 paper feature summary is missing competition-required columns")
    if len({row["sample_id"] for row in q1_rows}) != 100:
        raise RuntimeError("Q1 paper feature summary does not cover exactly 100 unique samples")
    if {row["modality"] for row in q1_rows} != {"Text", "Audio", "Vision"}:
        raise RuntimeError("Q1 paper feature summary must contain Text/Audio/Vision for every sample")

    # Q2 position×rate must use the same variant selected for final predictions;
    # otherwise the paper would mix two model definitions.
    grid = Q2 / "experiments" / "robustness" / "position_rate_grid.csv"
    selected = json.loads((Q2 / "analysis" / "selected_model.json").read_text(encoding="utf-8"))
    selected_variant = {
        "mrrf_no_recon": "no_recon",
        "mrrf_full": "full",
        "mrrf_dual_recon": "dual_recon",
    }[selected["selected_variant"]]
    with grid.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 90:
        raise RuntimeError(f"Q2 position-rate grid should contain 90 rows, found {len(rows)}")
    variants = {row.get("checkpoint_variant") for row in rows}
    if variants != {selected_variant}:
        raise RuntimeError(
            f"Q2 position-rate grid variant {sorted(variants)} != selected {selected_variant}"
        )

    print("\nPAPER FIGURE AUDIT | PASS", flush=True)
    print("✓ Q1: 4 main-text figures + 100×3 competition summary table", flush=True)
    print("✓ Q2: 6 main-text figures + supplement; position×rate uses selected model", flush=True)
    print("✓ Q3: 5 main-text figures + supplement + performance table", flush=True)


def main() -> None:
    # Fast syntax guard for all modules touched during the visualization pass.
    run(["-m", "compileall", "-q", "common", "question1", "question2", "question3"],
        "Compile check")

    run(["question1/run.py", "--reports-only"],
        "Q1 | Rebuild reports and compact publication figures from saved features")

    model_dir = selected_q2_model_dir()
    run([
        "question2/run.py", "experiment", "--name", "robustness",
        "--model-dir", str(model_dir),
    ], "Q2 | Refresh 10–50% position×rate grid with selected MRRF checkpoint")
    run(["question2/run.py", "paper-figures"],
        "Q2 | Build compact main-text and supplementary figures")

    run(["question3/run.py", "paper-figures"],
        "Q3 | Build compact main-text and supplementary figures")

    audit()


if __name__ == "__main__":
    main()
