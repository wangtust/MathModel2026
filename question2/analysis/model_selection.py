"""Select a Q2 configuration by three-seed validation score only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from question2.analysis.model_comparison import MODELS, read_repetitions
from question2.experiments.baselines import predict_attachment3
from question2.training.trainer import predict as predict_mrrf


ROOT = Path(__file__).resolve().parents[2]
from common.paths import Q2_OUTPUT_DIR
from question2.config import run_dir

OUT = Q2_OUTPUT_DIR / "analysis"


def export_submission(source: Path, destination: Path) -> None:
    """Write only contest-facing fields and fail on missing or inconsistent rows."""
    with source.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expected_ids = {f"附件3_{index:02d}" for index in range(1, 31)}
    ids = [row["sample_id"] for row in rows]
    if len(rows) != 30 or set(ids) != expected_ids or len(set(ids)) != 30:
        raise ValueError("Attachment 3 prediction CSV must contain each sample 01–30 exactly once")
    label_names = ("Negative", "Neutral", "Positive")
    cleaned = []
    for row in rows:
        label = int(row["classification_label"])
        annotation = row["annotation"]
        strength = float(row["sentiment_strength"])
        if label not in range(3) or annotation != label_names[label]:
            raise ValueError(f"Inconsistent polarity for {row['sample_id']}")
        if not (-3 <= strength <= 3) or not (
            (label == 0 and strength < 0) or
            (label == 1 and strength == 0) or
            (label == 2 and strength > 0)
        ):
            raise ValueError(f"Inconsistent strength for {row['sample_id']}")
        cleaned.append({"sample_id": row["sample_id"],
                        "annotation": annotation,
                        "classification_label": label,
                        "sentiment_strength": strength})
    cleaned.sort(key=lambda row: row["sample_id"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cleaned[0]))
        writer.writeheader()
        writer.writerows(cleaned)


def upgraded_candidates(root: Path) -> dict[str, tuple[float, Path]]:
    """Include complete three-seed runs trained under the original selection rule."""
    if not root.exists():
        return {}
    candidates = {}
    for first in sorted(root.glob("*_seed2026")):
        variant = first.name.removesuffix("_seed2026")
        paths = [root / f"{variant}_seed{seed}" / "metrics.json"
                 for seed in (2026, 2027, 2028)]
        if not all(path.exists() for path in paths):
            continue
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        if any(report.get("selection_metric") != "official" or
               report.get("variant") != variant for report in reports):
            raise ValueError(f"Incompatible validation rule or variant for {variant}")
        config_keys = ("hidden_dim", "lr", "regression_weight",
                       "reconstruction_weight", "consistency_weight",
                       "train_clean_prob", "class_weight_power",
                       "mask_padding", "mask_unavailable", "train_gap_layout")
        configurations = [tuple(report["training_config"].get(key) for key in config_keys)
                          for report in reports]
        if len(set(configurations)) != 1:
            raise ValueError(f"Three seeds have different training configurations: {variant}")
        score = sum(float(report["selection_score"]) for report in reports) / 3
        candidates[f"robust_v2_{variant}"] = (
            score, first / "attachment3_predictions_mrrf.csv")
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path,
                        default=Q2_OUTPUT_DIR / "experiments" / "robustness",
                        help="Optional three-seed upgraded MRRF runs")
    args = parser.parse_args()
    _, summary = read_repetitions()
    scores = {model: next(row["selection_score_mean"] for row in summary
                          if row["model"] == label and row["split"] == "validation"
                          and row["scenario"] == "clean")
              for label, model in MODELS}
    extras = upgraded_candidates(args.candidate_root)
    scores.update({name: score for name, (score, _) in extras.items()})
    selected = max(scores, key=scores.get)
    if selected in extras:
        prediction = extras[selected][1]
        if not prediction.exists():
            predict_mrrf(SimpleNamespace(output=str(prediction.parent), batch_size=32))
    elif selected in ("concat_mlp", "gru", "mult_no_aug", "tfr_feature"):
        output = run_dir(selected, 2026)
        predict_attachment3(SimpleNamespace(model=selected, seed=2026,
                                            output=str(output), batch_size=32))
        prediction = output / "attachment3_predictions_baseline.csv"
    elif selected == "mmsa_baseline":
        prediction = run_dir("mult", 2026) / "attachment3_predictions_baseline.csv"
    else:
        prediction = run_dir("mrrf", 2026, selected.removeprefix("mrrf_")) / "attachment3_predictions_mrrf.csv"
    if not prediction.exists():
        raise FileNotFoundError(prediction)
    submission = OUT / "attachment3_submission.csv"
    export_submission(prediction, submission)
    report = {"selected_variant": selected,
              "validation_selection_scores": scores,
              "candidate_configurations": list(scores),
              "selection_basis": "three_seed_validation_mean_across_complete_candidates",
              "selection_rule": "Mean across seeds 2026, 2027, 2028 of the average of clean and audio+vision 30% validation (macro_f1 - mae/3)",
              "prediction_checkpoint_seed": 2026,
              "prediction_checkpoint_seed_rule": "Fixed reporting seed 2026; epoch selected by validation, never by held-out test or unlabeled attachment 3",
              "candidate_root": str(args.candidate_root),
              "prediction_file": str(prediction),
              "submission_file": str(submission),
              "attachment3_has_labels": False}
    (OUT / "selected_model.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Selected {selected}, validation mean {scores[selected]:.4f}")
    print(prediction)
    print(submission)


if __name__ == "__main__":
    main()
