"""Compare two frozen Q2 variants' explanations on the same Q3 validation set.

This is a post-selection diagnostic.  Each variant explains its own predicted
class; paired rows share sample IDs, modality, deletion protocol and RNG seed.
No attachment-4 labels or held-out test outcomes are used for model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from common.paths import Q3_OUTPUT_DIR, Q2_OUTPUT_DIR

import numpy as np
import pandas as pd
import torch

from question3.pipeline import (DATA, MODS, Q2, ROOT, SEED, FeatureStore,
                         bootstrap_mean, choose_validation_indices, clean_batch,
                         explain_set, fidelity_tables, integrated_gradients, margin,
                         write_csv)
from question2.training.trainer import load_trained


def paired_rows(primary: pd.DataFrame, full: pd.DataFrame,
                primary_pred: pd.DataFrame, full_pred: pd.DataFrame) -> list[dict]:
    """One row per jointly eligible sample/modality, preserving signed effects."""
    a = primary[(primary.available == True) & (primary.matched_random_candidate_count >= 2)]
    b = full[(full.available == True) & (full.matched_random_candidate_count >= 2)]
    merged = a.merge(b, on=["sample_id", "modality"], suffixes=("_no_recon", "_full"),
                     validate="one_to_one")
    predicted_a = primary_pred.set_index("sample_id").prediction_class
    predicted_b = full_pred.set_index("sample_id").prediction_class
    rows = []
    for r in merged.itertuples(index=False):
        sid, mod = str(r.sample_id), r.modality
        row = {"sample_id": sid, "modality": mod,
               "same_predicted_class": bool(predicted_a[sid] == predicted_b[sid]),
               "no_recon_predicted_class": int(predicted_a[sid]),
               "full_predicted_class": int(predicted_b[sid]),
               "no_recon_window_width": int(r.window_width_no_recon),
               "full_window_width": int(r.window_width_full)}
        for metric in ("margin", "prob"):
            for variant in ("no_recon", "full"):
                top = float(getattr(r, f"top_delta_{metric}_{variant}"))
                random = float(getattr(r, f"random_delta_{metric}_{variant}"))
                low = float(getattr(r, f"low_delta_{metric}_{variant}"))
                row[f"{variant}_top_delta_{metric}"] = top
                row[f"{variant}_top_minus_random_{metric}"] = top - random
                row[f"{variant}_top_minus_low_{metric}"] = top - low
            row[f"full_minus_no_recon_top_minus_random_{metric}"] = (
                row[f"full_top_minus_random_{metric}"] -
                row[f"no_recon_top_minus_random_{metric}"])
        rows.append(row)
    return rows


def summarize(paired: list[dict]) -> list[dict]:
    rng = np.random.default_rng(SEED + 41)
    rows = []
    for mod in MODS:
        all_rows = [r for r in paired if r["modality"] == mod]
        for stratum, subset in (("all", all_rows),
                                ("same_predicted_class", [r for r in all_rows
                                                          if r["same_predicted_class"]])):
            if not subset:
                continue
            for metric in ("margin", "prob"):
                a = np.array([r[f"no_recon_top_minus_random_{metric}"] for r in subset])
                b = np.array([r[f"full_top_minus_random_{metric}"] for r in subset])
                mean, lo, hi = bootstrap_mean(b - a, rng)
                rows.append({"modality": mod, "stratum": stratum, "metric": metric,
                             "n": len(subset), "no_recon_mean": float(a.mean()),
                             "full_mean": float(b.mean()), "full_minus_no_recon": mean,
                             "ci95_low": lo, "ci95_high": hi,
                             "full_better_fraction": float((b > a).mean())})
    return rows


@torch.no_grad()
def full_ig_completeness(model, valid: dict, predictions: list[dict],
                         positions: list[dict], device: torch.device) -> list[dict]:
    classes = {str(r["sample_id"]): int(r["prediction_class"]) for r in predictions}
    sums = {}
    for row in positions:
        sid = str(row["sample_id"])
        sums[sid] = sums.get(sid, 0.0) + float(row["ig_signed"])
    results = []
    for start in range(0, len(valid["id"]), 32):
        ids = list(range(start, min(start + 32, len(valid["id"]))))
        batch = clean_batch(valid, ids, device)
        cls = torch.tensor([classes[str(valid["id"][i])] for i in ids],
                           dtype=torch.long, device=device)
        observed = margin(model(batch)[0], cls).cpu().numpy()
        baseline = dict(batch)
        for mod in MODS:
            baseline[mod] = .1 * batch[mod]
        reference = margin(model(baseline)[0], cls).cpu().numpy()
        for i, current, base in zip(ids, observed, reference):
            sid = str(valid["id"][i])
            expected = float(current - base)
            total = sums[sid]
            results.append({"sample_id": sid, "margin_difference": expected,
                            "sum_ig": total,
                            "absolute_completeness_error": abs(total - expected)})
    return results


def worst_case_quadrature_check(model, valid: dict, predictions: list[dict],
                                completeness: list[dict], device: torch.device,
                                steps: int) -> list[dict]:
    """Recheck the largest 48-step residual at finer quadrature resolutions."""
    sid = max(completeness, key=lambda r: r["absolute_completeness_error"])["sample_id"]
    idx = list(valid["id"]).index(sid)
    one = clean_batch(valid, [idx], device)
    cls = torch.tensor([next(int(r["prediction_class"]) for r in predictions
                              if r["sample_id"] == sid)], device=device)
    with torch.no_grad():
        observed = float(margin(model(one)[0], cls).item())
        reference = dict(one)
        for mod in MODS:
            reference[mod] = .1 * one[mod]
        expected = observed - float(margin(model(reference)[0], cls).item())
    rows = []
    for count in (steps, steps * 2, steps * 4):
        ig = integrated_gradients(model, one, cls, count)
        total = sum(float(ig[mod].sum()) for mod in MODS)
        rows.append({"sample_id": sid, "variant": "full", "ig_steps": count,
                     "margin_difference": expected, "sum_ig": total,
                     "absolute_completeness_error": abs(total - expected)})
    return rows


def run(out: Path):
    torch.set_num_threads(6)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_dir = Q2_OUTPUT_DIR / "runs" / "mrrf" / "full" / "seed_2026"
    full, checkpoint = load_trained(full_dir, device)
    full.eval()
    config = json.loads((out / "config.json").read_text())
    valid = FeatureStore(DATA).splits["valid"]
    indices = choose_validation_indices(valid, int(config["validation_explanation_subset_requested"]))
    primary_pred = pd.read_csv(out / "tables/validation_predictions_explained.csv",
                               dtype={"sample_id": str})
    primary_ev = pd.read_csv(out / "tables/validation_evidence.csv",
                             dtype={"sample_id": str})
    if len(indices) != 728 or set(primary_pred.sample_id) != {
            str(valid["id"][i]) for i in indices}:
        raise ValueError("Primary and full validation sample sets differ")
    predictions, evidence, positions = explain_set(
        full, valid, indices, device, int(config["ig_steps"]),
        float(config["window_fraction"]), int(config["random_reps"]), "full_validation")
    if set(primary_pred.sample_id) != {str(r["sample_id"]) for r in predictions}:
        raise ValueError("Full model did not explain the same validation IDs")
    tables = out / "tables"
    write_csv(tables / "full_validation_predictions_explained.csv", predictions)
    write_csv(tables / "full_validation_evidence.csv", evidence)
    write_csv(tables / "full_validation_position_ig.csv", positions)
    write_csv(tables / "full_validation_explanation_metrics.csv",
              fidelity_tables(predictions, evidence))
    completeness = full_ig_completeness(full, valid, predictions, positions, device)
    write_csv(tables / "full_validation_ig_completeness.csv", completeness)
    write_csv(tables / "full_validation_ig_quadrature_check.csv",
              worst_case_quadrature_check(full, valid, predictions, completeness,
                                          device, int(config["ig_steps"])))
    paired = paired_rows(primary_ev, pd.DataFrame(evidence), primary_pred,
                         pd.DataFrame(predictions))
    write_csv(tables / "validation_fidelity_model_paired.csv", paired)
    write_csv(tables / "validation_fidelity_model_summary.csv", summarize(paired))
    record = {"comparison": "post_selection_exploratory",
              "models": ["mrrf_no_recon", "mrrf_full"],
              "full_checkpoint": str(full_dir / "checkpoint.pt"),
              "full_checkpoint_sha256": hashlib.sha256(
                  (full_dir / "checkpoint.pt").read_bytes()).hexdigest(),
              "full_checkpoint_epoch": checkpoint["epoch"],
              "validation_samples": len(indices), "paired_modality_rows": len(paired),
              "ig_steps": int(config["ig_steps"]),
              "ig_baseline_factor": float(config["ig_baseline_factor"]),
              "window_fraction": float(config["window_fraction"]),
              "random_reps": int(config["random_reps"]),
              "bootstrap_reps": 2000,
              "comparison_scope": "Each model explains its own predicted class; same-class subset is reported separately.",
              "selection_policy": "Primary no_recon choice remains based on prior clean-validation score; this diagnostic does not use the held-out test or attachment 4 for selection."}
    (out / "full_validation_comparison.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Full model fidelity complete: {len(indices)} validation samples, "
          f"{len(paired)} paired modality rows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Q3_OUTPUT_DIR)
    run(parser.parse_args().output)
