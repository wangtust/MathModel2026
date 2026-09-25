"""Check Q3 coverage, consistency, localization and reproducible artifacts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import subprocess
from pathlib import Path
from common.paths import Q3_OUTPUT_DIR, Q2_OUTPUT_DIR

import numpy as np
import pandas as pd

from question3.pipeline import ROOT, MODS


def verify(out: Path):
    t = out / "tables"
    p = pd.read_csv(t / "attachment4_predictions.csv", dtype={"sample_id": str})
    e = pd.read_csv(t / "attachment4_evidence.csv", dtype={"sample_id": str})
    a = pd.read_csv(t / "attachment4_alignment_audit.csv", dtype={"sample_id": str})
    words = pd.read_csv(t / "attachment4_word_map.csv", dtype={"sample_id": str})
    token = pd.read_csv(t / "attachment4_token_map.csv", dtype={"sample_id": str})
    vf = pd.read_csv(t / "validation_explanation_metrics.csv")
    explained = pd.read_csv(t / "validation_predictions_explained.csv")
    perf = pd.read_csv(t / "model_performance.csv")
    ig_check = pd.read_csv(t / "attachment4_ig_completeness.csv")
    valid_ig_check = pd.read_csv(t / "validation_ig_completeness.csv")
    sensitivity = pd.read_csv(t / "attachment4_model_sensitivity.csv", dtype={"sample_id": str})
    full_pred = pd.read_csv(t / "full_validation_predictions_explained.csv",
                            dtype={"sample_id": str})
    full_evidence = pd.read_csv(t / "full_validation_evidence.csv",
                                dtype={"sample_id": str})
    full_ig = pd.read_csv(t / "full_validation_ig_completeness.csv",
                          dtype={"sample_id": str})
    quadrature = pd.read_csv(t / "full_validation_ig_quadrature_check.csv",
                             dtype={"sample_id": str})
    paired = pd.read_csv(t / "validation_fidelity_model_paired.csv",
                         dtype={"sample_id": str})
    paired_summary = pd.read_csv(t / "validation_fidelity_model_summary.csv")
    rank = pd.read_csv(t / "gate_response_rank.csv", dtype={"sample_id": str})
    rank_summary = pd.read_csv(t / "gate_response_rank_summary.csv")
    whole = pd.read_csv(t / "whole_modality_deletion.csv", dtype={"sample_id": str})
    issues = []
    ids = [f"{n:02d}" for n in range(1, 21)]
    if sorted(p.sample_id) != ids or sorted(a.sample_id) != ids:
        issues.append("prediction/alignment sample coverage is not exactly 01..20")
    if len(e) != 60 or e.groupby("sample_id").size().to_dict() != {sid: 3 for sid in ids}:
        issues.append("evidence must have exactly one row per sample and modality")
    if sorted(perf.split) != ["test", "valid"] or perf.set_index("split").loc["valid", "n"] != 728 or perf.set_index("split").loc["test", "n"] != 727:
        issues.append("fixed valid/test split sizes are wrong")
    if not all(np.isclose(p[[f"prob_{x}" for x in ("negative", "neutral", "positive")]].sum(axis=1), 1, atol=1e-5)):
        issues.append("class probabilities do not sum to one")
    if not all(np.sign(p.loc[p.prediction_class == 0, "sentiment_strength"]) <= 0) or not all(p.loc[p.prediction_class == 1, "sentiment_strength"] == 0) or not all(np.sign(p.loc[p.prediction_class == 2, "sentiment_strength"]) >= 0):
        issues.append("coherent strength contradicts polarity")
    for _, r in e.iterrows():
        sid, mod = r.sample_id, r.modality
        if sid not in ids or mod not in MODS:
            issues.append(f"invalid evidence ID/modality {sid}/{mod}")
            continue
        if not bool(r.available) and (pd.notna(r.token_start) or pd.notna(r.start_sec) or pd.notna(r.keyframe_sec)):
            issues.append(f"native-unavailable evidence got a location: {sid}/{mod}")
        if pd.notna(r.start_sec):
            dur = float(a.loc[a.sample_id == sid, "duration_sec"].iloc[0])
            if not (0 <= r.start_sec < r.end_sec <= dur + .1):
                issues.append(f"time out of source video: {sid}/{mod}")
            if r.localization_status != "trusted_word_alignment":
                issues.append(f"timed evidence lacks trust: {sid}/{mod}")
        if mod == "vision" and pd.notna(r.keyframe_sec) and not Path(r.keyframe_path).is_file():
            issues.append(f"missing keyframe: {sid}")
        if mod != "vision" and pd.notna(r.keyframe_sec):
            issues.append(f"non-visual keyframe: {sid}/{mod}")
        if bool(r.available):
            if not (1 <= r.token_start <= r.token_end <= 48):
                issues.append(f"invalid BERT content position: {sid}/{mod}")
            if pd.notna(r.get("source_char_end")) and r.source_char_end > float(a.loc[a.sample_id == sid, "retained_char_end"].iloc[0]):
                issues.append(f"evidence cites text after BERT truncation: {sid}/{mod}")
            matching = token[(token.sample_id == sid) & token.token_index.between(r.token_start, r.token_end)]
            if len(matching) != r.token_end - r.token_start + 1:
                issues.append(f"token map incomplete: {sid}/{mod}")
    if not (a.loc[a.sample_id == "15", "alignment_status"].iloc[0] != "trusted" and
            e.loc[e.sample_id == "15", "start_sec"].isna().all()):
        issues.append("sample 15 mismatch must not have trusted times")
    if not (e.loc[(e.sample_id == "13") & (e.modality == "vision"), "available"].iloc[0] == False):
        issues.append("sample 13 native vision absence not preserved")
    for sid in ("07", "18"):
        if not bool(a.loc[a.sample_id == sid, "text_truncated"].iloc[0]):
            issues.append(f"sample {sid} truncation not preserved")
    if set(vf.modality) != set(MODS) or not set(["all", "correct", "incorrect"]).issubset(vf.stratum):
        issues.append("fidelity comparison/strata incomplete")
    if len(explained) != 728:
        issues.append("not all 728 validation samples received explanations")
    if len(ig_check) != 20 or ig_check.absolute_completeness_error.max() >= .1:
        issues.append("IG numerical completeness exceeds the predeclared 0.1 margin tolerance")
    if len(valid_ig_check) != 728 or valid_ig_check.absolute_completeness_error.max() >= .1:
        issues.append("validation IG completeness or sample coverage failed")
    if sorted(sensitivity.sample_id) != ids:
        issues.append("frozen-model sensitivity lacks some attachment-4 samples")
    if len(full_pred) != 728 or set(full_pred.sample_id) != set(explained.sample_id):
        issues.append("full variant does not cover the same 728 validation IDs")
    if len(full_evidence) != 728 * 3 or full_evidence.groupby("sample_id").size().ne(3).any():
        issues.append("full variant modality evidence is incomplete")
    if len(full_ig) != 728 or set(full_ig.sample_id) != set(explained.sample_id):
        issues.append("full variant IG completeness lacks validation samples")
    worst_sid = full_ig.loc[full_ig.absolute_completeness_error.idxmax(), "sample_id"]
    if (len(quadrature) != 3 or quadrature.sample_id.nunique() != 1 or
            quadrature.sample_id.iloc[0] != worst_sid or
            not quadrature.ig_steps.is_monotonic_increasing or
            quadrature.absolute_completeness_error.iloc[-1] >= .1):
        issues.append("full variant worst-case higher-resolution IG check failed")
    if len(paired) < 2000 or paired.duplicated(["sample_id", "modality"]).any():
        issues.append("paired validation fidelity rows are incomplete or duplicated")
    if len(paired_summary) != 12 or not set(paired_summary.metric) == {"margin", "prob"}:
        issues.append("paired fidelity summary is incomplete")
    if len(whole) != (728 + 20) * 3 or whole.duplicated(["split", "sample_id", "modality"]).any():
        issues.append("whole-modality perturbation rows are incomplete")
    if len(rank) != (728 + 20) * 2 or rank.duplicated(["split", "sample_id", "perturbation"]).any():
        issues.append("gate-versus-response rank audit is incomplete")
    if len(rank_summary) != 4:
        issues.append("gate-versus-response rank summary is incomplete")
    figures = list((out / "figures").glob("*.pdf"))
    for pdf in figures:
        if not pdf.with_suffix(".png").is_file():
            issues.append(f"PNG companion missing for {pdf.name}")
    card_ids = sorted(pdf.stem.removeprefix("case_") for pdf in figures
                      if pdf.stem.startswith("case_"))
    if card_ids != ids:
        issues.append("not all attachment-4 samples have individual explanation cards")
    if len(figures) != 29:
        issues.append(f"expected 29 PDF figures, got {len(figures)}")
    supplement = out / "paper/attachment4_explanation_cards.pdf"
    if not supplement.is_file():
        issues.append("20-page attachment-4 explanation supplement is missing")
    else:
        info = subprocess.run(["pdfinfo", str(supplement)], capture_output=True,
                              text=True, check=True).stdout
        pages = next((int(line.split(":", 1)[1].strip()) for line in info.splitlines()
                      if line.startswith("Pages:")), 0)
        if pages != 20:
            issues.append(f"attachment-4 explanation supplement has {pages} pages, expected 20")
    summary = {"passed": not issues, "issues": issues, "predictions": len(p),
               "evidence_rows": len(e), "word_map_rows": len(words),
               "token_map_rows": len(token), "trusted_samples": int((a.alignment_status == "trusted").sum()),
               "trusted_local_evidence": int((e.localization_status == "trusted_word_alignment").sum()),
               "ig_completeness_mean_absolute_error": float(ig_check.absolute_completeness_error.mean()),
               "ig_completeness_max_absolute_error": float(ig_check.absolute_completeness_error.max()),
               "validation_ig_completeness_mean_absolute_error": float(valid_ig_check.absolute_completeness_error.mean()),
               "validation_ig_completeness_max_absolute_error": float(valid_ig_check.absolute_completeness_error.max()),
               "full_variant_class_agreement": int(sensitivity.class_agreement.sum()),
               "full_validation_explained_samples": len(full_pred),
               "full_validation_ig_completeness_mean_absolute_error":
                   float(full_ig.absolute_completeness_error.mean()),
               "full_validation_ig_completeness_max_absolute_error":
                   float(full_ig.absolute_completeness_error.max()),
               "full_validation_ig_above_0_1":
                   int((full_ig.absolute_completeness_error >= .1).sum()),
               "worst_case_ig_error_192_steps":
                   float(quadrature.absolute_completeness_error.iloc[-1]),
               "paired_fidelity_rows": len(paired),
               "gate_rank_rows": len(rank),
               "individual_explanation_cards": len(card_ids),
               "validation_explained_samples": len(explained),
               "fidelity_eligible_by_modality": {m: int(vf[(vf.modality == m) & (vf.stratum == "all") &
                                                        (vf.comparison == "top-minus-random")].n.iloc[0])
                                                 for m in MODS},
               "pdf_png_pairs": len(figures),
              "package_versions": {name: metadata.version(name) for name in
                                    ("torch", "numpy", "transformers", "whisperx",
                                     "faster-whisper", "matplotlib", "pandas",
                                     "scikit-learn", "scipy")},
               "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (ROOT / "question3" / "pipeline.py",
                                              ROOT / "question3" / "evidence" / "aligner.py",
                                              ROOT / "question3" / "evidence" / "mapper.py",
                                              ROOT / "question3" / "evaluation" / "ig_check.py",
                                              ROOT / "question3" / "evaluation" / "sensitivity.py",
                                              ROOT / "question3" / "evaluation" / "fidelity.py",
                                              ROOT / "question3" / "evaluation" / "alignment_quality.py",
                                              ROOT / "question3" / "visualization" / "report.py",
                                              ROOT / "question3" / "analysis" / "verification.py")}}
    (out / "verification.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Q3_OUTPUT_DIR)
    verify(parser.parse_args().output)
