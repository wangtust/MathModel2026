"""Frozen full-vs-no-reconstruction backbone sensitivity on the 20 unlabeled clips."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from common.paths import Q3_OUTPUT_DIR, Q2_OUTPUT_DIR

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from question3.pipeline import Q2, TOK, ROOT, clean_batch, load_a4, write_csv
from question2.training.trainer import load_trained
from question2.evaluation.metrics import coherent_strength
from question2.config import LABEL_NAMES


def run(out: Path):
    torch.set_num_threads(6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_dir = Q2_OUTPUT_DIR / "runs" / "mrrf" / "full" / "seed_2026"
    full, _ = load_trained(full_dir, device)
    full.eval()
    with np.load(full_dir / "train_scaler.npz") as f:
        scaler = {m: (f[f"{m}_mean"], f[f"{m}_std"]) for m in ("audio", "vision")}
    a4, _ = load_a4(scaler, AutoTokenizer.from_pretrained(TOK, use_fast=True, local_files_only=True))
    primary = pd.read_csv(out / "tables/attachment4_predictions.csv", dtype={"sample_id": str})
    primary.sample_id = primary.sample_id.str.zfill(2)
    primary_by_id = primary.set_index("sample_id")
    rows = []
    with torch.no_grad():
        for i, sid in enumerate(a4["id"]):
            batch = clean_batch(a4, [i], device)
            logits, strength = full(batch)
            probs = logits.softmax(-1)[0].cpu().numpy()
            cls = int(probs.argmax())
            raw = float(strength.item())
            adjusted = float(coherent_strength([cls], [raw])[0])
            old = primary_by_id.loc[sid]
            rows.append({"sample_id": sid,
                         "primary_annotation": old.annotation,
                         "primary_strength": float(old.sentiment_strength),
                         "full_annotation": LABEL_NAMES[cls],
                         "full_strength": adjusted,
                         "full_raw_strength": raw,
                         "full_prob_negative": float(probs[0]),
                         "full_prob_neutral": float(probs[1]),
                         "full_prob_positive": float(probs[2]),
                         "class_agreement": bool(old.annotation == LABEL_NAMES[cls]),
                         "absolute_strength_difference": abs(float(old.sentiment_strength) - adjusted)})
    write_csv(out / "tables/attachment4_model_sensitivity.csv", rows)
    compare = {r["sample_id"]: r for r in rows}
    enriched = []
    for row in primary.to_dict("records"):
        other = compare[row["sample_id"]]
        row["full_variant_annotation"] = other["full_annotation"]
        row["full_variant_strength"] = other["full_strength"]
        row["full_variant_agrees"] = other["class_agreement"]
        if not other["class_agreement"]:
            flags = str(row.get("explanation_quality_flags", "ok"))
            row["explanation_quality_flags"] = ("model_variant_disagreement" if flags == "ok"
                                                 else flags + ";model_variant_disagreement")
        enriched.append(row)
    write_csv(out / "tables/attachment4_predictions.csv", enriched)
    baseline = json.loads((full_dir / "metrics.json").read_text())
    summary = {"compared_samples": len(rows),
               "class_agreement_count": sum(r["class_agreement"] for r in rows),
               "mean_absolute_strength_difference": float(np.mean([r["absolute_strength_difference"] for r in rows])),
               "full_validation_clean": baseline["validation"]["clean"],
               "full_test_clean": baseline["heldout_test"]["clean"]}
    (out / "model_sensitivity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Q3_OUTPUT_DIR)
    run(p.parse_args().output)
