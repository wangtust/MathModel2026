"""Measure numerical completeness of the 12-step IG approximation on attachment 4."""
from __future__ import annotations

import argparse
from pathlib import Path
from common.paths import Q3_OUTPUT_DIR, Q2_OUTPUT_DIR

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from question3.pipeline import CKPT, TOK, ROOT, MODS, DATA, clean_batch, load_a4, margin, write_csv
from question2.training.trainer import load_trained
from question2.data.dataset import FeatureStore


def run(out: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(6)
    model, _ = load_trained(CKPT, device)
    model.eval()
    with np.load(CKPT / "train_scaler.npz") as f:
        scaler = {m: (f[f"{m}_mean"], f[f"{m}_std"]) for m in ("audio", "vision")}
    a4, _ = load_a4(scaler, AutoTokenizer.from_pretrained(TOK, use_fast=True, local_files_only=True))
    pred = pd.read_csv(out / "tables/attachment4_predictions.csv", dtype={"sample_id": str})
    pred.sample_id = pred.sample_id.str.zfill(2)
    pos = pd.read_csv(out / "tables/attachment4_position_ig.csv", dtype={"sample_id": str})
    pos.sample_id = pos.sample_id.str.zfill(2)
    rows = []
    with torch.no_grad():
        for i, sid in enumerate(a4["id"]):
            one = clean_batch(a4, [i], device)
            c = int(pred[pred.sample_id == sid].prediction_class.iloc[0])
            cls = torch.tensor([c], device=device)
            observed = float(margin(model(one)[0], cls).item())
            zero = dict(one)
            for m in MODS:
                zero[m] = .1 * one[m]
            base = float(margin(model(zero)[0], cls).item())
            ig = float(pos[pos.sample_id == sid].ig_signed.sum())
            expected = observed - base
            rows.append({"sample_id": sid, "observed_margin": observed,
                         "attenuated_baseline_margin": base, "margin_difference": expected,
                         "sum_ig": ig, "absolute_completeness_error": abs(ig - expected),
                         "relative_completeness_error": abs(ig - expected) / max(abs(expected), .1)})
    write_csv(out / "tables/attachment4_ig_completeness.csv", rows)
    print(f"Mean absolute IG completeness error: {np.mean([r['absolute_completeness_error'] for r in rows]):.4f}")
    store = FeatureStore(DATA)
    valid = store.splits["valid"]
    pred = pd.read_csv(out / "tables/validation_predictions_explained.csv", dtype={"sample_id": str})
    pos = pd.read_csv(out / "tables/validation_position_ig.csv", dtype={"sample_id": str})
    class_by_id = pred.set_index("sample_id").prediction_class.to_dict()
    ig_by_id = pos.groupby("sample_id").ig_signed.sum().to_dict()
    valid_rows = []
    with torch.no_grad():
        for start in range(0, len(valid["id"]), 32):
            indices = [i for i in range(start, min(start + 32, len(valid["id"])))
                       if str(valid["id"][i]) in class_by_id]
            if not indices:
                continue
            batch = clean_batch(valid, indices, device)
            classes = torch.tensor([class_by_id[str(valid["id"][i])] for i in indices],
                                   device=device, dtype=torch.long)
            observed = margin(model(batch)[0], classes).cpu().numpy()
            baseline = dict(batch)
            for m in MODS:
                baseline[m] = .1 * batch[m]
            base = margin(model(baseline)[0], classes).cpu().numpy()
            for i, v, b in zip(indices, observed, base):
                sid = str(valid["id"][i])
                ig = float(ig_by_id[sid])
                valid_rows.append({"sample_id": sid, "margin_difference": float(v - b),
                                   "sum_ig": ig, "absolute_completeness_error": abs(ig - float(v - b))})
    write_csv(out / "tables/validation_ig_completeness.csv", valid_rows)
    print(f"Validation IG completeness: n={len(valid_rows)}, mean absolute error="
          f"{np.mean([r['absolute_completeness_error'] for r in valid_rows]):.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Q3_OUTPUT_DIR)
    run(p.parse_args().output)
