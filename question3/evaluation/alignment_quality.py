"""Quantify agreement between internal fusion gates and deletion responses.

Top-window deletion is the in-distribution Q3 local probe.  Removing an entire
modality is a supplementary, stronger perturbation outside the main Q2
training range; its ranking is reported separately.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from common.paths import Q3_OUTPUT_DIR, Q2_OUTPUT_DIR

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from transformers import AutoTokenizer

from question3.pipeline import (A4, CKPT, DATA, MODS, ROOT, SEED, TOK, FeatureStore,
                         clean_batch, load_a4, margin, write_csv)
from question2.training.trainer import load_trained


def rank_agreement(gate: np.ndarray, response: np.ndarray) -> tuple[float, bool]:
    """Spearman rho and highest-ranked agreement, excluding native absence first."""
    if len(gate) < 2:
        return np.nan, False
    x, y = rankdata(gate), rankdata(response)
    rho = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan
    return rho, bool(np.argmax(gate) == np.argmax(response))


@torch.no_grad()
def whole_modality_responses(model, arr: dict, device: torch.device,
                             split: str) -> list[dict]:
    rows = []
    for start in range(0, len(arr["id"]), 32):
        indices = list(range(start, min(start + 32, len(arr["id"]))))
        clean = clean_batch(arr, indices, device)
        logits, _, aux = model(clean, return_aux=True)
        cls = logits.argmax(1)
        reference = margin(logits, cls)
        gates = aux["weights"].cpu().numpy()
        fractions = aux["availability"].cpu().numpy()
        for j, mod in enumerate(MODS):
            removed = dict(clean)
            removed[mod] = clean[mod].clone()
            removed[f"available_{mod}"] = clean[f"available_{mod}"].clone()
            observed = removed[f"available_{mod}"].clone()
            removed[mod][observed] = 0
            removed[f"available_{mod}"][observed] = False
            after = margin(model(removed)[0], cls)
            delta = (reference - after).cpu().numpy()
            for k, i in enumerate(indices):
                rows.append({"sample_id": str(arr["id"][i]), "split": split,
                             "modality": mod, "native_available": bool(fractions[k, j] > 0),
                             "gate_weight": float(gates[k, j]),
                             "whole_modality_delta_margin": float(delta[k])})
    return rows


def rank_rows(predictions: pd.DataFrame, whole: pd.DataFrame, split: str) -> list[dict]:
    wide = whole.pivot(index="sample_id", columns="modality",
                       values="whole_modality_delta_margin")
    rows = []
    for r in predictions.itertuples(index=False):
        sid = str(r.sample_id)
        mods = [m for m in MODS if float(getattr(r, f"availability_{m}")) > 0]
        if len(mods) < 2:
            continue
        gates = np.array([float(getattr(r, f"gate_{m}")) for m in mods])
        for perturbation in ("top_window", "whole_modality"):
            response = np.array([
                float(getattr(r, f"top_delta_margin_{m}")) if perturbation == "top_window"
                else float(wide.loc[sid, m]) for m in mods])
            rho, agreement = rank_agreement(gates, response)
            rows.append({"sample_id": sid, "split": split, "perturbation": perturbation,
                         "n_available_modalities": len(mods),
                         "available_modalities": "+".join(mods),
                         "gate_top_modality": mods[int(np.argmax(gates))],
                         "response_top_modality": mods[int(np.argmax(response))],
                         "top1_agreement": agreement, "spearman_rho": rho,
                         "max_signed_response": float(response.max()),
                         "any_positive_response": bool((response > 0).any()),
                         **{f"gate_{m}": float(getattr(r, f"gate_{m}")) for m in MODS},
                         **{f"response_{m}": (float(response[mods.index(m)])
                                               if m in mods else np.nan) for m in MODS}})
    return rows


def summary(rows: list[dict]) -> list[dict]:
    rng = np.random.default_rng(SEED + 52)
    result = []
    for split in ("validation", "attachment4"):
        for kind in ("top_window", "whole_modality"):
            subset = [r for r in rows if r["split"] == split and r["perturbation"] == kind]
            same = np.array([r["top1_agreement"] for r in subset], float)
            rhos = np.array([r["spearman_rho"] for r in subset], float)
            rhos = rhos[np.isfinite(rhos)]
            gate_top = np.array([r["gate_top_modality"] for r in subset])
            response_top = np.array([r["response_top_modality"] for r in subset])
            expected = float(sum(np.mean(gate_top == m) * np.mean(response_top == m)
                                 for m in MODS))
            observed = float(same.mean())
            kappa = ((observed - expected) / (1 - expected)
                     if expected < 1 - 1e-12 else np.nan)
            def ci(values):
                boots = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
                return float(np.quantile(boots, .025)), float(np.quantile(boots, .975))
            agree_lo, agree_hi = ci(same)
            rho_lo, rho_hi = ci(rhos)
            result.append({"split": split, "perturbation": kind, "n": len(subset),
                           "n_rho_defined": len(rhos),
                           "top1_agreement": observed,
                           "top1_ci95_low": agree_lo, "top1_ci95_high": agree_hi,
                           "independent_marginal_expected_top1": expected,
                           "cohen_kappa": kappa,
                           "gate_top_text_fraction": float(np.mean(gate_top == "text")),
                           "response_top_text_fraction": float(np.mean(response_top == "text")),
                           "mean_spearman_rho": float(rhos.mean()),
                           "rho_ci95_low": rho_lo, "rho_ci95_high": rho_hi,
                           "positive_response_fraction": float(np.mean(
                               [r["any_positive_response"] for r in subset]))})
    return result


def run(out: Path):
    torch.set_num_threads(6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_trained(CKPT, device)
    model.eval()
    store = FeatureStore(DATA)
    tokenizer = AutoTokenizer.from_pretrained(TOK, use_fast=True, local_files_only=True)
    a4, _ = load_a4(store.scaler, tokenizer)
    whole = whole_modality_responses(model, store.splits["valid"], device, "validation")
    whole += whole_modality_responses(model, a4, device, "attachment4")
    write_csv(out / "tables/whole_modality_deletion.csv", whole)
    all_rows = []
    for split, source in (("validation", "validation_predictions_explained.csv"),
                          ("attachment4", "attachment4_predictions.csv")):
        pred = pd.read_csv(out / "tables" / source, dtype={"sample_id": str})
        subset = pd.DataFrame([r for r in whole if r["split"] == split])
        all_rows.extend(rank_rows(pred, subset, split))
    write_csv(out / "tables/gate_response_rank.csv", all_rows)
    write_csv(out / "tables/gate_response_rank_summary.csv", summary(all_rows))
    print(f"Gate/response ranks: {len(all_rows)} per-sample comparisons", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Q3_OUTPUT_DIR)
    run(parser.parse_args().output)
