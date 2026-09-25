"""Question 3: reproducible predictions and independently tested explanations.

Run with the mathmodel interpreter.  All explanations are for the frozen Q2
checkpoint; attachment 4 is never used for fitting or parameter selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
from common.paths import ATTACHMENT2_DIR, ATTACHMENT4_DIR, Q2_OUTPUT_DIR, Q3_OUTPUT_DIR, CACHE_DIR
from question2.data.dataset import FeatureStore, content_positions, observed_positions
from question2.data.missingness import hide_batch
from question2.training.trainer import load_trained
from question2.config import LABEL_NAMES
from question2.evaluation.metrics import coherent_strength
DATA = ATTACHMENT2_DIR / "aligned_50.pkl"
A4 = ATTACHMENT4_DIR / "对齐版本"
Q2 = ROOT / "question2"
CKPT = Q2_OUTPUT_DIR / "runs" / "mrrf" / "no_recon" / "seed_2026"
LOCAL_TOK = CACHE_DIR / "models" / "bert-base-uncased"
TOK = LOCAL_TOK if LOCAL_TOK.is_dir() else "bert-base-uncased"
MODS = ("text", "audio", "vision")
SEED = 20260924


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_a4(scaler: dict, tokenizer) -> tuple[dict, dict]:
    items, texts, audios, visions, berts = {}, [], [], [], []
    paths = sorted(A4.glob("[0-9][0-9].pkl"))
    if len(paths) != 20:
        raise ValueError(f"Expected 20 attachment-4 files, got {len(paths)}")
    for path in paths:
        with path.open("rb") as handle:
            x = pickle.load(handle)
        sid = path.stem
        if str(x["id"]) != sid:
            raise ValueError(f"ID mismatch: {path}")
        raw = str(x["raw_text"])
        enc = tokenizer(raw, max_length=50, truncation=True, padding="max_length",
                        return_offsets_mapping=True)
        bert = np.asarray(x["text_bert"], dtype=np.int64)
        if bert.shape != (3, 50) or not np.array_equal(bert[0], enc["input_ids"]) or not np.array_equal(bert[1], enc["attention_mask"]):
            raise ValueError(f"Tokenizer/feature mismatch for {sid}")
        t = np.asarray(x["text"], dtype=np.float32)
        a = np.asarray(x["audio"], dtype=np.float32)
        v = np.asarray(x["vision"], dtype=np.float32)
        if t.shape != (50, 768) or a.shape != (50, 74) or v.shape != (50, 35):
            raise ValueError(f"Unexpected shape for {sid}")
        if not (A4 / "videos" / f"{sid}.mp4").is_file():
            raise FileNotFoundError(sid)
        items[sid] = {"raw_text": raw, "offsets": enc["offset_mapping"],
                      "retained_end": max((e for s, e in enc["offset_mapping"]), default=0),
                      "text_truncated": any(not c.isspace() for c in raw[max((e for s, e in enc["offset_mapping"]), default=0):])}
        texts.append(t); audios.append(a); visions.append(v); berts.append(bert)
    result = {"id": list(items), "text": np.stack(texts), "audio": np.stack(audios),
              "vision": np.stack(visions)}
    result["content"] = content_positions(np.stack(berts))
    for mod in ("audio", "vision"):
        mask = observed_positions(result[mod], result["content"])
        mean, std = scaler[mod]
        z = np.clip((result[mod] - mean) / std, -5, 5)
        result[mod] = (np.nan_to_num(z) * mask[..., None]).astype(np.float32)
        result[f"observed_{mod}"] = mask
    return result, items


def clean_batch(arr: dict, indices: list[int], device: torch.device) -> dict:
    keys = ("text", "audio", "vision", "content", "observed_audio", "observed_vision")
    raw = {k: torch.from_numpy(np.asarray(arr[k][indices])) for k in keys}
    clean = hide_batch(raw, np.random.default_rng(0), scenario="clean")
    return {k: v.to(device) for k, v in clean.items()}


def margin(logits: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
    own = logits.gather(1, cls[:, None]).squeeze(1)
    others = logits.scatter(1, cls[:, None], -torch.inf).max(1).values
    return own - others


@torch.no_grad()
def evaluate_split(model, arr, device, split: str) -> dict:
    cls_all, raw_all, ids = [], [], list(arr["id"])
    for start in range(0, len(ids), 32):
        idx = list(range(start, min(start + 32, len(ids))))
        batch = clean_batch(arr, idx, device)
        logits, strength = model(batch)
        cls_all.extend(logits.argmax(1).cpu().tolist())
        raw_all.extend(strength.cpu().tolist())
    true_cls = np.asarray(arr["classification"])
    true_str = np.asarray(arr["strength"])
    adjusted = coherent_strength(cls_all, raw_all)
    corr = float(np.corrcoef(true_str, adjusted)[0, 1])
    return {"split": split, "n": len(ids), "accuracy": accuracy_score(true_cls, cls_all),
            "macro_f1": f1_score(true_cls, cls_all, labels=[0, 1, 2], average="macro"),
            "mae": mean_absolute_error(true_str, adjusted), "pearson": corr}


def integrated_gradients(model, batch: dict, cls: torch.Tensor,
                         steps: int = 48, baseline_factor: float = .1) -> dict:
    """Midpoint IG from a 10%-amplitude reference, with fixed masks.

    Exact zero sits in a numerically steep LayerNorm region for this frozen
    MulT checkpoint.  The attenuated reference avoids a misleadingly poor
    quadrature approximation; completeness is measured separately.
    """
    x = {m: batch[m].detach() for m in MODS}
    grads = {m: torch.zeros_like(x[m]) for m in MODS}
    for k in range(steps):
        alpha = baseline_factor + (1 - baseline_factor) * (k + 0.5) / steps
        interp = dict(batch)
        for m in MODS:
            interp[m] = (alpha * x[m]).detach().requires_grad_(True)
        logits, _ = model(interp)
        g = torch.autograd.grad(margin(logits, cls).sum(),
                                [interp[m] for m in MODS])
        for m, gm in zip(MODS, g):
            grads[m] += gm.detach() * (1 - baseline_factor) / steps
    return {m: (x[m] * grads[m]).sum(-1).detach().cpu().numpy() for m in MODS}


def candidate_windows(ig: np.ndarray, valid: np.ndarray, content: np.ndarray,
                      rate: float) -> tuple[np.ndarray, np.ndarray, int, list[np.ndarray]]:
    positions = np.flatnonzero(content)
    width = max(1, min(len(positions), int(round(rate * len(positions)))))
    if not np.any(valid):
        return np.array([], int), np.array([], int), width, []
    # Fixed-width intervals in BERT content positions; native missingness is retained.
    candidates = [positions[s:s + width] for s in range(len(positions) - width + 1)]
    scores = np.asarray([np.sum(np.maximum(ig[p[valid[p]]], 0)) for p in candidates])
    if np.all(scores == 0):
        scores = np.asarray([np.abs(ig[p[valid[p]]]).sum() for p in candidates])
    top_idx = int(np.argmax(scores))
    # Same content width AND same number of native-observed positions.  This
    # prevents missing A/V positions from making random deletion artificially weak.
    counts = np.asarray([int(valid[p].sum()) for p in candidates])
    matched = np.flatnonzero(counts == counts[top_idx])
    low_idx = int(matched[np.argmin(scores[matched])])
    top, low = candidates[top_idx], candidates[low_idx]
    return top, low, width, [candidates[i] for i in matched]


@torch.no_grad()
def deletion_responses(model, one: dict, cls: int, spans: list[tuple[str, str, np.ndarray]],
                       device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Evaluate all matched perturbations together, while changing both feature
    # values and availability exactly as Q2's hide_batch does.
    n = len(spans)
    batch = {k: v.repeat((n,) + (1,) * (v.ndim - 1)) if isinstance(v, torch.Tensor) and v.ndim > 0
             else v for k, v in one.items()}
    for i, (_, mod, positions) in enumerate(spans):
        if len(positions):
            batch[mod][i, positions] = 0
            batch[f"available_{mod}"][i, positions] = False
    outs, strengths = [], []
    for start in range(0, n, 16):
        b = {k: v[start:start + 16] if isinstance(v, torch.Tensor) and v.ndim > 0 else v
             for k, v in batch.items()}
        l, s = model(b)
        outs.append(l); strengths.append(s)
    logits = torch.cat(outs)
    strength = torch.cat(strengths)
    c = torch.full((n,), cls, dtype=torch.long, device=device)
    return (margin(logits, c).cpu().numpy(),
            logits.softmax(-1)[:, cls].cpu().numpy(), strength.cpu().numpy())


def explain_one(model, one: dict, ig: dict, rng: np.random.Generator,
                rate: float, random_reps: int, device) -> tuple[dict, list[dict], list[dict]]:
    with torch.no_grad():
        logits, strength, aux = model(one, return_aux=True)
    cls = int(logits.argmax(1).item())
    base_margin = float(margin(logits, torch.tensor([cls], device=device)).item())
    base_prob = float(logits.softmax(-1)[0, cls].item())
    base_strength = float(strength.item())
    content = one["content"][0].cpu().numpy().astype(bool)
    spans, evidence, positions_log = [], [], []
    for m in MODS:
        valid = one[f"available_{m}"][0].cpu().numpy().astype(bool)
        top, low, width, matched = candidate_windows(ig[m], valid, content, rate)
        if len(top) == 0:
            evidence.append({"modality": m, "available": False, "reason": "native_features_unavailable"})
            continue
        positions = np.flatnonzero(content)
        start_top = int(np.flatnonzero(positions == top[0])[0])
        random_idx = rng.integers(0, len(matched), size=random_reps)
        spans.extend([("top", m, top), ("low", m, low)] +
                     [(f"random_{k}", m, matched[j]) for k, j in enumerate(random_idx)])
        evidence.append({"modality": m, "available": True, "token_start": int(top[0]),
                         "token_end": int(top[-1]), "window_width": width,
                         "native_valid_positions_in_window": int(valid[top].sum()),
                         "matched_random_candidate_count": len(matched),
                         "ig_signed_sum": float(ig[m][top].sum()),
                         "ig_abs_sum": float(np.abs(ig[m][top]).sum()),
                         "ig_positive_sum": float(np.maximum(ig[m][top], 0).sum()),
                         "ig_selection_start": start_top})
        for p in np.flatnonzero(content):
            positions_log.append({"modality": m, "token_index": int(p),
                                  "native_available": bool(valid[p]),
                                  "ig_signed": float(ig[m][p]),
                                  "ig_absolute": float(abs(ig[m][p])),
                                  "selected": bool(p in top)})
    if spans:
        pert_margin, pert_prob, pert_str = deletion_responses(model, one, cls, spans, device)
    else:
        pert_margin = pert_prob = pert_str = np.array([])
    response = {}
    for (kind, m, _), ma, pr, st in zip(spans, pert_margin, pert_prob, pert_str):
        response.setdefault(m, {}).setdefault(kind.split("_")[0], []).append(
            (base_margin - float(ma), base_prob - float(pr), abs(base_strength - float(st))))
    for e in evidence:
        m = e["modality"]
        if not e["available"]:
            continue
        for kind in ("top", "low", "random"):
            vals = np.asarray(response[m][kind])
            e[f"{kind}_delta_margin"] = float(vals[:, 0].mean())
            e[f"{kind}_delta_prob"] = float(vals[:, 1].mean())
            e[f"{kind}_abs_delta_strength"] = float(vals[:, 2].mean())
    effect = np.asarray([max(next((e.get("top_delta_margin", 0) for e in evidence if e["modality"] == m), 0), 0)
                         for m in MODS])
    total = float(effect.sum())
    contrib = effect / total if total > 1e-8 else np.zeros(3)
    dominant = MODS[int(np.argmax(effect))] if total > 0.05 and effect.max() > 0.02 else "unclear"
    weights = aux["weights"][0].cpu().numpy()
    avail = aux["availability"][0].cpu().numpy()
    summary = {"prediction_class": cls, "annotation": LABEL_NAMES[cls],
               "raw_strength": base_strength,
               "sentiment_strength": float(coherent_strength([cls], [base_strength])[0]),
               "margin": base_margin, "predicted_probability": base_prob,
               "dominant_modality": dominant,
               **{f"prob_{LABEL_NAMES[j].lower()}": float(logits.softmax(-1)[0, j]) for j in range(3)},
               **{f"gate_{m}": float(weights[j]) for j, m in enumerate(MODS)},
               **{f"availability_{m}": float(avail[j]) for j, m in enumerate(MODS)},
               **{f"deletion_contribution_{m}": float(contrib[j]) for j, m in enumerate(MODS)},
               **{f"top_delta_margin_{m}": float(next((e.get("top_delta_margin", 0) for e in evidence if e["modality"] == m), 0)) for m in MODS}}
    return summary, evidence, positions_log


def explain_set(model, arr, indices, device, steps, rate, random_reps, name):
    rng = np.random.default_rng(SEED)
    predictions, evidence, positions = [], [], []
    for n, idx in enumerate(indices, 1):
        sid = str(arr["id"][idx])
        one = clean_batch(arr, [idx], device)
        with torch.no_grad():
            cls = model(one)[0].argmax(1)
        ig = {m: v[0] for m, v in integrated_gradients(model, one, cls, steps).items()}
        summ, ev, pos = explain_one(model, one, ig, rng, rate, random_reps, device)
        summ.update({"sample_id": sid, "source_split": name})
        if "classification" in arr:
            summ.update({"true_class": int(arr["classification"][idx]),
                         "true_strength": float(arr["strength"][idx]),
                         "prediction_correct": int(summ["prediction_class"] == arr["classification"][idx])})
        predictions.append(summ)
        evidence.extend([{"sample_id": sid, "source_split": name, **e} for e in ev])
        positions.extend([{"sample_id": sid, "source_split": name, **p} for p in pos])
        if n == 1 or n % 20 == 0 or n == len(indices):
            print(f"{name} explanations {n}/{len(indices)}", flush=True)
    return predictions, evidence, positions


def choose_validation_indices(arr, n: int) -> list[int]:
    # Reproducible class-stratified subset, chosen without model or explanation results.
    y = np.asarray(arr["classification"])
    rng = np.random.default_rng(SEED)
    selected = []
    for c in range(3):
        pool = np.flatnonzero(y == c)
        count = min(len(pool), int(round(n * len(pool) / len(y))))
        selected.extend(rng.choice(pool, count, replace=False).tolist())
    return sorted(selected)


def bootstrap_mean(x: np.ndarray, rng: np.random.Generator, n=2000):
    x = np.asarray(x, float)
    samples = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def fidelity_tables(predictions: list[dict], evidence: list[dict]) -> list[dict]:
    pred = {r["sample_id"]: r for r in predictions}
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for m in MODS:
        subset = [e for e in evidence if e["modality"] == m and e["available"]
                  and e.get("matched_random_candidate_count", 0) >= 2]
        groups = [("all", subset),
                  ("correct", [e for e in subset if pred[e["sample_id"]]["prediction_correct"]]),
                  ("incorrect", [e for e in subset if not pred[e["sample_id"]]["prediction_correct"]]),
                  ("short_window_1to4", [e for e in subset if e["window_width"] <= 4]),
                  ("medium_window_5to8", [e for e in subset if 5 <= e["window_width"] <= 8]),
                  ("long_window_9plus", [e for e in subset if e["window_width"] >= 9])]
        for stratum, entries in groups:
            if not entries:
                continue
            for control in ("random", "low"):
                diff = np.asarray([e["top_delta_margin"] - e[f"{control}_delta_margin"] for e in entries])
                mean, lo, hi = bootstrap_mean(diff, rng)
                rows.append({"modality": m, "stratum": stratum, "comparison": f"top-minus-{control}",
                             "n": len(diff), "mean_delta_margin_difference": mean,
                             "ci95_low": lo, "ci95_high": hi,
                             "top_better_fraction": float((diff > 0).mean()),
                             "mean_top_delta_margin": float(np.mean([e["top_delta_margin"] for e in entries])),
                             "mean_control_delta_margin": float(np.mean([e[f"{control}_delta_margin"] for e in entries]))})
    return rows


def run_explanations(args, out: Path):
    torch.set_num_threads(6)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_trained(CKPT, device)
    model.eval()
    store = FeatureStore(DATA)
    token = AutoTokenizer.from_pretrained(TOK, use_fast=True, local_files_only=True)
    a4, items = load_a4(store.scaler, token)
    config = {"model": "MRRFNet_no_recon", "checkpoint": str(CKPT / "checkpoint.pt"),
              "checkpoint_sha256": hashlib.sha256((CKPT / "checkpoint.pt").read_bytes()).hexdigest(),
              "checkpoint_epoch": checkpoint["epoch"], "source_data": str(DATA),
              "attachment4": str(A4), "tokenizer": str(TOK), "device": str(device),
              "ig_steps": args.ig_steps, "ig_baseline": "0.1_times_input_fixed_masks",
              "ig_baseline_factor": .1,
              "window_fraction": args.window_fraction, "random_reps": args.random_reps,
              "validation_explanation_subset_requested": args.valid_n, "seed": SEED,
              "selection_rule": "fixed_window_max_positive_signed_IG; abs_IG_fallback",
              "dominant_rule": "max_positive_top_window_delta_margin_if_total>0.05_and_max>0.02"}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metrics = [evaluate_split(model, store.splits[s], device, s) for s in ("valid", "test")]
    write_csv(out / "tables/model_performance.csv", metrics)
    idx = choose_validation_indices(store.splits["valid"], args.valid_n)
    vp, ve, vpos = explain_set(model, store.splits["valid"], idx, device,
                               args.ig_steps, args.window_fraction, args.random_reps, "valid")
    ap, ae, apos = explain_set(model, a4, list(range(20)), device,
                               args.ig_steps, args.window_fraction, args.random_reps, "attachment4")
    write_csv(out / "tables/validation_predictions_explained.csv", vp)
    write_csv(out / "tables/validation_evidence.csv", ve)
    write_csv(out / "tables/validation_position_ig.csv", vpos)
    write_csv(out / "tables/validation_explanation_metrics.csv", fidelity_tables(vp, ve))
    write_csv(out / "tables/attachment4_predictions.csv", ap)
    write_csv(out / "tables/attachment4_evidence_raw.csv", ae)
    write_csv(out / "tables/attachment4_position_ig.csv", apos)
    (out / "attachment4_metadata.json").write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Prediction and explanation tables complete", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Q3_OUTPUT_DIR)
    p.add_argument("--valid-n", type=int, default=728)
    p.add_argument("--ig-steps", type=int, default=48)
    p.add_argument("--window-fraction", type=float, default=0.2)
    p.add_argument("--random-reps", type=int, default=8)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run_explanations(args, args.output)


if __name__ == "__main__":
    main()
