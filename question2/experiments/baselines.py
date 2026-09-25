"""Train independent MLP/GRU and clean-trained MulT controls on the same split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import torch
import torch.nn.functional as F

from question2.models.baselines import ConcatMLP, GRUFusion
from question2.data.dataset import FeatureStore, load_attachment3
from question2.data.missingness import hide_batch
from question2.models.mult import ContestMULT
from question2.models.mrrf import reconstruction_loss
from question2.models.tfr import TFRFeatureAdapter
from question2.training.trainer import make_loader
from common.paths import Q2_OUTPUT_DIR
from question2.config import ATTACHMENT3, DATA, LABEL_NAMES, VALID_SCENARIOS
from question2.evaluation.metrics import coherent_strength
from question2.evaluation.evaluator import evaluate, public_metrics, to_device
from common.reproducibility import seed_everything


def construct(name: str) -> torch.nn.Module:
    if name == "concat_mlp":
        return ConcatMLP()
    if name == "gru":
        return GRUFusion()
    if name == "mult_no_aug":
        return ContestMULT(hidden_dim=24)
    if name == "tfr_feature":
        return TFRFeatureAdapter(hidden_dim=24)
    raise ValueError(name)


def train(args):
    output = Path(args.output) if args.output else (Q2_OUTPUT_DIR / "runs" / args.model / f"seed_{args.seed}")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = FeatureStore(DATA)
    loaders = {split: make_loader(store, split, args.batch_size, device, args.seed)
               for split in ("train", "valid", "test")}
    model = construct(args.model).to(device)
    lr = args.lr if args.lr is not None else (1e-4 if args.model in ("mult_no_aug", "tfr_feature") else 3e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_score, best_epoch = -float("inf"), -1
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(args.seed + epoch)
        losses = []
        for original in loaders["train"]:
            scenario = "clean" if args.model == "mult_no_aug" else "train"
            masked = to_device(hide_batch(original, rng, scenario=scenario), device)
            if args.model == "tfr_feature":
                logits, strength, aux = model(masked, return_aux=True)
                rec = reconstruction_loss(aux["recovered"], to_device(original, device), masked)
            else:
                logits, strength = model(masked)
                rec = logits.new_zeros(())
            loss = (F.cross_entropy(logits, masked["classification"])
                    + .5 * F.smooth_l1_loss(strength, masked["strength"])
                    + .2 * rec)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        vals = [evaluate(model, loaders["valid"], device, scenario=s)
                for s in VALID_SCENARIOS]
        score = float(np.mean([v["macro_f1"] - v["mae"] / 3.
                               for v in (vals[0], vals[-1])]))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "selection_score": score}
        for val in vals:
            for metric in ("accuracy", "macro_f1", "mae", "pearson"):
                row[f"valid_{val['scenario']}_{metric}"] = val[metric]
        history.append(row)
        print(json.dumps({"model": args.model, "seed": args.seed, **row}), flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({"state_dict": model.state_dict(), "model": args.model,
                        "seed": args.seed, "epoch": epoch, "score": score},
                       output / "checkpoint.pt")
        if epoch - best_epoch >= args.patience:
            break
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    state = torch.load(output / "checkpoint.pt", map_location=device,
                       weights_only=True)
    model.load_state_dict(state["state_dict"])
    report = {"model": args.model, "seed": args.seed, "source": str(DATA),
              "training_missing_augmentation": args.model != "mult_no_aug",
              "tfr_adaptation": ("Official MMSA TFR-Net alignment/generator/C-GATE components on fixed text features; "
                                 "skip upstream temporal-final batch-axis permutation; masked content pooling"
                                 if args.model == "tfr_feature" else None),
              "best_epoch": best_epoch, "selection_score": best_score,
              "learning_rate": lr, "batch_size": args.batch_size,
              "train_size": len(loaders["train"].dataset),
              "valid_size": len(loaders["valid"].dataset),
              "test_size": len(loaders["test"].dataset),
              "validation": {}, "heldout_test": {}}
    for split in ("valid", "test"):
        for scenario in VALID_SCENARIOS:
            result = evaluate(model, loaders[split], device, scenario=scenario)
            report["validation" if split == "valid" else "heldout_test"][scenario] = public_metrics(result)
            if split == "valid" and scenario in ("clean", "audio+vision"):
                with (output / f"valid_predictions_{scenario.replace('+','_')}.csv").open(
                    "w", newline="", encoding="utf-8-sig"
                ) as handle:
                    writer = csv.writer(handle)
                    writer.writerow(("id", "true_class", "pred_class", "true_strength",
                                     "pred_strength", "raw_pred_strength"))
                    writer.writerows(zip(result["ids"], result["true_class"],
                                         result["pred_class"], result["true_strength"],
                                         result["pred_strength"], result["raw_pred_strength"]))
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(f"Saved {args.model}, seed {args.seed}, epoch {best_epoch}, score {best_score:.4f}", flush=True)


@torch.no_grad()
def predict_attachment3(args):
    output = Path(args.output) if args.output else (Q2_OUTPUT_DIR / "runs" / args.model / f"seed_{args.seed}")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(output / "checkpoint.pt", map_location=device,
                            weights_only=True)
    if checkpoint["model"] != args.model:
        raise ValueError("Checkpoint model differs from --model")
    model = construct(args.model).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scaler = FeatureStore(DATA).scaler
    samples = load_attachment3(ATTACHMENT3, scaler)
    rows = []
    for start in range(0, len(samples["id"]), args.batch_size):
        end = start + args.batch_size
        batch = {name: torch.from_numpy(samples[name][start:end]) for name in
                 ("text", "audio", "vision", "content", "observed_audio", "observed_vision")}
        batch = hide_batch(batch, np.random.default_rng(0), scenario="clean")
        logits, strength = model(to_device(batch, device))
        probs = logits.softmax(-1).cpu().numpy()
        raw = strength.cpu().numpy()
        for i, sample_id in enumerate(samples["id"][start:end]):
            cls = int(probs[i].argmax())
            rows.append({"sample_id": sample_id, "annotation": LABEL_NAMES[cls],
                         "classification_label": cls,
                         "sentiment_strength": float(coherent_strength([cls], [raw[i]])[0]),
                         "raw_sentiment_strength": float(raw[i]),
                         **{f"prob_{name.lower()}": float(probs[i, j])
                            for j, name in enumerate(LABEL_NAMES)}})
    path = output / "attachment3_predictions_baseline.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} attachment-three predictions: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("concat_mlp", "gru", "mult_no_aug", "tfr_feature"), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()
    if args.predict_only:
        predict_attachment3(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
