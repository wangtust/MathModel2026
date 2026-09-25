"""Train and evaluate MRRF-Net variants on the fixed contest split."""

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
from torch.utils.data import DataLoader

from question2.data.dataset import FeatureStore, load_attachment3
from question2.data.missingness import TRAIN_CHOICES, hide_batch, training_weights
from question2.models.mrrf import MRRFNet, reconstruction_loss
from question2.config import ATTACHMENT3, DATA, LABEL_NAMES, VALID_SCENARIOS
from question2.evaluation.metrics import coherent_strength
from question2.evaluation.evaluator import evaluate, public_metrics, to_device
from common.reproducibility import seed_everything
from common.paths import Q2_OUTPUT_DIR


def make_model(variant: str, hidden_dim: int,
               mask_padding: bool = True,
               mask_unavailable: bool = True) -> MRRFNet:
    if variant not in {"full", "no_recon", "no_reliability", "no_consistency",
                       "plain_fusion", "plain_no_aug", "recon_only", "dual_recon"}:
        raise ValueError(variant)
    if mask_unavailable and not mask_padding:
        raise ValueError("--mask-unavailable requires --mask-padding")
    return MRRFNet(hidden_dim=hidden_dim,
                   use_reconstruction=variant not in {"no_recon", "plain_fusion", "plain_no_aug"},
                   use_reliability=variant not in {"no_reliability", "plain_fusion",
                                                   "plain_no_aug", "recon_only"},
                   dual_reconstruction=variant == "dual_recon",
                   mask_padding=mask_padding,
                   mask_unavailable=mask_unavailable)


def make_loader(store, split, batch_size, device, seed=2026):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(store.dataset(split), batch_size=batch_size,
                      shuffle=split == "train", generator=generator,
                      num_workers=0, pin_memory=device.type == "cuda")


def load_trained(output: Path, device):
    checkpoint = torch.load(output / "checkpoint.pt", map_location=device,
                            weights_only=True)
    model = make_model(checkpoint["variant"], checkpoint["hidden_dim"],
                       checkpoint.get("mask_padding", False),
                       checkpoint.get("mask_unavailable", False)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def selection_score(validation: list[dict], metric: str) -> float:
    """Choose a checkpoint using the fixed validation set only."""
    clean, audio_vision = validation[0], validation[-1]
    if metric == "official":
        return float(np.mean([v["macro_f1"] - v["mae"] / 3.0
                              for v in (clean, audio_vision)]))
    if metric == "clean_accuracy":
        return float(clean["accuracy"])
    if metric in {"condition_mean_accuracy", "balanced_accuracy"}:
        return float(np.mean([clean["accuracy"], audio_vision["accuracy"]]))
    raise ValueError(f"Unknown selection metric: {metric}")


def classification_weights(labels: np.ndarray, power: float, device) -> torch.Tensor | None:
    """Mild inverse-frequency weighting; power zero is the original loss."""
    if not 0.0 <= power <= 1.0:
        raise ValueError("class-weight-power must be between 0 and 1")
    if power == 0.0:
        return None
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=3)
    if len(counts) != 3 or np.any(counts == 0):
        raise ValueError("All three training classes must be represented")
    weights = np.power(counts.mean() / counts, power)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {DATA} on {device}; variant={args.variant}", flush=True)
    store = FeatureStore(DATA)
    train_mix = training_weights(args.train_clean_prob)
    class_weights = classification_weights(store.splits["train"]["classification"],
                                           args.class_weight_power, device)
    config = {**vars(args), "training_scenario_weights": dict(zip(TRAIN_CHOICES, train_mix)),
              "classification_weights": (class_weights.detach().cpu().tolist()
                                         if class_weights is not None else None),
              "data_path": str(DATA)}
    (output / "training_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    store.save_scaler(output / "train_scaler.npz")
    loaders = {split: make_loader(store, split, args.batch_size, device, args.seed)
               for split in ("train", "valid", "test")}
    model = make_model(args.variant, args.hidden_dim, args.mask_padding,
                       args.mask_unavailable).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best_score, best_epoch = -float("inf"), -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(args.seed + epoch)
        totals = {name: [] for name in ("total", "classification", "regression", "reconstruction", "consistency")}
        for original in loaders["train"]:
            masked = hide_batch(original, rng,
                                scenario="clean" if args.variant == "plain_no_aug" else "train",
                                train_weights=train_mix,
                                gap_layout=args.train_gap_layout)
            full = to_device(original, device)
            masked = to_device(masked, device)
            logits, strength, aux = model(masked, return_aux=True)
            loss_cls = F.cross_entropy(logits, masked["classification"],
                                       weight=class_weights)
            loss_reg = F.smooth_l1_loss(strength, masked["strength"])
            loss_rec = reconstruction_loss(aux["recovered"], full, masked)
            if args.variant in {"full", "dual_recon", "no_recon", "no_reliability"} and args.consistency_weight:
                with torch.no_grad():
                    clean = hide_batch(original, np.random.default_rng(0), scenario="clean")
                    clean_logits, clean_strength = model(to_device(clean, device))
                loss_con = (F.kl_div(F.log_softmax(logits, dim=-1),
                                     F.softmax(clean_logits, dim=-1), reduction="batchmean")
                            + F.smooth_l1_loss(strength, clean_strength))
            else:
                loss_con = logits.new_zeros(())
            loss = (loss_cls + args.regression_weight * loss_reg
                    + args.reconstruction_weight * loss_rec
                    + args.consistency_weight * loss_con)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss in epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for name, value in zip(totals, (loss, loss_cls, loss_reg, loss_rec, loss_con)):
                totals[name].append(float(value.detach().cpu()))
        vals = [evaluate(model, loaders["valid"], device, scenario=s)
                for s in VALID_SCENARIOS]
        score = selection_score(vals, args.selection_metric)
        row = {"epoch": epoch, "selection_score": score,
               **{f"train_{name}": float(np.mean(values)) for name, values in totals.items()}}
        for val in vals:
            for key in ("accuracy", "macro_f1", "mae", "pearson"):
                row[f"valid_{val['scenario']}_{key}"] = val[key]
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({"state_dict": model.state_dict(), "variant": args.variant,
                        "hidden_dim": args.hidden_dim, "seed": args.seed,
                        "epoch": epoch, "selection_score": score,
                        "selection_metric": args.selection_metric,
                        "train_gap_layout": args.train_gap_layout,
                        "train_clean_prob": args.train_clean_prob,
                        "class_weight_power": args.class_weight_power,
                        "mask_padding": args.mask_padding,
                        "mask_unavailable": args.mask_unavailable},
                       output / "checkpoint.pt")
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping epoch={epoch}; best={best_epoch}", flush=True)
            break
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    model, _ = load_trained(output, device)
    report = {"model": ("MMSA MulT + independent masked self-encoders + dual reconstruction + reliability fusion"
                        if args.variant == "dual_recon" else
                        "MMSA MulT sequence encoder + masked reconstruction + reliability fusion"),
              "variant": args.variant, "source": str(DATA), "seed": args.seed,
              "training_missing_augmentation": args.variant != "plain_no_aug",
              "training_config": config,
              "selection_metric": args.selection_metric,
              "best_epoch": best_epoch, "selection_score": best_score,
              "train_size": len(loaders["train"].dataset),
              "valid_size": len(loaders["valid"].dataset),
              "test_size": len(loaders["test"].dataset),
              "weights": {"regression": args.regression_weight,
                          "reconstruction": args.reconstruction_weight,
                          "consistency": args.consistency_weight},
              "validation": {}, "heldout_test": {}}
    for split in ("valid", "test"):
        if split == "test" and args.skip_test:
            continue
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
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.variant in {"full", "dual_recon"} and not args.skip_predict:
        predict(args, model=model, store=store)


@torch.no_grad()
def predict(args, model=None, store=None):
    output = Path(args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model, _ = load_trained(output, device)
    if store is None:
        with np.load(output / "train_scaler.npz") as file:
            scaler = {name: (file[f"{name}_mean"], file[f"{name}_std"])
                      for name in ("audio", "vision")}
    else:
        scaler = store.scaler
    samples = load_attachment3(ATTACHMENT3, scaler)
    model.eval()
    rows = []
    for start in range(0, len(samples["id"]), args.batch_size):
        end = start + args.batch_size
        batch = {name: torch.from_numpy(samples[name][start:end]) for name in
                 ("text", "audio", "vision", "content",
                  "observed_audio", "observed_vision",
                  "available_text", "available_audio", "available_vision")}
        logits, strength, aux = model(to_device(batch, device), return_aux=True)
        probs = logits.softmax(-1).cpu().numpy()
        raw = strength.cpu().numpy()
        weights = aux["weights"].cpu().numpy()
        availability = aux["availability"].cpu().numpy()
        for i, sample_id in enumerate(samples["id"][start:end]):
            cls = int(probs[i].argmax())
            rows.append({"sample_id": sample_id, "annotation": LABEL_NAMES[cls],
                         "classification_label": cls,
                         "sentiment_strength": float(coherent_strength([cls], [raw[i]])[0]),
                         "raw_sentiment_strength": float(raw[i]),
                         **{f"prob_{name.lower()}": float(probs[i, j])
                            for j, name in enumerate(LABEL_NAMES)},
                         **{f"weight_{name}": float(weights[i, j])
                            for j, name in enumerate(("text", "audio", "vision"))},
                         **{f"availability_{name}": float(availability[i, j])
                            for j, name in enumerate(("text", "audio", "vision"))}})
    with (output / "attachment3_predictions_mrrf.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} attachment-three predictions", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "predict"))
    parser.add_argument("--variant", choices=("full", "no_recon", "no_reliability", "no_consistency",
                                              "plain_fusion", "plain_no_aug", "recon_only", "dual_recon"),
                        default="full")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--regression-weight", type=float, default=0.5)
    parser.add_argument("--reconstruction-weight", type=float, default=0.2)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--train-clean-prob", type=float, default=0.15,
                        help="Share of clean training examples; rescales missing-case mix")
    parser.add_argument("--class-weight-power", type=float, default=0.0,
                        help="Inverse-frequency class weighting exponent, 0 to 1")
    parser.add_argument("--train-gap-layout", choices=("shared", "independent", "mixed"),
                        default="shared", help="How two selected modalities receive contiguous gaps")
    parser.add_argument("--mask-padding", action=argparse.BooleanOptionalAction, default=True,
                        help="Exclude special/padding keys inside MulT attention")
    parser.add_argument("--mask-unavailable", action=argparse.BooleanOptionalAction, default=True,
                        help="Exclude locally missing keys from their source modality's attention")
    parser.add_argument("--selection-metric", choices=("official", "clean_accuracy",
                                                       "condition_mean_accuracy",
                                                       "balanced_accuracy"), default="official",
                        help="Validation-only criterion; condition_mean_accuracy averages clean and A+V 30%% Accuracy")
    parser.add_argument("--skip-test", action="store_true",
                        help="Do not evaluate the held-out test during tuning")
    parser.add_argument("--skip-predict", action="store_true",
                        help="Skip attachment-three inference for repeated-seed training")
    args = parser.parse_args()
        # 自动加载最佳参数
    best_config = Path(
        "outputs/q2/tuning/best_config.json"
    )

    if best_config.exists():

        print(
            f"Loading best config: {best_config}"
        )

        with open(
            best_config,
            "r",
            encoding="utf-8"
        ) as f:
            config=json.load(f)


        mapping = {
            "lr":"lr",
            "hidden_dim":"hidden_dim",
            "batch_size":"batch_size",
            "reconstruction_weight":"reconstruction_weight",
            "consistency_weight":"consistency_weight",
        }


        for key,arg_name in mapping.items():

            if key in config:

                old=getattr(args,arg_name)

                new=config[key]

                setattr(
                    args,
                    arg_name,
                    new
                )

                print(
                    f"{arg_name}: {old} -> {new}"
                )
    args.output = args.output or str(Q2_OUTPUT_DIR / "runs" / "mrrf" / args.variant / f"seed_{args.seed}")
    if args.action == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
