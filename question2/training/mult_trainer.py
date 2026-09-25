"""Train and evaluate a contest-ready dual-head MMSA MulT baseline.

Usage: python run_q2.py train [--epochs 8] [--batch-size 32]
       python run_q2.py predict
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, confusion_matrix
from torch.utils.data import DataLoader

from question2.data.dataset import FeatureStore, load_attachment3
from question2.data.missingness import hide_batch
from question2.models.mult import ContestMULT


from question2.config import DATA, ATTACHMENT3, LABEL_NAMES, VALID_SCENARIOS
from common.paths import Q2_OUTPUT_DIR


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor)
            else value for key, value in batch.items()}


def metrics(labels, predicted, strength, estimated) -> dict:
    labels = np.asarray(labels, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    strength = np.asarray(strength, dtype=float)
    estimated = np.asarray(estimated, dtype=float)
    corr = float(np.corrcoef(strength, estimated)[0, 1]) if (
        len(strength) > 1 and np.std(strength) > 1e-8 and np.std(estimated) > 1e-8
    ) else None
    return {"accuracy": float(accuracy_score(labels, predicted)),
            "macro_f1": float(f1_score(labels, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
            "mae": float(mean_absolute_error(strength, estimated)),
            "pearson": corr, "samples": len(labels)}


def coherent_strength(predicted, raw_strength) -> np.ndarray:
    """Enforce the contest's exclusive polarity/intensity intervals."""
    classes = np.asarray(predicted, dtype=int)
    raw = np.asarray(raw_strength, dtype=float)
    return np.where(classes == 1, 0.0,
                    np.where(classes == 0, np.minimum(raw, -1e-4),
                             np.maximum(raw, 1e-4)))


@torch.no_grad()
def evaluate(model, loader, device, scenario: str, seed: int = 20260923,
             rate: float = 0.3, position: str = "random",
             gap_layout: str = "shared") -> dict:
    model.eval()
    rng = np.random.default_rng(seed)
    labels, predicted, strength, estimated, ids = [], [], [], [], []
    for batch in loader:
        masked = hide_batch(batch, rng, scenario=scenario, rate=rate,
                            position=position, gap_layout=gap_layout)
        masked = to_device(masked, device)
        logits, output = model(masked)
        labels.extend(batch["classification"].numpy().tolist())
        predicted.extend(logits.argmax(-1).cpu().numpy().tolist())
        strength.extend(batch["strength"].numpy().tolist())
        estimated.extend(output.cpu().numpy().tolist())
        ids.extend(batch["id"])
    adjusted = coherent_strength(predicted, estimated)
    result = metrics(labels, predicted, strength, adjusted)
    raw_result = metrics(labels, predicted, strength, estimated)
    result["raw_mae"] = raw_result["mae"]
    result["raw_pearson"] = raw_result["pearson"]
    result["adjusted_predictions"] = int(np.sum(np.abs(adjusted - np.asarray(estimated)) > 1e-8))
    result.update({"scenario": scenario, "missing_rate": rate if scenario != "clean" else 0.0,
                   "position": position, "gap_layout": gap_layout,
                   "ids": ids, "true_class": labels,
                   "pred_class": predicted, "true_strength": strength,
                   "pred_strength": adjusted.tolist(), "raw_pred_strength": estimated})
    return result


def public_metrics(result: dict) -> dict:
    return {k: v for k, v in result.items()
            if k not in {"ids", "true_class", "pred_class", "true_strength",
                         "pred_strength", "raw_pred_strength"}}


def train(args) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {DATA} on {device}...", flush=True)
    store = FeatureStore(DATA)
    store.save_scaler(output / "train_scaler.npz")
    loaders = {
        split: DataLoader(store.dataset(split), batch_size=args.batch_size,
                          shuffle=split == "train", num_workers=0,
                          pin_memory=device.type == "cuda")
        for split in ("train", "valid", "test")
    }
    model = ContestMULT(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best_score = -float("inf")
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(args.seed + epoch)
        losses = []
        for batch in loaders["train"]:
            batch = hide_batch(batch, rng, scenario="train")
            batch = to_device(batch, device)
            logits, strength = model(batch)
            loss_cls = F.cross_entropy(logits, batch["classification"])
            loss_reg = F.smooth_l1_loss(strength, batch["strength"])
            loss = loss_cls + args.regression_weight * loss_reg
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = [evaluate(model, loaders["valid"], device, scenario=s)
                      for s in VALID_SCENARIOS]
        robust = [validation[0], validation[-1]]
        score = float(np.mean([m["macro_f1"] - m["mae"] / 3.0 for m in robust]))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "selection_score": score}
        for val in validation:
            for key in ("accuracy", "macro_f1", "mae", "pearson"):
                row[f"valid_{val['scenario']}_{key}"] = val[key]
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "hidden_dim": args.hidden_dim, "seed": args.seed,
                        "selection_score": score}, output / "checkpoint.pt")
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
            break
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    checkpoint = torch.load(output / "checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    report = {"source": str(DATA), "mmsa_commit": "a94e65d07fa1ae0d44e552390074b29b0898edfd",
              "model": "forked MMSA MulT, masked content pooling and dual heads",
              "seed": args.seed, "best_epoch": best_epoch, "best_validation_score": best_score,
              "train_size": len(loaders["train"].dataset),
              "valid_size": len(loaders["valid"].dataset),
              "test_size": len(loaders["test"].dataset),
              "validation": {}, "heldout_test": {}}
    for split in ("valid", "test"):
        for scenario in VALID_SCENARIOS:
            result = evaluate(model, loaders[split], device, scenario=scenario)
            report["validation" if split == "valid" else "heldout_test"][scenario] = public_metrics(result)
            if split == "valid" and scenario in ("clean", "audio+vision"):
                with (output / f"valid_predictions_{scenario.replace('+', '_')}.csv").open(
                    "w", newline="", encoding="utf-8-sig"
                ) as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["id", "true_class", "pred_class", "true_strength",
                                     "pred_strength", "raw_pred_strength"])
                    writer.writerows(zip(result["ids"], result["true_class"], result["pred_class"],
                                         result["true_strength"], result["pred_strength"],
                                         result["raw_pred_strength"]))
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not args.skip_predict:
        predict(args, store=store, model=model)


def predict(args, store: FeatureStore | None = None,
            model: ContestMULT | None = None) -> None:
    output = Path(args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        checkpoint = torch.load(output / "checkpoint.pt", map_location=device, weights_only=True)
        model = ContestMULT(hidden_dim=int(checkpoint["hidden_dim"])).to(device)
        model.load_state_dict(checkpoint["state_dict"])
    if store is None:
        with np.load(output / "train_scaler.npz", allow_pickle=False) as z:
            scaler = {"audio": (z["audio_mean"], z["audio_std"]),
                      "vision": (z["vision_mean"], z["vision_std"])}
    else:
        scaler = store.scaler
    print(f"Encoding attachment-three text from {ATTACHMENT3}...", flush=True)
    samples = load_attachment3(ATTACHMENT3, scaler)
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(samples["id"]), args.batch_size):
            end = start + args.batch_size
            batch = {name: torch.from_numpy(samples[name][start:end]) for name in
                     ("text", "audio", "vision", "content", "observed_audio", "observed_vision")}
            batch = hide_batch(batch, np.random.default_rng(0), scenario="clean")
            logits, strength = model(to_device(batch, device))
            prob = torch.softmax(logits, dim=-1).cpu().numpy()
            score = strength.cpu().numpy()
            for i, sample_id in enumerate(samples["id"][start:end]):
                cls = int(np.argmax(prob[i]))
                adjusted = float(coherent_strength([cls], [score[i]])[0])
                rows.append({"sample_id": sample_id, "annotation": LABEL_NAMES[cls],
                             "classification_label": cls, "sentiment_strength": adjusted,
                             "raw_sentiment_strength": float(score[i]),
                             "prob_negative": float(prob[i, 0]),
                             "prob_neutral": float(prob[i, 1]),
                             "prob_positive": float(prob[i, 2])})
    with (output / "attachment3_predictions_baseline.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} attachment-three baseline predictions", flush=True)


def analyze(args) -> None:
    """Fixed-mask validation study for the three factors required by the task."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = FeatureStore(DATA)
    loader = DataLoader(store.dataset("valid"), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    checkpoint = torch.load(output / "checkpoint.pt", map_location=device, weights_only=True)
    model = ContestMULT(hidden_dim=int(checkpoint["hidden_dim"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    scenarios = ("text", "audio", "vision", "audio+vision")
    rates = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    clean = evaluate(model, loader, device, "clean")
    rows = [public_metrics(clean)]
    for scenario in scenarios:
        for rate in rates:
            result = evaluate(model, loader, device, scenario, rate=rate)
            rows.append(public_metrics(result))
    for scenario in scenarios:
        for position in ("beginning", "middle", "end"):
            result = evaluate(model, loader, device, scenario, rate=0.3,
                              position=position)
            rows.append(public_metrics(result))
    with (output / "validation_robustness_grid.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    colors = {"text": "#245b78", "audio": "#c67828", "vision": "#4b98a8",
              "audio+vision": "#75558b"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for scenario in scenarios:
        curve = [r for r in rows if r["scenario"] == scenario and
                 r["position"] == "random"]
        x = [0.0] + [r["missing_rate"] for r in curve]
        for ax, metric in zip(axes, ("macro_f1", "mae")):
            ax.plot(x, [clean[metric]] + [r[metric] for r in curve], marker="o", markersize=3,
                    linewidth=1.3, label=scenario, color=colors[scenario])
    axes[0].set_ylabel("Validation macro-F1")
    axes[1].set_ylabel("Validation MAE")
    for ax in axes:
        ax.set_xlabel("Injected contiguous missing fraction")
        ax.set_xticks((0.0,) + rates)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("MMSA MulT baseline: sensitivity to local missing duration",
                 x=0.07, ha="left", fontsize=9, weight="bold")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.18, wspace=0.28)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"missing_rate_curves.{suffix}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    matrix = np.array([[next(r["mae"] - clean["mae"] for r in rows
                             if r["scenario"] == scenario and
                             r["position"] == position and r["missing_rate"] == 0.3)
                        for position in ("beginning", "middle", "end")]
                       for scenario in scenarios])
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    limit = max(0.02, float(np.max(np.abs(matrix))))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(3), ["Beginning", "Middle", "End"])
    ax.set_yticks(range(4), scenarios)
    ax.set_title("Validation MAE change at 30% missing", loc="left", fontsize=9)
    for i in range(4):
        for j in range(3):
            ax.text(j, i, f"{matrix[i,j]:+.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.8, label="MAE(masked) - MAE(clean)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"missing_position_heatmap.{suffix}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    av_result = evaluate(model, loader, device, "audio+vision", rate=0.3)
    matrices = [confusion_matrix(result["true_class"], result["pred_class"],
                                 labels=[0, 1, 2]) for result in (clean, av_result)]
    shared_max = max(int(cm.max()) for cm in matrices)
    for ax, scenario, cm in zip(axes, ("clean", "audio+vision"), matrices):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=shared_max)
        ax.set_xticks(range(3), LABEL_NAMES, rotation=30, ha="right")
        ax.set_yticks(range(3), LABEL_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("(a) Clean" if scenario == "clean" else "(b) Audio + vision, 30% missing")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.suptitle("MMSA MulT baseline: validation confusion matrices",
                 x=0.08, ha="left", fontsize=9, weight="bold")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.22, wspace=0.40)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"validation_confusion.{suffix}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    lines = ["# 问题二：MMSA MulT 双任务基线结果", "",
             f"训练/验证/留出测试：{report['train_size']}/{report['valid_size']}/{report['test_size']} 条；",
             f"训练种子 {report['seed']}；验证集选定第 {report['best_epoch']} 轮。", "",
             "| 数据划分 | 场景 | Accuracy | Macro-F1 | MAE | Pearson |",
             "|:--|:--|--:|--:|--:|--:|"]
    for split, field in (("验证", "validation"), ("留出测试", "heldout_test")):
        for scenario, val in report[field].items():
            lines.append(f"| {split} | {scenario} | {val['accuracy']:.3f} | "
                         f"{val['macro_f1']:.3f} | {val['mae']:.3f} | {val['pearson']:.3f} |")
    text60 = next(r for r in rows if r["scenario"] == "text" and
                  r["position"] == "random" and r["missing_rate"] == 0.6)
    av60 = next(r for r in rows if r["scenario"] == "audio+vision" and
                r["position"] == "random" and r["missing_rate"] == 0.6)
    lines += ["", "局部缺失为正文有效位置内的连续区间；上述缺失场景比例均为 30%。"
              " 选模只使用验证集；留出测试集不参与选模。",
              f"验证集在 60% 连续文本缺失时 Macro-F1={text60['macro_f1']:.3f}、"
              f"MAE={text60['mae']:.3f}；60% 语音+视觉缺失时分别为 "
              f"{av60['macro_f1']:.3f} 和 {av60['mae']:.3f}。"
              "这表明当前基线依赖文本，后续应重点验证文本缺失时的跨模态恢复。",
              "回归指标按最终输出的强度计算：预测 Neutral 时强度设为 0，"
              "Negative/Positive 时强度分别限制在 [-3,0)/(0,3]；"
              "原始回归头数值另存于预测 CSV 的 raw_pred_strength/raw_sentiment_strength 列与 metrics.json。",
              "附件三 30 条无标签预测见 `attachment3_predictions_baseline.csv`，该文件是基线预测，"
              "尚未经过重建/可靠性融合消融后的最终模型。",
              "单次训练中遮挡语音/视觉的指标略有上升，可能反映这些特征含噪；"
              "需多随机种子与后续模型对照，不能据此断言缺失总会改善性能。",
              "数值分布与位置实验见 `validation_robustness_grid.csv`；同名 PNG/PDF 图可用于论文初稿。", ""]
    (output / "baseline_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(rows)} validation conditions and paper figures", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "predict", "analyze"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--regression-weight", type=float, default=0.5)
    parser.add_argument("--skip-predict", action="store_true",
                        help="Skip attachment-three inference for repeated-seed training")
    args = parser.parse_args()
    args.output = args.output or str(Q2_OUTPUT_DIR / "runs" / "mult" / f"seed_{args.seed}")
    if args.action == "train":
        train(args)
    elif args.action == "predict":
        predict(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
