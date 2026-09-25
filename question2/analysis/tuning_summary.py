"""Summarize validation-only Q2 tuning with journal-sized PDF/PNG figures."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
from common.paths import Q2_OUTPUT_DIR
from question2.config import run_dir
INK = "#243642"
BLUE = "#245D83"
ORANGE = "#B8782C"
GREEN = "#527F6F"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#A4ADB5", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
})


def score(record: dict) -> float:
    clean = record["validation"]["clean"]
    missing = record["validation"]["audio+vision"]
    return (clean["accuracy"] + missing["accuracy"]) / 2


def load_runs(root: Path, baseline: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    paths = [baseline / "metrics.json", *sorted(root.glob("*/metrics.json"))]
    rows, directories = [], {}
    for path in paths:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        name = "historical_full" if path.parent == baseline else path.parent.name
        directories[name] = path.parent
        clean = data["validation"]["clean"]
        missing = data["validation"]["audio+vision"]
        pred_file = path.parent / "valid_predictions_clean.csv"
        if pred_file.is_file():
            pred = pd.read_csv(pred_file)
            neutral = pred.loc[pred.true_class == 1]
            neutral_recall = float((neutral.pred_class == 1).mean())
        else:
            neutral_recall = np.nan
        rows.append({
            "run": name, "variant": data.get("variant", "full"),
            "seed": data.get("seed", np.nan),
            "selection_metric": data.get("selection_metric", "official (historical)"),
            "best_epoch": data["best_epoch"],
            "mean_clean_av_accuracy": score(data),
            "clean_accuracy": clean["accuracy"], "clean_macro_f1": clean["macro_f1"],
            "clean_mae": clean["mae"], "clean_pearson": clean["pearson"],
            "av30_accuracy": missing["accuracy"], "av30_macro_f1": missing["macro_f1"],
            "av30_mae": missing["mae"], "av30_pearson": missing["pearson"],
            "clean_neutral_recall": neutral_recall,
        })
    if not rows:
        raise FileNotFoundError("No tuning metrics.json files found")
    return pd.DataFrame(rows).sort_values("mean_clean_av_accuracy", ascending=False), directories


def save(fig, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def accuracy_tradeoff(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.15, 3.5), constrained_layout=True)
    plotted = df[~df.run.str.startswith("baseline_seed")]
    for _, row in plotted.iterrows():
        old = row["run"] == "historical_full"
        ax.scatter(row.clean_accuracy, row.av30_accuracy, s=55, marker="s" if old else "o",
                   color=ORANGE if old else BLUE, edgecolor="white", linewidth=.6, zorder=3)
        offset = (5, 4)
        ax.annotate(row["run"].replace("_seed2026", ""),
                    (row.clean_accuracy, row.av30_accuracy), xytext=offset,
                    textcoords="offset points", fontsize=6.6)
    ax.axvline(float(df.loc[df.run == "historical_full", "clean_accuracy"].iloc[0]),
               color="#B6C1C8", lw=.8, ls="--")
    ax.axhline(float(df.loc[df.run == "historical_full", "av30_accuracy"].iloc[0]),
               color="#B6C1C8", lw=.8, ls="--")
    ax.grid(color="#E7EBEF", lw=.6, zorder=0)
    ax.set_xlabel("Validation accuracy: clean")
    ax.set_ylabel("Validation accuracy: audio + vision 30% missing")
    ax.set_title("Validation accuracy trade-off across tuning runs", loc="left", fontweight="bold")
    ax.margins(.17)
    save(fig, out, "fig01_validation_accuracy_tradeoff")


def metrics_panel(df: pd.DataFrame, out: Path) -> None:
    ordered = df.sort_values("mean_clean_av_accuracy")
    labels = ordered.run.str.replace("_seed2026", "", regex=False).tolist()
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(7.15, max(3.2, 1.9 + .43 * len(ordered))),
                             sharey=True, constrained_layout=True)
    for ax, key, title, xlab, limits in (
        (axes[0], "accuracy", "a  Classification", "Accuracy / Macro-F1", (0, 1)),
        (axes[1], "mae", "b  Strength prediction", "MAE ↓ / Pearson r ↑", None),
    ):
        if key == "accuracy":
            ax.plot(ordered.clean_accuracy, y + .16, "o", color=BLUE, label="Clean accuracy")
            ax.plot(ordered.av30_accuracy, y - .16, "s", color=ORANGE, label="A+V 30% accuracy")
            ax.plot(ordered.clean_macro_f1, y + .16, "x", color=BLUE, alpha=.7,
                    label="Clean Macro-F1")
            ax.plot(ordered.av30_macro_f1, y - .16, "+", color=ORANGE, alpha=.7,
                    label="A+V 30% Macro-F1")
        else:
            ax.plot(ordered.clean_mae, y + .12, "o", color=BLUE, label="Clean")
            ax.plot(ordered.av30_mae, y - .12, "s", color=ORANGE, label="A+V 30%")
            ax.plot(ordered.clean_pearson, y + .12, "x", color=BLUE, alpha=.7,
                    label="Clean Pearson")
            ax.plot(ordered.av30_pearson, y - .12, "+", color=ORANGE, alpha=.7,
                    label="A+V 30% Pearson")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(xlab)
        ax.set_yticks(y, labels)
        ax.grid(axis="x", color="#E7EBEF", lw=.6)
        if limits is not None:
            ax.set_xlim(*limits)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.28),
                  ncol=2, fontsize=6.3)
    low = min(ordered.clean_macro_f1.min(), ordered.av30_macro_f1.min(),
              ordered.clean_accuracy.min(), ordered.av30_accuracy.min())
    high = max(ordered.clean_macro_f1.max(), ordered.av30_macro_f1.max(),
               ordered.clean_accuracy.max(), ordered.av30_accuracy.max())
    axes[0].set_xlim(max(0, low - .025), min(1, high + .025))
    save(fig, out, "fig02_validation_metrics")


def confusion_panel(df: pd.DataFrame, directories: dict[str, Path], out: Path) -> None:
    candidates = df[(df.run != "historical_full") &
                    (~df.run.str.startswith("baseline_seed"))]
    if candidates.empty:
        candidates = df[df.run != "historical_full"]
    if candidates.empty:
        return
    best = candidates.iloc[0]["run"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.1), constrained_layout=True)
    for ax, name, letter in zip(axes, ("historical_full", best), ("a", "b")):
        p = pd.read_csv(directories[name] / "valid_predictions_clean.csv")
        cm = confusion_matrix(p.true_class, p.pred_class, labels=[0, 1, 2])
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i,j]}\n({norm[i,j]:.0%})", ha="center", va="center",
                        color="white" if norm[i,j] > .55 else INK, fontsize=7)
        ax.set_xticks(range(3), ["Neg", "Neu", "Pos"])
        ax.set_yticks(range(3), ["Neg", "Neu", "Pos"])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        label = "Full MRRF (reference)" if name == "historical_full" else (
            "Without reconstruction" if name.startswith("no_recon") else
            name.replace("_seed2026", "").replace("_", " ").title())
        ax.set_title(f"{letter}  {label}", loc="left", fontweight="bold")
    fig.colorbar(im, ax=axes, shrink=.75, pad=.02, label="Recall by true class")
    save(fig, out, "fig03_clean_confusion")


def paired_accuracy(df: pd.DataFrame, directories: dict[str, Path], out: Path) -> pd.DataFrame:
    """Same-sample accuracy changes against the original fixed checkpoint."""
    rng = np.random.default_rng(20260924)
    rows = []
    for condition, filename in (("clean", "valid_predictions_clean.csv"),
                                ("audio+vision", "valid_predictions_audio_vision.csv")):
        baseline = pd.read_csv(directories["historical_full"] / filename)
        if baseline.id.duplicated().any():
            raise ValueError("Duplicate validation IDs in historical checkpoint")
        for name, directory in directories.items():
            if name in {"historical_full", "baseline_seed2026"}:
                continue
            prediction = pd.read_csv(directory / filename)
            paired = baseline.merge(prediction, on="id", suffixes=("_base", "_trial"),
                                    validate="one_to_one")
            if len(paired) != len(baseline) or not np.array_equal(
                    paired.true_class_base, paired.true_class_trial):
                raise ValueError(f"Validation IDs/labels differ for {name}")
            base_correct = (paired.pred_class_base == paired.true_class_base).to_numpy()
            trial_correct = (paired.pred_class_trial == paired.true_class_trial).to_numpy()
            delta = trial_correct.astype(float) - base_correct.astype(float)
            boot = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
            rows.append({"run": name, "condition": condition, "n": len(delta),
                         "baseline_accuracy": float(base_correct.mean()),
                         "trial_accuracy": float(trial_correct.mean()),
                         "difference": float(delta.mean()),
                         "ci95_low": float(np.quantile(boot, .025)),
                         "ci95_high": float(np.quantile(boot, .975))})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result.to_csv(out / "paired_accuracy.csv", index=False)
    labels = sorted(result.run.unique())
    fig, ax = plt.subplots(figsize=(7.15, max(2.6, 1.25 + .47 * len(labels))),
                            constrained_layout=True)
    for j, (condition, marker, color) in enumerate((("clean", "o", BLUE),
                                                     ("audio+vision", "s", ORANGE))):
        sub = result[result.condition == condition].set_index("run").loc[labels]
        y = np.arange(len(labels)) + (j - .5) * .18
        value = sub.difference.to_numpy()
        ax.errorbar(value, y, xerr=[value - sub.ci95_low.to_numpy(),
                                    sub.ci95_high.to_numpy() - value],
                    fmt=marker, color=color, capsize=2, markersize=4,
                    linewidth=1.1, label="Clean" if j == 0 else "A+V 30%")
    ax.axvline(0, color="#65727C", lw=.8, ls="--")
    ax.set_yticks(range(len(labels)), [name.replace("_seed2026", "") for name in labels])
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy change versus original full checkpoint (95% paired bootstrap CI)")
    ax.set_title("Fixed-validation paired accuracy differences", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E7EBEF", lw=.6)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.25), ncol=2)
    save(fig, out / "figures", "fig04_paired_accuracy")
    return result


def summarize_seeds(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    trial = df[df.run != "historical_full"].copy()
    trial["configuration"] = trial.run.map(lambda name: re.sub(r"_seed\d+$", "", name))
    metrics = ("mean_clean_av_accuracy", "clean_accuracy", "clean_macro_f1",
               "clean_mae", "clean_pearson", "av30_accuracy", "av30_macro_f1",
               "av30_mae", "av30_pearson", "clean_neutral_recall")
    rows = []
    for name, group in trial.groupby("configuration", sort=True):
        row = {"configuration": name, "n_seeds": len(group),
               "seeds": ",".join(str(int(seed)) for seed in sorted(group.seed))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("mean_clean_av_accuracy_mean", ascending=False)
    summary.to_csv(out / "seed_summary.csv", index=False)
    if summary.n_seeds.max() > 1:
        fig, ax = plt.subplots(figsize=(7.15, max(2.7, 1.3 + .40 * len(summary))),
                                constrained_layout=True)
        order = summary.iloc[::-1]
        for j, r in enumerate(order.itertuples(index=False)):
            x = r.mean_clean_av_accuracy_mean
            sd = r.mean_clean_av_accuracy_sd
            ax.errorbar(x, j, xerr=sd if np.isfinite(sd) else None,
                        fmt="o", color=BLUE if r.n_seeds > 1 else ORANGE,
                        capsize=2, markersize=4, linewidth=1.1)
            ax.text(x + .002, j, f"n={r.n_seeds}", va="center", fontsize=6.5)
        ax.set_yticks(range(len(order)), order.configuration)
        ax.set_xlabel("Mean validation Accuracy across clean and A+V 30% (mean ± SD)")
        ax.set_title("Training-seed stability of tuning configurations", loc="left",
                     fontweight="bold")
        ax.grid(axis="x", color="#E7EBEF", lw=.6)
        ax.margins(x=.13)
        save(fig, out / "figures", "fig05_seed_stability")
    return summary


def report(df: pd.DataFrame, paired: pd.DataFrame, seeds: pd.DataFrame,
           out: Path) -> None:
    lines = ["# 问题二验证集调参实验", "",
             "这是一组基于附件二固定验证集的探索实验。排序分数是 clean 与 A+V 30% 两种验证条件 Accuracy 的平均值；论文既有正式模型仍按原先预设的 Macro-F1−MAE/3 规则选择。historical_full 使用旧规则选轮，供参照，不作为同规则消融。调参期间未评估留出 test，也未使用附件三无标签预测选模。", "",
             "|运行|选轮规则|最佳轮次|clean Acc|clean Macro-F1|clean MAE|clean Pearson|A+V30 Acc|A+V30 Macro-F1|A+V30 MAE|A+V30 Pearson|中性召回|",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        lines.append(f"|{r.run}|{r.selection_metric}|{int(r.best_epoch)}|{r.clean_accuracy:.3f}|"
                     f"{r.clean_macro_f1:.3f}|{r.clean_mae:.3f}|{r.clean_pearson:.3f}|"
                     f"{r.av30_accuracy:.3f}|{r.av30_macro_f1:.3f}|{r.av30_mae:.3f}|"
                     f"{r.av30_pearson:.3f}|{r.clean_neutral_recall:.3f}|")
    lines += ["", "图 1 比较完整输入与 A+V 局部缺失的分类准确率；图 2 同时保留 Accuracy、Macro-F1、MAE、Pearson；图 3 的每一行按真类归一化，括号是该类召回率。中性类样本较少，应结合其召回率及总体指标阅读。", "",
              "- [图1：验证集 Accuracy 权衡](figures/fig01_validation_accuracy_tradeoff.pdf)",
              "- [图2：分类与强度指标](figures/fig02_validation_metrics.pdf)",
              "- [图3：旧正式模型与表现最高的修改配置的 clean 验证集混淆矩阵](figures/fig03_clean_confusion.pdf)",
              "- [图4：相同验证样本上的 Accuracy 配对差值与 95% bootstrap 区间](figures/fig04_paired_accuracy.pdf)", "",
              "单个运行的全部四项指标与数值配置见 `run_summary.csv`、各运行目录的 `training_config.json`、`metrics.json` 和 `history.csv`。单种子筛选不能证明稳定提升。", ""]
    if (out / "historical_seed_results.md").exists():
        lines += ["利用已有 2026–2028 三种子训练历史重选相同 Accuracy 规则后的核查见 [三种子结果](historical_seed_results.md) 和 [图5](figures/fig05_three_seed_accuracy.pdf)。无重建版本的 2026 单种子优势在三种子平均中几乎消失，因此保留原正式模型与论文选模结论。", ""]
    if not paired.empty:
        lines += ["", "配对差值以旧正式 full 检查点为参照，在相同 728 条验证样本上计算；区间只反映验证样本抽样，不反映训练种子变动。", "",
                  "|运行|条件|Accuracy 差值|95% 区间|", "|---|---|---:|---:|"]
        for _, r in paired.iterrows():
            lines.append(f"|{r.run}|{r.condition}|{r.difference:+.3f}|"
                         f"[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]|")
    if seeds.n_seeds.max() > 1:
        lines += ["", "## 训练种子稳定性", "",
                  "下表的标准差是训练种子之间的差异；只训练一个种子的配置以单点显示，不能与多种子均值作稳定性结论。", "",
                  "|配置|种子数|两条件平均 Accuracy|clean Accuracy|A+V 30% Accuracy|",
                  "|---|---:|---:|---:|---:|"]
        for _, r in seeds.iterrows():
            def fmt(key):
                mean = r[f"{key}_mean"]
                sd = r[f"{key}_sd"]
                return f"{mean:.3f} ± {sd:.3f}" if np.isfinite(sd) else f"{mean:.3f}"
            lines.append(f"|{r.configuration}|{int(r.n_seeds)}|"
                         f"{fmt('mean_clean_av_accuracy')}|{fmt('clean_accuracy')}|"
                         f"{fmt('av30_accuracy')}|")
        lines += ["", "[图5：训练种子间的 Accuracy 稳定性](figures/fig05_seed_stability.pdf)"]
    (out / "validation_tuning_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Q2_OUTPUT_DIR / "experiments" / "tuning")
    parser.add_argument("--baseline", type=Path, default=run_dir("mrrf", 2026, "full"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.root / "comparison"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    df, directories = load_runs(args.root, args.baseline)
    df.to_csv(output / "run_summary.csv", index=False)
    accuracy_tradeoff(df, figures)
    metrics_panel(df, figures)
    confusion_panel(df, directories, figures)
    paired = paired_accuracy(df, directories, output)
    seeds = summarize_seeds(df, output)
    report(df, paired, seeds, output)
    print(f"Wrote {len(df)} validation runs to {output}")


if __name__ == "__main__":
    main()
