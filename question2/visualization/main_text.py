"""Build the compact Q2 main-text figure set from saved experiment outputs."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
import torch

from common.paths import Q2_OUTPUT_DIR
from common.plotting import (
    ACCENT, DANGER, DOUBLE_COLUMN, EDGE, GRID, INK, LIGHT, MODALITY_COLORS, MUTED,
    apply_paper_style, panel_title, save_figure,
)
from question2.config import ATTACHMENT3, DATA, run_dir
from question2.data.dataset import FeatureStore, content_positions
from question2.evaluation.evaluator import evaluate
from question2.training.trainer import load_trained, make_loader


OUT = Q2_OUTPUT_DIR / "paper_figures"
SCENARIOS = ("text", "audio", "vision", "text+audio", "text+vision", "audio+vision")
SCENARIO_NAMES = ("T", "A", "V", "T+A", "T+V", "A+V")
MODEL_COLORS = {
    "Baseline MulT": "#7D8790",
    "MRRF full": ACCENT,
    "MRRF no reconstruction": "#75558B",
}


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required Q2 result is missing: {path}\n"
            "Run the corresponding Q2 experiment/analysis stage before paper-figures."
        )
    return path


def _selection() -> tuple[str, int, str]:
    path = _require(Q2_OUTPUT_DIR / "analysis" / "selected_model.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = data["selected_variant"]
    seed = int(data.get("prediction_checkpoint_seed", 2026))
    if selected == "mrrf_no_recon":
        return "no_recon", seed, "No reconstruction"
    if selected == "mrrf_full":
        return "full", seed, "MRRF-Net"
    raise RuntimeError(
        "The compact Q2 paper figures currently expect the validation-selected "
        f"MRRF candidate; selected_variant={selected!r}."
    )


def _cell(value: float, digits: int = 2) -> str:
    threshold = .5 * 10 ** (-digits)
    return f"{0:.{digits}f}" if abs(value) < threshold else f"{value:+.{digits}f}"


def _one_row(df: pd.DataFrame, *, context: str, **filters) -> pd.Series:
    """Select exactly one experimental row; fail loudly on incomplete/duplicated grids."""
    mask = np.ones(len(df), dtype=bool)
    for key, value in filters.items():
        if isinstance(value, float):
            mask &= np.isclose(df[key].astype(float).to_numpy(), value)
        else:
            mask &= df[key].astype(str).to_numpy() == str(value)
    rows = df.loc[mask]
    if len(rows) != 1:
        raise RuntimeError(f"{context}: expected exactly one row for {filters}, found {len(rows)}")
    return rows.iloc[0]


def robustness_figure() -> None:
    """Q2-2: degradation heatmaps instead of twelve small duration curves."""
    path = _require(Q2_OUTPUT_DIR / "figures" / "validation_comparison_grid.csv")
    df = pd.read_csv(path)
    _, _, selected_label = _selection()
    labels = ("Baseline MulT", selected_label)
    rates = (.1, .2, .3, .4, .5, .6)

    apply_paper_style(chinese=True)
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN, 5.25), constrained_layout=True)

    matrices: dict[tuple[str, str], np.ndarray] = {}
    for label in labels:
        subset = df[(df.model == label) & (df.position == "random")]
        clean = _one_row(subset, context="robustness clean", scenario="clean", missing_rate=0.0)
        for metric in ("macro_f1", "mae"):
            matrix = np.empty((len(SCENARIOS), len(rates)), float)
            for i, scenario in enumerate(SCENARIOS):
                for j, rate in enumerate(rates):
                    row = _one_row(
                        subset, context="robustness missing grid",
                        scenario=scenario, missing_rate=rate,
                    )
                    matrix[i, j] = float(row[metric]) - float(clean[metric])
            matrices[(label, metric)] = matrix

    f1_bound = max(.03, max(np.abs(matrices[(label, "macro_f1")]).max() for label in labels))
    mae_bound = max(.03, max(np.abs(matrices[(label, "mae")]).max() for label in labels))

    for row_i, label in enumerate(labels):
        for col_i, metric in enumerate(("macro_f1", "mae")):
            ax = axes[row_i, col_i]
            matrix = matrices[(label, metric)]
            bound = f1_bound if metric == "macro_f1" else mae_bound
            cmap = "RdBu" if metric == "macro_f1" else "RdBu_r"
            im = ax.imshow(
                matrix, cmap=cmap,
                norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
                aspect="auto",
            )
            letter = chr(ord("a") + row_i * 2 + col_i)
            metric_name = "Δ Macro-F1" if metric == "macro_f1" else "Δ MAE"
            panel_title(ax, letter, f"{label} · {metric_name}")
            ax.set_xticks(range(len(rates)), [f"{int(r*100)}%" for r in rates])
            ax.set_yticks(range(len(SCENARIOS)), SCENARIO_NAMES)
            ax.set_xlabel("连续缺失比例")
            if col_i == 0:
                ax.set_ylabel("缺失模态")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix[i, j]
                    ax.text(
                        j, i, _cell(value),
                        ha="center", va="center", fontsize=5.8,
                        color="white" if abs(value) > .62 * bound else INK,
                    )
            fig.colorbar(im, ax=ax, shrink=.74, pad=.02)

    fig.text(
        .5, -.015,
        "所有格均为相对同一模型 clean 条件的变化；F1 负值和 MAE 正值表示性能退化。",
        ha="center", fontsize=6.2, color=MUTED,
    )
    save_figure(fig, OUT, "fig02_q2_missing_type_rate")


def position_rate_figure() -> None:
    """Q2-3: full 10%-50% position-by-rate interaction grid."""
    path = _require(Q2_OUTPUT_DIR / "experiments" / "robustness" / "position_rate_grid.csv")
    df = pd.read_csv(path)
    rates = sorted(float(x) for x in df.missing_rate.unique())
    expected = [.1, .2, .3, .4, .5]
    if len(rates) != 5 or not np.allclose(rates, expected):
        raise RuntimeError(f"Expected position-rate levels {expected}, found {rates}")
    if len(df) != 5 * len(SCENARIOS) * 3:
        raise RuntimeError(f"Expected 90 position-rate rows, found {len(df)}")

    matrices = []
    for rate in rates:
        matrix = np.empty((len(SCENARIOS), 3), float)
        for i, scenario in enumerate(SCENARIOS):
            for j, position in enumerate(("beginning", "middle", "end")):
                row = _one_row(
                    df, context="position-rate grid",
                    scenario=scenario, missing_rate=rate, position=position,
                )
                matrix[i, j] = float(row.delta_mae)
        matrices.append(matrix)

    bound = max(.03, max(np.abs(matrix).max() for matrix in matrices))
    apply_paper_style(chinese=True)
    fig, axes = plt.subplots(
        1, 5, figsize=(DOUBLE_COLUMN, 3.55), sharey=True,
        gridspec_kw={"wspace": .08}, constrained_layout=True,
    )
    for k, (ax, rate, matrix) in enumerate(zip(axes, rates, matrices)):
        im = ax.imshow(
            matrix, cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
            aspect="auto",
        )
        panel_title(ax, chr(ord("a")+k), f"{int(rate*100)}%")
        ax.set_xticks(range(3), ("Beg.", "Mid.", "End"))
        ax.set_yticks(range(len(SCENARIOS)), SCENARIO_NAMES)
        ax.set_xlabel("缺失位置")
        if k == 0:
            ax.set_ylabel("缺失模态")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(
                    j, i, _cell(value), ha="center", va="center", fontsize=5.2,
                    color="white" if abs(value) > .62*bound else INK,
                )
    fig.colorbar(im, ax=axes, shrink=.72, pad=.025, label="Δ MAE = MAE(masked) − MAE(clean)")
    fig.text(
        .5, -.03,
        "完整覆盖 5×6×3=90 个验证扰动条件；位置区间按各样本有效正文长度计算。",
        ha="center", fontsize=6.1, color=MUTED,
    )
    save_figure(fig, OUT, "fig03_q2_position_rate_interaction")


def baseline_ablation_figure() -> None:
    """Q2-4: separate candidate comparison from controlled cumulative ablation."""
    summary = pd.read_csv(_require(
        Q2_OUTPUT_DIR / "analysis" / "all_baselines_three_seed_summary.csv"
    ))
    ablation = pd.read_csv(_require(
        Q2_OUTPUT_DIR / "experiments" / "ablation" / "cumulative_ablation_seed2026.csv"
    ))

    models = [
        "Concat-MLP", "BiGRU", "MulT no augmentation", "MMSA MulT",
        "TFR feature adapter", "MRRF full", "MRRF no reconstruction",
    ]
    short = ["Concat", "BiGRU", "MulT-noaug", "MMSA MulT", "TFR", "MRRF-full", "MRRF-no-recon"]

    test = summary[
        (summary.split == "heldout_test") &
        (summary.scenario == "clean") &
        summary.model.isin(models)
    ].copy()
    test["order"] = test.model.map({name: i for i, name in enumerate(models)})
    test = test.sort_values("order")
    if test.model.tolist() != models:
        raise RuntimeError(f"Incomplete seven-model summary: {test.model.tolist()}")

    stages = [
        "Plain fusion", "+ Missing augmentation", "+ Reconstruction",
        "+ Reliability gate", "+ Consistency (MRRF)",
    ]
    val = ablation[
        (ablation.split == "validation") &
        ablation.stage.isin(stages)
    ].copy()

    apply_paper_style(chinese=True)
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN, 5.55), constrained_layout=True)

    y = np.arange(len(models))
    for ax, metric, letter in zip(axes[0], ("macro_f1", "mae"), ("a", "b")):
        mean_col, sd_col = f"{metric}_mean", f"{metric}_sd"
        for yi, (_, row) in enumerate(test.iterrows()):
            color = MODEL_COLORS.get(row.model, "#6F8792")
            ax.errorbar(
                float(row[mean_col]), yi, xerr=float(row[sd_col]),
                fmt="o", ms=4.5, capsize=2.2, lw=1.0, color=color,
            )
        ax.set_yticks(y, short)
        ax.invert_yaxis()
        ax.set_xlabel("Macro-F1 ↑" if metric == "macro_f1" else "MAE ↓")
        panel_title(ax, letter, "七种架构：held-out clean，3 seeds mean ± SD")
        ax.grid(axis="x", color=GRID, linewidth=.5)
        ax.set_axisbelow(True)

    y2 = np.arange(len(stages))
    stage_names = ["Plain", "+Aug", "+Recon", "+Reliability", "+Consistency"]
    for ax, metric, letter in zip(axes[1], ("macro_f1", "mae"), ("c", "d")):
        for scenario, offset, color, marker in (
            ("clean", -.10, "#627E8B", "o"),
            ("audio+vision", .10, MODALITY_COLORS["audio"], "s"),
        ):
            values = [
                float(_one_row(
                    val, context="controlled ablation",
                    stage=stage, scenario=scenario,
                )[metric])
                for stage in stages
            ]
            ax.scatter(values, y2 + offset, s=27, marker=marker, color=color,
                       label="Clean" if scenario == "clean" else "A+V 30%", zorder=3)
            ax.plot(values, y2 + offset, color=color, alpha=.30, lw=.7)
        ax.set_yticks(y2, stage_names)
        ax.invert_yaxis()
        ax.set_xlabel("Macro-F1 ↑" if metric == "macro_f1" else "MAE ↓")
        panel_title(ax, letter, "控制变量累计消融 · validation · seed 2026")
        ax.grid(axis="x", color=GRID, linewidth=.5)
        ax.set_axisbelow(True)
    axes[1, 0].legend(loc="lower left", frameon=False)

    fig.text(
        .5, -.018,
        "上排是重新调优后的候选架构比较；下排才是按组件逐步加入的控制消融，二者不混作同一 ablation。",
        ha="center", fontsize=6.15, color=MUTED,
    )
    save_figure(fig, OUT, "fig04_q2_models_and_ablation")


def _strength_group(value: float) -> str:
    mag = abs(float(value))
    if mag < 1e-8:
        return "Neutral"
    if mag <= 1:
        return "Weak"
    if mag <= 2:
        return "Moderate"
    return "Strong"


def error_attribution_figure() -> None:
    """Q2-5: classification transition plus regression error by emotion magnitude."""
    variant, seed, selected_label = _selection()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    model, _ = load_trained(run_dir("mrrf", seed, variant), device)
    loader = make_loader(FeatureStore(DATA), "valid", 64, device)

    clean = evaluate(model, loader, device, "clean")
    masked = evaluate(model, loader, device, "audio+vision", rate=.3)

    if clean["ids"] != masked["ids"]:
        raise RuntimeError("Error-attribution comparison requires identical validation sample order")
    true_cls = np.asarray(clean["true_class"], int)
    clean_cls = np.asarray(clean["pred_class"], int)
    masked_cls = np.asarray(masked["pred_class"], int)
    clean_ok = clean_cls == true_cls
    masked_ok = masked_cls == true_cls
    transition = np.array([
        [np.sum(clean_ok & masked_ok), np.sum(clean_ok & ~masked_ok)],
        [np.sum(~clean_ok & masked_ok), np.sum(~clean_ok & ~masked_ok)],
    ], dtype=int)

    true_strength = np.asarray(clean["true_strength"], float)
    clean_pred = np.asarray(clean["pred_strength"], float)
    masked_pred = np.asarray(masked["pred_strength"], float)
    groups = np.asarray([_strength_group(v) for v in true_strength])
    group_order = ("Neutral", "Weak", "Moderate", "Strong")
    clean_mae = np.array([
        np.mean(np.abs(true_strength[groups == g] - clean_pred[groups == g]))
        for g in group_order
    ])
    masked_mae = np.array([
        np.mean(np.abs(true_strength[groups == g] - masked_pred[groups == g]))
        for g in group_order
    ])

    audit = pd.DataFrame({
        "sample_id": clean["ids"],
        "true_class": true_cls,
        "clean_pred_class": clean_cls,
        "masked_pred_class": masked_cls,
        "true_strength": true_strength,
        "clean_pred_strength": clean_pred,
        "masked_pred_strength": masked_pred,
        "strength_group": groups,
    })
    OUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT / "fig05_error_attribution_samples.csv", index=False, encoding="utf-8-sig")

    apply_paper_style(chinese=True)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 3.15), constrained_layout=True)

    ax = axes[0]
    im = ax.imshow(transition, cmap="Blues", aspect="auto")
    ax.set_xticks((0, 1), ("Masked correct", "Masked wrong"))
    ax.set_yticks((0, 1), ("Clean correct", "Clean wrong"))
    panel_title(ax, "a", "Clean → A+V 30% 分类状态转移")
    total = transition.sum()
    for i in range(2):
        for j in range(2):
            n = int(transition[i, j])
            ax.text(j, i, f"{n}\n({n/total:.1%})", ha="center", va="center",
                    fontsize=7.2, color="white" if n > transition.max()*.55 else INK)
    fig.colorbar(im, ax=ax, shrink=.73, pad=.02, label="样本数")

    ax = axes[1]
    x = np.arange(len(group_order))
    ax.plot(x, clean_mae, marker="o", lw=1.35, ms=4, color="#627E8B", label="Clean")
    ax.plot(x, masked_mae, marker="s", lw=1.35, ms=4,
            color=MODALITY_COLORS["audio"], label="A+V 30%")
    ax.set_xticks(x, group_order)
    ax.set_ylabel("MAE")
    panel_title(ax, "b", "按真实情感强度分层的回归误差")
    ax.grid(axis="y", color=GRID, linewidth=.5)
    ax.legend(frameon=False)
    ax.set_axisbelow(True)

    fig.text(
        .5, -.025,
        f"模型：{selected_label}，validation n={total}。强度分组：0=Neutral，(0,1]=Weak，(1,2]=Moderate，(2,3]=Strong。",
        ha="center", fontsize=6.1, color=MUTED,
    )
    save_figure(fig, OUT, "fig05_q2_error_attribution")


def _attachment3_payload(path: Path) -> dict:
    """Read the official Attachment-3 pickle layout used by the established Q2 pipeline."""
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if isinstance(raw, dict) and "test" in raw:
        data = raw["test"]
    elif isinstance(raw, dict) and "text_bert" in raw:
        data = raw
    elif isinstance(raw, dict) and len(raw) == 1:
        data = next(iter(raw.values()))
    else:
        raise ValueError(f"Unexpected Attachment-3 structure: {path}")
    required = {"text_bert", "audio", "vision"}
    if not isinstance(data, dict) or not required.issubset(data):
        raise ValueError(f"Attachment-3 sample lacks {sorted(required)}: {path}")
    return data


def attachment3_figure() -> None:
    """Q2-6: real missingness map plus all 30 unlabeled predictions."""
    variant, seed, selected_label = _selection()
    prediction = _require(run_dir("mrrf", seed, variant) / "attachment3_predictions_mrrf.csv")
    pred = pd.read_csv(prediction)

    paths = sorted(ATTACHMENT3.glob("*.pkl"))
    if len(paths) != 30:
        raise RuntimeError(f"Expected 30 Attachment-3 samples, found {len(paths)}")
    patterns = {name: [] for name in ("text", "audio", "vision")}
    missing_rows = []
    for path in paths:
        data = _attachment3_payload(path)
        # This mirrors the already validated report.py loader: text_bert defines
        # aligned content positions; Attachment 3 injects zero-filled gaps into
        # audio/vision while text content itself remains available.
        content = np.asarray(content_positions(data["text_bert"])[0], dtype=bool)
        observed = {"text": content.copy()}
        for name in ("audio", "vision"):
            arr = np.asarray(data[name])
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 2 or arr.shape[0] != len(content):
                raise ValueError(f"Unexpected {name} shape {arr.shape} in {path}")
            observed[name] = content & np.any(np.abs(arr) > 1e-12, axis=-1)

        row = {"sample_id": path.stem, "content_length": int(content.sum())}
        for name in ("text", "audio", "vision"):
            state = np.zeros(len(content), np.int8)
            state[content] = 1
            state[content & ~observed[name]] = 2
            patterns[name].append(state)
            missing = int((content & ~observed[name]).sum())
            row[f"{name}_missing_count"] = missing
            row[f"{name}_missing_fraction"] = missing / max(int(content.sum()), 1)
        missing_rows.append(row)

    missing_df = pd.DataFrame(missing_rows)
    missing_df.to_csv(OUT / "fig06_attachment3_missingness.csv", index=False, encoding="utf-8-sig")

    required_pred = {"sample_id", "sentiment_strength", "prob_negative", "prob_neutral", "prob_positive"}
    if not required_pred.issubset(pred.columns):
        raise ValueError(f"Prediction file lacks columns: {sorted(required_pred - set(pred.columns))}")
    if len(pred) != 30 or pred.sample_id.astype(str).duplicated().any():
        raise RuntimeError("Attachment-3 prediction file must contain 30 unique sample IDs")
    pred = pred.sort_values("sample_id").reset_index(drop=True)
    strength = pred.sentiment_strength.to_numpy(float)
    probs = pred[["prob_negative", "prob_neutral", "prob_positive"]].to_numpy(float)
    if not np.allclose(probs.sum(axis=1), 1.0, atol=5e-3):
        raise RuntimeError("Attachment-3 class probabilities do not sum to one")

    apply_paper_style(chinese=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, .85, 1.15])
    state_cmap = ListedColormap((LIGHT, "#DCEAE7", DANGER))
    state_norm = BoundaryNorm((-.5, .5, 1.5, 2.5), state_cmap.N)

    for i, name in enumerate(("text", "audio", "vision")):
        ax = fig.add_subplot(gs[i])
        matrix = np.stack(patterns[name])
        ax.imshow(matrix, cmap=state_cmap, norm=state_norm, aspect="auto",
                  interpolation="nearest")
        panel_title(ax, chr(ord("a")+i), f"Attachment 3 · {name.title()} 可用性")
        ax.set_ylabel("sample")
        ax.set_yticks([0, 9, 19, 29], ["01", "10", "20", "30"])
        ax.set_xticks([0, 9, 19, 29, 39, 49], ["1", "10", "20", "30", "40", "50"])
        if i == 2:
            ax.set_xlabel("对齐序列位置")
        else:
            ax.tick_params(axis="x", labelbottom=False)

    ax = fig.add_subplot(gs[3])
    x = np.arange(len(pred))
    sentiment_colors = np.where(strength < 0, "#B65A55",
                                np.where(strength > 0, "#4E7FA6", "#7A8288"))
    ax.bar(x, strength, color=sentiment_colors, width=.78)
    ax.axhline(0, color=EDGE, lw=.7)
    ax.set_xlim(-.7, len(pred)-.3)
    ax.set_ylabel("预测强度")
    ax.set_xticks([])
    panel_title(ax, "d", "30 条无标签样本的情感强度预测")

    ax = fig.add_subplot(gs[4])
    left = np.zeros(len(pred))
    colors = ("#B65A55", "#8A9197", "#4E7FA6")
    for j, label in enumerate(("Negative", "Neutral", "Positive")):
        ax.bar(x, probs[:, j], bottom=left, color=colors[j], width=.78, label=label)
        left += probs[:, j]
    ax.set_xlim(-.7, len(pred)-.3); ax.set_ylim(0, 1)
    ax.set_ylabel("类别概率")
    ax.set_xlabel("Attachment 3 sample index")
    ax.set_xticks([0, 4, 9, 14, 19, 24, 29], ["01", "05", "10", "15", "20", "25", "30"])
    panel_title(ax, "e", "三分类概率")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, -.22), frameon=False)

    audio_any = int((missing_df.audio_missing_count > 0).sum())
    vision_any = int((missing_df.vision_missing_count > 0).sum())
    audio_mean = float(missing_df.audio_missing_fraction.mean())
    vision_mean = float(missing_df.vision_missing_fraction.mean())
    fig.text(
        .5, -.015,
        f"灰=padding/special，浅绿=observed，红=正文内零填充缺失；Audio {audio_any}/30、Vision {vision_any}/30 存在局部缺失，平均缺失率分别为 {audio_mean:.1%}/{vision_mean:.1%}。"
        f"预测由验证规则选出的 {selected_label} 生成；附件3无标签，不作性能评价。",
        ha="center", fontsize=5.9, color=MUTED,
    )
    save_figure(fig, OUT, "fig06_q2_attachment3")


def write_manifest() -> None:
    manifest = {
        "main_text_figures": {
            "fig01_q2_framework": "contiguous missingness construction + MRRF framework",
            "fig02_q2_missing_type_rate": "missing modality type x duration/rate degradation",
            "fig03_q2_position_rate_interaction": "missing position x rate interaction",
            "fig04_q2_models_and_ablation": "three-seed architecture comparison + controlled cumulative ablation",
            "fig05_q2_error_attribution": "classification transition + regression error stratification",
            "fig06_q2_attachment3": "Attachment 3 missingness + full unlabeled predictions",
        },
        "supplementary_legacy_figures": [
            "fig01_learning_curves", "fig02_duration_f1", "fig03_duration_mae",
            "fig04_ablation", "fig05_position_mae", "fig06_strength_calibration",
            "fig07_confusion", "fig08_reliability_gate", "fig11_three_seeds",
            "fig12_attachment3_disagreement", "fig16_all_baselines_missingness",
            "fig18_dual_reconstruction",
        ],
        "notes": [
            "No-reconstruction is treated as a tuned candidate model, not as a controlled ablation row.",
            "Cumulative ablation remains the controlled component-addition experiment.",
            "Position-rate figure uses the complete 10%-50%, 5x6x3=90-condition grid.",
            "Attachment 3 is unlabeled and is never used for performance claims.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    robustness_figure()
    position_rate_figure()
    baseline_ablation_figure()
    error_attribution_figure()
    attachment3_figure()
    write_manifest()
    print(f"Wrote compact Q2 paper figures to {OUT}", flush=True)


if __name__ == "__main__":
    main()
