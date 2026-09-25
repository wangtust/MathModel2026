"""Reproducible, paper-ready comparison figures from the saved checkpoints."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from matplotlib.colors import ListedColormap, BoundaryNorm

from question2.config import ATTACHMENT3, DATA
from question2.data.dataset import FeatureStore, content_positions
from question2.models.mult import ContestMULT
from question2.evaluation.evaluator import evaluate, to_device
from question2.training.trainer import load_trained, make_loader
from question2.data.missingness import hide_batch



ROOT = Path(__file__).resolve().parents[2]
from common.paths import Q2_OUTPUT_DIR
from question2.config import run_dir
OUT = Q2_OUTPUT_DIR / "figures"
LABELS = ("Negative", "Neutral", "Positive")
COLORS = {"Baseline MulT": "#7d8790", "MRRF-Net": "#176b89",
          "No reconstruction": "#c98b3e", "No reliability": "#715ba6",
          "No consistency": "#519f8c"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_history(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collect():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    store = FeatureStore(DATA)
    loader = make_loader(store, "valid", 64, device)
    base_ckpt = torch.load(run_dir("mult", 2026) / "checkpoint.pt",
                           map_location=device, weights_only=True)
    baseline = ContestMULT(hidden_dim=int(base_ckpt["hidden_dim"])).to(device)
    baseline.load_state_dict(base_ckpt["state_dict"])
    models = {"Baseline MulT": baseline}
    names = (("MRRF-Net", "full"), ("No reconstruction", "no_recon"),
             ("No reliability", "no_reliability"),
             ("No consistency", "no_consistency"))
    for label, variant in names:
        directory = run_dir("mrrf", 2026, variant)
        if (directory / "checkpoint.pt").exists():
            models[label], _ = load_trained(directory, device)
    scenarios = ("text", "audio", "vision", "text+audio", "text+vision",
                 "audio+vision")
    rates = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    results = {}
    rows = []
    for label, model in models.items():
        conditions = [("clean", 0., "random")]
        conditions += [(scenario, rate, "random") for scenario in scenarios for rate in rates]
        conditions += [(scenario, .3, pos) for scenario in scenarios
                       for pos in ("beginning", "middle", "end")]
        for scenario, rate, position in conditions:
            result = evaluate(model, loader, device, scenario, rate=rate,
                              position=position)
            results[(label, scenario, rate, position)] = result
            rows.append({"model": label, "scenario": scenario, "missing_rate": rate,
                         "position": position,
                         **{key: result[key] for key in
                            ("accuracy", "macro_f1", "mae", "pearson", "samples")}})
        print(f"Evaluated {label}: {len(conditions)} fixed-mask validation conditions", flush=True)
    with (OUT / "validation_comparison_grid.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return models, results, store, loader, device


def get(results, label, scenario, rate=0., position="random"):
    return results[(label, scenario, rate, position)]


def plot_learning():
    histories = (("Baseline MulT", run_dir("mult", 2026) / "history.csv"),
                 ("MRRF-Net", run_dir("mrrf", 2026, "full") / "history.csv"),
                 ("No reconstruction", run_dir("mrrf", 2026, "no_recon") / "history.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.65))
    for label, path in histories:
        if not path.exists():
            continue
        records = read_history(path)
        x = np.array([int(row["epoch"]) for row in records])
        values = (("train_loss" if label == "Baseline MulT" else "train_total", 0),
                  ("valid_clean_macro_f1", 1), ("valid_clean_mae", 2))
        for column, index in values:
            axes[index].plot(x, [float(row[column]) for row in records], label=label,
                             color=COLORS[label], linewidth=1.8)
    for ax, title in zip(axes, ("Training objective*", "Validation macro-F1", "Validation MAE")):
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=.18)
    axes[0].legend(frameon=False, fontsize=7)
    fig.text(.015, -.04, "* Training objectives differ between architectures; compare trends, not absolute values.",
             fontsize=7, color="#555555")
    fig.tight_layout(w_pad=2)
    save(fig, "fig01_learning_curves")


def plot_duration(results):
    scenarios = ("text", "audio", "vision", "text+audio", "text+vision",
                 "audio+vision")
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 5.6), sharex=True)
    for index, scenario in enumerate(scenarios):
        ax = axes.flat[index]
        for label in ("Baseline MulT", "MRRF-Net", "No reconstruction"):
            if (label, "clean", 0., "random") not in results:
                continue
            y = [get(results, label, "clean")["macro_f1"]]
            y += [get(results, label, scenario, rate)["macro_f1"]
                  for rate in (.1, .2, .3, .4, .5, .6)]
            ax.plot((0, .1, .2, .3, .4, .5, .6), y, marker="o", markersize=3,
                    linewidth=1.55, color=COLORS[label], label=label)
        ax.set_title(f"{scenario.capitalize()} missing", loc="left", fontsize=9)
        ax.set_ylabel("Validation macro-F1")
        ax.grid(alpha=.2)
    for ax in axes[-1]:
        ax.set_xlabel("Contiguous missing fraction")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False,
               fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.015))
    fig.tight_layout(rect=(0, .07, 1, 1))
    save(fig, "fig02_duration_f1")

    fig, axes = plt.subplots(2, 3, figsize=(10.6, 5.6), sharex=True)
    for index, scenario in enumerate(scenarios):
        ax = axes.flat[index]
        for label in ("Baseline MulT", "MRRF-Net", "No reconstruction"):
            if (label, "clean", 0., "random") not in results:
                continue
            y = [get(results, label, "clean")["mae"]]
            y += [get(results, label, scenario, rate)["mae"]
                  for rate in (.1, .2, .3, .4, .5, .6)]
            ax.plot((0, .1, .2, .3, .4, .5, .6), y, marker="o", markersize=3,
                    linewidth=1.55, color=COLORS[label], label=label)
        ax.set_title(f"{scenario.capitalize()} missing", loc="left", fontsize=9)
        ax.set_ylabel("Validation MAE")
        ax.grid(alpha=.2)
    for ax in axes[-1]:
        ax.set_xlabel("Contiguous missing fraction")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False,
               fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.015))
    fig.tight_layout(rect=(0, .07, 1, 1))
    save(fig, "fig03_duration_mae")


def plot_ablation(results):
    labels = [x for x in COLORS if (x, "clean", 0., "random") in results]
    conditions = (("Clean", "clean", 0.), ("Text", "text", .3),
                  ("Audio", "audio", .3), ("Vision", "vision", .3),
                  ("T+A", "text+audio", .3), ("T+V", "text+vision", .3),
                  ("A+V", "audio+vision", .3))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.1))
    x = np.arange(len(conditions))
    spacing = .15
    for i, label in enumerate(labels):
        for ax, metric in zip(axes, ("macro_f1", "mae")):
            vals = [get(results, label, scenario, rate)[metric]
                    for _, scenario, rate in conditions]
            ax.scatter(x + (i - (len(labels)-1)/2) * spacing, vals,
                       color=COLORS[label], s=27, label=label, zorder=3)
    for ax, metric in zip(axes, ("Macro-F1 (higher is better)", "MAE (lower is better)")):
        ax.set_xticks(x, [item[0] for item in conditions])
        ax.set_title(metric, loc="left", fontsize=9)
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
    for ax, metric in zip(axes, ("macro_f1", "mae")):
        all_values = [get(results, label, scenario, rate)[metric]
                      for label in labels for _, scenario, rate in conditions]
        lower, upper = min(all_values), max(all_values)
        pad = max(.008, (upper - lower) * .17)
        ax.set_ylim(lower - pad, upper + pad)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=7,
               loc="lower center", ncol=len(labels), bbox_to_anchor=(.5, -.015))
    fig.suptitle("Validation comparison at 30% contiguous missing", x=.07,
                 ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, .11, 1, .92))
    save(fig, "fig04_ablation")


def plot_positions(results):
    labels = [x for x in ("Baseline MulT", "MRRF-Net", "No reconstruction")
              if (x, "clean", 0., "random") in results]
    fig, axes = plt.subplots(1, len(labels), figsize=(4.3 * len(labels), 4.0),
                             squeeze=False)
    matrices = {label: np.array([[get(results, label, s, .3, p)["mae"]
                                  - get(results, label, "clean")["mae"]
                                  for p in ("beginning", "middle", "end")]
                                 for s in ("text", "audio", "vision", "text+audio",
                                           "text+vision", "audio+vision")])
                for label in labels}
    bound = max(.02, *(float(np.max(np.abs(matrix))) for matrix in matrices.values()))
    for ax, label in zip(axes[0], labels):
        matrix = matrices[label]
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound,
                          aspect="auto")
        ax.set_xticks(range(3), ("Beginning", "Middle", "End"))
        ax.set_yticks(range(6), ("Text", "Audio", "Vision", "Text+Audio",
                                 "Text+Vision", "Audio+Vision"))
        ax.set_title(label, loc="left", fontsize=9)
        for i in range(6):
            for j in range(3):
                ax.text(j, i, f"{matrix[i,j]:+.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, shrink=.72, label="Δ MAE from clean")
    fig.tight_layout()
    save(fig, "fig05_position_mae")


def plot_predictions(results):
    labels = [x for x in ("Baseline MulT", "MRRF-Net", "No reconstruction")
              if (x, "clean", 0., "random") in results]
    fig, axes = plt.subplots(1, len(labels), figsize=(4.1 * len(labels), 3.4),
                             squeeze=False)
    for ax, label in zip(axes[0], labels):
        r = get(results, label, "clean")
        true = np.asarray(r["true_strength"])
        pred = np.asarray(r["pred_strength"])
        ax.hexbin(true, pred, gridsize=24, extent=(-3, 3, -3, 3),
                  mincnt=1, cmap="Blues", linewidths=0)
        ax.plot([-3, 3], [-3, 3], color="#b45339", linewidth=1.1)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.set_xlabel("Reference strength")
        ax.set_ylabel("Predicted strength")
        ax.set_title(f"{label}  |  MAE={r['mae']:.3f}", loc="left", fontsize=9)
    fig.tight_layout()
    save(fig, "fig06_strength_calibration")

    fig, axes = plt.subplots(1, len(labels), figsize=(4.1 * len(labels), 3.3),
                             squeeze=False)
    matrices = [confusion_matrix(get(results, label, "clean")["true_class"],
                                 get(results, label, "clean")["pred_class"],
                                 labels=[0, 1, 2]) for label in labels]
    vmax = max(int(matrix.max()) for matrix in matrices)
    for ax, label, matrix in zip(axes[0], labels, matrices):
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(3), LABELS, rotation=30, ha="right")
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Reference")
        ax.set_title(label, loc="left", fontsize=9)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    save(fig, "fig07_confusion")


@torch.no_grad()
def plot_gate(model, loader, device, model_label):
    scenarios = ("clean", "text", "audio", "vision", "audio+vision")
    rates = (0., .3, .6)
    rows = []
    for scenario in scenarios:
        for rate in rates:
            if scenario == "clean" and rate != 0.:
                continue
            if scenario != "clean" and rate == 0.:
                continue
            rng = np.random.default_rng(20260923)
            weights, available = [], []
            model.eval()
            for batch in loader:
                masked = hide_batch(batch, rng, scenario=scenario, rate=rate)
                _, _, aux = model(to_device(masked, device), return_aux=True)
                weights.append(aux["weights"].cpu().numpy())
                available.append(aux["availability"].cpu().numpy())
            w = np.concatenate(weights)
            a = np.concatenate(available)
            for i, modality in enumerate(("text", "audio", "vision")):
                rows.append({"model": model_label, "scenario": scenario, "missing_rate": rate,
                             "modality": modality, "mean_gate_weight": float(w[:, i].mean()),
                             "mean_availability": float(a[:, i].mean()),
                             "samples": len(w)})
    with (OUT / "validation_reliability_weights.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.05))
    configs = (("clean", 0.), ("text", .3), ("audio", .3), ("vision", .3),
               ("audio+vision", .3), ("text", .6), ("audio+vision", .6))
    x = np.arange(len(configs))
    names = ["Clean", "T30", "A30", "V30", "A+V30", "T60", "A+V60"]
    colors = ("#245b78", "#c67828", "#4b98a8")
    for ax, field, title in zip(axes, ("mean_availability", "mean_gate_weight"),
                                ("Observed fraction", "Learned fusion weight")):
        bottom = np.zeros(len(configs))
        for modality, color in zip(("text", "audio", "vision"), colors):
            values = [next(row[field] for row in rows
                           if row["scenario"] == s and row["missing_rate"] == rate
                           and row["modality"] == modality) for s, rate in configs]
            ax.bar(x, values, bottom=bottom, color=color, width=.76,
                   label=modality.capitalize())
            bottom += values
        ax.set_xticks(x, names, rotation=35, ha="right")
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_ylim(0, 3.05 if field == "mean_availability" else 1.05)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.legend(*axes[1].get_legend_handles_labels(), frameon=False, fontsize=7,
               loc="lower center", ncol=3, bbox_to_anchor=(.5, -.02))
    fig.tight_layout(rect=(0, .08, 1, 1))
    save(fig, "fig08_reliability_gate")


def paired_bootstrap(results, repetitions=1000):
    """Paired validation intervals, sharing every resampled sample ID."""
    challengers = [name for name in ("MRRF-Net", "No reconstruction")
                   if (name, "clean", 0., "random") in results]
    if not challengers:
        return []
    rng = np.random.default_rng(20260923)
    rows = []
    for scenario, rate in (("clean", 0.), ("text", .3), ("audio+vision", .3),
                           ("text", .6), ("audio+vision", .6)):
        baseline = get(results, "Baseline MulT", scenario, rate)
        y = np.asarray(baseline["true_class"])
        strength = np.asarray(baseline["true_strength"])
        b_cls = np.asarray(baseline["pred_class"])
        b_str = np.asarray(baseline["pred_strength"])
        n = len(y)
        draws = rng.integers(0, n, size=(repetitions, n))
        for label in challengers:
            challenger = get(results, label, scenario, rate)
            if baseline["ids"] != challenger["ids"]:
                raise ValueError("Paired comparison requires identical validation sample order")
            f_cls = np.asarray(challenger["pred_class"])
            f_str = np.asarray(challenger["pred_strength"])
            f1_changes, mae_changes = [], []
            for index in draws:
                f1_changes.append(f1_score(y[index], f_cls[index], labels=[0, 1, 2],
                                           average="macro", zero_division=0)
                                  - f1_score(y[index], b_cls[index], labels=[0, 1, 2],
                                             average="macro", zero_division=0))
                mae_changes.append(np.abs(strength[index] - f_str[index]).mean()
                                   - np.abs(strength[index] - b_str[index]).mean())
            for metric, values, point in (
                ("macro_f1", f1_changes, challenger["macro_f1"] - baseline["macro_f1"]),
                ("mae", mae_changes, challenger["mae"] - baseline["mae"]),
            ):
                lower, upper = np.quantile(values, [.025, .975])
                rows.append({"challenger": label, "scenario": scenario,
                             "missing_rate": rate, "metric": metric,
                             "challenger_minus_baseline": float(point),
                             "ci95_lower": float(lower), "ci95_upper": float(upper),
                             "bootstrap_repetitions": repetitions})
    with (OUT / "paired_bootstrap_validation.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_attachment3_missingness():
    paths = sorted(ATTACHMENT3.glob("*.pkl"))
    patterns = {name: [] for name in ("text", "audio", "vision")}
    rows = []
    for path in paths:
        with path.open("rb") as handle:
            sample = pickle.load(handle)["test"]
        content = content_positions(sample["text_bert"])[0]
        observed = {"text": content}
        for name in ("audio", "vision"):
            observed[name] = content & np.any(sample[name][0] != 0, axis=-1)
        row = {"sample_id": path.stem, "content_length": int(content.sum())}
        for name in patterns:
            state = np.zeros(50, dtype=np.int8)
            state[content] = 2
            state[observed[name]] = 1
            patterns[name].append(state)
            row[f"{name}_missing_count"] = int((content & ~observed[name]).sum())
            row[f"{name}_missing_fraction"] = float(row[f"{name}_missing_count"] / content.sum())
        rows.append(row)
    with (OUT / "attachment3_missingness.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cmap = ListedColormap(("#eceff1", "#236a86", "#d26d46"))
    norm = BoundaryNorm((-.5, .5, 1.5, 2.5), 3)
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 6.3), sharey=True)
    for ax, name in zip(axes, patterns):
        matrix = np.stack(patterns[name])
        ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        ax.set_title(name.capitalize(), loc="left", fontsize=9)
        ax.set_xlabel("Aligned sequence position")
        ax.set_xticks((0, 9, 19, 29, 39, 49), (1, 10, 20, 30, 40, 50))
    axes[0].set_yticks(range(len(rows)), [r["sample_id"].replace("附件3_", "") for r in rows])
    axes[0].set_ylabel("Attachment 3 sample")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=color, label=label)
                        for color, label in (("#eceff1", "Padding / special token"),
                                             ("#236a86", "Observed"),
                                             ("#d26d46", "Zero-filled inside content"))],
               loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1))
    save(fig, "fig09_attachment3_missingness")


def selected_variant():
    repeated = {}
    for name, model, variant in (("baseline", "mult", None),
                                 ("full", "mrrf", "full"),
                                 ("no_recon", "mrrf", "no_recon")):
        paths = [run_dir(model, seed, variant) / "metrics.json"
                 for seed in (2026, 2027, 2028)]
        if all(path.exists() for path in paths):
            values = []
            for path in paths:
                report = json.loads(path.read_text(encoding="utf-8"))
                values.append(float(report.get("selection_score", report.get("best_validation_score"))))
            repeated[name] = float(np.mean(values))
    if len(repeated) == 3:
        return max(repeated, key=repeated.get), repeated, "three_seed_validation_mean"
    variants = ("full", "no_recon", "no_reliability", "no_consistency")
    scores = {}
    for variant in variants:
        path = run_dir("mrrf", 2026, variant) / "metrics.json"
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            scores[variant] = float(report["selection_score"])
    return max(scores, key=scores.get), scores, "single_seed_validation"


def plot_attachment3_predictions():
    variant, scores, basis = selected_variant()
    path = (run_dir("mult", 2026) / "attachment3_predictions_baseline.csv"
            if variant == "baseline" else
            run_dir("mrrf", 2026, variant) / "attachment3_predictions_mrrf.csv")
    if not path.exists():
        model = "mult" if variant == "baseline" else "mrrf"
        command = [sys.executable, str(ROOT / "question2" / "run.py"),
                   "predict", "--model", model, "--output", str(path.parent)]
        print(f"Prediction file missing; generating it from {path.parent}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        if not path.exists():
            raise RuntimeError(f"Prediction command completed without writing {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: row["sample_id"])
    if len(rows) != 30:
        raise ValueError("Expected 30 attachment-three predictions")
    x = np.arange(len(rows))
    names = [row["sample_id"].replace("附件3_", "") for row in rows]
    strengths = np.array([float(row["sentiment_strength"]) for row in rows])
    probs = np.array([[float(row[f"prob_{label}"]) for label in
                       ("negative", "neutral", "positive")] for row in rows])
    colors = ("#b45955", "#8c969d", "#397d9f")
    cls = np.argmax(probs, axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.9), sharex=True)
    axes[0].bar(x, strengths, color=[colors[i] for i in cls], width=.78)
    axes[0].axhline(0, color="#4d4d4d", linewidth=.8)
    axes[0].set_ylabel("Predicted strength")
    axes[0].set_ylim(-3.1, 3.1)
    axes[0].grid(axis="y", alpha=.17)
    bottom = np.zeros(len(rows))
    for j, label in enumerate(("Negative", "Neutral", "Positive")):
        axes[1].bar(x, probs[:, j], bottom=bottom, color=colors[j],
                    width=.78, label=label)
        bottom += probs[:, j]
    axes[1].set_ylabel("Predicted probability")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xticks(x, names, rotation=90)
    axes[1].set_xlabel("Attachment 3 sample ID")
    fig.legend(*axes[1].get_legend_handles_labels(), frameon=False, ncol=3,
               fontsize=7, loc="lower center", bbox_to_anchor=(.5, -.015))
    fig.suptitle(f"Unlabeled attachment 3 predictions | validation-selected: {variant}",
                 x=.09, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, .08, 1, .92))
    save(fig, "fig10_attachment3_predictions")
    # This is a visualization subset, not the seven-configuration final choice.
    (OUT / "visualization_selection.json").write_text(json.dumps({
        "selected_variant": variant, "validation_selection_scores": scores,
        "selection_basis": basis,
        "selection_rule": ("Across three training seeds, average each seed's clean and audio+vision 30% validation macro_f1 - mae/3"
                           if basis == "three_seed_validation_mean" else
                           "Single-seed mean over clean and audio+vision 30% of macro_f1 - mae/3"),
        "prediction_checkpoint_seed": 2026,
        "prediction_file": str(path)}, indent=2), encoding="utf-8")
    return variant, path


def write_validation_error_cases(results=None, selected=None, *, clean=None, av=None,
                                 label=None, output=OUT):
    if clean is None or av is None:
        label = {"baseline": "Baseline MulT", "full": "MRRF-Net",
                 "no_recon": "No reconstruction", "no_reliability": "No reliability",
                 "no_consistency": "No consistency"}[selected]
        clean = get(results, label, "clean")
        av = get(results, label, "audio+vision", .3)
    if label is None:
        raise ValueError("Error audit requires a model label")
    output.mkdir(parents=True, exist_ok=True)
    if clean["ids"] != av["ids"]:
        raise ValueError("Case comparison requires matching validation sample order")
    rows = []
    for i, sample_id in enumerate(clean["ids"]):
        true_strength = float(clean["true_strength"][i])
        pred_clean = float(clean["pred_strength"][i])
        pred_av = float(av["pred_strength"][i])
        rows.append({"id": sample_id, "true_class": LABELS[clean["true_class"][i]],
                     "pred_class_clean": LABELS[clean["pred_class"][i]],
                     "pred_class_audio_vision_30": LABELS[av["pred_class"][i]],
                     "true_strength": true_strength,
                     "pred_strength_clean": pred_clean,
                     "pred_strength_audio_vision_30": pred_av,
                     "absolute_error_clean": abs(true_strength - pred_clean),
                     "absolute_error_audio_vision_30": abs(true_strength - pred_av),
                     "class_changes_when_masked": int(clean["pred_class"][i] != av["pred_class"][i])})
    rows.sort(key=lambda row: max(row["absolute_error_clean"],
                                  row["absolute_error_audio_vision_30"]), reverse=True)
    with (output / "validation_case_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    transitions = {"correct_to_wrong": 0, "wrong_to_correct": 0,
                   "wrong_to_wrong": 0, "correct_to_correct": 0}
    confusion = {}
    for row in rows:
        was_correct = row["true_class"] == row["pred_class_clean"]
        is_correct = row["true_class"] == row["pred_class_audio_vision_30"]
        state = ("correct" if was_correct else "wrong") + "_to_" + (
            "correct" if is_correct else "wrong")
        transitions[state] += 1
        if not was_correct:
            key = (row["true_class"], row["pred_class_clean"])
            confusion[key] = confusion.get(key, 0) + 1
    bins = (("neutral", lambda value: value == 0),
            ("weak (0,1]", lambda value: 0 < value <= 1),
            ("moderate (1,2]", lambda value: 1 < value <= 2),
            ("strong (2,3]", lambda value: 2 < value <= 3))
    lines = ["# 验证集错误归因审计", "",
             f"模型：{label}；同一批 {len(rows)} 条验证样本，比较 clean 与音频+视觉共享区间 30% 遮挡。",
             "以下是错误分布与遮挡的关联，不是单个模态的因果贡献。", "",
             "## 遮挡前后分类状态", "",
             "| 状态 | 样本数 |", "|:--|--:|"]
    for state, count in transitions.items():
        lines.append(f"| {state} | {count} |")
    lines += ["", "## clean 条件下的类别混淆", "",
              "| 真实类别 → 预测类别 | 样本数 |", "|:--|--:|"]
    for pair, count in sorted(confusion.items(), key=lambda item: -item[1]):
        lines.append(f"| {pair[0]} → {pair[1]} | {count} |")
    lines += ["", "## 按真实情感强度分层", "",
              "| 绝对强度 | 样本数 | clean MAE | 遮挡 MAE |", "|:--|--:|--:|--:|"]
    for name, predicate in bins:
        group = [row for row in rows if predicate(abs(row["true_strength"]))]
        if group:
            clean_mae = np.mean([row["absolute_error_clean"] for row in group])
            masked_mae = np.mean([row["absolute_error_audio_vision_30"] for row in group])
            lines.append(f"| {name} | {len(group)} | {clean_mae:.3f} | {masked_mae:.3f} |")
    lines += ["", "## 最大强度误差样本", "",
              "| 样本 ID | 真实类别 | clean 预测 | 遮挡预测 | clean 绝对误差 | 遮挡绝对误差 |",
              "|:--|:--|:--|:--|--:|--:|"]
    for row in rows[:5]:
        lines.append(f"| {row['id']} | {row['true_class']} | {row['pred_class_clean']} | "
                     f"{row['pred_class_audio_vision_30']} | {row['absolute_error_clean']:.3f} | "
                     f"{row['absolute_error_audio_vision_30']:.3f} |")
    (output / "validation_error_attribution.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def write_report(results, bootstrap, selected, prediction_path):
    _, selection_scores, selection_basis = selected_variant()
    labels = [x for x in COLORS if (x, "clean", 0., "random") in results]
    conditions = (("Clean", "clean", 0.), ("Text 30%", "text", .3),
                  ("Audio 30%", "audio", .3), ("Vision 30%", "vision", .3),
                  ("Text+Audio 30%", "text+audio", .3),
                  ("Text+Vision 30%", "text+vision", .3),
                  ("Audio+Vision 30%", "audio+vision", .3))
    lines = ["# 问题二：缺失模态实验结果", "",
             "下表使用固定验证集、固定缺失掩码（种子 20260923）。模型选择仅依据验证集 clean 与 Audio+Vision 30% 场景；表内其余场景不参与选模。",
             "", "| 模型 | 场景 | Accuracy | Macro-F1 | MAE | Pearson |",
             "|:--|:--|--:|--:|--:|--:|"]
    for label in labels:
        for condition, scenario, rate in conditions:
            r = get(results, label, scenario, rate)
            lines.append(f"| {label} | {condition} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['mae']:.3f} | {r['pearson']:.3f} |")
    lines += ["", "## 留出测试集（只展示预设 clean 和 A+V 30% 条件）", "",
              "| 模型 | 场景 | Accuracy | Macro-F1 | MAE | Pearson |",
              "|:--|:--|--:|--:|--:|--:|"]
    for label in labels:
        path = (run_dir("mult", 2026) / "metrics.json" if label == "Baseline MulT"
                else run_dir("mrrf", 2026, {"MRRF-Net":"full","No reconstruction":"no_recon","No reliability":"no_reliability","No consistency":"no_consistency"}[label]) / "metrics.json")
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))["heldout_test"]
        for scenario in ("clean", "audio+vision"):
            r = data[scenario]
            lines.append(f"| {label} | {scenario} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['mae']:.3f} | {r['pearson']:.3f} |")
    broader = OUT / "all_baselines_three_seed_summary.csv"
    if broader.exists():
        with broader.open(newline="", encoding="utf-8-sig") as handle:
            baseline_rows = list(csv.DictReader(handle))
        all_scores = {row["model"]: float(row["selection_score_mean"])
                      for row in baseline_rows if row["split"] == "validation"
                      and row["scenario"] == "clean"}
        all_winner = max(all_scores, key=all_scores.get)
        selection_note = (f"七个配置按三种子验证选模分数比较，最高的是 {all_winner} "
                          f"({all_scores[all_winner]:.3f})；完整表见 `all_baselines_results.md`。"
                          f"本图表中的附件三预测来自 `{selected}` 的种子 2026 检查点：`{prediction_path}`。"
                          "留出测试排名与验证选模排名不完全一致，不能用测试结果重新挑选模型。")
    else:
        selection_note = (f"按预先设定的验证选择分数（{selection_basis}），当前候选模型是 `{selected}`，"
                          f"其附件三预测文件为 `{prediction_path}`；配置分数为 "
                          + ", ".join(f"{name}={value:.3f}" for name, value in selection_scores.items()) + "。")
    lines += ["", selection_note,
              "图 1–10 为种子 2026 的结果；三种子均值与标准差另见 `three_seed_results.md` 和图 11。细微差异不可写成显著优势。",
              "图 1–10 分别展示训练曲线、缺失时长、消融、缺失位置、强度预测、混淆矩阵、融合权重、附件三实际零值位置及无标签预测。强度 MAE/Pearson 使用与分类极性一致化的最终输出。",
              "附件三不含标签，不能计算其预测准确率；附件三预测图仅展示模型输出。", ""]
    full_t60 = get(results, "MRRF-Net", "text", .6)
    no_recon_t60 = get(results, "No reconstruction", "text", .6)
    lines += [f"消融提示重建存在场景权衡：验证集文本连续缺失 60% 时，完整模型 Macro-F1 "
              f"{full_t60['macro_f1']:.3f}、MAE {full_t60['mae']:.3f}；去重建版本为 "
              f"{no_recon_t60['macro_f1']:.3f}、{no_recon_t60['mae']:.3f}。"
              "单次训练中，去重建版在 clean 与 A+V 30% 的预设选模场景中略优；"
              "三种子验证均值须单独查看。附件三的文本 token 有效，"
              "不能据此断言重建在所有缺失机制下无用。", ""]
    with (OUT / "attachment3_missingness.csv").open(newline="", encoding="utf-8-sig") as handle:
        samples = list(csv.DictReader(handle))
    audio_missing = np.array([float(row["audio_missing_fraction"]) for row in samples])
    vision_missing = np.array([float(row["vision_missing_fraction"]) for row in samples])
    lines += [f"附件三 30 条样本的音频全零占正文比例为 {audio_missing.min():.0%}–{audio_missing.max():.0%}，"
              f"视觉为 {vision_missing.min():.0%}–{vision_missing.max():.0%}；"
              "文本 token 均有有效位置。零值比例仅描述输入可用性，不等于标注误差。", ""]
    if bootstrap:
        lines += ["## 配对 bootstrap：候选模型减去基线", "",
                  "仅反映固定验证样本和固定训练种子下的抽样不确定性；不覆盖训练随机性。Macro-F1 差值越高越好，MAE 差值越低越好。",
                  "", "| 模型 | 场景 | 指标 | 差值 | 95% 区间 |", "|:--|:--|:--|--:|:--|"]
        for row in bootstrap:
            lines.append(f"| {row['challenger']} | {row['scenario']} {row['missing_rate']:.0%} | {row['metric']} | "
                         f"{row['challenger_minus_baseline']:+.3f} | "
                         f"[{row['ci95_lower']:+.3f}, {row['ci95_upper']:+.3f}] |")
        lines.append("")
    (OUT / "paper_results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    models, results, _, loader, device = collect()
    plot_learning()
    plot_duration(results)
    plot_ablation(results)
    plot_positions(results)
    plot_predictions(results)
    plot_attachment3_missingness()
    selected, prediction_path = plot_attachment3_predictions()
    if selected == "full":
        plot_gate(models["MRRF-Net"], loader, device, "MRRF-Net")
    elif selected == "no_recon":
        plot_gate(models["No reconstruction"], loader, device, "No reconstruction")
    elif "MRRF-Net" in models:
        plot_gate(models["MRRF-Net"], loader, device, "MRRF-Net")
    write_validation_error_cases(results, selected)
    bootstrap = paired_bootstrap(results)
    write_report(results, bootstrap, selected, prediction_path)
    print(f"Wrote comparison tables, report and figures to {OUT}", flush=True)


if __name__ == "__main__":
    main()
