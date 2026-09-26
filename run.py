#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


# ============================================================================
# Generic helpers
# ============================================================================

def banner(text: str) -> None:
    line = "=" * 88
    print(f"\n{line}\n{text}\n{line}", flush=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def all_exist(paths: Iterable[Path]) -> bool:
    return all(path.exists() for path in paths)


def print_skip(title: str, outputs: Iterable[Path]) -> None:
    outputs = list(outputs)
    print(f"[SKIP] {title}", flush=True)
    for path in outputs:
        print(f"       ✓ {rel(path)}", flush=True)


def _display_cmd(cmd: list[str]) -> str:
    """Pretty-print subprocess commands without dumping multiline inline scripts."""
    shown = list(map(str, cmd))

    # Smoke tests use: python -c "<multiline script>".
    # Printing the whole script makes logs unreadable, so collapse it.
    if "-c" in shown:
        i = shown.index("-c")
        if i + 1 < len(shown):
            shown[i + 1] = "<inline smoke script>"

    return " ".join(shown)


def run_cmd(
    args: list[str],
    title: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    if title:
        banner(title)
    cmd = [PYTHON, *args]
    print("+", _display_cmd(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def run_module(module: str, *args: str, title: str | None = None) -> None:
    run_cmd(["-m", module, *args], title=title)


def run_stage(
    *,
    title: str,
    outputs: list[Path],
    args: list[str] | None = None,
    module: str | None = None,
    force: bool = False,
) -> None:
    """Run one stage only when its key outputs are incomplete."""
    if not force and outputs and all_exist(outputs):
        print_skip(title, outputs)
        return

    if module is not None:
        run_module(module, title=title)
    elif args is not None:
        run_cmd(args, title=title)
    else:
        raise ValueError("run_stage requires args or module")

    missing = [path for path in outputs if not path.exists()]
    if missing:
        raise RuntimeError(
            f"{title} command finished but expected outputs are missing:\n"
            + "\n".join(f"  - {rel(path)}" for path in missing)
        )


# ============================================================================
# Question 1
# ============================================================================

Q1_OUT = ROOT / "outputs" / "q1"


def q1_complete() -> bool:
    feature_dir = Q1_OUT / "features"
    feature_count = len(list(feature_dir.glob("*.npz"))) if feature_dir.exists() else 0

    required = [
        Q1_OUT / "tables" / "01_manifest.csv",
        Q1_OUT / "tables" / "02_sample_summary.csv",
        Q1_OUT / "tables" / "03_audit_issues.csv",
        Q1_OUT / "run_manifest.json",
        Q1_OUT / "paper" / "q1_results.md",
        Q1_OUT / "figures" / "fig01_modality_availability.pdf",
        Q1_OUT / "figures" / "fig02_sample_status_matrix.pdf",
        Q1_OUT / "figures" / "fig03_word_level_coverage.pdf",
    ]
    return feature_count >= 100 and all_exist(required)


def run_q1(*, force: bool, force_figures: bool) -> None:
    # Only rebuild reports/figures while preserving extracted features.
    if force_figures and not force:
        run_cmd(
            ["question1/run.py", "--reports-only"],
            title="Q1 | 强制重建报告、表格与图表",
        )
        return

    if q1_complete() and not force:
        banner("Q1 | 已完整完成")
        print(
            f"[SKIP] 检测到 >=100 个特征文件以及完整汇总/报告产物：{rel(Q1_OUT)}",
            flush=True,
        )
        return

    args = ["question1/run.py"]
    if force:
        # Explicit full rerun: disable Q1's own sample resume mechanism.
        args.append("--no-resume")

    run_cmd(
        args,
        title=(
            "Q1 | 强制从原始数据重新提取并生成全部结果"
            if force
            else "Q1 | 自动断点续跑：特征提取、时序对齐、跨语处理与全部报告"
        ),
    )


# ============================================================================
# Question 2
# ============================================================================

Q2_OUT = ROOT / "outputs" / "q2"
Q2_TUNING = Q2_OUT / "tuning"
Q2_TUNING_COMPARISON = Q2_TUNING / "comparison"


def q2_run_dir(model: str, seed: int, variant: str | None = None) -> Path:
    base = Q2_OUT / "runs"
    if model == "mrrf":
        if variant is None:
            raise ValueError("MRRF requires variant")
        return base / "mrrf" / variant / f"seed_{seed}"
    return base / model / f"seed_{seed}"


def q2_checkpoint(model: str, seed: int, variant: str | None = None) -> Path:
    return q2_run_dir(model, seed, variant) / "checkpoint.pt"


def q2_training_outputs(model: str, seed: int,
                        variant: str | None = None) -> list[Path]:
    """Artifacts that mean a training run truly finished, not merely saved a checkpoint."""
    directory = q2_run_dir(model, seed, variant)
    outputs = [
        directory / "checkpoint.pt",
        directory / "history.csv",
        directory / "metrics.json",
    ]
    if model in {"mrrf", "mult"}:
        outputs.append(directory / "train_scaler.npz")
    if model == "mrrf":
        outputs.append(directory / "training_config.json")
    return outputs


def train_q2(
    *,
    model: str,
    seed: int,
    variant: str | None = None,
    epochs: int = 30,
    patience: int = 5,
    force: bool = False,
) -> None:
    outputs = q2_training_outputs(model, seed, variant)
    label = f"{model}" + (f"/{variant}" if variant else "") + f" seed={seed}"

    if not force and all_exist(outputs):
        print_skip(f"Q2 TRAIN | {label}", outputs)
        return

    args = [
        "question2/run.py",
        "train",
        "--model", model,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--skip-predict",
    ]
    if variant is not None:
        args += ["--variant", variant]

    run_cmd(args, title=f"Q2 TRAIN | {label}")

    missing = [path for path in outputs if not path.exists()]
    if missing:
        raise RuntimeError(
            f"Q2 training {label} returned successfully but is incomplete:\n"
            + "\n".join(f"  - {rel(path)}" for path in missing)
        )


def q2_prediction_path(model: str, seed: int,
                       variant: str | None = None) -> Path:
    directory = q2_run_dir(model, seed, variant)
    return directory / (
        "attachment3_predictions_mrrf.csv"
        if model == "mrrf"
        else "attachment3_predictions_baseline.csv"
    )


def ensure_q2_prediction(
    *,
    model: str,
    seed: int = 2026,
    variant: str | None = None,
    force: bool = False,
) -> None:
    output = q2_prediction_path(model, seed, variant)
    label = f"{model}" + (f"/{variant}" if variant else "") + f" seed={seed}"

    if output.exists() and not force:
        print_skip(f"Q2 PREDICT | {label}", [output])
        return

    args = [
        "question2/run.py",
        "predict",
        "--model", model,
        "--seed", str(seed),
    ]
    if variant is not None:
        args += ["--variant", variant]

    run_cmd(args, title=f"Q2 PREDICT | {label}")

    if not output.exists():
        raise RuntimeError(f"Prediction finished but output is missing: {rel(output)}")


def q2_optuna_trial_metrics() -> list[Path]:
    return sorted(
        (Q2_OUT / "runs" / "mrrf").glob("optuna_trial_*/metrics.json")
    )


def ensure_q2_tuning() -> None:
    """
    Ensure the Optuna study has at least 30 completed trials.

    This uses the existing tuning module's objective/save_results functions, but
    runs only the number of trials still missing.  Crucially, best_config.json is
    hidden while tuning is active, because trainer.py otherwise auto-loads that
    file and would overwrite each trial's sampled parameters.
    """
    required = [
        Q2_TUNING / "study.db",
        Q2_TUNING / "trials.json",
        Q2_TUNING / "best_config.json",
    ]

    # Fast path: the study products and all 30 trained trial metrics already exist.
    if all_exist(required) and len(q2_optuna_trial_metrics()) >= 30:
        print_skip("Q2 TUNING | Optuna ≥30 completed trial artifacts", required)
        print(f"       ✓ optuna trial metrics: {len(q2_optuna_trial_metrics())}", flush=True)
        return

    banner("Q2 TUNING | 补齐 Optuna 至少 30 个完成 trial")

    try:
        import optuna
        from optuna.trial import TrialState
        from question2.experiments import tuning as tuning_module
    except Exception as exc:
        raise RuntimeError(
            "Q2 tuning requires optuna and question2.experiments.tuning"
        ) from exc

    Q2_TUNING.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{Q2_TUNING / 'study.db'}"

    study = optuna.create_study(
        study_name="mrrf_tuning",
        storage=storage,
        load_if_exists=True,
        direction=tuning_module.DIRECTION,
        sampler=optuna.samplers.TPESampler(seed=2026),
        pruner=optuna.pruners.MedianPruner(),
    )

    complete = sum(t.state == TrialState.COMPLETE for t in study.trials)
    remaining = max(0, 30 - complete)

    best = Q2_TUNING / "best_config.json"
    backup = Q2_TUNING / "best_config.before_resume.json"

    # Prevent trainer.py from replacing sampled Optuna parameters with an old best config.
    if remaining and best.exists():
        backup.write_bytes(best.read_bytes())
        best.unlink()

    try:
        if remaining:
            print(
                f"Optuna currently has {complete} completed trials; "
                f"running {remaining} additional trial(s).",
                flush=True,
            )
            study.optimize(tuning_module.objective, n_trials=remaining)
        tuning_module.save_results(study)
    except Exception:
        # Keep the previous known-good best config if the resumed tuning failed.
        if not best.exists() and backup.exists():
            best.write_bytes(backup.read_bytes())
        raise
    finally:
        if backup.exists():
            backup.unlink()

    missing = [path for path in required if not path.exists()]
    trial_count = len(q2_optuna_trial_metrics())
    if missing or trial_count < 30:
        raise RuntimeError(
            "Q2 tuning is still incomplete.\n"
            + "\n".join(f"  missing: {rel(path)}" for path in missing)
            + f"\n  trained optuna trial metrics: {trial_count} (<30)"
        )


def q2_tuning_summary(*, force: bool) -> None:
    """
    Summarize all Optuna trial runs using the actual directory layout of this project.

    tuning_summary.py defaults to outputs/q2/experiments/tuning, but the current
    tuning.py stores trial checkpoints in outputs/q2/runs/mrrf/optuna_trial_*.
    Therefore the root runner passes --root/--output explicitly.
    """
    outputs = [
        Q2_TUNING_COMPARISON / "run_summary.csv",
        Q2_TUNING_COMPARISON / "paired_accuracy.csv",
        Q2_TUNING_COMPARISON / "seed_summary.csv",
        Q2_TUNING_COMPARISON / "validation_tuning_results.md",
        Q2_TUNING_COMPARISON / "figures" / "fig01_validation_accuracy_tradeoff.pdf",
        Q2_TUNING_COMPARISON / "figures" / "fig02_validation_metrics.pdf",
        Q2_TUNING_COMPARISON / "figures" / "fig03_clean_confusion.pdf",
        Q2_TUNING_COMPARISON / "figures" / "fig04_paired_accuracy.pdf",
    ]
    run_stage(
        title="Q2 ANALYZE | Optuna 调参结果汇总与配对验证图",
        args=[
            "-m", "question2.analysis.tuning_summary",
            "--root", str(Q2_OUT / "runs" / "mrrf"),
            "--baseline", str(q2_run_dir("mrrf", 2026, "full")),
            "--output", str(Q2_TUNING_COMPARISON),
        ],
        outputs=outputs,
        force=force,
    )


def q2_mult_baseline_analysis(*, force: bool) -> None:
    """Run the MMSA MulT-specific robustness analysis that is separate from fig01–19."""
    directory = q2_run_dir("mult", 2026)
    outputs = [
        directory / "validation_robustness_grid.csv",
        directory / "missing_rate_curves.pdf",
        directory / "missing_position_heatmap.pdf",
        directory / "validation_confusion.pdf",
        directory / "baseline_results.md",
    ]
    run_stage(
        title="Q2 BASELINE | MMSA MulT 完整缺失率/位置/混淆矩阵分析",
        args=[
            "-m", "question2.training.mult_trainer",
            "analyze",
            "--seed", "2026",
            "--output", str(directory),
        ],
        outputs=outputs,
        force=force,
    )


def q2_main_figure_outputs() -> list[Path]:
    return [
        Q2_OUT / "figures" / f"fig{i:02d}_{name}.pdf"
        for i, name in [
            (1, "learning_curves"),
            (2, "duration_f1"),
            (3, "duration_mae"),
            (4, "ablation"),
            (5, "position_mae"),
            (6, "strength_calibration"),
            (7, "confusion"),
            (8, "reliability_gate"),
            (9, "attachment3_missingness"),
            (10, "attachment3_predictions"),
        ]
    ]


def ensure_q2_core_visualization(*, force: bool = False) -> None:
    """
    Create fig01–10 plus validation_comparison_grid.csv.

    model_comparison.py requires validation_comparison_grid.csv for fig16, so
    this stage must exist before the seven-model comparison.
    """
    outputs = [
        Q2_OUT / "figures" / "validation_comparison_grid.csv",
        Q2_OUT / "figures" / "paired_bootstrap_validation.csv",
        Q2_OUT / "figures" / "attachment3_missingness.csv",
        Q2_OUT / "figures" / "visualization_selection.json",
        Q2_OUT / "figures" / "paper_results.md",
        *q2_main_figure_outputs(),
    ]
    run_stage(
        title="Q2 FIGURES | 核心 fig01–fig10 与 validation_comparison_grid",
        args=["question2/run.py", "visualize"],
        outputs=outputs,
        force=force,
    )


def q2_numbered_figure_outputs() -> list[Path]:
    """All numbered Q2 figures fig01–fig19 defined by the current restructured code."""
    return [
        *q2_main_figure_outputs(),
        Q2_OUT / "experiments" / "repeats" / "fig11_three_seeds.pdf",
        Q2_OUT / "analysis" / "fig12_attachment3_disagreement.pdf",
        Q2_OUT / "figures" / "fig13_mrrf_architecture.pdf",
        Q2_OUT / "figures" / "fig14_contiguous_mask_example.pdf",
        Q2_OUT / "analysis" / "fig15_baseline_comparison.pdf",
        Q2_OUT / "analysis" / "fig16_all_baselines_missingness.pdf",
        Q2_OUT / "experiments" / "ablation" / "fig17_cumulative_ablation.pdf",
        Q2_OUT / "experiments" / "ablation" / "fig18_dual_reconstruction.pdf",
        Q2_OUT / "experiments" / "robustness" / "fig19_position_rate_interaction.pdf",
    ]


def q2_postprocessing(*, force: bool, figures_only: bool = False) -> None:
    """
    Full Q2 post-training pipeline.

    Dependency order matters:
      - visualize first creates validation_comparison_grid.csv
      - model_comparison consumes that grid for fig16
      - visualize is refreshed once after model_comparison so paper_results.md
        can include the seven-model three-seed comparison.
    """

    # Standalone MMSA MulT diagnostic analysis (3 extra PDFs).
    q2_mult_baseline_analysis(force=force)

    # Optuna tuning summary (4 guaranteed PDFs for the current trial layout).
    q2_tuning_summary(force=force)

    # fig11
    run_stage(
        title="Q2 EXP | 三随机种子稳定性",
        args=["question2/run.py", "experiment", "--name", "repeats"],
        outputs=[
            Q2_OUT / "experiments" / "repeats" / "three_seed_metrics.csv",
            Q2_OUT / "experiments" / "repeats" / "three_seed_summary.csv",
            Q2_OUT / "experiments" / "repeats" / "three_seed_results.md",
            Q2_OUT / "experiments" / "repeats" / "fig11_three_seeds.pdf",
        ],
        force=force,
    )

    # fig17
    run_stage(
        title="Q2 EXP | 累计消融",
        args=[
            "question2/run.py",
            "experiment",
            "--name", "ablation",
            "--summary-only",
        ],
        outputs=[
            Q2_OUT / "experiments" / "ablation" / "cumulative_ablation_seed2026.csv",
            Q2_OUT / "experiments" / "ablation" / "cumulative_ablation_results.md",
            Q2_OUT / "experiments" / "ablation" / "fig17_cumulative_ablation.pdf",
        ],
        force=force,
    )

    # fig18
    run_stage(
        title="Q2 EXP | 双路重建比较（55 条固定条件 + 三种子汇总）",
        args=["question2/run.py", "experiment", "--name", "dual_reconstruction"],
        outputs=[
            Q2_OUT / "experiments" / "ablation" / "dual_reconstruction_grid.csv",
            Q2_OUT / "experiments" / "ablation" / "dual_reconstruction_three_seed.csv",
            Q2_OUT / "experiments" / "ablation" / "dual_reconstruction_results.md",
            Q2_OUT / "experiments" / "ablation" / "fig18_dual_reconstruction.pdf",
        ],
        force=force,
    )

    # Missing duration is a required numerical experiment even though it has no
    # numbered PDF in its own module.
    if not figures_only:
        run_stage(
            title="Q2 EXP | 单模态缺失持续比例实验",
            args=["question2/run.py", "experiment", "--name", "missing_duration"],
            outputs=[
                Q2_OUT / "experiments" / "missing_duration" / "duration_results.csv",
            ],
            force=force,
        )

    # fig19 + independent-gap audit.
    run_stage(
        title="Q2 EXP | 缺失比例 × 缺失位置 + gap layout 鲁棒性",
        args=["question2/run.py", "experiment", "--name", "robustness"],
        outputs=[
            Q2_OUT / "experiments" / "robustness" / "position_rate_grid.csv",
            Q2_OUT / "experiments" / "robustness" / "gap_layout_validation.csv",
            Q2_OUT / "experiments" / "robustness" / "position_rate_results.md",
            Q2_OUT / "experiments" / "robustness" / "fig19_position_rate_interaction.pdf",
        ],
        force=force,
    )

    # First visualization pass is a hard dependency of model_comparison.fig16.
    ensure_q2_core_visualization(force=force)

    # fig15/16 + seven-model, three-seed metrics.
    comparison_outputs = [
        Q2_OUT / "analysis" / "all_baselines_three_seed_metrics.csv",
        Q2_OUT / "analysis" / "all_baselines_three_seed_summary.csv",
        Q2_OUT / "analysis" / "all_baselines_validation_robustness.csv",
        Q2_OUT / "analysis" / "all_baselines_results.md",
        Q2_OUT / "analysis" / "fig15_baseline_comparison.pdf",
        Q2_OUT / "analysis" / "fig16_all_baselines_missingness.pdf",
    ]
    analysis_was_missing = not all_exist(comparison_outputs)
    run_stage(
        title="Q2 ANALYZE | 七配置 × 三种子综合比较",
        args=["question2/run.py", "analyze", "--summary-only"],
        outputs=comparison_outputs,
        force=force,
    )

    # fig12
    run_stage(
        title="Q2 ANALYZE | 附件3 full vs no_recon 分歧",
        module="question2.analysis.attachment3_comparison",
        outputs=[
            Q2_OUT / "analysis" / "attachment3_model_disagreement.csv",
            Q2_OUT / "analysis" / "attachment3_disagreement_summary.md",
            Q2_OUT / "analysis" / "fig12_attachment3_disagreement.pdf",
        ],
        force=force,
    )

    # fig13/14
    run_stage(
        title="Q2 FIGURES | MRRF结构图与连续缺失示意图",
        module="question2.visualization.architecture",
        outputs=[
            Q2_OUT / "figures" / "fig13_mrrf_architecture.pdf",
            Q2_OUT / "figures" / "fig14_contiguous_mask_example.pdf",
        ],
        force=force,
    )

    if not figures_only:
        run_stage(
            title="Q2 ANALYZE | 验证集错误分析",
            args=["question2/run.py", "error-analysis"],
            outputs=[
                Q2_OUT / "analysis" / "error_analysis" / "error_cases.csv",
                Q2_OUT / "analysis" / "error_analysis" / "confidence_analysis.csv",
                Q2_OUT / "analysis" / "error_analysis" / "error_summary.json",
                Q2_OUT / "analysis" / "error_analysis" / "missing_pattern.json",
            ],
            force=force,
        )

    # Refresh paper_results.md after the seven-model summary first appears.
    # On later resume runs, this is skipped unless --force / --force-figures is used.
    if force or analysis_was_missing:
        run_cmd(
            ["question2/run.py", "visualize"],
            title="Q2 FIGURES | 刷新 fig01–fig10 与最终 paper_results.md",
        )

    if not figures_only:
        run_stage(
            title="Q2 SUBMIT | 三种子验证选模与附件3最终提交",
            args=["question2/run.py", "submit"],
            outputs=[
                Q2_OUT / "analysis" / "attachment3_submission.csv",
                Q2_OUT / "analysis" / "selected_model.json",
            ],
            force=force,
        )


def q2_final_audit() -> None:
    """Fail loudly if the root pipeline omitted any required Q2 product."""
    training_jobs: list[tuple[str, int, str | None]] = []

    for model in ("mult", "concat_mlp", "gru", "mult_no_aug", "tfr_feature"):
        for seed in (2026, 2027, 2028):
            training_jobs.append((model, seed, None))

    for variant in ("full", "no_recon", "dual_recon"):
        for seed in (2026, 2027, 2028):
            training_jobs.append(("mrrf", seed, variant))

    for variant in (
        "no_reliability",
        "no_consistency",
        "plain_no_aug",
        "plain_fusion",
        "recon_only",
    ):
        training_jobs.append(("mrrf", 2026, variant))

    missing_training = []
    for model, seed, variant in training_jobs:
        for path in q2_training_outputs(model, seed, variant):
            if not path.exists():
                missing_training.append(path)

    required = [
        Q2_TUNING / "best_config.json",
        Q2_TUNING / "trials.json",
        Q2_TUNING / "study.db",
        q2_prediction_path("mult", 2026),
        q2_prediction_path("mrrf", 2026, "full"),
        q2_prediction_path("mrrf", 2026, "no_recon"),
        Q2_OUT / "analysis" / "attachment3_submission.csv",
        Q2_OUT / "analysis" / "selected_model.json",
        *q2_numbered_figure_outputs(),
    ]
    missing = missing_training + [path for path in required if not path.exists()]

    if len(training_jobs) != 29:
        raise AssertionError(f"Internal Q2 training-job count changed: {len(training_jobs)}")

    if len(q2_optuna_trial_metrics()) < 30:
        missing.append(Q2_OUT / "runs" / "mrrf" / "optuna_trial_<30 completed>/metrics.json")

    if missing:
        raise RuntimeError(
            "Q2 final audit failed; the full pipeline is not complete:\n"
            + "\n".join(f"  - {rel(path)}" for path in missing)
        )

    # Write a small root-runner completion record only after every check passed.
    import json
    from datetime import datetime

    pdfs = sorted(Q2_OUT.rglob("*.pdf"))
    record = {
        "status": "complete",
        "fixed_training_checkpoints": 29,
        "optuna_trial_metrics": len(q2_optuna_trial_metrics()),
        "numbered_figures_fig01_to_fig19": 19,
        "all_pdf_files_currently_under_outputs_q2": len(pdfs),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    (Q2_OUT / "full_pipeline_complete.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    banner("Q2 FINAL AUDIT | PASS")
    print("✓ 29 个正式训练 checkpoint 全部完整", flush=True)
    print(f"✓ Optuna 完成 trial 产物：{len(q2_optuna_trial_metrics())}", flush=True)
    print("✓ fig01–fig19 全部存在", flush=True)
    print("✓ 附件3最终提交与模型选择文件存在", flush=True)
    print(f"✓ outputs/q2 当前 PDF 总数：{len(pdfs)}", flush=True)


def run_q2(*, force: bool, force_figures: bool) -> None:
    if force_figures and not force:
        banner("Q2 | 仅强制重跑所有图表/图表依赖分析，不重新训练")
        # Tuning summary and MulT diagnostics are also paper figures.
        q2_postprocessing(force=True, figures_only=True)
        return

    # 0) Hyper-parameter tuning: guarantee >=30 completed trial artifacts.
    # --force intentionally does NOT repeat a complete Optuna study.
    ensure_q2_tuning()

    # 1) Five independent architectures × three seeds = 15 checkpoints.
    for model in ("mult", "concat_mlp", "gru", "mult_no_aug", "tfr_feature"):
        for seed in (2026, 2027, 2028):
            train_q2(
                model=model,
                seed=seed,
                epochs=30,
                patience=5,
                force=force,
            )

    # 2) Main MRRF variants × three seeds = 9 checkpoints.
    for variant in ("full", "no_recon", "dual_recon"):
        for seed in (2026, 2027, 2028):
            train_q2(
                model="mrrf",
                variant=variant,
                seed=seed,
                epochs=30,
                patience=5,
                force=force,
            )

    # 3) Five seed-2026 component/cumulative ablation checkpoints.
    for variant in (
        "no_reliability",
        "no_consistency",
        "plain_no_aug",
        "plain_fusion",
        "recon_only",
    ):
        train_q2(
            model="mrrf",
            variant=variant,
            seed=2026,
            epochs=30,
            patience=5,
            force=force,
        )

    # Exactly 29 fixed formal checkpoints have now been requested.

    # 4) Attachment-3 predictions explicitly required downstream.
    ensure_q2_prediction(model="mult", seed=2026, force=force)
    ensure_q2_prediction(model="mrrf", variant="full", seed=2026, force=force)
    ensure_q2_prediction(model="mrrf", variant="no_recon", seed=2026, force=force)

    # 5) Every Q2 experimental / analysis / visualization / submission stage.
    q2_postprocessing(force=force, figures_only=False)

    # 6) Hard audit: q2 cannot report DONE unless the complete workflow exists.
    q2_final_audit()


# ============================================================================
# Question 3
# ============================================================================

Q3_OUT = ROOT / "outputs" / "q3"


Q3_STAGES = [
    (
        "pipeline",
        "预测 + Integrated Gradients + 初始解释",
        [
            Q3_OUT / "config.json",
            Q3_OUT / "tables" / "model_performance.csv",
            Q3_OUT / "tables" / "validation_predictions_explained.csv",
            Q3_OUT / "tables" / "validation_evidence.csv",
            Q3_OUT / "tables" / "attachment4_predictions.csv",
            Q3_OUT / "tables" / "attachment4_evidence_raw.csv",
        ],
    ),
    (
        "ig-check",
        "IG 完备性与数值检查",
        [
            Q3_OUT / "tables" / "attachment4_ig_completeness.csv",
            Q3_OUT / "tables" / "validation_ig_completeness.csv",
        ],
    ),
    (
        "evidence",
        "证据映射到文本、语音时段和原视频",
        [
            Q3_OUT / "tables" / "attachment4_alignment_audit.csv",
            Q3_OUT / "tables" / "attachment4_word_map.csv",
            Q3_OUT / "tables" / "attachment4_token_map.csv",
            Q3_OUT / "tables" / "attachment4_evidence.csv",
        ],
    ),
    (
        "fidelity",
        "解释忠实度、删除实验与模型变体对照",
        [
            Q3_OUT / "tables" / "full_validation_predictions_explained.csv",
            Q3_OUT / "tables" / "full_validation_evidence.csv",
            Q3_OUT / "tables" / "full_validation_explanation_metrics.csv",
            Q3_OUT / "tables" / "full_validation_ig_completeness.csv",
            Q3_OUT / "tables" / "validation_fidelity_model_summary.csv",
            Q3_OUT / "full_validation_comparison.json",
        ],
    ),
    (
        "sensitivity",
        "冻结模型敏感性分析",
        [
            Q3_OUT / "tables" / "attachment4_model_sensitivity.csv",
            Q3_OUT / "model_sensitivity.json",
        ],
    ),
    (
        "alignment",
        "融合门控与删除响应排序一致性",
        [
            Q3_OUT / "tables" / "whole_modality_deletion.csv",
            Q3_OUT / "tables" / "gate_response_rank.csv",
            Q3_OUT / "tables" / "gate_response_rank_summary.csv",
        ],
    ),
    (
        "verify",
        "全量覆盖、结果一致性与复现性检查",
        [
            Q3_OUT / "verification.json",
        ],
    ),
]


def q3_figure_outputs() -> list[Path]:
    names = [
        "fig01_performance",
        "fig02_fidelity",
        "fig03_modality_contributions",
        "fig04_local_importance",
        "fig05_alignment_quality",
        "fig06_timed_evidence",
        "fig07_model_sensitivity",
        "fig08_validation_variant_fidelity",
        "fig09_gate_response_rank",
    ]
    return [Q3_OUT / "figures" / f"{name}.pdf" for name in names] + [
        Q3_OUT / "paper" / "q3_results.md",
    ]


def run_q3(*, force: bool, force_figures: bool) -> None:
    if force_figures and not force:
        run_stage(
            title="Q3 FIGURES | 强制重建全部论文图与解释卡",
            args=["question3/run.py", "figures"],
            outputs=q3_figure_outputs(),
            force=True,
        )
        return

    for stage, description, outputs in Q3_STAGES:
        run_stage(
            title=f"Q3 | {stage} | {description}",
            args=["question3/run.py", stage],
            outputs=outputs,
            force=force,
        )

    run_stage(
        title="Q3 | figures | 论文级图表与解释卡",
        args=["question3/run.py", "figures"],
        outputs=q3_figure_outputs(),
        force=force,
    )



# ============================================================================
# Smoke tests
# ============================================================================

SMOKE_ROOT = ROOT / "outputs" / "_smoke"


def q1_smoke_env() -> dict[str, str]:
    """Mirror question1/run.py's CUDA/cuDNN bootstrap for smoke subprocesses."""
    import os

    env = os.environ.copy()
    py_root = Path(sys.executable).resolve().parents[1]
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    nvidia = py_root / "lib" / pyver / "site-packages" / "nvidia"

    libs = [
        nvidia / "cudnn" / "lib",
        nvidia / "cublas" / "lib",
    ]
    existing = env.get("LD_LIBRARY_PATH", "")
    valid = [str(p) for p in libs if p.is_dir()]
    if existing:
        valid.append(existing)
    if valid:
        env["LD_LIBRARY_PATH"] = ":".join(valid)

    # Same bundled NLTK data used by question1/run.py.
    nltk = ROOT / "question1" / "assets" / "nltk_data"
    if not nltk.exists():
        nltk = ROOT / "assets" / "nltk_data"
    if nltk.exists():
        env["NLTK_DATA"] = str(nltk)

    env["Q1_CUDA_PATH_READY"] = "1"
    return env


def smoke_compile(package: str) -> None:
    """Compile every .py file in one question package."""
    run_cmd(
        ["-m", "compileall", "-q", package],
        title=f"TEST | compileall | {package}",
    )


def smoke_import(modules: list[str], title: str) -> None:
    """Import modules in a clean interpreter so import/path errors are exposed."""
    code = (
        "import importlib\n"
        f"mods={modules!r}\n"
        "for name in mods:\n"
        "    importlib.import_module(name)\n"
        "    print('[IMPORT OK]', name)\n"
    )
    run_cmd(["-c", code], title=title)


def write_smoke_marker(question: str, details: dict) -> None:
    import json
    from datetime import datetime

    out = SMOKE_ROOT / question
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "pass",
        "question": question,
        "python": sys.executable,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        **details,
    }
    (out / "smoke_test_pass.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def smoke_q1() -> None:
    """
    Minimal real Q1 execution:
      - compile all Q1 modules
      - import extraction/alignment/analysis/visualization modules
      - run ONE real attachment-1 sample through text/audio/vision extraction
        and temporal alignment
      - write everything into outputs/_smoke/q1, never outputs/q1
    """
    banner("TEST Q1 | 单样本真实特征提取 + 对齐 smoke test")
    smoke_compile("question1")
    q1_env = q1_smoke_env()

    q1_modules = [
        "question1.config",
        "question1.pipeline",
        "question1.extraction.core",
        "question1.alignment.temporal",
        "question1.alignment.crosslingual",
        "question1.analysis.quality",
        "question1.visualization.cases",
        "question1.visualization.crosslingual",
        "question1.visualization.journal_cases",
    ]
    import_code = (
        "import importlib\n"
        f"mods={q1_modules!r}\n"
        "for name in mods:\n"
        "    importlib.import_module(name)\n"
        "    print('[IMPORT OK]', name)\n"
    )
    run_cmd(
        ["-c", import_code],
        title="TEST Q1 | import 全部关键模块",
        env=q1_env,
    )

    # Fail early with a readable CUDA/cuDNN diagnostic instead of aborting
    # deep inside WhisperX.
    cuda_probe = r"""
import os, torch
print("[Q1 CUDA PROBE] torch=", torch.__version__)
print("[Q1 CUDA PROBE] cuda_available=", torch.cuda.is_available())
print("[Q1 CUDA PROBE] torch_cuda=", torch.version.cuda)
print("[Q1 CUDA PROBE] cudnn=", torch.backends.cudnn.version())
print("[Q1 CUDA PROBE] LD_LIBRARY_PATH=", os.environ.get("LD_LIBRARY_PATH", ""))
if not torch.cuda.is_available():
    raise RuntimeError("Q1 is configured for CUDA ASR but torch.cuda.is_available() is False")
x = torch.randn(8, 8, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("[Q1 CUDA PROBE PASS]", float(y.abs().mean().cpu()))
"""
    run_cmd(
        ["-c", cuda_probe],
        title="TEST Q1 | CUDA/cuDNN 预检查",
        env=q1_env,
    )

    out = SMOKE_ROOT / "q1"
    code = f"""
from pathlib import Path
import json
import numpy as np
from question1 import config
from question1.pipeline import manifest, run

out = Path({str(out)!r})
out.mkdir(parents=True, exist_ok=True)
config.OUT = out

table = manifest()
usable = table[table.video_exists.astype(bool)]
if usable.empty:
    raise RuntimeError("Q1 smoke: attachment-1 contains no readable video")
sid = str(usable.iloc[0].sample_id)
print("[Q1 TEST SAMPLE]", sid, flush=True)

result = run({{sid}}, resume=False)
if len(result) != 1:
    raise RuntimeError(f"Q1 smoke expected exactly one processed sample, got {{len(result)}}")

feature = out / "features" / f"{{sid}}.npz"
log = out / "logs" / "samples" / f"{{sid}}.json"
if not feature.is_file() or not log.is_file():
    raise RuntimeError("Q1 smoke did not create feature/log files")

with np.load(feature) as z:
    required = {{
        "word_text", "word_audio", "word_vision",
        "word_valid", "word_audio_mask", "word_vision_mask",
        "second_audio", "second_vision",
    }}
    missing = sorted(required - set(z.files))
    if missing:
        raise RuntimeError(f"Q1 smoke feature archive missing keys: {{missing}}")
    if z["word_text"].shape != (config.MAX_WORDS, 768):
        raise RuntimeError(f"unexpected word_text shape: {{z['word_text'].shape}}")
    if z["word_audio"].shape[0] != config.MAX_WORDS:
        raise RuntimeError(f"unexpected word_audio shape: {{z['word_audio'].shape}}")
    if z["word_vision"].shape[0] != config.MAX_WORDS:
        raise RuntimeError(f"unexpected word_vision shape: {{z['word_vision'].shape}}")

meta = json.loads(log.read_text(encoding="utf-8"))
print("[Q1 TEST PASS]", {{
    "sample_id": sid,
    "text_audio_status": meta.get("text_audio_status"),
    "audio_frames": meta.get("audio_frames"),
    "vision_frames": meta.get("vision_frames"),
}}, flush=True)
"""
    run_cmd(
        ["-c", code],
        title="TEST Q1 | 实际跑 1 条视频",
        env=q1_env,
    )
    write_smoke_marker(
        "q1",
        {
            "scope": "one real attachment-1 sample; extraction + alignment",
            "output": str(out),
        },
    )
    banner("TEST Q1 | PASS")


def smoke_q2() -> None:
    """
    Minimal real Q2 execution without formal training:
      - compile/import every Q2 stage
      - load the real aligned_50.pkl
      - exercise missingness
      - run one forward+backward step for every independent architecture
      - run one forward+backward step for all 8 MRRF variants
      - verify attachment-3 can be loaded/encoded
      - check formal full-flow entry modules are importable

    No formal checkpoint under outputs/q2 is modified.
    """
    banner("TEST Q2 | 全模型单 batch forward/backward + 数据链路 smoke test")
    smoke_compile("question2")
    smoke_import(
        [
            "question2.data.dataset",
            "question2.data.missingness",
            "question2.models.baselines",
            "question2.models.mult",
            "question2.models.mrrf",
            "question2.models.tfr",
            "question2.training.trainer",
            "question2.training.mult_trainer",
            "question2.experiments.baselines",
            "question2.experiments.ablation",
            "question2.experiments.robustness",
            "question2.experiments.missing_duration",
            "question2.experiments.repeats",
            "question2.experiments.tuning",
            "question2.experiments.dual_reconstruction",
            "question2.analysis.model_comparison",
            "question2.analysis.error_analysis",
            "question2.analysis.attachment3_comparison",
            "question2.analysis.tuning_summary",
            "question2.visualization.architecture",
            "question2.visualization.report",
            "question2.analysis.model_selection",
        ],
        "TEST Q2 | import 训练/实验/分析/绘图/提交模块",
    )

    out = SMOKE_ROOT / "q2"
    code = f"""
from pathlib import Path
import gc
import json
import numpy as np
import torch
import torch.nn.functional as F

from question2.config import DATA, ATTACHMENT3
from question2.data.dataset import FeatureStore, load_attachment3
from question2.data.missingness import hide_batch
from question2.evaluation.evaluator import to_device
from question2.experiments.baselines import construct
from question2.models.mrrf import reconstruction_loss
from question2.training.trainer import make_loader, make_model

out = Path({str(out)!r})
out.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Q2 TEST DEVICE]", device, flush=True)

store = FeatureStore(DATA)
loader = make_loader(store, "train", batch_size=2, device=device, seed=987654)
original = next(iter(loader))

# Exercise clean + single + double-modality missingness, positions and independent gaps.
rng = np.random.default_rng(987654)
for scenario, kwargs in [
    ("clean", {{}}),
    ("text", {{"rate": .2, "position": "beginning"}}),
    ("audio", {{"duration": .2, "position": "middle"}}),
    ("vision", {{"rate": .2, "position": "end"}}),
    ("audio+vision", {{"rate": .3, "gap_layout": "independent"}}),
]:
    hidden = hide_batch(original, rng, scenario=scenario, **kwargs)
    if hidden["text"].shape[1:] != (50, 768):
        raise RuntimeError("Q2 smoke: text shape changed unexpectedly")

masked = hide_batch(
    original,
    np.random.default_rng(123),
    scenario="audio+vision",
    rate=.3,
    gap_layout="independent",
)
batch = to_device(masked, device)
original_device = to_device(original, device)

# Four independent controls.
for name in ("concat_mlp", "gru", "mult_no_aug", "tfr_feature"):
    model = construct(name).to(device)
    model.train()
    if name == "tfr_feature":
        logits, strength, aux = model(batch, return_aux=True)
        rec = reconstruction_loss(aux["recovered"], original_device, batch)
    else:
        logits, strength = model(batch)
        rec = logits.new_zeros(())
    loss = (
        F.cross_entropy(logits, batch["classification"])
        + .5 * F.smooth_l1_loss(strength, batch["strength"])
        + .2 * rec
    )
    if not torch.isfinite(loss):
        raise RuntimeError(f"Q2 smoke: non-finite loss for {{name}}")
    loss.backward()
    print("[Q2 MODEL OK]", name, "loss=", float(loss.detach().cpu()), flush=True)
    del model, logits, strength, loss
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# All MRRF code paths/variants.
variants = (
    "full", "no_recon", "no_reliability", "no_consistency",
    "plain_fusion", "plain_no_aug", "recon_only", "dual_recon",
)
for variant in variants:
    model = make_model(
        variant,
        hidden_dim=24,
        mask_padding=True,
        mask_unavailable=True,
    ).to(device)
    model.train()
    logits, strength, aux = model(batch, return_aux=True)
    rec = reconstruction_loss(aux["recovered"], original_device, batch)
    loss = (
        F.cross_entropy(logits, batch["classification"])
        + .5 * F.smooth_l1_loss(strength, batch["strength"])
        + .2 * rec
    )
    if not torch.isfinite(loss):
        raise RuntimeError(f"Q2 smoke: non-finite loss for MRRF {{variant}}")
    loss.backward()
    print("[Q2 MRRF OK]", variant, "loss=", float(loss.detach().cpu()), flush=True)
    del model, logits, strength, aux, loss
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Attachment 3 path + BERT reconstruction is part of the real prediction pipeline.
a3 = load_attachment3(ATTACHMENT3, store.scaler, bert_batch_size=8)
if len(a3["id"]) == 0:
    raise RuntimeError("Q2 smoke: attachment 3 loaded zero samples")
if a3["text"].shape[1:] != (50, 768):
    raise RuntimeError(f"Q2 smoke: attachment3 text shape {{a3['text'].shape}}")

summary = {{
    "device": str(device),
    "train_samples": len(store.splits["train"]["id"]),
    "valid_samples": len(store.splits["valid"]["id"]),
    "test_samples": len(store.splits["test"]["id"]),
    "attachment3_samples": len(a3["id"]),
    "independent_models_checked": 4,
    "mrrf_variants_checked": len(variants),
}}
(out / "runtime_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("[Q2 TEST PASS]", summary, flush=True)
"""
    run_cmd(["-c", code], title="TEST Q2 | 单 batch 实际执行全部模型代码路径")
    write_smoke_marker(
        "q2",
        {
            "scope": (
                "real aligned data; missingness; 4 independent model families; "
                "8 MRRF variants; forward/backward; attachment3 load"
            ),
            "output": str(out),
        },
    )
    banner("TEST Q2 | PASS")


def smoke_q3() -> None:
    """
    Minimal real Q3 execution:
      - compile/import every Q3 stage
      - validate frozen Q2 checkpoint + attachment4 loading
      - run ONE attachment4 sample through prediction, 2-step IG and deletion-based
        local explanation
      - do not write/overwrite outputs/q3
    """
    banner("TEST Q3 | 单样本预测 + IG + deletion explanation smoke test")
    smoke_compile("question3")
    smoke_import(
        [
            "question3.pipeline",
            "question3.attribution.integrated_gradients",
            "question3.attribution.modality_importance",
            "question3.evaluation.ig_check",
            "question3.evaluation.fidelity",
            "question3.evaluation.sensitivity",
            "question3.evaluation.alignment_quality",
            "question3.evidence.aligner",
            "question3.evidence.mapper",
            "question3.evidence.temporal_evidence",
            "question3.analysis.verification",
            "question3.visualization.report",
        ],
        "TEST Q3 | import pipeline/IG/evidence/fidelity/sensitivity/alignment/figures",
    )

    out = SMOKE_ROOT / "q3"
    code = f"""
from pathlib import Path
import json
import numpy as np
import torch
from transformers import AutoTokenizer

from question3.pipeline import (
    A4, CKPT, DATA, TOK, MODS, FeatureStore,
    clean_batch, integrated_gradients, explain_one, load_a4,
)
from question2.training.trainer import load_trained

out = Path({str(out)!r})
out.mkdir(parents=True, exist_ok=True)

checkpoint = CKPT / "checkpoint.pt"
if not checkpoint.is_file():
    raise FileNotFoundError(
        f"Q3 smoke requires frozen Q2 checkpoint: {{checkpoint}}"
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, ckpt = load_trained(CKPT, device)
model.eval()

store = FeatureStore(DATA)
tokenizer = AutoTokenizer.from_pretrained(
    TOK, use_fast=True, local_files_only=True
)
a4, items = load_a4(store.scaler, tokenizer)
if len(a4["id"]) != 20:
    raise RuntimeError(f"Q3 smoke expected 20 attachment4 samples, got {{len(a4['id'])}}")

one = clean_batch(a4, [0], device)
with torch.no_grad():
    cls = model(one)[0].argmax(1)

# Very small quadrature only for smoke testing the complete attribution path.
ig = {{
    m: v[0]
    for m, v in integrated_gradients(
        model, one, cls, steps=2, baseline_factor=.1
    ).items()
}}

summary, evidence, positions = explain_one(
    model,
    one,
    ig,
    np.random.default_rng(20260924),
    rate=.2,
    random_reps=1,
    device=device,
)

if len(evidence) != 3:
    raise RuntimeError(f"Q3 smoke expected 3 modality evidence rows, got {{len(evidence)}}")
if summary["prediction_class"] not in (0, 1, 2):
    raise RuntimeError("Q3 smoke produced invalid prediction class")

payload = {{
    "device": str(device),
    "checkpoint_variant": ckpt.get("variant"),
    "checkpoint_seed": ckpt.get("seed"),
    "sample_id": str(a4["id"][0]),
    "prediction_class": int(summary["prediction_class"]),
    "dominant_modality": summary["dominant_modality"],
    "evidence_rows": len(evidence),
    "position_rows": len(positions),
}}
(out / "runtime_summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("[Q3 TEST PASS]", payload, flush=True)
"""
    run_cmd(["-c", code], title="TEST Q3 | 实际跑 1 条附件4样本的解释链")
    write_smoke_marker(
        "q3",
        {
            "scope": "one real attachment4 sample; frozen Q2 model; prediction + 2-step IG + deletion explanation",
            "output": str(out),
        },
    )
    banner("TEST Q3 | PASS")


def run_smoke(question: str) -> None:
    if question == "q1":
        smoke_q1()
    elif question == "q2":
        smoke_q2()
    elif question == "q3":
        smoke_q3()
    elif question == "test":
        smoke_q1()
        smoke_q2()
        smoke_q3()
        banner("ALL SMOKE TESTS | PASS")
    else:
        raise ValueError(question)


# ============================================================================
# Root CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MathModel_restructured unified artifact-aware runner"
    )
    parser.add_argument(
        "question",
        choices=("q1", "q2", "q3", "test"),
        help=(
            "完整运行 q1/q2/q3；或使用 test 顺序执行三个问题的最小烟雾测试"
        ),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "对所选 q1/q2/q3 执行最小化真实运行检查，不启动正式全量实验，"
            "测试产物写入 outputs/_smoke/"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "忽略已有普通实验产物并强制重跑。Q2 的 Optuna 调参除外："
            "已有 best_config.json 时不会自动追加新的 30 个 trial。"
        ),
    )
    parser.add_argument(
        "--force-figures",
        action="store_true",
        help=(
            "只强制重建该问题的报告/图表相关阶段，不重新训练已有模型。"
            "适合修改绘图代码或论文配图后使用。"
        ),
    )
    args = parser.parse_args()

    if args.force and args.force_figures:
        parser.error("--force 和 --force-figures 二选一即可")
    if args.test and (args.force or args.force_figures):
        parser.error("--test 不能与 --force / --force-figures 同时使用")
    if args.question == "test" and (args.force or args.force_figures):
        parser.error("run.py test 不能与 --force / --force-figures 同时使用")

    started = time.time()
    mode = "TEST" if (args.test or args.question == "test") else "RUN"
    banner(
        f"START | {args.question.upper()} | mode={mode} | "
        f"resume={'OFF' if args.force else 'ON'} | project={ROOT}"
    )

    try:
        if args.test or args.question == "test":
            run_smoke(args.question)
        elif args.question == "q1":
            run_q1(force=args.force, force_figures=args.force_figures)
        elif args.question == "q2":
            run_q2(force=args.force, force_figures=args.force_figures)
        else:
            run_q3(force=args.force, force_figures=args.force_figures)

    except subprocess.CalledProcessError as exc:
        banner(f"FAILED | {args.question.upper()}")
        print(
            f"子命令失败，返回码 {exc.returncode}。\n"
            "已经成功生成的 checkpoint / CSV / PDF 会保留。"
            "修复问题后重新运行同一条命令，默认会自动从未完成阶段继续。",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(exc.returncode)

    except Exception as exc:
        banner(f"FAILED | {args.question.upper()}")
        print(str(exc), file=sys.stderr, flush=True)
        print(
            "\n已完成产物不会删除；修复后重新运行即可自动续跑。",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)

    elapsed = time.time() - started
    banner(f"DONE | {args.question.upper()} | elapsed={elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
