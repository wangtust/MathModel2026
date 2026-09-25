"""
Missing duration robustness experiment.

Analyze:
    - missing duration
    - missing modality

Output:
    outputs/q2/experiments/missing_duration/
"""

from __future__ import annotations

import csv
from pathlib import Path
import argparse
import torch

from question2.config import DATA
from question2.data.dataset import FeatureStore
from question2.training.trainer import load_trained, make_loader
from question2.evaluation.evaluator import evaluate, public_metrics

from common.paths import Q2_OUTPUT_DIR


OUT = (
    Q2_OUTPUT_DIR
    / "experiments"
    / "missing_duration"
)


MODALITIES = (
    "text",
    "audio",
    "vision"
)


# 缺失持续比例
DURATIONS = (
    0.05,
    0.10,
    0.20,
    0.30,
    0.50
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=
        Q2_OUTPUT_DIR
        / "runs"
        / "mrrf"
        / "full"
        / "seed_2026"
    )

    args = parser.parse_args()


    OUT.mkdir(
        parents=True,
        exist_ok=True
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    model, checkpoint = load_trained(
        args.model_dir,
        device
    )


    loader = make_loader(
        FeatureStore(DATA),
        "valid",
        64,
        device
    )


    rows=[]


    for modality in MODALITIES:

        for duration in DURATIONS:


            result = evaluate(
                model,
                loader,
                device,
                scenario=modality,
                duration=duration,
                position="random"
            )


            rows.append(
                {
                    "modality": modality,
                    "duration": duration,
                    **public_metrics(result)
                }
            )


            print(
                f"{modality} duration={duration}"
            )


    output = OUT / "duration_results.csv"


    with output.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)


    print(
        "Saved:",
        output
    )


if __name__=="__main__":
    main()