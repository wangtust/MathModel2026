"""
Validation error analysis for MRRF.

Generate:

1. error cases
2. missing pattern statistics
3. confidence analysis

Outputs:

error_cases.csv
error_summary.json
missing_pattern.json
confidence_analysis.csv

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import pandas as pd

from question2.data.dataset import FeatureStore
from question2.training.trainer import load_trained, make_loader
from question2.config import DATA
from question2.config import run_dir
from question2.evaluation.evaluator import evaluate
from common.paths import Q2_OUTPUT_DIR



def analyze_errors(
        result,
        scenario: str
):

    labels = result["true_class"]

    preds = result["pred_class"]

    strengths = result["true_strength"]

    pred_strength = result["pred_strength"]


    missing_rate = result.get(
        "missing_rate",
        0.0
    )

    position = result.get(
        "position",
        "none"
    )

    gap_layout = result.get(
        "gap_layout",
        "none"
    )


    confidence = result.get(
        "confidence",
        None
    )


    rows = []


    for i, (y, p) in enumerate(
        zip(labels, preds)
    ):

        if y != p:

            row = {

                "index":
                    i,


                "scenario":
                    scenario,


                "missing_rate":
                    missing_rate,


                "missing_position":
                    position,


                "gap_layout":
                    gap_layout,


                "true_label":
                    int(y),


                "pred_label":
                    int(p),


                "true_strength":
                    float(
                        strengths[i]
                    ),


                "pred_strength":
                    float(
                        pred_strength[i]
                    ),


                "strength_error":
                    float(
                        abs(
                            strengths[i]
                            -
                            pred_strength[i]
                        )
                    )
            }


            if confidence is not None:
                row["confidence"] = float(
                    confidence[i]
                )


            rows.append(row)


    return rows



def analyze_confidence(
        result,
        scenario
):

    confidence = result.get(
        "confidence",
        None
    )

    if confidence is None:
        return []


    labels = result["true_class"]

    preds = result["pred_class"]


    rows=[]


    for i in range(len(labels)):

        rows.append(
            {
                "index":
                    i,

                "scenario":
                    scenario,

                "true_label":
                    int(labels[i]),

                "pred_label":
                    int(preds[i]),

                "correct":
                    int(
                        labels[i]==preds[i]
                    ),

                "confidence":
                    float(
                        confidence[i]
                    )
            }
        )


    return rows



def build_pattern_statistics(
        summary
):

    rows=[]

    for name, item in summary.items():

        total = item["num_samples"]

        errors = item["num_errors"]


        rows.append(
            {
                "scenario":
                    name,

                "samples":
                    total,

                "errors":
                    errors,

                "error_rate":
                    errors / max(total,1),

                "macro_f1":
                    item["macro_f1"],

                "mae":
                    item["mae"],

                "missing_rate":
                    item["missing_rate"],

                "position":
                    item["position"],

                "gap_layout":
                    item["gap_layout"]
            }
        )


    return rows



def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--model-dir",
        type=Path,
        default=run_dir("mrrf", 2026, "full"),
        help="Trained MRRF checkpoint directory"
    )


    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Q2_OUTPUT_DIR / "analysis" / "error_analysis",
        help="Directory for error-analysis tables"
    )


    args = parser.parse_args()


    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    device=torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    model, checkpoint = load_trained(
        args.model_dir,
        device
    )


    store = FeatureStore(DATA)


    loader = make_loader(
        store,
        "valid",
        64,
        device
    )



    scenarios = [

        (
            "clean",
            {
                "scenario":
                    "clean"
            }
        ),


        (
            "audio_missing",
            {
                "scenario":
                    "audio",

                "rate":
                    0.3
            }
        ),


        (
            "vision_missing",
            {
                "scenario":
                    "vision",

                "rate":
                    0.3
            }
        ),


        (
            "text_missing",
            {
                "scenario":
                    "text",

                "rate":
                    0.3
            }
        ),


        (
            "audio_vision_missing",
            {
                "scenario":
                    "audio+vision",

                "rate":
                    0.3
            }
        )

    ]



    all_errors=[]

    all_confidence=[]

    summary={}



    for name, kwargs in scenarios:


        result = evaluate(
            model,
            loader,
            device,
            **kwargs
        )


        errors = analyze_errors(
            result,
            name
        )


        confidence_rows = analyze_confidence(
            result,
            name
        )


        all_errors.extend(
            errors
        )


        all_confidence.extend(
            confidence_rows
        )


        total_samples = len(
            result["true_class"]
        )


        summary[name] = {

            "num_samples":
                total_samples,


            "num_errors":
                len(errors),


            "macro_f1":
                result.get(
                    "macro_f1",
                    None
                ),


            "mae":
                result.get(
                    "mae",
                    None
                ),


            "missing_rate":
                result.get(
                    "missing_rate",
                    0.0
                ),


            "position":
                result.get(
                    "position",
                    "none"
                ),


            "gap_layout":
                result.get(
                    "gap_layout",
                    "none"
                )
        }



    # =====================
    # error cases
    # =====================

    pd.DataFrame(
        all_errors
    ).to_csv(
        args.output_dir /
        "error_cases.csv",
        index=False,
        encoding="utf-8-sig"
    )



    # =====================
    # confidence
    # =====================

    if len(all_confidence)>0:

        pd.DataFrame(
            all_confidence
        ).to_csv(
            args.output_dir /
            "confidence_analysis.csv",
            index=False,
            encoding="utf-8-sig"
        )



    # =====================
    # summary
    # =====================

    with open(
        args.output_dir /
        "error_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )



    # =====================
    # missing pattern
    # =====================

    pattern = build_pattern_statistics(
        summary
    )


    with open(
        args.output_dir /
        "missing_pattern.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            pattern,
            f,
            indent=2,
            ensure_ascii=False
        )



    print(
        "Saved error analysis to",
        args.output_dir
    )



if __name__ == "__main__":

    main()
