from __future__ import annotations

import numpy as np
import torch

from question2.data.missingness import hide_batch
from .metrics import metrics, coherent_strength


def to_device(batch, device):
    return {
        k: (
            v.to(device, non_blocking=True)
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in batch.items()
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    scenario,
    seed=20260923,
    rate=0.3,
    position="random",
    gap_layout="shared",
    duration=None
):

    model.eval()

    rng = np.random.default_rng(seed)

    labels = []
    predicted = []

    strength = []
    estimated = []

    ids = []
    confidence = []


    for batch in loader:

        masked = to_device(
            hide_batch(
                batch,
                rng,
                scenario=scenario,
                rate=rate,
                position=position,
                gap_layout=gap_layout,
                duration=duration
            ),
            device
        )


        out = model(masked)

        logits, output = out[0], out[1]

        # Per-sample classification confidence used by error analysis.
        # This is the maximum softmax probability of the predicted class.
        confidence.extend(
            torch.softmax(logits, dim=-1)
            .max(dim=-1)
            .values
            .cpu()
            .numpy()
            .tolist()
        )


        labels.extend(
            batch["classification"]
            .numpy()
            .tolist()
        )

        predicted.extend(
            logits.argmax(-1)
            .cpu()
            .numpy()
            .tolist()
        )


        strength.extend(
            batch["strength"]
            .numpy()
            .tolist()
        )

        estimated.extend(
            output
            .cpu()
            .numpy()
            .tolist()
        )


        ids.extend(
            batch["id"]
        )


    adjusted = coherent_strength(
        predicted,
        estimated
    )


    result = metrics(
        labels,
        predicted,
        strength,
        adjusted
    )


    raw = metrics(
        labels,
        predicted,
        strength,
        estimated
    )


    result.update(
        {
            "raw_mae": raw["mae"],

            "raw_pearson": raw["pearson"],


            "adjusted_predictions":
                int(
                    np.sum(
                        np.abs(
                            adjusted
                            -
                            np.asarray(
                                estimated
                            )
                        )
                        > 1e-8
                    )
                ),


            "scenario":
                scenario,


            "missing_rate":
                rate
                if scenario != "clean"
                else 0.0,


            "missing_duration":
                duration
                if (
                    duration is not None
                    and scenario != "clean"
                )
                else 0.0,


            "position":
                position,


            "gap_layout":
                gap_layout,


            "ids":
                ids,


            "true_class":
                labels,


            "pred_class":
                predicted,


            "true_strength":
                strength,


            "pred_strength":
                adjusted.tolist(),


            "raw_pred_strength":
                estimated,


            "confidence":
                confidence
        }
    )


    return result



def public_metrics(result):

    hidden = {
        "ids",
        "true_class",
        "pred_class",
        "true_strength",
        "pred_strength",
        "raw_pred_strength",
        "confidence"
    }


    return {
        k: v
        for k, v in result.items()
        if k not in hidden
    }