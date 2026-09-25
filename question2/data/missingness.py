"""Reproducible contiguous, within-content missingness for aligned sequences."""

from __future__ import annotations

import numpy as np
import torch


TRAIN_CHOICES = (
    "clean",
    "audio",
    "vision",
    "audio+vision",
    "text",
    "text+audio",
    "text+vision"
)

TRAIN_WEIGHTS = (
    0.15,
    0.20,
    0.20,
    0.20,
    0.10,
    0.075,
    0.075
)


def training_weights(
    clean_probability: float = TRAIN_WEIGHTS[0]
) -> tuple[float, ...]:

    """Change clean exposure while preserving the relative missing-case mix."""

    if not 0.0 <= clean_probability <= 1.0:
        raise ValueError(
            "clean_probability must be between 0 and 1"
        )

    if clean_probability == TRAIN_WEIGHTS[0]:
        return TRAIN_WEIGHTS

    scale = (
        1.0 - clean_probability
    ) / (
        1.0 - TRAIN_WEIGHTS[0]
    )

    weights = (
        clean_probability,
        *(weight * scale for weight in TRAIN_WEIGHTS[1:])
    )

    return tuple(
        float(value)
        for value in weights
    )


def hide_batch(
    batch: dict,
    rng: np.random.Generator,

    scenario: str = "train",

    rate: float | None = None,

    position: str = "random",

    train_weights: tuple[float, ...] | None = None,

    gap_layout: str = "shared",

    duration: float | None = None

) -> dict:


    """
    Mask a single continuous content interval per selected modality.

    Parameters
    ----------
    duration:
        Continuous missing duration ratio.
        Example:
            duration=0.2
            means 20% continuous sequence missing.
    """


    content = batch["content"]

    if content.ndim != 2 or content.shape[1] != 50:
        raise ValueError(
            "Expected (batch,50) content mask"
        )


    if gap_layout not in {
        "shared",
        "independent",
        "mixed"
    }:
        raise ValueError(
            f"Unknown gap layout: {gap_layout}"
        )


    if rate is not None and not 0.0 <= rate <= 1.0:
        raise ValueError(
            "missing rate must be between zero and one"
        )


    if duration is not None and not 0.0 <= duration <= 1.0:
        raise ValueError(
            "duration must be between zero and one"
        )


    hidden = {
        name: np.zeros(
            content.shape,
            dtype=bool
        )

        for name in (
            "text",
            "audio",
            "vision"
        )
    }


    weights = (
        TRAIN_WEIGHTS
        if train_weights is None
        else train_weights
    )


    if (
        scenario == "train"
        and
        (
            len(weights) != len(TRAIN_CHOICES)
            or not np.isfinite(weights).all()
            or any(weight < 0 for weight in weights)
            or not np.isclose(sum(weights),1.0)
        )
    ):
        raise ValueError(
            "train_weights must be nonnegative probabilities summing to one"
        )


    for row in range(content.shape[0]):


        chosen = (
            rng.choice(
                TRAIN_CHOICES,
                p=weights
            )

            if scenario == "train"

            else scenario
        )


        if chosen == "clean":
            continue


        if chosen not in TRAIN_CHOICES:
            raise ValueError(
                f"Unknown missing scenario: {chosen}"
            )


        valid = (
            torch.nonzero(
                content[row],
                as_tuple=False
            )
            .flatten()
            .cpu()
            .numpy()
        )


        if len(valid) == 0:
            continue


        modalities = chosen.split("+")


        layout = (
            "independent"
            if rng.random() < 0.5
            else "shared"
        ) if (
            gap_layout == "mixed"
            and len(modalities) > 1
        ) else gap_layout


        shared_interval = None


        for modality in modalities:


            if (
                layout == "shared"
                and shared_interval is not None
            ):

                start, width = shared_interval


            else:

                # ==========================
                # 核心修改位置
                # duration优先级最高
                # ==========================

                if duration is not None:

                    missing_rate = duration


                elif rate is not None:

                    missing_rate = rate


                else:

                    missing_rate = float(
                        rng.choice(
                            [
                                0.1,
                                0.2,
                                0.3,
                                0.4,
                                0.5
                            ]
                        )
                    )


                width = max(
                    1,
                    min(
                        len(valid),
                        int(
                            round(
                                missing_rate
                                *
                                len(valid)
                            )
                        )
                    )
                )


                if position == "beginning":

                    start = 0


                elif position == "middle":

                    start = (
                        len(valid)
                        -
                        width
                    ) // 2


                elif position == "end":

                    start = (
                        len(valid)
                        -
                        width
                    )


                elif position == "random":

                    start = int(
                        rng.integers(
                            0,
                            len(valid)-width+1
                        )
                    )


                else:

                    raise ValueError(
                        f"Unknown missing position: {position}"
                    )


                shared_interval = (
                    start,
                    width
                )


            hidden[modality][
                row,
                valid[start:start+width]
            ] = True



    result = dict(batch)


    for modality in (
        "text",
        "audio",
        "vision"
    ):

        hidden_mask = torch.from_numpy(
            hidden[modality]
        )


        base = (
            content
            if modality == "text"
            else batch[f"observed_{modality}"]
        )


        available = (
            base
            &
            ~hidden_mask
        )


        result[modality] = (
            batch[modality]
            *
            available.unsqueeze(-1)
        )


        result[
            f"available_{modality}"
        ] = available


        result[
            f"injected_{modality}"
        ] = (
            hidden_mask
            &
            base
        )


    return result