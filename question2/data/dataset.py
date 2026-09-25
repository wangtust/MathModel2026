"""Read the supplied aligned MOSEI files without confusing padding with loss."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def content_positions(text_bert: np.ndarray) -> np.ndarray:
    """BERT attention covers CLS, words/subwords and SEP; keep content only."""
    bert = np.asarray(text_bert)

    if bert.ndim != 3 or bert.shape[1:] != (3, 50):
        raise ValueError(
            f"Expected (N,3,50) text_bert, got {bert.shape}"
        )

    attention = bert[:, 1, :].astype(np.int64)
    lengths = attention.sum(axis=1)

    positions = np.arange(50)[None, :]

    if (
        np.any(lengths < 3)
        or np.any(lengths > 50)
        or not np.all(attention == (positions < lengths[:, None]))
        or not np.all(bert[:, 0, 0] == 101)
        or not np.all(
            bert[np.arange(len(bert)), 0, lengths - 1] == 102
        )
    ):
        raise ValueError(
            "Invalid or non-contiguous BERT attention/CLS/SEP structure"
        )

    return (
        (positions >= 1)
        &
        (positions < lengths[:, None] - 1)
    )


def observed_positions(
        features: np.ndarray,
        content: np.ndarray
) -> np.ndarray:
    """
    True:
        有效内容位置
        且特征非NaN
        且不是全0

    用于区分:
        padding
        模态缺失
    """

    return (
        content
        &
        np.isfinite(features).all(axis=-1)
        &
        np.any(features != 0, axis=-1)
    )


def _split(raw: dict, name: str) -> dict:

    part = raw[name]

    text = np.asarray(
        part["text"],
        dtype=np.float32
    )

    audio = np.asarray(
        part["audio"],
        dtype=np.float32
    )

    vision = np.asarray(
        part["vision"],
        dtype=np.float32
    )

    content = content_positions(
        np.asarray(part["text_bert"])
    )

    n = len(text)

    if (
        text.shape != (n, 50, 768)
        or audio.shape != (n, 50, 74)
        or vision.shape != (n, 50, 35)
        or len(part["id"]) != n
    ):
        raise ValueError(
            f"{name}: unexpected feature shapes"
        )


    labels = np.asarray(
        part["classification_labels"],
        dtype=np.int64
    )

    strength = np.asarray(
        part["regression_labels"],
        dtype=np.float32
    )

    expected = np.where(
        strength < 0,
        0,
        np.where(
            strength > 0,
            2,
            1
        )
    )

    if not np.array_equal(labels, expected):
        raise ValueError(
            f"{name}: classification/regression labels disagree"
        )


    return {
        "id": list(part["id"]),

        "text": text,

        "audio": audio,

        "vision": vision,

        "content": content,

        "observed_audio":
            observed_positions(audio, content),

        "observed_vision":
            observed_positions(vision, content),

        "classification": labels,

        "strength": strength,
    }



class FeatureStore:

    def __init__(self, path: str | Path):

        with Path(path).open("rb") as handle:
            raw = pickle.load(handle)


        self.splits = {
            name: _split(raw, name)
            for name in ("train", "valid", "test")
        }

        del raw


        self.scaler = {}

        train = self.splits["train"]


        for modality in ("audio", "vision"):

            values = train[modality][
                train[f"observed_{modality}"]
            ]

            mean = values.mean(axis=0).astype(
                np.float32
            )

            std = values.std(axis=0).astype(
                np.float32
            )

            std = np.maximum(
                std,
                1e-4
            )

            self.scaler[modality] = (
                mean,
                std
            )


            for split in self.splits.values():

                x = split[modality]

                x = np.clip(
                    (x - mean) / std,
                    -5.0,
                    5.0
                )

                x = np.nan_to_num(
                    x,
                    copy=False,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )

                x *= split[
                    f"observed_{modality}"
                ][..., None]

                split[modality] = x.astype(
                    np.float32,
                    copy=False
                )


    def dataset(self, name: str):

        return FeatureDataset(
            self.splits[name]
        )


    def save_scaler(self, path):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.savez_compressed(
            path,
            audio_mean=self.scaler["audio"][0],
            audio_std=self.scaler["audio"][1],
            vision_mean=self.scaler["vision"][0],
            vision_std=self.scaler["vision"][1],
        )



class FeatureDataset(Dataset):

    def __init__(self, arrays: dict):

        self.arrays = arrays


    def __len__(self):

        return len(
            self.arrays["id"]
        )


    def __getitem__(self, index):

        a = self.arrays


        content = torch.from_numpy(
            a["content"][index]
        )


        available_audio = torch.from_numpy(
            a["observed_audio"][index]
        )


        available_vision = torch.from_numpy(
            a["observed_vision"][index]
        )


        return {

            "id":
                a["id"][index],


            "text":
                torch.from_numpy(
                    a["text"][index]
                ),


            "audio":
                torch.from_numpy(
                    a["audio"][index]
                ),


            "vision":
                torch.from_numpy(
                    a["vision"][index]
                ),


            "content":
                content,


            # ===== MRRF新增接口 =====

            "available_text":
                content,


            "available_audio":
                available_audio,


            "available_vision":
                available_vision,


            # 保留旧接口

            "observed_audio":
                available_audio,


            "observed_vision":
                available_vision,



            "classification":
                torch.tensor(
                    a["classification"][index],
                    dtype=torch.long
                ),


            "strength":
                torch.tensor(
                    a["strength"][index],
                    dtype=torch.float32
                ),
        }



def load_attachment3(
        directory: str | Path,
        scaler: dict,
        bert_batch_size: int = 8
):

    from transformers import AutoModel


    paths = sorted(
        Path(directory).glob("*.pkl")
    )


    if not paths:
        raise FileNotFoundError(
            f"No attachment-three files in {directory}"
        )


    token_arrays = []
    audios = []
    visions = []


    for path in paths:

        with path.open("rb") as handle:

            item = pickle.load(handle)["test"]


        bert = np.asarray(
            item["text_bert"]
        )

        audio = np.asarray(
            item["audio"],
            dtype=np.float32
        )

        vision = np.asarray(
            item["vision"],
            dtype=np.float32
        )


        if (
            bert.shape != (1,3,50)
            or audio.shape != (1,50,74)
            or vision.shape != (1,50,35)
        ):
            raise ValueError(
                f"Unexpected attachment-three shapes in {path}"
            )


        token_arrays.append(
            bert[0].astype(np.int64)
        )

        audios.append(
            audio[0]
        )

        visions.append(
            vision[0]
        )



    tokens = np.stack(token_arrays)

    content = content_positions(tokens)



    model = AutoModel.from_pretrained(
        "bert-base-uncased",
        local_files_only=True
    ).eval()



    texts = []


    with torch.inference_mode():

        for start in range(
            0,
            len(tokens),
            bert_batch_size
        ):

            b = torch.from_numpy(
                tokens[start:start+bert_batch_size]
            )


            hidden = model(
                input_ids=b[:,0],
                attention_mask=b[:,1],
                token_type_ids=b[:,2]
            ).last_hidden_state


            texts.append(
                hidden.cpu()
                .numpy()
                .astype(np.float32)
            )



    del model



    result = {

        "id":
            [p.stem for p in paths],


        "text":
            np.concatenate(texts),


        "content":
            content,


        "audio":
            np.stack(audios),


        "vision":
            np.stack(visions),


        # 新增
        "available_text":
            content,

    }



    for modality in (
        "audio",
        "vision"
    ):

        mask = observed_positions(
            result[modality],
            content
        )


        mean, std = scaler[modality]


        x = np.clip(
            (result[modality]-mean)/std,
            -5.0,
            5.0
        )


        x = np.nan_to_num(
            x,
            copy=False,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )


        result[modality] = (
            x * mask[..., None]
        ).astype(np.float32)


        result[f"observed_{modality}"] = mask



        # MRRF接口

        result[f"available_{modality}"] = mask



    return result