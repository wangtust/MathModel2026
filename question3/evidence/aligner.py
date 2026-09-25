"""Align an English source transcript to its clip with the bundled WhisperX."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

from common.paths import PROJECT_ROOT, THIRD_PARTY_ROOT


class ForcedAligner:
    """Return only words with measured timestamps and alignment scores."""

    def __init__(self, device: str = "cuda"):
        self.device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
        vendor = THIRD_PARTY_ROOT / "whisperX"
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        import whisperx
        import nltk

        nltk_data = PROJECT_ROOT / "assets" / "nltk_data"
        if nltk_data.is_dir() and str(nltk_data) not in nltk.data.path:
            nltk.data.path.insert(0, str(nltk_data))
        self.whisperx = whisperx
        self.model, self.metadata = whisperx.load_align_model("en", self.device)

    def __call__(self, wav: str | Path, text: str) -> list[dict]:
        if not text.strip():
            return []
        audio = self.whisperx.load_audio(str(wav))
        duration = len(audio) / 16000
        if duration <= 0:
            return []
        result = self.whisperx.align(
            [{"start": 0.0, "end": duration, "text": text}],
            self.model, self.metadata, audio, self.device,
            interpolate_method="ignore", return_char_alignments=False,
        )
        words = []
        for item in result["word_segments"]:
            if not all(key in item for key in ("word", "start", "end", "score")):
                continue
            start, end, score = (float(item[key]) for key in ("start", "end", "score"))
            if not (math.isfinite(start) and math.isfinite(end) and math.isfinite(score)
                    and 0 <= start < end <= duration + 0.01):
                continue
            words.append({"word": str(item["word"]), "start": start,
                          "end": end, "score": score})
        return words
