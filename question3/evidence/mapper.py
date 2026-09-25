"""Auditably map attachment-4 BERT positions to source words and clip seconds."""
from __future__ import annotations

import difflib
import json
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from common.paths import Q3_OUTPUT_DIR, Q2_OUTPUT_DIR

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from question3.pipeline import A4, TOK, write_csv
from question3.evidence.aligner import ForcedAligner

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*(?:[.,!?;:])?")


def norm(s):
    return re.sub(r"[^a-z0-9']", "", str(s).lower())


def word_spans(raw):
    return [{"word_index": i, "word": m.group(), "char_start": m.start(),
             "char_end": m.end()} for i, m in enumerate(WORD_RE.finditer(raw))]


def match_words(source, aligned):
    """Use Q1's exact matching followed by conservative within-block matching."""
    out = [{**w, "start_sec": None, "end_sec": None, "score": None,
            "whisperx_word": ""} for w in source]
    a, b = [norm(w["word"]) for w in source], [norm(w["word"]) for w in aligned]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    used = set()
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            i, j = block.a + k, block.b + k
            if i < len(out) and j < len(aligned):
                out[i].update(start_sec=float(aligned[j]["start"]),
                              end_sec=float(aligned[j]["end"]),
                              score=float(aligned[j]["score"]),
                              whisperx_word=aligned[j]["word"])
                used.add(j)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        for i in range(i1, i2):
            if out[i]["start_sec"] is not None:
                continue
            pairs = [(difflib.SequenceMatcher(a=a[i], b=b[j], autojunk=False).ratio(), j)
                     for j in range(j1, j2) if j not in used]
            if pairs and max(pairs)[0] >= .8:
                _, j = max(pairs)
                out[i].update(start_sec=float(aligned[j]["start"]),
                              end_sec=float(aligned[j]["end"]),
                              score=float(aligned[j]["score"]),
                              whisperx_word=aligned[j]["word"])
                used.add(j)
    return out


def token_words(offsets, words):
    mapping = {}
    for i, (s, e) in enumerate(offsets):
        if e <= s:
            continue
        overlaps = [w["word_index"] for w in words
                    if w["char_start"] < e and w["char_end"] > s]
        mapping[i] = overlaps
    return mapping


@lru_cache(maxsize=1)
def local_asr():
    from faster_whisper import WhisperModel
    return WhisperModel("tiny.en", device="cpu", compute_type="int8",
                        local_files_only=True)


def asr_similarity(wav, raw):
    """Local tiny.en ASR is an independent mismatch screen, not ground truth."""
    try:
        model = local_asr()
        segments, _ = model.transcribe(str(wav), beam_size=3, vad_filter=True)
        heard = " ".join(s.text for s in segments).strip()
        a = [norm(w) for w in WORD_RE.findall(raw)]
        b = [norm(w) for w in WORD_RE.findall(heard)]
        sim = difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio() if b else 0.0
        return heard, sim, "ok"
    except Exception as exc:
        return "", None, f"asr_error:{type(exc).__name__}:{str(exc)[:150]}"


def duration(video):
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
                       text=True, capture_output=True, check=True)
    return float(p.stdout.strip())


def save_frame(video, sec, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{sec:.3f}",
                    "-i", str(video), "-frames:v", "1", "-q:v", "3", str(target)],
                   check=True)


def run(out: Path):
    metadata = json.loads((out / "attachment4_metadata.json").read_text())
    ev = pd.read_csv(out / "tables/attachment4_evidence_raw.csv", keep_default_na=False)
    pos = pd.read_csv(out / "tables/attachment4_position_ig.csv", keep_default_na=False)
    tokenizer = AutoTokenizer.from_pretrained(TOK, use_fast=True, local_files_only=True)
    aligner = ForcedAligner(device="cuda")
    audit, word_rows, token_rows, evidence_rows = [], [], [], []
    for sid in sorted(metadata):
        info = metadata[sid]
        raw = info["raw_text"]
        video = A4 / "videos" / f"{sid}.mp4"
        seconds = duration(video)
        words = word_spans(raw)
        enc = tokenizer(raw, max_length=50, truncation=True, padding="max_length",
                        return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        if [list(o) for o in offsets] != info["offsets"]:
            raise ValueError(f"Saved tokenizer offsets mismatch: {sid}")
        with tempfile.TemporaryDirectory(prefix="q3_audio_") as tmp:
            wav = Path(tmp) / f"{sid}.wav"
            audio_ok = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video),
                                       "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                                      capture_output=True).returncode == 0
            error, aligned, heard, sim, asr_state = "", [], "", None, "no_audio"
            if audio_ok and wav.exists() and wav.stat().st_size > 1000:
                heard, sim, asr_state = asr_similarity(wav, raw)
                try:
                    aligned = aligner(wav, raw)
                except Exception as exc:
                    error = f"{type(exc).__name__}:{str(exc)[:200]}"
            else:
                error = "audio_extract_failed"
        matched = match_words(words, aligned)
        finite = [w for w in matched if w["start_sec"] is not None]
        coverage = len(finite) / len(words) if words else 0.0
        scores = [w["score"] for w in finite if np.isfinite(w["score"])]
        mean_score = float(np.mean(scores)) if scores else None
        # Tiny ASR is a screening aid: a low match forbids a confident clock.
        mismatch = sim is not None and sim < .35
        trusted = bool(coverage >= .85 and mean_score is not None and
                       mean_score >= .30 and sim is not None and not mismatch and not error)
        status = ("trusted" if trusted else "asr_unavailable" if sim is None else
                  "asr_mismatch" if mismatch else
                  "alignment_failed" if error else "low_coverage_or_score")
        row = {"sample_id": sid, "video_path": str(video), "duration_sec": seconds,
               "source_word_count": len(words), "aligned_word_count": len(finite),
               "word_coverage": coverage, "mean_alignment_score": mean_score,
               "asr_similarity": sim, "asr_transcript": heard,
               "asr_status": asr_state, "alignment_status": status,
               "alignment_error": error, "text_truncated": info["text_truncated"],
               "retained_char_end": info["retained_end"]}
        audit.append(row)
        for w in matched:
            word_rows.append({"sample_id": sid, **w,
                              "timing_trusted": bool(trusted and w["score"] is not None and w["score"] >= .30)})
        token_map = token_words(offsets, matched)
        for idx, ids in token_map.items():
            span = offsets[idx]
            token_rows.append({"sample_id": sid, "token_index": idx,
                               "char_start": span[0], "char_end": span[1],
                               "token_text": raw[span[0]:span[1]],
                               "word_indices": ",".join(map(str, ids)),
                               "alignment_status": status})
        for _, e in ev[ev.sample_id.astype(str).str.zfill(2) == sid].iterrows():
            rec = e.to_dict()
            rec["sample_id"] = sid
            rec.update({"alignment_status": status, "video_path": str(video),
                        "source_text_truncated": info["text_truncated"],
                        "start_sec": None, "end_sec": None, "keyframe_sec": None,
                        "keyframe_path": "", "source_excerpt": "",
                        "word_indices": "", "timed_word_fraction": 0.0,
                        "localization_status": "native_unavailable" if not bool(rec["available"]) else "untrusted_alignment"})
            rec["reason"] = "native_features_unavailable" if not bool(rec["available"]) else ""
            effect = rec.get("top_delta_margin")
            rec["response_direction"] = ("unavailable" if not bool(rec["available"]) else
                                         "supportive" if float(effect) > .02 else
                                         "counterevidence" if float(effect) < -.02 else "weak")
            if not bool(rec["available"]):
                evidence_rows.append(rec)
                continue
            start, end = int(rec["token_start"]), int(rec["token_end"])
            ids = sorted(set(i for token in range(start, end + 1) for i in token_map.get(token, [])))
            rec["word_indices"] = ",".join(map(str, ids))
            if ids:
                char_start = min(matched[i]["char_start"] for i in ids)
                partial_boundary = any(matched[i]["char_end"] > info["retained_end"] for i in ids)
                char_end = min(max(matched[i]["char_end"] for i in ids), info["retained_end"])
                rec["source_excerpt"] = raw[char_start:char_end]
                rec["source_char_start"] = char_start
                rec["source_char_end"] = char_end
                good = [matched[i] for i in ids if matched[i]["start_sec"] is not None and
                        matched[i]["score"] is not None and matched[i]["score"] >= .30]
                rec["timed_word_fraction"] = len(good) / len(ids)
                if trusted and len(good) == len(ids) and not partial_boundary:
                    rec["start_sec"] = min(w["start_sec"] for w in good)
                    rec["end_sec"] = max(w["end_sec"] for w in good)
                    rec["localization_status"] = "trusted_word_alignment"
                    if rec["modality"] == "vision":
                        center = (rec["start_sec"] + rec["end_sec"]) / 2
                        rec["keyframe_sec"] = center
                        frame = out / "frames" / f"{sid}_vision_top.jpg"
                        save_frame(video, center, frame)
                        rec["keyframe_path"] = str(frame)
                elif partial_boundary:
                    rec["localization_status"] = "truncated_word_boundary"
            if not ids:
                rec["localization_status"] = "no_word_overlap"
            evidence_rows.append(rec)
        print(f"{sid}: {status} coverage={coverage:.2f} score={mean_score} ASR={sim}", flush=True)
    write_csv(out / "tables/attachment4_alignment_audit.csv", audit)
    write_csv(out / "tables/attachment4_word_map.csv", word_rows)
    write_csv(out / "tables/attachment4_token_map.csv", token_rows)
    write_csv(out / "tables/attachment4_evidence.csv", evidence_rows)
    predictions = pd.read_csv(out / "tables/attachment4_predictions.csv", dtype={"sample_id": str})
    predictions["sample_id"] = predictions.sample_id.str.zfill(2)
    by_id = {r["sample_id"]: r for r in audit}
    by_evidence = {(str(r["sample_id"]), r["modality"]): r for r in evidence_rows}
    enriched = []
    for rec in predictions.to_dict("records"):
        sid = rec["sample_id"]
        ar = by_id[sid]
        flags = []
        if ar["alignment_status"] != "trusted":
            flags.append("untrusted_transcript_clock")
        if ar["text_truncated"]:
            flags.append("bert_text_truncated")
        if rec["availability_vision"] == 0:
            flags.append("native_vision_absent")
        if rec["dominant_modality"] == "unclear":
            flags.append("dominant_unclear")
        if rec["predicted_probability"] < .5:
            flags.append("low_class_probability")
        if (rec["prediction_class"] == 2 and rec["raw_strength"] <= 0) or (
                rec["prediction_class"] == 0 and rec["raw_strength"] >= 0):
            flags.append("classification_regression_sign_conflict")
        dominant = by_evidence.get((sid, rec["dominant_modality"]))
        if dominant and dominant["localization_status"] != "trusted_word_alignment":
            flags.append("dominant_evidence_no_trusted_seconds")
        rec.update(alignment_status=ar["alignment_status"],
                   text_truncated=ar["text_truncated"],
                   trusted_evidence_count=sum(by_evidence[(sid, m)]["localization_status"] ==
                                              "trusted_word_alignment" for m in ("text", "audio", "vision")),
                   explanation_quality_flags=";".join(flags) if flags else "ok")
        enriched.append(rec)
    write_csv(out / "tables/attachment4_predictions.csv", enriched)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Q3_OUTPUT_DIR)
    run(parser.parse_args().output)
