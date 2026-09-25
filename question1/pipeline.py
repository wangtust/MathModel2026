"""Complete per-sample processing; each modality fails independently."""
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .alignment.temporal import interval_pool, match_words, pad, words, repetition_ratio
from .extraction import Extractors, video_metadata


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def manifest():
    table = pd.read_excel(config.DATA / 'label-100.xlsx')
    required = {'video_id', 'clip_id', 'text', 'label', 'annotation'}
    assert required.issubset(table.columns)
    assert len(table) == 100
    assert not table[['video_id', 'clip_id']].duplicated().any()
    records = []
    for row in table.itertuples(index=False):
        path = config.DATA / str(row.video_id) / f'{row.clip_id}.mp4'
        records.append({'sample_id': f'{row.video_id}_{row.clip_id}',
                        'video_id': str(row.video_id), 'clip_id': str(row.clip_id),
                        'video_path': str(path), 'video_exists': path.is_file(),
                        'official_text': str(row.text), 'label': float(row.label),
                        'annotation': str(row.annotation)})
    return pd.DataFrame(records)


def speech_on_times(times, segments):
    mask = np.zeros(len(times), np.uint8)
    for a, z in segments:
        mask |= ((times >= a) & (times < z)).astype(np.uint8)
    return mask


def speech_overlap(starts, ends, segments):
    result = np.zeros(len(starts), np.float32)
    for i, (a, z) in enumerate(zip(starts, ends)):
        if np.isfinite(a) and np.isfinite(z) and z > a:
            result[i] = min(1., sum(max(0., min(z, v)-max(a, u)) for u, v in segments)/(z-a))
    return result


def process_sample(row, extractors, output):
    sid, path = row.sample_id, Path(row.video_path)
    logs = Path(output)/'logs/samples'
    features_dir = Path(output)/'features'
    logs.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    failures = {}
    meta = row._asdict()
    official = str(row.official_text)
    official_words = words(official)
    count = len(official_words)
    info = video_metadata(path) if path.exists() else {'duration': 0, 'has_audio_stream': False, 'has_video_stream': False}
    duration = info['duration']
    text_token = np.empty((0, 768), np.float32)
    text_word = np.zeros((count, 768), np.float32)
    token_counts = np.zeros(count, np.int32)
    audio = np.empty((0, 25), np.float32)
    audio_time = np.empty(0, np.float32)
    try:
        text_token, text_word, token_counts, audio, audio_time = extractors.text_audio(path, official)
    except Exception:
        failures['fet_audio_text'] = traceback.format_exc()
        try:
            # Preserve the official-text feature even if audio extraction fails.
            feat = extractors.fet.text_extractor
            text_token = np.asarray(feat.extract(official), np.float32)
            for i, wid in enumerate(feat.get_word_ids(official)):
                if wid is not None and wid < count and i < len(text_token):
                    text_word[wid] += text_token[i]
                    token_counts[wid] += 1
            text_word /= np.maximum(token_counts, 1)[:, None]
        except Exception:
            failures['text'] = traceback.format_exc()
    segments, observed, asr_text, align_coverage = [], [], '', 0.0
    if info['has_audio_stream']:
        try:
            segments, observed, asr_text, align_coverage = extractors.voice_and_asr(path)
        except Exception:
            failures['asr_vad'] = traceback.format_exc()
    try:
        vision_time, vision, face, face_valid, multi, decoded, source_fps = extractors.vision(path)
    except Exception:
        failures['vision'] = traceback.format_exc()
        vision_time = np.empty(0, np.float32)
        vision = np.empty((0, 52), np.float32)
        face = face_valid = multi = np.empty(0, np.uint8)
        decoded, source_fps = 0, 0.0
    word_start, word_end, match = match_words(official_words, observed, config.MATCH_THRESHOLD)
    asr_repeat = repetition_ratio(observed)
    asr_unreliable = len(observed) >= 12 and asr_repeat >= .25
    if asr_unreliable:
        word_start[:] = np.nan
        word_end[:] = np.nan
        match['trusted'] = False
    if not segments:
        text_audio_status = 'no_speech'
    elif not observed:
        text_audio_status = 'asr_unavailable'
    elif asr_unreliable:
        text_audio_status = 'asr_unreliable'
    elif match['trusted']:
        text_audio_status = 'matched'
    else:
        text_audio_status = 'mismatch_candidate'
    speech_native = speech_on_times(audio_time, segments)
    word_speech_fraction = speech_overlap(word_start, word_end, segments)
    word_audio, word_audio_mask, audio_coverage = interval_pool(
        audio, audio_time, speech_native, word_start, word_end)
    word_audio_mask &= (word_speech_fraction >= .25).astype(np.uint8)
    word_audio[~word_audio_mask.astype(bool)] = 0
    vision_half = .5/config.FPS
    word_vision, word_vision_mask, vision_coverage = interval_pool(
        vision, vision_time, face_valid, word_start, word_end, support_half=vision_half)
    # A word containing ambiguous multi-face frames is not assigned arbitrarily.
    word_multi = np.array([bool(np.any((vision_time+vision_half > a) &
                                        (vision_time-vision_half < z) & multi.astype(bool)))
                           if np.isfinite(a) and np.isfinite(z) else False
                           for a, z in zip(word_start, word_end)], dtype=bool)
    word_vision_mask[word_multi] = 0
    word_vision[~word_vision_mask.astype(bool)] = 0
    sec_start = np.arange(0, max(1, np.ceil(duration)), config.SECOND_STEP, dtype=np.float32)
    sec_end = np.minimum(sec_start+config.SECOND_STEP, duration).astype(np.float32)
    sec_audio, sec_audio_mask, _ = interval_pool(audio, audio_time, speech_native, sec_start, sec_end)
    sec_vision, sec_vision_mask, _ = interval_pool(vision, vision_time, face_valid,
                                                   sec_start, sec_end, support_half=vision_half)
    sec_multi = np.array([np.any((vision_time+vision_half > a)&
                                  (vision_time-vision_half < z)&multi.astype(bool))
                          for a,z in zip(sec_start,sec_end)], dtype=bool)
    sec_vision_mask[sec_multi] = 0
    sec_vision[~sec_vision_mask.astype(bool)] = 0
    sec_speech_fraction = speech_overlap(sec_start, sec_end, segments)
    npz_path = features_dir/f'{sid}.npz'
    np.savez_compressed(npz_path,
        text_token=text_token, text_word_native=text_word,
        text_token_count=token_counts, audio_native=audio, audio_time=audio_time,
        audio_speech_mask=speech_native, vision_native=vision,
        vision_time=vision_time, vision_face_mask=face, vision_valid_mask=face_valid,
        vision_multiface_mask=multi, word_start=word_start, word_end=word_end,
        word_text=pad(text_word, config.MAX_WORDS),
        word_audio=pad(word_audio, config.MAX_WORDS),
        word_vision=pad(word_vision, config.MAX_WORDS),
        word_valid=pad(np.ones(count, np.uint8), config.MAX_WORDS),
        word_audio_mask=pad(word_audio_mask, config.MAX_WORDS),
        word_vision_mask=pad(word_vision_mask, config.MAX_WORDS),
        word_speech_fraction=pad(word_speech_fraction, config.MAX_WORDS),
        word_audio_coverage=pad(audio_coverage, config.MAX_WORDS),
        word_vision_coverage=pad(vision_coverage, config.MAX_WORDS),
        second_start=sec_start, second_end=sec_end,
        second_audio=sec_audio, second_audio_mask=sec_audio_mask,
        second_vision=sec_vision, second_vision_mask=sec_vision_mask,
        second_speech_fraction=sec_speech_fraction)
    flags = {'media_decoded': bool(decoded), 'audio_stream_present': info['has_audio_stream'],
             'speech_present': bool(segments), 'face_present': bool(np.any(face)),
             'face_feature_valid': bool(np.any(face_valid)),
             'target_face_uncertain': bool(np.any(multi)),
             'official_text_present': bool(official.strip()),
             'text_audio_match': text_audio_status == 'matched',
             'text_audio_status': text_audio_status,
             'asr_unreliable': asr_unreliable,
             'asr_repetition_ratio': asr_repeat,
             'asr_timing_coverage': align_coverage,
             'matched_word_fraction': match['coverage'],
             'word_audio_valid_fraction': float(np.mean(word_audio_mask)) if count else 0.,
             'word_vision_valid_fraction': float(np.mean(word_vision_mask)) if count else 0.}
    record = {**meta, **info, **flags,
              'sha256_video': sha256(path) if path.exists() else None,
              'word_count': count, 'text_token_count': int(len(text_token)),
              'audio_frames': int(len(audio)), 'vision_frames': int(len(vision)),
              'decoded_frames': decoded, 'source_fps': source_fps,
              'speech_segments': segments, 'asr_text': asr_text,
              'asr_words': observed, 'official_words': official_words,
              'match': match, 'face_blendshape_names': extractors.face_names,
              'feature_file': str(npz_path), 'failures': failures}
    with open(logs/f'{sid}.json', 'w') as f:
        json.dump(record, f, indent=2, ensure_ascii=False, default=float)
    return {key: value for key, value in record.items()
            if key not in ('asr_words', 'official_words', 'speech_segments', 'match',
                           'face_blendshape_names', 'failures')}


def run(ids=None, resume=True):
    config.OUT.mkdir(parents=True, exist_ok=True)
    table = manifest()
    (config.OUT/'tables').mkdir(exist_ok=True)
    table.to_csv(config.OUT/'tables/01_manifest.csv', index=False)
    if ids:
        table = table[table.sample_id.isin(ids)]
    engine = Extractors(config.OUT)
    rows = []
    try:
        for i, row in enumerate(table.itertuples(index=False), start=1):
            logfile = config.OUT/'logs/samples'/f'{row.sample_id}.json'
            if resume and logfile.exists():
                with open(logfile) as f:
                    r = json.load(f)
                rows.append({key: value for key, value in r.items()
                             if key not in ('asr_words','official_words','speech_segments',
                                            'match','face_blendshape_names','failures')})
            else:
                try:
                    rows.append(process_sample(row, engine, config.OUT))
                except Exception:
                    (config.OUT/'logs').mkdir(exist_ok=True)
                    with open(config.OUT/'logs'/f'FAILED_{row.sample_id}.txt','w') as f:
                        f.write(traceback.format_exc())
                    print(f'FAILED {row.sample_id}', flush=True)
            print(f'[{i}/{len(table)}] {row.sample_id}', flush=True)
            pd.DataFrame(rows).to_csv(config.OUT/'tables/02_sample_summary.csv', index=False)
    finally:
        engine.close()
    return pd.DataFrame(rows)
