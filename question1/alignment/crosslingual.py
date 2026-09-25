"""Conservative translated-transcript support on the original media clock.

An English translation has no source-language *word* timestamps. This module
therefore stores native-language ASR, English machine-translation hypotheses,
and clip/segment-level A/V support separately from the official English words.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from faster_whisper import WhisperModel
from nltk.stem import PorterStemmer

from .. import config
from .temporal import interval_pool


SCREEN_MODEL = 'base'  # multilingual, unlike the English-only base.en
TRANSCRIBE_MODEL = 'small'
LANGUAGE_THRESHOLD = .80
METHOD_VERSION = 1
STOPWORDS = set('a an and are as at be been but by for from had has have he her his i in into is it its '
                'of on or our she that the their them there these this those to was were which who will with you your'.split())
STEMMER = PorterStemmer()
SUMMARY_EXCLUDE = {'asr_words', 'official_words', 'speech_segments', 'match',
                   'face_blendshape_names', 'failures', 'source_asr_segments',
                   'machine_translation_segments', 'english_only_asr_words'}


def _terms(text: str) -> set[str]:
    return {STEMMER.stem(word) for word in re.findall(r'[a-z]+', text.lower())
            if len(word) > 2 and word not in STOPWORDS}


def _screen(table: pd.DataFrame, logs: Path, output: Path) -> pd.DataFrame:
    """Language ID for all speech clips; source hashes invalidate stale rows."""
    previous = pd.read_csv(output) if output.exists() else pd.DataFrame()
    old = previous.set_index('sample_id') if {'sample_id', 'source_sha256',
            'detected_language', 'language_probability'}.issubset(previous.columns) else None
    rows = []
    model = None
    try:
        for row in table.itertuples(index=False):
            meta = json.loads((logs/f'{row.sample_id}.json').read_text())
            if not meta['speech_present']:
                rows.append({'sample_id':row.sample_id,'source_sha256':meta['sha256_video'],
                             'detected_language':'','language_probability':np.nan,
                             'screen_model':SCREEN_MODEL})
                continue
            if old is not None and row.sample_id in old.index and \
                    old.loc[row.sample_id,'source_sha256'] == meta['sha256_video']:
                language = str(old.loc[row.sample_id,'detected_language'])
                confidence = float(old.loc[row.sample_id,'language_probability'])
            else:
                if model is None:
                    model = WhisperModel(SCREEN_MODEL,device=config.ASR_DEVICE,
                                         compute_type=config.ASR_COMPUTE)
                # Whisper's language decision precedes iterator decoding.
                _, info = model.transcribe(meta['video_path'],beam_size=1,
                                           vad_filter=True,task='transcribe')
                language, confidence = info.language, float(info.language_probability)
            rows.append({'sample_id':row.sample_id,'source_sha256':meta['sha256_video'],
                         'detected_language':language,'language_probability':confidence,
                         'screen_model':SCREEN_MODEL})
    finally:
        del model
    result = pd.DataFrame(rows)
    result.to_csv(output,index=False,float_format='%.6f')
    return result


def _segments(iterator, duration: float) -> list[dict]:
    result=[]
    for item in iterator:
        start,end=float(item.start),float(item.end)
        if not (np.isfinite(start) and np.isfinite(end) and 0 <= start < end):
            continue
        result.append({'start':start,'end':min(end,duration),
                       'text':item.text.strip(),'avg_logprob':float(item.avg_logprob),
                       'no_speech_prob':float(item.no_speech_prob)})
    return [row for row in result if row['end']>row['start']]


def _candidate_support(translation: list[dict], official: str) -> tuple[list[dict], str]:
    """Find lexical anchors in the *machine translation*, not bilingual words.

    Keep one leading segment when the first anchor starts after a brief lead-in.
    No anchor means whole-clip candidate support, explicitly marked uncertain.
    """
    target=_terms(official)
    for segment in translation:
        overlap=sorted(target & _terms(segment['text']))
        segment['shared_content_stems']=overlap
        segment['shared_content_count']=len(overlap)
        segment['candidate_support']=False
    anchors=[i for i,segment in enumerate(translation) if segment['shared_content_count']]
    if anchors:
        first=0 if anchors[0]==1 else anchors[0]
        for i in range(first,anchors[-1]+1):translation[i]['candidate_support']=True
        return [segment for segment in translation if segment['candidate_support']], 'lexical_anchor_candidate'
    for segment in translation:segment['candidate_support']=True
    return translation, 'no_lexical_anchor_review'


def _pool_segment_features(z: dict, intervals: list[dict], uncertain_face: bool) -> dict:
    start=np.array([s['start'] for s in intervals],np.float32)
    end=np.array([s['end'] for s in intervals],np.float32)
    audio,audio_mask,audio_coverage=interval_pool(
        z['audio_native'],z['audio_time'],z['audio_speech_mask'],start,end)
    half=.5/config.FPS
    vision,vision_mask,vision_coverage=interval_pool(
        z['vision_native'],z['vision_time'],z['vision_valid_mask'],start,end,support_half=half)
    multi=z['vision_multiface_mask'].astype(bool)
    vt=z['vision_time']
    for i,(a,b) in enumerate(zip(start,end)):
        if uncertain_face or np.any(multi & (vt+half>a) & (vt-half<b)):
            vision_mask[i]=0
            vision[i]=0
    if len(start):
        # The official text relates to the *selected spoken region as a whole*.
        # Aggregate only source-time frames belonging to selected intervals.
        at=z['audio_time']
        va=z['audio_speech_mask'].astype(bool) & np.any(
            (at[:,None]>=start[None,:]) & (at[:,None]<end[None,:]),axis=1)
        clip_audio=z['audio_native'][va].mean(axis=0) if va.any() else np.zeros(z['audio_native'].shape[1],np.float32)
        vv=z['vision_valid_mask'].astype(bool) & ~multi & np.any(
            (vt[:,None]+half>start[None,:]) & (vt[:,None]-half<end[None,:]),axis=1)
        clip_vision=z['vision_native'][vv].mean(axis=0) if vv.any() and not uncertain_face else np.zeros(z['vision_native'].shape[1],np.float32)
    else:
        clip_audio=np.zeros(z['audio_native'].shape[1],np.float32)
        clip_vision=np.zeros(z['vision_native'].shape[1],np.float32)
        va=np.empty(0,bool);vv=np.empty(0,bool)
    text=z['text_word_native']
    text_valid=z['text_token_count'].astype(bool)
    clip_text=text[text_valid].mean(axis=0) if text_valid.any() else np.zeros(text.shape[1],np.float32)
    return {'crosslingual_start':start,'crosslingual_end':end,
            'crosslingual_audio':audio,'crosslingual_audio_mask':audio_mask,
            'crosslingual_audio_coverage':audio_coverage,
            'crosslingual_vision':vision,'crosslingual_vision_mask':vision_mask,
            'crosslingual_vision_coverage':vision_coverage,
            'crosslingual_clip_start':np.float32(start.min() if len(start) else np.nan),
            'crosslingual_clip_end':np.float32(end.max() if len(end) else np.nan),
            'crosslingual_clip_text':clip_text.astype(np.float32),
            'crosslingual_clip_audio':clip_audio.astype(np.float32),
            'crosslingual_clip_vision':clip_vision.astype(np.float32),
            'crosslingual_clip_text_mask':np.uint8(text_valid.any()),
            'crosslingual_clip_audio_mask':np.uint8(bool(va.any())),
            'crosslingual_clip_vision_mask':np.uint8(bool(vv.any() and not uncertain_face))}


def _save_npz_atomic(path: Path, z: dict):
    temporary=path.with_suffix('.npz.tmp')
    with open(temporary,'wb') as stream:np.savez_compressed(stream,**z)
    os.replace(temporary,path)


def _enrich_foreign(meta: dict, path: Path, model: WhisperModel, language: str,
                    confidence: float) -> dict:
    if meta.get('crosslingual_method_version')==METHOD_VERSION and \
            meta.get('source_sha256_at_crosslingual_run')==meta['sha256_video']:
        return meta
    duration=float(meta['duration'])
    kwargs={'beam_size':5,'language':language,'vad_filter':True,
            'condition_on_previous_text':False}
    spoken,_=model.transcribe(meta['video_path'],task='transcribe',**kwargs)
    translated,_=model.transcribe(meta['video_path'],task='translate',**kwargs)
    source_segments=_segments(spoken,duration)
    translation_segments=_segments(translated,duration)
    selected,semantic_status=_candidate_support(translation_segments,meta['official_text'])
    if not source_segments or not selected:
        raise ValueError(f'No source-language/translation segments for {meta["sample_id"]}')
    with np.load(path) as archive:z={key:archive[key] for key in archive.files}
    if np.isfinite(z['word_start']).any() or z['word_audio_mask'].any() or z['word_vision_mask'].any():
        raise AssertionError('Cross-language official English words already have trusted source-speech times')
    z.update(_pool_segment_features(z,selected,bool(meta['target_face_uncertain'])))
    z['crosslingual_language']=np.array(language)
    z['crosslingual_selected_translation_indices']=np.array(
        [i for i,s in enumerate(translation_segments) if s['candidate_support']],np.int32)
    _save_npz_atomic(path,z)
    meta['english_only_asr_text']=meta['asr_text']
    meta['english_only_asr_words']=meta['asr_words']
    meta['english_only_asr_unreliable']=meta['asr_unreliable']
    meta['english_only_asr_repetition_ratio']=meta['asr_repetition_ratio']
    meta['english_only_asr_timing_coverage']=meta['asr_timing_coverage']
    meta['speech_language']=language
    meta['speech_language_confidence']=confidence
    meta['text_language']='en'
    meta['text_audio_status']='crosslingual_translation_candidate'
    meta['text_audio_match']=False
    meta['alignment_granularity']='clip_and_source_segments'
    meta['asr_unreliable']=False  # the old flag came from the English-only model
    meta['asr_repetition_ratio']=None  # source-language word alignment was not run
    meta['asr_timing_coverage']=None
    meta['asr_text']=' '.join(s['text'] for s in source_segments)
    meta['asr_words']=[]  # no foreign-word ↔ translated-English-word mapping
    meta['source_asr_segments']=source_segments
    meta['machine_translation_segments']=translation_segments
    meta['machine_translation_en']=' '.join(s['text'] for s in translation_segments)
    meta['crosslingual_semantic_status']=semantic_status
    meta['crosslingual_selected_segments']=len(selected)
    meta['crosslingual_support_start']=float(z['crosslingual_clip_start'])
    meta['crosslingual_support_end']=float(z['crosslingual_clip_end'])
    meta['crosslingual_audio_valid']=bool(z['crosslingual_clip_audio_mask'])
    meta['crosslingual_vision_valid']=bool(z['crosslingual_clip_vision_mask'])
    meta['crosslingual_model']=TRANSCRIBE_MODEL
    meta['crosslingual_method_version']=METHOD_VERSION
    meta['source_sha256_at_crosslingual_run']=meta['sha256_video']
    return meta


def upgrade_existing_outputs() -> pd.DataFrame:
    """Upgrade all 100 existing logs/NPZ in place without rerunning FET or vision."""
    output=config.OUT
    table=pd.read_csv(output/'tables/01_manifest.csv')
    logs=output/'logs/samples'
    language_table=_screen(table,logs,output/'tables/07_multilingual_language_screen.csv')
    by_id=language_table.set_index('sample_id')
    model=None
    summary=[]
    rows=[]
    try:
        for row in table.itertuples(index=False):
            path=logs/f'{row.sample_id}.json'
            meta=json.loads(path.read_text())
            language=by_id.loc[row.sample_id,'detected_language']
            confidence=by_id.loc[row.sample_id,'language_probability']
            confident=bool(pd.notna(confidence) and confidence>=LANGUAGE_THRESHOLD)
            if confident and language!='en':
                try:
                    current=(meta.get('crosslingual_method_version')==METHOD_VERSION and
                             meta.get('source_sha256_at_crosslingual_run')==meta['sha256_video'])
                    if current and 'english_only_asr_repetition_ratio' not in meta:
                        meta['english_only_asr_repetition_ratio']=meta['asr_repetition_ratio']
                        meta['english_only_asr_timing_coverage']=meta['asr_timing_coverage']
                        meta['asr_repetition_ratio']=None
                        meta['asr_timing_coverage']=None
                    if not current:
                        if model is None:
                            model=WhisperModel(TRANSCRIBE_MODEL,device=config.ASR_DEVICE,
                                               compute_type=config.ASR_COMPUTE)
                        meta=_enrich_foreign(meta,output/'features'/f'{row.sample_id}.npz',
                                             model,str(language),float(confidence))
                except Exception as error:
                    meta['crosslingual_failure']=f'{type(error).__name__}: {error}'
                    meta['alignment_granularity']='none'
                    meta['speech_language']=str(language)
                    meta['speech_language_confidence']=float(confidence)
                    meta['text_language']='en'
                    meta['text_audio_status']='crosslingual_processing_failed'
                    meta['text_audio_match']=False
            else:
                meta['speech_language']=str(language) if pd.notna(language) else ''
                meta['speech_language_confidence']=float(confidence) if pd.notna(confidence) else None
                meta['text_language']='en'
                meta['alignment_granularity']='word' if meta['text_audio_status']=='matched' else 'none'
                if meta['speech_present'] and not confident:
                    meta['language_review_required']=True
            path.write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=float))
            summary.append({key:value for key,value in meta.items() if key not in SUMMARY_EXCLUDE})
            rows.append({'sample_id':row.sample_id,'source_sha256':meta['sha256_video'],
                         'speech_language':meta['speech_language'],
                         'language_confidence':meta['speech_language_confidence'],
                         'official_text_language':'en',
                         'text_audio_status':meta['text_audio_status'],
                         'alignment_granularity':meta['alignment_granularity'],
                         'source_asr_text':meta.get('asr_text',''),
                         'machine_translation_en':meta.get('machine_translation_en',''),
                         'semantic_status':meta.get('crosslingual_semantic_status',''),
                         'selected_segments':meta.get('crosslingual_selected_segments',0),
                         'audio_valid':meta.get('crosslingual_audio_valid',False),
                         'vision_valid':meta.get('crosslingual_vision_valid',False),
                         'review_required':bool(meta.get('language_review_required') or
                                                meta['text_audio_status'].startswith('crosslingual'))})
    finally:
        del model
    result=pd.DataFrame(rows)
    result.to_csv(output/'tables/08_crosslingual_alignment.csv',index=False)
    pd.DataFrame(summary).to_csv(output/'tables/02_sample_summary.csv',index=False)
    return result
