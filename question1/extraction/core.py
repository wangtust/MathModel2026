"""Adapters around the checked-out GitHub modules and official MediaPipe API."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import mediapipe as mp

from .. import config
from ..alignment.temporal import temporal_face_presence

sys.path.insert(0, str(config.PROJECT / 'vendor/MMSA-FET/src'))
sys.path.insert(0, str(config.PROJECT / 'vendor/whisperX'))
sys.path.insert(0, str(config.PROJECT / 'vendor/silero-vad/src'))
from MSA_FET import FeatureExtractionTool
import whisperx
from silero_vad import load_silero_vad, get_speech_timestamps


def video_metadata(path):
    proc = subprocess.run(['ffprobe', '-v', 'error', '-show_streams',
                           '-show_format', '-of', 'json', str(path)],
                          capture_output=True, text=True, check=True)
    meta = json.loads(proc.stdout)
    streams = meta.get('streams', [])
    return {'duration': float(meta['format'].get('duration', 0)),
            'has_audio_stream': any(s['codec_type'] == 'audio' for s in streams),
            'has_video_stream': any(s['codec_type'] == 'video' for s in streams),
            'video_codec': next((s.get('codec_name') for s in streams if s['codec_type']=='video'), None),
            'audio_codec': next((s.get('codec_name') for s in streams if s['codec_type']=='audio'), None)}


class Extractors:
    def __init__(self, output):
        self.output = Path(output)
        self.fet = FeatureExtractionTool({
            'audio': {'tool': 'opensmile', 'sample_rate': 16000,
                      'args': {'feature_set': 'eGeMAPSv02', 'feature_level': 'LowLevelDescriptors',
                               'start': None, 'end': None}},
            'text': {'model': 'bert', 'device': 'cpu', 'pretrained': 'bert-base-uncased', 'args': {}}
        }, tmp_dir=self.output/'tmp/fet', log_dir=self.output/'logs/fet', verbose=0)
        self.fet._init_extractors()
        self.vad = load_silero_vad()
        self.asr = whisperx.load_model(config.ASR_MODEL, config.ASR_DEVICE,
                                       compute_type=config.ASR_COMPUTE, language='en',
                                       vad_method='silero')
        self.align_model, self.align_meta = whisperx.load_align_model('en', config.ASR_DEVICE)
        detector_opt = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(config.PROJECT/'assets/blaze_face_full_range.tflite')),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_detection_confidence=.45)
        landmarker_opt = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(config.PROJECT/'assets/face_landmarker.task')),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=5, output_face_blendshapes=True,
            min_face_detection_confidence=.45,
            min_face_presence_confidence=.45)
        self.detector_opt = detector_opt
        self.landmarker_opt = landmarker_opt
        self.detector = None
        self.landmarker = None
        self.face_names = None

    def close(self):
        if self.detector is not None:
            self.detector.close()
        if self.landmarker is not None:
            self.landmarker.close()

    def text_audio(self, path, official):
        # FET native feature extraction. Official text is supplied explicitly.
        self.fet.tmp_dir.mkdir(parents=True, exist_ok=True)
        result = self.fet.run_single(path, text=official)
        text = np.asarray(result['text'], np.float32)
        audio = np.asarray(result['audio'], np.float32)
        audio_time = np.asarray(self.fet.audio_extractor.get_timestamps(), np.float32)
        if audio.shape[0] != len(audio_time) and audio.shape[1] == len(audio_time):
            audio = audio.T
        if audio.shape[0] != len(audio_time):
            raise ValueError('FET audio timestamp/feature mismatch')
        word_ids = self.fet.text_extractor.get_word_ids(official)
        official_words = official.split()
        text_word = np.zeros((len(official_words), text.shape[1]), np.float32)
        token_counts = np.zeros(len(official_words), np.int32)
        for j, wid in enumerate(word_ids):
            if wid is not None and wid < len(official_words) and j < len(text):
                text_word[wid] += text[j]
                token_counts[wid] += 1
        divisor = np.maximum(token_counts, 1)[:, None]
        text_word /= divisor
        return text, text_word, token_counts, audio, audio_time

    def voice_and_asr(self, path):
        wave = whisperx.load_audio(str(path)).astype(np.float32)
        vad_raw = get_speech_timestamps(torch.from_numpy(wave), self.vad,
                                        sampling_rate=16000, threshold=config.VAD_THRESHOLD)
        segments = [[float(s['start'])/16000, float(s['end'])/16000] for s in vad_raw]
        if not segments:
            return segments, [], '', 0.0
        transcribed = self.asr.transcribe(wave, batch_size=8)
        chunks = transcribed.get('segments', [])
        asr_text = ' '.join(s.get('text', '').strip() for s in chunks).strip()
        if not chunks:
            return segments, [], asr_text, 0.0
        aligned = whisperx.align(chunks, self.align_model, self.align_meta, wave,
                                 config.ASR_DEVICE, return_char_alignments=False)
        observed = aligned.get('word_segments', [])
        timed = sum('start' in w and 'end' in w for w in observed)
        return segments, observed, asr_text, timed/max(1, len(observed))

    def vision(self, path):
        # VIDEO mode keeps timestamp state; re-create for every source clip.
        self.close()
        self.detector = mp.tasks.vision.FaceDetector.create_from_options(self.detector_opt)
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(self.landmarker_opt)
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or fps <= 0:
            cap.release()
            return np.empty(0, np.float32), np.empty((0, 52), np.float32), \
                   np.empty(0, np.uint8), np.empty(0, np.uint8), np.empty(0, np.uint8), 0, 0
        stride = max(1, round(fps/config.FPS))
        times, features, face, valid, multi = [], [], [], [], []
        decoded, idx = 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded += 1
            if idx % stride == 0:
                h, w = frame.shape[:2]
                if w > 640:
                    frame = cv2.resize(frame, (640, round(640*h/w)))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ms = round(idx/fps*1000)
                d = self.detector.detect_for_video(image, ms)
                l = self.landmarker.detect_for_video(image, ms)
                n_detect, n_land = len(d.detections), len(l.face_landmarks)
                n = max(n_detect, n_land)
                has_face = n > 0
                unique = n == 1 and len(l.face_blendshapes) == 1
                vector = np.zeros(52, np.float32)
                if unique:
                    blend = l.face_blendshapes[0]
                    if self.face_names is None:
                        self.face_names = [x.category_name for x in blend]
                    vector[:min(52, len(blend))] = [x.score for x in blend[:52]]
                times.append(idx/fps)
                features.append(vector)
                face.append(has_face)
                valid.append(unique)
                multi.append(n > 1)
            idx += 1
        cap.release()
        # A detector-only hit in one isolated sample is usually a background
        # false positive; retain one-frame appearances confirmed by landmarks.
        face = temporal_face_presence(face, valid, multi)
        return (np.asarray(times, np.float32), np.asarray(features, np.float32),
                np.asarray(face, np.uint8), np.asarray(valid, np.uint8),
                np.asarray(multi, np.uint8), decoded, float(fps))
