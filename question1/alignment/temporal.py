"""Conservative official-text ↔ observed-speech matching and masked pooling."""
import re
from collections import Counter
from difflib import SequenceMatcher
import numpy as np


def words(text):
    return str(text).split()


def normalize(word):
    return re.sub(r"[^a-z0-9']", '', str(word).lower().replace('’', "'"))


def repetition_ratio(observed, n=4):
    tokens=[normalize(w.get('word','')) for w in observed]
    grams=[tuple(tokens[i:i+n]) for i in range(max(0,len(tokens)-n+1))]
    if not grams:return 0.
    counts=Counter(grams)
    return sum(max(0,v-1) for v in counts.values())/len(grams)


def match_words(official, observed, threshold=0.55):
    """Only identical lexical units receive observed times; never stretch text."""
    ref = [normalize(w) for w in official]
    hyp = [normalize(w.get('word', '')) for w in observed]
    blocks = SequenceMatcher(None, ref, hyp, autojunk=False).get_matching_blocks()
    mapping = {}
    for block in blocks:
        for offset in range(block.size):
            i, j = block.a + offset, block.b + offset
            if (ref[i] and 'start' in observed[j] and 'end' in observed[j]
                    and float(observed[j].get('score', 0)) >= .1):
                a, z = float(observed[j]['start']), float(observed[j]['end'])
                if np.isfinite(a) and np.isfinite(z) and 0 <= a < z:
                    mapping[i] = (j, a, z)
    lexical = sum(bool(x) for x in ref)
    coverage = len(mapping) / max(1, lexical)
    trusted = coverage >= threshold
    starts = np.full(len(ref), np.nan, dtype=np.float32)
    ends = np.full(len(ref), np.nan, dtype=np.float32)
    if trusted:
        for i, (_, a, z) in mapping.items():
            starts[i], ends[i] = a, z
    return starts, ends, {'coverage': coverage, 'trusted': trusted,
                          'matched_words': len(mapping), 'official_words': lexical,
                          'observed_words': len(hyp), 'matched_pairs':
                          [[i, mapping[i][0]] for i in sorted(mapping)]}


def interval_pool(values, times, valid, starts, ends, support_half=0.0):
    """Temporal pooling with a separate effective-data mask."""
    values = np.asarray(values, np.float32)
    times = np.asarray(times, np.float32)
    valid = np.asarray(valid, bool)
    if values.ndim != 2:
        raise ValueError('values must have shape [time, channels]')
    out = np.zeros((len(starts), values.shape[1]), np.float32)
    masks = np.zeros(len(starts), np.uint8)
    coverage = np.zeros(len(starts), np.float32)
    if len(times) == 0:
        return out, masks, coverage
    for i, (a, z) in enumerate(zip(starts, ends)):
        if not (np.isfinite(a) and np.isfinite(z) and z > a):
            continue
        if support_half:
            weight = np.maximum(0, np.minimum(times + support_half, z) -
                                np.maximum(times - support_half, a))
        else:
            weight = ((times >= a) & (times < z)).astype(np.float32)
        sel = (weight > 0) & valid
        if np.any(sel):
            out[i] = np.average(values[sel], axis=0, weights=weight[sel])
            masks[i] = 1
            coverage[i] = float(np.sum(weight[sel]) / max(1e-6, np.sum(weight)))
    return out, masks, coverage


def pad(arr, length, fill=0):
    arr = np.asarray(arr)
    result = np.full((length,) + arr.shape[1:], fill, dtype=arr.dtype)
    n = min(length, len(arr))
    result[:n] = arr[:n]
    return result


def temporal_face_presence(raw, features_valid, multiple):
    """Reject isolated detector-only hits, while retaining confirmed short faces."""
    raw=np.asarray(raw,dtype=bool)
    confirmed=np.asarray(features_valid,dtype=bool)|np.asarray(multiple,dtype=bool)
    neighbor=np.zeros(len(raw),dtype=bool)
    if len(raw)>1:
        neighbor[1:] |= raw[:-1]
        neighbor[:-1] |= raw[1:]
    return (raw & (confirmed|neighbor)).astype(np.uint8)
