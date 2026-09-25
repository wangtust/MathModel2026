"""Three auditable, word-synchronous journal figures for question 1."""
from __future__ import annotations

import html
import json
import subprocess
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.io import wavfile

from .. import config


# Curated after checking all word, speech, and single-face masks. Each has a
# different official sample-level polarity; these are not word-level labels.
CASES = (
    ('a', '-UacrmKiTn4_4', 'Negative', '负向'),
    ('b', '-s9qJ7ATP7w_6', 'Neutral', '中性'),
    ('c', '-THoVjtIkeU_2', 'Positive', '正向'),
)
FIGURES = config.OUT / 'figures' / 'alignment'
ASSETS = config.OUT / 'case_assets'
TABLES = config.OUT / 'tables'


def _read_frame_pts(video: str) -> np.ndarray:
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
         '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', video],
        capture_output=True, text=True, check=True)
    frames = json.loads(result.stdout)['frames']
    pts = np.asarray([float(f['best_effort_timestamp_time']) for f in frames], dtype=np.float64)
    if len(pts) == 0 or np.any(np.diff(pts) < 0):
        raise ValueError(f'Missing or nonmonotone video PTS: {video}')
    return pts


def _decode_word_audio(video: str, output: Path) -> tuple[int, np.ndarray]:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i', video,
         '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(output)],
        check=True, capture_output=True)
    rate, signal = wavfile.read(output)
    if rate != 16000 or signal.ndim != 1:
        raise AssertionError('Expected mono 16 kHz PCM')
    return rate, signal.astype(np.float32) / 32768.0


def _representative_frames(video: str, pts: np.ndarray, window: tuple[float,float], number: int = 6):
    start, end = window
    targets = start + (np.arange(number) + .5) * (end-start) / number
    indices = np.asarray([int(np.argmin(np.abs(pts - t))) for t in targets], dtype=int)
    cap = cv2.VideoCapture(video)
    wanted = set(indices.tolist())
    frames = {}
    index = 0
    while wanted:
        ok, image = cap.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            wanted.remove(index)
        index += 1
    cap.release()
    if wanted:
        raise AssertionError(f'Could not decode source video frames: {sorted(wanted)}')
    return [(int(index), float(pts[index]), frames[int(index)]) for index in indices]


def _letterbox_frame(rgb: np.ndarray, target_ratio: float = 1.42) -> np.ndarray:
    height, width = rgb.shape[:2]
    if width/height > target_ratio:
        target_height = int(np.ceil(width/target_ratio))
        result = np.full((target_height,width,3), 248, dtype=np.uint8)
        top=(target_height-height)//2
        result[top:top+height]=rgb
    else:
        target_width = int(np.ceil(height*target_ratio))
        result = np.full((height,target_width,3), 248, dtype=np.uint8)
        left=(target_width-width)//2
        result[:,left:left+width]=rgb
    return result


def _check_case(meta: dict, z: dict, polarity: str) -> None:
    count = int(meta['word_count'])
    if meta['annotation'] != polarity or meta['text_audio_status'] != 'matched':
        raise AssertionError(f'Polarity or transcript pairing changed: {meta["sample_id"]}')
    if not (meta['speech_present'] and meta['face_feature_valid']) or meta['target_face_uncertain']:
        raise AssertionError(f'This source is not a clean three-modality case: {meta["sample_id"]}')
    if not (np.isfinite(z['word_start']).all() and np.isfinite(z['word_end']).all()):
        raise AssertionError(f'Untimed official word in selected case: {meta["sample_id"]}')
    if not (np.all(z['word_audio_mask'][:count]) and np.all(z['word_vision_mask'][:count])):
        raise AssertionError(f'Invalid speech or face feature in case: {meta["sample_id"]}')
    if not (np.all(z['text_token_count'] > 0) and len(z['text_word_native']) == count):
        raise AssertionError(f'Official text feature mismatch: {meta["sample_id"]}')
    if not np.allclose(z['text_word_native'], z['word_text'][:count], rtol=0, atol=1e-6):
        raise AssertionError('Text word vectors differ from exported sequence')


def _word_mapping(meta: dict, z: dict, pts: np.ndarray, rate: int,
                  waveform: np.ndarray) -> pd.DataFrame:
    asr_pairs = {int(i): int(j) for i, j in meta['match']['matched_pairs']}
    rows = []
    audio_times = z['audio_time'].astype(float)
    vision_times = z['vision_time'].astype(float)
    support_half = .5 / config.FPS
    for i, (word, start, end) in enumerate(zip(meta['official_words'], z['word_start'], z['word_end'])):
        start, end = float(start), float(end)
        observed = meta['asr_words'][asr_pairs[i]]
        score = float(observed['score'])
        sample_start, sample_end = int(np.floor(start * rate)), int(np.ceil(end * rate))
        audio_index = np.flatnonzero((audio_times >= start) & (audio_times < end) &
                                     z['audio_speech_mask'].astype(bool))
        vision_index = np.flatnonzero((vision_times + support_half > start) &
                                      (vision_times - support_half < end) &
                                      z['vision_valid_mask'].astype(bool) &
                                      ~z['vision_multiface_mask'].astype(bool))
        video_index = np.flatnonzero((pts >= start) & (pts < end))
        if (not 0 <= sample_start < sample_end <= len(waveform) or
                len(audio_index) == 0 or len(vision_index) == 0 or len(video_index) == 0 or
                score < .1 or end > float(meta['duration']) + .05):
            raise AssertionError(f'Invalid original-media mapping: {meta["sample_id"]} word {i+1} {word}')
        # This holds because the plot uses the stored word vectors; it detects
        # accidental display of an unrelated array after data regeneration.
        if not np.allclose(z['audio_native'][audio_index].mean(axis=0),
                           z['word_audio'][i], rtol=2e-4, atol=2e-4):
            raise AssertionError(f'Audio feature does not correspond to word {i+1}')
        overlap = np.maximum(0, np.minimum(vision_times[vision_index] + support_half, end) -
                             np.maximum(vision_times[vision_index] - support_half, start))
        if not np.allclose(np.average(z['vision_native'][vision_index], axis=0, weights=overlap),
                           z['word_vision'][i], rtol=2e-4, atol=2e-4):
            raise AssertionError(f'Face feature does not correspond to word {i+1}')
        rows.append({
            'sample_id': meta['sample_id'], 'word_index_1based': i+1,
            'official_word': word, 'asr_word': observed['word'],
            'asr_alignment_score': score, 'start_s': start, 'end_s': end,
            'audio_start_sample_0based': sample_start,
            'audio_end_sample_exclusive': sample_end,
            'original_video_first_frame_0based': int(video_index[0]),
            'original_video_last_frame_0based': int(video_index[-1]),
            'original_video_first_pts_s': float(pts[video_index[0]]),
            'original_video_last_pts_s': float(pts[video_index[-1]]),
            'audio_feature_first_row_0based': int(audio_index[0]),
            'audio_feature_last_row_0based': int(audio_index[-1]),
            'visual_feature_first_row_0based': int(vision_index[0]),
            'visual_feature_last_row_0based': int(vision_index[-1]),
            'text_feature_row_0based': i,
            'audio_mask': int(z['word_audio_mask'][i]),
            'face_mask': int(z['word_vision_mask'][i]),
        })
    mapping = pd.DataFrame(rows)
    if (not mapping['start_s'].is_monotonic_increasing or
            not mapping['word_index_1based'].tolist() == list(range(1, len(rows)+1))):
        raise AssertionError('Word order/timing is not monotone')
    return mapping


def _standardize_channels(features: np.ndarray) -> np.ndarray:
    """Display only: per-channel z-score across official words in this clip."""
    matrix = np.asarray(features, np.float32)
    mean = matrix.mean(axis=0, keepdims=True)
    sd = matrix.std(axis=0, keepdims=True)
    return np.clip((matrix - mean) / np.where(sd > 1e-7, sd, 1), -2, 2).T


def _plot_case(letter: str, meta: dict, z: dict, mapping: pd.DataFrame,
               waveform: np.ndarray, frames: list, chinese_polarity: str):
    duration = float(meta['duration'])
    intervals = mapping[['start_s', 'end_s']].to_numpy(dtype=float)
    palette = {'Negative': '#bd5952', 'Neutral': '#67717b', 'Positive': '#356f9d'}
    polarity_color = palette[meta['annotation']]
    fig = plt.figure(figsize=(7.5, 7.55), facecolor='white')
    grid = fig.add_gridspec(7, 1, left=.105, right=.9, top=.91, bottom=.17,
                            hspace=.43, height_ratios=[1.37, 1.0, 1.13, .62, .62, .62, .5])
    axes = [fig.add_subplot(grid[i]) for i in range(7)]
    video_ax, wave_ax, word_ax, text_ax, audio_ax, face_ax, label_ax = axes
    fig.suptitle(f'图1{letter}  多模态视频片段的逐词时间对齐：{chinese_polarity}样本',
                 fontsize=11, fontweight='semibold', x=.105, ha='left', y=.983)
    fig.text(.105,.933,
             f'{meta["sample_id"]}   ·   原视频 {duration:.2f} s   ·   官方文本 {len(mapping)} 词   ·   '
             f'有效对齐 {len(mapping)}/{len(mapping)} 词',
             fontsize=7.8, color='#44505b')

    # Video thumbnails are extracted from decoded original frames. Their
    # labels use ffprobe presentation times, which share the audio origin.
    video_ax.set_ylim(0, 1)
    video_ax.set_title('A  原视频帧  |  t 为原视频显示时间；表情特征采样率 2 fps',
                       loc='left', fontsize=8, pad=2)
    card_width = duration / 6 * .9
    for _, pts, rgb in frames:
        left, right = max(0, pts-card_width/2), min(duration, pts+card_width/2)
        video_ax.imshow(_letterbox_frame(rgb), extent=(left,right,.24,.91),
                        aspect='auto', interpolation='bilinear')
        video_ax.text(pts,.10,f'{pts:.2f} s',ha='center',va='center',fontsize=6.1,color='#36454d')
        video_ax.plot([pts,pts],[.02,.08],color='#2e786d',lw=1.5)

    # Dense waveform is reduced by min/max bins, preserving transients.
    wave_ax.set_title('B  原声道波形  |  单声道 16 kHz；橙色为语音检测，灰色为词外音频',
                      loc='left', fontsize=8, pad=2)
    step = max(1, int(np.ceil(len(waveform)/2200)))
    count = len(waveform)//step
    binned = waveform[:count*step].reshape(count,step)
    times = (np.arange(count)+.5)*step/16000
    for start,end in meta['speech_segments']:
        wave_ax.axvspan(start,end,color='#e9c0a7',alpha=.28,zorder=0)
    for start,end in ((0,intervals[0,0]),(intervals[-1,1],duration)):
        if end-start>.12:
            wave_ax.axvspan(start,end,color='#aeb6bd',alpha=.22,zorder=.2)
    wave_ax.fill_between(times,binned.min(axis=1),binned.max(axis=1),
                         color='#cb7250',linewidth=0,alpha=.88)
    peak = max(.05,float(np.max(np.abs(waveform))))
    wave_ax.set_ylim(-peak*1.18,peak*1.18)
    if duration-intervals[-1,1]>.3:
        wave_ax.text((intervals[-1,1]+duration)/2,peak*.85,'词外音频',
                     ha='center',va='top',fontsize=5.5,color='#647077')
    wave_ax.set_ylabel('幅值',fontsize=7)
    wave_ax.tick_params(axis='y',labelsize=6)

    word_ax.set_ylim(0,1)
    word_ax.set_title('C  官方单词  |  每个色块严格对应上方语音的 [起点, 终点) 区间',
                      loc='left', fontsize=8, pad=2)
    lanes = (.43,.62,.81)
    for i,(row,(start,end)) in enumerate(zip(mapping.itertuples(index=False), intervals)):
        center=(start+end)/2
        word_ax.barh(.13,end-start,left=start,height=.18,
                     color='#74b9a5' if i%2==0 else '#a8d8c7',edgecolor='#387b6e',lw=.5)
        lane = lanes[i%3]
        word_ax.plot([center,center],[.23,lane-.07],color='#8ca99e',lw=.5)
        word_ax.text(center,lane,f'{i+1:02d} {row.official_word}',ha='center',va='center',
                     fontsize=6.3, color='#174c44',
                     bbox={'facecolor':'white','edgecolor':'none','pad':.25,'alpha':.77})
    word_ax.text(duration,.02,'编号按官方词顺序',ha='right',va='bottom',fontsize=6,color='#58646c')

    feature_cmap = plt.get_cmap('PRGn')
    for ax,title,key,dims in [
        (text_ax,'D  文本词特征  |  BERT WordPiece 均值','text_word_native',768),
        (audio_ax,'E  语音词特征  |  eGeMAPS LLD 窗口均值','word_audio',25),
        (face_ax,'F  人脸词特征  |  MediaPipe blendshape 加权均值','word_vision',52),
    ]:
        matrix=z[key] if key=='text_word_native' else z[key][:len(mapping)]
        if matrix.shape!=(len(mapping),dims):
            raise AssertionError(f'Unexpected feature shape {key}: {matrix.shape}')
        display=_standardize_channels(matrix)
        ax.set_ylim(0,dims)
        ax.set_facecolor('#f5f5f5')
        for i,(start,end) in enumerate(intervals):
            ax.imshow(display[:,i:i+1],extent=(start,end,0,dims),aspect='auto',
                      origin='lower',cmap=feature_cmap,vmin=-2,vmax=2,
                      interpolation='nearest',rasterized=True)
        ax.set_title(title,loc='left',fontsize=8,pad=2)
        ax.set_ylabel(f'{dims}维',fontsize=6.5)
        ax.set_yticks([])

    # The continuous label belongs to the complete clip and carries no word
    # timestamps. Render it on a separate intensity scale.
    label_ax.set_xlim(-3,3)
    label_ax.set_ylim(0,1)
    sentiment_cmap=LinearSegmentedColormap.from_list('label',
        ['#b85151','#e8b1ae','#f2f3f4','#a6c7e6','#427bb1'])
    label_ax.imshow(np.linspace(0,1,600)[None,:],extent=(-3,3,.18,.48),
                    aspect='auto',cmap=sentiment_cmap,vmin=0,vmax=1)
    label=float(meta['label'])
    label_ax.plot(label,.6,marker='v',color=polarity_color,markersize=7,clip_on=False)
    label_ax.text(label,.68,f'{label:+.2f}  {meta["annotation"]}',
                  ha='center',va='bottom',fontsize=8,fontweight='bold',color=polarity_color)
    label_ax.set_title('G  整段样本情感标签  |  原始 label-100.xlsx 标注，非逐词情感值',
                       loc='left',fontsize=8,pad=2)
    label_ax.set_xticks(np.arange(-3,4,1))
    label_ax.tick_params(axis='x',labelsize=6,length=2)
    label_ax.set_yticks([])
    for spine in ('left','right','top'):label_ax.spines[spine].set_visible(False)

    for ax in axes[:6]:
        ax.set_xlim(0,duration)
        ax.spines[['left','right','top']].set_visible(False)
        ax.tick_params(axis='x',labelsize=6,length=2)
        if ax not in (word_ax,face_ax):ax.tick_params(axis='x',labelbottom=False)
        if ax not in (wave_ax,):ax.set_yticks([])
    for ax in (wave_ax,text_ax,audio_ax,face_ax):
        for start,end in intervals:
            ax.axvline(start,color='#7b8d8c',lw=.32,alpha=.38)
        ax.grid(axis='x',color='#a6b5b2',lw=.4,alpha=.25)

    colorbar_ax=fig.add_axes([.923,.288,.011,.202])
    plt.colorbar(ScalarMappable(norm=Normalize(-2,2),cmap=feature_cmap),
                 cax=colorbar_ax,ticks=[-2,0,2])
    colorbar_ax.tick_params(labelsize=6,length=2)
    fig.text(.94,.498,'z',ha='center',fontsize=7)
    fig.text(.105,.087,
             '词边界来自 WhisperX 的实际语音对齐；文本保持官方原词。三行特征共用词时间，灰白间隙表示原素材中的词间空白。',
             fontsize=6.45,color='#4d5960')
    fig.text(.105,.065,
             '特征热图按本片段每个通道在词间标准化，仅用于显示；原始 768/25/52 维数值保存在 NPZ。视频帧为原视频抽样。',
             fontsize=6.45,color='#4d5960')
    fig.text(.105,.043,'A–F 共用原音视频时间轴（秒）；时间戳为自动估计，逐词原始帧和特征行号可在配套 CSV 与试听页面核对。',
             fontsize=6.45,color='#4d5960')
    return fig


def _review_page(case_data: list[dict]) -> str:
    cards=[]
    for case in case_data:
        sid=case['sample_id']
        buttons=''.join(
            f'<button class="word" data-start="{r.start_s:.3f}" data-end="{r.end_s:.3f}" '
            f'title="{r.start_s:.3f}–{r.end_s:.3f} s · ASR score {r.asr_alignment_score:.2f}">'
            f'<small>{r.word_index_1based:02d}</small> {html.escape(r.official_word)}</button>'
            for r in case['mapping'].itertuples(index=False))
        cards.append(f'''<section class="case" id="case-{html.escape(sid)}">
<h2>{case['chinese_polarity']} · {html.escape(sid)} · {case['label']:+.2f}</h2>
<p>官方文本：{html.escape(case['official_text'])}</p>
<audio controls preload="metadata" src="../case_assets/{html.escape(sid)}/audio_16k.wav"></audio>
<p class="hint">点击单词只播放该词对应的原声音频区间；悬停可看起止时间与自动对齐分数。</p>
<div class="words">{buttons}</div>
<figure><img src="../figures/alignment/fig01{case['letter']}_{sid}.png" alt="{case['chinese_polarity']}样本逐词时序对齐图"><figcaption>图1{case['letter']}：原始视频帧、语音波形、官方词区间、文本/语音/视觉特征及整段标签。</figcaption></figure>
<p><a href="../case_assets/{html.escape(sid)}/word_mapping.csv">下载逐词源素材与特征行号映射 CSV</a></p>
</section>''')
    return '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>问题一：三个样本的逐词时序对齐核验</title><style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:auto;max-width:1050px;padding:24px;color:#263640;background:#f6f8f7;line-height:1.55}
h1{font-size:1.5rem}h2{font-size:1.15rem;margin-bottom:0}a{color:#17695d}.case{background:white;border:1px solid #dce5e1;border-radius:8px;padding:18px;margin:22px 0}
audio{width:100%}.words{display:flex;flex-wrap:wrap;gap:7px}.word{background:#e4f2ec;border:1px solid #8ab7a4;border-radius:14px;padding:6px 10px;cursor:pointer;color:#124f43;font-size:1rem}
.word:hover,.word.active{background:#186b5d;color:white}.word small{opacity:.65}.hint{font-size:.85rem;color:#647077}
figure{margin:15px 0}figure img{width:100%;height:auto;border:1px solid #d4dcd8}figcaption{font-size:.8rem;color:#586770}
</style></head><body><h1>三个典型样本：逐词时间对齐核验</h1>
<p>负向、中性、正向各选一条。按钮控制下方原声音频；所有词区间均取自动转写与官方文本核对后的真实发音时间。情感标签是整段标注。</p>'''+''.join(cards)+'''<script>
document.querySelectorAll('.case').forEach(section=>{
const player=section.querySelector('audio');let stopAt=Infinity;let active=null;let frame=0;
function finish(){player.pause();stopAt=Infinity;cancelAnimationFrame(frame);if(active)active.classList.remove('active')}
function monitor(){if(player.currentTime>=stopAt-.008){finish();return}frame=requestAnimationFrame(monitor)}
section.querySelectorAll('button.word').forEach(button=>button.addEventListener('click',async()=>{
cancelAnimationFrame(frame);
if(active)active.classList.remove('active');active=button;button.classList.add('active');
stopAt=Number(button.dataset.end);player.currentTime=Number(button.dataset.start);
try{await player.play();frame=requestAnimationFrame(monitor)}catch(error){button.classList.remove('active')}
}));
player.addEventListener('timeupdate',()=>{if(player.currentTime>=stopAt)finish()});
player.addEventListener('ended',finish);
});
</script></body></html>'''


def create_alignment_cases() -> dict:
    """Build three figures, a multi-page PDF, CSV evidence, and listening page."""
    matplotlib.rcParams.update({
        'font.family': ['Microsoft YaHei','DejaVu Sans'], 'font.size': 8,
        'pdf.fonttype':42, 'ps.fonttype':42,
        'axes.edgecolor':'#66757b', 'axes.linewidth':.6,
        'savefig.dpi':300, 'figure.dpi':140,
    })
    FIGURES.mkdir(parents=True,exist_ok=True)
    TABLES.mkdir(parents=True,exist_ok=True)
    case_data=[]
    summary=[]
    multipage=FIGURES/'fig01_three_typical_cases.pdf'
    with PdfPages(multipage) as pdf:
        for letter,sid,polarity,chinese in CASES:
            log_path=config.OUT/'logs/samples'/f'{sid}.json'
            feature_path=config.OUT/'features'/f'{sid}.npz'
            if not log_path.exists() or not feature_path.exists():
                continue
            meta=json.loads(log_path.read_text())
            with np.load(feature_path) as archive:
                z={key:archive[key] for key in archive.files}
            _check_case(meta,z,polarity)
            pts=_read_frame_pts(meta['video_path'])
            if len(pts)!=int(meta['decoded_frames']):
                raise AssertionError(f'Decoded frame count changed: {sid}')
            asset=ASSETS/sid
            rate,waveform=_decode_word_audio(meta['video_path'],asset/'audio_16k.wav')
            mapping=_word_mapping(meta,z,pts,rate,waveform)
            mapping.to_csv(asset/'word_mapping.csv',index=False,float_format='%.6f')
            frames=_representative_frames(meta['video_path'],pts,
                                           (float(mapping.start_s.min()),float(mapping.end_s.max())))
            figure=_plot_case(letter,meta,z,mapping,waveform,frames,chinese)
            stem=f'fig01{letter}_{sid}'
            figure.savefig(FIGURES/f'{stem}.pdf',bbox_inches='tight')
            figure.savefig(FIGURES/f'{stem}.png',dpi=300,bbox_inches='tight')
            pdf.savefig(figure,bbox_inches='tight')
            plt.close(figure)
            summary.append({'figure':stem,'sample_id':sid,'polarity':polarity,
                            'label':float(meta['label']), 'duration_s':float(meta['duration']),
                            'official_words':len(mapping), 'accepted_words':len(mapping),
                            'min_asr_score':mapping['asr_alignment_score'].min(),
                            'mean_asr_score':mapping['asr_alignment_score'].mean(),
                            'source_video_sha256':meta['sha256_video'],
                            'original_video_path':meta['video_path'],
                            'figure_frame_indices_0based':';'.join(str(index) for index,_,_ in frames),
                            'figure_frame_pts_s':';'.join(f'{time:.6f}' for _,time,_ in frames),
                            'audio_rate_hz':rate,'original_frame_count':len(pts),
                            'audio_sample_count':len(waveform)})
            case_data.append({'letter':letter,'sample_id':sid,'chinese_polarity':chinese,
                              'label':float(meta['label']), 'official_text':meta['official_text'],
                              'mapping':mapping})
    if len(summary)!=3:
        raise AssertionError(f'Expected all three cases; only {len(summary)} were available')
    pd.DataFrame(summary).to_csv(TABLES/'05_typical_case_validation.csv',index=False,float_format='%.6f')
    page=config.OUT/'paper/alignment_review.html'
    page.parent.mkdir(parents=True,exist_ok=True)
    page.write_text(_review_page(case_data),encoding='utf-8')
    manifest={'method':'Original video PTS; 16 kHz mono source waveform; WhisperX accepted word intervals; stored native/pooled vectors',
              'figure_channel_normalization':'Per-channel z-score across official words within each clip, clipped to [-2,2]; visualization only',
              'word_time_acceptance':'Official-ASR lexical match, ASR word score >=0.1, clip match coverage >=0.55',
              'visual_frame_support_s':1/config.FPS,
              'cases':summary,'combined_pdf':str(multipage),'review_html':str(page)}
    (FIGURES/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'cases':summary,'pdf':multipage,'review_html':page}
