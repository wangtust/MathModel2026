"""Auditable journal figure and listening page for translated-text examples."""
from __future__ import annotations

import html
import json
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .. import config
from .cases import (_decode_word_audio, _letterbox_frame,
                           _read_frame_pts, _representative_frames)


EXAMPLE = '-yRb-Jum7EQ_6'


def _hebrew_image(text: str) -> np.ndarray:
    """Pillow/RAQM handles right-to-left glyph ordering in the exported PDF."""
    canvas=Image.new('RGB',(1800,92),'white')
    draw=ImageDraw.Draw(canvas)
    # DejaVu Sans covers both Hebrew letters and Latin punctuation used by ASR.
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',30)
    draw.text((1780,10),text[:120],font=font,fill='#33454d',
              direction='rtl',language='he',anchor='ra')
    return np.asarray(canvas)


def _figure(meta: dict, z: dict, waveform: np.ndarray, frames: list):
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],
                         'font.size':8,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.edgecolor':'#66757b','axes.linewidth':.6})
    duration=float(meta['duration'])
    a,b=float(z['crosslingual_clip_start']),float(z['crosslingual_clip_end'])
    fig=plt.figure(figsize=(7.35,5.35),facecolor='white')
    grid=fig.add_gridspec(3,1,left=.105,right=.94,top=.82,bottom=.25,hspace=.52,
                          height_ratios=[1.12,1.15,.92])
    video,wave,features=[fig.add_subplot(grid[i]) for i in range(3)]
    fig.suptitle('跨语言案例：译文与原声的片段级时间对应',
                 x=.105,y=.978,ha='left',fontsize=10.5,fontweight='semibold')
    fig.text(.105,.918,
             f'{meta["sample_id"]}   ·   希伯来语原声（语种概率 {meta["speech_language_confidence"]:.3f}）'
             f'   ·   原视频 {duration:.2f} s',
             fontsize=7.5,color='#44505b')
    fig.text(.105,.871,'官方英语译文：'+textwrap.shorten(meta['official_text'],
             width=114,placeholder='…'),fontsize=7.3,color='#245f54')

    video.set_xlim(0,duration);video.set_ylim(0,1)
    video.set_yticks([])
    video.set_title('(a) 原视频帧  |  时间戳为源视频 PTS，均在译文候选区间内',
                    loc='left',fontsize=8,pad=2)
    card_width=(b-a)/4*.91
    for index,pts,rgb in frames:
        video.imshow(_letterbox_frame(rgb),extent=(max(0,pts-card_width/2),
                     min(duration,pts+card_width/2),.25,.90),aspect='auto')
        video.text(pts,.10,f'{pts:.2f} s',ha='center',fontsize=6.2,color='#36454d')
        video.plot([pts,pts],[.02,.08],color='#2e786d',lw=1.3)

    wave.set_title('(b) 原声波形  |  蓝色区间关联官方译文；灰色结尾无关联',
                   loc='left',fontsize=8,pad=2)
    step=max(1,int(np.ceil(len(waveform)/2500)))
    count=len(waveform)//step
    bins=waveform[:count*step].reshape(count,step)
    time=(np.arange(count)+.5)*step/16000
    wave.axvspan(a,b,color='#b8d9ea',alpha=.63,zorder=0)
    wave.axvspan(b,duration,color='#dfe5e8',alpha=.65,zorder=0)
    wave.fill_between(time,bins.min(axis=1),bins.max(axis=1),
                      color='#bb704e',linewidth=0,alpha=.88)
    peak=max(.05,float(np.max(np.abs(waveform))))
    wave.set_ylim(-peak*1.18,peak*1.18);wave.set_xlim(0,duration)
    wave.set_ylabel('幅值',fontsize=7)
    wave.text((a+b)/2,peak*.86,f'译文候选 {a:.2f}–{b:.2f} s',ha='center',
              va='top',fontsize=6,color='#285b78')
    wave.text((b+duration)/2,peak*.86,'未关联语音',ha='center',va='top',
              fontsize=6,color='#5b6570')

    features.set_title('(c) 三模态特征有效区间  |  数字为原始向量维数',
                       loc='left',fontsize=8,pad=2)
    tracks=[('T  官方英语译文',768,bool(z['crosslingual_clip_text_mask']),'#64a68e'),
            ('A  真人语音',25,bool(z['crosslingual_clip_audio_mask']),'#688fb5'),
            ('V  人脸表情',52,bool(z['crosslingual_clip_vision_mask']),'#a581ac')]
    features.set_xlim(0,duration);features.set_ylim(-.15,3.05)
    for i,(name,dims,valid,color) in enumerate(tracks):
        y=2.4-i*1.0
        features.broken_barh([(a,b-a)],(y-.2,.4),facecolors=color if valid else '#d4d8db',
                              edgecolors='#65747a',linewidth=.5)
        features.text(a+.10,y,f'{name}  ·  {dims}维  ·  {"有效" if valid else "不确定"}',
                      va='center',fontsize=7,color='white' if valid else '#33414a')
    features.axvline(b,color='#74848d',lw=.6,ls='--')
    features.set_yticks([]);features.set_xlabel('原视频时间（秒）',fontsize=7)
    features.grid(axis='x',color='#a6b5b2',lw=.4,alpha=.3)

    for ax in (video,wave,features):
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='x',labelsize=7,length=2)
    for ax in (video,wave):ax.tick_params(axis='x',labelbottom=False)
    fig.text(.105,.171,
             f'整段情感标签：{float(meta["label"]):+.1f}（{meta["annotation"]}）；标签不是逐词或逐帧标注。',
             fontsize=7.4,color='#374850')
    fig.text(.105,.116,
             '译文内容词只帮助定位语音句段；不会据此生成官方英语词的发音时间。',
             fontsize=7.1,color='#4d5960')
    fig.text(.105,.072,
             '自动识别和英译仍需原声核听；源语转写与两版英文对照见补充图 6b。',
             fontsize=7.1,color='#4d5960')
    return fig


def _text_evidence(meta: dict):
    """Supplementary text evidence kept apart from the time-axis figure."""
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],
                         'font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
    fig=plt.figure(figsize=(7.35,4.15),facecolor='white')
    ax=fig.add_axes((.095,.10,.85,.73));ax.axis('off')
    fig.suptitle('补充图 6b  跨语言文本核验依据',x=.095,y=.97,
                 ha='left',fontsize=10.5,fontweight='semibold')
    fig.text(.095,.885,f'{meta["sample_id"]}  ·  源语言 ASR 与机器英译均由模型估计',
             fontsize=7.5,color='#53616a')
    ax.text(0,.91,'(a) 希伯来语原声自动转写',fontsize=8,fontweight='semibold',
            color='#344b52',transform=ax.transAxes)
    source=' '.join(s['text'] for s in meta['source_asr_segments'])
    ax.imshow(_hebrew_image(source),extent=(0,1,.76,.86),aspect='auto',
              transform=ax.transAxes)
    ax.text(0,.68,'(b) 附件官方英语译文',fontsize=8,fontweight='semibold',
            color='#245f54',transform=ax.transAxes)
    ax.text(0,.61,textwrap.fill(meta['official_text'],105),fontsize=7.7,
            color='#344b52',va='top',linespacing=1.5,transform=ax.transAxes)
    selected=[s for s in meta['machine_translation_segments'] if s['candidate_support']]
    unmatched=[s for s in meta['machine_translation_segments'] if not s['candidate_support']]
    ax.text(0,.43,'(c) 模型英译候选及原声区间',fontsize=8,fontweight='semibold',
            color='#344b52',transform=ax.transAxes)
    y=.36
    for s in selected+unmatched:
        state='候选关联' if s['candidate_support'] else '未关联'
        prefix=f'{s["start"]:.2f}–{s["end"]:.2f} s  [{state}]  '
        content=textwrap.fill(prefix+s['text'],width=100,
                              subsequent_indent=' '*len(prefix))
        ax.text(.015,y,content,
                fontsize=7.2,color='#285b78' if s['candidate_support'] else '#76635e',
                va='top',linespacing=1.4,transform=ax.transAxes)
        y-=.10+.05*content.count('\n')
    fig.text(.095,.054,'候选关联基于英文内容词重合；自动翻译无法独立证明语义等价，需人工核听。',
             fontsize=7.1,color='#53616a')
    return fig


def _review_html(cases: list[dict]) -> str:
    sections=[]
    for meta in cases:
        sid=meta['sample_id']
        rows=[]
        for i,segment in enumerate(meta['machine_translation_segments']):
            mark='候选关联' if segment['candidate_support'] else '未关联'
            rows.append(f'<tr><td>{i+1}</td><td>{segment["start"]:.2f}–{segment["end"]:.2f}</td>'
                        f'<td>{mark}</td><td>{html.escape(segment["text"])}</td>'
                        f'<td>{segment["shared_content_count"]}</td></tr>')
        native=' '.join(s['text'] for s in meta['source_asr_segments'])
        sections.append(f'''<section><h2>{html.escape(sid)} · {meta['speech_language_confidence']:.3f}</h2>
<p><b>官方英语译文：</b>{html.escape(meta['official_text'])}</p>
<p><b>源语言自动转写（希伯来语）：</b><span dir="rtl" lang="he">{html.escape(native)}</span></p>
<audio controls src="../case_assets/{html.escape(sid)}/audio_16k.wav"></audio>
<p>源视频中的候选语音区间：{meta['crosslingual_support_start']:.2f}–{meta['crosslingual_support_end']:.2f} 秒。</p>
<table><thead><tr><th>序号</th><th>原声时间/s</th><th>候选状态</th><th>机器英译</th><th>共有词干数</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>原声译文是自动模型结果；官方英语词均无独立发音时间。原音频可在播放器中按表内时间核听。</p></section>''')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>跨语言语音与英语译文核验</title>
<style>body{max-width:960px;margin:auto;padding:24px;font-family:system-ui,"Microsoft YaHei",sans-serif;color:#273742;background:#f7f8f7;line-height:1.5}section{padding:20px;margin:20px 0;background:white;border:1px solid #d8dfdd;border-radius:7px}audio{width:100%}table{border-collapse:collapse;width:100%;font-size:.88rem}th,td{border-bottom:1px solid #dce3df;padding:6px;text-align:left}th{background:#edf3f1}span[dir=rtl]{display:block;text-align:right;font-size:1.08rem;padding:8px;background:#f3f5f6}</style>
<h1>跨语言译文与希伯来语原声的片段级核验</h1>
<p>这些样本的英文 text 是译文候选。表中时间是原音频的机器翻译片段时间；“共有词干”仅用于辅助选片，不证明语义等价。</p>'''+''.join(sections)+'</html>'


def create_crosslingual_figure() -> dict:
    logs=config.OUT/'logs/samples'
    ids=('-yRb-Jum7EQ_1','-yRb-Jum7EQ_5','-yRb-Jum7EQ_6')
    cases=[json.loads((logs/f'{sid}.json').read_text()) for sid in ids]
    if any(meta.get('text_audio_status')!='crosslingual_translation_candidate' for meta in cases):
        raise AssertionError('The three verified source-language cases are not ready')
    for meta in cases:
        _decode_word_audio(meta['video_path'],config.OUT/'case_assets'/meta['sample_id']/'audio_16k.wav')
    meta=cases[2]
    with np.load(config.OUT/'features'/f'{EXAMPLE}.npz') as archive:
        z={key:archive[key] for key in archive.files}
    _,waveform=_decode_word_audio(meta['video_path'],
                                  config.OUT/'case_assets'/EXAMPLE/'audio_16k.wav')
    pts=_read_frame_pts(meta['video_path'])
    frames=_representative_frames(meta['video_path'],pts,
             (float(z['crosslingual_clip_start']),float(z['crosslingual_clip_end'])),number=4)
    figure=_figure(meta,z,waveform,frames)
    output=config.OUT/'figures'
    output.mkdir(parents=True,exist_ok=True)
    stem='fig06_crosslingual_hebrew_translation'
    figure.savefig(output/f'{stem}.pdf',bbox_inches='tight')
    figure.savefig(output/f'{stem}.png',dpi=300,bbox_inches='tight')
    plt.close(figure)
    supplement=_text_evidence(meta)
    supplement_stem='fig06b_crosslingual_text_evidence'
    supplement.savefig(output/f'{supplement_stem}.pdf',bbox_inches='tight')
    supplement.savefig(output/f'{supplement_stem}.png',dpi=300,bbox_inches='tight')
    plt.close(supplement)
    page=config.OUT/'paper/crosslingual_review.html'
    page.write_text(_review_html(cases),encoding='utf-8')
    table=pd.DataFrame([{'sample_id':m['sample_id'],
                         'official_english_text':m['official_text'],
                         'source_language':m['speech_language'],
                         'language_probability':m['speech_language_confidence'],
                         'source_video_sha256':m['sha256_video'],
                         'candidate_start_s':m['crosslingual_support_start'],
                         'candidate_end_s':m['crosslingual_support_end'],
                         'selected_segments':m['crosslingual_selected_segments'],
                         'word_times_created':0,
                         'clip_audio_valid':m['crosslingual_audio_valid'],
                         'clip_vision_valid':m['crosslingual_vision_valid']}
                        for m in cases])
    table.to_csv(config.OUT/'tables/09_crosslingual_case_validation.csv',index=False)
    return {'figure':output/f'{stem}.pdf',
            'supplementary_figure':output/f'{supplement_stem}.pdf',
            'review_html':page,
            'validation_table':config.OUT/'tables/09_crosslingual_case_validation.csv'}
