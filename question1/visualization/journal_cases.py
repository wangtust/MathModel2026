"""Compact, publication-oriented case figures built from auditable Q1 masks."""
from __future__ import annotations

import json
import textwrap

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config
from .cases import _read_frame_pts, _representative_frames


TEAL='#39786f'
BLUE='#638cac'
OCHRE='#b3834a'
AMBER='#bb7156'
INK='#2d4149'
LIGHT='#f2f5f4'


def _open_case(sid):
    meta=json.loads((config.OUT/'logs/samples'/f'{sid}.json').read_text())
    with np.load(config.OUT/'features'/f'{sid}.npz') as archive:
        z={key:archive[key] for key in archive.files}
    return meta,z


def _intervals(times,mask,half,duration):
    selected=[]
    for t,valid in zip(times,mask):
        if valid:
            start,end=max(0.,float(t)-half),min(duration,float(t)+half)
            if selected and start<=selected[-1][1]+.025:
                selected[-1]=(selected[-1][0],max(end,selected[-1][1]))
            else:selected.append((start,end))
    return selected


def _bands(ax,intervals,y,color,height=.38):
    if intervals:
        ax.broken_barh([(float(a),float(b-a)) for a,b in intervals if b>a],
                       (y-height/2,height),facecolors=color,edgecolors='none')


def _timeline(ax,meta,z,*,compact=False,number_words=False):
    duration=float(meta['duration'])
    lanes=[2.45,1.45,.45]
    for y in lanes:ax.axhspan(y-.23,y+.23,color=LIGHT,zorder=-2)
    speech=[(float(a),float(b)) for a,b in meta['speech_segments']]
    face=_intervals(z['vision_time'],z['vision_valid_mask'],.5/config.FPS,duration)
    multiple=_intervals(z['vision_time'],z['vision_multiface_mask'],.5/config.FPS,duration)
    ws=z['word_start'];we=z['word_end']
    words=[(float(a),float(b)) for a,b in zip(ws,we) if np.isfinite(a) and np.isfinite(b) and b>a]
    _bands(ax,speech,lanes[0],BLUE)
    _bands(ax,face,lanes[1],TEAL)
    _bands(ax,multiple,lanes[1],AMBER,height=.47)
    _bands(ax,words,lanes[2],OCHRE)
    if number_words:
        for i,(a,b) in enumerate(zip(ws,we)):
            if np.isfinite(a) and np.isfinite(b) and b>a and b-a>.095:
                ax.text((a+b)/2,lanes[2],str(i+1),ha='center',va='center',
                        fontsize=5.8,color='white',fontweight='semibold')
    if not speech:ax.text(duration/2,lanes[0],'无真人语音',ha='center',va='center',fontsize=6.8,color='#69777c')
    if not face and not multiple:
        message='检测到人脸，表情特征无效' if meta['face_present'] else '未检测到目标人脸'
        ax.text(duration/2,lanes[1],message,ha='center',va='center',fontsize=6.8,color='#69777c')
    if not words:ax.text(duration/2,lanes[2],'无可靠官方词时间',ha='center',va='center',fontsize=6.8,color='#69777c')
    ax.set_xlim(0,max(duration,.5));ax.set_ylim(0,3.0)
    ax.set_yticks(lanes,['真人语音','可用人脸','官方词时间'])
    ax.set_xlabel('原视频时间（秒）',fontsize=7)
    ax.tick_params(axis='both',labelsize=6.5,length=2)
    ax.grid(axis='x',color='#b9c7c3',lw=.4,alpha=.4)
    ax.set_axisbelow(True)
    ax.spines[['top','right','left']].set_visible(False)
    if compact:
        ax.set_xlabel('时间 / s',fontsize=6.5,labelpad=1)
        ax.tick_params(axis='both',labelsize=5.8)
        ax.set_yticklabels(['语音','人脸','词时间'],fontsize=6.2)


def _source_frame(meta,z,kind):
    times=z['vision_time']
    if kind=='multiple_faces':mask=z['vision_multiface_mask'].astype(bool)
    elif kind in ('typical','no_speech'):mask=z['vision_valid_mask'].astype(bool)
    else:mask=np.ones(len(times),bool)
    idx=np.flatnonzero(mask)
    target=float(times[idx[len(idx)//2]]) if len(idx) else float(meta['duration'])/2
    pts=_read_frame_pts(meta['video_path'])
    frame_index,t,frame=_representative_frames(meta['video_path'],pts,(target,target+.0001),number=1)[0]
    return frame_index,t,frame


def _save(fig,stem,out):
    fig.savefig(out/f'{stem}.pdf',bbox_inches='tight')
    fig.savefig(out/f'{stem}.png',dpi=300,bbox_inches='tight')
    plt.close(fig)


def _mismatch(meta,z,out):
    sid=meta['sample_id']
    fig=plt.figure(figsize=(7.35,4.3),facecolor='white')
    grid=fig.add_gridspec(2,2,left=.08,right=.96,top=.86,bottom=.12,
                          hspace=.38,wspace=.1,height_ratios=[1.33,1.05],width_ratios=[.39,.61])
    image_ax=fig.add_subplot(grid[0,0]);text_ax=fig.add_subplot(grid[0,1]);timeline=fig.add_subplot(grid[1,:])
    fig.suptitle('文本与原声不符的案例：保留模态，停止逐词配对',
                 x=.08,y=.98,ha='left',fontsize=10.5,fontweight='semibold')
    fig.text(.08,.916,f'{sid}  ·  官方词 {meta["word_count"]} 个  ·  接受词时间 0 个',
             fontsize=7.5,color='#53616a')
    _,t,frame=_source_frame(meta,z,'mismatch')
    image_ax.imshow(frame);image_ax.axis('off')
    image_ax.set_title('(a) 原视频帧',loc='left',fontsize=8,pad=3)
    image_ax.text(.0,-.08,f'原视频 PTS {t:.2f} s；此片段未检测到目标人脸',
                  transform=image_ax.transAxes,fontsize=6.4,color='#53616a')
    text_ax.axis('off')
    text_ax.set_title('(b) 官方文本与 ASR 假设',loc='left',fontsize=8,pad=3)
    text_ax.text(0,.91,'附件官方文本',fontweight='semibold',fontsize=7.2,color=TEAL,
                 transform=text_ax.transAxes)
    text_ax.text(0,.80,textwrap.fill(meta['official_text'],64),fontsize=7.1,
                 color=INK,va='top',linespacing=1.45,transform=text_ax.transAxes)
    text_ax.text(0,.42,'自动识别的原声文字',fontweight='semibold',fontsize=7.2,
                 color=AMBER,transform=text_ax.transAxes)
    text_ax.text(0,.32,textwrap.fill(meta['asr_text'],64),fontsize=7.1,
                 color=INK,va='top',linespacing=1.45,transform=text_ax.transAxes)
    text_ax.text(0,.05,f'词形覆盖 {meta["match"]["matched_words"]}/{meta["word_count"]}；'
                    '这是复核候选，不直接判定附件标注错误。',
                 fontsize=6.5,color='#815e55',transform=text_ax.transAxes)
    timeline.set_title('(c) 原声与模态状态  |  灰色词轨说明没有可接受的逐词时间',
                       loc='left',fontsize=8,pad=4)
    _timeline(timeline,meta,z)
    _save(fig,f'fig04_mismatch_alignment_{sid}',out)
    return {'figure':'fig04_mismatch_alignment','panel':'(a)–(c)','sample_id':sid,
            'original_frame_pts_s':t,'finding':'text_audio_mismatch_candidate'}


def _visual_grid(cases,out):
    kinds=[('typical','(a) 有效说话人'),('no_face','(b) 画面无可用目标脸'),
           ('multiple_faces','(c) 多人脸目标不确定'),('no_speech','(d) 无真人语音')]
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.65))
    fig.subplots_adjust(left=.04,right=.98,top=.91,bottom=.07,wspace=.065,hspace=.30)
    fig.suptitle('原视频画面中的模态有效性案例',x=.04,y=.985,ha='left',
                 fontsize=10.5,fontweight='semibold')
    rows=[]
    for ax,(kind,title) in zip(axes.flat,kinds):
        sid='-NFrJFQijFE_2' if kind=='no_face' and \
            (config.OUT/'logs/samples/-NFrJFQijFE_2.json').exists() else cases[kind]
        meta,z=_open_case(sid)
        frame_index,t,frame=_source_frame(meta,z,kind)
        ax.imshow(frame,aspect='equal');ax.axis('off')
        ax.set_title(title,loc='left',fontsize=8,pad=4)
        ax.text(0,-.06,f'{meta["sample_id"]}  |  原视频帧 {frame_index}，PTS {t:.2f} s',
                transform=ax.transAxes,fontsize=6.4,color='#53616a',va='top')
        rows.append({'figure':'fig05_visual_evidence','panel':title[:3],
                     'sample_id':meta['sample_id'],'original_frame_pts_s':t,
                     'original_frame_index_0based':frame_index,'finding':kind})
    _save(fig,'fig05_visual_evidence',out)
    return rows


def _typical(meta,z,out):
    sid=meta['sample_id']
    fig,ax=plt.subplots(figsize=(7.25,3.45))
    fig.subplots_adjust(left=.12,right=.97,top=.78,bottom=.27)
    fig.suptitle('典型样本的语音—人脸—官方词时间轴',x=.12,y=.975,
                 ha='left',fontsize=10.5,fontweight='semibold')
    timed=int(np.isfinite(z['word_start']).sum())
    fig.text(.12,.884,f'{sid}  ·  {timed}/{meta["word_count"]} 个官方词获得真实发音时间',
             fontsize=7.4,color='#53616a')
    _timeline(ax,meta,z,number_words=True)
    fig.text(.12,.10,'官方文本：'+meta['official_text'],fontsize=7.1,color=INK)
    fig.text(.12,.045,'词轨中的数字对应官方词序；逐词词形、边界及原视频帧号见配套映射表。',
             fontsize=6.5,color='#53616a')
    _save(fig,f'fig_case_typical_{sid}',out)
    return {'figure':'fig_case_typical','panel':'whole','sample_id':sid,
            'original_frame_pts_s':np.nan,'finding':'direct_word_alignment'}


def _anomalies(cases,out):
    kinds=[('no_face','(a) 无可用目标人脸'),('no_speech','(b) 无真人语音'),
           ('multiple_faces','(c) 多人脸目标歧义'),('intermittent_face','(d) 人脸间歇出现')]
    fig,axes=plt.subplots(2,2,figsize=(7.25,5.25))
    fig.subplots_adjust(left=.12,right=.98,top=.87,bottom=.13,wspace=.28,hspace=.55)
    fig.suptitle('四种异常样本的模态有效区间',x=.12,y=.982,ha='left',
                 fontsize=10.5,fontweight='semibold')
    fig.text(.12,.92,'蓝：真人语音   绿：可用人脸   黄：已接受词时间   橙：多人脸目标不确定',
             fontsize=7,color='#53616a')
    rows=[]
    for ax,(kind,title) in zip(axes.flat,kinds):
        meta,z=_open_case(cases[kind])
        ax.set_title(f'{title}  |  {meta["sample_id"]}',loc='left',fontsize=7.2,pad=4)
        _timeline(ax,meta,z,compact=True)
        rows.append({'figure':'fig07_quality_exception_timeline','panel':title[:3],
                     'sample_id':meta['sample_id'],'original_frame_pts_s':np.nan,'finding':kind})
    fig.text(.12,.04,'多人人脸以时间区间带表示；空模态显示明确的缺失状态，而不以大面积空白代替结果。',
             fontsize=6.5,color='#53616a')
    _save(fig,'fig07_quality_exception_timeline',out)
    return rows


def create_journal_case_figures(cases: dict, out) -> list[dict]:
    """Replace stand-alone diagnostic screenshots with reviewer-ready panels."""
    out.mkdir(parents=True,exist_ok=True)
    needed={'typical','no_face','multiple_faces','no_speech','mismatch','intermittent_face'}
    if not needed.issubset(cases):
        return []
    meta,z=_open_case(cases['mismatch'])
    records=[_mismatch(meta,z,out)]
    records.extend(_visual_grid(cases,out))
    meta,z=_open_case(cases['typical'])
    records.append(_typical(meta,z,out))
    records.extend(_anomalies(cases,out))
    pd.DataFrame(records).to_csv(config.OUT/'tables/10_journal_case_manifest.csv',index=False)
    return records
