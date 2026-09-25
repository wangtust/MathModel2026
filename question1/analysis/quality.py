"""Reproducibility tables, independent invariants, and journal-size figures."""
import json
import subprocess
import hashlib
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config


def _style():
    plt.rcParams.update({'font.family': ['Microsoft YaHei','DejaVu Sans'], 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .65, 'axes.edgecolor':'#66757b',
                         'pdf.fonttype': 42, 'ps.fonttype':42,
                         'savefig.dpi': 300, 'figure.dpi': 130})


def _save(fig, stem, out):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(out/f'{stem}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def _git_info():
    paths = {'MMSA-FET':'third_party/MMSA-FET', 'TalkNet-ASD':'third_party/TalkNet-ASD',
             'silero-vad':'third_party/silero-vad', 'whisperX':'third_party/whisperX',
             'CMU-MultimodalSDK':'third_party/CMU-MultimodalSDK'}
    info = {}
    for name, rel in paths.items():
        path = config.PROJECT/rel
        if path.exists():
            commit = subprocess.run(['git','rev-parse','HEAD'],cwd=path,
                                    text=True,capture_output=True)
            dirty = subprocess.run(['git','status','--porcelain'],cwd=path,
                                   text=True,capture_output=True)
            info[name] = {'commit':commit.stdout.strip(), 'local_patch':bool(dirty.stdout.strip())}
    return info


def _sha(path):
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def _versions():
    result={}
    for name in ['numpy','pandas','torch','mediapipe','opensmile','transformers',
                 'whisperx','silero-vad','opencv-python','matplotlib']:
        try:result[name]=version(name)
        except PackageNotFoundError:result[name]='not installed as distribution'
    return result


def _cached_model_hashes():
    cache=Path.home()/'.cache/huggingface/hub'
    models={'bert-base-uncased':('models--bert-base-uncased','model.safetensors'),
            'faster-whisper-base.en':('models--Systran--faster-whisper-base.en','model.bin'),
            'faster-whisper-base-multilingual':('models--Systran--faster-whisper-base','model.bin'),
            'faster-whisper-small-multilingual':('models--Systran--faster-whisper-small','model.bin')}
    result={}
    for label,(directory,filename) in models.items():
        root=cache/directory
        ref=root/'refs/main'
        if ref.exists():
            revision=ref.read_text().strip()
            weight=root/'snapshots'/revision/filename
            result[label]={'revision':revision,'weight_sha256':_sha(weight) if weight.exists() else None}
    align=Path.home()/'.cache/torch/hub/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth'
    if align.exists():result['whisperx_en_align']={'weight_sha256':_sha(align)}
    return result


def audit(manifest, summary):
    actual = set(summary.sample_id) if len(summary) else set()
    expected = set(manifest.sample_id)
    by_id=manifest.set_index('sample_id')
    errors = []
    for sid in sorted(expected):
        if sid not in actual:
            errors.append({'sample_id':sid,'level':'error','issue':'missing_sample_output'})
            continue
        feature = config.OUT/'features'/f'{sid}.npz'
        log = config.OUT/'logs/samples'/f'{sid}.json'
        if not feature.exists() or not log.exists():
            errors.append({'sample_id':sid,'level':'error','issue':'missing_feature_or_log'})
            continue
        with open(log) as f:
            meta = json.load(f)
        source=Path(by_id.loc[sid,'video_path'])
        if not source.is_file() or source != Path(meta['video_path']):
            errors.append({'sample_id':sid,'level':'error','issue':'source_path_mismatch'})
        elif meta.get('sha256_video') != _sha(source):
            errors.append({'sample_id':sid,'level':'error','issue':'source_hash_mismatch'})
        if meta.get('official_text') != by_id.loc[sid,'official_text']:
            errors.append({'sample_id':sid,'level':'error','issue':'official_text_mismatch'})
        if not np.isclose(float(meta['label']),float(by_id.loc[sid,'label'])):
            errors.append({'sample_id':sid,'level':'error','issue':'label_mismatch'})
        if meta.get('annotation') != by_id.loc[sid,'annotation']:
            errors.append({'sample_id':sid,'level':'error','issue':'annotation_mismatch'})
        with np.load(feature) as z:
            count = meta['word_count']
            if count != len(meta['official_words']) or count != len(z['word_start']):
                errors.append({'sample_id':sid,'level':'error','issue':'word_count_mismatch'})
            if z['word_text'].shape != (config.MAX_WORDS,768):
                errors.append({'sample_id':sid,'level':'error','issue':'fixed_text_shape'})
            if z['word_audio'].shape[0] != config.MAX_WORDS or z['word_vision'].shape[0] != config.MAX_WORDS:
                errors.append({'sample_id':sid,'level':'error','issue':'fixed_av_shape'})
            if not np.all(z['word_valid'][:count] == 1) or np.any(z['word_valid'][count:]):
                errors.append({'sample_id':sid,'level':'error','issue':'padding_mask'})
            for key in ('audio_time','vision_time','second_start'):
                if np.any(np.diff(z[key]) < 0):
                    errors.append({'sample_id':sid,'level':'error','issue':key+'_not_monotone'})
            timed=np.isfinite(z['word_start']) & np.isfinite(z['word_end'])
            if np.any(timed):
                start=z['word_start'][timed];end=z['word_end'][timed]
                if np.any(start<0) or np.any(end<=start) or np.any(end>meta['duration']+.5):
                    errors.append({'sample_id':sid,'level':'error','issue':'invalid_word_times'})
                if np.any(np.diff(start)<0):
                    errors.append({'sample_id':sid,'level':'error','issue':'word_times_not_monotone'})
            if np.any(z['word_audio_mask'] > z['word_valid']) or np.any(z['word_vision_mask'] > z['word_valid']):
                errors.append({'sample_id':sid,'level':'error','issue':'invalid_modality_mask'})
            if np.any(z['word_audio_mask'][count:]) or np.any(z['word_vision_mask'][count:]):
                errors.append({'sample_id':sid,'level':'error','issue':'padding_has_modality'})
            if not meta['speech_present'] and np.any(z['word_audio_mask']):
                errors.append({'sample_id':sid,'level':'error','issue':'audio_without_speech'})
            if not meta['face_present'] and np.any(z['word_vision_mask']):
                errors.append({'sample_id':sid,'level':'error','issue':'vision_without_face'})
            if meta.get('asr_unreliable') and np.any(timed):
                errors.append({'sample_id':sid,'level':'error','issue':'unreliable_asr_has_word_times'})
            if meta.get('text_audio_status')=='crosslingual_translation_candidate':
                needed={'crosslingual_start','crosslingual_end','crosslingual_clip_text',
                        'crosslingual_clip_audio','crosslingual_clip_vision',
                        'crosslingual_clip_text_mask','crosslingual_clip_audio_mask',
                        'crosslingual_clip_vision_mask'}
                if not needed.issubset(z.files):
                    errors.append({'sample_id':sid,'level':'error','issue':'missing_crosslingual_features'})
                else:
                    a=z['crosslingual_start'];b=z['crosslingual_end']
                    if (not len(a) or len(a)!=len(b) or np.any(a<0) or
                            np.any(b<=a) or np.any(b>float(meta['duration'])+.05)):
                        errors.append({'sample_id':sid,'level':'error','issue':'invalid_crosslingual_intervals'})
                    if np.any(timed) or np.any(z['word_audio_mask']) or np.any(z['word_vision_mask']):
                        errors.append({'sample_id':sid,'level':'error','issue':'translated_english_has_false_word_alignment'})
                    if (z['crosslingual_clip_text'].shape!=(768,) or
                            z['crosslingual_clip_audio'].shape!=(25,) or
                            z['crosslingual_clip_vision'].shape!=(52,)):
                        errors.append({'sample_id':sid,'level':'error','issue':'crosslingual_clip_feature_shape'})
                    if meta['target_face_uncertain'] and z['crosslingual_clip_vision_mask']:
                        errors.append({'sample_id':sid,'level':'error','issue':'uncertain_face_used_crosslingually'})
                if (meta.get('speech_language','en')=='en' or
                        float(meta.get('speech_language_confidence',0))<.8 or
                        not meta.get('source_asr_segments')):
                    errors.append({'sample_id':sid,'level':'error','issue':'unsupported_crosslingual_claim'})
    if len(summary) != len(actual):
        errors.append({'sample_id':'ALL','level':'error','issue':'duplicate_output_ids'})
    return pd.DataFrame(errors,columns=['sample_id','level','issue'])


def _fig_availability(table, out):
    fields = [('可解码音轨','audio_stream_present'),
              ('真人语音','speech_present'),
              ('检测到人脸','face_present'),
              ('有效人脸表情','face_feature_valid'),
              ('英语词级对齐','text_audio_match')]
    fig, ax = plt.subplots(figsize=(6.7,2.7))
    fig.subplots_adjust(left=.22,right=.9,top=.89,bottom=.19)
    y = np.arange(len(fields))
    vals = [int(table[k].fillna(False).astype(bool).sum()) for _,k in fields]
    ax.barh(y,vals,color='#3d7771',height=.53,edgecolor='none')
    ax.set_yticks(y,[x for x,_ in fields]); ax.invert_yaxis()
    ax.set_xlim(0,len(table)+8);ax.set_xlabel('样本数（n=100）')
    ax.set_title('原始信号与情感证据的有效覆盖',loc='left',fontsize=9,pad=8)
    for i,v in enumerate(vals): ax.text(v+1,i,f'{v}  ({v/len(table):.0%})',va='center',fontsize=7.5,color='#33444b')
    ax.grid(axis='x',alpha=.14,linewidth=.45);ax.set_axisbelow(True)
    _save(fig,'fig01_modality_availability',out)


def _fig_matrix(table, out):
    events=np.column_stack([
        ~table.speech_present.astype(bool),
        ~table.face_present.astype(bool),
        ~table.face_feature_valid.astype(bool),
        ~table.text_audio_match.astype(bool),
        table.text_audio_status.eq('crosslingual_translation_candidate'),
        table.target_face_uncertain.astype(bool),
    ]).astype(int)
    names=['无人声','无人脸','无有效表情','无可信英语词对齐','跨语译文候选','目标人脸不确定']
    severity=events.sum(axis=1)
    # First group by the number of quality events, then by the event pattern.
    order=np.lexsort((np.arange(len(table)),*tuple(-events[:,j] for j in reversed(range(events.shape[1]))),-severity))
    shown=events[order].T
    pd.DataFrame({'figure_column_1based':np.arange(1,len(order)+1),
                  'sample_id':table.iloc[order].sample_id.to_numpy(),
                  'quality_event_count':severity[order]}).to_csv(
                      config.OUT/'tables/06_status_matrix_order.csv',index=False)
    fig, ax = plt.subplots(figsize=(7.3,2.85))
    fig.subplots_adjust(left=.19,right=.91,top=.83,bottom=.24)
    from matplotlib.colors import ListedColormap
    ax.pcolormesh(np.arange(len(order)+1),np.arange(len(names)+1),shown,
                  cmap=ListedColormap(['#f4f6f5','#39766e']),vmin=0,vmax=1,
                  edgecolors='white',linewidth=.12,rasterized=True)
    ax.set_yticks(np.arange(len(names))+.5,names)
    ax.invert_yaxis()
    ax.set_xticks([.5,19.5,39.5,59.5,79.5,99.5],['1','20','40','60','80','100'])
    ax.set_xlim(0,len(order)+9)
    ax.set_xlabel('按质量事件数量与类型排序的样本编号（原 ID 见配套表）')
    ax.set_title('100 条样本的质量事件矩阵',loc='left',fontsize=9,pad=7)
    ax.text(1,1.065,'深色＝出现该事件；浅色＝未出现',transform=ax.transAxes,
            ha='right',fontsize=7,color='#526068')
    for i,n in enumerate(shown.sum(axis=1)):
        ax.text(len(order)+1,i+.5,f'n={n}',va='center',fontsize=7,color='#38505a')
    for boundary in (20,40,60,80):ax.axvline(boundary,color='#8ea7a0',lw=.45,alpha=.5)
    ax.tick_params(length=0,labelsize=7)
    _save(fig,'fig02_sample_status_matrix',out)


def _fig_temporal_coverage(table, out):
    fig, ax = plt.subplots(figsize=(7.2,4.1))
    fig.subplots_adjust(left=.14,right=.72,top=.88,bottom=.18)
    x=table.word_audio_valid_fraction.fillna(0).to_numpy(float)
    y=table.word_vision_valid_fraction.fillna(0).to_numpy(float)
    paired=table.text_audio_match.astype(bool).to_numpy()
    uncertain=table.target_face_uncertain.astype(bool).to_numpy()
    nonzero=(x>0)|(y>0)
    def bubbles(mask,color,face,label):
        xy=np.column_stack((x[mask],y[mask]))
        if not len(xy):return
        points,counts=np.unique(np.round(xy,6),axis=0,return_counts=True)
        ax.scatter(points[:,0],points[:,1],s=24+16*counts,
                   edgecolor=color,facecolor=face,linewidth=.75,alpha=.82,label=label)
        for (u,v),count in zip(points,counts):
            if count>=4:ax.text(u,v,str(count),ha='center',va='center',fontsize=6.5,
                                color='white' if face!='none' else color)
    bubbles(nonzero & paired & ~uncertain,'#276e68','#3d8f82','已接受英语词时间')
    bubbles(nonzero & (~paired | uncertain),'#9b6557','none','词时间待核/目标脸不确定')
    ax.set(xlim=(-.03,1.09),ylim=(-.03,1.09),
           xlabel='有效语音词 / 官方词',ylabel='有效人脸词 / 官方词')
    ax.set_title('逐词语音与人脸特征的有效覆盖',loc='left',fontsize=9,pad=7)
    ax.set_xticks(np.linspace(0,1,6));ax.set_yticks(np.linspace(0,1,6))
    ax.grid(color='#a5b5b1',alpha=.3,lw=.45);ax.set_axisbelow(True)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=7,markerscale=.68,
               loc='upper left',bbox_to_anchor=(.745,.87))
    zero=~nonzero
    counts=table.loc[zero,'text_audio_status'].value_counts()
    lines=[f'{int(zero.sum())} 条位于 (0, 0)', '',
           f'文本不匹配候选  {int(counts.get("mismatch_candidate",0))}',
           f'跨语译文候选    {int(counts.get("crosslingual_translation_candidate",0))}',
           f'无真人语音      {int(counts.get("no_speech",0))}']
    other=int(zero.sum())-sum(int(counts.get(k,0)) for k in
             ('mismatch_candidate','crosslingual_translation_candidate','no_speech'))
    if other:lines.append(f'其他待复核      {other}')
    fig.text(.765,.64,'零覆盖样本',fontsize=9,fontweight='semibold',color='#2c4b4a')
    fig.text(.765,.59,'\n'.join(lines),fontsize=7.3,color='#41545b',va='top',linespacing=1.7)
    fig.text(.765,.19,'圆面积表示同坐标\n重合的样本数。',fontsize=7,color='#526068',linespacing=1.5)
    fig.text(.14,.055,'跨语言样本在句段级另有特征；此图仅统计官方英语词级有效覆盖。',
             fontsize=6.8,color='#526068')
    _save(fig,'fig03_word_level_coverage',out)



def create_reports():
    _style()
    manifest=pd.read_csv(config.OUT/'tables/01_manifest.csv')
    summary_path=config.OUT/'tables/02_sample_summary.csv'
    table=pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    issues=audit(manifest,table)
    issues.to_csv(config.OUT/'tables/03_audit_issues.csv',index=False)
    if table.empty:return
    table=table.fillna({'matched_word_fraction':0})
    review=table[(~table['text_audio_match'].astype(bool)) |
                 (table['target_face_uncertain'].astype(bool)) |
                 (~table['face_present'].astype(bool)) |
                 (~table['speech_present'].astype(bool))].copy()
    def reason(row):
        tags=[]
        if row['text_audio_status']=='asr_unreliable':tags.append('asr_repetition_review')
        if row['text_audio_status']=='mismatch_candidate':tags.append('text_audio_mismatch_review')
        if row['text_audio_status']=='asr_unavailable':tags.append('asr_unavailable')
        if row['text_audio_status']=='crosslingual_translation_candidate':
            tags.append('crosslingual_translation_review')
        if row.get('language_review_required',False)==True:tags.append('language_id_uncertain')
        if not row['speech_present']:tags.append('no_human_speech')
        if not row['face_present']:tags.append('no_detected_face')
        if row['target_face_uncertain']:tags.append('target_face_uncertain')
        return ';'.join(tags)
    review['review_reason']=review.apply(reason,axis=1)
    review.to_csv(config.OUT/'tables/04_review_candidates.csv',index=False)
    out=config.OUT/'figures'
    out.mkdir(parents=True,exist_ok=True)
    for stale in out.glob('fig*.pdf'):
        stale.unlink()
    for stale in out.glob('fig*.png'):
        stale.unlink()
    _fig_availability(table,out)
    _fig_matrix(table,out)
    _fig_temporal_coverage(table,out)
    cases={}
    full=table[table.speech_present.astype(bool)&table.face_feature_valid.astype(bool)&table.text_audio_match.astype(bool)]
    if len(full):
        preferred='-THoVjtIkeU_2'
        cases['typical']=preferred if preferred in set(full.sample_id) else full.iloc[0].sample_id
    noface=table[~table.face_present.astype(bool)]
    if len(noface):
        # Matched speech with no usable target-face detection isolates this event.
        preferred='-HwX2H8Z4hY_9'
        cases['no_face']=preferred if preferred in set(noface.sample_id) else noface.iloc[0].sample_id
    nospeech=table[~table.speech_present.astype(bool)]
    if len(nospeech):
        face_only=nospeech[nospeech.face_present.astype(bool)]
        cases['no_speech']=(face_only.iloc[0] if len(face_only) else nospeech.iloc[0]).sample_id
    mismatch=table[table.text_audio_status=='mismatch_candidate']
    if len(mismatch):cases['mismatch']=mismatch.sort_values('matched_word_fraction').iloc[0].sample_id
    multiface=table[table.target_face_uncertain.astype(bool)]
    if len(multiface):cases['multiple_faces']=multiface.iloc[0].sample_id
    unreliable=table[table.asr_unreliable.astype(bool)]
    if len(unreliable):cases['asr_unreliable']=unreliable.iloc[0].sample_id
    intermittent=[]
    for sid in table.sample_id:
        with np.load(config.OUT/'features'/f'{sid}.npz') as z:
            f=z['vision_face_mask']
            if len(f)>4 and 0.1<float(np.mean(f))<0.9:intermittent.append(sid)
    if intermittent:cases['intermittent_face']=intermittent[0]
    from question1.visualization.journal_cases import create_journal_case_figures
    journal_cases=create_journal_case_figures(cases,out)
    # The curated word-synchronous cases require all three full-run outputs.
    # A pilot with selected IDs still produces its ordinary reports.
    from question1.visualization.cases import CASES, create_alignment_cases
    case_ready=all((config.OUT/'features'/f'{sid}.npz').exists() and
                   (config.OUT/'logs/samples'/f'{sid}.json').exists()
                   for _,sid,_,_ in CASES)
    alignment=create_alignment_cases() if case_ready else None
    cross_ready=all((config.OUT/'logs/samples'/f'{sid}.json').exists()
                    for sid in ('-yRb-Jum7EQ_1','-yRb-Jum7EQ_5','-yRb-Jum7EQ_6'))
    crosslingual=None
    if cross_ready:
        from question1.visualization.crosslingual import create_crosslingual_figure
        statuses=[json.loads((config.OUT/'logs/samples'/f'{sid}.json').read_text())
                  .get('text_audio_status') for sid in
                  ('-yRb-Jum7EQ_1','-yRb-Jum7EQ_5','-yRb-Jum7EQ_6')]
        if all(status=='crosslingual_translation_candidate' for status in statuses):
            crosslingual=create_crosslingual_figure()
    meta={'data_root':str(config.DATA),'n_manifest':len(manifest),
          'n_outputs':len(table),'n_audit_errors':len(issues),
          'repositories':_git_info(),
          'software_versions':_versions(),
          'asset_sha256':{p.name:_sha(p) for p in (config.PROJECT/'assets').iterdir() if p.is_file()},
          'cached_model_hashes':_cached_model_hashes(),
          'models': {'asr':config.ASR_MODEL,'language_screen':'multilingual Whisper base',
                     'foreign_speech':'multilingual Whisper small',
                     'text':'bert-base-uncased',
                     'audio':'eGeMAPSv02 LowLevelDescriptors','vision':'MediaPipe Face Landmarker blendshapes'},
          'parameters':{'vision_fps':config.FPS,'vad_threshold':config.VAD_THRESHOLD,
                        'match_threshold':config.MATCH_THRESHOLD,'max_words':config.MAX_WORDS,
                        'foreign_language_probability_threshold':.8},
          'case_ids':cases,
          'journal_case_panels':len(journal_cases),
          'alignment_case_ids':[case['sample_id'] for case in alignment['cases']] if alignment else [],
          'crosslingual_case_ids':['-yRb-Jum7EQ_1','-yRb-Jum7EQ_5','-yRb-Jum7EQ_6']
              if crosslingual else []}
    (config.OUT/'run_manifest.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    total_words=total_timed=total_audio_words=total_visual_words=0
    for sid in table.sample_id:
        with np.load(config.OUT/'features'/f'{sid}.npz') as z:
            count=int(np.sum(z['word_valid']))
            total_words+=count
            total_timed+=int(np.sum(np.isfinite(z['word_start'])))
            total_audio_words+=int(np.sum(z['word_audio_mask']))
            total_visual_words+=int(np.sum(z['word_vision_mask']))
    doc=["# 问题一：重建实验结果", "", f"已处理 {len(table)}/{len(manifest)} 条；审计错误 {len(issues)} 条。", "",
         "| 判定项 | 数量 |", "|---|---:|"]
    for label,col in [('真人语音','speech_present'),('检测到人脸','face_present'),
                      ('有效人脸表情特征','face_feature_valid'),('目标人脸不确定','target_face_uncertain'),
                      ('文本与语音匹配','text_audio_match')]:
        doc.append(f'| {label} | {int(table[col].astype(bool).sum())} |')
    doc.append(f'| ASR 重复幻觉待复核 | {int(table.asr_unreliable.astype(bool).sum())} |')
    translated=table.text_audio_status.eq('crosslingual_translation_candidate')
    language_review_count=int(table['language_review_required'].eq(True).sum()) \
        if 'language_review_required' in table else 0
    doc.append(f'| 跨语言译文候选（句段级） | {int(translated.sum())} |')
    doc += ['', f'官方文本共 {total_words} 词；其中有可靠实际发音时间的词 {total_timed} 个，'
            f'有效语音词特征 {total_audio_words} 个，有效人脸表情词特征 {total_visual_words} 个。']
    doc += ['', '所有状态均为模型估计，异常列表需人工复核；无可靠时间的官方词没有跨模态配对。',
            '固定长度词序列长度为 80，原始长度和填充掩码保存在每条 NPZ 中。',
            '音频和视觉同时保留原生时间轴与一秒时间格。',
            f'有人声但无检测人脸：{int((table.speech_present.astype(bool)&~table.face_present.astype(bool)).sum())} 条。',
            f'无人声但有人脸：{int((~table.speech_present.astype(bool)&table.face_present.astype(bool)).sum())} 条。',
            f'人声与人脸均缺失：{int((~table.speech_present.astype(bool)&~table.face_present.astype(bool)).sum())} 条。',
            f'有人声且官方文本未达匹配阈值：{int((table.text_audio_status=="mismatch_candidate").sum())} 条。',
            f'高置信度非英语语音且官方为英语译文：{int(translated.sum())} 条；跨语言特征仅在句段/片段级对应。',
            f'语言识别低于 0.80、需要人工核听：{language_review_count} 条。',
            '', '图中的文本匹配状态由自动转写估计；低于阈值的样本只能定为“待核对”，不可直接宣称标签错误。',
            '', '## 可用于论文的方法表述', '',
            '以官方词序列为文本主轴，用 BERT 的 WordPiece 隐层向量按词求均值得到 768 维词特征；语音用 eGeMAPSv02 低层描述符得到逐帧 25 维声学特征；视觉按 2 fps 检测人脸并提取 52 维表情 blendshape。',
            'Silero VAD 标记真人语音时间段；只有与官方词同形、WhisperX 对齐分数至少 0.1 且样本词覆盖率至少 0.55 的词，才接受其真实发音区间。明显重复的 ASR 结果单列待复核。',
            '对有效词区间 [s_i,e_i)，声学帧按是否落在区间内求均值，视觉帧按相邻采样支持区间与该词的交叠长度加权求均值。人声不足或目标人脸不确定时，特征位填零且独立掩码为 0；填零不是模态存在的证据。',
            '多语言 Whisper 先识别源语种；高置信度非英语样本由多语言模型生成源语转写与英语机器翻译候选，在原音视频时间轴上按翻译片段聚合音频/人脸特征。官方英语文本以整句/片段为单位关联，英语词级时间保持 NaN，不能将译文单词强配到外语发音。自动翻译只帮助定位，语义关系仍列入人工复核。',
            '保留所有原生时间点、官方词原始长度、定长 80 词序列与填充掩码，允许逐样本回溯到原始 MP4 与标签。',
            '', '## 图表', '']
    if alignment:
        aligned_words=sum(case['accepted_words'] for case in alignment['cases'])
        doc += [f'图1a–c 选取负向、中性、正向各一条，共 {aligned_words} 个官方词均有可靠语音时间、语音特征和人脸特征；三个案例的标签均是整段标签，不是逐词情感值。',
                '同一时间轴展示原视频帧、原声波形、逐词区间及文本/声学/表情特征；逐词源素材帧号、音频采样点及特征行号见 `../case_assets/<sample_id>/word_mapping.csv`。',
                '组合图：`../figures/alignment/fig01_three_typical_cases.pdf`；交互试听页：`alignment_review.html`；验证表：`../tables/05_typical_case_validation.csv`。',
                '彩色特征热图逐通道仅在当前片段内标准化用于显示，计算用原始值仍保存在 NPZ；图中时间戳由自动模型估计，需以原声人工复核。', '']
    if crosslingual:
        doc += ['图6 展示希伯来语原声与官方英语译文的片段级候选对应。蓝色为译文内容词支持的片段，灰色为未关联语音；图中三模态特征只在片段级聚合。',
                '正文图：`../figures/fig06_crosslingual_hebrew_translation.pdf`；源语 ASR 与两版英文对照放在补充图 `../figures/fig06b_crosslingual_text_evidence.pdf`。三条样本的原声核验页：`crosslingual_review.html`；验证表：`../tables/09_crosslingual_case_validation.csv`。',
                '机器译文会出错，内容词交集不能证明语义等价；这三条仍留在人工复核清单内。', '']
    captions={'fig01':'五种证据状态的样本数',
              'fig02':'100 条样本的质量事件热图',
              'fig03':'词级语音与视觉有效覆盖率',
              'fig04':'官方文本与原声不符的合并案例图',
              'fig05':'标准 (a)–(d) 原视频画面案例',
              'fig06b':'跨语言源语转写与译文对照（补充图）',
              'fig06':'跨语言原声与英语译文的片段级对应',
              'fig07':'四种异常模态时间轴',
              'fig_case':'典型语音—人脸—词时间轴'}
    for figure in sorted(out.glob('*.pdf')):
        caption=next((value for key,value in captions.items() if figure.name.startswith(key)), '案例')
        doc.append(f'- `{figure.name}`：{caption}。')
    (config.OUT/'paper').mkdir(exist_ok=True)
    (config.OUT/'paper/q1_results.md').write_text('\n'.join(doc)+'\n')
