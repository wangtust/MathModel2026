#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ENV_LIB = Path(sys.executable).resolve().parents[1] / 'lib/python3.11/site-packages/nvidia'
libs = [str(ENV_LIB/'cudnn/lib'), str(ENV_LIB/'cublas/lib')]
if __name__ == '__main__' and os.environ.get('Q1_CUDA_PATH_READY') != '1':
    env=os.environ.copy(); env['LD_LIBRARY_PATH']=':'.join(libs+[env.get('LD_LIBRARY_PATH','')]); env['Q1_CUDA_PATH_READY']='1'
    nltk = ROOT / 'assets' / 'nltk_data'
    if nltk.exists(): env['NLTK_DATA']=str(nltk)
    os.execvpe(sys.executable,[sys.executable,__file__,*sys.argv[1:]],env)

import pandas as pd

from question1 import config
from question1.pipeline import run
from question1.analysis.quality import create_reports
from question1.alignment.crosslingual import upgrade_existing_outputs
from question1.visualization.overview import create_overview_figures


MAIN_FIGURE_MARKER = '## 正文核心可视化（期刊版）'


def _write_paper_feature_summary(table: pd.DataFrame) -> None:
    """Write the explicit 100-sample × 3-modality table required by the contest.

    The processing summary is intentionally wide and audit-oriented.  This long
    table provides the exact paper-facing fields requested by the statement:
    sample id, modality, source duration, feature dimension and alignment
    granularity, plus sequence length and valid fraction for traceability.
    """
    specs = (
        ('Text', 768, None),
        ('Audio', 25, 'word_audio_valid_fraction'),
        ('Vision', 52, 'word_vision_valid_fraction'),
    )
    rows = []
    for _, sample in table.iterrows():
        raw_word_count = sample.get('word_count', 0)
        word_count = 0 if pd.isna(raw_word_count) else int(raw_word_count)
        raw_duration = sample.get('duration', 0.0)
        duration = 0.0 if pd.isna(raw_duration) else float(raw_duration)
        granularity = sample.get('alignment_granularity', 'word')
        granularity = 'word' if pd.isna(granularity) else str(granularity)
        text_present = sample.get('official_text_present', True)
        text_present = False if pd.isna(text_present) else bool(text_present)

        for modality, dim, fraction_col in specs:
            if fraction_col is None:
                valid_fraction = 1.0 if text_present else 0.0
            else:
                value = sample.get(fraction_col, 0.0)
                valid_fraction = 0.0 if pd.isna(value) else float(value)
            feature_file = sample.get('feature_file', '')
            feature_file = '' if pd.isna(feature_file) else str(feature_file)
            rows.append({
                'sample_id': sample['sample_id'],
                'modality': modality,
                'source_duration_sec': duration,
                'feature_dim': dim,
                'alignment_granularity': granularity,
                'aligned_sequence_length': word_count,
                'valid_fraction': valid_fraction,
                'feature_file': feature_file,
            })
    paper = pd.DataFrame(rows)
    expected = len(table) * 3
    if len(paper) != expected:
        raise RuntimeError(f'Q1 paper feature summary expected {expected} rows, found {len(paper)}')
    paper.to_csv(config.OUT / 'tables/11_paper_feature_summary.csv',
                 index=False, encoding='utf-8-sig')


def _create_main_text_figures():
    """Build the consolidated Q1 journal figures from already-audited outputs."""
    summary = config.OUT / 'tables/02_sample_summary.csv'
    if not summary.exists():
        return
    table = pd.read_csv(summary)
    if table.empty:
        return
    _write_paper_feature_summary(table)
    create_overview_figures(table, config.OUT / 'figures')

    report = config.OUT / 'paper/q1_results.md'
    if report.exists():
        text = report.read_text(encoding='utf-8')
        if MAIN_FIGURE_MARKER not in text:
            text += (
                '\n\n' + MAIN_FIGURE_MARKER + '\n\n'
                '- `../figures/fig01_q1_framework.pdf`：三模态特征提取、词级时间锚定、聚合、mask 与审计输出的总体框架。\n'
                '- `../figures/fig02_q1_quality_overview.pdf`：100 条样本的模态可用性、质量事件矩阵与 Audio–Vision 逐词覆盖三联总览。\n'
                '- `../figures/fig03_q1_representative_alignment.pdf`：正文典型样本；原视频帧、波形、官方词区间与三类特征摘要共享同一时间轴。\n'
                '- `../figures/fig04_q1_quality_control.pdf`：无目标脸、无真人语音、多人脸歧义、文本/原声不符四类复杂场景的源画面与时间状态。\n\n'
                '- `../tables/11_paper_feature_summary.csv`：赛题要求的 100 条样本 × Text/Audio/Vision 长表，显式给出原始时长、特征维度、对齐粒度、序列长度和有效比例。\n\n'
                '原 `alignment/fig01a–c` 的 768/25/52 维完整热图继续保留为审计/补充材料；跨语言图继续作为特殊异常补充证据。\n'
            )
            report.write_text(text, encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description='Question 1 multimodal feature/alignment pipeline')
    p.add_argument('--ids',default=''); p.add_argument('--no-resume',action='store_true'); p.add_argument('--reports-only',action='store_true'); p.add_argument('--crosslingual-only',action='store_true')
    a=p.parse_args()
    if a.reports_only and a.crosslingual_only: p.error('mutually exclusive')
    if not a.reports_only and not a.crosslingual_only: run(set(filter(None,a.ids.split(','))),resume=not a.no_resume)
    if a.crosslingual_only or (not a.reports_only and not a.ids): upgrade_existing_outputs()
    create_reports()
    _create_main_text_figures()
if __name__=='__main__': main()
