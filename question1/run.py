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

from question1.pipeline import run
from question1.analysis.quality import create_reports
from question1.alignment.crosslingual import upgrade_existing_outputs

def main():
    p=argparse.ArgumentParser(description='Question 1 multimodal feature/alignment pipeline')
    p.add_argument('--ids',default=''); p.add_argument('--no-resume',action='store_true'); p.add_argument('--reports-only',action='store_true'); p.add_argument('--crosslingual-only',action='store_true')
    a=p.parse_args()
    if a.reports_only and a.crosslingual_only: p.error('mutually exclusive')
    if not a.reports_only and not a.crosslingual_only: run(set(filter(None,a.ids.split(','))),resume=not a.no_resume)
    if a.crosslingual_only or (not a.reports_only and not a.ids): upgrade_existing_outputs()
    create_reports()
if __name__=='__main__': main()
