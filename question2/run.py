#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _dispatch(module_main, argv):
    old=sys.argv; sys.argv=[old[0],*argv]
    try: module_main()
    finally: sys.argv=old

def _paper_figures():
    from question2.visualization.architecture import paper_framework
    from question2.visualization.main_text import main as paper_main
    from question2.visualization.supplement import main as supplement_main
    paper_framework()
    paper_main()
    supplement_main()

def main():
    if len(sys.argv)<2:
        raise SystemExit('Usage: python question2/run.py <train|predict|experiment|error-analysis|analyze|visualize|paper-figures|submit> ...')
    cmd=sys.argv[1]; args=sys.argv[2:]
    if cmd in {'train','predict'}:
        model='mrrf'
        if '--model' in args:
            i=args.index('--model'); model=args[i+1]; args=args[:i]+args[i+2:]
        if model=='mrrf':
            from question2.training.trainer import main as m; _dispatch(m,[cmd,*args]); return
        if model=='mult':
            from question2.training.mult_trainer import main as m; _dispatch(m,[cmd,*args]); return
        from question2.experiments.baselines import main as m
        mapped = {'gru':'gru','concat_mlp':'concat_mlp','tfr':'tfr_feature','mult_no_aug':'mult_no_aug'}.get(model,model)
        extra=['--model',mapped,*args]
        if cmd=='predict': extra.append('--predict-only')
        _dispatch(m,extra); return
    if cmd=='experiment':
        if '--name' not in args: raise SystemExit('experiment requires --name')
        i=args.index('--name'); name=args[i+1]; rest=args[:i]+args[i+2:]
        modules={
            'ablation':'question2.experiments.ablation',
            'robustness':'question2.experiments.robustness',
            'missing_duration':
            'question2.experiments.missing_duration',
            'repeats':'question2.experiments.repeats',
            'tuning':'question2.experiments.tuning',
            'baselines':'question2.experiments.baselines',
            'dual_reconstruction':'question2.experiments.dual_reconstruction'
            }
        mod=__import__(modules[name],fromlist=['main']); _dispatch(mod.main,rest); return
    if cmd=='analyze':
        from question2.analysis.model_comparison import main as m; _dispatch(m,args); return
    if cmd in {'error-analysis','error_analysis'}:
        from question2.analysis.error_analysis import main as m; _dispatch(m,args); return
    if cmd=='visualize':
        from question2.visualization.report import main as m
        _dispatch(m,args)
        # On completed runs, refresh the compact paper set too. During the first
        # pre-selection visualization pass this is intentionally skipped.
        from common.paths import Q2_OUTPUT_DIR
        if (Q2_OUTPUT_DIR / 'analysis' / 'selected_model.json').exists():
            _paper_figures()
        return
    if cmd in {'paper-figures','paper_figures'}:
        if args:
            raise SystemExit('paper-figures does not accept additional arguments')
        _paper_figures(); return
    if cmd=='submit':
        from question2.analysis.model_selection import main as m
        _dispatch(m,args)
        # Selection is now fixed, so all main-text figures can safely label and
        # visualize the validation-selected candidate without touching training.
        _paper_figures()
        return
    raise SystemExit(f'Unknown command: {cmd}')
if __name__=='__main__': main()
