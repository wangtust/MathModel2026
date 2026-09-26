#!/usr/bin/env python3
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _dispatch(module_main, argv):
    old=sys.argv; sys.argv=[old[0],*argv]
    try: module_main()
    finally: sys.argv=old


def _selected_model_dir() -> tuple[Path, str, int]:
    from common.paths import Q2_OUTPUT_DIR
    selected_path = Q2_OUTPUT_DIR / 'analysis' / 'selected_model.json'
    if not selected_path.exists():
        raise FileNotFoundError(
            f'Missing {selected_path}; run Q2 model selection before paper figures.'
        )
    data = json.loads(selected_path.read_text(encoding='utf-8'))
    selected = data.get('selected_variant')
    seed = int(data.get('prediction_checkpoint_seed', 2026))
    mapping = {
        'mrrf_no_recon': 'no_recon',
        'mrrf_full': 'full',
        'mrrf_dual_recon': 'dual_recon',
    }
    if selected not in mapping:
        raise RuntimeError(f'Unsupported selected Q2 variant: {selected!r}')
    variant = mapping[selected]
    model_dir = Q2_OUTPUT_DIR / 'runs' / 'mrrf' / variant / f'seed_{seed}'
    if not (model_dir / 'checkpoint.pt').exists():
        raise FileNotFoundError(f'Selected Q2 checkpoint is missing: {model_dir}')
    return model_dir, variant, seed


def _ensure_selected_position_rate_grid():
    """Keep the main-text position×rate analysis on the final selected Q2 model.

    The full root pipeline creates this grid before validation-based model selection,
    so historical outputs can legitimately be based on MRRF full.  Once selection
    is fixed, paper figures should not silently mix that checkpoint with the final
    no-reconstruction predictor.  Recompute only when the saved grid does not
    already match the selected variant/seed and the complete 90-condition layout.
    """
    from common.paths import Q2_OUTPUT_DIR
    model_dir, variant, seed = _selected_model_dir()
    grid = Q2_OUTPUT_DIR / 'experiments' / 'robustness' / 'position_rate_grid.csv'

    matched = False
    if grid.exists():
        try:
            with grid.open('r', encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            variants = {row.get('checkpoint_variant') for row in rows}
            seeds = {int(row.get('checkpoint_seed', -1)) for row in rows}
            rates = {round(float(row.get('missing_rate', -1)), 6) for row in rows}
            scenarios = {row.get('scenario') for row in rows}
            positions = {row.get('position') for row in rows}
            matched = (
                len(rows) == 90
                and variants == {variant}
                and seeds == {seed}
                and rates == {.1, .2, .3, .4, .5}
                and scenarios == {'text','audio','vision','text+audio','text+vision','audio+vision'}
                and positions == {'beginning','middle','end'}
            )
        except Exception:
            matched = False

    if matched:
        print(f'Q2 paper grid already matches selected {variant}/seed_{seed}; skip refresh.', flush=True)
        return

    print(
        f'Q2 paper grid is missing/stale; recomputing 90 conditions with selected '
        f'{variant}/seed_{seed}.', flush=True
    )
    from question2.experiments.robustness import main as robustness_main
    _dispatch(robustness_main, ['--model-dir', str(model_dir)])


def _paper_figures():
    # The position×rate panel is the only compact Q2 figure that depends on a
    # pre-selection experiment file.  Ensure it uses the validation-selected
    # model before any publication-facing rendering.
    _ensure_selected_position_rate_grid()
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
        # Selection is now fixed.  _paper_figures first reconciles the
        # position×rate grid with that selected checkpoint, then renders all
        # publication-facing figures without retraining.
        _paper_figures()
        return
    raise SystemExit(f'Unknown command: {cmd}')
if __name__=='__main__': main()
