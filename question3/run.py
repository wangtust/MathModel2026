#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _call_pipeline(modname, argv):
    mod = __import__(modname, fromlist=['main'])
    old = sys.argv
    sys.argv = [old[0], *argv]
    try:
        mod.main()
    finally:
        sys.argv = old


def _paper_figures() -> None:
    # Publication figures intentionally use the tightened renderer.  The legacy
    # main_text/report modules remain available as audit/supplementary material.
    from question3.visualization.publication import main as paper_main
    from question3.visualization.supplement import run as supplement_run

    paper_main()
    supplement_run()


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else 'pipeline'
    args = sys.argv[2:]
    table = {
        'pipeline': 'question3.pipeline',
        'ig-check': 'question3.evaluation.ig_check',
        'fidelity': 'question3.evaluation.fidelity',
        'sensitivity': 'question3.evaluation.sensitivity',
        'alignment': 'question3.evaluation.alignment_quality',
        'evidence': 'question3.evidence.mapper',
        'verify': 'question3.analysis.verification',
        'figures': 'question3.visualization.report',
        'paper-figures': 'question3.visualization.publication',
    }
    if stage not in table:
        raise SystemExit('Stages: ' + ', '.join(table))

    if stage == 'pipeline':
        _call_pipeline(table[stage], args)
        return

    # Compact journal figures read the already-generated Q3 tables directly and
    # deliberately do not rerun training, attribution or model selection.
    if stage == 'paper-figures':
        if args:
            raise SystemExit('paper-figures takes no extra arguments; it uses outputs/q3')
        _paper_figures()
        return

    from common.paths import Q3_OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Q3_OUTPUT_DIR)
    output = parser.parse_args(args).output

    if stage == 'figures':
        # Preserve the full legacy evidence package required by verification:
        # diagnostic figures + case cards + combined supplement PDF.
        legacy = __import__(table[stage], fromlist=['run'])
        legacy.run(output)

        # The root runner uses the canonical outputs/q3 location. Generate the
        # compact publication layer afterwards without changing verification
        # expectations for the legacy package.
        if output.resolve() == Q3_OUTPUT_DIR.resolve():
            _paper_figures()
        else:
            print(
                f"Skipped compact paper figures for non-default --output={output}; "
                "run `python question3/run.py paper-figures` for outputs/q3.",
                flush=True,
            )
        return

    mod = __import__(table[stage], fromlist=['run', 'verify'])
    (mod.verify if stage == 'verify' else mod.run)(output)


if __name__ == '__main__':
    main()
