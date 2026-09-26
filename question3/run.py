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
        'paper-figures': 'question3.visualization.main_text',
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
        mod = __import__(table[stage], fromlist=['main'])
        mod.main()
        return

    from common.paths import Q3_OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Q3_OUTPUT_DIR)
    output = parser.parse_args(args).output
    mod = __import__(table[stage], fromlist=['run', 'verify'])
    (mod.verify if stage == 'verify' else mod.run)(output)


if __name__ == '__main__':
    main()
