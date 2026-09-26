"""Organize the complete legacy Q3 evidence package as supplementary material.

The files under outputs/q3/figures remain untouched because verification.py expects
that layout.  This module only copies publication-facing supplementary artifacts
into outputs/q3/paper_figures/supplementary so the main-text and audit layers are
clearly separated.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from common.paths import Q3_OUTPUT_DIR


SOURCE = Q3_OUTPUT_DIR / "figures"
DEST = Q3_OUTPUT_DIR / "paper_figures" / "supplementary"

LEGACY_FIGURES = (
    "fig01_performance",
    "fig02_fidelity",
    "fig03_modality_contributions",
    "fig04_local_importance",
    "fig05_alignment_quality",
    "fig06_timed_evidence",
    "fig07_model_sensitivity",
    "fig08_validation_variant_fidelity",
    "fig09_gate_response_rank",
)


def _copy_pair(name: str, destination: Path) -> list[str]:
    copied = []
    for suffix in ("pdf", "png"):
        src = SOURCE / f"{name}.{suffix}"
        if not src.exists():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        dst = destination / src.name
        shutil.copy2(src, dst)
        copied.append(str(dst.relative_to(Q3_OUTPUT_DIR)))
    return copied


def run() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    diagnostics = DEST / "diagnostics"
    for name in LEGACY_FIGURES:
        copied.extend(_copy_pair(name, diagnostics))

    cards = DEST / "case_cards"
    for i in range(1, 21):
        copied.extend(_copy_pair(f"case_{i:02d}", cards))

    bundle = Q3_OUTPUT_DIR / "paper" / "attachment4_explanation_cards.pdf"
    if bundle.exists():
        dst = DEST / "attachment4_explanation_cards.pdf"
        shutil.copy2(bundle, dst)
        copied.append(str(dst.relative_to(Q3_OUTPUT_DIR)))

    notes = {
        "purpose": "supplementary evidence package; main text uses outputs/q3/paper_figures/fig01_q3_* through fig05_q3_*",
        "diagnostic_roles": {
            "fig01_performance": "legacy combined performance/fidelity overview; prediction performance is a main-text table now",
            "fig02_fidelity": "full fidelity display including top-minus-low control",
            "fig03_modality_contributions": "legacy deletion response plus internal gate diagnostic; gate is not interpreted as causal contribution",
            "fig04_local_importance": "all 20 samples x 3 modalities local IG maps; row-wise normalization only",
            "fig05_alignment_quality": "legacy per-sample alignment audit bars",
            "fig06_timed_evidence": "legacy Sample-19 source-time evidence",
            "fig07_model_sensitivity": "unlabeled Attachment-4 model-choice sensitivity only",
            "fig08_validation_variant_fidelity": "post-selection frozen-variant explanation diagnostic",
            "fig09_gate_response_rank": "sanity check showing gate rank is not a modality-contribution percentage",
            "case_01 ... case_20": "full explanation cards for all Attachment-4 samples",
        },
        "copied_files": copied,
    }
    (DEST / "supplement_manifest.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Organized {len(copied)} Q3 supplementary artifacts under {DEST}", flush=True)


if __name__ == "__main__":
    run()
