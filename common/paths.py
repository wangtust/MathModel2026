"""Central path registry for the whole project.

By default the project lives beside ``../data``. Override with MM_DATA_ROOT or
MM_OUTPUT_ROOT when running on another machine/server.
"""
from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("MM_DATA_ROOT", PROJECT_ROOT.parent / "data")).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("MM_OUTPUT_ROOT", PROJECT_ROOT / "outputs")).expanduser().resolve()
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"

ATTACHMENT1_DIR = DATA_ROOT / "附件1-数据集原始多模态样本"
ATTACHMENT2_DIR = DATA_ROOT / "附件2-数据集特征文件"
ATTACHMENT3_DIR = DATA_ROOT / "附件3-模态缺失特征样本"
ATTACHMENT4_DIR = DATA_ROOT / "附件4-可解释专项视频样本与特征文件"

PROCESSED_DIR = DATA_ROOT / "processed"
FEATURES_DIR = DATA_ROOT / "features"
CACHE_DIR = DATA_ROOT / "cache"

Q1_OUTPUT_DIR = OUTPUT_ROOT / "q1"
Q2_OUTPUT_DIR = OUTPUT_ROOT / "q2"
Q3_OUTPUT_DIR = OUTPUT_ROOT / "q3"

def ensure_data_workdirs() -> None:
    for path in (PROCESSED_DIR, FEATURES_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)
