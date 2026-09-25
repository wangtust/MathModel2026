from pathlib import Path
import sys
from common.paths import THIRD_PARTY_ROOT
for p in (THIRD_PARTY_ROOT / "MMSA-FET" / "src", THIRD_PARTY_ROOT / "whisperX", THIRD_PARTY_ROOT / "silero-vad" / "src", THIRD_PARTY_ROOT / "CMU-MultimodalSDK"):
    if p.exists() and str(p) not in sys.path: sys.path.insert(0, str(p))
