import sys
from common.paths import THIRD_PARTY_ROOT
p = THIRD_PARTY_ROOT / "MMSA" / "src"
if p.exists() and str(p) not in sys.path: sys.path.insert(0, str(p))
