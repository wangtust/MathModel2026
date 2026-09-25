from common.paths import ATTACHMENT2_DIR, ATTACHMENT3_DIR, Q2_OUTPUT_DIR
DATA = ATTACHMENT2_DIR / 'aligned_50.pkl'
ATTACHMENT3 = ATTACHMENT3_DIR / '对齐版本'
LABEL_NAMES = ['Negative','Neutral','Positive']
VALID_SCENARIOS = ('clean','audio','vision','audio+vision')


def run_dir(model: str, seed: int = 2026, variant: str | None = None):
    from common.paths import Q2_OUTPUT_DIR
    if model == "mrrf":
        return Q2_OUTPUT_DIR / "runs" / "mrrf" / (variant or "full") / f"seed_{seed}"
    return Q2_OUTPUT_DIR / "runs" / model / f"seed_{seed}"
