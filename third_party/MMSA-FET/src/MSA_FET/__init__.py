from .single import FeatureExtractionTool
# The dataset workflow imports ctc_segmentation eagerly.  Keep the single-file
# extractor usable when only its native text/audio features are requested.
# Callers needing dataset processing can import MSA_FET.dataset explicitly.
from .utils import *
