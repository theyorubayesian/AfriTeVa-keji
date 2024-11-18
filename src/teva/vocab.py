import os

import seqio

DEFAULT_SPM_PATH = os.getenv("DEFAULT_SPM_PATH")
DEFAULT_VOCAB = seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH)
