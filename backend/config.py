from pathlib import Path

# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "model"

MODEL_PATH = MODEL_DIR / "final_model_tuned distilbert.joblib"
LABEL_PATH = MODEL_DIR / "label_encoder.json"

# ============================================================
# API Settings
# ============================================================

API_TITLE = "OffensEval NLP API"
API_DESCRIPTION = "REST API for Offensive Language Detection"
API_VERSION = "1.0.0"

# ============================================================
# Model Settings
# ============================================================

EMBEDDING_MODEL_NAME = "distilbert-base-nli-stsb-mean-tokens"

EXPECTED_FEATURES = 769

MAX_INPUT_LENGTH = 5000