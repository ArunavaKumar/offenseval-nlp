import sys
import os

import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
from preprocessing import preprocess_for_inference

# --------------------------------------------------
# Model Path
# --------------------------------------------------
MODEL_PATH = "model/final_model_tuned distilbert.joblib"

# --------------------------------------------------
# Test Sentences
# 6 Not Offensive + 6 Offensive
# (2 short, 2 medium, 2 long per class)
# --------------------------------------------------

TEST_SENTENCES = [
    # -----------------------------
    # NOT OFFENSIVE
    # -----------------------------

    # Short (5–10 words)
    "I respectfully disagree with your opinion",
    "This decision could have been handled better",

    # Medium (15–25 words)
    "The service quality has declined recently, but I believe the team is making genuine efforts to improve the situation",
    "I am frustrated by the delays, yet I understand that unexpected challenges can affect even well planned projects",

    # Long (25–30 words)
    "While the recent management decisions were disappointing, I acknowledge the complexity of the situation and hope future strategies will better address customer expectations",
    "The communication could have been clearer, but I appreciate the transparency shown by the organization in acknowledging its shortcomings",

    # -----------------------------
    # OFFENSIVE
    # -----------------------------

    # Short (5–10 words)
    "You are an absolute idiot",
    "This is the dumbest thing ever",

    # Medium (15–25 words)
    "You clearly have no idea what you are talking about and your comments only show how ignorant you really are",
    "The way you handled this task proves your complete lack of competence and basic understanding",

    # Long (25–30 words)
    "You are a useless fool who keeps making stupid mistakes and everyone is tired of cleaning up the mess caused by your incompetence",
    "Your behavior is annoying and insulting and shows a total absence of professionalism intelligence and respect for others"
]

# --------------------------------------------------
# Tests
# --------------------------------------------------

def test_model_loads():
    """Ensure the trained model loads correctly."""
    print("\n[TEST] Loading trained model...")
    model = joblib.load(MODEL_PATH)
    assert model is not None
    print("[PASS] Model loaded successfully.")


def test_preprocessing_shape():
    """Ensure preprocessing returns correct feature shape."""
    print("\n[TEST] Running preprocessing pipeline...")
    X = preprocess_for_inference(TEST_SENTENCES[0])

    assert isinstance(X, np.ndarray)
    assert X.shape == (1, 769)

    print(f"[PASS] Preprocessing output shape verified: {X.shape}")


def test_prediction_runs_for_all_sentences():
    """Ensure prediction works for all test sentences."""
    print("\n[TEST] Running inference on all test sentences...")
    model = joblib.load(MODEL_PATH)

    for idx, sentence in enumerate(TEST_SENTENCES, start=1):
        X = preprocess_for_inference(sentence)
        probs = model.predict_proba(X)

        assert probs.shape == (1, 2)
        assert np.isclose(probs.sum(), 1.0, atol=1e-6)

        print(f"[PASS] Inference successful for sentence {idx}")

    print("[PASS] All test sentences processed correctly.")