# ============================================================
# preprocessing.py
# Centralized preprocessing & feature extraction for inference
# ============================================================

# -----------------------------
# Imports
# -----------------------------
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
import torch

# -----------------------------
# NLTK Setup
# -----------------------------
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# -----------------------------
# Global Objects (Loaded Once)
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDING_MODEL_NAME = "distilbert-base-nli-stsb-mean-tokens"
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    device=DEVICE
)

sentiment_analyzer = SentimentIntensityAnalyzer()

ADDITIONAL_STOP_WORDS = {"amp"}

# ============================================================
# Text Cleaning
# ============================================================

def clean_tweet(text: str) -> str:
    """
    Cleans and normalizes tweet text for model inference.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"\bRT\b[\s]+", "", text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\bu\b", "you", text)
    text = re.sub(r"\bim\b", "i am", text)

    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"[\"“”‘’]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    tokens = [
        t for t in tokens
        if t.isalpha()
        and not re.search(r"(.)\1{2,}", t)
        and t not in ADDITIONAL_STOP_WORDS
    ]

    return " ".join(tokens)

# ============================================================
# Sentiment Extraction
# ============================================================

def extract_sentiment(text: str) -> float:
    """
    Returns VADER compound sentiment score.
    """
    return sentiment_analyzer.polarity_scores(text)["compound"]

def sentiment_category(score: float) -> str:
    """
    Maps compound sentiment score to category.
    """
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def extract_sentiment_with_label(text: str) -> dict:
    """
    Returns sentiment score and category (for UI transparency).
    """
    score = extract_sentiment(text)
    category = sentiment_category(score)

    return {
        "score": score,
        "category": category
    }

# ============================================================
# Final Inference Preprocessing
# ============================================================

def preprocess_for_inference(raw_text: str) -> np.ndarray:
    """
    Converts raw user input into model-ready feature vector.

    Output shape:
    (1, 769) -> [768 DistilBERT embedding + 1 sentiment feature]
    """

    clean_text = clean_tweet(raw_text)

    if len(clean_text.split()) < 2:
        raise ValueError("Input text too short after preprocessing.")

    # Generate embedding
    embedding = embedding_model.encode(
        [clean_text],
        convert_to_numpy=True
    )

    # Extract sentiment feature
    sentiment_score = extract_sentiment(clean_text)
    sentiment_feature = np.array([[sentiment_score]])

    # Concatenate embedding + sentiment
    final_features = np.hstack([embedding, sentiment_feature])

    return final_features
