# ============================================================
# preprocessing.py
# Centralized preprocessing and feature extraction module
# ============================================================

# ------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------
import re

import nltk
import numpy as np
import torch

from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ------------------------------------------------------------
# Download Required NLTK Resources
# ------------------------------------------------------------
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ------------------------------------------------------------
# Device Configuration
# ------------------------------------------------------------
# Automatically select the best available device.
# Priority:
#   1. CUDA GPU
#   2. Apple Silicon (MPS)
#   3. CPU

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# ------------------------------------------------------------
# Global Configuration
# ------------------------------------------------------------

EMBEDDING_MODEL_NAME = "distilbert-base-nli-stsb-mean-tokens"

# 768 embedding features + 1 sentiment feature
EXPECTED_FEATURES = 769


# ------------------------------------------------------------
# Load Models (Loaded Once)
# ------------------------------------------------------------

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    device=DEVICE
)

sentiment_analyzer = SentimentIntensityAnalyzer()


# ------------------------------------------------------------
# Additional Stop Words
# ------------------------------------------------------------

ADDITIONAL_STOP_WORDS = {"amp"}


# ============================================================
# Text Cleaning
# ============================================================

def clean_tweet(text: str) -> str:
    """
    Clean and normalize raw input text before feature extraction.

    Parameters
    ----------
    text : str
        Raw user input.

    Returns
    -------
    str
        Cleaned text suitable for embedding generation.
    """

    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    # Remove retweet token
    text = re.sub(r"\bRT\b[\s]+", "", text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol
    text = re.sub(r"#", "", text)

    # Expand common abbreviations
    text = re.sub(r"\bu\b", "you", text)
    text = re.sub(r"\bim\b", "i am", text)

    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    # Remove quotation marks
    text = re.sub(r"[\"“”‘’]", "", text)

    # Remove digits
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    # Tokenize text
    tokens = word_tokenize(text)

    # Keep only valid alphabetic tokens
    tokens = [
        token
        for token in tokens
        if token.isalpha()
        and not re.search(r"(.)\1{2,}", token)
        and token not in ADDITIONAL_STOP_WORDS
    ]

    return " ".join(tokens)


# ============================================================
# Sentiment Analysis
# ============================================================

def extract_sentiment(text: str) -> float:
    """
    Calculate the VADER compound sentiment score.

    Parameters
    ----------
    text : str

    Returns
    -------
    float
        Compound sentiment score.
    """

    if not isinstance(text, str):
        raise ValueError("Sentiment input must be a string.")

    return float(
        sentiment_analyzer.polarity_scores(text)["compound"]
    )


def sentiment_category(score: float) -> str:
    """
    Convert the compound sentiment score into
    Positive, Neutral or Negative.
    """

    if score >= 0.05:
        return "Positive"

    if score <= -0.05:
        return "Negative"

    return "Neutral"


def extract_sentiment_with_label(text: str) -> dict:
    """
    Return both sentiment score and category.
    """

    score = extract_sentiment(text)

    return {
        "score": score,
        "category": sentiment_category(score)
    }


# ============================================================
# Feature Extraction
# ============================================================

def preprocess_for_inference(raw_text: str) -> np.ndarray:
    """
    Convert raw user text into the feature vector expected
    by the trained classifier.

    Output
    ------
    Shape: (1, 769)

        • 768 DistilBERT embedding features
        • 1 VADER sentiment feature
    """

    # Clean input text
    cleaned_text = clean_tweet(raw_text)

    # Validate cleaned text
    if len(cleaned_text.split()) < 2:
        raise ValueError(
            "Input text is too short after preprocessing."
        )

    # Generate sentence embedding
    embedding = embedding_model.encode(
        [cleaned_text],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    # Calculate sentiment feature
    sentiment_score = extract_sentiment(cleaned_text)

    sentiment_feature = np.array(
        [[sentiment_score]],
        dtype=np.float32
    )

    # Concatenate embedding and sentiment
    features = np.hstack(
        [embedding, sentiment_feature]
    ).astype(np.float32)

    # Validate feature dimensions
    if features.shape != (1, EXPECTED_FEATURES):
        raise RuntimeError(
            f"Unexpected feature shape: {features.shape}. "
            f"Expected (1, {EXPECTED_FEATURES})."
        )

    return features