# ============================================================
# inference.py
# Model loading and prediction module
# ============================================================

# ------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------
import json

import joblib
import numpy as np

from backend.config import (
    MODEL_PATH,
    LABEL_PATH
)

from backend.preprocessing import (
    clean_tweet,
    preprocess_for_inference,
    sentiment_analyzer,
    sentiment_category
)


# ------------------------------------------------------------
# Verify Required Files
# ------------------------------------------------------------
# Ensure the trained model and label encoder
# are available before loading.

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

if not LABEL_PATH.exists():
    raise FileNotFoundError(
        f"Label encoder not found: {LABEL_PATH}"
    )


# ------------------------------------------------------------
# Load Trained Model
# ------------------------------------------------------------
# Load the trained classifier and label mapping once
# when the application starts.

model = joblib.load(MODEL_PATH)

with LABEL_PATH.open(
    "r",
    encoding="utf-8"
) as file:
    label_encoder = json.load(file)


# ============================================================
# Prediction Function
# ============================================================

def predict_sentence(text: str) -> dict:
    """
    Predict whether the supplied text contains
    offensive language.

    Parameters
    ----------
    text : str
        Input text supplied by the user.

    Returns
    -------
    dict
        Prediction result together with preprocessing
        and sentiment analysis information.
    """

    # ------------------------------------------
    # Clean Input Text
    # ------------------------------------------

    cleaned_text = clean_tweet(text)

    # ------------------------------------------
    # Generate Features
    # ------------------------------------------

    features = preprocess_for_inference(text)

    # ------------------------------------------
    # Predict Class
    # ------------------------------------------

    predicted_class = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    class_positions = np.where(
        model.classes_ == predicted_class
    )[0]

    if len(class_positions) == 0:
        raise RuntimeError(
            "Predicted class was not found in the model."
        )

    predicted_index = int(class_positions[0])

    confidence = float(
        probabilities[predicted_index]
    )

    # ------------------------------------------
    # Decode Prediction Label
    # ------------------------------------------

    label_key = str(predicted_class)

    if label_key not in label_encoder:
        raise RuntimeError(
            f"No label mapping found for class {label_key}."
        )

    predicted_label = label_encoder[label_key]

    # ------------------------------------------
    # Sentiment Analysis
    # ------------------------------------------

    sentiment_scores = sentiment_analyzer.polarity_scores(
        cleaned_text
    )

    compound_score = float(
        sentiment_scores["compound"]
    )

    sentiment_label = sentiment_category(
        compound_score
    )

    # ------------------------------------------
    # Return API Response
    # ------------------------------------------

    return {
        "prediction": predicted_label,
        "confidence": round(confidence, 4),

        "cleaned_text": cleaned_text,

        "sentiment_score": round(
            compound_score,
            4
        ),

        "sentiment_category": sentiment_label,

        "positive_score": round(
            float(sentiment_scores["pos"]),
            4
        ),

        "neutral_score": round(
            float(sentiment_scores["neu"]),
            4
        ),

        "negative_score": round(
            float(sentiment_scores["neg"]),
            4
        )
    }


# ============================================================
# Model Status
# ============================================================

def model_is_ready() -> bool:
    """
    Check whether the trained model has been
    successfully loaded.

    Returns
    -------
    bool
        True if the model is ready for inference.
    """

    return (
        model is not None
        and hasattr(model, "predict")
        and hasattr(model, "predict_proba")
    )