# ============================================================
# app.py
# Streamlit frontend for offensive-language detection
# ============================================================

# ------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------

import os

import requests
import streamlit as st


# ------------------------------------------------------------
# API Configuration
# ------------------------------------------------------------

API_BASE_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"


# ------------------------------------------------------------
# Streamlit Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Offensive Language Detection",
    page_icon="🚨",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ------------------------------------------------------------
# Application Header
# ------------------------------------------------------------

st.title("🚨 Offensive Language Detection")

st.write(
    "Transformer-based offensive-language classification using "
    "**Streamlit, FastAPI, DistilBERT embeddings, and sentiment analysis**."
)


# ------------------------------------------------------------
# Backend Status
# ------------------------------------------------------------

def check_backend_status() -> bool:
    """
    Check whether the FastAPI backend is available.
    """

    try:
        response = requests.get(
            HEALTH_ENDPOINT,
            timeout=5
        )

        if response.status_code != 200:
            return False

        health_data = response.json()

        return bool(
            health_data.get("status") == "healthy"
            and health_data.get("model_ready", False)
        )

    except requests.exceptions.RequestException:
        return False


backend_available = check_backend_status()

if backend_available:
    st.success(
        "Backend API and prediction model are available.",
        icon="✅"
    )
else:
    st.warning(
        "Backend API is currently unavailable. "
        "Start the backend before submitting a prediction.",
        icon="⚠️"
    )


# ------------------------------------------------------------
# User Input
# ------------------------------------------------------------

user_text = st.text_area(
    label="Enter text to analyse:",
    height=120,
    max_chars=5000,
    placeholder="Type a tweet, comment, or sentence here..."
)


# ============================================================
# Helper Functions
# ============================================================

def sentiment_bar(
    label: str,
    value: float,
    color: str
) -> None:
    """
    Display a sentiment percentage bar.
    """

    safe_value = max(0.0, min(float(value), 1.0))
    percentage = round(safe_value * 100)

    st.markdown(
        f"""
        <div style="margin-bottom: 12px;">
            <b>{label}</b> — {percentage}%
            <div style="
                width: 100%;
                background-color: #2b2b2b;
                border-radius: 6px;
                overflow: hidden;
                margin-top: 5px;
            ">
                <div style="
                    width: {percentage}%;
                    min-height: 12px;
                    background-color: {color};
                    border-radius: 6px;
                ">
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_error_message(response: requests.Response) -> str:
    """
    Extract a readable error message from an API response.
    """

    try:
        response_data = response.json()

        detail = response_data.get(
            "detail",
            "The API returned an error."
        )

        if isinstance(detail, list):
            messages = []

            for item in detail:
                if isinstance(item, dict):
                    messages.append(
                        str(item.get("msg", item))
                    )
                else:
                    messages.append(str(item))

            return "; ".join(messages)

        return str(detail)

    except (requests.exceptions.JSONDecodeError, ValueError):
        return (
            f"The API request failed with status code "
            f"{response.status_code}."
        )


def display_prediction_explanation(
    predicted_label: str,
    compound_score: float
) -> None:
    """
    Display a brief explanation of the prediction.
    """

    explanation_points = []

    if compound_score <= -0.5:
        explanation_points.append(
            "Strong negative sentiment was detected."
        )

    elif compound_score < -0.05:
        explanation_points.append(
            "Mild negative sentiment was detected."
        )

    elif compound_score >= 0.5:
        explanation_points.append(
            "Strong positive sentiment was detected."
        )

    elif compound_score > 0.05:
        explanation_points.append(
            "Mild positive sentiment was detected."
        )

    else:
        explanation_points.append(
            "Neutral or mixed sentiment was detected."
        )

    if predicted_label.lower() == "offensive":
        explanation_points.append(
            "The classifier identified contextual patterns "
            "associated with offensive language."
        )
    else:
        explanation_points.append(
            "The classifier did not identify strong contextual "
            "patterns associated with offensive language."
        )

    for point in explanation_points:
        st.write(f"• {point}")


# ============================================================
# Prediction Workflow
# ============================================================

analyse_button = st.button(
    "Analyse Text",
    type="primary",
    use_container_width=True,
    disabled=not backend_available
)

if analyse_button:
    text_to_analyse = user_text.strip()

    if not text_to_analyse:
        st.warning(
            "Please enter some text before running the analysis.",
            icon="⚠️"
        )

    else:
        try:
            # Send request
            with st.spinner("Analysing text..."):
                response = requests.post(
                    PREDICT_ENDPOINT,
                    json={"text": text_to_analyse},
                    timeout=60
                )

            # Handle API error
            if response.status_code != 200:
                st.error(
                    get_error_message(response),
                    icon="❌"
                )
                st.stop()

            result = response.json()

            # Read API response
            cleaned_text = result["cleaned_text"]
            predicted_label = result["prediction"]
            confidence = float(result["confidence"])

            compound_score = float(
                result["sentiment_score"]
            )

            sentiment_category = result[
                "sentiment_category"
            ]

            positive_score = float(
                result["positive_score"]
            )

            neutral_score = float(
                result["neutral_score"]
            )

            negative_score = float(
                result["negative_score"]
            )

            # --------------------------------------------
            # Preprocessed Text
            # --------------------------------------------

            st.subheader("🧹 Preprocessed Text")

            if cleaned_text:
                st.code(
                    cleaned_text,
                    language=None,
                    wrap_lines=True
                )
            else:
                st.info(
                    "No meaningful text remained after preprocessing."
                )

            st.divider()

            # --------------------------------------------
            # Sentiment Analysis
            # --------------------------------------------

            st.subheader("📊 Sentiment Analysis")

            sentiment_bar(
                "Positive sentiment",
                positive_score,
                "#2ecc71"
            )

            sentiment_bar(
                "Neutral sentiment",
                neutral_score,
                "#f1c40f"
            )

            sentiment_bar(
                "Negative sentiment",
                negative_score,
                "#e74c3c"
            )

            sentiment_col_1, sentiment_col_2 = st.columns(2)

            with sentiment_col_1:
                st.metric(
                    label="Compound score",
                    value=f"{compound_score:.3f}"
                )

            with sentiment_col_2:
                st.metric(
                    label="Sentiment category",
                    value=sentiment_category
                )

            with st.expander(
                "View sentiment interpretation"
            ):
                st.markdown(
                    """
- **Positive:** compound score ≥ +0.05
- **Neutral:** −0.05 < compound score < +0.05
- **Negative:** compound score ≤ −0.05

The compound sentiment score ranges from −1 to +1.
"""
                )

            st.divider()

            # --------------------------------------------
            # Prediction Result
            # --------------------------------------------

            st.subheader("📌 Prediction Result")

            if predicted_label.lower() == "offensive":
                st.error(
                    f"Prediction: **{predicted_label}**",
                    icon="🚨"
                )
            else:
                st.success(
                    f"Prediction: **{predicted_label}**",
                    icon="✅"
                )

            st.metric(
                label="Prediction confidence",
                value=f"{confidence:.2%}"
            )

            st.caption(
                "Confidence represents the classifier's estimated "
                "probability for the predicted class."
            )

            # --------------------------------------------
            # Prediction Explanation
            # --------------------------------------------

            st.subheader("🧠 Prediction Explanation")

            display_prediction_explanation(
                predicted_label=predicted_label,
                compound_score=compound_score
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Unable to connect to the FastAPI backend. "
                "For local execution, start it with "
                "`uvicorn backend.main:app --reload`. "
                "For Docker, run `docker compose up --build`.",
                icon="❌"
            )

        except requests.exceptions.Timeout:
            st.error(
                "The prediction request timed out. "
                "Please try again.",
                icon="❌"
            )

        except requests.exceptions.RequestException as error:
            st.error(
                f"API request failed: {error}",
                icon="❌"
            )

        except KeyError as error:
            st.error(
                f"The API response is missing the field: {error}",
                icon="❌"
            )

        except (TypeError, ValueError) as error:
            st.error(
                f"The API returned an invalid value: {error}",
                icon="❌"
            )

        except Exception as error:
            st.error(
                f"An unexpected error occurred: {error}",
                icon="❌"
            )


# ============================================================
# Model Information
# ============================================================

st.divider()
st.subheader("📊 Model Information")

st.markdown(
    """
**Application architecture**

- Frontend: Streamlit
- Backend: FastAPI
- Communication: REST API
- Deployment: Docker and Azure Container Apps

**Machine-learning pipeline**

- Embedding model: DistilBERT SentenceTransformer
- Additional feature: VADER compound sentiment
- Classifier: Logistic Regression
- Training dataset: TweetEval offensive-language data

**Validation performance**

- Macro F1-score: approximately 0.72
- ROC-AUC score: approximately 0.81
"""
)


# ============================================================
# Disclaimer
# ============================================================

st.divider()
st.subheader("⚠️ Disclaimer")

st.markdown(
    """
This application predicts offensive language using patterns
learned from social-media data.

**Limitations**

- Sarcasm and humour may not be interpreted correctly.
- Cultural context and implicit meanings may be missed.
- Some offensive expressions may be incorrectly classified.
- Predictions should support, not replace, human judgement.
"""
)


# ============================================================
# Developer Information
# ============================================================

st.divider()
st.subheader("👨‍💻 Developer Information")

st.markdown(
    """
**Arunava Kumar Chakraborty**

*Data Analyst | Machine Learning Enthusiast*

[LinkedIn](https://www.linkedin.com/in/arunava-kr-chakraborty)  
[GitHub](https://github.com/ArunavaKumar)
"""
)