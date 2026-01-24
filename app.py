import streamlit as st
import json
import joblib
import numpy as np

from preprocessing import (
    clean_tweet,
    preprocess_for_inference,
    extract_sentiment,
    sentiment_category
)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -----------------------------
# Load Model & Metadata
# -----------------------------
MODEL_PATH = "model/final_model_tuned distilbert.joblib"
LABEL_PATH = "model/label_encoder.json"

model = joblib.load(MODEL_PATH)

with open(LABEL_PATH, "r") as f:
    label_map = json.load(f)

# Reverse mapping
inv_label_map = {v: k for k, v in label_map.items()}

# Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Offensive Language Detection", layout="centered")

st.title("🚨 Offensive Language Detection")
st.write(
    "Transformer-based offensive language classification using "
    "**Tuned DistilBERT + Sentiment**"
)

# -----------------------------
# Input Section
# -----------------------------
user_text = st.text_area(
    "Enter text input to analyze:",
    height=120,
    placeholder="Type a tweet or sentence here..."
)

# -----------------------------
# Prediction Logic
# -----------------------------
if st.button("Analyze"):

    if not user_text.strip():
        st.warning("⚠ Please enter some text.")
        st.stop()

    try:
        # -----------------------------
        # Preprocessing
        # -----------------------------
        cleaned = clean_tweet(user_text)
        X = preprocess_for_inference(user_text)

        # -----------------------------
        # Show Preprocessed Text FIRST
        # -----------------------------
        st.subheader("🧹 Preprocessed Text (Model Input)")
        st.code(cleaned)

        # -----------------------------
        # Prediction
        # -----------------------------
        probs = model.predict_proba(X)[0]
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]
        predicted_label = label_map[str(pred_class)]

        # -----------------------------
        # Prediction Result
        # -----------------------------
        st.subheader("📌 Prediction Result")

        st.markdown(
            f""" 
            **Confidence Score: `{confidence:.4f}`**

            **Predicted Offensive Language Category: `{predicted_label}`** 
            """
        )

        # ------------------
        # Sentiment Analysis
        # ------------------
        st.subheader("📊 Sentiment Analysis")

        sentiment_scores = sentiment_analyzer.polarity_scores(cleaned)

        pos_pct = sentiment_scores["pos"]
        neu_pct = sentiment_scores["neu"]
        neg_pct = sentiment_scores["neg"]
        compound = sentiment_scores["compound"]

        sent_category = sentiment_category(compound)

        def sentiment_bar(label, value, color):
            percent = int(value * 100)
            st.markdown(
                f"""
                <div style="margin-bottom:12px;">
                    <b>{label}</b> — {percent}%
                    <div style="background-color:#2b2b2b;border-radius:6px;">
                        <div style="
                            width:{percent}%;
                            background-color:{color};
                            padding:6px;
                            border-radius:6px;">
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        sentiment_bar("Positive Sentiment", pos_pct, "#2ecc71")
        sentiment_bar("Neutral Sentiment", neu_pct, "#f1c40f")
        sentiment_bar("Negative Sentiment", neg_pct, "#e74c3c")

        st.markdown(
            f"""
            **Compound Sentiment Score: `{compound:.3f}`**  
            **Overall Sentiment Category: `{sent_category}`**

            **Sentiment Interpretation Ranges:**  
            • **Positive:** compound ≥ +0.05  
            • **Neutral:** -0.05 < compound < +0.05  
            • **Negative:** compound ≤ -0.05  

            *(Overall scale: -1 = very negative, +1 = very positive)*
            """
        )

        # -----------------------------
        # Explanation Section
        # -----------------------------
        st.subheader("🧠 Why this prediction?")

        explanation_points = []

        if compound <= -0.5:
            explanation_points.append("Strong negative sentiment detected")
        elif compound < 0:
            explanation_points.append("Mild negative sentiment detected")
        elif compound > 0.5:
            explanation_points.append("Strong positive sentiment detected")
        else:
            explanation_points.append("Neutral or mixed sentiment detected")

        if predicted_label == "Offensive":
            explanation_points.append(
                "Contextual language patterns resemble offensive expressions"
            )
        else:
            explanation_points.append(
                "No strong offensive contextual signals detected"
            )

        for point in explanation_points:
            st.write(f"• {point}")

    except ValueError as e:
        st.error(f"❌ Input rejected: {str(e)}")

# -----------------------------
# Model Information Panel
# -----------------------------
st.subheader("📊 Model Information")

st.markdown(
    """
    **Embedding Model:** DistilBERT (SentenceTransformer)  
    **Additional Feature:** VADER sentiment (compound score)  
    **Classifier:** Logistic Regression  

    **Training Dataset:** TweetEval – Offensive Language  

    **Validation Performance:**  
    • Macro F1 ≈ 0.72  
    • ROC-AUC ≈ 0.81  
    """
)

# -----------------------------
# Disclaimer
# -----------------------------
st.subheader("⚠ Disclaimer")

st.markdown(
    """
    This system predicts offensive language based on patterns learned from
    social media data.

    **Limitations:**
    - Sarcasm and humor may not be interpreted correctly
    - Cultural context and implicit meanings may be missed
    - Predictions should support—not replace—human judgment
    """
)

# -----------------------------
# Developer Information
# -----------------------------
st.subheader("👨‍💻 Developer Information")

st.markdown(
    """
    **Arunava Kumar Chakraborty**
    
    *Data Analyst@Paperpedia*

    🔗 [https://www.linkedin.com/in/arunava-kr-chakraborty](https://www.linkedin.com/in/arunava-kr-chakraborty)

    🐙 [https://github.com/ArunavaKumar](https://github.com/ArunavaKumar)

    ---
    *This application is developed as part of an academic and research-oriented
    project on transformer-based offensive language detection.*
    """
)
