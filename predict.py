# predict.py

import json
import joblib
import numpy as np

from preprocessing import preprocess_for_inference

# -----------------------------
# Load Model & Label Encoder
# -----------------------------

MODEL_PATH = "model/final_model_tuned distilbert.joblib"
LABEL_PATH = "model/label_encoder.json"

model = joblib.load(MODEL_PATH)

with open(LABEL_PATH, "r") as f:
    label_encoder = json.load(f)

# -----------------------------
# Prediction Function
# -----------------------------

def predict_sentence(text: str):
    X = preprocess_for_inference(text)

    pred_class = model.predict(X)[0]
    pred_prob = model.predict_proba(X)[0][1]

    label = label_encoder[str(pred_class)]

    return label, float(pred_prob)

# -----------------------------
# User Input
# -----------------------------

if __name__ == "__main__":
    user_text = input("Enter a sentence to analyze: ").strip()

    if len(user_text) == 0:
        print("Error: Empty input.")
    else:
        label, confidence = predict_sentence(user_text)

        print("\nPrediction Result")
        print("-----------------")
        print(f"Text       : {user_text}")
        print(f"Prediction : {label}")
        print(f"Confidence : {confidence:.4f}")
