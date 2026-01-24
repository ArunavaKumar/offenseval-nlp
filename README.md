# 🚨 OffensEval NLP

### Transformer-Based Offensive Language Detection with Sentiment-Aware Modeling

A **production-ready NLP system** for offensive language detection, leveraging **DistilBERT sentence embeddings**, **sentiment-aware feature augmentation**, and a **Logistic Regression classifier**, deployed as an interactive **Streamlit web application** with **CI-validated inference and automated testing**.

---

## 🌐 Live Demo

👉 **Deployed Application:**
[https://offenseval-nlp-ojkd85r386zbffvjbvrkst.streamlit.app/](https://offenseval-nlp-ojkd85r386zbffvjbvrkst.streamlit.app/)

> The app demonstrates real-time offensive language detection with sentiment transparency and confidence-based predictions.

---

## 📌 Project Overview

This project implements a **complete end-to-end offensive language detection pipeline**, covering:

* Text preprocessing & normalization
* Transformer-based semantic embeddings
* Sentiment-aware feature engineering
* Supervised classification
* Model evaluation & validation
* Automated inference testing (CI-ready)
* Interactive web-based deployment

The system is trained and evaluated on the **TweetEval – Offensive Language** benchmark and designed with **interpretability, robustness, and deployability** in mind.

---

## 🎯 Key Features

* 🔍 **Transformer-based embeddings** (DistilBERT via SentenceTransformers)
* 😊 **Sentiment-aware modeling** using VADER compound scores
* ⚖️ **Interpretable classifier** (Logistic Regression)
* 📊 **Confidence scores & transparent explanations**
* 🧪 **Automated inference testing using PyTest**
* 🌐 **Interactive Streamlit web application**
* 🚀 **Deployment-ready architecture**
* 📁 **Clean, modular, industry-standard repository structure**

---

## 🧠 Model Architecture

```
Input Text
   ↓
Text Cleaning & Normalization
   ↓
DistilBERT Sentence Embeddings (768D)
   ↓
VADER Sentiment Score (+1 feature)
   ↓
Feature Concatenation (769D)
   ↓
Logistic Regression Classifier
   ↓
Offensive / Not Offensive Prediction
```

---

## 📊 Dataset

* **Dataset:** TweetEval – Offensive Language
* **Task:** Binary classification

  * `Offensive`
  * `Not Offensive`

---

## 📈 Model Performance (Validation)

| Metric   | Score |
| -------- | ----- |
| Macro F1 | ~0.72 |
| ROC-AUC  | ~0.81 |

> Metrics are reported on a held-out validation set following standard NLP evaluation practices.

---

## 🌐 Web Application Features

The deployed application provides:

* ✔️ Preprocessed text preview (model input transparency)
* ✔️ Predicted label with confidence score
* ✔️ Sentiment analysis (Positive / Neutral / Negative)
* ✔️ Color-coded sentiment bars with percentage distribution
* ✔️ Clear explanation of model reasoning
* ✔️ Explicit disclaimers and limitations

---

## 🧪 Automated Testing (CI-Ready)

Inference tests validate:

* Model loading and compatibility
* Correct preprocessing output shape
* Robust prediction across:

  * Short, medium, and long inputs
  * Offensive and non-offensive language
* Proper probability normalization

Run locally with:

```bash
pytest tests/test_inference.py
```

---

## 📁 Repository Structure

```
OffensEval-NLP/
│
├── app.py                     # Streamlit web application
├── predict.py                 # CLI inference script
├── preprocessing.py           # Centralized preprocessing & features
│
├── model/
│   ├── final_model_tuned_distilbert.joblib
│   └── label_encoder.json
│
├── embeddings/                # Saved embedding artifacts
│
├── tests/
│   └── test_inference.py      # Automated inference tests
│
├── Model_Training_and_Evaluation.ipynb
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚠️ Disclaimer & Limitations

This system predicts offensive language based on patterns learned from social media data.

**Limitations:**

* Sarcasm and humor may not be interpreted correctly
* Cultural and contextual nuances may be missed
* Predictions are intended to **support**, not replace, human judgment

Use responsibly.

---

## 👨‍💻 Developer

**Arunava Kumar Chakraborty**

*Data Analyst | ML Enthusiast*

🔗 [https://www.linkedin.com/in/arunava-kr-chakraborty](https://www.linkedin.com/in/arunava-kr-chakraborty)

🐙 [https://github.com/ArunavaKumar](https://github.com/ArunavaKumar)

---

## 📜 License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

## ⭐ Acknowledgements

* HuggingFace SentenceTransformers
* VADER Sentiment Analysis
* TweetEval Benchmark
* Streamlit
