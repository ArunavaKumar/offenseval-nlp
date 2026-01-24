# 🚨 OffensEval NLP – Offensive Language Detection using Transformers

A **production-ready NLP system** for detecting offensive language in text using **DistilBERT sentence embeddings**, **sentiment-aware feature augmentation**, and a **Logistic Regression classifier**, deployed via **Streamlit** with CI-validated inference.

---

## 📌 Project Overview

This project implements an **end-to-end offensive language detection pipeline**, covering:

* Text preprocessing & normalization
* Transformer-based semantic embeddings
* Sentiment-aware feature engineering
* Supervised classification
* Model evaluation & validation
* Automated inference testing (CI-ready)
* Interactive web-based deployment

The system is trained and evaluated on the **TweetEval – Offensive Language** dataset and designed with **transparency, reproducibility, and deployability** in mind.

---

## 🎯 Key Features

* 🔍 **Transformer-based embeddings** (DistilBERT – SentenceTransformers)
* 😊 **Sentiment-aware modeling** using VADER compound scores
* ⚖️ **Interpretable classifier** (Logistic Regression)
* 📊 **Confidence scores & explanations**
* 🧪 **Automated inference tests using PyTest**
* 🌐 **Streamlit web application**
* 🚀 **Deployment-ready for Streamlit Cloud**
* 📁 **Clean, modular, industry-standard repo structure**

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

## 🌐 Web Application (Streamlit)

The Streamlit app provides:

* ✔️ Preprocessed text preview
* ✔️ Predicted label & confidence score
* ✔️ Sentiment analysis (Positive / Neutral / Negative)
* ✔️ Color-coded sentiment bars with percentages
* ✔️ Model explanation & transparency
* ✔️ Clear disclaimers and limitations

To run locally:

```bash
streamlit run app.py
```

---

## 🧪 Automated Testing (CI-Ready)

Inference tests ensure:

* Model loads correctly
* Preprocessing returns valid feature vectors
* Predictions run successfully across:

  * Short, medium, and long inputs
  * Offensive and non-offensive language
* Probability outputs are valid and normalized

Run tests with:

```bash
pytest tests/
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
* Cultural context and implicit meanings may be missed
* Not intended to replace human moderation or judgment

Use predictions responsibly.

---

## 👨‍💻 Developer Information

**Name:** Arunava Kumar Chakraborty

🔗 **LinkedIn:**
[https://www.linkedin.com/in/arunava-kr-chakraborty](https://www.linkedin.com/in/arunava-kr-chakraborty)

🐙 **GitHub:**
[https://github.com/ArunavaKumar](https://github.com/ArunavaKumar)

---

## 📜 License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

## 🚀 Deployment

This application can be deployed **for free** using **Streamlit Cloud**:

1. Fork or clone this repository
2. Log in to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Select `app.py` as the entry point
4. Deploy 🚀

---

## ⭐ Acknowledgements

* HuggingFace SentenceTransformers
* VADER Sentiment Analysis
* TweetEval Benchmark
* Streamlit

---
