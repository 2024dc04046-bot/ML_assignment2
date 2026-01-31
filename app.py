# =========================
# 1. IMPORT LIBRARIES
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score
)

import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# 2. PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ML Assignment – Breast Cancer Prediction",
    layout="centered"
)

st.title("Breast Cancer Classification App")
st.write("Upload test data and evaluate ML models")


# =========================
# 3. MODEL LOADING
# =========================
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "kNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl")
    }
    return models


models = load_models()


# =========================
# 4. MODEL SELECTION
# =========================
model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]


# =========================
# 5. FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload CSV file (Test data only)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())

    # =========================
    # 6. FEATURE / TARGET SPLIT
    # =========================
    # 1. Identify Target
    if "diagnosis" in df.columns:
        y = df["diagnosis"].map({"M": 1, "B": 0})
        evaluation_mode = True
    else:
        y = None
        evaluation_mode = False

    # 2. Extract Features (Ensure exactly 30 columns in correct order)
    # Most sklearn models store feature names in 'feature_names_in_'
    if hasattr(model, "feature_names_in_"):
        expected_features = model.feature_names_in_
        X = df[expected_features]
    else:
        # Fallback if names aren't stored: drop metadata manually
        cols_to_drop = [c for c in ["id", "ID", "diagnosis", "Diagnosis"] if c in df.columns]
        X = df.drop(cols_to_drop, axis=1)

    # =========================
    # 7. PREDICTION
    # =========================
    y_pred = model.predict(X)

    st.subheader("Predictions")
    st.write(y_pred[:10])


    # =========================
    # 8. METRICS (ONLY IF LABEL EXISTS)
    # =========================
    if evaluation_mode:

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

        # AUC only if probability is supported
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_prob)
        else:
            auc = "Not Available"

        st.subheader("Evaluation Metrics")

        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "MCC", "AUC"],
            "Value": [accuracy, precision, recall, f1, mcc, auc]
        })

        st.table(metrics_df)

        # =========================
        # 9. CONFUSION MATRIX
        # =========================
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    else:
        st.info("No target column found. Showing predictions only.")
