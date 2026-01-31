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
    # Clean up column names in the uploaded file (remove spaces/lowercase)
    df.columns = df.columns.str.strip()

    # Get the 30 feature names the model expects
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    else:
        # Fallback if model doesn't store names
        expected_features = df.columns.tolist()
        expected_features = [c for c in expected_features if c.lower() not in ["id", "diagnosis"]]

    # Ensure all expected features are present in the uploaded DF
    missing = [f for f in expected_features if f not in df.columns]

    if missing:
        st.error(f"Missing columns in CSV: {missing}")
        st.stop() # Stop the app execution here to show the error clearly

    X = df[expected_features]

    # Handle Target for Evaluation
    # Look for 'diagnosis' or 'Diagnosis'
    target_col = next((c for c in df.columns if c.lower() == "diagnosis"), None)
    
    if target_col:
        # Convert M/B to 1/0
        y = df[target_col].map({'M': 1, 'B': 0, 1: 1, 0: 0})
        evaluation_mode = True
    else:
        y = None
        evaluation_mode = False

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
