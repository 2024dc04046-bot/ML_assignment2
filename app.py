import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns


st.title("Breast Cancer Classification App")

uploaded_file = st.file_uploader(
    "Upload CSV test dataset (UCI format)",
    type=["csv"]
)

model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None)

    X = df.iloc[:, 2:]
    y = df.iloc[:, 1].map({"M": 1, "B": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_choice == "kNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_choice == "Naive Bayes":
        model = GaussianNB()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    st.subheader("Evaluation Metrics")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("AUC:", roc_auc_score(y_test, y_prob))
    st.write("Precision:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("F1 Score:", f1_score(y_test, y_pred))
    st.write("MCC:", matthews_corrcoef(y_test, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
