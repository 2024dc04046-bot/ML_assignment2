import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("breast_cancer_wisconsin.csv")

X = df.drop(["id", "diagnosis"], axis=1)
y = df["diagnosis"].map({"M": 1, "B": 0})


# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# =========================
# MODELS
# =========================
models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "decision_tree": DecisionTreeClassifier(random_state=42),

    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "naive_bayes": GaussianNB(),

    "random_forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),

    "xgboost": XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
}


# =========================
# TRAIN & SAVE
# =========================
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"model/{name}.pkl")

print("✅ All models trained and saved")
