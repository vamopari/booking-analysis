"""
Train all 5 models for the Hotel Booking Cancellation assignment.
Run from project root: pixi run python model/train.py
"""
import os
import sys
import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import load_clean_data, split_data, build_preprocessor, TARGET

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(MODEL_DIR)
KNN_SAMPLE_SIZE = 15000  # cap so the saved KNN model stays deployment-friendly

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "decision_tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
    "knn": KNeighborsClassifier(n_neighbors=15),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
}

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "kNN",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest (Ensemble)",
}


def main():
    df = load_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)

    results = []
    for key, model in MODELS.items():
        preprocessor = build_preprocessor()
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])

        if key == "knn" and len(X_train) > KNN_SAMPLE_SIZE:
            X_fit, _, y_fit, _ = train_test_split(
                X_train, y_train, train_size=KNN_SAMPLE_SIZE,
                random_state=RANDOM_STATE, stratify=y_train,
            )
        else:
            X_fit, y_fit = X_train, y_train

        pipe.fit(X_fit, y_fit)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": DISPLAY_NAMES[key],
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_proba),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred),
        }
        results.append(metrics)
        print(f"[OK] {DISPLAY_NAMES[key]:28s}", {k: round(v, 3) for k, v in metrics.items() if k != "Model"})

        joblib.dump(pipe, os.path.join(MODEL_DIR, f"{key}.joblib"))

    metrics_df = pd.DataFrame(results)[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    metrics_df.to_csv(os.path.join(MODEL_DIR, "metrics_comparison.csv"), index=False)
    print("\nSaved model/metrics_comparison.csv")
    print(metrics_df.to_string(index=False))

    # Save a sample of the test split (features + true label) for the Streamlit app upload demo
    test_sample = X_test.copy()
    test_sample[TARGET] = y_test.values
    test_sample = test_sample.sample(n=min(1000, len(test_sample)), random_state=RANDOM_STATE)
    test_sample.to_csv(os.path.join(BASE_DIR, "test_data.csv"), index=False)
    print(f"Saved {len(test_sample)}-row test_data.csv to project root")


if __name__ == "__main__":
    main()
