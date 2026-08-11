import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

TARGET = "is_canceled"

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest (Ensemble)": "model/random_forest.joblib",
}

st.set_page_config(layout="wide")
st.title("Hotel Booking Cancellation — Model Comparison App")

model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
model = joblib.load(MODEL_FILES[model_name])

uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(data.head())

    if TARGET not in data.columns:
        st.error(f"CSV must contain the '{TARGET}' column with true labels.")
    else:
        X = data.drop(columns=[TARGET])
        y_true = data[TARGET].astype(int)

        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        st.subheader(f"Evaluation Metrics — {model_name}")
        cols = st.columns(6)
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_proba),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }
        for col, (k, v) in zip(cols, metrics.items()):
            col.metric(k, f"{v:.3f}")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(4, 4))
            ConfusionMatrixDisplay(cm, display_labels=["Not Cancelled", "Cancelled"]).plot(ax=ax, cmap="Blues", colorbar=False)
            st.pyplot(fig)

        with col_right:
            st.subheader("Classification Report")
            report = classification_report(y_true, y_pred, target_names=["Not Cancelled", "Cancelled"], output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(3))

        st.subheader("Compare All Models")
        comp_df = pd.read_csv("model/metrics_comparison.csv")
        st.dataframe(comp_df)
else:
    st.info("Upload a CSV file to get started.")
