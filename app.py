import io
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
    "Random Forest": "model/random_forest.joblib",
}

DEFAULT_SELECTION = ["Random Forest", "Naive Bayes"]

st.set_page_config(page_title="Hotel Booking Cancellation Classifier", layout="wide")
st.title("🏨 Hotel Booking Cancellation — Model Comparison")
st.caption("Upload test data, pick up to 2 models, and compare them side by side.")
st.caption("Download provided test_data ad upload it again or directly attach provided data.")

# --- Sidebar ---------------------------------------------------------
st.sidebar.header("Configuration")

with open("test_data.csv", "rb") as f:
    st.sidebar.download_button(
        label="⬇️ Download test_data.csv",
        data=f,
        file_name="test_data.csv",
        mime="text/csv",
        help="Downloads the bundled 1,000+ row sample test dataset to your machine.",
    )

uploaded_file = st.sidebar.file_uploader(
    "Upload test data (CSV)",
    type=["csv"],
    help="Upload your own CSV with the same columns as test_data.csv, "
         "including the true 'is_canceled' label column.",
)

use_sample_clicked = st.sidebar.button(
    "📎 Use bundled test_data.csv",
    help="Loads the bundled sample test dataset directly, without needing "
         "to download and re-upload it.",
)

selected_models = st.sidebar.multiselect(
    "Choose up to 2 models to compare",
    options=list(MODEL_FILES.keys()),
    default=DEFAULT_SELECTION,
    max_selections=2,
)

if use_sample_clicked:
    st.session_state["data_source"] = "sample"
elif uploaded_file is not None:
    st.session_state["data_source"] = "uploaded"
    st.session_state["uploaded_data"] = pd.read_csv(uploaded_file)

data_source = st.session_state.get("data_source")

if data_source == "sample":
    data = pd.read_csv("test_data.csv")
    st.sidebar.success("Using bundled test_data.csv")
elif data_source == "uploaded":
    data = st.session_state["uploaded_data"]
    st.sidebar.success("Using your uploaded CSV")
else:
    data = None


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def compute_metrics(model, X, y_true):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return y_pred, y_proba, {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def render_model_column(model_name, X, y_true):
    model = load_model(MODEL_FILES[model_name])
    y_pred, y_proba, metrics = compute_metrics(model, X, y_true)

    st.markdown(f"### {model_name}")

    metric_cols = st.columns(3)
    for i, (k, v) in enumerate(metrics.items()):
        metric_cols[i % 3].metric(k, f"{v:.3f}")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Not\nCancelled", "Cancelled"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title("")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    st.image(buf, width=380)

    with st.expander("Classification Report"):
        report = classification_report(
            y_true, y_pred, target_names=["Not Cancelled", "Cancelled"], output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3))

if data is None:
    st.info(
        "👈 Get started from the sidebar: download the sample data, "
        "upload your own CSV, or click 'Use bundled test_data.csv'."
    )
else:
    st.subheader("Preview of data")
    st.dataframe(data.head())

    if TARGET not in data.columns:
        st.error(f"CSV must contain the '{TARGET}' column with true labels.")
    elif not selected_models:
        st.warning("Select at least one model from the sidebar.")
    else:
        X = data.drop(columns=[TARGET])
        y_true = data[TARGET].astype(int)

        st.subheader("Side-by-Side Comparison")
        cols = st.columns(len(selected_models))
        for col, model_name in zip(cols, selected_models):
            with col:
                render_model_column(model_name, X, y_true)

st.divider()
st.subheader("All Models — Full Comparison")
comp_df = pd.read_csv("model/metrics_comparison.csv")
styled = comp_df.set_index("Model").style.background_gradient(cmap="Greens", axis=0).format("{:.3f}")
st.dataframe(styled, width="stretch")
