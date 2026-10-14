import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.datasets import make_classification

st.set_page_config(page_title="Transaction Fraud Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ Transaction Fraud Detection")
st.caption("Educational demo using synthetic data. It is not a production banking fraud detector.")

@st.cache_data
def make_demo_data(n=8000, seed=42):
    X, y = make_classification(
        n_samples=n, n_features=8, n_informative=5, n_redundant=1,
        n_clusters_per_class=2, weights=[0.975, 0.025],
        class_sep=1.15, random_state=seed
    )
    df = pd.DataFrame(X, columns=[
        "amount_signal", "velocity_signal", "distance_signal", "merchant_risk",
        "account_age_signal", "time_signal", "device_signal", "history_signal"
    ])
    # Add human-readable features while retaining synthetic patterns.
    rng = np.random.default_rng(seed)
    df["amount"] = np.clip(np.exp(df["amount_signal"] + 3.0), 1, 5000).round(2)
    df["transactions_last_hour"] = np.clip((df["velocity_signal"] * 2 + 3).round(), 0, 25).astype(int)
    df["distance_from_home_km"] = np.clip((df["distance_signal"] * 30 + 25), 0, 1500).round(1)
    df["merchant_category"] = rng.choice(["grocery", "travel", "electronics", "food", "digital"], n,
                                          p=[.25, .12, .18, .25, .20])
    df["device_type"] = rng.choice(["known", "new", "mobile", "desktop"], n, p=[.60, .12, .20, .08])
    df["is_fraud"] = y
    return df

@st.cache_resource
def train_model():
    df = make_demo_data()
    features = ["amount", "transactions_last_hour", "distance_from_home_km",
                "merchant_category", "device_type"]
    X = df[features]
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    numeric = ["amount", "transactions_last_hour", "distance_from_home_km"]
    categorical = ["merchant_category", "device_type"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    model = Pipeline([
        ("preprocess", pre),
        ("classifier", RandomForestClassifier(
            n_estimators=180, class_weight="balanced", random_state=42, n_jobs=-1,
            min_samples_leaf=2
        ))
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "auc": roc_auc_score(y_test, probs),
        "cm": confusion_matrix(y_test, preds),
        "report": classification_report(y_test, preds, zero_division=0)
    }
    return model, df, metrics

model, demo_df, metrics = train_model()
tab1, tab2, tab3 = st.tabs(["Predict a transaction", "Model evaluation", "Explore demo data"])

with tab1:
    st.subheader("Enter transaction details")
    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input("Transaction amount", min_value=0.01, max_value=100000.0, value=249.99, step=10.0)
        velocity = st.number_input("Transactions in the last hour", min_value=0, max_value=100, value=2)
        distance = st.number_input("Distance from usual location (km)", min_value=0.0, max_value=20000.0, value=8.0)
    with c2:
        merchant = st.selectbox("Merchant category", ["grocery", "travel", "electronics", "food", "digital"])
        device = st.selectbox("Device type", ["known", "new", "mobile", "desktop"])
        threshold = st.slider("Flagging threshold", 0.10, 0.90, 0.50, 0.05)
    if st.button("Analyze transaction", type="primary"):
        sample = pd.DataFrame([{
            "amount": amount, "transactions_last_hour": velocity,
            "distance_from_home_km": distance, "merchant_category": merchant, "device_type": device
        }])
        probability = float(model.predict_proba(sample)[0, 1])
        flagged = probability >= threshold
        st.metric("Estimated fraud probability", f"{probability:.1%}")
        if flagged:
            st.error("Flag for review — the model score is above your selected threshold.")
        else:
            st.success("Not flagged by this demo model — this does not guarantee the transaction is safe.")
        st.caption("This probability is generated by a model trained on synthetic data and must not be used for real financial decisions.")

with tab2:
    st.subheader("Held-out test-set metrics")
    cols = st.columns(4)
    for col, label, key in zip(cols, ["Precision", "Recall", "F1-score", "ROC-AUC"],
                               ["precision", "recall", "f1", "auc"]):
        col.metric(label, f"{metrics[key]:.3f}")
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=metrics["cm"], display_labels=["Legitimate", "Fraud"])
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Confusion matrix (synthetic test data)")
    st.pyplot(fig)
    with st.expander("Classification report"):
        st.code(metrics["report"])
    st.info("These metrics describe a synthetic demonstration dataset, not real-world performance.")

with tab3:
    st.subheader("Synthetic sample data")
    st.dataframe(demo_df.drop(columns=[c for c in demo_df.columns if c.endswith("_signal")]).head(100), use_container_width=True)
    st.download_button("Download synthetic dataset CSV", demo_df.to_csv(index=False), "synthetic_transactions.csv", "text/csv")
