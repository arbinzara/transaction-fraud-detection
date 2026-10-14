# Transaction Fraud Detection System

A Streamlit machine-learning demo that scores transactions and presents model evaluation metrics.

## Features
- Synthetic, imbalanced transaction dataset generated reproducibly.
- Scikit-learn preprocessing pipeline for numeric and categorical features.
- Random Forest classifier with class weighting.
- Interactive transaction scoring and adjustable flagging threshold.
- Precision, recall, F1-score, ROC-AUC, confusion matrix, and CSV download.

## Run locally
Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Important limitations
The included data is synthetic and is only for learning/demo purposes. Metrics are not evidence of real banking performance. For a serious project, replace it with a properly licensed dataset (for example, a public anonymized transaction dataset), document its source, avoid data leakage, tune the decision threshold, and evaluate precision-recall trade-offs. Do not use this demo to approve or block real transactions.

## Suggested next improvements
- Load a documented public dataset from a configurable CSV.
- Add time-aware validation and drift monitoring.
- Save the trained pipeline and add unit tests.
- Add a FastAPI prediction endpoint and Docker deployment.
