# src/evaluate.py

import joblib
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

from config import MODEL_PATH, DATA_PATH, THRESHOLD
from preprocess import load_data, clean_data

def evaluate():
    model = joblib.load(MODEL_PATH)

    df = load_data(DATA_PATH)
    df = clean_data(df)

    X = df.drop(columns=["readmitted_binary"])
    y = df["readmitted_binary"]

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)

    print(f"Threshold: {THRESHOLD}")
    print(classification_report(y, y_pred))
    print("ROC-AUC:", roc_auc_score(y, y_proba))

if __name__ == "__main__":
    evaluate()
