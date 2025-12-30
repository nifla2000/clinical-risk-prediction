# src/train.py

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

from config import DATA_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE
from preprocess import load_data, clean_data, build_preprocessor

def train():
    df = load_data(DATA_PATH)
    df = clean_data(df)

    X = df.drop(columns=["readmitted_binary"])
    y = df["readmitted_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X)

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
