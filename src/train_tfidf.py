import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from config import TEXT_COL, LABEL_COLS, RANDOM_STATE
from metrics import tune_thresholds, apply_thresholds, evaluate_multilabel


def main() -> None:
    data_dir = Path("data/processed")
    artifact_dir = Path("artifacts/tfidf")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    x_train = train_df[TEXT_COL].astype(str)
    x_val = val_df[TEXT_COL].astype(str)
    x_test = test_df[TEXT_COL].astype(str)

    y_train = train_df[LABEL_COLS].values
    y_val = val_df[LABEL_COLS].values
    y_test = test_df[LABEL_COLS].values

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=200_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="liblinear",
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    )
                ),
            ),
        ]
    )

    print("[!] Обучение TF-IDF...")
    model.fit(x_train, y_train)

    print("[!] Подбор thresholds на validation...")
    val_proba = model.predict_proba(x_val)
    thresholds = tune_thresholds(y_val, val_proba, LABEL_COLS)

    print("[!] Оценка на test...")
    test_proba = model.predict_proba(x_test)
    test_pred = apply_thresholds(test_proba, thresholds)

    metrics, per_class = evaluate_multilabel(y_test, test_pred, LABEL_COLS)

    joblib.dump(model, artifact_dir / "tfidf_logreg.joblib")
    np.save(artifact_dir / "thresholds.npy", thresholds)

    with open(report_dir / "tfidf_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    per_class.to_csv(report_dir / "tfidf_per_class.csv", index=False)

    print("[+] Результат:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
