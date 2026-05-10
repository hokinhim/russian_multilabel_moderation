import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support


def tune_thresholds(y_true: np.ndarray, y_proba: np.ndarray, label_cols: list[str]) -> np.ndarray:
    """
    Подбирает индивидуальный threshold для каждого класса, оптимизируя F1.
    """
    thresholds = np.zeros(len(label_cols), dtype=float)
    grid = np.linspace(0.05, 0.7, 14)

    for j, _ in enumerate(label_cols):
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in grid:
            y_pred_j = (y_proba[:, j] >= threshold).astype(int)
            f1 = f1_score(
                y_true[:, j],
                y_pred_j,
                zero_division=0,
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        thresholds[j] = best_threshold

    return thresholds


def apply_thresholds(y_proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Применяет индивидуальные пороги к матрице вероятностей.
    """
    return (y_proba >= thresholds.reshape(1, -1)).astype(int)


def evaluate_multilabel(y_true: np.ndarray, y_pred: np.ndarray, label_cols: list[str]) -> tuple[dict, pd.DataFrame]:
    """
    Считает метрики.
    """
    metrics = {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred)
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    per_class = pd.DataFrame(
        {
            "label": label_cols,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    ).sort_values("support")

    # Редкий класс = из нижнего квартиля по числу положительных примеров
    rare_threshold = per_class["support"].quantile(0.25)
    rare_classes = per_class[per_class["support"] <= rare_threshold]

    metrics["rare_recall_mean"] = float(rare_classes["recall"].mean())
    metrics["rare_classes"] = rare_classes["label"].tolist()

    return metrics, per_class
