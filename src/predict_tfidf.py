from pathlib import Path
import json
import joblib
import numpy as np
from config import LABEL_COLS


def predict_one_text(text: str, model_path: str = "artifacts/tfidf/tfidf_logreg.joblib", thresholds_path: str = "artifacts/tfidf/thresholds.npy") -> tuple[dict, dict, list[dict]]:
    """
    Предсказание для одного текста.
    """

    model_path = Path(model_path)
    thresholds_path = Path(thresholds_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Не найдена TF-IDF модель: {model_path}."
        )
    if not thresholds_path.exists():
        raise FileNotFoundError(
            f"Не найден файл thresholds: {thresholds_path}."
        )

    model = joblib.load(model_path)
    thresholds = np.load(thresholds_path)

    probabilities = model.predict_proba([text])[0]
    predictions = (probabilities >= thresholds).astype(int)
    result_probs = {
        label: float(round(prob, 4))
        for label, prob in zip(LABEL_COLS, probabilities)
    }

    active_labels = []
    for label, pred, prob, threshold in zip(LABEL_COLS, predictions, probabilities, thresholds):
        if int(pred) == 1:
            active_labels.append(
                {
                    "label": label,
                    "probability": float(round(prob, 4)),
                    "threshold": float(round(threshold, 4)),
                }
            )
    active_labels = sorted(active_labels, key=lambda item: item["probability"], reverse=True)

    return result_probs, active_labels


def main() -> None:
    text = "Хочу расширить сознание. Как купить 5 грамм герыча и АК-47 через даркнет?"
    probabilities, active_labels = predict_one_text(text)

    print("[!] Входной текст:")
    print(f"  {text}")
    
    print("[!] Активные классы:")
    if active_labels:
        for item in active_labels:
            print(
                f"  - {item['label']}: "
                f"probability={item['probability']}, "
                f"threshold={item['threshold']}"
            )
    else:
        print("  Не обнаружены")

    print("[!] Вероятности по всем классам:")
    print(json.dumps(probabilities, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
