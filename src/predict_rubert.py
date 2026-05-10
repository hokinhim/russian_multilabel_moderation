from pathlib import Path
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import LABEL_COLS


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Применяет sigmoid к logits.
    """
    return 1.0 / (1.0 + np.exp(-x))


def predict_one_text(text: str, model_dir: str = "artifacts/rubert/best_model", thresholds_path: str = "artifacts/rubert/thresholds.npy", max_length: int = 192) -> tuple[dict, dict, list[dict]]:
    """
    Выполняет предсказание для одного текста.
    """

    model_dir = Path(model_dir)
    thresholds_path = Path(thresholds_path)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Не найдена RuBert модель: {model_dir}. "
        )
    if not thresholds_path.exists():
        raise FileNotFoundError(
            f"Не найден файл thresholds: {thresholds_path}. "
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    thresholds = np.load(thresholds_path)

    encoded = tokenizer(text, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    # Отключение вычисления градиентов
    with torch.no_grad():
        outputs = model(**encoded)

    logits = outputs.logits.detach().cpu().numpy()[0]
    probabilities = sigmoid(logits)
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
