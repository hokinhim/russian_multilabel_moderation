import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from config import TEXT_COL, LABEL_COLS, RUBERT_MODEL_NAME, RANDOM_STATE
from metrics import tune_thresholds, apply_thresholds, evaluate_multilabel


class WeightedMultilabelTrainer(Trainer):
    """
    Trainer для multilabel-классификации с учетом дисбаланса классов.
    pos_weight увеличивает штраф за ошибку на положительных примерах редких классов.
    """
    def __init__(self, *args, pos_weight: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(logits.device)
        else:
            pos_weight = None

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fn(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Применение sigmoid к logits.
    """
    return 1.0 / (1.0 + np.exp(-x))


def dataframe_to_hf_dataset(df: pd.DataFrame) -> Dataset:
    """
    Переводит pandas DataFrame в Hugging Face Dataset.
    """
    return Dataset.from_pandas(df[[TEXT_COL] + LABEL_COLS], preserve_index=False)


def build_tokenize_fn(tokenizer, max_length: int = 192) -> callable:
    """
    Создает функцию токенизации.
    """

    def tokenize_fn(batch):
        tokenized = tokenizer(batch[TEXT_COL], truncation=True, max_length=max_length)

        labels = []
        for i in range(len(batch[TEXT_COL])):
            row_labels = [float(batch[col][i]) for col in LABEL_COLS]
            labels.append(row_labels)

        tokenized["labels"] = labels
        return tokenized

    return tokenize_fn


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """
    Считает pos_weight для BCEWithLogitsLoss.
    """
    positives = y_train.sum(axis=0)
    negatives = y_train.shape[0] - positives

    pos_weight = negatives / np.clip(positives, a_min=1, a_max=None)
    pos_weight = np.clip(pos_weight, a_min=1.0, a_max=10.0)

    return torch.tensor(pos_weight, dtype=torch.float32)


def evaluate_from_logits(logits: np.ndarray, y_true: np.ndarray, thresholds: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """
    Переводит logits в вероятности, применяет thresholds и считает метрики.
    """
    proba = sigmoid(logits)
    pred = apply_thresholds(proba, thresholds)
    return evaluate_multilabel(y_true, pred, LABEL_COLS)


def main() -> None:
    data_dir = Path("data/processed")
    artifact_dir = Path("artifacts/rubert")
    report_dir = Path("reports")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    y_train = train_df[LABEL_COLS].values.astype(np.float32)
    y_val = val_df[LABEL_COLS].values.astype(np.float32)
    y_test = test_df[LABEL_COLS].values.astype(np.float32)

    tokenizer = AutoTokenizer.from_pretrained(RUBERT_MODEL_NAME)

    id2label = {i: label for i, label in enumerate(LABEL_COLS)}
    label2id = {label: i for i, label in enumerate(LABEL_COLS)}

    model = AutoModelForSequenceClassification.from_pretrained(
        RUBERT_MODEL_NAME,
        num_labels=len(LABEL_COLS),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )

    train_ds = dataframe_to_hf_dataset(train_df)
    val_ds = dataframe_to_hf_dataset(val_df)
    test_ds = dataframe_to_hf_dataset(test_df)

    tokenize_fn = build_tokenize_fn(tokenizer)

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=[TEXT_COL] + LABEL_COLS)
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=[TEXT_COL] + LABEL_COLS)
    test_ds = test_ds.map(tokenize_fn, batched=True, remove_columns=[TEXT_COL] + LABEL_COLS)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    pos_weight = compute_pos_weight(y_train)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        proba = sigmoid(logits)
        pred = (proba >= 0.5).astype(int)

        metrics, _ = evaluate_multilabel(labels.astype(int), pred.astype(int), LABEL_COLS)

        return {
            "micro_f1": metrics["micro_f1"],
            "macro_f1": metrics["macro_f1"],
            "samples_f1": metrics["samples_f1"],
            "hamming_loss": metrics["hamming_loss"],
            "rare_recall_mean": metrics["rare_recall_mean"],
        }

    training_args = TrainingArguments(
        output_dir=str(artifact_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=RANDOM_STATE,
    )

    trainer = WeightedMultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        pos_weight=pos_weight,
    )

    print("[!] Обучение RuBERT...")
    trainer.train()

    print("[!] Подбор thresholds на validation...")
    val_logits = trainer.predict(val_ds).predictions
    val_proba = sigmoid(val_logits)
    thresholds = tune_thresholds(y_val.astype(int), val_proba, LABEL_COLS)

    print("[!] Оценка на test...")
    test_logits = trainer.predict(test_ds).predictions
    test_metrics, test_per_class = evaluate_from_logits(test_logits, y_test.astype(int), thresholds)

    trainer.save_model(str(artifact_dir / "best_model"))
    tokenizer.save_pretrained(str(artifact_dir / "best_model"))
    np.save(artifact_dir / "thresholds.npy", thresholds)

    with open(report_dir / "rubert_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    test_per_class.to_csv(report_dir / "rubert_per_class.csv", index=False)

    print("[+] Результат:")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
