from pathlib import Path
import numpy as np
import pandas as pd
from datasets import load_dataset
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from config import DATASET_NAME, TEXT_COL, LABEL_COLS, LABEL_THRESHOLD, RANDOM_STATE, TEST_SIZE, VAL_SIZE


def binarize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует float-метки в бинарные значения.
    """
    df = df.copy()
    for col in LABEL_COLS:
        df[col] = (df[col].astype(float) >= LABEL_THRESHOLD).astype(int)
    return df


def multilabel_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Делит датасет на train/val/test с сохранением пропорций по всем меткам.
    """
    y = df[LABEL_COLS].values
    dummy_x = np.zeros((len(df), 1))    # Признаки не важны 

    splitter_1 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_val_idx, test_idx = next(splitter_1.split(dummy_x, y))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    y_train_val = train_val_df[LABEL_COLS].values
    dummy_x_train_val = np.zeros((len(train_val_df), 1))

    splitter_2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=VAL_SIZE / (1.0 - TEST_SIZE),
        random_state=RANDOM_STATE,
    )

    train_idx, val_idx = next(splitter_2.split(dummy_x_train_val, y_train_val))
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def main() -> None:
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[!] Загрузка датасета...")
    dataset = load_dataset(DATASET_NAME)
    print("[+] Датасет загружен")

    print("[!] Подготовка данных...")
    df = dataset["train"].to_pandas()
    df = df[[TEXT_COL] + LABEL_COLS].copy()

    # Удаление пустых тестов
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df = df[df[TEXT_COL].str.len() > 0].reset_index(drop=True)

    # Бинаризация меток
    df = binarize_labels(df)

    # Разделение на train/val/test
    train_df, val_df, test_df = multilabel_split(df)

    # Сохранение в CSV
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("[+] Данные подготовлены")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")

    print("[+] Частоты классов в train:")
    print(train_df[LABEL_COLS].sum().sort_values())


if __name__ == "__main__":
    main()
