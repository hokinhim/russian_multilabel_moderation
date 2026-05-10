import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_DISPLAY_NAMES = {
    "micro_f1": "Micro-F1",
    "macro_f1": "Macro-F1",
    "samples_f1": "Samples-F1",
    "hamming_loss": "Hamming loss",
    "rare_recall_mean": "Rare recall",
}


def setup_plot_style() -> None:
    """
    Настройка графиков.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.facecolor": "white",
            "legend.edgecolor": "#DDDDDD",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def load_metrics(path: Path) -> dict:
    """
    Загружает JSON-файл с метриками модели.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл с метриками не найден: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_per_class_metrics(path: Path) -> pd.DataFrame:
    """
    Загружает CSV-файл с поклассовыми метриками модели.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл с поклассовыми метриками не найден: {path}")

    return pd.read_csv(path)


def build_comparison_table(tfidf_metrics: dict, rubert_metrics: dict) -> pd.DataFrame:
    """
    Формирует таблицу сравнения.
    """
    rows = []

    for model_name, metrics in [
        ("TF-IDF + LogisticRegression", tfidf_metrics),
        ("RuBERT classifier", rubert_metrics),
    ]:
        rows.append(
            {
                "model": model_name,
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
                "samples_f1": metrics["samples_f1"],
                "hamming_loss": metrics["hamming_loss"],
                "rare_recall_mean": metrics["rare_recall_mean"],
            }
        )

    return pd.DataFrame(rows)


def merge_per_class_metrics(tfidf_per_class: pd.DataFrame, rubert_per_class: pd.DataFrame) -> pd.DataFrame:
    """
    Объединяет поклассовые метрики TF-IDF и RuBERT.
    """
    tfidf = tfidf_per_class.copy()
    rubert = rubert_per_class.copy()

    tfidf = tfidf.rename(
        columns={
            "support": "support_tfidf",
            "precision": "precision_tfidf",
            "recall": "recall_tfidf",
            "f1": "f1_tfidf",
        }
    )

    rubert = rubert.rename(
        columns={
            "support": "support_rubert",
            "precision": "precision_rubert",
            "recall": "recall_rubert",
            "f1": "f1_rubert",
        }
    )

    merged = tfidf.merge(rubert, on="label", how="inner")
    merged = merged.sort_values("support_tfidf", ascending=True).reset_index(drop=True)
    return merged


def add_horizontal_bar_labels(ax, fmt: str = "{:.3f}", padding: float = 0.01, fontsize: int = 8) -> None:
    """
    Добавляет числовые значения справа от горизонтальных столбцов.
    """
    for container in ax.containers:
        for bar in container:
            width = bar.get_width()

            if np.isnan(width):
                continue

            ax.text(
                width + padding,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(width),
                ha="left",
                va="center",
                fontsize=fontsize,
                color="#222222",
            )


def save_metrics_plot(comparison: pd.DataFrame, plots_dir: Path) -> None:
    """
    Сохраняет столбчатые диаграммы для всех метрик.
    """
    metric_cols = ["micro_f1", "macro_f1", "samples_f1", "hamming_loss", "rare_recall_mean"]
    tfidf_row = comparison[comparison["model"] == "TF-IDF + LogisticRegression"].iloc[0]
    rubert_row = comparison[comparison["model"] == "RuBERT classifier"].iloc[0]

    x = np.arange(len(metric_cols))
    width = 0.3

    tfidf_values = [tfidf_row[col] for col in metric_cols]
    rubert_values = [rubert_row[col] for col in metric_cols]

    _, ax = plt.subplots(figsize=(16, 7))

    bars_1 = ax.bar(
        x - width / 2,
        tfidf_values,
        width,
        label="TF-IDF + логистическая регрессия",
        color="#4C78A8",
        edgecolor="#2F4B66",
        linewidth=0.8,
    )

    bars_2 = ax.bar(
        x + width / 2,
        rubert_values,
        width,
        label="RuBERT классификатор",
        color="#F58518",
        edgecolor="#9A530F",
        linewidth=0.8,
    )

    ax.set_xlabel("Метрика")
    ax.set_ylabel("Значение")
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_DISPLAY_NAMES[col] for col in metric_cols], rotation=0)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper center", ncols=2)

    for bars in [bars_1, bars_2]:
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_comparison.png", dpi=300)
    plt.close()


def save_per_class_f1_plot(per_class: pd.DataFrame, plots_dir: Path) -> None:
    """
    Сохраняет горизонтальные столбчатые диаграммы для F1 по каждому классу.
    """
    labels = per_class["label"]

    y = np.arange(len(per_class))
    height = 0.3

    _, ax = plt.subplots(figsize=(14, 11))

    ax.barh(
        y - height / 2,
        per_class["f1_tfidf"],
        height,
        label="TF-IDF + логистическая регрессия",
        color="#4C78A8",
        edgecolor="#2F4B66",
        linewidth=0.6,
    )

    ax.barh(
        y + height / 2,
        per_class["f1_rubert"],
        height,
        label="RuBERT классификатор",
        color="#F58518",
        edgecolor="#9A530F",
        linewidth=0.6,
    )

    ax.set_xlabel("F1")
    ax.set_ylabel("Класс")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, rotation=0)
    ax.set_xlim(0, 1.02)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.legend(loc="lower right")

    add_horizontal_bar_labels(ax, fmt="{:.3f}", padding=0.01, fontsize=8)

    plt.tight_layout()
    plt.savefig(plots_dir / "per_class_f1.png", dpi=300)
    plt.close()


def save_per_class_recall_plot(per_class: pd.DataFrame, plots_dir: Path) -> None:
    """
    Сохраняет горизонтальный grouped bar chart для recall по каждому классу.
    """
    labels = per_class["label"]

    y = np.arange(len(per_class))
    height = 0.3

    fig, ax = plt.subplots(figsize=(14, 11))

    ax.barh(
        y - height / 2,
        per_class["recall_tfidf"],
        height,
        label="TF-IDF + логистическая регрессия",
        color="#72B7B2",
        edgecolor="#3E7774",
        linewidth=0.6,
    )

    ax.barh(
        y + height / 2,
        per_class["recall_rubert"],
        height,
        label="RuBERT классификатор",
        color="#E45756",
        edgecolor="#8A2E2D",
        linewidth=0.6,
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Класс")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, rotation=0)
    ax.set_xlim(0, 1.02)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.legend(loc="lower right")

    add_horizontal_bar_labels(ax, fmt="{:.3f}", padding=0.01, fontsize=8)

    plt.tight_layout()
    plt.savefig(plots_dir / "per_class_recall.png", dpi=300)
    plt.close()


def main() -> None:
    setup_plot_style()

    report_dir = Path("reports")
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    tfidf_metrics = load_metrics(report_dir / "tfidf_metrics.json")
    rubert_metrics = load_metrics(report_dir / "rubert_metrics.json")
    tfidf_per_class = load_per_class_metrics(report_dir / "tfidf_per_class.csv")
    rubert_per_class = load_per_class_metrics(report_dir / "rubert_per_class.csv")

    comparison = build_comparison_table(tfidf_metrics=tfidf_metrics, rubert_metrics=rubert_metrics)
    comparison_path = report_dir / "comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    per_class_comparison = merge_per_class_metrics(tfidf_per_class=tfidf_per_class, rubert_per_class=rubert_per_class)
    per_class_comparison_path = report_dir / "per_class_comparison.csv"
    per_class_comparison.to_csv(per_class_comparison_path, index=False)

    save_metrics_plot(comparison, plots_dir)
    save_per_class_f1_plot(per_class_comparison, plots_dir)
    save_per_class_recall_plot(per_class_comparison, plots_dir)

    print("[+] Результат:")
    print(comparison)


if __name__ == "__main__":
    main()
