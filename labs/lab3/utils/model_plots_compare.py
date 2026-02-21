import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_model_comparison(results_df: pd.DataFrame, save_path: str):
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
    available_metrics = [m for m in metrics_to_plot if m in results_df.columns]

    if not available_metrics:
        raise ValueError("Нет доступных метрик для построения графика.")

    results_plot = results_df[available_metrics].copy()

    plt.figure(figsize=(12, 6))
    x = np.arange(len(available_metrics))
    models = results_plot.index.tolist()
    n_models = len(models)
    width = 0.8 / max(1, n_models)

    offsets = [(i - (n_models - 1) / 2) * width for i in range(n_models)]

    for i, model in enumerate(models):
        plt.bar(x + offsets[i], results_plot.loc[model], width,
                label=model, alpha=0.8)

    plt.xlabel('Метрика')
    plt.ylabel('Значение')
    plt.title('Сравнение метрик качества моделей')
    plt.xticks(x, available_metrics, rotation=45)
    plt.legend()
    if 'ROC-AUC' in available_metrics:
        plt.ylim(0.7, 1.0)
    else:
        ymin = max(0.0, results_plot.min().min() - 0.05)
        ymax = min(1.0, results_plot.max().max() + 0.05)
        if ymin == ymax:
            ymin = max(0.0, ymin - 0.05)
            ymax = min(1.0, ymax + 0.05)
        plt.ylim(ymin, ymax)

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
