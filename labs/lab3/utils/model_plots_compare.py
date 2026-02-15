import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_model_comparison(results_df: pd.DataFrame, save_path: str):
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
    results_plot = results_df[metrics_to_plot].copy()

    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    models = results_plot.index
    for i, model in enumerate(models):
        plt.bar(x + i*width - width, results_plot.loc[model], width,
                label=model, alpha=0.8)

    plt.xlabel('Метрика')
    plt.ylabel('Значение')
    plt.title('Сравнение метрик качества моделей')
    plt.xticks(x, metrics_to_plot, rotation=45)
    plt.legend()
    plt.ylim(0.7, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
