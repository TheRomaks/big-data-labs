import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from typing import List, Tuple


def plot_class_distribution(df: pd.DataFrame, output_dir: str = None, show: bool = False):
    target_col = 'Class' if 'Class' in df.columns else None

    if not target_col:
        print("Колонка 'Class' не найдена для распределения")
        return None

    class_dist = df[target_col].value_counts(normalize=True) * 100
    print(f"  Класс 0: {class_dist.get(0, 0):.2f}% ({df[target_col].value_counts().get(0, 0):,} объектов)")
    print(f"  Класс 1: {class_dist.get(1, 0):.2f}% ({df[target_col].value_counts().get(1, 0):,} объектов)")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x=target_col, hue=target_col, palette='husl', legend=False, ax=ax)
    ax.set_title('Распределение классов', fontsize=14, fontweight='bold')
    ax.set_xlabel('Класс')
    ax.set_ylabel('Количество объектов')

    if output_dir:
        fig.savefig(output_dir, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_feature_distributions(df: pd.DataFrame, numeric_cols: List[str], save_path: str = None, show: bool = False):
    cols_to_plot = numeric_cols[:8]
    n_cols = len(cols_to_plot)
    rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(cols_to_plot):
        sns.histplot(df[col], kde=True, bins=30, ax=axes[i])
        axes[i].set_title(f'Распределение {col}')
        axes[i].set_xlabel('')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_correlation_matrix(df: pd.DataFrame, numeric_cols: List[str], save_path: str = None, target_col: str = 'Class',
                            show: bool = False):
    cols_for_corr = numeric_cols.copy()
    if target_col in df.columns and target_col not in cols_for_corr:
        cols_for_corr.append(target_col)

    corr = df[cols_for_corr].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title('Матрица корреляции признаков')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def compute_summary_stats(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    pd.set_option('display.max_columns', None)

    stats_df = pd.DataFrame({
        'min': df[numeric_cols].min(),
        '25%': df[numeric_cols].quantile(0.25),
        'median': df[numeric_cols].median(),
        'mean': df[numeric_cols].mean(),
        '75%': df[numeric_cols].quantile(0.75),
        'max': df[numeric_cols].max()
    })

    print("\nСтатистика данных:")
    print(stats_df.round(3))

    return stats_df