import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List

def plot_class_distribution(df: pd.DataFrame, output_dir: str):
    class_dist = df['Class'].value_counts(normalize=True) * 100

    print(f"  Класс 0 (не пульсар): {class_dist[0]:.2f}% ({df['Class'].value_counts()[0]:,} объектов)")
    print(f"  Класс 1 (пульсар):    {class_dist[1]:.2f}% ({df['Class'].value_counts()[1]:,} объектов)")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Class', hue='Class', palette='husl', legend=False)
    plt.title('Распределение классов', fontsize=14, fontweight='bold')
    plt.xlabel('Класс')
    plt.ylabel('Количество объектов')
    plt.savefig(output_dir, dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_distributions(df: pd.DataFrame, numeric_cols: List[str], save_path: str):
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols[:8], 1):
        plt.subplot(3, 3, i)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Распределение {col}')
        plt.xlabel('')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, numeric_cols: List[str], save_path: str):
    plt.figure(figsize=(12, 10))
    corr = df[numeric_cols + ['Class']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Корреляционная матрица признаков')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def compute_summary_stats(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    stats_df = pd.DataFrame({
        'min': df[numeric_cols].min(),
        '25%': df[numeric_cols].quantile(0.25),
        'median': df[numeric_cols].median(),
        'mean': df[numeric_cols].mean(),
        '75%': df[numeric_cols].quantile(0.75),
        'max': df[numeric_cols].max()
    })
    print(stats_df.round(3))
    return stats_df