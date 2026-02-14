import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

def ensure_output_dir():
    Path('data').mkdir(exist_ok=True)

def plot_distribution(df: pd.DataFrame, column: str, title: str, xlabel: str, ylabel: str,
                      save_path: str = None):
    ensure_output_dir()
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column].dropna(), bins=20, kde=True, color='steelblue')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axvline(df[column].mean(), color='red', linestyle='--',
                label=f'Среднее: {df[column].mean():.1f}')
    plt.axvline(df[column].median(), color='green', linestyle='--',
                label=f'Медиана: {df[column].median():.1f}')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранен: {save_path}")
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, features: list, title: str,
                            save_path: str = None, figsize: tuple = (12, 10)):
    ensure_output_dir()
    corr_matrix = df[features].corr()

    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, square=True, annot_kws={"size": 9},
                cbar_kws={"label": "Коэффициент корреляции"})
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Корреляционная матрица сохранена: {save_path}")
    plt.show()