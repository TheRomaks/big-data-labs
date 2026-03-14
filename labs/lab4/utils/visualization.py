import matplotlib.pyplot as plt
import pandas as pd


def plot_knn_k_selection(k_range, k_scores, optimal_k, save_path='utils/data/knn_k_selection.png', show=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, k_scores, marker='o', linewidth=2)
    ax.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
    ax.set_xlabel('Количество соседей (k)')
    ax.set_ylabel('R² Score')
    ax.set_title('Зависимость качества модели от количества соседей')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig


def print_comparison_table(metrics_dict):
    comparison_df = pd.DataFrame({
        'Метрика': ['MAE', 'MSE', 'RMSE', 'MAPE (%)', 'R²'],
        'Линейная регрессия': [
            metrics_dict['Linear Regression']['MAE'],
            metrics_dict['Linear Regression']['MSE'],
            metrics_dict['Linear Regression']['RMSE'],
            metrics_dict['Linear Regression']['MAPE'],
            metrics_dict['Linear Regression']['R2']
        ],
        'Lasso': [
            metrics_dict['Lasso']['MAE'],
            metrics_dict['Lasso']['MSE'],
            metrics_dict['Lasso']['RMSE'],
            metrics_dict['Lasso']['MAPE'],
            metrics_dict['Lasso']['R2']
        ],
        'ElasticNet': [
            metrics_dict['ElasticNet']['MAE'],
            metrics_dict['ElasticNet']['MSE'],
            metrics_dict['ElasticNet']['RMSE'],
            metrics_dict['ElasticNet']['MAPE'],
            metrics_dict['ElasticNet']['R2']
        ],
        'KNN Регрессия': [
            metrics_dict['KNN']['MAE'],
            metrics_dict['KNN']['MSE'],
            metrics_dict['KNN']['RMSE'],
            metrics_dict['KNN']['MAPE'],
            metrics_dict['KNN']['R2']
        ]
    })

    print("\nСравнительная таблица метрик")

    float_format = '{:,.4f}'.format
    styled_df = comparison_df.copy()

    for col in comparison_df.columns[1:]:
        styled_df[col] = comparison_df[col].apply(float_format)

    print(styled_df.to_string(index=False))

    return comparison_df


def plot_model_comparison(metrics_dict, save_path='utils/data/model_comparison.png', show=False):
    models = ['Linear Regression', 'Lasso', 'ElasticNet', 'KNN']
    model_labels = ['Линейная', 'Lasso', 'ElasticNet', 'KNN']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    metrics_to_plot = ['MAE', 'MSE', 'RMSE', 'R2']
    n_metrics = len(metrics_to_plot)

    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = [metrics_dict[m][metric] for m in models]

        if metric in ['MAE', 'MSE', 'RMSE']:
            bars = ax.bar(model_labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max(values) * 0.02),
                        f'{val:,.0f}', ha='center', va='bottom', fontsize=10)
            ax.set_ylabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(f'{metric}', fontsize=12, fontweight='bold', pad=10)
        else:
            bars = ax.bar(model_labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)
            ax.set_ylabel('R² Score', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1.05)
            ax.set_title(f'R²', fontsize=12, fontweight='bold', pad=10)

        ax.set_xlabel('Модель', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Сравнение всех моделей по всем метрикам', fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def print_best_model(models):
    best_name = max(models.keys(), key=lambda k: models[k]['metrics']['R2'])
    best_model = models[best_name]

    model_names_rus = {
        'Linear Regression': 'Линейная регрессия',
        'Lasso': 'Lasso Регрессия',
        'ElasticNet': 'ElasticNet Регрессия',
        'KNN': 'KNN Регрессия'
    }

    print(f"\n✓ Лучшая модель: {model_names_rus[best_name]}")
    print(f"  R² Score: {best_model['metrics']['R2']:.4f}")
    print(f"  RMSE: {best_model['metrics']['RMSE']:.4f}")
    print(f"  MAE: {best_model['metrics']['MAE']:.4f}")

    return best_name, best_model

def get_comparison_df(metrics_dict):
    return pd.DataFrame(metrics_dict).T.reset_index().rename(columns={'index': 'Модель'})