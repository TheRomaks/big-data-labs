import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_knn_k_selection(k_range, k_scores, optimal_k, save_path='utils/data/knn_k_selection.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, k_scores, marker='o', linewidth=2)
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
    plt.xlabel('Количество соседей (k)')
    plt.ylabel('R² Score')
    plt.title('Зависимость качества модели от количества соседей')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150)
    plt.show()


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

    print("Сравнительная таблица метрик")

    # Форматирование вывода для читаемости (убираем научную нотацию)
    float_format = '{:,.2f}'.format
    styled_df = comparison_df.copy()

    # Применяем форматирование к числовым колонкам
    for col in comparison_df.columns[1:]:
        styled_df[col] = comparison_df[col].apply(float_format)

    print(styled_df.to_string(index=False))

    return comparison_df


def plot_model_comparison(metrics_dict, save_path='utils/data/car_model_comparison.png'):
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    metrics_to_plot = ['MAE', 'RMSE', 'R2']
    x = np.arange(len(metrics_to_plot))
    width = 0.2

    models = ['Linear Regression', 'Lasso', 'ElasticNet', 'KNN']
    model_labels = ['Линейная', 'Lasso', 'ElasticNet', 'KNN']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (model_name, color, label) in enumerate(zip(models, colors, model_labels)):
        mae = metrics_dict[model_name]['MAE']
        rmse = metrics_dict[model_name]['RMSE']
        r2 = metrics_dict[model_name]['R2']

        max_error = max([metrics_dict[m]['RMSE'] for m in models])
        values = [mae / max_error, rmse / max_error, r2]

        ax.bar(x + i * width - width * 1.5, values, width,
               label=label, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Метрика', fontsize=12, fontweight='bold')
    ax.set_ylabel('Нормализованное значение', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, fontsize=11)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Сравнение метрик качества всех моделей (нормализовано)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


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

    print("Сравнительная таблица метрик")

    float_format = '{:,.4f}'.format
    styled_df = comparison_df.copy()

    for col in comparison_df.columns[1:]:
        styled_df[col] = comparison_df[col].apply(float_format)

    print(styled_df.to_string(index=False))

    return comparison_df

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