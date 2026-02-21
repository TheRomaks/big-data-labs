import os
import pandas as pd
from utils.data_loading import load_data, initial_inspection, get_numeric_features
from utils.eda import plot_feature_distributions, plot_correlation_matrix, \
    compute_summary_stats
from utils.preprocessing import handle_missing_values, detect_outliers, prepare_data
from utils.evaluation import evaluate_model
from utils.model_plots_compare import plot_model_comparison
from utils.trainers import get_models

os.makedirs('utils/data/wine', exist_ok=True)

df = load_data('utils/data/WineQT.csv')
initial_inspection(df)

features = ['fixed acidity', 'volatile acidity', 'citric acid', 'alcohol', 'residual sugar']
print(f"Выбранные признаки ({len(features)}):")
for i, feat in enumerate(features, 1):
    corr = df[feat].corr(df['quality'])
    print(f"  {i}. {feat:25s} (корреляция с качеством: {corr:+.3f})")

df_reduced = df[features + ['quality']].copy()
print(f"\nФорма датасета после фильтрации: {df_reduced.shape}")

df_reduced['quality_binary'] = (df_reduced['quality'] >= 6).astype(int)

print("\nРаспределение исходных классов (качество 3-8):")
quality_dist = df['quality'].value_counts().sort_index()
for q, count in quality_dist.items():
    pct = count / len(df) * 100
    print(f"  Качество {q}: {count:3d} ({pct:5.2f}%)")

print("\nРаспределение бинарных классов:")
binary_dist = df_reduced['quality_binary'].value_counts().sort_index()

numeric_cols = get_numeric_features(df_reduced, target_col='quality_binary')
compute_summary_stats(df_reduced, numeric_cols)

plot_feature_distributions(
    df_reduced,
    numeric_cols,
    'utils/data/wine/feature_distributions.png'
)

plot_correlation_matrix(
    df_reduced,
    numeric_cols,
    'utils/data/wine/correlation_matrix.png',
    target_col='quality_binary'
)

df_clean = handle_missing_values(df_reduced, numeric_cols)
detect_outliers(df_clean, numeric_cols)

X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(
    df_clean,
    target_col='quality_binary'
)

results = {}
models = get_models()

for name, model in models.items():
    metrics, _, _, _ = evaluate_model(
        model,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        name,
        'utils/data/wine',
        is_multiclass = True
    )

    results[name] = metrics
    print(pd.DataFrame([metrics]).round(4))

results_df = pd.DataFrame(results).T.round(4)
print("\nСводная таблица метрик:")
print(results_df)

plot_model_comparison(
    results_df,
    'utils/data/wine/models_comparison.png'
)

best_model = results_df['F1-score'].idxmax()
best_score = results_df.loc[best_model, 'F1-score']
print(f"\n Лучшая модель по F1-score: {best_model} ({best_score:.4f})")
