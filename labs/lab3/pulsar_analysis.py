import pandas as pd
from utils.data_loading import initial_inspection, load_data, get_numeric_features
from utils.eda import plot_class_distribution, plot_feature_distributions, plot_correlation_matrix, \
    compute_summary_stats
from utils.evaluation import evaluate_model
from utils.model_plots_compare import plot_model_comparison
from utils.preprocessing import  handle_missing_values, \
    detect_outliers, prepare_data
from utils.trainers import get_models
from utils.overfitting_diagnostics import (
    train_test_gap,
    cross_validation_scores,
    plot_learning_curve
)

df = load_data('utils/data/train.csv')
df = df.drop(columns=['id', 'Id'], errors='ignore')
initial_inspection(df)

numeric_cols = get_numeric_features(df)
compute_summary_stats(df, numeric_cols)
plot_class_distribution(df,'utils/data/target_distribution.png')

plot_feature_distributions(df, numeric_cols, 'utils/data/feature_distributions.png')
plot_correlation_matrix(df, numeric_cols, 'utils/data/correlation_matrix.png')

df = handle_missing_values(df, numeric_cols)
detect_outliers(df, numeric_cols)

X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(df)

results = {}
models = get_models()

for name, model in models.items():
        print(f"\n{name}")
        metrics, _, _, _ = evaluate_model(
            model, X_train_scaled, X_test_scaled,
            y_train, y_test, name, 'utils/data'
        )
        results[name] = metrics
        print(pd.DataFrame([metrics]).round(3))

        print("\n--- Проверка train/test gap ---")
        gap_metrics = train_test_gap(
            model,
            X_train_scaled, X_test_scaled,
            y_train, y_test
        )
        print(pd.DataFrame([gap_metrics]).round(3))

        print("\n--- Cross-validation ---")
        cv_metrics = cross_validation_scores(
            model,
            X_train_scaled,
            y_train
        )
        print(pd.DataFrame([cv_metrics]).round(3))

        print("\n--- Learning curve ---")
        plot_learning_curve(
            model,
            X_train_scaled,
            y_train,
            name
        )

results_df = pd.DataFrame(results).T.round(3)
print(results_df)
plot_model_comparison(results_df, 'utils/data/models_comparison.png')

best_model = results_df['F1-score'].idxmax()
print(f"\nЛучшая модель по F1-score: {best_model} ({results_df.loc[best_model, 'F1-score']:.3f})")