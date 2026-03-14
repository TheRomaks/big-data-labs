import warnings
from labs.lab3.utils.data_loading import load_data, initial_inspection, get_numeric_features
from labs.lab3.utils.eda import compute_summary_stats, plot_feature_distributions, plot_correlation_matrix
from utils.models import train_all_models, save_models
from utils.preprocessing import handle_missing_values, remove_outliers_iqr, prepare_data
from utils.hypotheses import test_hypotheses, print_hypothesis_summary
from utils.visualization import (
    plot_knn_k_selection,
    plot_model_comparison,
    print_comparison_table,
    print_best_model,
    get_comparison_df
)

warnings.filterwarnings('ignore')


def diabet_analysis(filepath="utils/data/diabetes.csv", show_plots=False):
    df = load_data(filepath)
    initial_inspection(df)

    numeric_cols = get_numeric_features(df, target_col='BMI')

    stats_df = compute_summary_stats(df, numeric_cols)

    print("\nАнализ целевой переменной BMI:")
    bmi_mode = df['BMI'].mode()[0]
    bmi_counts = df['BMI'].value_counts()
    print(f"Мода: {bmi_mode}")
    print(f"Распределение:\n{bmi_counts}")

    fig_dist = plot_feature_distributions(df, numeric_cols, 'utils/data/diabetes_distribution.png', show=show_plots)
    fig_corr = plot_correlation_matrix(df, numeric_cols, 'utils/data/diabetes_correlation.png', target_col='BMI',
                                       show=show_plots)

    hypotheses_list = [
        ("Уровень глюкозы влияет на BMI", 'Glucose', 0.3),
        ("Возраст влияет на BMI", 'Age', 0.2),
        ("Количество беременностей влияет на BMI", 'Pregnancies', 0.2),
        ("Давление влияет на BMI", 'BloodPressure', 0.2)
    ]

    hypothesis_results = test_hypotheses(df, hypotheses_list, target_column='BMI')
    print_hypothesis_summary(hypothesis_results)

    df_processed, missing_stats = handle_missing_values(
        df,
        zero_columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin'],
        handle_zeros=True,
        handle_nan=True,
        strategy='median'
    )

    feature_cols = [col for col in numeric_cols if col != 'BMI']
    df_processed, outlier_stats = remove_outliers_iqr(df_processed, feature_cols)

    data = prepare_data(df_processed, target_column='BMI', stratify=False)

    models = train_all_models(data['X_train'], data['y_train'], data['X_test'], data['y_test'])

    fig_knn = None
    if 'KNN' in models and 'k_range' in models['KNN']:
        fig_knn = plot_knn_k_selection(
            models['KNN']['k_range'],
            models['KNN']['k_scores'],
            models['KNN']['optimal_k'],
            save_path='utils/data/diabet_knn_k_selection.png',
            show=show_plots
        )

    metrics_dict = {name: model['metrics'] for name, model in models.items()}
    print_comparison_table(metrics_dict)

    metrics_df = get_comparison_df(metrics_dict)
    fig_comp = plot_model_comparison(metrics_dict, save_path='utils/data/diabet_model_comparison.png', show=show_plots)

    save_models(models, data['scaler'], "diabet")
    best_name, best_model = print_best_model(models)

    return {
        'stats_df': stats_df,
        'fig_dist': fig_dist,
        'fig_corr': fig_corr,
        'fig_comp': fig_comp,
        'fig_knn': fig_knn,
        'metrics_df': metrics_df,
        'best_model_name': best_name
    }


if __name__ == "__main__":
    diabet_analysis(show_plots=True)