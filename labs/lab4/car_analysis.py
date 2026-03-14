import warnings
import pandas as pd
from labs.lab3.utils.data_loading import load_data, initial_inspection, get_numeric_features
from labs.lab3.utils.eda import plot_feature_distributions, plot_correlation_matrix, compute_summary_stats
from labs.lab4.utils.preprocessing import exploratory_analysis
from utils.hypotheses import test_hypotheses, print_hypothesis_summary
from utils.models import train_all_models, save_models
from utils.preprocessing import (
    handle_missing_values,
    remove_outliers_iqr,
    prepare_data,
    encode_categorical_variables,
)
from utils.visualization import (
    plot_knn_k_selection,
    plot_model_comparison,
    print_comparison_table,
    print_best_model,
    get_comparison_df
)

warnings.filterwarnings('ignore')


def preprocess_car_data(df: pd.DataFrame) -> pd.DataFrame:
    df_processed = df.copy()
    df_processed = df_processed.dropna()
    df_processed['CarName'] = df_processed['CarName'].str.split().str[0]
    return df_processed


def car_price_analysis(filepath: str = 'utils/data/CarPrice_Assignment.csv', show_plots=False):
    df = load_data(filepath)
    initial_inspection(df)

    numeric_cols = get_numeric_features(df)

    stats_df = compute_summary_stats(df, numeric_cols)

    exploratory_analysis(df)
    df_processed = preprocess_car_data(df)

    fig_dist = plot_feature_distributions(df, numeric_cols, 'utils/data/car_distribution.png', show=show_plots)
    fig_corr = plot_correlation_matrix(df, numeric_cols, 'utils/data/car_correlation.png', target_col='price',
                                       show=show_plots)

    categorical_cols = ['fueltype', 'aspiration', 'doornumber', 'carbody',
                        'drivewheel', 'enginelocation', 'enginetype',
                        'cylindernumber', 'fuelsystem']

    df_processed, encoders = encode_categorical_variables(df_processed, categorical_cols, encoding_method='label')

    df_processed, missing_stats = handle_missing_values(
        df_processed, zero_columns=[], handle_zeros=False, handle_nan=True, strategy='median'
    )

    feature_cols = [col for col in df_processed.columns if col not in ['car_ID', 'price', 'CarName']]
    df_processed, outlier_stats = remove_outliers_iqr(df_processed, feature_cols)

    target_column = 'price'
    hypotheses_list = [
        ("Мощность двигателя", 'horsepower', 0.5),
        ("Размер двигателя", 'enginesize', 0.5),
        ("Вес автомобиля", 'curbweight', 0.5),
        ("Расход в городе", 'citympg', -0.3),
        ("Длина автомобиля", 'carlength', 0.3)
    ]

    hypothesis_results = test_hypotheses(df_processed, hypotheses_list, target_column)
    print_hypothesis_summary(hypothesis_results)

    data = prepare_data(df_processed[feature_cols + [target_column]], target_column=target_column, stratify=False)

    models = train_all_models(data['X_train'], data['y_train'], data['X_test'], data['y_test'])

    fig_knn = None
    if 'KNN' in models and 'k_range' in models['KNN']:
        fig_knn = plot_knn_k_selection(
            models['KNN']['k_range'],
            models['KNN']['k_scores'],
            models['KNN']['optimal_k'],
            save_path='utils/data/car_knn_k_selection.png',
            show=show_plots
        )

    metrics_dict = {name: model['metrics'] for name, model in models.items()}
    print_comparison_table(metrics_dict)

    metrics_df = get_comparison_df(metrics_dict)
    fig_comp = plot_model_comparison(metrics_dict, save_path='utils/data/car_model_comparison.png', show=show_plots)

    save_models(models, data['scaler'], "car")
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
    car_price_analysis(show_plots=True)