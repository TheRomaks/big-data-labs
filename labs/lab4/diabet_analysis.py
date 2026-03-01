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

)

warnings.filterwarnings('ignore')

def diabet_analysis():
    df = load_data("utils/data/diabetes.csv")
    initial_inspection(df)

    numeric_cols = get_numeric_features(df)
    compute_summary_stats(df, numeric_cols)

    print("\nАнализ целевой переменной (Outcome):")
    outcome_mode = df['Outcome'].mode()[0]
    outcome_counts = df['Outcome'].value_counts()
    print(f"Мода: {outcome_mode}")
    print(f"Распределение:\n{outcome_counts}")
    print(f"Мода встречается {outcome_counts[outcome_mode]} раз")

    plot_feature_distributions(df, numeric_cols, 'utils/data/diabetes_distribution.png')
    plot_correlation_matrix(df, numeric_cols, 'utils/data/diabetes_correlation.png')

    hypotheses_list = [
        ("Уровень глюкозы", 'Glucose', 0.3),
        ("BMI", 'BMI', 0.2),
        ("Возраст", 'Age', 0.2)
    ]

    hypothesis_results = test_hypotheses(df, hypotheses_list)
    print_hypothesis_summary(hypothesis_results)

    df_processed, missing_stats = handle_missing_values(
        df,
        zero_columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'],
        handle_zeros=True,
        handle_nan=True,
        strategy='median'
    )

    feature_cols = [col for col in numeric_cols if col != 'Outcome']
    df_processed, outlier_stats = remove_outliers_iqr(df_processed, feature_cols)

    data = prepare_data(df_processed, target_column='Outcome')

    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    scaler = data['scaler']
    feature_names = data['feature_names']

    print(f"\nОбучающая выборка: {X_train.shape[0]} образцов")
    print(f"Тестовая выборка: {X_test.shape[0]} образцов")
    print(f"Количество признаков: {len(feature_names)}")

    models = train_all_models(X_train, y_train, X_test, y_test)

    plot_knn_k_selection(
        models['KNN']['k_range'],
        models['KNN']['k_scores'],
        models['KNN']['optimal_k']
    )

    metrics_dict = {name: model['metrics'] for name, model in models.items()}
    print_comparison_table(metrics_dict)

    plot_model_comparison(metrics_dict)

    save_models(models, scaler,"diabet")

diabet_analysis()
