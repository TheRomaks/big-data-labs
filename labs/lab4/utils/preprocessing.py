import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Dict


def exploratory_analysis(df: pd.DataFrame) -> Dict:
    stats = {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    print(f"\nЧисловые переменные ({len(numeric_cols)}): {numeric_cols}")
    print(f"Категориальные переменные ({len(categorical_cols)}): {categorical_cols}")

    print("Статистика по числовым переменным:")
    print(df[numeric_cols].describe().round(2))

    print("Статистика по категориальным переменным:")
    for col in categorical_cols:
        mode_val = df[col].mode()[0]
        mode_count = (df[col] == mode_val).sum()
        unique_count = df[col].nunique()
        print(f"\n{col}:")
        print(f"  Уникальных значений: {unique_count}")
        print(f"  Мода: '{mode_val}' (встречается {mode_count} раз)")
        stats[col] = {'mode': mode_val, 'count': mode_count, 'unique': unique_count}

    return stats


def encode_categorical_variables(df: pd.DataFrame,
                                 categorical_cols: List[str],
                                 encoding_method: str = 'label') -> Tuple[pd.DataFrame, Dict]:
    df_processed = df.copy()
    encoders = {}

    for col in categorical_cols:
        if col in df_processed.columns:
            if encoding_method == 'label':
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                encoders[col] = le
            elif encoding_method == 'onehot':
                dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
                df_processed = pd.concat([df_processed, dummies], axis=1)
                df_processed.drop(col, axis=1, inplace=True)

    return df_processed, encoders

def handle_missing_values(df: pd.DataFrame,
                          zero_columns: List[str] = None,
                          handle_zeros: bool = True,
                          handle_nan: bool = True,
                          strategy: str = 'median') -> Tuple[pd.DataFrame, Dict]:
    df_processed = df.copy()
    stats = {
        'columns': [],
        'zero_counts': [],
        'zero_percentages': [],
        'nan_counts': [],
        'nan_percentages': [],
        'replacement_values': []
    }

    if zero_columns is None:
        zero_columns = []

    print(f"\nКолонки для обработки нулей: {zero_columns}")

    for col in zero_columns:
        if col in df_processed.columns:
            zero_count = (df_processed[col] == 0).sum()
            zero_percentage = zero_count / len(df_processed) * 100

            stats['columns'].append(col)
            stats['zero_counts'].append(zero_count)
            stats['zero_percentages'].append(zero_percentage)

            print(f"{col}: {zero_count} нулевых значений ({zero_percentage:.2f}%)")

            if handle_zeros and zero_count > 0:
                if strategy == 'median':
                    replacement_val = df_processed[df_processed[col] != 0][col].median()
                elif strategy == 'mean':
                    replacement_val = df_processed[df_processed[col] != 0][col].mean()
                else:
                    replacement_val = 0

                df_processed[col] = df_processed[col].replace(0, replacement_val)
                stats['replacement_values'].append(replacement_val)
            else:
                stats['replacement_values'].append(None)

    if handle_nan:
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col not in zero_columns and col in df_processed.columns:
                nan_count = df_processed[col].isna().sum()
                if nan_count > 0:
                    nan_percentage = nan_count / len(df_processed) * 100
                    print(f"{col}: {nan_count} пропущенных значений ({nan_percentage:.2f}%)")

                    if strategy == 'median':
                        replacement_val = df_processed[col].median()
                    elif strategy == 'mean':
                        replacement_val = df_processed[col].mean()
                    else:
                        replacement_val = 0

                    df_processed[col] = df_processed[col].fillna(replacement_val)

    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        if col in df_processed.columns:
            nan_count = df_processed[col].isna().sum()
            if nan_count > 0:
                nan_percentage = nan_count / len(df_processed) * 100
                print(f"{col}: {nan_count} пропущенных значений ({nan_percentage:.2f}%)")
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'Unknown')

    return df_processed, stats

def remove_outliers_iqr(df: pd.DataFrame,
                        columns: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
    df_processed = df.copy()

    if columns is None:
        columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()

    stats = {
        'columns': [],
        'outlier_counts': [],
        'lower_bounds': [],
        'upper_bounds': []
    }
    for col in columns:
        if col in df_processed.columns:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((df_processed[col] < lower_bound) |
                        (df_processed[col] > upper_bound)).sum()

            stats['columns'].append(col)
            stats['outlier_counts'].append(outliers)
            stats['lower_bounds'].append(lower_bound)
            stats['upper_bounds'].append(upper_bound)

            print(f"{col}: найдено {outliers} выбросов "
                  f"(границы: [{lower_bound:.2f}, {upper_bound:.2f}])")

            # Ограничение выбросов вместо удаления (clip)
            df_processed[col] = df_processed[col].clip(lower=lower_bound,
                                                       upper=upper_bound)

    return df_processed, stats


def prepare_data(df: pd.DataFrame,
                 target_column: str = 'Outcome',
                 test_size: float = 0.2,
                 random_state: int = 42,
                 stratify: bool = True) -> Dict:
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=test_size,
            random_state=random_state
        )

    print(f"\nОбучающая выборка: {X_train.shape[0]} образцов")
    print(f"Тестовая выборка: {X_test.shape[0]} образцов")
    data_dict = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_names,
        'target_column': target_column
    }

    return data_dict


def calculate_metrics(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      model_name: str = "") -> Dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # MAPE (с обработкой нулей)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0

    if model_name:
        print(f"\n{model_name}:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²:   {r2:.4f}")

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }