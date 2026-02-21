import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from typing import Tuple, List

def handle_missing_values(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    if df.isnull().sum().sum() > 0:
        print("Обнаружены пропуски. Выполняется заполнение медианой...")
        df.fillna(df[numeric_cols].median(), inplace=True)
    else:
        print("Пропусков в данных не обнаружено.")
    return df

def detect_outliers(df: pd.DataFrame, numeric_cols: List[str]) -> dict:
    outliers_info = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outliers_info[col] = outliers
        print(f"{col:25s}: {outliers:4d} выбросов ({outliers / len(df) * 100:.2f}%)")
    return outliers_info

def prepare_data(
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        target_col: str = 'Class'
) -> Tuple:
    X = df.drop(['id', 'Id','quality', target_col], axis=1, errors='ignore')
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nФорма X: {X.shape}, форма y: {y.shape}")
    print(f"Обучающая выборка: {X_train.shape[0]} объектов")
    print(f"Тестовая выборка:  {X_test.shape[0]} объектов")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
