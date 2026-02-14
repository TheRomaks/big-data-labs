import pandas as pd

def get_column_types(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime = df.select_dtypes(include=['datetime64']).columns.tolist()

    return {
        'numeric': numeric,
        'categorical': categorical,
        'datetime': datetime
    }

def calculate_numeric_stats(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    stats = pd.DataFrame()
    for col in columns:
        stats.loc[col, 'Пропуски (%)'] = df[col].isnull().mean() * 100
        stats.loc[col, 'Пропуски'] = df[col].isnull().sum()
        stats.loc[col, 'Минимум'] = df[col].min()
        stats.loc[col, 'Максимум'] = df[col].max()
        stats.loc[col, 'Среднее'] = df[col].mean()
        stats.loc[col, 'Медиана'] = df[col].median()
        stats.loc[col, 'Дисперсия'] = df[col].var()
        stats.loc[col, 'Квантиль 0.1'] = df[col].quantile(0.1)
        stats.loc[col, 'Q1 (0.25)'] = df[col].quantile(0.25)
        stats.loc[col, 'Q3 (0.75)'] = df[col].quantile(0.75)
        stats.loc[col, 'Квантиль 0.9'] = df[col].quantile(0.9)
    return stats.round(2)

def calculate_categorical_stats(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    stats = pd.DataFrame()
    for col in columns:
        stats.loc[col, 'Пропуски (%)'] = df[col].isnull().mean() * 100
        stats.loc[col, 'Пропуски'] = df[col].isnull().sum()
        stats.loc[col, 'Уникальных'] = df[col].nunique()
        mode = df[col].mode()
        stats.loc[col, 'Мода'] = mode[0] if not mode.empty else 'N/A'
        stats.loc[col, 'Частота моды'] = df[col].value_counts().iloc[0] if not mode.empty else 0
    return stats