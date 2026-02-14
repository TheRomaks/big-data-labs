import pandas as pd
import seaborn as sns
from pathlib import Path

def load_mpg_dataset() -> pd.DataFrame:
    df = sns.load_dataset('mpg')
    print(f"Загружен датасет 'mpg': {df.shape[0]:,} строк × {df.shape[1]} столбцов")
    return df


def load_sales_dataset(filepath: str = 'data/final_cleaned_data.csv') -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path.absolute()}")

    df = pd.read_csv(filepath)
    print(f"Загружен датасет продаж: {df.shape[0]:,} строк × {df.shape[1]} столбцов")
    return df


def convert_date_columns(df: pd.DataFrame, date_columns: list) -> pd.DataFrame:
    df = df.copy()
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            nulls = df[col].isnull().sum()
            if nulls > 0:
                print(f"Столбец '{col}': {nulls} некорректных дат преобразовано в NaT")
    return df