from typing import List
import numpy as np
import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(f"Память: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB\n")
    return df

def initial_inspection(df: pd.DataFrame) -> dict:
    info = {
        'info': df.info(),
        'missing': df.isnull().sum(),
        'shape': df.shape
    }
    print("\nПропуски в столбцах:")
    print(df.isnull().sum())
    return info

def get_numeric_features(df: pd.DataFrame, target_col: str = 'Class') -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    return numeric_cols