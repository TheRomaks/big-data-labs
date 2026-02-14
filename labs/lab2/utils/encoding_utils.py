import pandas as pd
from sklearn.preprocessing import LabelEncoder

def label_encode_column(df: pd.DataFrame, column: str) -> tuple:
    le = LabelEncoder()
    encoded = le.fit_transform(df[column].astype(str))
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    return encoded, le, mapping

def one_hot_encode_column(df: pd.DataFrame, column: str, drop_first: bool = True) -> pd.DataFrame:
    return pd.get_dummies(df, columns=[column], prefix=column, drop_first=drop_first)