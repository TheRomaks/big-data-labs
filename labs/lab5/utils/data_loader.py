import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np


def load_and_preprocess_data(choice, session_state=None):
    if "Penguins" in choice:
        df = pd.read_csv("utils/data/penguins.csv")
        target_col = "sex"
        dataset_name = "Penguins"
    else:
        df = pd.read_csv("utils/data/final_cleaned_data.csv")
        target_col = "brand_name"
        dataset_name = "Bike Sales"

    df = df.dropna()

    numerical_for_outliers = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for col in numerical_for_outliers:
        if col in df.columns and col != target_col:
            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)
            df[col] = np.clip(df[col], q01, q99)

    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)

    label_encoder = None
    if df[target_col].dtype == 'object':
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
        label_encoder = le

    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    if session_state is not None:
        if label_encoder is not None:
            session_state['label_encoder'] = label_encoder
        else:
            session_state.pop('label_encoder', None)
        session_state['scaler'] = scaler
        session_state['numerical_cols'] = numerical_cols

    return df, numerical_cols, target_col, dataset_name