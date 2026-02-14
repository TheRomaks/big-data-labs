import pandas as pd

def calculate_order_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'shipped_date' in df.columns and 'required_date' in df.columns:
        df['order_delay_days'] = (df['shipped_date'] - df['required_date']).dt.days
        df['is_late'] = (df['order_delay_days'] > 0).astype(int)
        print("Добавлены признаки: order_delay_days, is_late")

    required_cols = ['product_price', 'quantity', 'discount']
    if all(col in df.columns for col in required_cols):
        df['total_price'] = df['product_price'] * df['quantity'] * (1 - df['discount'])
        print("Добавлен признак: total_price")

    return df