import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv('data/merged_sales_data.csv')

print("Исходная форма:", df.shape)
print("\nКолонки:", df.columns.tolist())


#константа
df = df.drop('staff_active', axis=1, errors='ignore')

#корреляция
cols_to_drop = [
    'order_id',
    'model_year',
    'customer_id',
    'staff_first_name',
    'staff_last_name',
    'store_city',
    'store_name',
    'customer_state'
]

df = df.drop(cols_to_drop, axis=1, errors='ignore')

#пропуски
if 'shipped_date' in df.columns:
    df['shipped_date'] = df['shipped_date'].fillna(df['shipped_date'].mode()[0])

#нули
if 'stock_quantity' in df.columns:
    median_stock = df['stock_quantity'].replace(0, pd.NA).median()
    df['stock_quantity'] = df['stock_quantity'].replace(0, median_stock)

df = df.drop_duplicates()

final_file = 'data/final_cleaned_data.csv'
df.to_csv(final_file, index=False)

profile = ProfileReport(df, title="Финальные чистые данные")
profile.to_file("data/final_report.html")

