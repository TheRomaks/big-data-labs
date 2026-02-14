import pandas as pd
from scipy import stats
from utils.data_loader import load_sales_dataset, convert_date_columns
from utils.feature_engineering import calculate_order_metrics
from utils.eda_analyzer import get_column_types, calculate_numeric_stats, calculate_categorical_stats
from utils.hypothesis import compare_two_groups, chi_square_test
from utils.encoding_utils import label_encode_column
from utils.visualizer import plot_correlation_matrix

df = load_sales_dataset()
df = convert_date_columns(df, ['order_date', 'required_date', 'shipped_date'])
df = calculate_order_metrics(df)

col_types = get_column_types(df)
print(f"\nЧисловые признаки ({len(col_types['numeric'])}): {col_types['numeric']}")
print(f"Категориальные признаки ({len(col_types['categorical'])}): {col_types['categorical']}")
print(f"Временные признаки ({len(col_types['datetime'])}): {col_types['datetime']}")

num_stats = calculate_numeric_stats(df, col_types['numeric'])
print("\nСтатистика числовых признаков:")
print(num_stats.to_string())

cat_stats = calculate_categorical_stats(df, col_types['categorical'])
print("\nСтатистика категориальных признаков:")
print(cat_stats.to_string())

print("\nГипотеза 1: Сравнение цен брендов Trek и Electra")
trek_prices = df[df['brand_name'] == 'Trek']['product_price'].dropna()
electra_prices = df[df['brand_name'] == 'Electra']['product_price'].dropna()

print(f"Trek: Среднее: ${trek_prices.mean():,.2f} | Медиана: ${trek_prices.median():,.2f}")
print(f"Electra: Среднее: ${electra_prices.mean():,.2f} | Медиана: ${electra_prices.median():,.2f}")

result = compare_two_groups(trek_prices, electra_prices, 'Trek', 'Electra')
_, p_levene = stats.levene(trek_prices, electra_prices)

print(f"\nWelch's t-test: t={result['statistic']:.4f}, p-value={result['p_value']:.6f}")
print(f"Дисперсии {'равны' if p_levene > 0.05 else 'не равны'} (Levene p={p_levene:.4f})")

if result['significant']:
    print(f"Средняя цена Trek (${result['mean_group1']:,.2f}) значительно отличается от Electra (${result['mean_group2']:,.2f})")
else:
    print("Нет статистически значимых различий в средних ценах")

print("\nГипотеза 2: Распределение категорий по штатам")
top_states = df['store_state'].value_counts().nlargest(2).index.tolist()
state1, state2 = top_states[0], top_states[1]

contingency = pd.crosstab(
    df[df['store_state'].isin([state1, state2])]['store_state'],
    df[df['store_state'].isin([state1, state2])]['category_name']
)
print("\nТаблица сопряженности:")
print(contingency.to_string())

chi2_result = chi_square_test(contingency)
print(f"\nМинимальная ожидаемая частота: {chi2_result['expected_min']:.2f}")
print(f"χ²-статистика: {chi2_result['chi2']:.4f}, p-value: {chi2_result['p_value']:.6f}")

if chi2_result['significant']:
    print(f"Распределение категорий значимо различается между {state1} и {state2}")

    for state in [state1, state2]:
        dist = df[df['store_state'] == state]['category_name'].value_counts(normalize=True) * 100
        print(f"\nТоп-3 категории в {state}:")
        for cat, pct in dist.head(3).items():
            print(f"  • {cat}: {pct:.1f}%")
else:
    print("Распределение категорий не различается статистически значимо")

print("\nКорреляционная матрица")
df_corr = df.copy()

for col in ['brand_name', 'category_name', 'store_state']:
    if col in df_corr.columns:
        encoded, _, _ = label_encode_column(df_corr, col)
        df_corr[f'{col}_encoded'] = encoded

corr_features = [
    'quantity', 'discount', 'stock_quantity', 'order_delay_days',
    'is_late', 'total_price', 'product_price',
    'brand_name_encoded', 'category_name_encoded', 'store_state_encoded'
]
corr_features = [col for col in corr_features if col in df_corr.columns]

plot_correlation_matrix(
    df_corr.dropna(), corr_features,
    'Матрица корреляции признаков (Продажи)',
    'data/correlation_matrix_sales.png'
)