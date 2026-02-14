from utils.data_loader import load_mpg_dataset
from utils.eda_analyzer import get_column_types, calculate_numeric_stats, calculate_categorical_stats
from utils.hypothesis import compare_two_groups, test_correlation
from utils.encoding_utils import label_encode_column
from utils.visualizer import plot_distribution, plot_correlation_matrix

df = load_mpg_dataset()

col_types = get_column_types(df)
print(f"\nЧисловые признаки ({len(col_types['numeric'])}): {col_types['numeric']}")
print(f"Категориальные признаки ({len(col_types['categorical'])}): {col_types['categorical']}")

num_stats = calculate_numeric_stats(df, col_types['numeric'])
print("\nСтатистика числовых признаков:")
print(num_stats.to_string())

cat_stats = calculate_categorical_stats(df, col_types['categorical'])
print("\nСтатистика категориальных признаков:")
print(cat_stats.to_string())

plot_distribution(
    df, 'mpg',
    'Распределение расхода топлива (mpg)',
    'Миль на галлон (mpg)', 'Частота',
    'data/mpg_distribution.png'
)

print("\nГипотеза 1: Сравнение расхода топлива США и Японии")
usa_mpg = df[df['origin'] == 'usa']['mpg'].dropna()
japan_mpg = df[df['origin'] == 'japan']['mpg'].dropna()

result = compare_two_groups(usa_mpg, japan_mpg, 'США', 'Япония')
print(f"Нормальность: США (p={result['normality']['США']['p_value']:.4f}), "
      f"Япония (p={result['normality']['Япония']['p_value']:.4f})")
print(f"Тест: {result['test']}, statistic={result['statistic']:.4f}, p-value={result['p_value']:.4f}")

if result['significant']:
    print(f"Средний mpg в США ({result['mean_group1']:.2f}) значимо отличается от Японии ({result['mean_group2']:.2f})")
else:
    print("Статистически значимых различий в mpg между США и Японией нет")

print("\nГипотеза 2: Зависимость количества цилиндров и ускорения")
cylinders = df['cylinders'].dropna()
acceleration = df['acceleration'].dropna()

corr_result = test_correlation(cylinders, acceleration)
print(f"Метод: корреляция {corr_result['method']} ({corr_result['corr_type']})")
print(f"Коэффициент: {corr_result['correlation']:.4f}, p-value: {corr_result['p_value']:.4f}")

if corr_result['significant']:
    print(f"Обнаружена {corr_result['strength']} {corr_result['direction']} корреляция (ρ = {corr_result['correlation']:.2f})")
else:
    print("Нет статистически значимой корреляции")

print("\nКодирование и корреляционная матрица")
df_encoded = df.copy()
encoded, le, mapping = label_encode_column(df_encoded, 'origin')
df_encoded['origin_encoded'] = encoded
print(f"Label Encoding для 'origin': {mapping}")

numeric_df = df[col_types['numeric']].copy()
numeric_df['origin_num'] = df['origin'].map({'usa': 0, 'europe': 1, 'japan': 2})

plot_correlation_matrix(
    numeric_df, numeric_df.columns.tolist(),
    'Корреляционная матрица признаков (MPG)',
    'data/correlation_matrix_mpg.png',
    figsize=(10, 8)
)