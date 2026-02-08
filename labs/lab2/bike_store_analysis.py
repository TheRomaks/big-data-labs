import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder

sns.set_theme(style="whitegrid")

df = pd.read_csv('data/final_cleaned_data.csv')
print(f"Размер датасета: {df.shape[0]:,} строк × {df.shape[1]} столбцов\n")

date_cols = ['order_date', 'required_date', 'shipped_date']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

df['order_delay_days'] = (df['shipped_date'] - df['required_date']).dt.days
df['is_late'] = (df['order_delay_days'] > 0).astype(int)
df['total_price'] = df['product_price'] * df['quantity'] * (1 - df['discount'])

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

print(f"Числовые признаки ({len(numeric_cols)}): {numeric_cols}")
print(f"Категориальные признаки ({len(categorical_cols)}): {categorical_cols}")
print(f"Временные признаки ({len(datetime_cols)}): {datetime_cols}")

stats_numeric = pd.DataFrame()
for col in numeric_cols:
    stats_numeric.loc[col, 'Пропуски (%)'] = df[col].isnull().mean() * 100
    stats_numeric.loc[col, 'Пропуски '] = df[col].isnull().sum()
    stats_numeric.loc[col, 'Минимум'] = df[col].min()
    stats_numeric.loc[col, 'Максимум'] = df[col].max()
    stats_numeric.loc[col, 'Среднее'] = df[col].mean()
    stats_numeric.loc[col, 'Медиана'] = df[col].median()
    stats_numeric.loc[col, 'Дисперсия'] = df[col].var()
    stats_numeric.loc[col, 'Квантиль 0.1'] = df[col].quantile(0.1)
    stats_numeric.loc[col, 'Квартиль Q1 (0.25)'] = df[col].quantile(0.25)
    stats_numeric.loc[col, 'Квартиль Q3 (0.75)'] = df[col].quantile(0.75)
    stats_numeric.loc[col, 'Квантиль 0.9'] = df[col].quantile(0.9)

stats_numeric = stats_numeric.round(2)
print(stats_numeric.to_string())

stats_categorical = pd.DataFrame()
for col in categorical_cols:
    stats_categorical.loc[col, 'Пропуски (%)'] = df[col].isnull().mean() * 100
    stats_categorical.loc[col, 'Пропуски '] = df[col].isnull().sum()
    stats_categorical.loc[col, 'Уникальных'] = df[col].nunique()
    stats_categorical.loc[col, 'Мода'] = df[col].mode()[0]
    stats_categorical.loc[col, 'Частота моды'] = df[col].value_counts().iloc[0]

print(stats_categorical.to_string())

#гипотеза 1: сравнение цен брендов Trek и Electra
trek_prices = df[df['brand_name'] == 'Trek']['product_price']
electra_prices = df[df['brand_name'] == 'Electra']['product_price']

print(f"Trek: Среднее: ${trek_prices.mean():,.2f} | Медиана: ${trek_prices.median():,.2f} | Std: ${trek_prices.std():,.2f}")
print(f"Electra: Среднее: ${electra_prices.mean():,.2f} | Медиана: ${electra_prices.median():,.2f} | Std: ${electra_prices.std():,.2f}")

# Проверка равенства дисперсий (тест Левена)
_, p_levene = stats.levene(trek_prices, electra_prices)

t_stat, p_value = stats.ttest_ind(trek_prices, electra_prices, equal_var=False, alternative='greater')

print(f"\nWelch's t-test:")
print(f"  t-статистика: {t_stat:.4f}")
print(f"  p-value: {p_value:.6f}")
print(f"  Дисперсии {'равны' if p_levene > 0.05 else 'не равны'} (p_levene = {p_levene:.4f})")

if p_value < 0.05:
    print(f"Вывод: Средняя цена велосипедов Trek {trek_prices.mean()} значительно отличается, чем у Electra {electra_prices.mean()}.")
else:
    print("Вывод: Нет статистически значимых различий в средних ценах между брендами.")

# Гипотеза 2: Распределение категорий по штатам
top_states = df['store_state'].value_counts().nlargest(2).index.tolist()
state1, state2 = top_states[0], top_states[1]

# Формирование таблицы сопряженности
contingency_table = pd.crosstab(
    df[df['store_state'].isin([state1, state2])]['store_state'],
    df[df['store_state'].isin([state1, state2])]['category_name']
)

print("\nТаблица сопряженности (частоты):")
print(contingency_table.to_string())

# Проверка условия применимости хи-квадрат
chi2_stat, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)
min_expected = expected.min()

print(f"\nМинимальная ожидаемая частота: {min_expected:.2f}")
print(f"Результаты χ²-теста:")
print(f"  χ²-статистика: {chi2_stat:.4f}")
print(f"  Степени свободы: {dof}")
print(f"  p-value: {p_value_chi2:.6f}")

if p_value_chi2 < 0.05:
    print(f"Вывод: Распределение категорий велосипедов статистически значимо различается между {state1} и {state2}.")

    # Анализ различий в распределениях
    dist_state1 = df[df['store_state'] == state1]['category_name'].value_counts(normalize=True) * 100
    dist_state2 = df[df['store_state'] == state2]['category_name'].value_counts(normalize=True) * 100

    print(f"\nТоп-3 категории в {state1}:")
    for cat, pct in dist_state1.head(3).items():
        print(f"  - {cat}: {pct:.1f}%")

    print(f"\nТоп-3 категории в {state2}:")
    for cat, pct in dist_state2.head(3).items():
        print(f"  - {cat}: {pct:.1f}%")
else:
    print("Вывод: Распределение категорий не различается статистически значимо между штатами.")

# Обоснование выбора целевой переменной
print("\nОбоснование выбора целевой переменной:")
print("- product_price — количественная переменная, подлежащая прогнозированию")
print("- Имеет бизнес-значимость: цена определяет выручку и маржинальность")
print("- Зависит от множества факторов: бренд, категория, остатки на складе, регион")

# Подготовка данных для корреляции
# Создаем копию с закодированными категориальными признаками
df_corr = df.copy()

# Кодирование только информативных категориальных признаков
categorical_for_corr = ['brand_name', 'category_name', 'store_state']
for col in categorical_for_corr:
    if col in df_corr.columns:
        le = LabelEncoder()
        df_corr[f'{col}_encoded'] = le.fit_transform(df_corr[col].astype(str))

# Список признаков для корреляции
corr_features = [
    'quantity', 'discount', 'order_status', 'stock_quantity',
    'order_delay_days', 'is_late', 'total_price',
    'brand_name_encoded', 'category_name_encoded', 'store_state_encoded'
]

# Фильтрация существующих столбцов
corr_features = [col for col in corr_features if col in df_corr.columns]

# Добавляем целевую переменную
corr_features.append('product_price')

# Удаляем строки с пропусками
df_corr_clean = df_corr[corr_features].dropna()

print(f"\nАнализ корреляции для {len(corr_features) - 1} признаков с целевой переменной 'product_price'")
print(f"Размер выборки после удаления пропусков: {df_corr_clean.shape[0]:,} наблюдений")

# Расчет корреляции Пирсона
corr_matrix = df_corr_clean.corr()

# Вывод корреляций с целевой переменной
print("\nКорреляция признаков с 'product_price' (Пирсон):")
target_corr = corr_matrix['product_price'].drop('product_price').sort_values(ascending=False)
for feature, corr_val in target_corr.items():
    direction = "↑ положительная" if corr_val > 0 else "↓ отрицательная"
    strength = "сильная" if abs(corr_val) > 0.5 else ("умеренная" if abs(corr_val) > 0.3 else "слабая")
    print(f"  {feature:25s}: {corr_val:6.3f} ({strength} {direction})")

# Визуализация матрицы корреляции
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, mask=mask, square=True, annot_kws={"size": 9},
            cbar_kws={"label": "Коэффициент корреляции Пирсона"})
plt.title('Матрица корреляции признаков', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✅ Матрица корреляции сохранена: correlation_matrix.png")

# Ключевые выводы
print("\nКлючевые выводы по корреляционному анализу:")
strong_corr = target_corr[abs(target_corr) > 0.3]
if not strong_corr.empty:
    print("\nСильные корреляции (|r| > 0.3):")
    for feature, corr_val in strong_corr.items():
        if corr_val > 0:
            print(f"- {feature}: {corr_val:.3f} → более высокие значения признака ассоциированы с ростом цены")
        else:
            print(f"- {feature}: {corr_val:.3f} → более высокие значения признака ассоциированы со снижением цены")

        # Предметная интерпретация
        if 'brand' in feature.lower():
            print("  Интерпретация: Бренд является ключевым фактором ценообразования (премиальные бренды = высокие цены)")
        elif 'stock' in feature.lower() or 'quantity' in feature.lower():
            print("  Интерпретация: Дорогие модели обычно представлены в меньших количествах на складе")
        elif 'category' in feature.lower():
            print("  Интерпретация: Категория продукта напрямую определяет ценовой сегмент (электровелосипеды > горные > городские)")

moderate_corr = target_corr[(abs(target_corr) <= 0.3) & (abs(target_corr) > 0.1)]
if not moderate_corr.empty:
    print("\nУмеренные корреляции (0.1 < |r| ≤ 0.3):")
    for feature, corr_val in moderate_corr.items():
        print(f"- {feature}: {corr_val:.3f}")