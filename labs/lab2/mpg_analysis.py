import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder

sns.set_theme(style="whitegrid")

df = sns.load_dataset('mpg')
print(f"Размер датасета: {df.shape[0]:,} строк × {df.shape[1]} столбцов")

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nЧисловые признаки ({len(numeric_cols)}): {numeric_cols}")
print(f"Категориальные признаки ({len(categorical_cols)}): {categorical_cols}")

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

plt.figure(figsize=(10, 6))
sns.histplot(df['mpg'], bins=20, kde=True, color='steelblue')
plt.title('Распределение расхода топлива (mpg)', fontsize=14, fontweight='bold')
plt.xlabel('Миль на галлон (mpg)')
plt.ylabel('Частота')
plt.axvline(df['mpg'].mean(), color='red', linestyle='--', label=f'Среднее: {df["mpg"].mean():.1f}')
plt.axvline(df['mpg'].median(), color='green', linestyle='--', label=f'Медиана: {df["mpg"].median():.1f}')
plt.legend()
plt.tight_layout()
plt.savefig('mpg_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

#Гипотеза 1: различие расхода между usa и japan
usa_mpg = df[df['origin'] == 'usa']['mpg'].dropna()
japan_mpg = df[df['origin'] == 'japan']['mpg'].dropna()

_, p_usa = stats.shapiro(usa_mpg.sample(min(5000, len(usa_mpg)), random_state=42))
_, p_japan = stats.shapiro(japan_mpg.sample(min(5000, len(japan_mpg)), random_state=42))

print(f"  США: p = {p_usa:.4f} → {'нормальное' if p_usa > 0.05 else 'ненормальное'} распределение")
print(f"  Япония: p = {p_japan:.4f} → {'нормальное' if p_japan > 0.05 else 'ненормальное'} распределение")

if p_usa > 0.05 and p_japan > 0.05:
    stat, p_value = stats.ttest_ind(usa_mpg, japan_mpg, equal_var=False)
    test_name = "t-тест Стьюдента (Welch)"
else:
    stat, p_value = stats.mannwhitneyu(usa_mpg, japan_mpg, alternative='two-sided')
    test_name = "Mann-Whitney U"

print(f"Использованный тест: {test_name}, statistic={stat:.4f}, p-value={p_value:.4f}")

if p_value < 0.05:
    mean_usa = usa_mpg.mean()
    mean_jpn = japan_mpg.mean()
    print(f"Вывод: средний mpg в США ({mean_usa:.2f}) существенно отличается от Японии ({mean_jpn:.2f}).")
else:
    print("Вывод: статистически значимых различий в mpg между США и Японией нет.")

#Гипотеза 2: Зависимость количества цилиндров и ускорения
cylinders = df['cylinders'].dropna()
acceleration = df['acceleration'].dropna()

_, p_cyl = stats.shapiro(cylinders.sample(min(5000, len(cylinders)), random_state=42))
_, p_acc = stats.shapiro(acceleration.sample(min(5000, len(acceleration)), random_state=42))

print(f"\nПроверка нормальности:")
print(f"  Цилиндры: p = {p_cyl:.4f} → {'нормальное' if p_cyl > 0.05 else 'ненормальное'}")
print(f"  Ускорение: p = {p_acc:.4f} → {'нормальное' if p_acc > 0.05 else 'ненормальное'}")

if p_cyl > 0.05 and p_acc > 0.05:
    corr, p_value = stats.pearsonr(cylinders, acceleration)
    test_name = "Корреляция Пирсона"
    corr_type = "линейная"
else:
    corr, p_value = stats.spearmanr(cylinders, acceleration)
    test_name = "Корреляция Спирмена"
    corr_type = "монотонная"

print(f"\nПрименён критерий: {test_name}")
print(f"Коэффициент корреляции ({corr_type}): {corr:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    direction = "отрицательная" if corr < 0 else "положительная"
    strength = "сильная" if abs(corr) > 0.7 else ("умеренная" if abs(corr) > 0.3 else "слабая")
    print(f"Вывод: Обнаружена {strength} {direction} корреляция (ρ = {corr:.2f}).")
else:
    print("Вывод: Нет статистически значимой корреляции между цилиндрами и ускорением.")

df_encoded = df.copy()

le = LabelEncoder()
df_encoded['origin_encoded'] = le.fit_transform(df['origin'])
print("\nLabel Encoding для 'origin':")
print(f"  Кодировка: {dict(zip(le.classes_, le.transform(le.classes_)))}")

df_one = pd.get_dummies(df, columns=['origin'], prefix='origin', drop_first=True)
print("\nOne-Hot Encoding для 'origin' (первый столбец удалён для избежания мультиколлинеарности):")
print(f"  Новые столбцы: {df_one.filter(like='origin').columns.tolist()}")

# Выбор числовых признаков + закодированный 'origin'
numeric_df = df.select_dtypes(include=['int64', 'float64']).copy()
numeric_df['origin_num'] = df['origin'].map({'usa': 0, 'europe': 1, 'japan': 2})

corr_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Корреляционная матрица признаков', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix_mpg.png', dpi=300, bbox_inches='tight')
plt.show()
