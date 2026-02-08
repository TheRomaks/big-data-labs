import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

df = pd.read_csv('data/final_cleaned_data.csv')

# Гистограмма 1: Распределение цен товаров
plt.figure(figsize=(10, 6))
sns.histplot(df['product_price'], bins=40, kde=True, color='steelblue')
plt.title('Распределение цен товаров (product_price)', fontsize=14, fontweight='bold')
plt.xlabel('Цена, $')
plt.ylabel('Частота')
plt.axvline(df['product_price'].mean(), color='red', linestyle='--',
            label=f'Среднее: ${df["product_price"].mean():.2f}')
plt.axvline(df['product_price'].median(), color='green', linestyle='--',
            label=f'Медиана: ${df["product_price"].median():.2f}')
plt.legend()
plt.tight_layout()
plt.savefig('data/hist_product_price_improved.png', dpi=300, bbox_inches='tight')
plt.show()

# Гистограмма 2: Распределение количества товаров в заказе, нормированное распределение
plt.figure(figsize=(10, 6))
sns.histplot(df['quantity'], bins=range(1, df['quantity'].max() + 2),
             discrete=True, color='seagreen', edgecolor='black')
plt.title('Распределение количества товаров в заказе (quantity)', fontsize=14, fontweight='bold')
plt.xlabel('Количество штук')
plt.ylabel('Частота')
plt.xticks(range(1, min(df['quantity'].max() + 1, 11)))
plt.tight_layout()
plt.savefig('data/hist_quantity_improved.png', dpi=300, bbox_inches='tight')
plt.show()

# Гистограмма 3: Распределение скидок
plt.figure(figsize=(10, 6))
df['discount_pct'] = df['discount'] * 100
sns.histplot(df['discount_pct'], bins=20, kde=True, color='coral')
plt.title('Распределение скидок (discount)', fontsize=14, fontweight='bold')
plt.xlabel('Скидка, %')
plt.ylabel('Частота')
plt.axvline(df['discount_pct'].mean(), color='red', linestyle='--',
            label=f'Среднее: {df["discount_pct"].mean():.1f}%')
plt.axvline(df['discount_pct'].median(), color='green', linestyle='--',
            label=f'Медиана: {df["discount_pct"].median():.1f}%')
plt.legend()
plt.tight_layout()
plt.savefig('data/hist_discount_improved.png', dpi=300, bbox_inches='tight')
plt.show()
