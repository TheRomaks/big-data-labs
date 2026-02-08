import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

df = pd.read_csv('data/final_cleaned_data.csv')

plt.figure(figsize=(14, 7))
top_categories = df['category_name'].value_counts().nlargest(6).index
top_brands = df['brand_name'].value_counts().nlargest(8).index
filtered = df[df['category_name'].isin(top_categories) & df['brand_name'].isin(top_brands)]

sns.boxplot(data=filtered, x='category_name', y='product_price', hue='brand_name')
plt.title('Распределение цен по категориям и брендам', fontsize=14, fontweight='bold')
plt.xlabel('Категория товара')
plt.ylabel('Цена, $')
plt.xticks(rotation=15)
plt.legend(title='Бренд', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('data/price_by_category_brand_imp.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 7))

df_filtered = df[df['order_status'].isin([1, 2, 3, 4])].copy()
df_filtered['order_status_str'] = df_filtered['order_status'].map({
    1: 'Ожидание', 2: 'В обработке', 3: 'Отправлен', 4: 'Доставлен'
})

sns.scatterplot(
    data=df_filtered,
    x='quantity',
    y='discount',
    hue='order_status_str',
    size='product_price',
    sizes=(40, 300),
    alpha=0.6,
    edgecolor='w',
    linewidth=0.5
)
plt.title('Связь количества товаров, скидки и статуса заказа\n(размер точки = цена товара)',
          fontsize=14, fontweight='bold')
plt.xlabel('Количество товаров в заказе')
plt.ylabel('Скидка, доля')
plt.legend(title='Статус заказа', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/quantity_discount_status_imp.png', dpi=300, bbox_inches='tight')
plt.show()