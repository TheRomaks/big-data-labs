import psycopg2
import pandas as pd
from config import DB_CONFIG

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

print("Подключение к PostgreSQL успешно!\n")

queries = [
    ("1. Категории и продукты (2 таблицы)",
     """
     SELECT 
         c.category_name,
         COUNT(p.product_id) AS product_count,
         ROUND(AVG(p.list_price)::numeric, 2) AS avg_price
     FROM categories c
     JOIN products p ON c.category_id = p.category_id
     GROUP BY c.category_name
     ORDER BY product_count DESC;
     """),

    ("2. Магазины, заказы и позиции (3 таблицы)",
     """
     SELECT 
         s.store_name,
         COUNT(DISTINCT o.order_id) AS order_count,
         SUM(oi.quantity) AS total_quantity,
         ROUND(SUM(oi.list_price * oi.quantity * (1 - COALESCE(oi.discount, 0)))::numeric, 2) AS total_revenue
     FROM stores s
     JOIN orders o ON s.store_id = o.store_id
     JOIN order_items oi ON o.order_id = oi.order_id
     GROUP BY s.store_name
     ORDER BY total_revenue DESC;
     """),

    ("3. Полная информация по заказу №1 (все таблицы)",
     """
     SELECT 
         o.order_id,
         o.order_date,
         o.order_status,
         c.first_name || ' ' || c.last_name AS customer,
         s.store_name,
         st.first_name || ' ' || st.last_name AS staff,
         p.product_name,
         b.brand_name,
         cat.category_name,
         oi.quantity,
         oi.list_price,
         oi.discount,
         ROUND(oi.list_price * oi.quantity * (1 - COALESCE(oi.discount, 0))::numeric, 2) AS item_total
     FROM orders o
     JOIN customers c     ON o.customer_id = c.customer_id
     JOIN stores s        ON o.store_id    = s.store_id
     JOIN staffs st       ON o.staff_id    = st.staff_id
     JOIN order_items oi  ON o.order_id    = oi.order_id
     JOIN products p      ON oi.product_id = p.product_id
     JOIN brands b        ON p.brand_id    = b.brand_id
     JOIN categories cat  ON p.category_id = cat.category_id
     WHERE o.order_id = 1                     
     ORDER BY oi.item_id;
     """),

    ("4. Количество строк после JOIN orders + order_items",
     """
     SELECT COUNT(*) AS joined_rows
     FROM orders o
     JOIN order_items oi ON o.order_id = oi.order_id;
     """),

    ("5.1 Топ-5 клиентов по общей сумме покупок",
     """
     SELECT 
         c.first_name || ' ' || c.last_name AS customer_name,
         COUNT(DISTINCT o.order_id) AS order_count,
         ROUND(SUM(oi.list_price * oi.quantity * (1 - COALESCE(oi.discount, 0)))::numeric, 2) AS total_spent
     FROM customers c
     JOIN orders o ON c.customer_id = o.customer_id
     JOIN order_items oi ON o.order_id = oi.order_id
     GROUP BY c.customer_id, c.first_name, c.last_name
     ORDER BY total_spent DESC
     LIMIT 5;
     """),

    ("5.2 Продажи по месяцам (с DATE_TRUNC)",
     """
     SELECT 
         DATE_TRUNC('month', o.order_date)::date AS month,
         COUNT(DISTINCT o.order_id) AS orders_count,
         ROUND(SUM(oi.list_price * oi.quantity * (1 - COALESCE(oi.discount, 0)))::numeric, 2) AS monthly_revenue
     FROM orders o
     JOIN order_items oi ON o.order_id = oi.order_id
     GROUP BY month
     ORDER BY month DESC;
     """),

    ("5.3 Товары с нулевым остатком (stocks + продукты)",
     """
     SELECT 
         p.product_name,
         b.brand_name,
         COUNT(DISTINCT st.store_id) AS stores_without_stock
     FROM products p
     JOIN brands b ON p.brand_id = b.brand_id
     JOIN stocks st ON p.product_id = st.product_id
     WHERE st.quantity = 0
     GROUP BY p.product_name, b.brand_name
     ORDER BY stores_without_stock DESC;
     """)
]

for i, (name, sql) in enumerate(queries, 1):
    print(f"ЗАПРОС {i}: {name}")

    cur.execute(sql)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]

    df = pd.DataFrame(rows, columns=columns)
    print(df.to_string(index=False))
    print(f"Запрос {i} выполнен ({len(rows)} строк)\n")

cur.close()
conn.close()
