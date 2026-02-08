import pandas as pd
from sqlalchemy import create_engine

DB_CONFIG = {
    "user": "postgres",
    "password": "root1234",
    "host": "localhost",
    "port": 5432,
    "dbname": "bike_store"
}

engine = create_engine(
    f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}",
)

query = """
SELECT
    oi.order_id,
    oi.item_id,
    oi.product_id,
    oi.quantity,
    oi.list_price,
    oi.discount,

    o.order_status,
    o.order_date,
    o.required_date,
    o.shipped_date,

    c.customer_id,
    c.first_name AS customer_first_name,
    c.last_name  AS customer_last_name,
    c.city       AS customer_city,
    c.state      AS customer_state,

    p.product_name,
    p.model_year,
    p.list_price AS product_price,

    cat.category_name,
    b.brand_name,

    s.store_name,
    s.city  AS store_city,
    s.state AS store_state,

    st.first_name AS staff_first_name,
    st.last_name  AS staff_last_name,
    st.active     AS staff_active,

    sk.quantity AS stock_quantity

FROM order_items oi
LEFT JOIN orders o       ON oi.order_id   = o.order_id
LEFT JOIN customers c   ON o.customer_id = c.customer_id
LEFT JOIN products p    ON oi.product_id = p.product_id
LEFT JOIN categories cat ON p.category_id = cat.category_id
LEFT JOIN brands b      ON p.brand_id    = b.brand_id
LEFT JOIN stores s      ON o.store_id    = s.store_id
LEFT JOIN staffs st     ON o.staff_id    = st.staff_id
LEFT JOIN stocks sk     ON sk.product_id = p.product_id
                         AND sk.store_id = s.store_id
"""
conn = engine.raw_connection()
df = pd.read_sql_query(query, conn)
print("Пропуски в данных:\n", df.isnull().sum())
conn.close()
df.to_csv("data/merged_sales_data.csv", index=False)
print(f"Строк: {len(df)}, столбцов: {len(df.columns)}")
engine.dispose()

