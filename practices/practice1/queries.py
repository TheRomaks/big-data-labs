from pyspark.sql.functions import (
    col, count, avg, sum as _sum, round, coalesce,
    concat_ws, lit, trunc
)


def run_queries(tables):
    customers = tables["customers"]
    orders = tables["orders"]
    order_items = tables["order_items"]
    products = tables["products"]
    brands = tables["brands"]
    categories = tables["categories"]
    staffs = tables["staffs"]
    stores = tables["stores"]
    stocks = tables["stocks"]

    q1 = categories.alias("c").join(
        products.alias("p"), col("c.category_id") == col("p.category_id"), "inner"
    ).groupBy(col("c.category_name")) \
        .agg(count(col("p.product_id")).alias("product_count"),
             round(avg(col("p.list_price")), 2).alias("avg_price")) \
        .orderBy(col("product_count").desc())

    q2 = stores.alias("s").join(orders.alias("o"), col("s.store_id") == col("o.store_id")) \
        .join(order_items.alias("oi"), col("o.order_id") == col("oi.order_id")) \
        .groupBy(col("s.store_name")) \
        .agg(count("o.order_id").alias("order_count"),
             _sum(col("oi.quantity")).alias("total_quantity"),
             round(_sum(col("oi.list_price") * col("oi.quantity") *
                        (1 - coalesce(col("oi.discount"), lit(0)))), 2).alias("total_revenue")) \
        .orderBy(col("total_revenue").desc())

    q3 = orders.alias("o").filter(col("o.order_id") == 1) \
        .join(customers.alias("c"), col("o.customer_id") == col("c.customer_id")) \
        .join(stores.alias("s"), col("o.store_id") == col("s.store_id")) \
        .join(staffs.alias("st"), col("o.staff_id") == col("st.staff_id")) \
        .join(order_items.alias("oi"), col("o.order_id") == col("oi.order_id")) \
        .join(products.alias("p"), col("oi.product_id") == col("p.product_id")) \
        .join(brands.alias("b"), col("p.brand_id") == col("b.brand_id")) \
        .join(categories.alias("cat"), col("p.category_id") == col("cat.category_id")) \
        .select(
        col("o.order_id"), col("o.order_date"), col("o.order_status"),
        concat_ws(" ", col("c.first_name"), col("c.last_name")).alias("customer"),
        col("s.store_name"),
        concat_ws(" ", col("st.first_name"), col("st.last_name")).alias("staff"),
        col("p.product_name"), col("b.brand_name"), col("cat.category_name"),
        col("oi.quantity"), col("oi.list_price"), col("oi.discount"),
        round(col("oi.list_price") * col("oi.quantity") *
              (1 - coalesce(col("oi.discount"), lit(0))), 2).alias("item_total")
    ).orderBy("oi.item_id")

    joined_count = orders.join(order_items, "order_id").count()

    q51 = customers.alias("c").join(orders.alias("o"), col("c.customer_id") == col("o.customer_id")) \
        .join(order_items.alias("oi"), col("o.order_id") == col("oi.order_id")) \
        .groupBy(col("c.customer_id"), col("c.first_name"), col("c.last_name")) \
        .agg(count("o.order_id").alias("order_count"),
             round(_sum(col("oi.list_price") * col("oi.quantity") *
                        (1 - coalesce(col("oi.discount"), lit(0)))), 2).alias("total_spent")) \
        .orderBy(col("total_spent").desc()) \
        .limit(5)

    q52 = orders.alias("o").join(order_items.alias("oi"), col("o.order_id") == col("oi.order_id")) \
        .groupBy(trunc(col("o.order_date"), "month").alias("month")) \
        .agg(count("o.order_id").alias("orders_count"),
             round(_sum(col("oi.list_price") * col("oi.quantity") *
                        (1 - coalesce(col("oi.discount"), lit(0)))), 2).alias("monthly_revenue")) \
        .orderBy(col("month").desc())

    q53 = products.alias("p").join(brands.alias("b"), col("p.brand_id") == col("b.brand_id")) \
        .join(stocks.alias("st"), col("p.product_id") == col("st.product_id")) \
        .filter(col("st.quantity") == 0) \
        .groupBy(col("p.product_name"), col("b.brand_name")) \
        .agg(count("st.store_id").alias("stores_without_stock")) \
        .orderBy(col("stores_without_stock").desc())

    return {
        "q1": q1, "q2": q2, "q3": q3, "joined_count": joined_count,
        "q51": q51, "q52": q52, "q53": q53
    }