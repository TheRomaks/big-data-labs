from pyspark.sql import SparkSession
import schemas


def load_all_tables(spark: SparkSession):

    customers = spark.read.option("header", "true").schema(schemas.customers_schema) \
        .csv(spark.conf.get("spark.data.path") + "customers.csv")

    orders = spark.read.option("header", "true").schema(schemas.orders_schema) \
        .csv(spark.conf.get("spark.data.path") + "orders.csv")

    order_items = spark.read.option("header", "true").schema(schemas.order_items_schema) \
        .csv(spark.conf.get("spark.data.path") + "order_items.csv")

    products = spark.read.option("header", "true").schema(schemas.products_schema) \
        .csv(spark.conf.get("spark.data.path") + "products.csv")

    brands = spark.read.option("header", "true").schema(schemas.brands_schema) \
        .csv(spark.conf.get("spark.data.path") + "brands.csv")

    categories = spark.read.option("header", "true").schema(schemas.categories_schema) \
        .csv(spark.conf.get("spark.data.path") + "categories.csv")

    staffs = spark.read.option("header", "true") \
        .csv(spark.conf.get("spark.data.path") + "staffs.csv")

    stores = spark.read.option("header", "true") \
        .csv(spark.conf.get("spark.data.path") + "stores.csv")

    stocks = spark.read.option("header", "true").schema(schemas.stocks_schema) \
        .csv(spark.conf.get("spark.data.path") + "stocks.csv")

    return {
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "products": products,
        "brands": brands,
        "categories": categories,
        "staffs": staffs,
        "stores": stores,
        "stocks": stocks
    }