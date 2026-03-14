from pyspark.sql.types import *

customers_schema = StructType([
    StructField("customer_id", IntegerType(), False),
    StructField("first_name",  StringType(),  False),
    StructField("last_name",   StringType(),  False),
    StructField("phone",       StringType(),  True),
    StructField("email",       StringType(),  True),
    StructField("street",      StringType(),  True),
    StructField("city",        StringType(),  True),
    StructField("state",       StringType(),  True),
    StructField("zip_code",    StringType(),  True)
])

orders_schema = StructType([
    StructField("order_id",       IntegerType(), False),
    StructField("customer_id",    IntegerType(), True),
    StructField("order_status",   IntegerType(), False),
    StructField("order_date",     DateType(),    False),
    StructField("required_date",  DateType(),    True),
    StructField("shipped_date",   DateType(),    True),
    StructField("store_id",       IntegerType(), True),
    StructField("staff_id",       IntegerType(), True)
])

order_items_schema = StructType([
    StructField("order_id",   IntegerType(), False),
    StructField("item_id",    IntegerType(), False),
    StructField("product_id", IntegerType(), True),
    StructField("quantity",   IntegerType(), False),
    StructField("list_price", DecimalType(10,2), False),
    StructField("discount",   DecimalType(4,2),  True)
])

products_schema = StructType([
    StructField("product_id",   IntegerType(), False),
    StructField("product_name", StringType(),  False),
    StructField("brand_id",     IntegerType(), True),
    StructField("category_id",  IntegerType(), True),
    StructField("model_year",   IntegerType(), True),
    StructField("list_price",   DecimalType(10,2), True)
])

brands_schema = StructType([
    StructField("brand_id",   IntegerType(), False),
    StructField("brand_name", StringType(),  False)
])

categories_schema = StructType([
    StructField("category_id",   IntegerType(), False),
    StructField("category_name", StringType(),  False)
])

stocks_schema = StructType([
    StructField("store_id",   IntegerType(), False),
    StructField("product_id", IntegerType(), False),
    StructField("quantity",   IntegerType(), False)
])