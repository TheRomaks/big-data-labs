from pyspark.sql import SparkSession
from data_loader import load_all_tables
from queries import run_queries

DATA_PATH = "data/"

def main():
    spark = (SparkSession.builder
             .appName("BikeStore — PySpark Задача 2")
             .config("spark.driver.memory", "4g")
             .config("spark.executor.memory", "4g")
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())

    spark.sparkContext.setLogLevel("WARN")

    spark.conf.set("spark.data.path", DATA_PATH)

    tables = load_all_tables(spark)
    results = run_queries(tables)

    print("\n1. Категории и продукты (2 таблицы)")
    results["q1"].show(truncate=False)

    print("\n2. Магазины, заказы и позиции (3 таблицы)")
    results["q2"].show(truncate=False)

    print("\n3. Полная информация по заказу №1 (все таблицы)")
    results["q3"].show(truncate=False)

    print("\n4. Количество строк после JOIN orders + order_items")
    print(f"Результат: {results['joined_count']:,} строк\n")

    print("\n5.1 Топ-5 клиентов по общей сумме покупок")
    results["q51"].show(truncate=False)

    print("\n5.2 Продажи по месяцам")
    results["q52"].show(truncate=False)

    print("\n5.3 Товары с нулевым остатком")
    results["q53"].show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()