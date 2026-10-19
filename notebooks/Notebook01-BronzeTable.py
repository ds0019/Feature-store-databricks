# Databricks notebook source
from pyspark.sql import functions as F

base = "/Volumes/experiments_sammy/default/nyc_taxi_yellow_trips"
batch_id = "2026-07"

spark.sql("CREATE SCHEMA IF NOT EXISTS bronze")

# Contratto dei dati: lo schema del file Parquet originale
schema = spark.read.parquet(f"{base}/yellow_tripdata_2026-07.parquet").schema

df = (spark.read
      .schema(schema)
      .option("header", True)
      .option("enforceSchema", False)   # verifica i nomi dell'intestazione
      .csv(f"{base}/landing/csv/{batch_id}"))

bronze = (df
    .withColumn("_ingestion_ts", F.current_timestamp())
    .withColumn("_source_file",  F.col("_metadata.file_path"))
    .withColumn("_batch_id",     F.lit(batch_id)))

# Scrittura idempotente: replaceWhere sostituisce atomicamente solo il batch corrente
if spark.catalog.tableExists("bronze.trips_raw"):
    (bronze.write.mode("overwrite")
        .option("replaceWhere", f"_batch_id = '{batch_id}'")
        .saveAsTable("bronze.trips_raw"))
else:
    bronze.write.mode("overwrite").saveAsTable("bronze.trips_raw")

# COMMAND ----------

print(spark.table("bronze.trips_raw").filter(F.col("_batch_id") == batch_id).count())
display(spark.sql("DESCRIBE HISTORY bronze.trips_raw")
        .select("version", "operation", "operationParameters").limit(3))

# COMMAND ----------

df = spark.table("workspace.bronze.trips_raw")

display(df)