# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 02 — Layer Silver
# MAGIC Legge `bronze.trips_raw`, applica regole di validità, deduplica,
# MAGIC aggiunge colonne derivate e scrive `silver.trips`.
# MAGIC Registra in `silver.quality_report` quante righe viola ciascuna regola.

# COMMAND ----------

from functools import reduce
from pyspark.sql import functions as F

batch_id = "2026-07"
start, end = "2026-07-01", "2026-08-01"   # periodo valido del batch

spark.sql("CREATE SCHEMA IF NOT EXISTS silver")

bronze = spark.table("bronze.trips_raw").filter(F.col("_batch_id") == batch_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Rinomina in snake_case e durata della corsa

# COMMAND ----------

t = (bronze.select(
        F.col("VendorID").alias("vendor_id"),
        F.col("tpep_pickup_datetime").alias("pickup_ts"),
        F.col("tpep_dropoff_datetime").alias("dropoff_ts"),
        F.col("passenger_count").cast("int").alias("passenger_count"),
        F.col("trip_distance").alias("trip_distance_mi"),
        F.col("PULocationID").alias("pu_zone_id"),
        F.col("DOLocationID").alias("do_zone_id"),
        F.col("payment_type").cast("int").alias("payment_type"),
        "fare_amount", "tip_amount", "total_amount",
        "_ingestion_ts", "_batch_id")
     .withColumn("durata_min",
                 F.expr("timestampdiff(SECOND, pickup_ts, dropoff_ts) / 60.0")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Regole di validità
# MAGIC Ogni regola descrive una **violazione**. Un valore nullo nella condizione
# MAGIC conta come violazione (coalesce a True).

# COMMAND ----------

regole = {
    "chiavi_nulle":        F.col("pickup_ts").isNull() | F.col("dropoff_ts").isNull()
                           | F.col("pu_zone_id").isNull() | F.col("do_zone_id").isNull(),
    "fuori_periodo":       (F.col("pickup_ts") < start) | (F.col("pickup_ts") >= end),
    "durata_non_valida":   ~F.col("durata_min").between(1, 180),
    "distanza_non_valida": ~F.col("trip_distance_mi").between(0.1, 100),
    "importo_non_valido":  (F.col("fare_amount") <= 0) | (F.col("total_amount") <= 0),
    "zona_sconosciuta":    ~F.col("pu_zone_id").between(1, 263)
                           | ~F.col("do_zone_id").between(1, 263),
}

viol_cols = [f"_viol_{k}" for k in regole]
t = t.withColumns({f"_viol_{k}": F.coalesce(c, F.lit(True)) for k, c in regole.items()})
t = t.withColumn("_valida", ~reduce(lambda a, b: a | b, [F.col(c) for c in viol_cols]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Conteggi per il report di qualità
# MAGIC Le regole si sovrappongono: una riga può violarne più d'una,
# MAGIC quindi i conteggi per regola non si sommano al totale scartato.

# COMMAND ----------

stats = t.agg(
    F.count("*").alias("righe_bronze"),
    *[F.sum(F.col(c).cast("int")).alias(c.replace("_viol_", "")) for c in viol_cols],
    F.sum((~F.col("_valida")).cast("int")).alias("righe_non_valide"),
).first().asDict()

valide = t.filter("_valida").drop(*viol_cols, "_valida")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Deduplica e colonne derivate
# MAGIC Il dataset non ha chiave primaria: la chiave naturale della corsa è
# MAGIC definita come (vendor, pickup, dropoff, zona partenza, zona arrivo, importo).

# COMMAND ----------

chiave_naturale = ["vendor_id", "pickup_ts", "dropoff_ts",
                   "pu_zone_id", "do_zone_id", "total_amount"]

silver = (valide
    .dropDuplicates(chiave_naturale)
    .withColumn("pickup_date",  F.to_date("pickup_ts"))
    .withColumn("pickup_hour",  F.hour("pickup_ts"))
    .withColumn("day_of_week",  F.dayofweek("pickup_ts"))          # 1 = domenica
    .withColumn("is_weekend",   F.dayofweek("pickup_ts").isin(1, 7))
    .withColumn("velocita_mph", F.col("trip_distance_mi") / (F.col("durata_min") / 60)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Scrittura idempotente
# MAGIC `replaceWhere` sostituisce atomicamente solo il batch corrente:
# MAGIC rilanciare il notebook non duplica i dati.

# COMMAND ----------

if spark.catalog.tableExists("silver.trips"):
    (silver.write.mode("overwrite")
        .option("replaceWhere", f"_batch_id = '{batch_id}'")
        .saveAsTable("silver.trips"))
else:
    silver.write.mode("overwrite").saveAsTable("silver.trips")

righe_silver = spark.table("silver.trips").filter(F.col("_batch_id") == batch_id).count()
stats["duplicati_rimossi"] = (stats["righe_bronze"] - stats["righe_non_valide"]) - righe_silver
stats["righe_silver"] = righe_silver

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Report di qualità

# COMMAND ----------

report = (spark.createDataFrame([stats])
          .withColumn("_batch_id", F.lit(batch_id))
          .withColumn("_report_ts", F.current_timestamp()))

if spark.catalog.tableExists("silver.quality_report"):
    spark.sql(f"DELETE FROM silver.quality_report WHERE _batch_id = '{batch_id}'")
report.write.mode("append").option("mergeSchema", True).saveAsTable("silver.quality_report")

display(report)

# COMMAND ----------

