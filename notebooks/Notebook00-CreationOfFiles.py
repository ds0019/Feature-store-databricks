# Databricks notebook source

base = "/Volumes/experiments_sammy/default/nyc_taxi_yellow_trips"
src = spark.read.parquet(f"{base}/yellow_tripdata_2026-07.parquet")
src.write.mode("overwrite").option("header", True).csv(f"{base}/landing/csv/2026-07")
src.write.mode("overwrite").parquet(f"{base}/benchmark/parquet/2026-07")
src.write.mode("overwrite").format("delta").save(f"{base}/benchmark/delta/2026-07")

print(src.count())
src.printSchema()

# COMMAND ----------



# COMMAND ----------

src.selectExpr(
  "min(tpep_pickup_datetime)", "max(tpep_pickup_datetime)",
  "min(trip_distance)", "max(trip_distance)",
  "min(fare_amount)", "max(fare_amount)",
  "min(passenger_count)", "max(passenger_count)"
).show(truncate=False)

# COMMAND ----------

import pyarrow.parquet as pq

base  = "/Volumes/experiments_sammy/default/nyc_taxi_yellow_trips"
batch = "2026-07"

def file_in(path):
    """Tutti i file sotto path, anche nelle sottocartelle."""
    out = []
    for f in dbutils.fs.ls(path):
        out += file_in(f.path) if f.isDir() else [f]
    return out

def info_parquet(file_path):
    """Codec e numero di row group, letti dal footer del file."""
    meta = pq.ParquetFile(file_path.replace("dbfs:", "").replace("file:", "")).metadata
    return (f"   codec: {meta.row_group(0).column(0).compression}"
            f"   row group nel primo file: {meta.num_row_groups}")

# File TLC, CSV e Parquet: si sommano i file di dati
for nome, path in [("Parquet TLC", f"{base}/yellow_tripdata_{batch}.parquet"),
                   ("CSV",         f"{base}/landing/csv/{batch}"),
                   ("Parquet",     f"{base}/benchmark/parquet/{batch}")]:
    dati = [f for f in file_in(path) if not f.name.startswith("_")]
    riga = f"{nome:12s} file: {len(dati):3d}   MB: {sum(f.size for f in dati) / 1024**2:8.1f}"
    pq_files = [f for f in dati if f.name.endswith(".parquet")]
    if pq_files:
        riga += info_parquet(pq_files[0].path)
    print(riga)

# Delta: solo i file della versione corrente
delta_path = f"{base}/benchmark/delta/{batch}"
det   = spark.sql(f"DESCRIBE DETAIL delta.`{delta_path}`").first()
tutti = [f for f in file_in(delta_path) if "_delta_log" not in f.path]
print(f"{'Delta':12s} file: {det.numFiles:3d}   MB: {det.sizeInBytes / 1024**2:8.1f}"
      + info_parquet(spark.read.format("delta").load(delta_path).inputFiles()[0])
      + f"   (cartella intera: {len(tutti)} file, {sum(f.size for f in tutti) / 1024**2:.1f} MB)")