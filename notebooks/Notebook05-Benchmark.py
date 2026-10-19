# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "databricks-feature-engineering",
# ]
# ///
# MAGIC %md
# MAGIC # 05 — Benchmark
# MAGIC Misure per il Capitolo 5. Il notebook legge soltanto: non modifica né le copie
# MAGIC di benchmark né le tabelle della pipeline.
# MAGIC
# MAGIC **Parte A (Sezione 5.2): formati di archiviazione.** Spazio su disco delle tre
# MAGIC copie del NB00 (CSV, Parquet, Delta) e tempo di tre query:
# MAGIC - **Q1 scansione completa**: tutte le colonne di tutte le righe;
# MAGIC - **Q2 proiezione e aggregazione**: corse e importo medio per zona (2 colonne su 21);
# MAGIC - **Q3 filtro selettivo**: le corse di un solo giorno (circa 1/31 delle righe).
# MAGIC
# MAGIC **Parte B (Sezione 5.3): generazione del training set.** Lo stesso training set
# MAGIC del NB04 ottenuto (1) dalle Feature Table con `create_training_set`, (2) dalle
# MAGIC Feature Table con una join di uguaglianza, (3) ricalcolando le feature dal Silver,
# MAGIC cioè senza Feature Store. La variante (1) serve da riferimento per verificare che
# MAGIC le tre diano lo stesso risultato; i tempi si misurano per (2) e (3), che usano la
# MAGIC stessa join e differiscono solo per il ricalcolo.
# MAGIC
# MAGIC **Metodo.** Ogni query viene eseguita con la sorgente `noop`, che calcola il
# MAGIC risultato completo senza scriverlo: si misura la lettura e l'elaborazione, non la
# MAGIC scrittura. Ogni misura è preceduta da un'esecuzione di riscaldamento, esclusa, e
# MAGIC ripetuta `RIPETIZIONI` volte; si riportano media, deviazione standard e mediana.

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time, statistics, datetime
import pyarrow.parquet as pq
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

# Stessa convenzione dei Notebook 03 e 04
spark.conf.set("spark.sql.session.timeZone", "America/New_York")

base    = "/Volumes/experiments_sammy/default/nyc_taxi_yellow_trips"
batch   = "2026-07"
TLC     = f"{base}/yellow_tripdata_{batch}.parquet"
CSV     = f"{base}/landing/csv/{batch}"
PARQUET = f"{base}/benchmark/parquet/{batch}"
DELTA   = f"{base}/benchmark/delta/{batch}"

RIPETIZIONI = 10
MB = 1e6          # MB decimali, come nel Notebook 03

print("Esecuzione:", datetime.datetime.now().isoformat(timespec="seconds"), "UTC - Spark", spark.version)

# Su serverless la configurazione della cache potrebbe non essere modificabile

try:
    spark.conf.set("spark.databricks.io.cache.enabled", "false")
    print("Disk cache: disattivata")
except Exception as e:
    print("Disk cache: non modificabile su questo calcolo (" + type(e).__name__ + ")")

# COMMAND ----------

def esegui(df):
    """Calcola il risultato completo senza scriverlo."""
    df.write.format("noop").mode("overwrite").save()

def misura(costruisci):
    """costruisci() crea il DataFrame da zero, quindi la lettura rientra nella misura.
    Restituisce il tempo del riscaldamento e i tempi delle RIPETIZIONI misure (s)."""
    t0 = time.perf_counter(); esegui(costruisci()); primo = time.perf_counter() - t0
    tempi = []
    for _ in range(RIPETIZIONI):
        t0 = time.perf_counter()
        esegui(costruisci())
        tempi.append(time.perf_counter() - t0)
    return primo, tempi

risultati = []

def registra(sezione, variante, primo, tempi):
    med = statistics.median(tempi)
    media = statistics.mean(tempi)
    dev = statistics.stdev(tempi)
    risultati.append((sezione, variante, len(tempi), round(media, 3), round(dev, 3),
                      round(med, 3), round(min(tempi), 3), round(max(tempi), 3), round(primo, 2)))
    print(f"{sezione:8s} {variante:42s} media {media:6.2f} s  dev.std {dev:5.2f}  "
          f"mediana {med:6.2f}  (min {min(tempi):.2f}, max {max(tempi):.2f}, n={len(tempi)})")

def file_in(path):
    """Tutti i file sotto path, anche nelle sottocartelle."""
    out = []
    for f in dbutils.fs.ls(path):
        out += file_in(f.path) if f.isDir() else [f]
    return out

def locale(path):
    return path.replace("dbfs:", "").replace("file:", "")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parte A — Formati di archiviazione (Sezione 5.2)
# MAGIC ### A.1 Spazio su disco
# MAGIC Per la copia Delta si conta solo la versione corrente (`DESCRIBE DETAIL`); il log
# MAGIC delle transazioni è riportato a parte.

# COMMAND ----------

spazio = {}
for nome, path in [("Parquet TLC", TLC), ("CSV", CSV), ("Parquet", PARQUET)]:
    dati = [f for f in file_in(path) if not f.name.startswith("_")]
    spazio[nome] = sum(f.size for f in dati) / MB

det = spark.sql(f"DESCRIBE DETAIL delta.`{DELTA}`").first()
spazio["Delta"] = det.sizeInBytes / MB
log_delta = sum(f.size for f in file_in(f"{DELTA}/_delta_log")) / MB

for nome, v in spazio.items():
    print(f"{nome:12s} {v:8.1f} MB")
print(f"Log Delta    {log_delta:8.3f} MB")
print(f"CSV / Parquet = {spazio['CSV'] / spazio['Parquet']:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### A.2 Spazio occupato da ciascuna colonna nella copia Parquet
# MAGIC Dal footer del file: quanti byte deve leggere una query che usa solo alcune colonne.

# COMMAND ----------

dim = {}
for f in [f for f in file_in(PARQUET) if f.name.endswith(".parquet")]:
    meta = pq.ParquetFile(locale(f.path)).metadata
    for i in range(meta.num_row_groups):
        rg = meta.row_group(i)
        for j in range(rg.num_columns):
            c = rg.column(j)
            dim[c.path_in_schema] = dim.get(c.path_in_schema, 0) + c.total_compressed_size

tot = sum(dim.values())
for col, b in sorted(dim.items(), key=lambda x: -x[1]):
    print(f"{col:25s} {b / MB:6.2f} MB  {100 * b / tot:5.1f}%")
q2 = dim["PULocationID"] + dim["total_amount"]
print(f"\nColonne lette da Q2: {q2 / MB:.2f} MB su {tot / MB:.2f} MB ({100 * q2 / tot:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### A.3 Tempi delle query
# MAGIC Il CSV è letto con lo schema imposto, come nel Notebook 01. A parte si misura Q1 con
# MAGIC `inferSchema`, che richiede una lettura aggiuntiva del file (Sezione 2.1).

# COMMAND ----------

schema = spark.read.parquet(TLC).schema

def leggi(formato):
    if formato == "CSV":
        return (spark.read.schema(schema).option("header", True)
                .option("enforceSchema", False).csv(CSV))
    if formato == "CSV (inferSchema)":
        return spark.read.option("header", True).option("inferSchema", True).csv(CSV)
    if formato == "Parquet":
        return spark.read.parquet(PARQUET)
    if formato == "Delta":
        return spark.read.format("delta").load(DELTA)

QUERY = {
    "Q1": lambda df: df,
    "Q2": lambda df: (df.groupBy("PULocationID")
                        .agg(F.count("*").alias("corse"), F.avg("total_amount").alias("importo_medio"))),
    "Q3": lambda df: df.filter(F.expr(
        "tpep_pickup_datetime >= TIMESTAMP_NTZ'2026-07-15 00:00:00' AND "
        "tpep_pickup_datetime <  TIMESTAMP_NTZ'2026-07-16 00:00:00'")),
}
FORMATI = ["CSV", "Parquet", "Delta"]

# Controllo (non misurato): i tre formati devono dare gli stessi risultati
for q in QUERY:
    conteggi = {fmt: QUERY[q](leggi(fmt)).count() for fmt in FORMATI}
    print(q, "righe nel risultato:", conteggi)

# COMMAND ----------

for q, query in QUERY.items():
    for fmt in FORMATI:
        primo, tempi = misura(lambda: query(leggi(fmt)))
        registra("5.2 " + q, fmt, primo, tempi)

primo, tempi = misura(lambda: QUERY["Q1"](leggi("CSV (inferSchema)")))
registra("5.2 Q1", "CSV (inferSchema)", primo, tempi)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parte B — Generazione del training set (Sezione 5.3)
# MAGIC Le etichette sono lette dal training set salvato dal NB04, così il loro calcolo
# MAGIC non pesa su nessuna delle tre varianti.

# COMMAND ----------

FT_ORARIA      = "workspace.gold.fs_zona_oraria"
FT_GIORNALIERA = "workspace.gold.fs_zona_giornaliera"
FT_ANAGRAFICA  = "workspace.gold.fs_zona_anagrafica"

FEATURE_ORARIE = ["corse_1h", "arrivi_1h", "corse_3h", "corse_24h", "arrivi_24h",
                  "corse_1h_giorno_prima", "corse_1h_settimana_prima",
                  "durata_media_1h", "distanza_media_1h", "velocita_media_1h",
                  "importo_medio_1h"]
FEATURE_GIORNALIERE = ["corse_giorno", "arrivi_giorno", "media_corse_7g",
                       "importo_medio_giorno", "distanza_media_giorno",
                       "quota_carta_giorno"]
FEATURE_STATICHE = ["borough", "service_zone", "is_aeroporto"]
COLONNE = ["zone_id", "t", "etichetta"] + FEATURE_ORARIE + FEATURE_GIORNALIERE + FEATURE_STATICHE

etichette = spark.table("workspace.gold.training_domanda_oraria").select("zone_id", "t", "etichetta")

fe = FeatureEngineeringClient()
feature_lookups = [
    FeatureLookup(table_name=FT_ORARIA, feature_names=FEATURE_ORARIE,
                  lookup_key="zone_id", timestamp_lookup_key="t"),
    FeatureLookup(table_name=FT_GIORNALIERA, feature_names=FEATURE_GIORNALIERE,
                  lookup_key="zone_id", timestamp_lookup_key="t"),
    FeatureLookup(table_name=FT_ANAGRAFICA, feature_names=FEATURE_STATICHE,
                  lookup_key="zone_id"),
]

# (1) Feature Table + join point-in-time della libreria (come nel NB04)
def training_point_in_time():
    return (fe.create_training_set(df=etichette, feature_lookups=feature_lookups,
                                   label="etichetta")
              .load_df().select(COLONNE))

# Join di uguaglianza: equivalente alla point-in-time grazie alla griglia densa
# (verificato nel NB04)
def training_uguaglianza(oraria, giornaliera, anagrafica):
    return (etichette
        .join(oraria.withColumnRenamed("feature_ts", "t"), ["zone_id", "t"], "left")
        .withColumn("_giorno", F.expr("date_trunc('DAY', t)"))
        .join(giornaliera.withColumnRenamed("feature_ts", "_giorno"), ["zone_id", "_giorno"], "left")
        .join(anagrafica, ["zone_id"], "left")
        .select(COLONNE))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ricalcolo delle feature dal Silver
# MAGIC Stesse trasformazioni del NB03 (sezioni 3–6), senza scrittura delle tabelle.

# COMMAND ----------

def feature_dal_silver():
    trips = spark.table("silver.trips")
    zone  = spark.table("silver.zone")

    anagrafica = zone.select("zone_id", "borough", "service_zone",
                             F.col("service_zone").isin("Airports", "EWR").alias("is_aeroporto"))

    g_min, g_max = trips.agg(F.min("pickup_date"), F.max("pickup_date")).first()
    n_ore = ((g_max - g_min).days + 1) * 24
    inizio = g_min.strftime("%Y-%m-%d 00:00:00")
    ore = spark.range(n_ore).select(
        F.expr(f"timestampadd(HOUR, CAST(id AS INT), TIMESTAMP_NTZ'{inizio}')").alias("ora_inizio"))
    griglia = zone.select("zone_id").crossJoin(ore)

    def inizio_ora(c):
        return F.expr(f"make_timestamp_ntz(year({c}), month({c}), day({c}), hour({c}), 0, 0)")

    partenze = (trips
        .groupBy(F.col("pu_zone_id").alias("zone_id"), inizio_ora("pickup_ts").alias("ora_inizio"))
        .agg(F.count("*").alias("corse_1h")))
    # Esiti noti a fine corsa: per ora di arrivo (come nel NB03)
    esiti = (trips
        .groupBy(F.col("pu_zone_id").alias("zone_id"), inizio_ora("dropoff_ts").alias("ora_inizio"))
        .agg(F.avg("durata_min").alias("durata_media_1h"),
             F.avg("trip_distance_mi").alias("distanza_media_1h"),
             F.avg("velocita_mph").alias("velocita_media_1h"),
             F.avg("total_amount").alias("importo_medio_1h")))
    arrivi = (trips
        .groupBy(F.col("do_zone_id").alias("zone_id"), inizio_ora("dropoff_ts").alias("ora_inizio"))
        .agg(F.count("*").alias("arrivi_1h")))
    orario = (griglia
        .join(partenze, ["zone_id", "ora_inizio"], "left")
        .join(esiti,    ["zone_id", "ora_inizio"], "left")
        .join(arrivi,   ["zone_id", "ora_inizio"], "left")
        .fillna(0, subset=["corse_1h", "arrivi_1h"]))

    w = Window.partitionBy("zone_id").orderBy("ora_inizio")
    def somma_ultime(col, n):
        wn = w.rowsBetween(-(n - 1), 0)
        return F.when(F.count(F.lit(1)).over(wn) == n, F.sum(col).over(wn))

    oraria = (orario
        .withColumn("corse_3h",  somma_ultime("corse_1h", 3))
        .withColumn("corse_24h", somma_ultime("corse_1h", 24))
        .withColumn("arrivi_24h", somma_ultime("arrivi_1h", 24))
        .withColumn("corse_1h_giorno_prima",    F.lag("corse_1h", 24).over(w))
        .withColumn("corse_1h_settimana_prima", F.lag("corse_1h", 168).over(w))
        .withColumn("feature_ts", F.expr("timestampadd(HOUR, 1, ora_inizio)").cast("timestamp"))
        .select("zone_id", "feature_ts", *FEATURE_ORARIE))

    per_giorno = (trips   # esiti per giorno di arrivo (come nel NB03)
        .groupBy(F.col("pu_zone_id").alias("zone_id"), F.to_date("dropoff_ts").alias("giorno"))
        .agg(F.avg("total_amount").alias("importo_medio_giorno"),
             F.avg("trip_distance_mi").alias("distanza_media_giorno"),
             F.avg((F.col("payment_type") == 1).cast("double")).alias("quota_carta_giorno")))
    giornaliero = (orario
        .groupBy("zone_id", F.to_date("ora_inizio").alias("giorno"))
        .agg(F.sum("corse_1h").alias("corse_giorno"), F.sum("arrivi_1h").alias("arrivi_giorno"))
        .join(per_giorno, ["zone_id", "giorno"], "left"))
    wg7 = Window.partitionBy("zone_id").orderBy("giorno").rowsBetween(-6, 0)
    giornaliera = (giornaliero
        .withColumn("media_corse_7g",
                    F.when(F.count(F.lit(1)).over(wg7) == 7, F.avg("corse_giorno").over(wg7)))
        .withColumn("feature_ts", F.date_add("giorno", 1).cast("timestamp"))
        .select("zone_id", "feature_ts", *FEATURE_GIORNALIERE))

    return oraria, giornaliera, anagrafica

# COMMAND ----------

# MAGIC %md
# MAGIC ### Controllo: le tre varianti producono lo stesso training set
# MAGIC Le medie ricalcolate possono differire nelle ultime cifre decimali (l'ordine delle
# MAGIC somme cambia da un'esecuzione all'altra): le colonne decimali sono arrotondate a 6 cifre.

# COMMAND ----------

def arrotonda(df):
    return df.select([F.round(c, 6).alias(c) if t == "double" else F.col(c) for c, t in df.dtypes])

def differenze(a, b):
    a, b = arrotonda(a), arrotonda(b)
    return a.exceptAll(b).count() + b.exceptAll(a).count()

rif = training_point_in_time()
ft  = (spark.table(FT_ORARIA), spark.table(FT_GIORNALIERA), spark.table(FT_ANAGRAFICA))
print("Righe del training set:", rif.count())
print("Differenze (2) uguaglianza vs (1) point-in-time:", differenze(training_uguaglianza(*ft), rif))
print("Differenze (3) ricalcolo   vs (1) point-in-time:", differenze(training_uguaglianza(*feature_dal_silver()), rif))

# Dati letti da ciascuna variante
def mb_tabella(nome):
    return spark.sql(f"DESCRIBE DETAIL {nome}").first().sizeInBytes / MB
print(f"\nFeature Table: {sum(mb_tabella(t) for t in [FT_ORARIA, FT_GIORNALIERA, FT_ANAGRAFICA]):.2f} MB")
print(f"Silver (trips + zone): {mb_tabella('silver.trips') + mb_tabella('silver.zone'):.2f} MB")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tempi

# COMMAND ----------

primo, tempi = misura(lambda: training_uguaglianza(
    spark.table(FT_ORARIA), spark.table(FT_GIORNALIERA), spark.table(FT_ANAGRAFICA)))
registra("5.3", "(2) Feature Table, join di uguaglianza", primo, tempi)

primo, tempi = misura(lambda: training_uguaglianza(*feature_dal_silver()))
registra("5.3", "(3) Ricalcolo dal Silver", primo, tempi)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Riepilogo

# COMMAND ----------

display(spark.createDataFrame(
    risultati,
    "sezione STRING, variante STRING, n INT, media_s DOUBLE, dev_std_s DOUBLE, "
    "mediana_s DOUBLE, min_s DOUBLE, max_s DOUBLE, riscaldamento_s DOUBLE"))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Parte C — Letture a freddo e copie divise in 31 file (Sezione 5.2)
# MAGIC La cache su disco di Databricks conserva in locale i file Parquet (e quindi Delta)
# MAGIC già letti, e su serverless non si può disattivare. Per escluderla, **ogni misura
# MAGIC legge una copia appena scritta in una cartella nuova**: un file mai letto non può
# MAGIC essere in cache. Ogni copia viene letta una sola volta e poi cancellata; la
# MAGIC scrittura non è misurata.
# MAGIC
# MAGIC Oltre alle tre copie del NB00 si provano copie Parquet e Delta **divise in 31 file
# MAGIC ordinati per istante di partenza** (circa un giorno per file): con il filtro su un
# MAGIC giorno (Q3) le statistiche di minimo e massimo permettono di saltare gli altri file.
# MAGIC
# MAGIC Richiede le celle iniziali del notebook (import, costanti, funzioni, `schema` e
# MAGIC `QUERY` della parte A), non le celle dei tempi.

# COMMAND ----------

FREDDO     = f"{base}/benchmark/freddo"
RIP_FREDDO = 10
src = spark.read.parquet(TLC)          # stesso DataFrame da cui il NB00 crea le copie

def per_giorno(df):
    """31 file ordinati per istante di partenza, circa uno per giorno."""
    return (df.repartitionByRange(31, "tpep_pickup_datetime")
              .sortWithinPartitions("tpep_pickup_datetime"))

def scrivi(df, fmt, path):
    w = df.write.mode("overwrite")
    if fmt.startswith("CSV"):
        w.option("header", True).csv(path)
    elif fmt.startswith("Parquet"):
        w.parquet(path)
    else:
        # niente accorpamento automatico dei file piccoli: servono 31 file
        (w.format("delta")
          .option("delta.autoOptimize.optimizeWrite", "false")
          .option("delta.autoOptimize.autoCompact", "false")
          .save(path))

def leggi_da(fmt, path):
    if fmt.startswith("CSV"):
        return (spark.read.schema(schema).option("header", True)
                .option("enforceSchema", False).csv(path))
    if fmt.startswith("Parquet"):
        return spark.read.parquet(path)
    return spark.read.format("delta").load(path)

COPIE = {"CSV": src, "Parquet": src, "Delta": src,
         "Parquet (31 file)": per_giorno(src), "Delta (31 file)": per_giorno(src)}

# COMMAND ----------

# MAGIC %md
# MAGIC ### C.1 Controllo delle copie divise
# MAGIC Quanti file ha ciascuna copia e quanti possono contenere corse del 15 luglio,
# MAGIC secondo le statistiche che il motore usa per saltare i dati: il footer dei file per
# MAGIC Parquet, il log delle transazioni per Delta.

# COMMAND ----------

def puo_contenere_il_15(mn, mx):
    # confronto tra stringhe ISO: funziona sia con "2026-07-15 08:00" sia con "2026-07-15T08:00"
    return mx >= "2026-07-15" and mn < "2026-07-16"

p_pq, p_dl = f"{FREDDO}/controllo_parquet", f"{FREDDO}/controllo_delta"
scrivi(COPIE["Parquet (31 file)"], "Parquet", p_pq)
scrivi(COPIE["Delta (31 file)"], "Delta", p_dl)

# Parquet: statistiche nel footer di ogni file
file_pq = [f for f in file_in(p_pq) if f.name.endswith(".parquet")]
utili = 0
for f in file_pq:
    meta = pq.ParquetFile(locale(f.path)).metadata
    for i in range(meta.num_row_groups):
        rg = meta.row_group(i)
        c = [rg.column(j) for j in range(rg.num_columns)
             if rg.column(j).path_in_schema == "tpep_pickup_datetime"][0]
        if c.statistics is None or not c.statistics.has_min_max or \
           puo_contenere_il_15(str(c.statistics.min), str(c.statistics.max)):
            utili += 1
            break
print(f"Parquet (31 file): {len(file_pq)} file, da leggere per Q3: {utili}")

# Delta: statistiche nel log delle transazioni
stats = (spark.read.json(f"{p_dl}/_delta_log/*.json")
         .where("add IS NOT NULL")
         .select(F.get_json_object("add.stats", "$.minValues.tpep_pickup_datetime").alias("mn"),
                 F.get_json_object("add.stats", "$.maxValues.tpep_pickup_datetime").alias("mx"))
         .collect())
utili = sum(1 for s in stats if s.mn is None or puo_contenere_il_15(s.mn, s.mx))
print(f"Delta (31 file):   {len(stats)} file, da leggere per Q3: {utili}")

for fmt, p in [("Parquet (31 file)", p_pq), ("Delta (31 file)", p_dl)]:
    print(fmt, "- righe di Q3:", QUERY["Q3"](leggi_da(fmt, p)).count(), "(atteso 135900)")
dbutils.fs.rm(p_pq, True); dbutils.fs.rm(p_dl, True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### C.2 Tempi a freddo
# MAGIC Circa 30–35 minuti con 10 ripetizioni: ogni misura richiede di scrivere una copia nuova.

# COMMAND ----------

PROVE = ([(q, fmt) for q in ["Q1", "Q2", "Q3"] for fmt in ["CSV", "Parquet", "Delta"]]
         + [("Q3", "Parquet (31 file)"), ("Q3", "Delta (31 file)")])

risultati_freddo = []
for q, fmt in PROVE:
    tempi = []
    for r in range(RIP_FREDDO):
        nome = fmt.replace(" (31 file)", "_31") + f"_{q}_{r}"
        path = f"{FREDDO}/{nome}"
        scrivi(COPIE[fmt], fmt, path)                    # non misurata
        t0 = time.perf_counter()
        esegui(QUERY[q](leggi_da(fmt, path)))            # prima e unica lettura
        tempi.append(time.perf_counter() - t0)
        dbutils.fs.rm(path, True)
    med, media, dev = statistics.median(tempi), statistics.mean(tempi), statistics.stdev(tempi)
    risultati_freddo.append((q, fmt, len(tempi), round(media, 3), round(dev, 3), round(med, 3),
                             round(min(tempi), 3), round(max(tempi), 3)))
    print(f"A freddo {q} {fmt:18s} media {media:6.2f} s  dev.std {dev:5.2f}  mediana {med:6.2f}  "
          f"(min {min(tempi):.2f}, max {max(tempi):.2f}, n={len(tempi)})")

dbutils.fs.rm(FREDDO, True)
display(spark.createDataFrame(risultati_freddo,
        "query STRING, copia STRING, n INT, media_s DOUBLE, dev_std_s DOUBLE, "
        "mediana_s DOUBLE, min_s DOUBLE, max_s DOUBLE"))