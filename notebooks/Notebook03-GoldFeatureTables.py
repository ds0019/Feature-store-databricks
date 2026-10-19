# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 03 — Layer Gold: Feature Table
# MAGIC Legge `silver.trips` e costruisce le Feature Table dell'entità **zona**,
# MAGIC a tre granularità diverse:
# MAGIC
# MAGIC | Tabella | Chiave primaria | Granularità | Quando è noto il valore |
# MAGIC |---|---|---|---|
# MAGIC | `gold.fs_zona_anagrafica` | `zone_id` | statica | sempre |
# MAGIC | `gold.fs_zona_giornaliera` | `zone_id`, `feature_ts` | 1 giorno | a mezzanotte, a giorno concluso |
# MAGIC | `gold.fs_zona_oraria` | `zone_id`, `feature_ts` | 1 ora | allo scadere dell'ora |
# MAGIC
# MAGIC **Regola fondamentale:** `feature_ts` è l'istante in cui il valore diventa
# MAGIC *disponibile*, cioè la **fine** della finestra di aggregazione, non l'inizio.
# MAGIC Una riga con `feature_ts = 11:00` descrive l'ora 10:00–11:00.
# MAGIC Su questa convenzione si basano le point-in-time join del notebook 04.
# MAGIC
# MAGIC **Regola sugli esiti della corsa:** una corsa ha un inizio e una fine. La
# MAGIC *partenza* è nota all'inizio, mentre durata, distanza, importo e metodo di
# MAGIC pagamento sono noti solo alla *fine*. Ogni informazione va quindi attribuita alla
# MAGIC finestra in cui diventa nota: le partenze all'ora (o al giorno) di inizio corsa,
# MAGIC gli esiti all'ora (o al giorno) di fine corsa. Altrimenti una riga con
# MAGIC `feature_ts = 10:00` conterrebbe la durata di una corsa partita alle 9:50 e
# MAGIC conclusa alle 10:20, cioè un valore non ancora noto alle 10:00.
# MAGIC
# MAGIC Le tabelle non contengono etichette né riferimenti a uno specifico modello:
# MAGIC sono riutilizzabili da qualsiasi caso d'uso che abbia la zona come entità.
# MAGIC
# MAGIC **Prerequisito:** caricare `taxi_zone_lookup.csv` (sito TLC) nel volume, in
# MAGIC `lookup/taxi_zone_lookup.csv`.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

base = "/Volumes/experiments_sammy/default/nyc_taxi_yellow_trips"

# I timestamp TLC sono ora locale di New York (timestamp_ntz). Le chiavi
# temporali delle feature table devono essere di tipo TIMESTAMP: con il fuso
# della sessione impostato su New York la conversione conserva l'istante corretto.
spark.conf.set("spark.sql.session.timeZone", "America/New_York")

spark.sql("CREATE SCHEMA IF NOT EXISTS gold")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Tabella di riferimento delle zone (Silver)
# MAGIC La Taxi Zone Lookup è un dato di riferimento: viene standardizzata nel Silver
# MAGIC come le corse. Si tengono solo le 263 zone geografiche, coerentemente con la
# MAGIC regola `zona_sconosciuta` del notebook 02.

# COMMAND ----------

zone = (spark.read
        .option("header", True)
        .schema("LocationID INT, Borough STRING, Zone STRING, service_zone STRING")
        .csv(f"{base}/lookup/taxi_zone_lookup.csv")
        .select(F.col("LocationID").alias("zone_id"),
                F.col("Borough").alias("borough"),
                F.col("Zone").alias("zone_name"),
                "service_zone")
        .filter(F.col("zone_id").between(1, 263)))

zone.write.mode("overwrite").option("overwriteSchema", True).saveAsTable("silver.zone")
print(spark.table("silver.zone").count(), "zone")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Funzione di scrittura delle feature table
# MAGIC Ogni tabella viene ricreata per intero (*full refresh*) perché le feature a
# MAGIC finestra mobile di un mese dipendono anche dai giorni del mese precedente.
# MAGIC `CREATE OR REPLACE` mantiene la cronologia Delta: le versioni precedenti
# MAGIC restano interrogabili con il time travel.
# MAGIC
# MAGIC Metadati registrati in Unity Catalog:
# MAGIC - **chiave primaria** (con `TIMESERIES` sulla colonna temporale): è ciò che
# MAGIC   rende la tabella una *time series feature table* per Databricks;
# MAGIC - **commenti** su tabella e colonne (il catalogo delle feature);
# MAGIC - **tag** (layer, entità, granularità);
# MAGIC - **liquid clustering** su chiave e tempo al posto del partizionamento.

# COMMAND ----------

def sql_str(s):
    return s.replace("\\", "\\\\").replace("'", "\\'")

def scrivi_feature_table(df, nome, chiavi, ts=None, commento="",
                         commenti_colonne=None, tag=None):
    df.createOrReplaceTempView("_ft_tmp")
    cluster = ", ".join(chiavi + ([ts] if ts else []))
    spark.sql(f"""
        CREATE OR REPLACE TABLE {nome}
        CLUSTER BY ({cluster})
        COMMENT '{sql_str(commento)}'
        AS SELECT * FROM _ft_tmp
    """)

    # Chiave primaria: le colonne devono essere NOT NULL
    for c in chiavi + ([ts] if ts else []):
        spark.sql(f"ALTER TABLE {nome} ALTER COLUMN {c} SET NOT NULL")
    pk = ", ".join(chiavi + ([f"{ts} TIMESERIES"] if ts else []))
    nome_pk = "pk_" + nome.split(".")[-1]
    spark.sql(f"ALTER TABLE {nome} DROP PRIMARY KEY IF EXISTS")
    spark.sql(f"ALTER TABLE {nome} ADD CONSTRAINT {nome_pk} PRIMARY KEY ({pk})")

    for col, txt in (commenti_colonne or {}).items():
        spark.sql(f"ALTER TABLE {nome} ALTER COLUMN {col} COMMENT '{sql_str(txt)}'")
    if tag:
        coppie = ", ".join(f"'{k}' = '{v}'" for k, v in tag.items())
        spark.sql(f"ALTER TABLE {nome} SET TAGS ({coppie})")

    spark.sql(f"OPTIMIZE {nome}")
    print(f"{nome}: {spark.table(nome).count()} righe")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature statiche: `gold.fs_zona_anagrafica`

# COMMAND ----------

anagrafica = (spark.table("silver.zone")
    .select("zone_id", "borough", "service_zone",
            F.col("service_zone").isin("Airports", "EWR").alias("is_aeroporto")))

scrivi_feature_table(
    anagrafica, "gold.fs_zona_anagrafica", chiavi=["zone_id"],
    commento="Feature statiche della zona TLC (fonte: Taxi Zone Lookup)",
    commenti_colonne={
        "zone_id": "Identificativo della zona TLC (1-263)",
        "borough": "Distretto di New York a cui appartiene la zona",
        "service_zone": "Tipologia di area di servizio (Yellow Zone, Boro Zone, Airports, EWR)",
        "is_aeroporto": "Vero se la zona e un aeroporto (JFK, LaGuardia, Newark)",
    },
    tag={"layer": "gold", "entita": "zona", "granularita": "statica"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Griglia densa zona × ora
# MAGIC Un'ora senza corse non compare in `silver.trips`, ma per il modello è
# MAGIC un'informazione (zero corse). Si costruisce quindi il prodotto cartesiano
# MAGIC di tutte le zone con tutte le ore del periodo: ogni zona ha esattamente una
# MAGIC riga per ora. La densità è anche ciò che rende corrette le finestre mobili
# MAGIC basate sul numero di righe (sezione 5).

# COMMAND ----------

trips = spark.table("silver.trips")

g_min, g_max = trips.agg(F.min("pickup_date"), F.max("pickup_date")).first()
n_ore = ((g_max - g_min).days + 1) * 24
inizio = g_min.strftime("%Y-%m-%d 00:00:00")

ore = spark.range(n_ore).select(
    F.expr(f"timestampadd(HOUR, CAST(id AS INT), TIMESTAMP_NTZ'{inizio}')").alias("ora_inizio"))
griglia = spark.table("silver.zone").select("zone_id").crossJoin(ore)

def inizio_ora(c):
    return F.expr(f"make_timestamp_ntz(year({c}), month({c}), day({c}), hour({c}), 0, 0)")

print(f"Periodo {g_min} - {g_max}: {n_ore} ore x {spark.table('silver.zone').count()} zone")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4b. Corse a cavallo di due finestre
# MAGIC Quante corse finiscono in un'ora (o in un giorno) diversa da quella in cui sono
# MAGIC partite. Sono le corse i cui esiti, se attribuiti all'ora di partenza, sarebbero
# MAGIC entrati in una riga con `feature_ts` precedente alla fine della corsa (Sezione 3.4).

# COMMAND ----------

n_corse = trips.count()
ora_diversa = trips.filter(inizio_ora("pickup_ts") != inizio_ora("dropoff_ts")).count()
giorno_diverso = trips.filter(F.to_date("pickup_ts") != F.to_date("dropoff_ts")).count()
print(f"Corse nel Silver:                   {n_corse}")
print(f"Finiscono in un'ora diversa:        {ora_diversa} ({100 * ora_diversa / n_corse:.1f}%)")
print(f"Finiscono in un giorno diverso:     {giorno_diverso} ({100 * giorno_diverso / n_corse:.2f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Feature orarie: `gold.fs_zona_oraria`
# MAGIC - aggregati dell'ora appena conclusa: partenze (per ora di inizio corsa), arrivi
# MAGIC   (per ora di fine corsa) e medie degli esiti delle corse partite dalla zona e
# MAGIC   **concluse** nell'ora (per ora di fine corsa, regola sugli esiti);
# MAGIC - somme mobili sulle ultime 3 e 24 ore;
# MAGIC - valori della stessa ora del giorno prima e della settimana prima.
# MAGIC
# MAGIC Una finestra mobile incompleta (ad esempio le prime 23 ore del periodo per
# MAGIC `corse_24h`) produce `NULL` e non un valore sottostimato.

# COMMAND ----------

# Partenze: note all'inizio della corsa, quindi per ora di partenza
partenze = (trips
    .groupBy(F.col("pu_zone_id").alias("zone_id"), inizio_ora("pickup_ts").alias("ora_inizio"))
    .agg(F.count("*").alias("corse_1h")))

# Esiti (durata, distanza, velocità, importo): noti solo alla fine della corsa,
# quindi per ora di ARRIVO. Restano le corse partite dalla zona (pu_zone_id).
esiti = (trips
    .groupBy(F.col("pu_zone_id").alias("zone_id"), inizio_ora("dropoff_ts").alias("ora_inizio"))
    .agg(F.avg("durata_min").alias("durata_media_1h"),
         F.avg("trip_distance_mi").alias("distanza_media_1h"),
         F.avg("velocita_mph").alias("velocita_media_1h"),
         F.avg("total_amount").alias("importo_medio_1h")))

# Arrivi: noti alla fine della corsa, per zona e ora di arrivo
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

fs_oraria = (orario
    .withColumn("corse_3h",  somma_ultime("corse_1h", 3))
    .withColumn("corse_24h", somma_ultime("corse_1h", 24))
    .withColumn("arrivi_24h", somma_ultime("arrivi_1h", 24))
    .withColumn("corse_1h_giorno_prima",    F.lag("corse_1h", 24).over(w))
    .withColumn("corse_1h_settimana_prima", F.lag("corse_1h", 168).over(w))
    # Il valore dell'ora [t, t+1h) diventa noto a t+1h
    .withColumn("feature_ts",
                F.expr("timestampadd(HOUR, 1, ora_inizio)").cast("timestamp"))
    .select("zone_id", "feature_ts",
            "corse_1h", "arrivi_1h", "corse_3h", "corse_24h", "arrivi_24h",
            "corse_1h_giorno_prima", "corse_1h_settimana_prima",
            "durata_media_1h", "distanza_media_1h", "velocita_media_1h", "importo_medio_1h"))

scrivi_feature_table(
    fs_oraria, "gold.fs_zona_oraria", chiavi=["zone_id"], ts="feature_ts",
    commento="Feature orarie della zona di partenza. feature_ts = fine dell ora aggregata (istante di disponibilita)",
    commenti_colonne={
        "feature_ts": "Fine della finestra oraria: istante da cui i valori sono noti",
        "corse_1h": "Corse partite dalla zona nell ora conclusa a feature_ts",
        "arrivi_1h": "Corse arrivate nella zona nell ora conclusa a feature_ts",
        "corse_3h": "Corse partite nelle ultime 3 ore (NULL se finestra incompleta)",
        "corse_24h": "Corse partite nelle ultime 24 ore (NULL se finestra incompleta)",
        "arrivi_24h": "Corse arrivate nelle ultime 24 ore (NULL se finestra incompleta)",
        "corse_1h_giorno_prima": "corse_1h della stessa ora del giorno precedente",
        "corse_1h_settimana_prima": "corse_1h della stessa ora di sette giorni prima",
        "durata_media_1h": "Durata media in minuti delle corse partite dalla zona e concluse nell ora (NULL se nessuna)",
        "distanza_media_1h": "Distanza media in miglia delle corse partite dalla zona e concluse nell ora",
        "velocita_media_1h": "Velocita media in mph delle corse partite dalla zona e concluse nell ora (traffico)",
        "importo_medio_1h": "Importo totale medio in dollari delle corse partite dalla zona e concluse nell ora",
    },
    tag={"layer": "gold", "entita": "zona", "granularita": "oraria"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Feature giornaliere: `gold.fs_zona_giornaliera`
# MAGIC Profilo della zona sul giorno concluso e media mobile a 7 giorni.
# MAGIC Il valore del giorno `g` diventa noto a mezzanotte, cioè a `g + 1 giorno`.
# MAGIC Come nella tabella oraria, le medie degli esiti (importo, distanza, pagamento)
# MAGIC sono calcolate sulle corse partite dalla zona e **concluse** nel giorno.

# COMMAND ----------

# Esiti noti a fine corsa: per giorno di ARRIVO
per_giorno = (trips
    .groupBy(F.col("pu_zone_id").alias("zone_id"), F.to_date("dropoff_ts").alias("giorno"))
    .agg(F.avg("total_amount").alias("importo_medio_giorno"),
         F.avg("trip_distance_mi").alias("distanza_media_giorno"),
         F.avg((F.col("payment_type") == 1).cast("double")).alias("quota_carta_giorno")))

giornaliero = (orario
    .groupBy("zone_id", F.to_date("ora_inizio").alias("giorno"))
    .agg(F.sum("corse_1h").alias("corse_giorno"),
         F.sum("arrivi_1h").alias("arrivi_giorno"))
    .join(per_giorno, ["zone_id", "giorno"], "left"))

wg = Window.partitionBy("zone_id").orderBy("giorno")
wg7 = wg.rowsBetween(-6, 0)

fs_giornaliera = (giornaliero
    .withColumn("media_corse_7g",
                F.when(F.count(F.lit(1)).over(wg7) == 7, F.avg("corse_giorno").over(wg7)))
    .withColumn("feature_ts", F.date_add("giorno", 1).cast("timestamp"))
    .select("zone_id", "feature_ts", "corse_giorno", "arrivi_giorno", "media_corse_7g",
            "importo_medio_giorno", "distanza_media_giorno", "quota_carta_giorno"))

scrivi_feature_table(
    fs_giornaliera, "gold.fs_zona_giornaliera", chiavi=["zone_id"], ts="feature_ts",
    commento="Feature giornaliere della zona di partenza. feature_ts = mezzanotte successiva al giorno aggregato",
    commenti_colonne={
        "feature_ts": "Mezzanotte successiva al giorno aggregato: istante da cui i valori sono noti",
        "corse_giorno": "Corse partite dalla zona nel giorno concluso",
        "arrivi_giorno": "Corse arrivate nella zona nel giorno concluso",
        "media_corse_7g": "Media giornaliera delle corse negli ultimi 7 giorni (NULL se finestra incompleta)",
        "importo_medio_giorno": "Importo totale medio in dollari delle corse partite dalla zona e concluse nel giorno",
        "distanza_media_giorno": "Distanza media in miglia delle corse partite dalla zona e concluse nel giorno",
        "quota_carta_giorno": "Quota di corse pagate con carta di credito tra quelle partite dalla zona e concluse nel giorno",
    },
    tag={"layer": "gold", "entita": "zona", "granularita": "giornaliera"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Verifiche
# MAGIC 1. Unicità della chiave primaria (i vincoli in Unity Catalog sono
# MAGIC    informativi: non vengono imposti dal motore, quindi vanno controllati).
# MAGIC 2. Coerenza delle finestre mobili: `corse_3h` a `t` deve essere la somma di
# MAGIC    `corse_1h` a `t`, `t-1h`, `t-2h`.
# MAGIC 3. Conservazione dei conteggi: le corse nella tabella oraria devono essere
# MAGIC    uguali alle righe del Silver.

# COMMAND ----------

risultati = []
for nome, chiave in [("gold.fs_zona_anagrafica", ["zone_id"]),
                     ("gold.fs_zona_giornaliera", ["zone_id", "feature_ts"]),
                     ("gold.fs_zona_oraria", ["zone_id", "feature_ts"])]:
    t = spark.table(nome)
    risultati.append((nome, t.count(), t.select(*chiave).distinct().count()))
display(spark.createDataFrame(risultati, "tabella STRING, righe LONG, chiavi_distinte LONG"))

fo = spark.table("gold.fs_zona_oraria")
w_chk = Window.partitionBy("zone_id").orderBy("feature_ts")
errori_3h = (fo
    .withColumn("atteso", F.col("corse_1h") + F.lag("corse_1h", 1).over(w_chk)
                                            + F.lag("corse_1h", 2).over(w_chk))
    .filter(F.col("corse_3h").isNotNull() & (F.col("corse_3h") != F.col("atteso")))
    .count())

print("Righe con corse_3h incoerente:", errori_3h)
print("Corse nel Silver:        ", trips.count())
print("Somma corse_1h nel Gold: ", fo.agg(F.sum("corse_1h")).first()[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Esempio: una zona in un'ora di punta

# COMMAND ----------

display(spark.sql("""
    SELECT o.zone_id, z.zone_name, o.feature_ts, o.corse_1h, o.corse_3h, o.corse_24h,
           o.corse_1h_giorno_prima, o.corse_1h_settimana_prima, o.velocita_media_1h
    FROM gold.fs_zona_oraria o JOIN silver.zone z USING (zone_id)
    WHERE z.zone_name = 'Midtown Center'
      AND o.feature_ts BETWEEN '2026-07-15 07:00:00' AND '2026-07-15 12:00:00'
    ORDER BY o.feature_ts
"""))

# COMMAND ----------

# Metadati registrati in Unity Catalog (chiave primaria, commenti, clustering)
display(spark.sql("DESCRIBE TABLE EXTENDED gold.fs_zona_oraria"))

# COMMAND ----------

for t in ["gold.fs_zona_anagrafica", "gold.fs_zona_giornaliera", "gold.fs_zona_oraria"]:
    d = spark.sql(f"DESCRIBE DETAIL {t}").select("numFiles", "sizeInBytes").first()
    print(t, d.numFiles, "file,", round(d.sizeInBytes / 1e6, 2), "MB")

display(spark.sql("DESCRIBE HISTORY gold.fs_zona_oraria")
        .select("version", "timestamp", "operation"))