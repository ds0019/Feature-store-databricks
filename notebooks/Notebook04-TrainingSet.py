# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "databricks-feature-engineering",
# ]
# ///
# MAGIC %md
# MAGIC # 04 — Generazione del training set
# MAGIC Costruisce il training set del caso d'uso (previsione della domanda per zona e ora):
# MAGIC
# MAGIC 1. **tabella delle etichette** (*spine*): una riga per zona e per ora `t`, con
# MAGIC    l'etichetta = corse partite dalla zona in `[t, t+1h)`, calcolata dal Silver;
# MAGIC 2. **join point-in-time** con le Feature Table tramite `FeatureEngineeringClient`:
# MAGIC    per ogni riga si prende la riga più recente con `feature_ts <= t`;
# MAGIC 3. **verifica**: la stessa join fatta a mano deve dare lo stesso risultato
# MAGIC    (e chiarisce se Databricks usa `<=` oppure `<`);
# MAGIC 4. **prova del leakage**: con il timestamp all'inizio della finestra la feature
# MAGIC    `corse_1h` coinciderebbe con l'etichetta;
# MAGIC 5. **salvataggio** del training set come tabella Delta.
# MAGIC
# MAGIC Le Feature Table non vengono modificate: il notebook le legge soltanto.

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

# Stessa convenzione del Notebook 03: gli istanti TLC sono ora locale di New York
spark.conf.set("spark.sql.session.timeZone", "America/New_York")

FT_ORARIA      = "workspace.gold.fs_zona_oraria"
FT_GIORNALIERA = "workspace.gold.fs_zona_giornaliera"
FT_ANAGRAFICA  = "workspace.gold.fs_zona_anagrafica"
TRAINING_SET   = "workspace.gold.training_domanda_oraria"

FEATURE_ORARIE = ["corse_1h", "arrivi_1h", "corse_3h", "corse_24h", "arrivi_24h",
                  "corse_1h_giorno_prima", "corse_1h_settimana_prima",
                  "durata_media_1h", "distanza_media_1h", "velocita_media_1h",
                  "importo_medio_1h"]
FEATURE_GIORNALIERE = ["corse_giorno", "arrivi_giorno", "media_corse_7g",
                       "importo_medio_giorno", "distanza_media_giorno",
                       "quota_carta_giorno"]
FEATURE_STATICHE = ["borough", "service_zone", "is_aeroporto"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Tabella delle etichette
# MAGIC Le etichette non stanno nelle Feature Table (che non devono dipendere dal caso d'uso):
# MAGIC si calcolano dal Silver. Si usa la stessa griglia densa zona × ora del Notebook 03,
# MAGIC così anche le ore senza corse compaiono, con etichetta 0.
# MAGIC `t` è l'**inizio** dell'ora da prevedere: l'istante in cui si fa la previsione.

# COMMAND ----------

trips = spark.table("silver.trips")

g_min, g_max = trips.agg(F.min("pickup_date"), F.max("pickup_date")).first()
n_ore = ((g_max - g_min).days + 1) * 24
inizio = g_min.strftime("%Y-%m-%d 00:00:00")

ore = spark.range(n_ore).select(
    F.expr(f"timestampadd(HOUR, CAST(id AS INT), TIMESTAMP_NTZ'{inizio}')").alias("ora_inizio"))

corse_per_ora = (trips
    .groupBy(F.col("pu_zone_id").alias("zone_id"),
             F.expr("make_timestamp_ntz(year(pickup_ts), month(pickup_ts), day(pickup_ts), "
                    "hour(pickup_ts), 0, 0)").alias("ora_inizio"))
    .agg(F.count("*").alias("etichetta")))

etichette = (spark.table("silver.zone").select("zone_id")
    .crossJoin(ore)
    .join(corse_per_ora, ["zone_id", "ora_inizio"], "left")
    .fillna(0, subset=["etichetta"])
    .withColumn("t", F.col("ora_inizio").cast("timestamp"))
    .select("zone_id", "t", "etichetta"))

print("Righe della tabella delle etichette:", etichette.count())
print("Somma delle etichette (deve essere uguale alle corse del Silver):",
      etichette.agg(F.sum("etichetta")).first()[0], "vs", trips.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Join point-in-time con il Feature Engineering Client
# MAGIC Una `FeatureLookup` per ciascuna Feature Table:
# MAGIC - per le tabelle temporali `timestamp_lookup_key="t"` attiva la join point-in-time;
# MAGIC - per la tabella anagrafica basta la chiave `zone_id`.

# COMMAND ----------

fe = FeatureEngineeringClient()

feature_lookups = [
    FeatureLookup(table_name=FT_ORARIA, feature_names=FEATURE_ORARIE,
                  lookup_key="zone_id", timestamp_lookup_key="t"),
    FeatureLookup(table_name=FT_GIORNALIERA, feature_names=FEATURE_GIORNALIERE,
                  lookup_key="zone_id", timestamp_lookup_key="t"),
    FeatureLookup(table_name=FT_ANAGRAFICA, feature_names=FEATURE_STATICHE,
                  lookup_key="zone_id"),
]

training_set = fe.create_training_set(df=etichette,
                                      feature_lookups=feature_lookups,
                                      label="etichetta")
training_df = training_set.load_df()

print("Righe del training set:", training_df.count())
print("Colonne:", training_df.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Valori mancanti
# MAGIC All'inizio del periodo alcune feature non esistono ancora (finestre incomplete,
# MAGIC nessun giorno precedente al 1° luglio): restano `NULL`.

# COMMAND ----------

feature_cols = FEATURE_ORARIE + FEATURE_GIORNALIERE + FEATURE_STATICHE
display(training_df.select(
    [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in feature_cols]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Esempio: Midtown Center, 15 luglio
# MAGIC L'etichetta delle 9:00 (corse tra 9:00 e 10:00) deve ricomparire come `corse_1h`
# MAGIC nella riga delle 10:00: un'ora dopo, quel valore è diventato noto.

# COMMAND ----------

display(training_df
    .join(spark.table("silver.zone").select("zone_id", "zone_name"), "zone_id")
    .filter((F.col("zone_name") == "Midtown Center")
            & F.col("t").between("2026-07-15 08:00:00", "2026-07-15 11:00:00"))
    .select("zone_name", "t", "etichetta", "corse_1h", "corse_3h", "corse_24h",
            "corse_giorno", "media_corse_7g", "borough")
    .orderBy("t"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Verifica con una join manuale
# MAGIC Con la griglia densa ogni zona ha una riga per ogni ora e per ogni giorno, quindi
# MAGIC "la riga più recente con `feature_ts <= t`" è semplicemente:
# MAGIC - tabella oraria: `feature_ts = t`;
# MAGIC - tabella giornaliera: `feature_ts =` mezzanotte del giorno di `t`.
# MAGIC
# MAGIC Si prova anche la variante con `<` stretto (`t - 1h` e mezzanotte del giorno di
# MAGIC `t - 1 secondo`): una sola delle due deve dare zero differenze.

# COMMAND ----------

oraria      = spark.table(FT_ORARIA)
giornaliera = spark.table(FT_GIORNALIERA)
anagrafica  = spark.table(FT_ANAGRAFICA)

def join_manuale(stretta=False):
    ts_ora    = F.expr("t - INTERVAL 1 HOUR") if stretta else F.col("t")
    ts_giorno = (F.expr("date_trunc('DAY', t - INTERVAL 1 SECOND)") if stretta
                 else F.expr("date_trunc('DAY', t)"))
    return (etichette
        .withColumn("_ts_ora", ts_ora)
        .withColumn("_ts_giorno", ts_giorno)
        .join(oraria.select("zone_id", F.col("feature_ts").alias("_ts_ora"), *FEATURE_ORARIE),
              ["zone_id", "_ts_ora"], "left")
        .join(giornaliera.select("zone_id", F.col("feature_ts").alias("_ts_giorno"),
                                 *FEATURE_GIORNALIERE),
              ["zone_id", "_ts_giorno"], "left")
        .join(anagrafica.select("zone_id", *FEATURE_STATICHE), ["zone_id"], "left")
        .drop("_ts_ora", "_ts_giorno"))

colonne = ["zone_id", "t", "etichetta"] + feature_cols
ts = training_df.select(colonne)
for nome, stretta in [("feature_ts <= t", False), ("feature_ts <  t", True)]:
    m = join_manuale(stretta).select(colonne)
    diverse = ts.exceptAll(m).count() + m.exceptAll(ts).count()
    print(f"Join manuale con {nome}: righe diverse dal training set = {diverse}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Prova del leakage
# MAGIC Se la riga oraria fosse etichettata con l'**inizio** della finestra, una join sul
# MAGIC timestamp le assegnerebbe come `corse_1h` proprio le corse dell'ora da prevedere.
# MAGIC Si confronta quante volte la feature coincide con l'etichetta nei due casi
# MAGIC (risultati per la Sezione 5.3).

# COMMAND ----------

ingenua = (etichette
    .join(oraria.select("zone_id",
                        F.expr("feature_ts - INTERVAL 1 HOUR").alias("t"),  # inizio della finestra
                        F.col("corse_1h").alias("corse_1h_ingenua")),
          ["zone_id", "t"], "left")
    .join(training_df.select("zone_id", "t", "corse_1h"), ["zone_id", "t"]))

display(ingenua.agg(
    F.avg((F.col("corse_1h_ingenua") == F.col("etichetta")).cast("double"))
        .alias("quota_uguali_ingenua"),
    F.avg((F.col("corse_1h") == F.col("etichetta")).cast("double"))
        .alias("quota_uguali_point_in_time"),
    F.corr("corse_1h_ingenua", "etichetta").alias("correlazione_ingenua"),
    F.corr("corse_1h", "etichetta").alias("correlazione_point_in_time")))

# COMMAND ----------


# Le ore senza corse (0 = 0) gonfiano la quota di righe uguali:
# si ripete il confronto sulle sole ore con almeno una corsa.

con_corse = F.col("etichetta") > 0
display(ingenua.agg(
    F.avg(con_corse.cast("double")).alias("quota_ore_con_corse"),
    F.avg(F.when(con_corse, (F.col("corse_1h") == F.col("etichetta")).cast("double")))
        .alias("uguali_pit_ore_con_corse")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Salvataggio
# MAGIC Il training set viene salvato come tabella Delta: resta riproducibile anche
# MAGIC quando le versioni precedenti delle Feature Table non sono più disponibili.

# COMMAND ----------

(training_df.write.mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(TRAINING_SET))

spark.sql(f"""COMMENT ON TABLE {TRAINING_SET} IS
  'Training set per la previsione della domanda: etichetta = corse partite dalla zona in [t, t+1h)'""")

print(TRAINING_SET, ":", spark.table(TRAINING_SET).count(), "righe")