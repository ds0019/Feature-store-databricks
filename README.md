# Feature Store batch su Databricks

Codice della tesi di laurea triennale in Informatica *Progettazione e
implementazione di un Feature Store batch su Databricks: modellazione dei dati e
analisi dei formati di archiviazione* (Università degli Studi di Torino, A.A.
2025/2026). Candidato: Sammy D'Agostino. Relatore: Prof. Luca Anselma.

Il progetto costruisce l'Offline Store di un Feature Store a partire dalle corse
dei taxi gialli di New York (luglio 2026). La pipeline segue l'architettura a
medaglione:
- **Bronze**: corse grezze;
- **Silver**: corse validate e deduplicate;
- **Gold**: tre Feature Table dell'entità zona.

Dalle Feature Table si genera poi un training set con join point-in-time.
Un notebook di benchmark confronta CSV, Parquet e Delta Lake e misura il
vantaggio del riuso delle feature.

## Contenuto

| File | Cosa fa | Sezione della tesi |
|---|---|---|
| `notebooks/Notebook00-CreationOfFiles.py` | Dal file Parquet della TLC crea la copia CSV nella landing zone e le copie Parquet e Delta per il benchmark | 4.2 |
| `notebooks/Notebook01-BronzeTable.py` | Carica il CSV in `bronze.trips_raw`, con lo schema del Parquet originale | 4.2 |
| `notebooks/Notebook02-SilverTable.py` | Regole di validità, deduplica e colonne derivate: `silver.trips` e `silver.quality_report` | 4.2 |
| `notebooks/Notebook03-GoldFeatureTables.py` | Tabella delle zone e Feature Table `gold.fs_zona_anagrafica`, `gold.fs_zona_giornaliera`, `gold.fs_zona_oraria` | 4.3 |
| `notebooks/Notebook04-TrainingSet.py` | Etichette, join point-in-time, verifiche, prova del leakage, salvataggio del training set | 4.4, 5.3 |
| `notebooks/Notebook05-Benchmark.py` | Confronto tra i formati, letture a freddo, generazione del training set con e senza Feature Store | 5.2, 5.3 |
| `job/Job_Medallion_FeatureStore.yml` | Job con i tre layer: NB01 → NB02 → NB03 | 4.1 |
| `job/Tesi_InteraPipeline.yaml` | Job che esegue il job precedente, poi NB04 e NB05 | — |

## Requisiti

- Un workspace Databricks con Unity Catalog. Il progetto è stato sviluppato su
  Databricks Free Edition, con calcolo serverless.
- La libreria `databricks-feature-engineering`, installata automaticamente da
  NB04 e NB05 con `%pip`.

## Dati

I dati non sono inclusi nel repository. Si scaricano dalla pagina
[TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page):

- Yellow Taxi Trip Records di luglio 2026: `yellow_tripdata_2026-07.parquet`;
- Taxi Zone Lookup Table: `taxi_zone_lookup.csv`.

Vanno caricati in un Volume di Unity Catalog con questa struttura:

```
/Volumes/<catalogo>/<schema>/<volume>/
├── yellow_tripdata_2026-07.parquet
└── lookup/
    └── taxi_zone_lookup.csv
```

Le altre cartelle del Volume (`landing/csv/`, `benchmark/`) sono create dai
notebook.

## Configurazione

1. Importare i file `.py` nel workspace con Workspace → Import. Databricks li
   riconosce come notebook grazie all'intestazione `# Databricks notebook source`.
2. Impostare il percorso del proprio Volume nella variabile `base` di NB00, NB01,
   NB03 e NB05. Il valore attuale è
   `/Volumes/experiments_sammy/default/nyc_taxi_yellow_trips`.
3. Le tabelle sono create negli schemi `bronze`, `silver` e `gold` del catalogo
   predefinito, che in Free Edition è `workspace`. NB04 e NB05 si riferiscono
   alle Feature Table con il nome completo `workspace.gold.*`.
4. Il batch elaborato è `2026-07`. Per un altro mese vanno cambiati:
   - il nome del file Parquet in NB00, NB01 e NB05;
   - `batch_id` in NB01 e NB02;
   - il periodo `start, end` in NB02;
   - `batch` in NB05;
   - il percorso del CSV in NB00.

## Ordine di esecuzione

1. **NB00**, una volta sola: simula il sistema esterno che deposita i dati nella
   landing zone.
2. **Job `Job_Medallion_FeatureStore`**: esegue NB01 → NB02 → NB03; ogni task
   parte solo se il precedente è terminato con successo.
3. **NB04**: genera il training set `gold.training_domanda_oraria`.
4. **NB05**: esegue i benchmark. Le letture a freddo (Parte C) richiedono circa
   30–35 minuti.

I passi 2–4 si possono eseguire insieme con il job `Tesi_InteraPipeline`.

I file YAML descrivono la configurazione dei job, esportata da Databricks. Per
ricrearli basta un job con un task per ogni passo, nell'ordine indicato. In
`Tesi_InteraPipeline.yaml` il primo task avvia il job
`Job_Medallion_FeatureStore` tramite il suo identificativo, che va sostituito con
quello del proprio workspace.

## Risultati attesi

Con il batch di luglio 2026:

| Oggetto | Righe |
|---|---|
| `bronze.trips_raw` | 3.530.109 |
| `silver.trips` | 3.276.089 |
| `gold.fs_zona_anagrafica` | 263 |
| `gold.fs_zona_giornaliera` | 8.153 |
| `gold.fs_zona_oraria` | 195.672 |
| `gold.training_domanda_oraria` | 195.672 esempi |

I tempi dei benchmark dipendono dall'ambiente di calcolo e possono differire da
quelli riportati nella tesi.