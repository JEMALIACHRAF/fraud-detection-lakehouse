<div align="center">

<img src="https://img.shields.io/badge/Azure_Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white"/>
<img src="https://img.shields.io/badge/Delta_Lake-00ADD8?style=for-the-badge&logo=delta&logoColor=white"/>
<img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"/>
<img src="https://img.shields.io/badge/BigQuery-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white"/>
<img src="https://img.shields.io/badge/PySpark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white"/>
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>

# Fraud Detection Lakehouse
### Multi-Cloud · Azure Databricks · Delta Lake · MLflow · BigQuery

[![CI](https://github.com/JEMALIACHRAF/fraud-detection-lakehouse/actions/workflows/ci.yml/badge.svg)](https://github.com/JEMALIACHRAF/fraud-detection-lakehouse/actions/workflows/ci.yml)

*150,000 transactions · 30 fraud features · 5 ML models · ROC-AUC 0.97 · Azure + GCP*

</div>

---

## Overview

End-to-end **fraud detection data platform** for a retail bank, built on a multi-cloud architecture:

- **Azure Databricks** — Spark compute, Delta Lake, MLflow tracking
- **Azure Blob Storage** — Bronze / Silver / Gold medallion layers
- **Google BigQuery** — Serving layer for BI and compliance reporting
- **Looker Studio** — Real-time fraud analytics dashboard

**Business impact simulated:**
- Detect fraudulent transactions before settlement (T+1 latency)
- Compare 5 ML models to find the optimal fraud detector
- Full audit trail via Delta Lake time travel
- Automated CI/CD via GitHub Actions

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                  │
│   BigQuery Public Data (Chicago Taxi)  +  Synthetic Banking Data    │
│                    150,000 transactions · 1.32% fraud rate          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  Python ingestion scripts
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              AZURE BLOB STORAGE — Medallion Architecture            │
│                                                                      │
│  Bronze  wasbs://bronze@fraudlakehouse...                           │
│  ├── JSON Lines, raw untouched                                      │
│  └── Partitioned by year/month/day                                  │
│                                                                      │
│  Silver  wasbs://silver@fraudlakehouse...  (Delta Lake)             │
│  ├── Clean + typed + deduplicated                                   │
│  └── MERGE upsert, partitioned by transaction_date                 │
│                                                                      │
│  Gold    wasbs://gold@fraudlakehouse...    (Delta Lake)             │
│  ├── 30 fraud detection features                                    │
│  └── Velocity · Statistical · Behavioral · Account                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
         Azure Databricks            Google BigQuery
         ├── PySpark transforms      ├── fraud_alerts (1,983 rows)
         ├── Delta Lake ACID         ├── model_performance (13 rows)
         ├── MLflow experiments      └── account_risk_profile (12,089 rows)
         └── 5 ML models                      │
                    │                         ▼
                    └──────────────► Looker Studio Dashboard
```

**CI/CD:**
```
PR → CI (lint + 17 unit tests)     Merge/Manual → CD (deploy to Databricks)
```

---

## Tech Stack

| Layer | Tool | Version |
|---|---|---|
| Cloud Compute | Azure Databricks | 13.3 LTS ML |
| Storage Format | Delta Lake | 3.0.0 |
| Raw Storage | Azure Blob Storage | — |
| Serving Layer | Google BigQuery | — |
| ML Tracking | MLflow | 2.19.0 |
| Imbalance | SMOTE (imbalanced-learn) | 0.12.3 |
| CI/CD | GitHub Actions | — |
| Dashboard | Looker Studio | — |
| Language | Python 3.11 + PySpark 3.5 | — |

---

## Project Structure

```
fraud-detection-lakehouse/
│
├── src/                              # Production code
│   ├── common/
│   │   ├── logger.py                 # Structured JSON logging
│   │   ├── config.py                 # Typed config loader (YAML + env vars)
│   │   └── exceptions.py            # Custom exception hierarchy
│   ├── ingestion/
│   │   └── transaction_ingester.py  # PostgreSQL → Azure Blob Bronze
│   ├── bronze_to_silver/
│   │   └── transformer.py           # PySpark clean + Delta Lake MERGE
│   ├── silver_to_gold/
│   │   └── feature_engineer.py      # 30 fraud features
│   ├── ml/
│   │   └── trainer.py               # 5 models + MLflow + SMOTE
│   ├── serving/
│   │   └── bigquery_exporter.py     # Gold → BigQuery
│   └── jobs.py                      # Databricks job entrypoints
│
├── scripts/                          # Infrastructure + data scripts
│   ├── setup_azure_databricks.py    # IaC — create Databricks workspace
│   ├── setup_azure_storage.py       # IaC — create Azure Blob Storage
│   ├── setup_databricks_cluster.py  # IaC — create Databricks cluster
│   ├── create_databricks_notebook.py # Deploy pipeline notebook
│   ├── upload_to_dbfs.py            # Upload src/ to Databricks
│   ├── upload_data_to_blob.py       # Upload Bronze data to Azure
│   ├── upload_gcp_credentials.py    # Upload GCP credentials to Databricks
│   ├── extract_taxi_trips.py        # Extract from BigQuery public data
│   ├── generate_synthetic_transactions.py # Generate 100k transactions
│   ├── create_bq_analytics_views.py # IaC — create 6 BigQuery views
│   └── run_local_pipeline.py        # Run full pipeline locally
│
├── tests/unit/
│   ├── test_bronze_to_silver.py     # 12 PySpark unit tests
│   └── test_feature_engineering.py  # 5 feature tests
│
├── config/
│   ├── dev.yml                      # Dev environment config
│   └── prod.yml                     # Prod environment config
│
├── .github/workflows/
│   ├── ci.yml                       # Lint + tests on every PR
│   └── cd.yml                       # Deploy to Databricks (manual)
│
└── .env.example                     # Environment variables template
```

---

## ML Results — 5 Models Comparison

| Model | ROC-AUC | Avg Precision | F1 | Recall | Precision |
|---|---|---|---|---|---|
| **Random Forest** ✓ | **0.9734** | **0.6765** | **0.701** | 0.586 | 0.872 |
| XGBoost | 0.9663 | 0.6630 | 0.238 | 0.810 | 0.139 |
| Gradient Boosting | 0.9617 | 0.6299 | 0.197 | 0.787 | 0.113 |
| Neural Network (128-64-32) | 0.7790 | 0.5623 | 0.634 | 0.558 | 0.735 |
| Logistic Regression | 0.7883 | 0.1039 | 0.056 | 0.644 | 0.029 |

**Random Forest wins** — highest F1 (0.70) + precision (0.87): 87% of fraud alerts are real, minimizing costly false positives for fraud analysts.

All runs tracked in MLflow with hyperparameters, metrics, feature importance, and model signatures.

---

## Feature Engineering — 30 Fraud Signals

**Velocity (sliding windows)**
```
tx_count_24h     → card testing signal — many tx in one day
tx_amount_24h    → cumulative amount in 24h
tx_count_7d      → weekly volume baseline
tx_amount_7d     → weekly spending pattern
```

**Statistical (30-day rolling)**
```
amount_mean_30d  → account's normal spending level
amount_std_30d   → spending volatility
amount_zscore    → (amount - mean) / std — key anomaly signal
```

**Behavioral**
```
is_night_transaction      → 23:00 - 05:00
is_weekend                → Saturday or Sunday
time_since_last_tx_seconds → card testing signal (rapid succession)
```

---

## Setup Guide

### Prerequisites

- Python 3.11+
- Azure account ($200 free credit at azure.microsoft.com/free)
- GCP account with BigQuery enabled
- Azure CLI (`az`) and Google Cloud SDK (`gcloud`) installed

### Step 1 — Clone and configure

```bash
git clone https://github.com/JEMALIACHRAF/fraud-detection-lakehouse
cd fraud-detection-lakehouse

python -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your credentials
```

### Step 2 — Configure `.env`

```env
# Azure
AZURE_SUBSCRIPTION_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
AZURE_STORAGE_ACCOUNT=fraudlakehouse
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=fraudlakehouse;AccountKey=...

# Databricks
DATABRICKS_HOST=https://adb-xxxx.azuredatabricks.net
DATABRICKS_TOKEN=dapi_xxxx
DATABRICKS_CLUSTER_ID=xxxx-xxxxxx-xxxxxxxx

# GCP
GCP_PROJECT_ID=your-gcp-project-id
```

### Step 3 — Provision Azure infrastructure

```bash
# Login
az login

# Create Databricks workspace (~10 min)
python scripts/setup_azure_databricks.py

# Create Azure Blob Storage (Bronze/Silver/Gold/MLflow containers)
python scripts/setup_azure_storage.py

# Add AZURE_STORAGE_CONNECTION_STRING to .env from script output
```

### Step 4 — Generate and upload data

```bash
# 100k synthetic banking transactions (30 seconds)
python scripts/generate_synthetic_transactions.py --rows 100000 --output-dir data

# 50k real taxi trips from BigQuery public data
python scripts/extract_taxi_trips.py --project your-gcp-project-id --limit 50000

# Upload to Azure Blob Storage Bronze container
python scripts/upload_data_to_blob.py
```

### Step 5 — Deploy to Databricks

```bash
# Create cluster (or manually via Databricks UI)
# Recommended: Standard_D4s_v5, Runtime 13.3 LTS ML, Single node

# Add DATABRICKS_CLUSTER_ID to .env

# Upload source code to Databricks Workspace
python scripts/upload_to_dbfs.py

# Upload GCP credentials for BigQuery export
python scripts/upload_gcp_credentials.py

# Create pipeline notebook on Databricks
python scripts/create_databricks_notebook.py
```

### Step 6 — Run the pipeline on Databricks

Open Databricks workspace → Workspace → Users → your@email → fraud-pipeline → pipeline_notebook

Attach cluster and run cells in order:

```
Cell 0  → %pip install dependencies (wait for restart)
Cell 1  → Configure Azure + GCP credentials + paths
Cell 2  → Read Bronze (150,000 rows)
Cell 3  → Bronze → Silver (Delta Lake, clean + type)
Cell 4  → Silver → Gold (30 fraud features)
Cell 5  → Train 5 ML models + MLflow tracking
Cell 6  → MLflow experiment dashboard
Cell 7  → Delta Lake time travel demo
Cell 8  → Export to BigQuery (3 tables)
```

### Step 7 — Create BigQuery analytical views

```bash
python scripts/create_bq_analytics_views.py
# Creates 6 views: vw_daily_fraud_summary, vw_fraud_by_country,
# vw_fraud_by_channel, vw_top_risky_accounts, vw_model_comparison,
# vw_hourly_fraud_pattern
```

Connect Looker Studio to `fraud_detection` dataset:
```
https://lookerstudio.google.com/
```

---

## Run Locally (No Cloud Required)

```bash
# Windows setup — required for PySpark
mkdir C:\hadoop\bin
curl -L -o C:\hadoop\bin\winutils.exe https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.5/bin/winutils.exe
curl -L -o C:\hadoop\bin\hadoop.dll https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.5/bin/hadoop.dll

# Generate data
python scripts/generate_synthetic_transactions.py --rows 100000 --output-dir data

# Set environment (Windows)
set PYSPARK_PYTHON=C:\path\to\python.exe
set HADOOP_HOME=C:\hadoop

# Run full pipeline locally (Bronze → Silver → Gold → ML → Parquet)
python scripts/run_local_pipeline.py --data-dir data

# Open MLflow UI
mlflow ui --backend-store-uri file:///C:/path/to/mlflow_runs
# Navigate to http://localhost:5000
```

---

## Unit Tests

```bash
# Windows — set Python for PySpark workers
set PYSPARK_PYTHON=C:\path\to\python.exe

# Run 17 unit tests
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ -v --cov=src/bronze_to_silver --cov=src/silver_to_gold

# Expected: 17 passed in ~120s
```

**Tests cover:**
- Deduplication on `transaction_id` (keep latest `created_at`)
- Currency normalization (`"usd"` → `"USD"`)
- Channel normalization (`"ATM"` → `"atm"`)
- Null handling (`merchant_category` → `"unknown"`, `city` → `"UNKNOWN"`)
- Negative amount removal
- Data quality flags (`_dq_amount_suspicious`, `_dq_unknown_channel`)
- Night transaction detection (23:00+ flagged)
- New merchant detection (first transaction with merchant)
- Account fraud rate aggregation

---

## CI/CD

### CI — GitHub Actions (every Pull Request)

```
ruff lint → pytest 17 unit tests → coverage report
```

### CD — GitHub Actions (manual trigger)

1. Go to **Actions** → **"CD — Deploy to Azure Databricks"**
2. Click **"Run workflow"**
3. Choose `run_pipeline: false` (code deploy only) or `true` (deploy + run pipeline)

**Required GitHub Secrets:**
```
DATABRICKS_HOST        → https://adb-xxxx.azuredatabricks.net
DATABRICKS_TOKEN       → dapi_xxxx
DATABRICKS_CLUSTER_ID  → xxxx-xxxxxx-xxxxxxxx
```

Add secrets at: `Settings → Secrets and variables → Actions → New repository secret`

---

## Delta Lake Features Used

**ACID transactions** — concurrent writes never corrupt data:
```python
# MERGE upsert — idempotent, safe to re-run
delta_table.merge(df, "silver.transaction_id = updates.transaction_id") \
    .whenMatchedUpdateAll() \
    .whenNotMatchedInsertAll() \
    .execute()
```

**Time travel** — read data at any past version:
```python
# Audit: what did the data look like at version 0?
df = spark.read.format("delta").option("versionAsOf", 0).load(SILVER_PATH)
```


---

## Why Multi-Cloud (Azure + GCP)?

```
Azure Databricks        GCP BigQuery
────────────────        ─────────────────────────────────────
Best Spark managed      Best serverless SQL analytics engine
Delta Lake native       Looker Studio native integration
MLflow integrated       Free tier: 1TB queries/month
$200 free credit        Already configured (dbt project)
Dominant in EU banks    Standard for analytics in data teams
```

This architecture mirrors what large European banks actually use in production — Azure for compute and data engineering, GCP/BigQuery for analytics and reporting.

---


## Author

**Achraf Jemali** — Data & AI Engineer


[![GitHub](https://img.shields.io/badge/GitHub-JEMALIACHRAF-black?logo=github)](https://github.com/JEMALIACHRAF)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Achraf_Jemali-0077B5?logo=linkedin)](https://linkedin.com/in/achraf-jemali-54a417239)

---

<div align="center">

Built with Databricks · Delta Lake · MLflow · GCP · 

</div>
