<div align="center">

<img src="https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white"/>
<img src="https://img.shields.io/badge/Apache_Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white"/>
<img src="https://img.shields.io/badge/Delta_Lake-00ADD8?style=for-the-badge&logo=delta&logoColor=white"/>
<img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"/>
<img src="https://img.shields.io/badge/BigQuery-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white"/>
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>

# Fraud Detection Lakehouse
### Medallion Architecture on GCP · Databricks · Delta Lake · MLflow

*Production-grade ELT pipeline · 25+ PySpark features · XGBoost + MLflow Model Registry · BigQuery serving layer*

</div>

---

## Overview

End-to-end **fraud detection data platform** for a retail bank, built on GCP with Databricks at the core.

Raw transactions are ingested from PostgreSQL into GCS (Bronze), cleaned and typed in Databricks (Silver), enriched with 25+ behavioral fraud features (Gold), used to train an XGBoost model tracked in MLflow, and finally exported to BigQuery for BI dashboards and compliance reporting.

**Business impact simulated:**
- Detect fraudulent transactions before settlement (T+1 latency)
- Reduce false positives via RFM-style behavioral scoring
- Full audit trail via Delta Lake time travel
- Model auto-promotion when ROC-AUC ≥ 0.85

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCE                                    │
│              PostgreSQL — transactions, accounts, merchants              │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │  Python batch extract (psycopg2)
                               │  Incremental by watermark · Schema validation
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               GCS BRONZE  gs://bucket/bronze/transactions/              │
│                    JSON Lines · Partitioned by day                      │
│               Raw data preserved as-is (immutable source of truth)      │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATABRICKS  (PySpark)                                │
│                                                                         │
│  Bronze → Silver                                                        │
│  ├── Schema enforcement (StructType)                                    │
│  ├── Type casting + null handling                                       │
│  ├── Enum standardization (channel, currency, tx_type)                  │
│  ├── Deduplication via Window + row_number()                            │
│  ├── Data quality flags (_dq_*)                                         │
│  └── MERGE upsert → Delta Lake Silver                                   │
│                                                                         │
│  Silver → Gold  (Feature Engineering)                                   │
│  ├── Velocity features  (tx count/amount over 1h, 6h, 24h, 7d)         │
│  ├── Statistical features  (mean, std, z-score over 30d)               │
│  ├── Behavioral features  (night tx, weekend, new merchant, time gap)   │
│  ├── Account features  (lifetime stats, historical fraud rate)          │
│  └── MERGE upsert → Delta Lake Gold (Z-ORDER on account_id)            │
│                                                                         │
│  ML Training  (weekly)                                                  │
│  ├── Load Gold features (180d lookback)                                 │
│  ├── SMOTE oversampling (fraud typically < 1%)                          │
│  ├── XGBoost training + cross-validation                               │
│  ├── MLflow: log params · metrics · feature importance · model          │
│  └── Auto-promote to Production if ROC-AUC ≥ 0.85                      │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          MLflow Model Registry    BigQuery Serving Layer
          fraud_classifier         ├── fraud_features (daily snapshot)
          Production / Staging     ├── fraud_alerts   (score ≥ 0.8)
                                   └── account_risk_profile
                                              │
                                              ▼
                                   BI Dashboards · Compliance · API
```

**Databricks Job — daily at 03:00 CET:**

```
ingest_bronze
      │
      ▼
bronze_to_silver ──────────────────────────────────► bq_export
      │                                                    ▲
      ▼                                                    │
silver_to_gold ─────────────────────────────────────────► ┘
      │
      ▼ (weekly)
ml_training  →  MLflow Model Registry  →  Production promotion
```

---

## Tech Stack

| Layer | Tool | Role |
|---|---|---|
| Compute | **Databricks** (PySpark 3.5) | All transformations + ML training |
| Storage format | **Delta Lake** | ACID, time travel, Z-ordering, MERGE |
| Raw storage | **GCS** (Bronze/Silver/Gold) | Medallion architecture |
| ML tracking | **MLflow** | Experiment tracking + Model Registry |
| ML model | **XGBoost** + SMOTE | Fraud classifier |
| Serving | **BigQuery** | BI layer + compliance |
| Source DB | **PostgreSQL** | Transactional source |
| Orchestration | **Databricks Jobs** | DAG with retries + email alerts |
| CI/CD | **GitHub Actions** | Lint + type check + pytest |
| Languages | **Python 3.11** | All pipeline code |

---

## Project Structure

```
fraud-detection-lakehouse/
│
├── src/
│   ├── common/
│   │   ├── logger.py           # Structured JSON logging (GCP Cloud Logging compatible)
│   │   ├── config.py           # Config loader — YAML + env vars, typed dataclasses
│   │   └── exceptions.py       # Custom exception hierarchy
│   │
│   ├── ingestion/
│   │   └── transaction_ingester.py   # PostgreSQL → GCS Bronze
│   │                                 # Incremental · batched · schema validated
│   │
│   ├── bronze_to_silver/
│   │   └── transformer.py      # PySpark Bronze → Silver
│   │                           # MERGE upsert · dedup · quality flags
│   │
│   ├── silver_to_gold/
│   │   └── feature_engineer.py # 25+ fraud features
│   │                           # Velocity · statistical · behavioral · account
│   │
│   ├── ml/
│   │   └── trainer.py          # XGBoost + MLflow
│   │                           # SMOTE · evaluation · Model Registry promotion
│   │
│   ├── serving/
│   │   └── bigquery_exporter.py # Gold → BigQuery
│   │                            # fraud_features · fraud_alerts · account_risk_profile
│   │
│   └── jobs.py                 # Databricks job entrypoints (argparse)
│
├── tests/
│   └── unit/
│       ├── test_bronze_to_silver.py    # 11 unit tests — cleaning, dedup, flags
│       └── test_feature_engineering.py # behavioral + account feature tests
│
├── config/
│   ├── dev.yml                 # Dev environment config
│   └── prod.yml                # Prod environment (env var references)
│
├── databricks/
│   └── jobs/
│       └── daily_pipeline.json # Databricks Job definition (5 tasks)
│
├── .github/workflows/
│   └── ci.yml                  # lint → type check → pytest → coverage
│
└── requirements.txt
```

---

## Feature Engineering — 25+ Fraud Signals

### Velocity features (sliding time windows)

| Feature | Window | Description |
|---|---|---|
| `tx_count_1h` | 1 hour | Transactions in last hour — card testing signal |
| `tx_amount_1h` | 1 hour | Total amount in last hour |
| `tx_count_6h` | 6 hours | Medium-term velocity |
| `tx_count_24h` | 24 hours | Daily velocity |
| `tx_merchants_24h` | 24 hours | Unique merchants in 24h — dispersion signal |
| `tx_count_7d` | 7 days | Weekly baseline |

### Statistical features

| Feature | Description |
|---|---|
| `amount_mean_30d` | Rolling 30-day mean amount per account |
| `amount_std_30d` | Rolling 30-day std — measures volatility |
| `amount_zscore` | (amount - mean) / std — key anomaly signal |
| `amount_ratio_to_max` | Current / max amount in 30d |

### Behavioral features

| Feature | Description |
|---|---|
| `is_night_transaction` | Between 23:00 and 05:00 |
| `is_weekend` | Saturday or Sunday |
| `time_since_last_tx_seconds` | Time gap from previous transaction |
| `is_new_merchant` | First transaction with this merchant |
| `merchant_category_diversity_7d` | Unique categories in 7 days |

### Account lifetime features

| Feature | Description |
|---|---|
| `account_historical_fraud_rate` | % of past transactions flagged as fraud |
| `account_total_tx` | Lifetime transaction count |
| `account_distinct_countries` | Countries transacted in |
| `account_age_days` | Days since first transaction |

---

## ML Pipeline

**Model:** XGBoost classifier with SMOTE oversampling

**Class imbalance handling:**
- Fraud typically represents < 1% of transactions
- SMOTE upsamples fraud to 10% in training set
- `scale_pos_weight` computed from original class distribution

**Evaluation metrics (production thresholds):**

| Metric | Threshold to promote |
|---|---|
| ROC-AUC | ≥ 0.85 |
| Average Precision | ≥ 0.70 |
| F1 (fraud class) | ≥ 0.65 |

**MLflow tracking:**
- All hyperparameters logged
- All metrics logged (including confusion matrix components)
- Feature importance CSV as artifact
- Model registered in MLflow Model Registry
- Auto-promotion to Production when thresholds met
- Previous Production version archived automatically

---

## Delta Lake Design

**Bronze** — append-only, immutable raw data. Never modified after write.

**Silver** — MERGE upsert on `transaction_id`. Handles late arrivals and corrections.

**Gold** — MERGE upsert on `transaction_id`. OPTIMIZE + Z-ORDER on `account_id, transaction_date` for fast BI queries.

```python
# Z-ordering example — dramatically reduces data scanned for account queries
OPTIMIZE delta.`gs://bucket/gold/features`
ZORDER BY (account_id, transaction_date)
```

**Time travel** — Delta Lake retains 30 days of history by default:

```python
# Audit: read Gold as of 7 days ago
df = spark.read.format("delta") \
    .option("versionAsOf", "7 days ago") \
    .load("gs://bucket/gold/features")
```

---

## Local Setup

### Prerequisites

- Python 3.11+
- Java 11+ (required for PySpark)
- GCP project with GCS + BigQuery enabled
- Databricks Community Edition account (free)

```bash
# Clone
git clone https://github.com/JEMALIACHRAF/fraud-detection-lakehouse
cd fraud-detection-lakehouse

# Virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp config/dev.yml config/local.yml
# Edit config/local.yml with your GCP project and DB credentials

# Set secrets as env vars
export GCP_PROJECT_ID="your-project-id"
export DB_PASSWORD="your-db-password"
export DATABRICKS_HOST="https://community.cloud.databricks.com"
export DATABRICKS_TOKEN="your-token"

# Run unit tests
pytest tests/unit/ -v --cov=src
```

### Run individual pipeline stages

```bash
# Ingest Bronze
python -m src.jobs bronze_to_silver --env dev --date 2024-01-15

# Bronze → Silver
python -m src.jobs bronze_to_silver --env dev --date 2024-01-15

# Silver → Gold (feature engineering)
python -m src.jobs silver_to_gold --env dev --date 2024-01-15

# Train fraud model
python -m src.jobs ml_training --env dev --lookback-days 180

# Export to BigQuery
python -m src.jobs bq_export --env dev --date 2024-01-15
```

---

## Databricks Deployment

### Upload and run on Databricks Community Edition

```bash
# 1. Create a cluster on community.cloud.databricks.com
#    Runtime: 13.3 LTS ML (includes MLflow + Delta Lake)

# 2. Install dependencies on the cluster
#    Cluster → Libraries → Install New → PyPI:
#    xgboost==2.0.3
#    imbalanced-learn==0.11.0
#    google-cloud-storage==2.14.0

# 3. Upload src/ to DBFS
databricks fs cp -r src/ dbfs:/fraud-pipeline/src/ --overwrite

# 4. Upload config/
databricks fs cp -r config/ dbfs:/fraud-pipeline/config/ --overwrite

# 5. Run a job task
databricks runs submit --json @databricks/jobs/daily_pipeline.json
```

---

## CI/CD

Every pull request runs:

```
Ruff lint → mypy type check → pytest unit tests → coverage report
```

Coverage threshold: **70% minimum** on `src/`.

---

## Key Engineering Decisions

**Medallion architecture** — Bronze/Silver/Gold separation ensures:
- Raw data is never modified (Bronze = audit log)
- Cleaning is separated from feature engineering
- Each layer can be reprocessed independently

**MERGE instead of overwrite** — idempotent writes allow safe re-runs without duplicates, critical for at-least-once delivery semantics.

**Z-ordering on account_id** — fraud queries always filter by account. Z-ordering co-locates data for the same account, reducing GCS reads by up to 90% on large datasets.

**SMOTE on training only** — oversampling applied only to training split, never to test set. Prevents overly optimistic evaluation metrics.

**Structured JSON logging** — all logs are machine-parseable and compatible with GCP Cloud Logging. Enables log-based alerting and dashboards without code changes.

---
## Author

**Achraf Jemali** — Data & AI Engineer


[![GitHub](https://img.shields.io/badge/GitHub-JEMALIACHRAF-black?logo=github)](https://github.com/JEMALIACHRAF)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Achraf_Jemali-0077B5?logo=linkedin)](https://linkedin.com/in/achraf-jemali-54a417239)

---

<div align="center">

Built with Databricks · Delta Lake · MLflow · GCP · 

</div>
