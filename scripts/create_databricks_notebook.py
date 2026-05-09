"""
Create the main pipeline notebook on Databricks Workspace.
Reads ALL credentials from .env — never hardcoded.
Includes GCP credentials loading for BigQuery export.
"""

import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat, Language

load_dotenv(Path(__file__).parents[1] / ".env")

client = WorkspaceClient(
    host=os.getenv("DATABRICKS_HOST"),
    token=os.getenv("DATABRICKS_TOKEN"),
)

me              = client.current_user.me()
user_email      = me.user_name
workspace_base  = f"/Users/{user_email}/fraud-pipeline"

STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT", "fraudlakehouse")
GCP_PROJECT     = os.getenv("GCP_PROJECT_ID", "projet-dbt-495310")

conn_str    = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
STORAGE_KEY = ""
if "AccountKey=" in conn_str:
    STORAGE_KEY = conn_str.split("AccountKey=")[1].split(";")[0]

NOTEBOOK_CONTENT = f'''# Databricks notebook source
# MAGIC %md
# MAGIC # Fraud Detection Lakehouse — Azure Databricks
# MAGIC **Stack:** PySpark · Delta Lake · MLflow · Azure Blob Storage · BigQuery
# MAGIC
# MAGIC ## Architecture
# MAGIC ```
# MAGIC Azure Blob (Bronze JSON)
# MAGIC     -> Silver (Delta Lake, clean + typed)
# MAGIC         -> Gold (Delta Lake, 30 fraud features)
# MAGIC             -> MLflow (5 models comparison + signatures)
# MAGIC                 -> BigQuery (fraud_alerts, model_performance, account_risk_profile)
# MAGIC ```

# COMMAND ----------

# MAGIC %md ## 0. Install dependencies

# COMMAND ----------

%pip install mlflow==2.10.0 xgboost==2.0.3 imbalanced-learn==0.12.3 scikit-learn==1.4.2 typing_extensions==4.9.0 google-cloud-bigquery google-cloud-bigquery-storage
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## 1. Configure Storage + Credentials

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from delta.tables import DeltaTable
import pandas as pd
import numpy as np
import json
import os

# ── Azure Blob Storage ────────────────────────────────────────────────────────
STORAGE_ACCOUNT = "{STORAGE_ACCOUNT}"
STORAGE_KEY     = "{STORAGE_KEY}"
GCP_PROJECT     = "{GCP_PROJECT}"
BQ_DATASET      = "fraud_detection"

spark.conf.set(
    f"fs.azure.account.key.{{STORAGE_ACCOUNT}}.blob.core.windows.net",
    STORAGE_KEY
)

BRONZE_PATH = f"wasbs://bronze@{{STORAGE_ACCOUNT}}.blob.core.windows.net/bronze/transactions"
SILVER_PATH = f"wasbs://silver@{{STORAGE_ACCOUNT}}.blob.core.windows.net/silver/transactions"
GOLD_PATH   = f"wasbs://gold@{{STORAGE_ACCOUNT}}.blob.core.windows.net/gold/features"

# ── GCP Credentials ───────────────────────────────────────────────────────────
gcp_creds_workspace = f"/Workspace/Users/{user_email}/fraud-pipeline/gcp_credentials.json"
gcp_creds_tmp       = "/tmp/gcp_credentials.json"

try:
    with open(gcp_creds_workspace) as f:
        creds = json.load(f)
    with open(gcp_creds_tmp, "w") as f:
        json.dump(creds, f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_creds_tmp
    print(f"GCP credentials loaded — project: {{creds.get('quota_project_id', GCP_PROJECT)}}")
except FileNotFoundError:
    print("WARNING: GCP credentials not found — BigQuery export will fail")
    print(f"Run: python scripts/upload_gcp_credentials.py")

print("\\nPaths configured:")
print(f"  BRONZE: {{BRONZE_PATH}}")
print(f"  SILVER: {{SILVER_PATH}}")
print(f"  GOLD:   {{GOLD_PATH}}")
print(f"  BQ:     {{GCP_PROJECT}}.{{BQ_DATASET}}")

files = dbutils.fs.ls(BRONZE_PATH)
print(f"\\nBronze partitions: {{len(files)}}")

# COMMAND ----------

# MAGIC %md ## 2. Read Bronze Layer

# COMMAND ----------

df_bronze   = spark.read.json(BRONZE_PATH)
row_count   = df_bronze.count()
fraud_count = df_bronze.filter("is_fraud = true").count()

print(f"Bronze rows:  {{row_count:,}}")
print(f"Fraud:        {{fraud_count:,}} ({{fraud_count/row_count:.2%}})")
print(f"Columns:      {{len(df_bronze.columns)}}")
display(df_bronze.limit(5))

# COMMAND ----------

# MAGIC %md ## 3. Bronze -> Silver (PySpark + Delta Lake)

# COMMAND ----------

VALID_TYPES    = ["purchase", "withdrawal", "transfer", "refund", "payment", "cash", "credit_card", "unknown"]
VALID_CHANNELS = ["online", "atm", "pos", "mobile", "wire", "taxi"]

df_silver = (
    df_bronze
    .withColumn("transaction_timestamp", F.to_timestamp(F.col("timestamp")))
    .withColumn("transaction_date", F.to_date(F.col("timestamp")))
    .drop("timestamp", "year", "month", "day")
    .withColumn("transaction_type",
        F.when(F.lower(F.col("transaction_type")).isin(VALID_TYPES),
               F.lower(F.col("transaction_type"))
        ).otherwise(F.lit("unknown")))
    .withColumn("channel",
        F.when(F.lower(F.col("channel")).isin(VALID_CHANNELS),
               F.lower(F.col("channel"))
        ).otherwise(F.lit("unknown")))
    .withColumn("currency",          F.upper(F.col("currency")))
    .withColumn("merchant_category", F.coalesce(F.col("merchant_category"), F.lit("unknown")))
    .withColumn("city",              F.coalesce(F.upper(F.col("city")), F.lit("UNKNOWN")))
    .withColumn("country_code",      F.upper(F.col("country_code")))
    .withColumn("amount",
        F.when(F.col("amount") < 0, F.lit(None)).otherwise(F.col("amount")))
    .dropna(subset=["transaction_id", "account_id", "amount"])
)

(
    df_silver.write
    .format("delta")
    .mode("overwrite")
    .partitionBy("transaction_date")
    .save(SILVER_PATH)
)

silver_count = df_silver.count()
print(f"Silver rows: {{silver_count:,}}")
print(f"Written to Delta Lake: {{SILVER_PATH}}")
display(DeltaTable.forPath(spark, SILVER_PATH).history())

# COMMAND ----------

# MAGIC %md ## 4. Silver -> Gold (Feature Engineering — 30 features)

# COMMAND ----------

df_silver = spark.read.format("delta").load(SILVER_PATH)

w_24h = (Window.partitionBy("account_id")
         .orderBy(F.col("transaction_timestamp").cast("long"))
         .rangeBetween(-86400, 0))
w_7d  = (Window.partitionBy("account_id")
         .orderBy(F.col("transaction_timestamp").cast("long"))
         .rangeBetween(-604800, 0))
w_30d = (Window.partitionBy("account_id")
         .orderBy(F.col("transaction_timestamp").cast("long"))
         .rangeBetween(-2592000, 0))
w_acc = Window.partitionBy("account_id").orderBy("transaction_timestamp")

df_gold = (
    df_silver
    # Velocity
    .withColumn("tx_count_24h",    F.count("transaction_id").over(w_24h))
    .withColumn("tx_amount_24h",   F.sum("amount").over(w_24h))
    .withColumn("tx_count_7d",     F.count("transaction_id").over(w_7d))
    .withColumn("tx_amount_7d",    F.sum("amount").over(w_7d))
    # Statistical
    .withColumn("amount_mean_30d", F.mean("amount").over(w_30d))
    .withColumn("amount_std_30d",  F.stddev("amount").over(w_30d))
    .withColumn("amount_zscore",
        F.when(F.col("amount_std_30d") > 0,
            (F.col("amount") - F.col("amount_mean_30d")) / F.col("amount_std_30d")
        ).otherwise(F.lit(0.0)))
    # Behavioral
    .withColumn("tx_hour",         F.hour("transaction_timestamp"))
    .withColumn("tx_dow",          F.dayofweek("transaction_timestamp"))
    .withColumn("is_night_transaction",
        (F.col("tx_hour") >= 23) | (F.col("tx_hour") <= 5))
    .withColumn("is_weekend",      F.col("tx_dow").isin([1, 7]))
    .withColumn("prev_ts",         F.lag("transaction_timestamp").over(w_acc))
    .withColumn("time_since_last_tx_seconds",
        F.when(F.col("prev_ts").isNotNull(),
            F.col("transaction_timestamp").cast("long") - F.col("prev_ts").cast("long")
        ).otherwise(F.lit(-1)))
    .drop("prev_ts")
    .withColumn("feature_date", F.col("transaction_date"))
)

(
    df_gold.write
    .format("delta")
    .mode("overwrite")
    .partitionBy("feature_date")
    .save(GOLD_PATH)
)

gold_count = df_gold.count()
print(f"Gold rows:     {{gold_count:,}}")
print(f"Gold features: {{len(df_gold.columns)}}")
print(f"Written to:    {{GOLD_PATH}}")
display(DeltaTable.forPath(spark, GOLD_PATH).history())

# COMMAND ----------

# MAGIC %md ## 5. ML Training — 5 Models + MLflow with Signatures

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

mlflow.set_experiment("/Users/{user_email}/fraud_detection_azure")

FEATURES = [
    "amount", "tx_hour", "tx_dow",
    "tx_count_24h", "tx_amount_24h",
    "tx_count_7d", "tx_amount_7d",
    "amount_mean_30d", "amount_std_30d", "amount_zscore",
    "time_since_last_tx_seconds",
]

# Load Gold features
pdf = (
    spark.read.format("delta").load(GOLD_PATH)
    .select(FEATURES + ["is_fraud"])
    .dropna()
    .toPandas()
)
pdf["is_fraud"] = pdf["is_fraud"].astype(int)
print(f"Training samples: {{len(pdf):,}}")
print(f"Fraud rate:       {{pdf['is_fraud'].mean():.2%}}")

X = pdf[FEATURES].fillna(-1).values
y = pdf["is_fraud"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE oversampling
smote        = SMOTE(sampling_strategy=0.1, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {{len(X_train_res):,}} samples | fraud rate: {{y_train_res.mean():.2%}}")

MODELS = {{
    "logistic_regression": {{
        "model": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, C=0.1),
        "description": "Baseline lineaire — simple et interpretable"
    }},
    "random_forest": {{
        "model": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1),
        "description": "Ensemble arbres — robuste et stable"
    }},
    "gradient_boosting": {{
        "model": GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42),
        "description": "Boosting sequentiel — bon sur donnees desequilibrees"
    }},
    "xgboost": {{
        "model": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, scale_pos_weight=50, random_state=42, eval_metric="aucpr", use_label_encoder=False, subsample=0.8, colsample_bytree=0.8),
        "description": "XGBoost — state of the art fraude"
    }},
    "neural_network": {{
        "model": MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu", max_iter=300, early_stopping=True, validation_fraction=0.1, random_state=42, learning_rate_init=0.001),
        "description": "Reseau de neurones 3 couches (128-64-32)"
    }},
}}

results  = {{}}
X_sample = pd.DataFrame(X_train[:5], columns=FEATURES)

for name, config in MODELS.items():
    print(f"\\n== Training: {{name}} ==")
    print(f"   {{config['description']}}")
    model = config["model"]

    with mlflow.start_run(run_name=name):
        model.fit(X_train_res, y_train_res)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred  = (y_proba >= 0.5).astype(int)

        # Signature required for Unity Catalog
        signature = infer_signature(
            X_sample,
            pd.DataFrame(
                model.predict_proba(X_sample.values),
                columns=["prob_normal", "prob_fraud"]
            )
        )

        cm         = confusion_matrix(y_test, y_pred)
        tp, fp, fn = int(cm[1][1]), int(cm[0][1]), int(cm[1][0])

        metrics = {{
            "roc_auc":           round(roc_auc_score(y_test, y_proba), 4),
            "average_precision": round(average_precision_score(y_test, y_proba), 4),
            "f1_score":          round(f1_score(y_test, y_pred, zero_division=0), 4),
            "precision_fraud":   round(tp / (tp + fp) if (tp + fp) > 0 else 0, 4),
            "recall_fraud":      round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
            "true_positives":    tp,
            "false_positives":   fp,
            "false_negatives":   fn,
        }}

        mlflow.log_params({{
            "model_type":  name,
            "description": config["description"],
            "n_features":  len(FEATURES),
            "smote":       True,
            "train_size":  len(X_train_res),
            "test_size":   len(X_test),
        }})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_sample,
            registered_model_name=f"fraud_classifier_{{name}}",
        )

        results[name] = metrics
        print(f"   ROC-AUC: {{metrics['roc_auc']}} | AP: {{metrics['average_precision']}} | F1: {{metrics['f1_score']}}")
        print(f"   Recall:  {{metrics['recall_fraud']}} | Precision: {{metrics['precision_fraud']}}")
        print(f"   TP={{tp}} FP={{fp}} FN={{fn}}")

# Summary
results_df = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
results_df.index.name = "model"
print("\\n== RESULTS SUMMARY ==")
display(results_df)

best = results_df.index[0]
print(f"\\nBest model: {{best}} (ROC-AUC={{results[best]['roc_auc']}})")

# COMMAND ----------

# MAGIC %md ## 6. MLflow Experiment Dashboard

# COMMAND ----------

mlflow_client = MlflowClient()
experiment    = mlflow_client.get_experiment_by_name("/Users/{user_email}/fraud_detection_azure")
runs          = mlflow_client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.roc_auc DESC"]
)

perf_rows = []
for run in runs:
    perf_rows.append({{
        "model":             run.data.params.get("model_type", run.info.run_name),
        "roc_auc":           run.data.metrics.get("roc_auc"),
        "average_precision": run.data.metrics.get("average_precision"),
        "f1_score":          run.data.metrics.get("f1_score"),
        "precision_fraud":   run.data.metrics.get("precision_fraud"),
        "recall_fraud":      run.data.metrics.get("recall_fraud"),
        "true_positives":    int(run.data.metrics.get("true_positives", 0)),
        "false_positives":   int(run.data.metrics.get("false_positives", 0)),
        "false_negatives":   int(run.data.metrics.get("false_negatives", 0)),
        "run_id":            run.info.run_id[:8],
        "status":            run.info.status,
    }})

perf_pdf = pd.DataFrame(perf_rows).sort_values("roc_auc", ascending=False)
print("All MLflow runs:")
display(perf_pdf)

best_run  = runs[0]
best_name = best_run.data.params.get("model_type", best_run.info.run_name)
print(f"\\nBest registered model: fraud_classifier_{{best_name}}")
print(f"View in MLflow UI: Experiments -> fraud_detection_azure")

# COMMAND ----------

# MAGIC %md ## 7. Delta Lake Time Travel

# COMMAND ----------

print("Silver table history (audit trail):")
display(DeltaTable.forPath(spark, SILVER_PATH).history())

print("\\nGold table history:")
display(DeltaTable.forPath(spark, GOLD_PATH).history())

# Time travel — read at version 0
print("\\nRead Silver at version 0 (initial load):")
df_v0 = spark.read.format("delta").option("versionAsOf", 0).load(SILVER_PATH)
print(f"Version 0 rows: {{df_v0.count():,}}")
display(df_v0.groupBy("transaction_type").count().orderBy("count", ascending=False))

# COMMAND ----------

# MAGIC %md ## 8. Export to BigQuery (Serving Layer)

# COMMAND ----------

from google.cloud import bigquery

# Reload GCP credentials (in case of kernel restart)
gcp_creds_workspace = f"/Workspace/Users/{user_email}/fraud-pipeline/gcp_credentials.json"
gcp_creds_tmp       = "/tmp/gcp_credentials.json"
with open(gcp_creds_workspace) as f:
    creds_data = json.load(f)
with open(gcp_creds_tmp, "w") as f:
    json.dump(creds_data, f)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_creds_tmp
print(f"GCP credentials ready")

bq_client   = bigquery.Client(project=GCP_PROJECT)
dataset_ref = bigquery.Dataset(f"{{GCP_PROJECT}}.{{BQ_DATASET}}")
dataset_ref.location = "EU"
try:
    bq_client.create_dataset(dataset_ref)
    print(f"Dataset created: {{BQ_DATASET}}")
except Exception:
    print(f"Dataset exists:  {{BQ_DATASET}}")

df_gold = spark.read.format("delta").load(GOLD_PATH)

# ── Table 1: fraud_alerts ─────────────────────────────────────────────────────
df_alerts  = (
    df_gold.filter("is_fraud = true")
    .select(
        "transaction_id", "account_id", "merchant_id", "amount",
        "transaction_type", "channel", "country_code",
        "amount_zscore", "tx_count_24h", "tx_amount_24h",
        "is_night_transaction", "is_weekend",
        "time_since_last_tx_seconds", "feature_date"
    )
)
alerts_pdf = df_alerts.toPandas()
alerts_pdf["feature_date"]         = alerts_pdf["feature_date"].astype(str)
alerts_pdf["is_night_transaction"]  = alerts_pdf["is_night_transaction"].astype(bool)
alerts_pdf["is_weekend"]            = alerts_pdf["is_weekend"].astype(bool)

bq_client.load_table_from_dataframe(
    alerts_pdf,
    f"{{GCP_PROJECT}}.{{BQ_DATASET}}.fraud_alerts",
    job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
).result()
print(f"fraud_alerts: {{len(alerts_pdf):,}} rows -> BigQuery")

# ── Table 2: model_performance ────────────────────────────────────────────────
bq_client.load_table_from_dataframe(
    perf_pdf,
    f"{{GCP_PROJECT}}.{{BQ_DATASET}}.model_performance",
    job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
).result()
print(f"model_performance: {{len(perf_pdf):,}} rows -> BigQuery")

# ── Table 3: account_risk_profile ─────────────────────────────────────────────
df_risk  = (
    df_gold.groupBy("account_id")
    .agg(
        F.count("transaction_id").alias("total_transactions"),
        F.sum("amount").alias("total_amount"),
        F.avg("amount").alias("avg_amount"),
        F.sum(F.col("is_fraud").cast("int")).alias("fraud_count"),
        F.avg(F.col("is_fraud").cast("int")).alias("fraud_rate"),
        F.max("amount_zscore").alias("max_zscore"),
        F.max("tx_count_24h").alias("max_tx_count_24h"),
    )
)
risk_pdf = df_risk.toPandas()
bq_client.load_table_from_dataframe(
    risk_pdf,
    f"{{GCP_PROJECT}}.{{BQ_DATASET}}.account_risk_profile",
    job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
).result()
print(f"account_risk_profile: {{len(risk_pdf):,}} rows -> BigQuery")

print(f"\\n{'='*50}")
print(f"BigQuery export complete!")
print(f"  fraud_alerts:         {{len(alerts_pdf):,}} rows")
print(f"  model_performance:    {{len(perf_pdf):,}} rows")
print(f"  account_risk_profile: {{len(risk_pdf):,}} rows")
print(f"\\nVerify: https://console.cloud.google.com/bigquery?project={{GCP_PROJECT}}")
'''

notebook_path = f"{workspace_base}/pipeline_notebook"
content_b64   = base64.b64encode(NOTEBOOK_CONTENT.encode()).decode()

client.workspace.import_(
    path=notebook_path,
    content=content_b64,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    overwrite=True,
)

print(f"Notebook updated: {notebook_path}")
print(f"Open: {os.getenv('DATABRICKS_HOST')}#notebook{notebook_path}")