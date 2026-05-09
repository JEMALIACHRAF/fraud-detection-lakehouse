"""
scripts/run_local_pipeline.py

Run the full fraud detection pipeline locally without cloud dependencies.

Replaces:
- GCS         → local data/ directory
- Delta Lake  → Parquet files
- BigQuery    → local Parquet output
- MLflow      → local MLflow server (localhost:5000)

Usage:
    python scripts/run_local_pipeline.py --data-dir data --date 2024-06-15
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Force correct Python + Hadoop for PySpark on Windows ─────────────────────
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] = "C:\\hadoop\\bin;" + os.environ.get("PATH", "")

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report


def get_spark(app_name: str = "fraud-pipeline-local") -> SparkSession:
    """Create local Spark session — no Delta, no GCS."""
    return (
        SparkSession.builder
        .master("local[*]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


# ── Step 1: Read Bronze ───────────────────────────────────────────────────────

def read_bronze(spark: SparkSession, data_dir: str) -> DataFrame:
    """Read all JSON Lines from Bronze layer."""
    bronze_path = str(Path(data_dir) / "bronze" / "transactions")
    print(f"\n[1/5] Reading Bronze layer: {bronze_path}")

    df = spark.read.json(bronze_path)
    count = df.count()
    fraud_count = df.filter(F.col("is_fraud") == True).count()
    print(f"  Rows:       {count:,}")
    print(f"  Fraud:      {fraud_count:,} ({fraud_count/count:.2%})")
    print(f"  Columns:    {len(df.columns)}")
    return df


# ── Step 2: Bronze → Silver ───────────────────────────────────────────────────

VALID_TRANSACTION_TYPES = ["purchase", "withdrawal", "transfer", "refund", "payment"]
VALID_CHANNELS = ["online", "atm", "pos", "mobile", "wire"]
VALID_CURRENCIES = ["EUR", "USD", "GBP", "CHF", "JPY", "CAD"]

def bronze_to_silver(df: DataFrame) -> DataFrame:
    """Clean and normalize Bronze data → Silver."""
    print(f"\n[2/5] Bronze → Silver transformation")

    # Cast timestamp
    df = df.withColumn(
        "transaction_timestamp",
        F.to_timestamp(F.col("timestamp"))
    ).withColumn(
        "transaction_date",
        F.to_date(F.col("timestamp"))
    ).drop("timestamp")

    # Normalize enums
    df = df.withColumn(
        "transaction_type",
        F.when(F.lower(F.col("transaction_type")).isin(VALID_TRANSACTION_TYPES),
               F.lower(F.col("transaction_type"))
        ).otherwise(F.lit("unknown"))
    ).withColumn(
        "channel",
        F.when(F.lower(F.col("channel")).isin(VALID_CHANNELS),
               F.lower(F.col("channel"))
        ).otherwise(F.lit("unknown"))
    ).withColumn(
        "currency",
        F.when(F.upper(F.col("currency")).isin(VALID_CURRENCIES),
               F.upper(F.col("currency"))
        ).otherwise(F.lit("OTHER"))
    )

    # Null handling
    df = df.withColumn(
        "merchant_category",
        F.coalesce(F.col("merchant_category"), F.lit("unknown"))
    ).withColumn(
        "city", F.coalesce(F.upper(F.col("city")), F.lit("UNKNOWN"))
    ).withColumn(
        "country_code", F.upper(F.col("country_code"))
    ).withColumn(
        "amount",
        F.when(F.col("amount") < 0, F.lit(None)).otherwise(F.col("amount"))
    )

    # Deduplication
    from pyspark.sql.window import Window
    w = Window.partitionBy("transaction_id").orderBy(F.col("created_at").desc())
    df = (
        df.withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    # Drop nulls on key fields
    df = df.dropna(subset=["transaction_id", "account_id", "amount", "transaction_timestamp"])

    count = df.count()
    print(f"  Silver rows: {count:,}")
    return df


# ── Step 3: Silver → Gold (features) ─────────────────────────────────────────

def silver_to_gold(df: DataFrame) -> DataFrame:
    """Compute fraud detection features → Gold."""
    print(f"\n[3/5] Silver → Gold feature engineering")

    from pyspark.sql.window import Window

    # Velocity windows
    w_24h = (
        Window.partitionBy("account_id")
        .orderBy(F.col("transaction_timestamp").cast("long"))
        .rangeBetween(-86400, 0)
    )
    w_7d = (
        Window.partitionBy("account_id")
        .orderBy(F.col("transaction_timestamp").cast("long"))
        .rangeBetween(-604800, 0)
    )
    w_30d = (
        Window.partitionBy("account_id")
        .orderBy(F.col("transaction_timestamp").cast("long"))
        .rangeBetween(-2592000, 0)
    )
    w_account = Window.partitionBy("account_id").orderBy("transaction_timestamp")

    df = (
        df
        # Velocity
        .withColumn("tx_count_24h",     F.count("transaction_id").over(w_24h))
        .withColumn("tx_amount_24h",    F.sum("amount").over(w_24h))
        .withColumn("tx_count_7d",      F.count("transaction_id").over(w_7d))
        .withColumn("tx_amount_7d",     F.sum("amount").over(w_7d))
        # Statistical
        .withColumn("amount_mean_30d",  F.mean("amount").over(w_30d))
        .withColumn("amount_std_30d",   F.stddev("amount").over(w_30d))
        .withColumn(
            "amount_zscore",
            F.when(
                F.col("amount_std_30d") > 0,
                (F.col("amount") - F.col("amount_mean_30d")) / F.col("amount_std_30d")
            ).otherwise(F.lit(0.0))
        )
        # Behavioral
        .withColumn("tx_hour",          F.hour("transaction_timestamp"))
        .withColumn("tx_dow",           F.dayofweek("transaction_timestamp"))
        .withColumn(
            "is_night_transaction",
            (F.col("tx_hour") >= 23) | (F.col("tx_hour") <= 5)
        )
        .withColumn(
            "is_weekend",
            F.col("tx_dow").isin([1, 7])
        )
        .withColumn(
            "prev_tx_timestamp",
            F.lag("transaction_timestamp").over(w_account)
        )
        .withColumn(
            "time_since_last_tx_seconds",
            F.when(
                F.col("prev_tx_timestamp").isNotNull(),
                F.col("transaction_timestamp").cast("long") - F.col("prev_tx_timestamp").cast("long")
            ).otherwise(F.lit(-1))
        )
        .drop("prev_tx_timestamp")
        # Feature date
        .withColumn("feature_date", F.col("transaction_date"))
    )

    count = df.count()
    print(f"  Gold rows:   {count:,}")
    print(f"  Features:    {len(df.columns)}")
    return df


# ── Step 4: ML Training with MLflow ──────────────────────────────────────────

FEATURES = [
    "amount", "tx_hour", "tx_dow",
    "tx_count_24h", "tx_amount_24h",
    "tx_count_7d", "tx_amount_7d",
    "amount_mean_30d", "amount_std_30d", "amount_zscore",
    "time_since_last_tx_seconds",
]

def train_model(df: DataFrame, mlflow_dir: str) -> None:
    """Train fraud classifier and log to local MLflow."""
    print(f"\n[4/5] ML Training with MLflow")

    mlflow_path = Path(mlflow_dir).absolute().as_posix()
    mlflow.set_tracking_uri(f"file:///{mlflow_path}")
    mlflow.set_experiment("fraud_detection_local")

    # Convert to Pandas
    pdf = df.select(FEATURES + ["is_fraud"]).dropna().toPandas()
    pdf["is_fraud"] = pdf["is_fraud"].astype(int)

    print(f"  Training samples: {len(pdf):,}")
    print(f"  Fraud rate:       {pdf['is_fraud'].mean():.2%}")

    X = pdf[FEATURES].fillna(-1).values
    y = pdf["is_fraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="random_forest_local"):

        # Train
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        # Evaluate
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred  = (y_proba >= 0.5).astype(int)

        roc_auc  = roc_auc_score(y_test, y_proba)
        avg_prec = average_precision_score(y_test, y_proba)
        f1       = f1_score(y_test, y_pred, zero_division=0)

        # Log to MLflow
        mlflow.log_params({
            "n_estimators": 100,
            "max_depth": 8,
            "class_weight": "balanced",
            "features": len(FEATURES),
            "train_samples": len(X_train),
        })
        mlflow.log_metrics({
            "roc_auc":           round(roc_auc, 4),
            "average_precision": round(avg_prec, 4),
            "f1_score":          round(f1, 4),
            "training_time_s":   round(train_time, 2),
        })

        # Feature importance
        fi = pd.DataFrame({
            "feature":    FEATURES,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        fi_path = Path(mlflow_dir) / "feature_importance.csv"
        fi.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path))

        mlflow.sklearn.log_model(model, "model")

        print(f"\n  ✓ MLflow run complete")
        print(f"    ROC-AUC:           {roc_auc:.4f}")
        print(f"    Average Precision: {avg_prec:.4f}")
        print(f"    F1 Score:          {f1:.4f}")
        print(f"    Training time:     {train_time:.1f}s")
        print(f"\n  Top 5 features:")
        for _, row in fi.head(5).iterrows():
            print(f"    {row['feature']:<35} {row['importance']:.4f}")


# ── Step 5: Write Gold to Parquet (serving) ───────────────────────────────────

def write_serving_layer(df: DataFrame, data_dir: str) -> None:
    """Write Gold features to Parquet serving layer."""
    print(f"\n[5/5] Writing serving layer (Parquet)")
    output_path = str(Path(data_dir) / "gold" / "features")

    (
        df.write
        .mode("overwrite")
        .partitionBy("feature_date")
        .parquet(output_path)
    )
    print(f"  Written to: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run fraud detection pipeline locally")
    parser.add_argument("--data-dir",   default="data",          help="Data directory")
    parser.add_argument("--mlflow-dir", default="mlflow_runs",   help="MLflow tracking directory")
    parser.add_argument("--skip-ml",    action="store_true",     help="Skip ML training")
    args = parser.parse_args()

    # Set PYSPARK_PYTHON
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    print("=" * 60)
    print("  FRAUD DETECTION PIPELINE — LOCAL RUN")
    print("=" * 60)

    start_total = time.time()
    spark = get_spark()

    try:
        df_bronze = read_bronze(spark, args.data_dir)
        df_silver = bronze_to_silver(df_bronze)
        df_gold   = silver_to_gold(df_silver)

        if not args.skip_ml:
            Path(args.mlflow_dir).mkdir(exist_ok=True)
            train_model(df_gold, args.mlflow_dir)

        write_serving_layer(df_gold, args.data_dir)

    finally:
        spark.stop()

    elapsed = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  MLflow UI: mlflow ui --backend-store-uri file://{Path(args.mlflow_dir).absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
