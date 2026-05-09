"""
Databricks Job Entrypoints — one script per pipeline stage.

Each script is the entrypoint for a Databricks Job task.
Args are passed via Databricks job parameters (widgets or task values).
"""

import argparse
import sys
from datetime import datetime

from pyspark.sql import SparkSession

from src.common.config import load_config
from src.common.logger import get_logger

logger = get_logger(__name__)


def get_spark() -> SparkSession:
    """Get or create Spark session configured for GCS + Delta Lake."""
    return (
        SparkSession.builder
        .appName("fraud-detection-lakehouse")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.databricks.delta.retentionDurationCheck.enabled", "false")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


def parse_date(date_str: str | None) -> datetime:
    """Parse date string or default to today."""
    if date_str:
        return datetime.strptime(date_str, "%Y-%m-%d")
    return datetime.utcnow()


# ── Job: Bronze → Silver ──────────────────────────────────────────────────────

def run_bronze_to_silver(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bronze → Silver transformation")
    parser.add_argument("--date", type=str, help="Processing date YYYY-MM-DD")
    parser.add_argument("--env", type=str, default="prod")
    parsed = parser.parse_args(args)

    config = load_config(parsed.env)
    spark = get_spark()

    from src.bronze_to_silver.transformer import BronzeToSilverTransformer
    transformer = BronzeToSilverTransformer(spark, config)

    date = parse_date(parsed.date)
    summary = transformer.run(date)

    logger.info("Job complete", extra=summary)
    if summary["status"] != "success":
        sys.exit(1)


# ── Job: Silver → Gold ────────────────────────────────────────────────────────

def run_silver_to_gold(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Silver → Gold feature engineering")
    parser.add_argument("--date", type=str, help="Processing date YYYY-MM-DD")
    parser.add_argument("--env", type=str, default="prod")
    parsed = parser.parse_args(args)

    config = load_config(parsed.env)
    spark = get_spark()

    from src.silver_to_gold.feature_engineer import SilverToGoldTransformer
    transformer = SilverToGoldTransformer(spark, config)

    date = parse_date(parsed.date)
    summary = transformer.run(date)

    logger.info("Job complete", extra=summary)
    if summary["status"] != "success":
        sys.exit(1)


# ── Job: ML Training ──────────────────────────────────────────────────────────

def run_ml_training(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fraud model training + MLflow")
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--env", type=str, default="prod")
    parsed = parser.parse_args(args)

    config = load_config(parsed.env)
    spark = get_spark()

    from src.ml.trainer import FraudModelTrainer
    trainer = FraudModelTrainer(spark, config)

    summary = trainer.run(lookback_days=parsed.lookback_days)
    logger.info("Job complete", extra=summary)


# ── Job: BigQuery Export ──────────────────────────────────────────────────────

def run_bq_export(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gold → BigQuery export")
    parser.add_argument("--date", type=str, help="Export date YYYY-MM-DD")
    parser.add_argument("--env", type=str, default="prod")
    parsed = parser.parse_args(args)

    config = load_config(parsed.env)
    spark = get_spark()

    from src.serving.bigquery_exporter import BigQueryExporter
    exporter = BigQueryExporter(spark, config)

    date = parse_date(parsed.date)
    summary = exporter.run(date)

    logger.info("Job complete", extra=summary)
    if summary["status"] != "success":
        sys.exit(1)


if __name__ == "__main__":
    # Usage: python -m src.jobs <job_name> [args]
    if len(sys.argv) < 2:
        print("Usage: python -m src.jobs <bronze_to_silver|silver_to_gold|ml_training|bq_export>")
        sys.exit(1)

    job_name = sys.argv[1]
    job_args = sys.argv[2:]

    jobs = {
        "bronze_to_silver": run_bronze_to_silver,
        "silver_to_gold":   run_silver_to_gold,
        "ml_training":      run_ml_training,
        "bq_export":        run_bq_export,
    }

    if job_name not in jobs:
        print(f"Unknown job: {job_name}. Choose from: {list(jobs.keys())}")
        sys.exit(1)

    jobs[job_name](job_args)
