"""
Serving module — Export Gold features + fraud scores to BigQuery.

BigQuery acts as the serving layer for:
- BI dashboards (fraud KPIs, trends)
- Real-time scoring API (reads from BQ)
- Compliance reporting
"""

from __future__ import annotations

from datetime import datetime

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.common.config import PipelineConfig
from src.common.exceptions import ServingError
from src.common.logger import get_logger

logger = get_logger(__name__)

BQ_WRITE_OPTIONS = {
    "temporaryGcsBucket": None,  # set from config
    "createDisposition": "CREATE_IF_NEEDED",
    "writeDisposition": "WRITE_APPEND",
    "partitionField": "feature_date",
    "partitionType": "DAY",
    "clusteredFields": "account_id,transaction_type",
}


class BigQueryExporter:
    """
    Exports Gold Delta Lake data to BigQuery for BI and serving.

    Tables exported:
    - fraud_features: daily feature snapshot per transaction
    - fraud_alerts: high-risk transactions (score >= threshold)
    - account_risk_profile: aggregated risk per account
    """

    def __init__(self, spark: SparkSession, config: PipelineConfig) -> None:
        self.spark = spark
        self.config = config
        self._gold_path = f"gs://{config.gcs.bucket}/{config.gcs.gold_prefix}"
        self._bq_dataset = f"{config.bigquery.project_id}.{config.bigquery.dataset}"

    def _read_gold(self, date: datetime) -> DataFrame:
        """Read Gold Delta for a specific feature_date."""
        try:
            df = (
                self.spark.read
                .format("delta")
                .load(self._gold_path)
                .filter(F.col("feature_date") == date.date().isoformat())
            )
            count = df.count()
            logger.info("Gold read for serving", extra={"rows": count, "date": str(date.date())})
            return df
        except Exception as e:
            raise ServingError(f"Failed to read Gold layer: {e}") from e

    def _write_to_bq(self, df: DataFrame, table: str, mode: str = "append") -> None:
        """Write DataFrame to a BigQuery table."""
        full_table = f"{self._bq_dataset}.{table}"
        logger.info("Writing to BigQuery", extra={"table": full_table, "mode": mode})

        options = {
            **BQ_WRITE_OPTIONS,
            "temporaryGcsBucket": self.config.gcs.bucket,
        }

        try:
            (
                df.write
                .format("bigquery")
                .mode(mode)
                .options(**options)
                .save(full_table)
            )
            logger.info("BigQuery write complete", extra={"table": full_table})
        except Exception as e:
            raise ServingError(f"BigQuery write failed for {full_table}: {e}") from e

    def export_fraud_features(self, df: DataFrame) -> None:
        """Export full feature set to BQ for BI and ad-hoc analysis."""
        self._write_to_bq(df, "fraud_features")

    def export_fraud_alerts(self, df: DataFrame, threshold: float = 0.8) -> None:
        """
        Export high-risk transactions to BQ fraud_alerts table.

        Only transactions with fraud_score >= threshold are exported.
        This table is used for real-time alerting and compliance.
        """
        if "fraud_score" not in df.columns:
            logger.warning("fraud_score column not found — skipping alerts export")
            return

        alerts = df.filter(F.col("fraud_score") >= threshold).select(
            "transaction_id", "account_id", "merchant_id",
            "amount", "currency", "transaction_type", "channel",
            "country_code", "fraud_score", "is_fraud",
            "transaction_timestamp", "feature_date",
        )

        count = alerts.count()
        logger.info("Exporting fraud alerts", extra={"count": count, "threshold": threshold})
        self._write_to_bq(alerts, "fraud_alerts")

    def export_account_risk_profile(self, df: DataFrame) -> None:
        """
        Aggregate risk profile per account and export to BQ.

        Used by CRM and compliance teams.
        """
        profile = (
            df.groupBy("account_id", "feature_date")
            .agg(
                F.count("transaction_id").alias("daily_tx_count"),
                F.sum("amount").alias("daily_tx_amount"),
                F.mean("fraud_score").alias("avg_fraud_score") if "fraud_score" in df.columns
                    else F.lit(None).cast("double").alias("avg_fraud_score"),
                F.max("fraud_score").alias("max_fraud_score") if "fraud_score" in df.columns
                    else F.lit(None).cast("double").alias("max_fraud_score"),
                F.sum(F.col("is_fraud").cast("int")).alias("confirmed_fraud_count"),
                F.first("account_historical_fraud_rate").alias("historical_fraud_rate"),
                F.first("account_age_days").alias("account_age_days"),
            )
        )

        self._write_to_bq(profile, "account_risk_profile")

    def run(self, date: datetime) -> dict:
        """
        Run full export for a given date.

        Args:
            date: Feature date to export

        Returns:
            Export summary dict
        """
        logger.info("Starting BigQuery export", extra={"date": str(date.date())})

        try:
            df = self._read_gold(date)
            row_count = df.cache().count()

            self.export_fraud_features(df)
            self.export_fraud_alerts(df)
            self.export_account_risk_profile(df)

            df.unpersist()

            summary = {
                "status": "success",
                "date": date.isoformat(),
                "rows_exported": row_count,
                "tables": ["fraud_features", "fraud_alerts", "account_risk_profile"],
            }
            logger.info("BigQuery export complete", extra=summary)
            return summary

        except ServingError as e:
            logger.error("BigQuery export failed", extra={"error": str(e)})
            raise
