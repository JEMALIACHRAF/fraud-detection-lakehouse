"""
Bronze → Silver transformation — PySpark on Databricks.

Responsibilities:
- Read raw JSON Lines from GCS Bronze
- Cast and validate types
- Deduplicate on transaction_id
- Standardize nulls and enumerations
- Write to Delta Lake Silver (partitioned by transaction_date)
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from delta.tables import DeltaTable

from src.common.config import PipelineConfig
from src.common.exceptions import TransformationError, DeltaWriteError
from src.common.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# ── Schema definition ─────────────────────────────────────────────────────────

BRONZE_SCHEMA = T.StructType([
    T.StructField("transaction_id",    T.StringType(),    False),
    T.StructField("account_id",        T.StringType(),    False),
    T.StructField("amount",            T.DoubleType(),    False),
    T.StructField("currency",          T.StringType(),    True),
    T.StructField("merchant_id",       T.StringType(),    True),
    T.StructField("merchant_category", T.StringType(),    True),
    T.StructField("transaction_type",  T.StringType(),    True),
    T.StructField("channel",           T.StringType(),    True),
    T.StructField("country_code",      T.StringType(),    True),
    T.StructField("city",              T.StringType(),    True),
    T.StructField("timestamp",         T.StringType(),    False),
    T.StructField("is_fraud",          T.BooleanType(),   True),
    T.StructField("created_at",        T.StringType(),    True),
    T.StructField("_ingested_at",      T.StringType(),    True),
])

VALID_TRANSACTION_TYPES = ["purchase", "withdrawal", "transfer", "refund", "payment"]
VALID_CHANNELS = ["online", "atm", "pos", "mobile", "wire"]
VALID_CURRENCIES = ["EUR", "USD", "GBP", "CHF", "JPY", "CAD"]


class BronzeToSilverTransformer:
    """
    PySpark transformer: Bronze JSON Lines → Silver Delta Lake.

    Uses MERGE (upsert) on transaction_id for idempotent writes —
    safe to re-run without creating duplicates.
    """

    def __init__(self, spark: SparkSession, config: PipelineConfig) -> None:
        self.spark = spark
        self.config = config
        self._silver_path = f"gs://{config.gcs.bucket}/{config.gcs.silver_prefix}"

    # ── Read ──────────────────────────────────────────────────────────────────

    def read_bronze(self, date: datetime) -> DataFrame:
        """
        Read Bronze JSON Lines for a specific date partition.

        Args:
            date: Extraction date to read

        Returns:
            Raw DataFrame with enforced schema
        """
        path = (
            f"gs://{self.config.gcs.bucket}/{self.config.gcs.bronze_prefix}"
            f"/year={date.year}/month={date.month:02d}/day={date.day:02d}/"
        )

        logger.info("Reading Bronze layer", extra={"path": path})

        try:
            df = (
                self.spark.read
                .schema(BRONZE_SCHEMA)
                .option("mode", "PERMISSIVE")           # log corrupt rows, don't fail
                .option("columnNameOfCorruptRecord", "_corrupt_record")
                .json(path)
            )

            raw_count = df.count()
            logger.info("Bronze read complete", extra={"raw_rows": raw_count})
            return df

        except Exception as e:
            raise TransformationError(f"Failed to read Bronze layer: {e}") from e

    # ── Transform ─────────────────────────────────────────────────────────────

    def clean(self, df: DataFrame) -> DataFrame:
        """
        Apply cleaning transformations.

        Steps:
        1. Drop corrupt records
        2. Cast timestamp string to proper TimestampType
        3. Standardize enumerations (lowercase → canonical)
        4. Null handling
        5. Derive transaction_date partition column
        6. Deduplicate on transaction_id (keep latest created_at)
        """
        logger.info("Starting cleaning transformations")

        # 1. Drop corrupt records
        if "_corrupt_record" in df.columns:
            corrupt_count = df.filter(F.col("_corrupt_record").isNotNull()).count()
            if corrupt_count > 0:
                logger.warning("Corrupt records found", extra={"count": corrupt_count})
            df = df.filter(F.col("_corrupt_record").isNull()).drop("_corrupt_record")

        # 2. Cast timestamps
        df = df.withColumn(
            "transaction_timestamp",
            F.to_timestamp(F.col("timestamp"), "yyyy-MM-dd'T'HH:mm:ss")
        ).withColumn(
            "created_at",
            F.to_timestamp(F.col("created_at"), "yyyy-MM-dd'T'HH:mm:ss")
        ).withColumn(
            "_ingested_at",
            F.to_timestamp(F.col("_ingested_at"))
        ).drop("timestamp")

        # 3. Standardize enumerations
        df = df.withColumn(
            "transaction_type",
            F.when(
                F.lower(F.col("transaction_type")).isin(VALID_TRANSACTION_TYPES),
                F.lower(F.col("transaction_type"))
            ).otherwise(F.lit("unknown"))
        ).withColumn(
            "channel",
            F.when(
                F.lower(F.col("channel")).isin(VALID_CHANNELS),
                F.lower(F.col("channel"))
            ).otherwise(F.lit("unknown"))
        ).withColumn(
            "currency",
            F.when(
                F.upper(F.col("currency")).isin(VALID_CURRENCIES),
                F.upper(F.col("currency"))
            ).otherwise(F.lit("OTHER"))
        )

        # 4. Null handling
        df = df.withColumn(
            "amount",
            F.when(F.col("amount") < 0, F.lit(None)).otherwise(F.col("amount"))
        ).withColumn(
            "merchant_category",
            F.coalesce(F.col("merchant_category"), F.lit("unknown"))
        ).withColumn(
            "city",
            F.coalesce(F.upper(F.col("city")), F.lit("UNKNOWN"))
        ).withColumn(
            "country_code",
            F.upper(F.col("country_code"))
        )

        # 5. Partition column
        df = df.withColumn("transaction_date", F.to_date(F.col("transaction_timestamp")))

        # 6. Deduplication — keep latest record per transaction_id
        window = (
            __import__("pyspark.sql.window", fromlist=["Window"])
            .Window.partitionBy("transaction_id")
            .orderBy(F.col("created_at").desc())
        )
        df = (
            df.withColumn("_row_num", F.row_number().over(window))
            .filter(F.col("_row_num") == 1)
            .drop("_row_num")
        )

        # Drop nulls on non-nullable fields
        df = df.dropna(subset=["transaction_id", "account_id", "amount", "transaction_timestamp"])

        clean_count = df.count()
        logger.info("Cleaning complete", extra={"clean_rows": clean_count})
        return df

    def add_quality_flags(self, df: DataFrame) -> DataFrame:
        """
        Add data quality flags for monitoring.

        These columns are not used downstream but help track
        data quality degradation over time.
        """
        return df.withColumn(
            "_dq_amount_suspicious",
            F.col("amount") > 50_000
        ).withColumn(
            "_dq_unknown_channel",
            F.col("channel") == "unknown"
        ).withColumn(
            "_dq_unknown_type",
            F.col("transaction_type") == "unknown"
        )

    # ── Write (Delta MERGE) ───────────────────────────────────────────────────

    def write_silver(self, df: DataFrame) -> None:
        """
        Write cleaned DataFrame to Silver Delta Lake using MERGE (upsert).

        Partitioned by transaction_date.
        If Silver table doesn't exist, creates it.

        Args:
            df: Cleaned DataFrame to write
        """
        logger.info("Writing to Silver Delta Lake", extra={"path": self._silver_path})

        try:
            if DeltaTable.isDeltaTable(self.spark, self._silver_path):
                logger.info("Silver table exists — performing MERGE upsert")
                delta_table = DeltaTable.forPath(self.spark, self._silver_path)

                (
                    delta_table.alias("silver")
                    .merge(
                        df.alias("updates"),
                        "silver.transaction_id = updates.transaction_id"
                    )
                    .whenMatchedUpdateAll()
                    .whenNotMatchedInsertAll()
                    .execute()
                )
            else:
                logger.info("Silver table does not exist — creating with initial write")
                (
                    df.write
                    .format("delta")
                    .mode("overwrite")
                    .partitionBy("transaction_date")
                    .option("overwriteSchema", "true")
                    .save(self._silver_path)
                )

            # Optimize Delta table after write (Z-ordering on frequent filter columns)
            self.spark.sql(f"""
                OPTIMIZE delta.`{self._silver_path}`
                ZORDER BY (account_id, transaction_date)
            """)

            logger.info("Silver write and OPTIMIZE complete")

        except Exception as e:
            raise DeltaWriteError(f"Failed to write Silver layer: {e}") from e

    # ── Orchestration ─────────────────────────────────────────────────────────

    def run(self, date: datetime) -> dict:
        """
        Run the full Bronze → Silver transformation for a given date.

        Args:
            date: Date to process

        Returns:
            Transformation summary dict
        """
        logger.info("Starting Bronze → Silver", extra={"date": date.isoformat()})

        try:
            raw_df = self.read_bronze(date)
            raw_count = raw_df.count()

            clean_df = self.clean(raw_df)
            flagged_df = self.add_quality_flags(clean_df)
            clean_count = flagged_df.count()

            self.write_silver(flagged_df)

            summary = {
                "status": "success",
                "date": date.isoformat(),
                "raw_rows": raw_count,
                "clean_rows": clean_count,
                "dropped_rows": raw_count - clean_count,
                "silver_path": self._silver_path,
            }

            logger.info("Bronze → Silver complete", extra=summary)
            return summary

        except (TransformationError, DeltaWriteError) as e:
            logger.error("Bronze → Silver failed", extra={"error": str(e)})
            raise
