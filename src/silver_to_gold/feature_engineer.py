"""
Silver → Gold transformation — Feature Engineering for Fraud Detection.

Computes behavioral features per account used by the ML model:
- Velocity features (transaction count/amount over sliding windows)
- Statistical features (mean, std, z-score of amounts)
- Behavioral features (night transactions, cross-border, merchant diversity)
- Historical aggregates (lifetime stats)
"""

from __future__ import annotations

from datetime import datetime

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T
from delta.tables import DeltaTable

from src.common.config import PipelineConfig
from src.common.exceptions import TransformationError, DeltaWriteError
from src.common.logger import get_logger

logger = get_logger(__name__)


class SilverToGoldTransformer:
    """
    PySpark transformer: Silver Delta Lake → Gold feature store.

    Computes fraud detection features at two granularities:
    - Transaction-level features (per row)
    - Account-level aggregates (joined back to each transaction)

    Output written to Gold Delta Lake, partitioned by feature_date.
    """

    def __init__(self, spark: SparkSession, config: PipelineConfig) -> None:
        self.spark = spark
        self.config = config
        self._silver_path = f"gs://{config.gcs.bucket}/{config.gcs.silver_prefix}"
        self._gold_path = f"gs://{config.gcs.bucket}/{config.gcs.gold_prefix}"

    # ── Read Silver ───────────────────────────────────────────────────────────

    def read_silver(self, lookback_days: int = 90) -> DataFrame:
        """
        Read Silver Delta Lake with a lookback window for feature computation.

        Args:
            lookback_days: Days of history needed for rolling features

        Returns:
            Silver DataFrame
        """
        logger.info("Reading Silver layer", extra={"lookback_days": lookback_days})

        try:
            df = (
                self.spark.read
                .format("delta")
                .load(self._silver_path)
                .filter(
                    F.col("transaction_date") >= F.date_sub(F.current_date(), lookback_days)
                )
            )
            count = df.count()
            logger.info("Silver read complete", extra={"rows": count})
            return df
        except Exception as e:
            raise TransformationError(f"Failed to read Silver: {e}") from e

    # ── Velocity features ─────────────────────────────────────────────────────

    def compute_velocity_features(self, df: DataFrame) -> DataFrame:
        """
        Compute rolling velocity features per account.

        Windows: 1h, 6h, 24h, 7d
        Features: tx count, total amount, unique merchants
        """
        logger.info("Computing velocity features")

        # Window specs (rows-based on ordered partitions)
        w_1h = (
            Window.partitionBy("account_id")
            .orderBy(F.col("transaction_timestamp").cast("long"))
            .rangeBetween(-3600, 0)
        )
        w_6h = (
            Window.partitionBy("account_id")
            .orderBy(F.col("transaction_timestamp").cast("long"))
            .rangeBetween(-21600, 0)
        )
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

        df = (
            df
            # 1h velocity
            .withColumn("tx_count_1h",      F.count("transaction_id").over(w_1h))
            .withColumn("tx_amount_1h",     F.sum("amount").over(w_1h))
            # 6h velocity
            .withColumn("tx_count_6h",      F.count("transaction_id").over(w_6h))
            .withColumn("tx_amount_6h",     F.sum("amount").over(w_6h))
            # 24h velocity
            .withColumn("tx_count_24h",     F.count("transaction_id").over(w_24h))
            .withColumn("tx_amount_24h",    F.sum("amount").over(w_24h))
            .withColumn("tx_merchants_24h", F.approx_count_distinct("merchant_id").over(w_24h))
            # 7d velocity
            .withColumn("tx_count_7d",      F.count("transaction_id").over(w_7d))
            .withColumn("tx_amount_7d",     F.sum("amount").over(w_7d))
            .withColumn("tx_merchants_7d",  F.approx_count_distinct("merchant_id").over(w_7d))
        )

        return df

    # ── Statistical features ──────────────────────────────────────────────────

    def compute_statistical_features(self, df: DataFrame) -> DataFrame:
        """
        Compute statistical features per account over 30-day window.

        Features: mean, std, z-score of transaction amounts.
        Z-score = (current_amount - mean) / std — key fraud signal.
        """
        logger.info("Computing statistical features")

        w_30d = (
            Window.partitionBy("account_id")
            .orderBy(F.col("transaction_timestamp").cast("long"))
            .rangeBetween(-2592000, 0)  # 30 days
        )

        df = (
            df
            .withColumn("amount_mean_30d", F.mean("amount").over(w_30d))
            .withColumn("amount_std_30d",  F.stddev("amount").over(w_30d))
            .withColumn(
                "amount_zscore",
                F.when(
                    F.col("amount_std_30d") > 0,
                    (F.col("amount") - F.col("amount_mean_30d")) / F.col("amount_std_30d")
                ).otherwise(F.lit(0.0))
            )
            .withColumn("amount_max_30d", F.max("amount").over(w_30d))
            .withColumn(
                "amount_ratio_to_max",
                F.when(
                    F.col("amount_max_30d") > 0,
                    F.col("amount") / F.col("amount_max_30d")
                ).otherwise(F.lit(0.0))
            )
        )

        return df

    # ── Behavioral features ───────────────────────────────────────────────────

    def compute_behavioral_features(self, df: DataFrame) -> DataFrame:
        """
        Compute behavioral flags and patterns.

        Features:
        - is_night_transaction: 23:00 - 05:00
        - is_weekend: Saturday or Sunday
        - is_cross_border: transaction country != account home country
        - time_since_last_tx: seconds since previous transaction
        - is_new_merchant: first time transacting with this merchant
        - merchant_category_diversity_7d: unique categories in 7 days
        """
        logger.info("Computing behavioral features")

        w_account = Window.partitionBy("account_id").orderBy("transaction_timestamp")
        w_7d = (
            Window.partitionBy("account_id")
            .orderBy(F.col("transaction_timestamp").cast("long"))
            .rangeBetween(-604800, 0)
        )

        df = (
            df
            # Time-based flags
            .withColumn("tx_hour", F.hour("transaction_timestamp"))
            .withColumn("tx_dow",  F.dayofweek("transaction_timestamp"))
            .withColumn(
                "is_night_transaction",
                (F.col("tx_hour") >= 23) | (F.col("tx_hour") <= 5)
            )
            .withColumn(
                "is_weekend",
                F.col("tx_dow").isin([1, 7])  # 1=Sunday, 7=Saturday in Spark
            )
            # Time since last transaction (in seconds)
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
            # First occurrence of merchant for this account
            .withColumn(
                "first_tx_merchant",
                F.first("transaction_timestamp").over(
                    Window.partitionBy("account_id", "merchant_id").orderBy("transaction_timestamp")
                )
            )
            .withColumn(
                "is_new_merchant",
                F.col("transaction_timestamp") == F.col("first_tx_merchant")
            )
            .drop("first_tx_merchant")
            # Merchant category diversity over 7 days
            .withColumn(
                "merchant_category_diversity_7d",
                F.approx_count_distinct("merchant_category").over(w_7d)
            )
        )

        return df

    # ── Account lifetime features ─────────────────────────────────────────────

    def compute_account_features(self, df: DataFrame) -> DataFrame:
        """
        Compute account-level lifetime aggregates joined back to transactions.

        Features: total lifetime transactions, avg amount, fraud rate history.
        """
        logger.info("Computing account lifetime features")

        account_stats = (
            df.groupBy("account_id")
            .agg(
                F.count("transaction_id").alias("account_total_tx"),
                F.mean("amount").alias("account_avg_amount"),
                F.sum("amount").alias("account_total_amount"),
                F.mean(F.col("is_fraud").cast("int")).alias("account_historical_fraud_rate"),
                F.min("transaction_timestamp").alias("account_first_tx_date"),
                F.countDistinct("country_code").alias("account_distinct_countries"),
                F.countDistinct("merchant_category").alias("account_distinct_categories"),
            )
        )

        df = df.join(account_stats, on="account_id", how="left")

        # Account age in days
        df = df.withColumn(
            "account_age_days",
            F.datediff(F.col("transaction_date"), F.to_date(F.col("account_first_tx_date")))
        ).drop("account_first_tx_date")

        return df

    # ── Final feature selection ───────────────────────────────────────────────

    def select_features(self, df: DataFrame) -> DataFrame:
        """
        Select and order final feature columns for the Gold layer.

        Keeps only columns needed by the ML model + audit fields.
        """
        feature_columns = [
            # Keys
            "transaction_id", "account_id", "merchant_id",
            # Target
            "is_fraud",
            # Transaction attributes
            "amount", "currency", "transaction_type", "channel",
            "country_code", "merchant_category",
            "tx_hour", "tx_dow",
            # Velocity features
            "tx_count_1h", "tx_amount_1h",
            "tx_count_6h", "tx_amount_6h",
            "tx_count_24h", "tx_amount_24h", "tx_merchants_24h",
            "tx_count_7d", "tx_amount_7d", "tx_merchants_7d",
            # Statistical features
            "amount_mean_30d", "amount_std_30d", "amount_zscore",
            "amount_max_30d", "amount_ratio_to_max",
            # Behavioral features
            "is_night_transaction", "is_weekend",
            "time_since_last_tx_seconds", "is_new_merchant",
            "merchant_category_diversity_7d",
            # Account features
            "account_total_tx", "account_avg_amount",
            "account_historical_fraud_rate",
            "account_distinct_countries", "account_distinct_categories",
            "account_age_days",
            # Audit
            "transaction_date", "transaction_timestamp",
        ]

        # Add feature_date partition column
        df = df.withColumn("feature_date", F.col("transaction_date"))

        available = [c for c in feature_columns if c in df.columns]
        return df.select(available + ["feature_date"])

    # ── Write Gold ────────────────────────────────────────────────────────────

    def write_gold(self, df: DataFrame) -> None:
        """
        Write feature DataFrame to Gold Delta Lake.

        Uses MERGE on transaction_id for idempotency.
        Applies Z-ordering on account_id and transaction_date for fast reads.
        """
        logger.info("Writing to Gold Delta Lake", extra={"path": self._gold_path})

        try:
            if DeltaTable.isDeltaTable(self.spark, self._gold_path):
                delta_table = DeltaTable.forPath(self.spark, self._gold_path)
                (
                    delta_table.alias("gold")
                    .merge(df.alias("updates"), "gold.transaction_id = updates.transaction_id")
                    .whenMatchedUpdateAll()
                    .whenNotMatchedInsertAll()
                    .execute()
                )
            else:
                (
                    df.write
                    .format("delta")
                    .mode("overwrite")
                    .partitionBy("feature_date")
                    .save(self._gold_path)
                )

            self.spark.sql(f"""
                OPTIMIZE delta.`{self._gold_path}`
                ZORDER BY (account_id, transaction_date)
            """)

            logger.info("Gold write and OPTIMIZE complete")

        except Exception as e:
            raise DeltaWriteError(f"Failed to write Gold layer: {e}") from e

    # ── Orchestration ─────────────────────────────────────────────────────────

    def run(self, date: datetime) -> dict:
        """
        Run full Silver → Gold feature engineering for a given date.

        Args:
            date: Processing date

        Returns:
            Summary dict
        """
        logger.info("Starting Silver → Gold", extra={"date": date.isoformat()})

        try:
            df = self.read_silver(lookback_days=90)
            df = self.compute_velocity_features(df)
            df = self.compute_statistical_features(df)
            df = self.compute_behavioral_features(df)
            df = self.compute_account_features(df)
            df = self.select_features(df)

            # Filter to only write today's features (history used for computation only)
            df_today = df.filter(F.col("feature_date") == date.date().isoformat())
            count = df_today.count()

            self.write_gold(df_today)

            summary = {
                "status": "success",
                "date": date.isoformat(),
                "feature_rows": count,
                "gold_path": self._gold_path,
            }
            logger.info("Silver → Gold complete", extra=summary)
            return summary

        except (TransformationError, DeltaWriteError) as e:
            logger.error("Silver → Gold failed", extra={"error": str(e)})
            raise
