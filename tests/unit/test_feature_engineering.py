"""
Unit tests — SilverToGoldTransformer (feature engineering)

Run: pytest tests/unit/test_feature_engineering.py -v
"""

import pytest
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


@pytest.fixture(scope="session")
def spark():
    return (
        SparkSession.builder
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


@pytest.fixture
def silver_data(spark):
    """Sample Silver data for feature computation tests."""
    base_ts = datetime(2024, 1, 15, 10, 0, 0)
    data = [
        # Account ACC001 — 3 transactions close together
        {
            "transaction_id": f"TX{i:03d}",
            "account_id": "ACC001",
            "merchant_id": f"M{i % 3:03d}",
            "amount": float(100 * (i + 1)),
            "currency": "EUR",
            "transaction_type": "purchase",
            "channel": "online",
            "country_code": "FR",
            "merchant_category": ["grocery", "electronics", "travel"][i % 3],
            "is_fraud": i == 2,  # 3rd transaction is fraud
            "transaction_timestamp": base_ts + timedelta(minutes=i * 30),
            "transaction_date": (base_ts + timedelta(minutes=i * 30)).date(),
            "_ingested_at": base_ts,
            "_dq_amount_suspicious": False,
            "_dq_unknown_channel": False,
            "_dq_unknown_type": False,
        }
        for i in range(3)
    ] + [
        # Account ACC002 — 1 transaction
        {
            "transaction_id": "TX100",
            "account_id": "ACC002",
            "merchant_id": "M010",
            "amount": 9999.0,
            "currency": "USD",
            "transaction_type": "transfer",
            "channel": "online",
            "country_code": "DE",
            "merchant_category": "unknown",
            "is_fraud": True,
            "transaction_timestamp": base_ts,
            "transaction_date": base_ts.date(),
            "_ingested_at": base_ts,
            "_dq_amount_suspicious": True,
            "_dq_unknown_channel": False,
            "_dq_unknown_type": False,
        }
    ]
    return spark.createDataFrame(data)


class TestBehavioralFeatures:

    def test_night_transaction_flag_at_23h(self, spark):
        """Transactions at 23:00+ should be flagged as night transactions."""
        from src.silver_to_gold.feature_engineer import SilverToGoldTransformer
        from unittest.mock import MagicMock
        config = MagicMock()
        config.gcs.bucket = "test"
        config.gcs.silver_prefix = "silver"
        config.gcs.gold_prefix = "gold"

        t = SilverToGoldTransformer(spark, config)

        data = [{
            "transaction_id": "TX_NIGHT",
            "account_id": "ACC001",
            "merchant_id": "M001",
            "amount": 100.0,
            "currency": "EUR",
            "transaction_type": "purchase",
            "channel": "online",
            "country_code": "FR",
            "merchant_category": "grocery",
            "is_fraud": False,
            "transaction_timestamp": datetime(2024, 1, 15, 23, 30, 0),
            "transaction_date": datetime(2024, 1, 15).date(),
            "_ingested_at": datetime(2024, 1, 15, 0, 0, 0),
            "_dq_amount_suspicious": False,
            "_dq_unknown_channel": False,
            "_dq_unknown_type": False,
        }]
        df = spark.createDataFrame(data)
        result = t.compute_behavioral_features(df)
        row = result.first()
        assert row["is_night_transaction"] is True

    def test_day_transaction_not_flagged(self, spark):
        """Transactions at 10:00 should NOT be flagged as night."""
        from src.silver_to_gold.feature_engineer import SilverToGoldTransformer
        from unittest.mock import MagicMock
        config = MagicMock()
        config.gcs.bucket = "test"
        config.gcs.silver_prefix = "silver"
        config.gcs.gold_prefix = "gold"

        t = SilverToGoldTransformer(spark, config)

        data = [{
            "transaction_id": "TX_DAY",
            "account_id": "ACC001",
            "merchant_id": "M001",
            "amount": 100.0,
            "currency": "EUR",
            "transaction_type": "purchase",
            "channel": "online",
            "country_code": "FR",
            "merchant_category": "grocery",
            "is_fraud": False,
            "transaction_timestamp": datetime(2024, 1, 15, 10, 0, 0),
            "transaction_date": datetime(2024, 1, 15).date(),
            "_ingested_at": datetime(2024, 1, 15, 0, 0, 0),
            "_dq_amount_suspicious": False,
            "_dq_unknown_channel": False,
            "_dq_unknown_type": False,
        }]
        df = spark.createDataFrame(data)
        result = t.compute_behavioral_features(df)
        row = result.first()
        assert row["is_night_transaction"] is False

    def test_new_merchant_flag_first_transaction(self, spark, silver_data):
        """First transaction with a merchant should be flagged as new merchant."""
        from src.silver_to_gold.feature_engineer import SilverToGoldTransformer
        from unittest.mock import MagicMock
        config = MagicMock()
        config.gcs.bucket = "test"
        config.gcs.silver_prefix = "silver"
        config.gcs.gold_prefix = "gold"

        t = SilverToGoldTransformer(spark, config)
        result = t.compute_behavioral_features(silver_data)

        # TX000 is first for ACC001 with M000 → should be new merchant
        tx000 = result.filter(F.col("transaction_id") == "TX000").first()
        assert tx000["is_new_merchant"] is True


class TestAccountFeatures:

    def test_account_total_tx_count(self, spark, silver_data):
        """ACC001 has 3 transactions — account_total_tx should be 3."""
        from src.silver_to_gold.feature_engineer import SilverToGoldTransformer
        from unittest.mock import MagicMock
        config = MagicMock()
        config.gcs.bucket = "test"
        config.gcs.silver_prefix = "silver"
        config.gcs.gold_prefix = "gold"

        t = SilverToGoldTransformer(spark, config)
        result = t.compute_account_features(silver_data)

        acc001_rows = result.filter(F.col("account_id") == "ACC001").collect()
        for row in acc001_rows:
            assert row["account_total_tx"] == 3

    def test_account_historical_fraud_rate(self, spark, silver_data):
        """ACC001: 1 fraud / 3 tx = 0.333 fraud rate."""
        from src.silver_to_gold.feature_engineer import SilverToGoldTransformer
        from unittest.mock import MagicMock
        config = MagicMock()
        config.gcs.bucket = "test"
        config.gcs.silver_prefix = "silver"
        config.gcs.gold_prefix = "gold"

        t = SilverToGoldTransformer(spark, config)
        result = t.compute_account_features(silver_data)

        acc001 = result.filter(F.col("account_id") == "ACC001").first()
        assert abs(acc001["account_historical_fraud_rate"] - (1/3)) < 0.01
