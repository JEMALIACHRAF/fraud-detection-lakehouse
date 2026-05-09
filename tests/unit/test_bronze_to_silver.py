"""
Unit tests — BronzeToSilverTransformer

Tests use a local SparkSession (no cluster needed).
Run: pytest tests/unit/test_bronze_to_silver.py -v
"""

import pytest
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from unittest.mock import MagicMock, patch

from src.bronze_to_silver.transformer import BronzeToSilverTransformer, BRONZE_SCHEMA
from src.common.config import (
    PipelineConfig, GCSConfig, DatabaseConfig,
    DatabricksConfig, BigQueryConfig, MLConfig
)


@pytest.fixture(scope="session")
def spark():
    """Local Spark session for unit tests."""
    return (
        SparkSession.builder
        .master("local[2]")
        .appName("test-fraud-pipeline")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


@pytest.fixture
def config():
    """Minimal config for unit tests."""
    return PipelineConfig(
        environment="test",
        gcs=GCSConfig(bucket="test-bucket"),
        database=DatabaseConfig(
            host="localhost", port=5432, database="test",
            username="user", password="pass"
        ),
        databricks=DatabricksConfig(
            host="https://test.azuredatabricks.net",
            token="test-token",
            cluster_id="test-cluster",
            mlflow_tracking_uri="databricks",
        ),
        bigquery=BigQueryConfig(project_id="test-project", dataset="test_dataset"),
        ml=MLConfig(),
    )


@pytest.fixture
def transformer(spark, config):
    return BronzeToSilverTransformer(spark, config)


@pytest.fixture
def sample_raw_data(spark):
    """Sample raw transaction data matching Bronze schema."""
    data = [
        {
            "transaction_id": "TX001",
            "account_id": "ACC001",
            "amount": 150.0,
            "currency": "EUR",
            "merchant_id": "M001",
            "merchant_category": "grocery",
            "transaction_type": "purchase",
            "channel": "online",
            "country_code": "fr",
            "city": "paris",
            "timestamp": "2024-01-15T10:30:00",
            "is_fraud": False,
            "created_at": "2024-01-15T10:30:01",
            "_ingested_at": "2024-01-15T11:00:00",
        },
        {
            "transaction_id": "TX002",
            "account_id": "ACC002",
            "amount": 5000.0,
            "currency": "usd",  # lowercase — should be normalized
            "merchant_id": "M002",
            "merchant_category": None,  # null — should default to "unknown"
            "transaction_type": "PURCHASE",  # uppercase — should be normalized
            "channel": "ATM",  # uppercase — should be normalized
            "country_code": "de",
            "city": None,
            "timestamp": "2024-01-15T23:45:00",
            "is_fraud": True,
            "created_at": "2024-01-15T23:45:01",
            "_ingested_at": "2024-01-15T11:00:00",
        },
        {
            "transaction_id": "TX001",  # duplicate — should be deduped
            "account_id": "ACC001",
            "amount": 150.0,
            "currency": "EUR",
            "merchant_id": "M001",
            "merchant_category": "grocery",
            "transaction_type": "purchase",
            "channel": "online",
            "country_code": "fr",
            "city": "paris",
            "timestamp": "2024-01-15T10:30:00",
            "is_fraud": False,
            "created_at": "2024-01-15T10:35:00",  # later created_at
            "_ingested_at": "2024-01-15T11:00:00",
        },
    ]
    return spark.createDataFrame(data)


class TestClean:

    def test_deduplication(self, transformer, sample_raw_data):
        """Duplicate transaction_ids should be removed — keep latest created_at."""
        result = transformer.clean(sample_raw_data)
        tx001_count = result.filter(result.transaction_id == "TX001").count()
        assert tx001_count == 1, "Duplicate TX001 should be deduplicated"

    def test_output_row_count(self, transformer, sample_raw_data):
        """3 rows with 1 duplicate → 2 unique transactions."""
        result = transformer.clean(sample_raw_data)
        assert result.count() == 2

    def test_currency_normalized_to_uppercase(self, transformer, sample_raw_data):
        """Currency 'usd' → 'USD'."""
        result = transformer.clean(sample_raw_data)
        tx002 = result.filter(result.transaction_id == "TX002").first()
        assert tx002["currency"] == "USD"

    def test_transaction_type_normalized_to_lowercase(self, transformer, sample_raw_data):
        """Transaction type 'PURCHASE' → 'purchase'."""
        result = transformer.clean(sample_raw_data)
        tx002 = result.filter(result.transaction_id == "TX002").first()
        assert tx002["transaction_type"] == "purchase"

    def test_channel_normalized_to_lowercase(self, transformer, sample_raw_data):
        """Channel 'ATM' → 'atm'."""
        result = transformer.clean(sample_raw_data)
        tx002 = result.filter(result.transaction_id == "TX002").first()
        assert tx002["channel"] == "atm"

    def test_null_merchant_category_filled(self, transformer, sample_raw_data):
        """Null merchant_category → 'unknown'."""
        result = transformer.clean(sample_raw_data)
        tx002 = result.filter(result.transaction_id == "TX002").first()
        assert tx002["merchant_category"] == "unknown"

    def test_null_city_filled(self, transformer, sample_raw_data):
        """Null city → 'UNKNOWN'."""
        result = transformer.clean(sample_raw_data)
        tx002 = result.filter(result.transaction_id == "TX002").first()
        assert tx002["city"] == "UNKNOWN"

    def test_country_code_uppercase(self, transformer, sample_raw_data):
        """country_code 'fr' → 'FR'."""
        result = transformer.clean(sample_raw_data)
        tx001 = result.filter(result.transaction_id == "TX001").first()
        assert tx001["country_code"] == "FR"

    def test_transaction_date_derived(self, transformer, sample_raw_data):
        """transaction_date should be derived from transaction_timestamp."""
        result = transformer.clean(sample_raw_data)
        assert "transaction_date" in result.columns
        assert result.filter(result.transaction_date.isNull()).count() == 0

    def test_negative_amount_nulled(self, spark, transformer):
        """Negative amounts should be set to null and row dropped."""
        data = [{
            "transaction_id": "TX_NEG",
            "account_id": "ACC001",
            "amount": -50.0,  # negative
            "currency": "EUR",
            "merchant_id": "M001",
            "merchant_category": "grocery",
            "transaction_type": "purchase",
            "channel": "online",
            "country_code": "FR",
            "city": "Paris",
            "timestamp": "2024-01-15T10:00:00",
            "is_fraud": False,
            "created_at": "2024-01-15T10:00:01",
            "_ingested_at": "2024-01-15T11:00:00",
        }]
        df = spark.createDataFrame(data)
        result = transformer.clean(df)
        # Negative amount → null → row dropped by dropna
        assert result.count() == 0


class TestQualityFlags:

    def test_suspicious_amount_flag(self, transformer, sample_raw_data):
        """Amounts > 50000 should be flagged."""
        result = transformer.clean(sample_raw_data)
        flagged = transformer.add_quality_flags(result)
        assert "_dq_amount_suspicious" in flagged.columns

    def test_unknown_channel_flag(self, spark, transformer):
        """Unknown channel should trigger _dq_unknown_channel flag."""
        data = [{
            "transaction_id": "TX_UNK",
            "account_id": "ACC001",
            "amount": 100.0,
            "currency": "EUR",
            "merchant_id": "M001",
            "merchant_category": "grocery",
            "transaction_type": "purchase",
            "channel": "fax",  # invalid → unknown
            "country_code": "FR",
            "city": "Paris",
            "timestamp": "2024-01-15T10:00:00",
            "is_fraud": False,
            "created_at": "2024-01-15T10:00:01",
            "_ingested_at": "2024-01-15T11:00:00",
        }]
        df = spark.createDataFrame(data)
        cleaned = transformer.clean(df)
        flagged = transformer.add_quality_flags(cleaned)
        row = flagged.first()
        assert row["_dq_unknown_channel"] is True
