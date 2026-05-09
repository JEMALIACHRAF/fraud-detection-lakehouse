"""
Configuration loader — reads YAML config files and merges with env vars.
Supports dev / staging / prod environments.
"""

import os
import yaml
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

from src.common.exceptions import ConfigurationError
from src.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GCSConfig:
    bucket: str
    bronze_prefix: str = "bronze/transactions"
    silver_prefix: str = "silver/transactions"
    gold_prefix: str = "gold/features"


@dataclass
class DatabaseConfig:
    host: str
    port: int
    database: str
    username: str
    password: str = field(repr=False)
    schema: str = "public"


@dataclass
class DatabricksConfig:
    host: str
    token: str = field(repr=False)
    cluster_id: str
    mlflow_tracking_uri: str


@dataclass
class BigQueryConfig:
    project_id: str
    dataset: str
    location: str = "EU"


@dataclass
class MLConfig:
    experiment_name: str = "fraud_detection"
    model_name: str = "fraud_classifier"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    max_depth: int = 8
    fraud_threshold: float = 0.5


@dataclass
class PipelineConfig:
    environment: str
    gcs: GCSConfig
    database: DatabaseConfig
    databricks: DatabricksConfig
    bigquery: BigQueryConfig
    ml: MLConfig
    batch_size: int = 100_000
    max_retries: int = 3
    retry_delay_seconds: int = 30


def load_config(environment: str | None = None) -> PipelineConfig:
    """
    Load configuration for the given environment.

    Priority: env vars > config file > defaults.

    Args:
        environment: 'dev' | 'staging' | 'prod'. Defaults to ENV_NAME env var.

    Returns:
        PipelineConfig instance

    Raises:
        ConfigurationError: if required fields are missing
    """
    env = environment or os.getenv("ENV_NAME", "dev")
    config_path = Path(__file__).parents[2] / "config" / f"{env}.yml"

    if not config_path.exists():
        raise ConfigurationError(f"Config file not found: {config_path}")

    logger.info("Loading configuration", extra={"environment": env, "config_path": str(config_path)})

    with open(config_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    try:
        gcs = GCSConfig(
            bucket=_resolve(raw["gcs"]["bucket"], "GCS_BUCKET"),
            bronze_prefix=raw["gcs"].get("bronze_prefix", "bronze/transactions"),
            silver_prefix=raw["gcs"].get("silver_prefix", "silver/transactions"),
            gold_prefix=raw["gcs"].get("gold_prefix", "gold/features"),
        )

        database = DatabaseConfig(
            host=_resolve(raw["database"]["host"], "DB_HOST"),
            port=int(_resolve(raw["database"]["port"], "DB_PORT", "5432")),
            database=_resolve(raw["database"]["database"], "DB_NAME"),
            username=_resolve(raw["database"]["username"], "DB_USER"),
            password=_resolve(raw["database"]["password"], "DB_PASSWORD"),
            schema=raw["database"].get("schema", "public"),
        )

        databricks = DatabricksConfig(
            host=_resolve(raw["databricks"]["host"], "DATABRICKS_HOST"),
            token=_resolve(raw["databricks"]["token"], "DATABRICKS_TOKEN"),
            cluster_id=_resolve(raw["databricks"]["cluster_id"], "DATABRICKS_CLUSTER_ID"),
            mlflow_tracking_uri=_resolve(
                raw["databricks"]["mlflow_tracking_uri"], "MLFLOW_TRACKING_URI"
            ),
        )

        bigquery = BigQueryConfig(
            project_id=_resolve(raw["bigquery"]["project_id"], "GCP_PROJECT_ID"),
            dataset=raw["bigquery"].get("dataset", "fraud_detection"),
            location=raw["bigquery"].get("location", "EU"),
        )

        ml_raw = raw.get("ml", {})
        ml = MLConfig(
            experiment_name=ml_raw.get("experiment_name", "fraud_detection"),
            model_name=ml_raw.get("model_name", "fraud_classifier"),
            test_size=float(ml_raw.get("test_size", 0.2)),
            random_state=int(ml_raw.get("random_state", 42)),
            n_estimators=int(ml_raw.get("n_estimators", 300)),
            max_depth=int(ml_raw.get("max_depth", 8)),
            fraud_threshold=float(ml_raw.get("fraud_threshold", 0.5)),
        )

        return PipelineConfig(
            environment=env,
            gcs=gcs,
            database=database,
            databricks=databricks,
            bigquery=bigquery,
            ml=ml,
            batch_size=int(raw.get("batch_size", 100_000)),
            max_retries=int(raw.get("max_retries", 3)),
            retry_delay_seconds=int(raw.get("retry_delay_seconds", 30)),
        )

    except KeyError as e:
        raise ConfigurationError(f"Missing required config key: {e}") from e


def _resolve(value: Any, env_var: str, default: str | None = None) -> str:
    """Resolve value from env var with fallback to config file value."""
    env_value = os.getenv(env_var)
    if env_value:
        return env_value
    if value and str(value) not in ("None", "null", ""):
        return str(value)
    if default is not None:
        return default
    raise ConfigurationError(
        f"Required value missing: set '{env_var}' env var or define in config file"
    )
