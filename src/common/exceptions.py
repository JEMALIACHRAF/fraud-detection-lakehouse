"""
Custom exceptions for the fraud detection lakehouse pipeline.
"""


class LakehouseError(Exception):
    """Base exception for all pipeline errors."""


class IngestionError(LakehouseError):
    """Raised when data extraction from source fails."""


class TransformationError(LakehouseError):
    """Raised when a PySpark transformation fails."""


class DeltaWriteError(LakehouseError):
    """Raised when writing to Delta Lake fails."""


class SchemaValidationError(LakehouseError):
    """Raised when input data does not match expected schema."""


class MLflowError(LakehouseError):
    """Raised when MLflow tracking or registry operations fail."""


class ServingError(LakehouseError):
    """Raised when exporting data to BigQuery fails."""


class ConfigurationError(LakehouseError):
    """Raised when required configuration is missing or invalid."""
