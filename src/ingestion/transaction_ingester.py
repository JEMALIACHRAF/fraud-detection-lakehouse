"""
Ingestion module — Extract transactions from PostgreSQL and land to GCS Bronze layer.

Design:
- Incremental extraction based on last watermark
- Batch processing to handle large volumes
- Idempotent writes (partition by extraction date)
- Schema validation before writing
"""

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterator

import psycopg2
import psycopg2.extras
from google.cloud import storage
from google.cloud.storage import Bucket

from src.common.config import PipelineConfig
from src.common.exceptions import IngestionError, SchemaValidationError
from src.common.logger import get_logger

logger = get_logger(__name__)

TRANSACTION_SCHEMA = {
    "transaction_id": str,
    "account_id": str,
    "amount": float,
    "currency": str,
    "merchant_id": str,
    "merchant_category": str,
    "transaction_type": str,
    "channel": str,
    "country_code": str,
    "city": str,
    "timestamp": str,
    "is_fraud": bool,
}

EXTRACT_QUERY = """
    SELECT
        transaction_id::text,
        account_id::text,
        amount::float,
        currency,
        merchant_id::text,
        merchant_category,
        transaction_type,
        channel,
        country_code,
        city,
        timestamp::text,
        is_fraud,
        created_at::text
    FROM {schema}.transactions
    WHERE created_at >= %(start_date)s
      AND created_at <  %(end_date)s
    ORDER BY created_at ASC
    LIMIT %(limit)s OFFSET %(offset)s
"""


class TransactionIngester:
    """
    Extracts transactions from PostgreSQL and writes to GCS Bronze layer.

    The Bronze layer stores raw data as-is in JSON Lines format,
    partitioned by extraction date: bronze/transactions/year=YYYY/month=MM/day=DD/
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._db_conn: psycopg2.extensions.connection | None = None
        self._gcs_client: storage.Client | None = None
        self._bucket: Bucket | None = None

    # ── Connection management ─────────────────────────────────────────────────

    def _get_db_connection(self) -> psycopg2.extensions.connection:
        if self._db_conn is None or self._db_conn.closed:
            db = self.config.database
            logger.info("Connecting to PostgreSQL", extra={"host": db.host, "database": db.database})
            try:
                self._db_conn = psycopg2.connect(
                    host=db.host,
                    port=db.port,
                    dbname=db.database,
                    user=db.username,
                    password=db.password,
                    connect_timeout=10,
                    options=f"-c search_path={db.schema}",
                )
                self._db_conn.set_session(readonly=True, autocommit=True)
            except psycopg2.Error as e:
                raise IngestionError(f"Failed to connect to PostgreSQL: {e}") from e
        return self._db_conn

    def _get_bucket(self) -> Bucket:
        if self._bucket is None:
            self._gcs_client = storage.Client()
            self._bucket = self._gcs_client.bucket(self.config.gcs.bucket)
        return self._bucket

    # ── Core extraction ───────────────────────────────────────────────────────

    def extract_batch(
        self,
        start_date: datetime,
        end_date: datetime,
        offset: int = 0,
    ) -> list[dict]:
        """
        Extract a single batch of transactions from PostgreSQL.

        Args:
            start_date: Inclusive lower bound
            end_date: Exclusive upper bound
            offset: Pagination offset

        Returns:
            List of transaction dicts
        """
        conn = self._get_db_connection()
        params = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "limit": self.config.batch_size,
            "offset": offset,
        }
        query = EXTRACT_QUERY.format(schema=self.config.database.schema)

        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                rows = [dict(row) for row in cur.fetchall()]
                logger.info(
                    "Extracted batch",
                    extra={"rows": len(rows), "offset": offset, "start_date": start_date.isoformat()},
                )
                return rows
        except psycopg2.Error as e:
            raise IngestionError(f"Extraction failed at offset {offset}: {e}") from e

    def extract_all(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Iterator[list[dict]]:
        """
        Generator that yields batches of transactions for the given window.

        Args:
            start_date: Inclusive lower bound
            end_date: Exclusive upper bound

        Yields:
            Batches of transaction dicts
        """
        offset = 0
        total_extracted = 0

        while True:
            batch = self.extract_batch(start_date, end_date, offset)
            if not batch:
                break

            yield batch
            total_extracted += len(batch)
            offset += self.config.batch_size

            if len(batch) < self.config.batch_size:
                break

        logger.info("Extraction complete", extra={"total_rows": total_extracted})

    # ── Schema validation ─────────────────────────────────────────────────────

    def validate_schema(self, records: list[dict]) -> None:
        """
        Validate that records contain required fields.

        Args:
            records: List of transaction dicts

        Raises:
            SchemaValidationError: if required fields are missing
        """
        required_fields = set(TRANSACTION_SCHEMA.keys())
        errors: list[str] = []

        for i, record in enumerate(records[:10]):  # sample first 10
            missing = required_fields - set(record.keys())
            if missing:
                errors.append(f"Row {i}: missing fields {missing}")

        if errors:
            raise SchemaValidationError(
                f"Schema validation failed ({len(errors)} errors):\n" + "\n".join(errors)
            )

    # ── GCS write ─────────────────────────────────────────────────────────────

    def write_to_bronze(
        self,
        records: list[dict],
        extraction_date: datetime,
        batch_index: int,
    ) -> str:
        """
        Write a batch to GCS Bronze layer as JSON Lines.

        Path pattern: bronze/transactions/year=YYYY/month=MM/day=DD/batch_{index}.jsonl

        Args:
            records: Validated transaction records
            extraction_date: Date used for partitioning
            batch_index: Batch number (for idempotent filenames)

        Returns:
            GCS URI of the written file
        """
        bucket = self._get_bucket()
        prefix = self.config.gcs.bronze_prefix

        blob_path = (
            f"{prefix}/"
            f"year={extraction_date.year}/"
            f"month={extraction_date.month:02d}/"
            f"day={extraction_date.day:02d}/"
            f"batch_{batch_index:04d}.jsonl"
        )

        # Add ingestion metadata to each record
        enriched = [
            {**record, "_ingested_at": datetime.now(timezone.utc).isoformat()}
            for record in records
        ]

        content = "\n".join(json.dumps(row, default=str) for row in enriched)

        try:
            blob = bucket.blob(blob_path)
            blob.upload_from_string(content, content_type="application/json")
            gcs_uri = f"gs://{self.config.gcs.bucket}/{blob_path}"
            logger.info(
                "Written to Bronze",
                extra={"gcs_uri": gcs_uri, "rows": len(records)},
            )
            return gcs_uri
        except Exception as e:
            raise IngestionError(f"Failed to write to GCS: {e}") from e

    # ── Orchestration ─────────────────────────────────────────────────────────

    def run(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict:
        """
        Run full ingestion for a time window.

        Args:
            start_date: Defaults to yesterday 00:00 UTC
            end_date: Defaults to today 00:00 UTC

        Returns:
            Ingestion summary dict
        """
        now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start = start_date or (now - timedelta(days=1))
        end = end_date or now

        logger.info(
            "Starting ingestion run",
            extra={"start_date": start.isoformat(), "end_date": end.isoformat()},
        )

        total_rows = 0
        total_batches = 0
        gcs_uris: list[str] = []
        errors: list[str] = []

        for batch_idx, batch in enumerate(self.extract_all(start, end)):
            try:
                self.validate_schema(batch)
                uri = self.write_to_bronze(batch, start, batch_idx)
                gcs_uris.append(uri)
                total_rows += len(batch)
                total_batches += 1

                # Retry-friendly: small sleep between batches
                if batch_idx > 0 and batch_idx % 10 == 0:
                    time.sleep(0.5)

            except SchemaValidationError as e:
                logger.error("Schema validation failed — skipping batch", extra={"error": str(e)})
                errors.append(str(e))
            except IngestionError as e:
                logger.error("Write failed — skipping batch", extra={"error": str(e)})
                errors.append(str(e))

        summary = {
            "status": "success" if not errors else "partial",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "total_rows": total_rows,
            "total_batches": total_batches,
            "gcs_uris": gcs_uris,
            "errors": errors,
        }

        logger.info("Ingestion run complete", extra=summary)
        return summary

    def close(self) -> None:
        if self._db_conn and not self._db_conn.closed:
            self._db_conn.close()
            logger.info("PostgreSQL connection closed")
