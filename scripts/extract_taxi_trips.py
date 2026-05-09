"""
scripts/extract_taxi_trips.py

Extract Chicago Taxi Trips from BigQuery public dataset.
Used as a proxy for financial transactions (amount, time, location, payment type).

Maps to our pipeline:
- trip_id          → transaction_id
- trip_total       → amount
- payment_type     → transaction_type
- company          → merchant_id
- pickup_community_area → merchant_category
- trip_start_timestamp → timestamp
- taxi_id          → account_id

Usage:
    python scripts/extract_taxi_trips.py --project your-project-id --limit 50000
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import bigquery

QUERY = """
SELECT
    COALESCE(unique_key, GENERATE_UUID())          AS transaction_id,
    COALESCE(taxi_id, 'UNKNOWN')                   AS account_id,
    COALESCE(company, 'UNKNOWN_COMPANY')            AS merchant_id,
    COALESCE(
        CAST(pickup_community_area AS STRING),
        'UNKNOWN'
    )                                               AS merchant_category,
    COALESCE(payment_type, 'unknown')               AS transaction_type,
    'taxi'                                          AS channel,
    'US'                                            AS country_code,
    'CHICAGO'                                       AS city,
    'USD'                                           AS currency,
    COALESCE(trip_total, 0.0)                       AS amount,
    TIMESTAMP(trip_start_timestamp)                 AS timestamp,
    CASE
        WHEN trip_total > 200 THEN TRUE
        WHEN trip_seconds < 60 AND trip_total > 50 THEN TRUE
        WHEN trip_miles < 0.1 AND trip_total > 30 THEN TRUE
        ELSE FALSE
    END                                             AS is_fraud,
    CURRENT_TIMESTAMP()                             AS created_at
FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE trip_start_timestamp IS NOT NULL
  AND trip_total IS NOT NULL
  AND trip_total > 0
  AND trip_total < 500
  AND taxi_id IS NOT NULL
  AND RAND() < 0.001  -- sample ~0.1% for speed
LIMIT @limit
"""


def extract_taxi_trips(
    project_id: str,
    output_dir: str,
    limit: int = 50_000,
    batch_size: int = 10_000,
) -> None:
    """
    Extract taxi trips from BigQuery and save as JSON Lines in Bronze format.

    Args:
        project_id: GCP project ID
        output_dir: Local output directory (Bronze layer)
        limit: Max rows to extract
        batch_size: Rows per output file
    """
    client = bigquery.Client(project=project_id)

    output_path = Path(output_dir) / "bronze" / "transactions" / "year=2024" / "month=01" / "day=15"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Extracting up to {limit:,} taxi trips from BigQuery...")
    print(f"Output: {output_path}")

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("limit", "INT64", limit)
        ]
    )

    query_job = client.query(QUERY, job_config=job_config)
    rows = list(query_job.result())

    print(f"Extracted {len(rows):,} rows")

    # Write in batches as JSON Lines
    total_written = 0
    batch_idx = 0

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        output_file = output_path / f"batch_{batch_idx:04d}.jsonl"

        with open(output_file, "w") as f:
            for row in batch:
                record = {
                    "transaction_id": row.transaction_id,
                    "account_id":     row.account_id,
                    "merchant_id":    row.merchant_id,
                    "merchant_category": str(row.merchant_category),
                    "transaction_type": str(row.transaction_type).lower().replace(" ", "_"),
                    "channel":        row.channel,
                    "country_code":   row.country_code,
                    "city":           row.city,
                    "currency":       row.currency,
                    "amount":         float(row.amount),
                    "timestamp":      row.timestamp.isoformat() if row.timestamp else None,
                    "is_fraud":       bool(row.is_fraud),
                    "created_at":     row.created_at.isoformat() if row.created_at else None,
                    "_ingested_at":   datetime.now(timezone.utc).isoformat(),
                    "_source":        "bigquery_chicago_taxi",
                }
                f.write(json.dumps(record) + "\n")

        total_written += len(batch)
        batch_idx += 1
        print(f"  Written batch {batch_idx}: {len(batch):,} rows → {output_file.name}")

    print(f"\nDone. Total written: {total_written:,} rows in {batch_idx} files")
    print(f"Fraud rate: {sum(1 for r in rows if r.is_fraud) / len(rows):.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Chicago taxi trips → Bronze layer")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--output-dir", default="data", help="Local output directory")
    parser.add_argument("--limit", type=int, default=50_000, help="Max rows to extract")
    parser.add_argument("--batch-size", type=int, default=10_000, help="Rows per file")
    args = parser.parse_args()

    extract_taxi_trips(
        project_id=args.project,
        output_dir=args.output_dir,
        limit=args.limit,
        batch_size=args.batch_size,
    )
