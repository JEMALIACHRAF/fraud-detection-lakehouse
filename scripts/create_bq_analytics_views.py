"""
Create BigQuery analytical views for fraud detection dashboard.
IaC approach — all views defined as code, versioned in GitHub.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv(Path(__file__).parents[1] / ".env")

GCP_PROJECT = os.getenv("GCP_PROJECT_ID", "projet-dbt-495310")
BQ_DATASET  = "fraud_detection"

client = bigquery.Client(project=GCP_PROJECT)

VIEWS = {
    "vw_daily_fraud_summary": f"""
        SELECT
            DATE(feature_date)                          AS date,
            COUNT(*)                                    AS total_alerts,
            SUM(amount)                                 AS total_fraud_amount,
            AVG(amount)                                 AS avg_fraud_amount,
            MAX(amount)                                 AS max_fraud_amount,
            AVG(amount_zscore)                          AS avg_zscore,
            COUNTIF(is_night_transaction)               AS night_transactions,
            COUNTIF(is_weekend)                         AS weekend_transactions,
            COUNT(DISTINCT account_id)                  AS distinct_accounts
        FROM `{GCP_PROJECT}.{BQ_DATASET}.fraud_alerts`
        GROUP BY 1
        ORDER BY 1 DESC
    """,

    "vw_fraud_by_country": f"""
        SELECT
            country_code,
            COUNT(*)                                    AS fraud_count,
            SUM(amount)                                 AS total_amount,
            AVG(amount)                                 AS avg_amount,
            AVG(amount_zscore)                          AS avg_zscore
        FROM `{GCP_PROJECT}.{BQ_DATASET}.fraud_alerts`
        GROUP BY 1
        ORDER BY 2 DESC
    """,

    "vw_fraud_by_channel": f"""
        SELECT
            channel,
            transaction_type,
            COUNT(*)                                    AS fraud_count,
            SUM(amount)                                 AS total_amount,
            AVG(amount)                                 AS avg_amount
        FROM `{GCP_PROJECT}.{BQ_DATASET}.fraud_alerts`
        GROUP BY 1, 2
        ORDER BY 3 DESC
    """,

    "vw_top_risky_accounts": f"""
        SELECT
            account_id,
            total_transactions,
            fraud_count,
            ROUND(fraud_rate * 100, 2)                  AS fraud_rate_pct,
            ROUND(total_amount, 2)                      AS total_amount,
            ROUND(avg_amount, 2)                        AS avg_amount,
            ROUND(max_zscore, 2)                        AS max_zscore,
            max_tx_count_24h
        FROM `{GCP_PROJECT}.{BQ_DATASET}.account_risk_profile`
        WHERE fraud_count > 0
        ORDER BY fraud_rate DESC
        LIMIT 100
    """,

    "vw_model_comparison": f"""
        SELECT
            model,
            ROUND(roc_auc, 4)                           AS roc_auc,
            ROUND(average_precision, 4)                 AS average_precision,
            ROUND(f1_score, 4)                          AS f1_score,
            ROUND(precision_fraud, 4)                   AS precision_fraud,
            ROUND(recall_fraud, 4)                      AS recall_fraud,
            true_positives,
            false_positives,
            false_negatives,
            true_positives + false_negatives            AS total_fraud_cases,
            ROUND(
                true_positives * 100.0 /
                NULLIF(true_positives + false_negatives, 0), 1
            )                                           AS detection_rate_pct
        FROM `{GCP_PROJECT}.{BQ_DATASET}.model_performance`
        ORDER BY roc_auc DESC
    """,

    "vw_hourly_fraud_pattern": f"""
        SELECT
            tx_count_24h                                AS velocity_24h,
            COUNT(*)                                    AS fraud_count,
            AVG(amount)                                 AS avg_amount,
            AVG(amount_zscore)                          AS avg_zscore
        FROM `{GCP_PROJECT}.{BQ_DATASET}.fraud_alerts`
        GROUP BY 1
        ORDER BY 1
    """,
}

print(f"Creating {len(VIEWS)} analytical views in {GCP_PROJECT}.{BQ_DATASET}...\n")

for view_name, query in VIEWS.items():
    full_name = f"{GCP_PROJECT}.{BQ_DATASET}.{view_name}"
    view = bigquery.Table(full_name)
    view.view_query = query.strip()

    try:
        client.delete_table(full_name, not_found_ok=True)
        client.create_table(view)
        print(f"  ✓ {view_name}")
    except Exception as e:
        print(f"  ✗ {view_name}: {e}")

print(f"\nDone! Views available in BigQuery:")
print(f"  https://console.cloud.google.com/bigquery?project={GCP_PROJECT}")
print(f"\nConnect to Looker Studio:")
print(f"  https://lookerstudio.google.com/")