"""
Setup Databricks cluster + install libraries via Python SDK.
Reads credentials from .env file — never hardcode secrets.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient

# Load .env from project root
load_dotenv(Path(__file__).parents[1] / ".env")

DATABRICKS_HOST  = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
    raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN must be set in .env")

client = WorkspaceClient(host=DATABRICKS_HOST, token=DATABRICKS_TOKEN)

print("Creating cluster...")
from databricks.sdk.service.compute import DataSecurityMode

from databricks.sdk.service.compute import DataSecurityMode

cluster = client.clusters.create(
    cluster_name="fraud-pipeline-cluster",
    spark_version="13.3.x-scala2.12",
    node_type_id="Standard_D4s_v3",
    num_workers=1,
    data_security_mode=DataSecurityMode.SINGLE_USER,
    single_user_name=client.current_user.me().user_name,
    spark_conf={
        "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
        "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        "spark.databricks.delta.retentionDurationCheck.enabled": "false",
    },
    autotermination_minutes=60,
).result()

print(f"  Cluster ID: {cluster.cluster_id}")
print(f"  State:      {cluster.state}")

print("Installing libraries...")
client.libraries.install(
    cluster_id=cluster.cluster_id,
    libraries=[
        {"pypi": {"package": "xgboost==2.0.3"}},
        {"pypi": {"package": "imbalanced-learn==0.11.0"}},
        {"pypi": {"package": "google-cloud-bigquery==3.14.0"}},
        {"pypi": {"package": "google-cloud-storage==2.14.0"}},
    ]
)

print(f"\nCluster ID: {cluster.cluster_id}")
print("Add this to your .env: DATABRICKS_CLUSTER_ID=" + cluster.cluster_id)