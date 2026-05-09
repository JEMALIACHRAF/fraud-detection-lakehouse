"""
Upload local Bronze data to Azure Blob Storage.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv(Path(__file__).parents[1] / ".env")

conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
blob_service = BlobServiceClient.from_connection_string(conn_str)
container = blob_service.get_container_client("bronze")

data_dir = Path(__file__).parents[1] / "data" / "bronze" / "transactions"
files = list(data_dir.rglob("*.jsonl"))
print(f"Uploading {len(files)} files to Azure Blob Storage...")

for i, file in enumerate(files):
    relative = file.relative_to(data_dir.parent.parent)
    blob_name = str(relative).replace("\\", "/")
    with open(file, "rb") as f:
        container.upload_blob(blob_name, f, overwrite=True)
    print(f"  [{i+1}/{len(files)}] {blob_name}")

print(f"\nDone! {len(files)} files uploaded to container 'bronze'")
print(f"URL: https://fraudlakehouse.blob.core.windows.net/bronze/")