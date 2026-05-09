"""
Create Azure Blob Storage account + container for the lakehouse.
Bronze/Silver/Gold layers will be stored here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from azure.identity import AzureCliCredential
from azure.mgmt.storage import StorageManagementClient
from azure.mgmt.storage.models import (
    StorageAccountCreateParameters,
    Sku,
    SkuName,
    Kind,
)

load_dotenv(Path(__file__).parents[1] / ".env")

SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP  = "fraud-pipeline-rg"
LOCATION        = "westeurope"
STORAGE_ACCOUNT = "fraudlakehouse"   # lowercase, no dashes, max 24 chars
CONTAINER_NAME  = "lakehouse"

credential = AzureCliCredential()
storage_client = StorageManagementClient(credential, SUBSCRIPTION_ID)

# Create storage account
print("Creating storage account...")
poller = storage_client.storage_accounts.begin_create(
    RESOURCE_GROUP,
    STORAGE_ACCOUNT,
    StorageAccountCreateParameters(
        sku=Sku(name=SkuName.STANDARD_LRS),
        kind=Kind.STORAGE_V2,
        location=LOCATION,
        enable_https_traffic_only=True,
    )
)
account = poller.result()
print(f"  Storage account: {account.name}")

# Get connection string
keys = storage_client.storage_accounts.list_keys(RESOURCE_GROUP, STORAGE_ACCOUNT)
connection_string = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={STORAGE_ACCOUNT};"
    f"AccountKey={keys.keys[0].value};"
    f"EndpointSuffix=core.windows.net"
)

# Create container
from azure.storage.blob import BlobServiceClient
blob_service = BlobServiceClient.from_connection_string(connection_string)

for container in ["bronze", "silver", "gold", "mlflow"]:
    try:
        blob_service.create_container(container)
        print(f"  Container created: {container}")
    except Exception:
        print(f"  Container exists: {container}")

print(f"\nAdd this to your .env:")
print(f"AZURE_STORAGE_ACCOUNT={STORAGE_ACCOUNT}")
print(f"AZURE_STORAGE_CONNECTION_STRING={connection_string}")
print(f"AZURE_STORAGE_CONTAINER=lakehouse")