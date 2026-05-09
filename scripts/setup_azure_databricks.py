"""
Setup Azure Databricks workspace via Python SDK.
Run once to provision the infrastructure.
"""

from azure.identity import AzureCliCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.databricks import AzureDatabricksManagementClient
from azure.mgmt.databricks.models import Workspace, Sku

SUBSCRIPTION_ID = "c6ae49d1-b762-4e11-8d7e-68271a5c31dd"
RESOURCE_GROUP  = "fraud-pipeline-rg"
LOCATION        = "westeurope"
WORKSPACE_NAME  = "fraud-lakehouse-ws"
SKU             = "trial"  # 14 jours gratuit

credential = AzureCliCredential()

# 1. Resource Group
print("Creating resource group...")
rg_client = ResourceManagementClient(credential, SUBSCRIPTION_ID)
rg_client.resource_groups.create_or_update(
    RESOURCE_GROUP,
    {"location": LOCATION}
)
print(f"  Resource group '{RESOURCE_GROUP}' ready")

# 2. Databricks Workspace
print("Creating Databricks workspace (5-10 min)...")
db_client = AzureDatabricksManagementClient(credential, SUBSCRIPTION_ID)
poller = db_client.workspaces.begin_create_or_update(
    RESOURCE_GROUP,
    WORKSPACE_NAME,
    Workspace(
        location=LOCATION,
        sku=Sku(name=SKU),
        managed_resource_group_id=f"/subscriptions/{SUBSCRIPTION_ID}/resourceGroups/{WORKSPACE_NAME}-managed-rg",
    )
)
workspace = poller.result()
print(f"  Workspace URL: https://{workspace.workspace_url}")
print(f"  Workspace ID:  {workspace.workspace_id}")
print(f"\nDone! Open: https://{workspace.workspace_url}")