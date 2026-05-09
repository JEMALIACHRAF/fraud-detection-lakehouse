"""
Upload local data and src/ code to Databricks Workspace Files.
Uses /Workspace/Users/{email}/fraud-pipeline/ instead of DBFS.
"""

import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat, Language

load_dotenv(Path(__file__).parents[1] / ".env")

client = WorkspaceClient(
    host=os.getenv("DATABRICKS_HOST"),
    token=os.getenv("DATABRICKS_TOKEN"),
)

# Get current user email
me = client.current_user.me()
user_email = me.user_name
workspace_base = f"/Users/{user_email}/fraud-pipeline"

print(f"Uploading to: {workspace_base}")

def upload_file(local_path: Path, workspace_path: str) -> None:
    """Upload a single file to Databricks Workspace."""
    with open(local_path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")

    # Determine language
    suffix = local_path.suffix
    language = None
    fmt = ImportFormat.AUTO

    if suffix == ".py":
        language = Language.PYTHON
        fmt = ImportFormat.SOURCE

    try:
        client.workspace.import_(
            path=workspace_path,
            content=content,
            format=fmt,
            language=language,
            overwrite=True,
        )
        print(f"  ✓ {local_path.name}")
    except Exception as e:
        print(f"  ✗ {local_path.name}: {e}")

def upload_dir(local_dir: Path, workspace_prefix: str) -> None:
    """Recursively upload a directory to Databricks Workspace."""
    for file in sorted(local_dir.rglob("*")):
        if file.is_file() and not any(
            p in str(file) for p in ["__pycache__", ".pyc", ".egg-info", ".git"]
        ):
            relative = file.relative_to(local_dir.parent)
            ws_path = f"{workspace_prefix}/{relative}".replace("\\", "/")

            # Create parent directory
            parent = "/".join(ws_path.split("/")[:-1])
            try:
                client.workspace.mkdirs(parent)
            except Exception:
                pass

            upload_file(file, ws_path)

project_root = Path(__file__).parents[1]

print("\n[1/3] Uploading source code (src/)...")
upload_dir(project_root / "src", workspace_base)

print("\n[2/3] Uploading config/...")
upload_dir(project_root / "config", workspace_base)

print("\n[3/3] Uploading scripts/...")
upload_dir(project_root / "scripts", workspace_base)

print(f"\nDone! Files available at: {workspace_base}")
print(f"Open in Databricks: {os.getenv('DATABRICKS_HOST')}#workspace{workspace_base}")