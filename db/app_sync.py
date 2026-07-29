"""
App Registry Synchronization.
"""
from __future__ import annotations
import hashlib
import os
import yaml
from db.postgres import get_sync_session
from db.models.app import App
from shared.logger import get_logger

logger = get_logger(__name__)

_APPS_YAML_PATH = os.path.join(
    os.path.dirname(__file__), "..", "mcp_server", "apps.yaml"
)

def sync_apps() -> None:
    """Synchronize YAML configuration to the database."""
    if not os.path.exists(_APPS_YAML_PATH):
        logger.error("Apps YAML not found at %s", _APPS_YAML_PATH)
        return

    logger.info("Starting App Registry sync from YAML...")

    with open(_APPS_YAML_PATH, "r") as f:
        config = yaml.safe_load(f)

    apps_list = config.get("apps", [])
    if not apps_list:
        logger.warning("No apps defined in YAML.")
        return

    with get_sync_session() as session:
        for app in apps_list:
            app_id = app.get("app_id")
            
            # Using session.merge handles the upsert
            app_obj = App(
                app_id=app_id,
                app_name=app["name"],
                public_key=app.get("public_key", ""),
                domain=app.get("domain", "http://localhost:3000"),
                allowed_workflows=app.get("allowed_workflows", []),
                allowed_tools=app.get("allowed_tools", []),
                rate_limit_rpm=app.get("rate_limit_rpm", 60),
                token_quota_monthly=app.get("token_quota_monthly"),
                cache_ttl_seconds=app.get("cache_ttl_seconds", 3600),
                app_config=app.get("app_config", {}),
                metadata_=app.get("metadata", {}),
            )
            session.merge(app_obj)
            logger.info("Synchronized app: %s", app["name"])
        session.commit()

    logger.info("App Registry synchronization complete.")

if __name__ == "__main__":
    sync_apps()
