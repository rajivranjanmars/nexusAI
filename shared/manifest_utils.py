"""
Shared utilities for loading app manifest files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_app_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load an app credentials manifest JSON file.
    
    Args:
        manifest_path: Path to app_credentials.json
        
    Returns:
        Parsed manifest dictionary
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)
