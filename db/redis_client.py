"""
DEPRECATED: Use `shared.redis_client` directly instead.

This module is kept for backward compatibility but will be removed in a future version.
Importing from this module will raise a DeprecationWarning.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from shared.redis_client import RateLimitState

# Emit a strong deprecation warning
warnings.warn(
    "db.redis_client is deprecated and will be removed in a future version. "
    "Use shared.redis_client directly.",
    DeprecationWarning,
    stacklevel=2
)

# Import from shared.redis_client
from shared.redis_client import get, set, delete, close, redis_client

# Re-export for backward compatibility
__all__ = ["get", "set", "delete", "close", "redis_client"]
