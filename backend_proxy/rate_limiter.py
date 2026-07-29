"""
DEPRECATED: Use `shared.redis_client` directly instead.

This module is kept for backward compatibility but will be removed in a future version.
Importing from this module will raise a DeprecationWarning.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.redis_client import RateLimitState, RateLimitExceeded

# Emit a strong deprecation warning
warnings.warn(
    "backend_proxy.rate_limiter is deprecated and will be removed in a future version. "
    "Use shared.redis_client directly.",
    DeprecationWarning,
    stacklevel=2
)

# Import from shared.redis_client
from shared.redis_client import RateLimitState, RateLimitExceeded, enforce_rate_limit, close

# Re-export for backward compatibility
__all__ = ["RateLimitState", "RateLimitExceeded", "enforce_rate_limit", "close"]
