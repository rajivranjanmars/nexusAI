"""
Base model mixins for common functionality across database models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


class ToDictMixin:
    """Mixin class providing a generic to_dict() method for SQLAlchemy models."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary representation.
        
        Returns:
            Dictionary with column names as keys and column values as values.
            datetime objects are converted to ISO format strings.
        """
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            result[column.name] = value
        return result
