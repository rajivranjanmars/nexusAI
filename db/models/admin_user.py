"""
SQLAlchemy ORM model for admin dashboard users.

Maps to the ``admin_users`` table in PostgreSQL.  Admin users authenticate
with email + password (separate from the app-signed JWT flow) and are
assigned one of two roles: ``super_admin`` (platform-wide) or ``app_admin``
(scoped to a single registered application).
"""

from __future__ import annotations

import datetime
import hashlib
import os
import uuid

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.models.base import Base


class AdminUser(Base):
    """Persistent admin account for the Super Admin Portal and App Admin Dashboard."""

    __tablename__ = "admin_users"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4,
    )
    email: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True,
    )
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(
        String(32), nullable=False, default="app_admin",
        comment="super_admin | app_admin",
    )
    app_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("apps.app_id", ondelete="SET NULL"), nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_login_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(),
    )

    # ── Password helpers ───────────────────────────────────────────────────

    _PBKDF2_ITERATIONS = 600_000
    _SALT_BYTES = 32
    _HASH_NAME = "sha256"

    @classmethod
    def hash_password(cls, plaintext: str) -> str:
        """Return a ``$pbkdf2$<iterations>$<salt_hex>$<hash_hex>`` string."""
        salt = os.urandom(cls._SALT_BYTES)
        dk = hashlib.pbkdf2_hmac(
            cls._HASH_NAME,
            plaintext.encode("utf-8"),
            salt,
            cls._PBKDF2_ITERATIONS,
        )
        return f"$pbkdf2${cls._PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"

    def verify_password(self, plaintext: str) -> bool:
        """Check *plaintext* against the stored password hash."""
        try:
            _, _, iterations_s, salt_hex, expected_hex = self.password_hash.split("$", 4)
        except ValueError:
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(expected_hex)
        dk = hashlib.pbkdf2_hmac(
            self._HASH_NAME,
            plaintext.encode("utf-8"),
            salt,
            int(iterations_s),
        )
        return dk == expected