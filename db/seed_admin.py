"""
Seed the first super_admin account from environment variables.

Usage::

    ADMIN_EMAIL=admin@lpu.in ADMIN_PASSWORD=changeme python -m db.seed_admin

If a super_admin already exists, the script is a no-op.
"""

from __future__ import annotations

import os
import sys

from db.models.admin_user import AdminUser
import db.models.app  # noqa: F401  # required for FK resolution
from db.postgres import get_sync_session
from sqlalchemy import select


def seed_super_admin() -> None:
    email = os.getenv("ADMIN_EMAIL")
    password = os.getenv("ADMIN_PASSWORD")

    if not email or not password:
        print("ADMIN_EMAIL and ADMIN_PASSWORD env vars required", file=sys.stderr)
        sys.exit(1)

    with get_sync_session() as session:
        existing = session.execute(
            select(AdminUser).where(AdminUser.role == "super_admin")
        ).scalar_one_or_none()

        if existing:
            print(f"super_admin already exists: {existing.email}")
            return

        admin = AdminUser(
            email=email,
            password_hash=AdminUser.hash_password(password),
            display_name="Super Admin",
            role="super_admin",
        )
        session.add(admin)
        session.commit()
        print(f"Created super_admin: {email}")


if __name__ == "__main__":
    seed_super_admin()