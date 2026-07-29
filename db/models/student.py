"""
SQLAlchemy ORM model for the Student table.

Maps to the ``students`` table in PostgreSQL and exposes audit-friendly
``updated_at`` tracking.
"""

from __future__ import annotations

import datetime
from typing import Optional

from sqlalchemy import Date, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from db.models.base import Base


class Student(Base):
    """Represents a student record in PostgreSQL."""

    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    first_name: Mapped[str] = mapped_column(String(128), nullable=False)
    last_name: Mapped[str] = mapped_column(String(128), nullable=False)
    email: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    program: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    enrollment_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="active"
    )
    gpa: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    attendance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_classes_held: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    classes_attended: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_semester_classes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fee_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    next_exam: Mapped[Optional[datetime.date]] = mapped_column(Date, nullable=True)
    active_courses: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def to_dict(self) -> dict:
        """Serialize the model to a plain dictionary."""
        return {
            "student_id": self.student_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "program": self.program,
            "enrollment_status": self.enrollment_status,
            "gpa": self.gpa,
            "attendance": self.attendance,
            "total_classes_held": self.total_classes_held,
            "classes_attended": self.classes_attended,
            "total_semester_classes": self.total_semester_classes,
            "fee_status": self.fee_status,
            "next_exam": self.next_exam.isoformat() if self.next_exam else None,
            "active_courses": self.active_courses,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
