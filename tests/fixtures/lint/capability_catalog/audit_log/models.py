"""SQLAlchemy models stub for audit_log (test fixture)."""
from __future__ import annotations
from sqlalchemy import Column, String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
	pass


class AuditLogEntry(Base):
	__tablename__ = "audit_log_entries"
	id = Column(String, primary_key=True)
	tenant_id = Column(String, nullable=False)
	event = Column(String, nullable=False)
