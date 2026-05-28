"""Standalone SQLAlchemy model shims for legacy CRM subpackages."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, String
from sqlalchemy.orm import declarative_base
from uuid_extensions import uuid7str


Model = declarative_base()


class BaseMixin:
	"""Minimal primary-key mixin used when APG auth_rbac models are absent."""

	id = Column(String(36), primary_key=True, default=uuid7str)


class AuditMixin:
	"""Minimal audit mixin for standalone model import and local execution."""

	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
	created_by = Column(String(100), nullable=True)
	updated_by = Column(String(100), nullable=True)
