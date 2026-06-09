"""auth_rbac stub — APG base model mixins and common auth classes."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String, Boolean, Text
from sqlalchemy.orm import declarative_base


Model = declarative_base()


class BaseMixin:
	"""Provides id, created_at, updated_at to any Model subclass."""
	__abstract__ = True

	id = Column(String(36), primary_key=True)
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	updated_at = Column(
		DateTime(timezone=True),
		default=lambda: datetime.now(timezone.utc),
		onupdate=lambda: datetime.now(timezone.utc),
	)


class AuditMixin:
	"""Provides created_by, updated_by, is_deleted audit fields."""
	__abstract__ = True

	created_by = Column(String(255), nullable=True)
	updated_by = Column(String(255), nullable=True)
	is_deleted = Column(Boolean, default=False, nullable=False)
	tenant_id = Column(String(36), nullable=True, index=True)


class User(Model, BaseMixin):
	"""Minimal User stub for capabilities that reference auth_rbac.User."""
	__tablename__ = "apg_users_stub"

	username = Column(String(255), unique=True, nullable=False, default="")
	email = Column(String(255), unique=True, nullable=True)
	first_name = Column(String(255), nullable=True)
	last_name = Column(String(255), nullable=True)
	is_active = Column(Boolean, default=True)
	role = Column(String(100), nullable=True)


class Role(Model, BaseMixin):
	"""Minimal Role stub."""
	__tablename__ = "apg_roles_stub"

	name = Column(String(100), unique=True, nullable=False, default="")
	description = Column(Text, nullable=True)


class Permission(Model, BaseMixin):
	"""Minimal Permission stub."""
	__tablename__ = "apg_permissions_stub"

	name = Column(String(255), unique=True, nullable=False, default="")
	resource = Column(String(255), nullable=True)
	action = Column(String(100), nullable=True)


# Common exports
__all__ = ["Model", "BaseMixin", "AuditMixin", "User", "Role", "Permission"]
