"""Domain adapters for SACCO GL — APG infrastructure integration."""
from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)


def get_audit_adapter(capability_id: str = "fintech_sacco_gl"):
	"""Return NATS event adapter if available, else None."""
	nats_url = os.environ.get("NATS_URL")
	if nats_url:
		try:
			from capabilities.common.nats.nats_adapter import NATSEventAdapter
			return NATSEventAdapter(capability_id)
		except Exception as exc:
			_log.debug("NATS unavailable: %s", exc)
	return None


def get_db_adapter():
	"""Return async SQLAlchemy session factory if DATABASE_URL is set."""
	db_url = os.environ.get("DATABASE_URL")
	if db_url:
		try:
			from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
			from sqlalchemy.orm import sessionmaker
			engine = create_async_engine(db_url, echo=False)
			return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
		except Exception as exc:
			_log.debug("DB adapter unavailable: %s", exc)
	return None
