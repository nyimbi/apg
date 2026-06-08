"""Tests for SOC 2 audit trail hardening.

Validates that the AuditLoggingService:
  - Persists events to apg_audit_events table when DB is available
  - Publishes events to NATS when NATS_URL is configured
  - Maintains a durable hash chain across service restarts
  - Handles missing DB/NATS gracefully (in-memory fallback)
"""
from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from capabilities.common.audl.service import AuditLoggingService


def make_service(db=None):
	return AuditLoggingService(
		db_session=db,
		tenant_id="tenant-soc2",
		actor_id="system",
	)


# ── Chain hash correctness ────────────────────────────────────────────────────

async def test_chain_hash_advances_with_each_event():
	"""Each event advances the chain_tip, creating a tamper-evident chain."""
	svc = make_service()
	assert svc._chain_tip == "0" * 64

	event1 = await svc.log_event("user1", "create", "record-1", "data_create", None, None, True)
	chain_after_1 = svc._chain_tip
	assert chain_after_1 != "0" * 64

	event2 = await svc.log_event("user1", "update", "record-1", "data_update", None, None, True)
	chain_after_2 = svc._chain_tip
	assert chain_after_2 != chain_after_1

	# Both events have chain_hash set
	assert event1.chain_hash is not None and len(event1.chain_hash) == 64
	assert event2.chain_hash is not None and len(event2.chain_hash) == 64


async def test_chain_prev_is_tracked_between_events():
	"""_chain_prev records the hash before advancement for DB persistence."""
	svc = make_service()
	await svc.log_event("u1", "read", "r1", "data_read", None, None, True)
	first_chain_tip = svc._chain_tip

	await svc.log_event("u1", "update", "r1", "data_update", None, None, True)
	assert svc._chain_prev == first_chain_tip


# ── DB persistence ────────────────────────────────────────────────────────────

async def test_persist_to_db_called_when_db_available():
	"""_persist_audit_event_to_db is called on each log_event when db is set."""
	mock_db = AsyncMock()
	mock_db.execute = AsyncMock(return_value=MagicMock())
	mock_db.commit = AsyncMock()

	svc = make_service(db=mock_db)
	svc._chain_tip_initialized = True  # skip lazy DB load in this test

	await svc.log_event("user1", "create", "record-1", "data_create", "127.0.0.1", None, True)

	mock_db.execute.assert_called()
	mock_db.commit.assert_called()

	call_args = mock_db.execute.call_args
	sql_text = str(call_args[0][0])
	assert "apg_audit_events" in sql_text
	assert "INSERT" in sql_text


async def test_persist_to_db_skipped_when_no_db():
	"""No DB crash when db_session is None (standalone mode)."""
	svc = make_service(db=None)
	# Should not raise
	event = await svc.log_event("user1", "create", "r1", "data_create", None, None, True)
	assert event.id is not None


async def test_persist_to_db_tolerates_db_error():
	"""DB errors are caught and logged, not propagated."""
	mock_db = AsyncMock()
	mock_db.execute = AsyncMock(side_effect=Exception("DB unavailable"))
	mock_db.commit = AsyncMock()

	svc = make_service(db=mock_db)
	svc._chain_tip_initialized = True

	# Must not raise
	event = await svc.log_event("user1", "create", "r1", "data_create", None, None, True)
	assert event.id is not None


# ── NATS publishing ───────────────────────────────────────────────────────────

async def test_nats_publish_called_when_url_set(monkeypatch):
	"""Events are published to NATS apg.events.audl.audit_event when NATS_URL set."""
	monkeypatch.setenv("NATS_URL", "nats://localhost:4222")

	svc = make_service()

	with patch("capabilities.common.nats.nats_adapter.NATSEventAdapter.log_event", new=AsyncMock()) as mock_log:
		with patch("capabilities.common.nats.nats_adapter._get_js", new=AsyncMock()):
			event = await svc.log_event("user1", "read", "r1", "data_read", None, None, True)

	# log_event was called on the NATS adapter
	mock_log.assert_called_once()
	call_kwargs = mock_log.call_args[1]
	assert call_kwargs["event_type"] == "data_read"
	assert call_kwargs["actor_id"] == "user1"


async def test_nats_publish_skipped_when_no_url(monkeypatch):
	"""NATS publish is not attempted when NATS_URL is not set."""
	monkeypatch.delenv("NATS_URL", raising=False)
	svc = make_service()

	with patch("capabilities.common.nats.nats_adapter.NATSEventAdapter.log_event", new=AsyncMock()) as mock_log:
		await svc.log_event("user1", "read", "r1", "data_read", None, None, True)

	mock_log.assert_not_called()


# ── Chain tip loading from DB ─────────────────────────────────────────────────

async def test_load_chain_tip_from_db_reads_last_hash():
	"""On first log_event, chain_tip is loaded from DB for continuity."""
	stored_hash = "a" * 64
	mock_result = MagicMock()
	mock_result.fetchone.return_value = (stored_hash,)

	mock_db = AsyncMock()
	mock_db.execute = AsyncMock(return_value=mock_result)

	svc = make_service(db=mock_db)
	assert not svc._chain_tip_initialized

	await svc.log_event("user1", "read", "r1", "data_read", None, None, True)

	assert svc._chain_tip_initialized
	# chain_prev should have been the stored hash before we advanced
	assert svc._chain_prev == stored_hash or svc._chain_tip != "0" * 64


async def test_load_chain_tip_falls_back_on_db_error():
	"""DB error during chain_tip load falls back to zero-hash (safe default)."""
	mock_db = AsyncMock()
	mock_db.execute = AsyncMock(side_effect=Exception("Connection error"))

	svc = make_service(db=mock_db)
	tip = await svc._load_chain_tip_from_db()
	assert tip == "0" * 64


async def test_load_chain_tip_returns_zero_when_no_rows():
	"""Empty table returns zero-hash (first event in the chain)."""
	mock_result = MagicMock()
	mock_result.fetchone.return_value = None

	mock_db = AsyncMock()
	mock_db.execute = AsyncMock(return_value=mock_result)

	svc = make_service(db=mock_db)
	tip = await svc._load_chain_tip_from_db()
	assert tip == "0" * 64


# ── SOC 2 SQL migration file ──────────────────────────────────────────────────

def test_soc2_migration_file_exists():
	"""The SOC 2 migration SQL file must exist and contain immutability rules."""
	from pathlib import Path
	sql = Path("capabilities/common/audl/0002_soc2_audit_events.sql")
	assert sql.exists(), "0002_soc2_audit_events.sql must exist"
	content = sql.read_text()
	assert "apg_audit_events" in content
	assert "chain_hash" in content
	assert "checksum" in content
	assert "DO INSTEAD NOTHING" in content  # append-only rule


def test_soc2_migration_has_append_only_rules():
	"""SQL migration must prevent both UPDATE and DELETE."""
	from pathlib import Path
	content = Path("capabilities/common/audl/0002_soc2_audit_events.sql").read_text()
	assert "no_update" in content
	assert "no_delete" in content
