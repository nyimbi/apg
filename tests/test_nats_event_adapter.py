"""Tests for NATS JetStream event adapter.

These tests run without a live NATS server — they use pytest-httpserver or
direct mocking to validate adapter behaviour. Integration tests that require
a running NATS instance live in tests/integration/.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from capabilities.common.nats.subject_registry import (
	subject_for,
	parse_subject,
	subscribe_all_capability_events,
)
from capabilities.common.nats.nats_adapter import NATSEventAdapter, get_nats_audit_adapter


# ── subject_registry ──────────────────────────────────────────────────────────

def test_subject_for_normalizes_separators():
	assert subject_for("ckm_wfa", "workflow_started") == "apg.events.ckm_wfa.workflow_started"
	assert subject_for("fintech-gwy", "payment-received") == "apg.events.fintech_gwy.payment_received"


def test_subject_for_lowercases():
	assert subject_for("CKM_WFA", "WorkflowStarted") == "apg.events.ckm_wfa.workflowstarted"


def test_parse_subject_roundtrip():
	subject = subject_for("ckm_wfa", "task_assigned")
	result = parse_subject(subject)
	assert result == ("ckm_wfa", "task_assigned")


def test_parse_subject_invalid():
	assert parse_subject("not.an.apg.subject") is None
	assert parse_subject("apg.events") is None


def test_subscribe_all_capability_events_wildcard():
	pattern = subscribe_all_capability_events("ckm_wfa")
	assert pattern == "apg.events.ckm_wfa.>"


# ── NATSEventAdapter ──────────────────────────────────────────────────────────

async def test_nats_adapter_publishes_correct_subject():
	"""NATSEventAdapter publishes to the correct NATS subject."""
	mock_js = AsyncMock()
	adapter = NATSEventAdapter(capability_id="ckm_wfa")

	with patch("capabilities.common.nats.nats_adapter._get_js", return_value=mock_js):
		await adapter.log_event(
			"workflow_started", "user1", "tenant1", "wf-001", {"key": "val"}
		)

	mock_js.publish.assert_called_once()
	call_args = mock_js.publish.call_args
	subject = call_args[0][0]
	payload = json.loads(call_args[0][1].decode())

	assert subject == "apg.events.ckm_wfa.workflow_started"
	assert payload["event_type"] == "workflow_started"
	assert payload["actor_id"] == "user1"
	assert payload["tenant_id"] == "tenant1"
	assert payload["resource_id"] == "wf-001"
	assert "timestamp" in payload


async def test_nats_adapter_includes_msg_id_header_for_dedup():
	"""Exactly-once deduplication requires Msg-Id header."""
	mock_js = AsyncMock()
	adapter = NATSEventAdapter(capability_id="test_cap")

	with patch("capabilities.common.nats.nats_adapter._get_js", return_value=mock_js):
		await adapter.log_event("test_event", "u1", "t1", "r1", {})

	_, kwargs = mock_js.publish.call_args
	assert "Msg-Id" in (kwargs.get("headers") or mock_js.publish.call_args[1].get("headers", {}))


async def test_nats_adapter_retries_on_transient_failure():
	"""Adapter retries up to 3 times on publish failure."""
	attempt_count = 0

	async def flaky_publish(*args, **kwargs):
		nonlocal attempt_count
		attempt_count += 1
		if attempt_count < 3:
			raise Exception("transient NATS error")

	mock_js = AsyncMock()
	mock_js.publish = flaky_publish
	adapter = NATSEventAdapter(capability_id="test_cap")

	with patch("capabilities.common.nats.nats_adapter._get_js", return_value=mock_js):
		with patch("asyncio.sleep", new_callable=AsyncMock):
			await adapter.log_event("evt", "u", "t", "r", {})

	assert attempt_count == 3


async def test_nats_adapter_does_not_raise_after_all_retries_exhausted():
	"""Adapter logs error but does not propagate after 3 failed attempts."""
	mock_js = AsyncMock()
	mock_js.publish.side_effect = Exception("permanent failure")
	adapter = NATSEventAdapter(capability_id="test_cap")

	with patch("capabilities.common.nats.nats_adapter._get_js", return_value=mock_js):
		with patch("asyncio.sleep", new_callable=AsyncMock):
			# Must not raise
			await adapter.log_event("evt", "u", "t", "r", {})


def test_get_nats_audit_adapter_returns_none_without_env(monkeypatch):
	monkeypatch.delenv("NATS_URL", raising=False)
	assert get_nats_audit_adapter() is None


def test_get_nats_audit_adapter_returns_adapter_with_env(monkeypatch):
	monkeypatch.setenv("NATS_URL", "nats://localhost:4222")
	adapter = get_nats_audit_adapter("ckm_wfa")
	assert isinstance(adapter, NATSEventAdapter)
	assert adapter._capability_id == "ckm_wfa"


# ── Adapter factory integration ───────────────────────────────────────────────

def test_get_audit_adapter_uses_nats_when_url_set(monkeypatch):
	"""get_audit_adapter() returns NATSEventAdapter when NATS_URL is set."""
	monkeypatch.setenv("NATS_URL", "nats://localhost:4222")
	from capabilities.ckm.wfa.domain.adapters import get_audit_adapter
	adapter = get_audit_adapter()
	assert isinstance(adapter, NATSEventAdapter)


def test_get_audit_adapter_falls_back_to_null_without_nats(monkeypatch):
	"""get_audit_adapter() falls back to NullAuditAdapter with no env vars."""
	monkeypatch.delenv("NATS_URL", raising=False)
	from capabilities.ckm.wfa.domain.adapters import NullAuditAdapter, get_audit_adapter
	adapter = get_audit_adapter()
	assert isinstance(adapter, NullAuditAdapter)
