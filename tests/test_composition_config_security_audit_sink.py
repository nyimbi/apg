"""Regressions for Central Configuration security audit persistence."""

from __future__ import annotations

import json

import pytest

from capabilities.composition.config import security_engine


@pytest.mark.asyncio
async def test_security_engine_imports_without_optional_jose_dependency(tmp_path):
	engine = security_engine.CentralConfigurationSecurity(
		audit_store_path=tmp_path / "audit.jsonl"
	)

	assert hasattr(security_engine, "JOSE_AVAILABLE")
	assert engine.audit_log_path.name == "audit.jsonl"


@pytest.mark.asyncio
async def test_security_audit_event_persists_jsonl_and_forwards_to_siem_client(tmp_path):
	delivered: list[dict] = []

	async def siem_client(payload: dict) -> None:
		delivered.append(payload)

	audit_log = tmp_path / "audit" / "security-events.jsonl"
	engine = security_engine.CentralConfigurationSecurity(
		audit_store_path=audit_log,
		siem_client=siem_client,
	)

	await engine._audit_security_event(
		event_type="authorization",
		resource_id="config/payment/limits",
		action="read_config",
		result="success",
		user_id="user-a",
		metadata={"tenant_id": "tenant-a", "source": "regression"},
		ip_address="10.0.0.7",
		user_agent="pytest",
	)

	lines = audit_log.read_text(encoding="utf-8").splitlines()
	assert len(lines) == 1

	stored = json.loads(lines[0])
	assert stored["event_type"] == "authorization"
	assert stored["resource_id"] == "config/payment/limits"
	assert stored["user_id"] == "user-a"
	assert stored["metadata"] == {"tenant_id": "tenant-a", "source": "regression"}
	assert "timestamp" in stored

	assert delivered == [stored]
	assert engine.siem_delivery_failures == []


@pytest.mark.asyncio
async def test_security_siem_failures_are_recorded_without_losing_audit_event(tmp_path):
	def failing_siem_client(payload: dict) -> None:
		raise RuntimeError("siem unavailable")

	audit_log = tmp_path / "security-events.jsonl"
	engine = security_engine.CentralConfigurationSecurity(
		audit_store_path=audit_log,
		siem_client=failing_siem_client,
	)

	await engine._audit_security_event(
		event_type="authorization",
		resource_id="config/payment/limits",
		action="deny_config",
		result="denied",
		user_id="user-a",
		metadata={"tenant_id": "tenant-a"},
	)

	assert audit_log.exists()
	assert len(audit_log.read_text(encoding="utf-8").splitlines()) == 1
	assert len(engine.siem_delivery_failures) == 1
	assert engine.siem_delivery_failures[0]["error"] == "siem unavailable"
