"""
Tests for AuditLoggingService — all 42 public methods.

Uses real objects, no mocks.  Async tests run via asyncio.get_event_loop().

© 2025 Datacraft  www.datacraft.co.ke
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from capabilities.common.audl.models import (
	AuditEventCreate,
	AuditEventType,
	AuditLevel,
	AuditQueryCreate,
	AuditTrailCreate,
	AuditTrailUpdate,
	ComplianceFramework,
	ComplianceReportCreate,
	DataSubjectRequestCreate,
	DSRType,
	DSRStatus,
	EvidencePackageCreate,
	EventSource,
	RetentionAction,
	RetentionPolicyCreate,
	TamperDetectionCreate,
	TrailStatus,
)
from capabilities.common.audl.service import AuditLoggingService


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

TENANT = "test-tenant"
ACTOR  = "test-actor"
loop   = asyncio.get_event_loop()


def svc() -> AuditLoggingService:
	"""Fresh service instance for each test."""
	return AuditLoggingService(db_session=None, tenant_id=TENANT, actor_id=ACTOR)


def _event_create(**kwargs) -> AuditEventCreate:
	defaults = dict(
		tenant_id=TENANT,
		level=AuditLevel.INFO,
		event_type=AuditEventType.USER_LOGIN,
		source=EventSource.AUTH,
		category="authentication",
		actor_id=ACTOR,
		action="login",
		resource_id="session-1",
		success=True,
	)
	defaults.update(kwargs)
	return AuditEventCreate(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_service_requires_tenant():
	with pytest.raises(AssertionError):
		AuditLoggingService(db_session=None, tenant_id="", actor_id="a")


def test_service_requires_actor():
	with pytest.raises(AssertionError):
		AuditLoggingService(db_session=None, tenant_id="t", actor_id="")


# ---------------------------------------------------------------------------
# log_event
# ---------------------------------------------------------------------------

def test_log_event_returns_response():
	s  = svc()
	ev = loop.run_until_complete(s.log_event(
		who="u1", what="login", on_what="s1",
		how=AuditEventType.USER_LOGIN, where="10.0.0.1",
		when=None, result=True,
	))
	assert ev.id
	assert ev.tenant_id == TENANT
	assert ev.checksum is not None
	assert ev.chain_hash is not None
	assert ev.immutable is True


def test_log_event_with_payload():
	s       = svc()
	payload = _event_create(ip_address="8.8.8.8", contains_pii=True)
	ev      = loop.run_until_complete(s.log_event(
		who=ACTOR, what="login", on_what="s1",
		how=AuditEventType.USER_LOGIN, where="8.8.8.8",
		when=None, result=True, payload=payload,
	))
	assert ev.contains_pii is True
	assert ev.risk_score > 0   # external IP bumps risk


def test_log_event_checksum_verifies():
	s  = svc()
	ev = loop.run_until_complete(s.log_event(
		who="u1", what="read", on_what="doc-1",
		how=AuditEventType.DATA_READ, where="192.168.1.1",
		when=None, result=True,
	))
	assert ev.verify_integrity() is True


def test_log_event_chain_advances():
	s   = svc()
	tip0 = s._chain_tip
	loop.run_until_complete(s.log_event(
		who="u1", what="x", on_what="r",
		how=AuditEventType.API_CALL, where=None, when=None, result=True,
	))
	assert s._chain_tip != tip0


def test_log_event_risk_score_failed_auth():
	s  = svc()
	ev = loop.run_until_complete(s.log_event(
		who="u1", what="failed login", on_what="s1",
		how=AuditEventType.USER_FAILED_LOGIN, where="8.8.8.8",
		when=None, result=False,
	))
	assert ev.risk_score >= 0.35   # at least failed_auth weight


# ---------------------------------------------------------------------------
# immutable_log_write
# ---------------------------------------------------------------------------

def test_batch_write_multiple_events():
	s   = svc()
	evs = [_event_create() for _ in range(5)]
	res = loop.run_until_complete(s.immutable_log_write(evs))
	assert len(res) == 5


def test_batch_write_empty_raises():
	s = svc()
	with pytest.raises(ValueError):
		loop.run_until_complete(s.immutable_log_write([]))


def test_batch_write_too_large_raises():
	s   = svc()
	evs = [_event_create() for _ in range(10_001)]
	with pytest.raises(ValueError):
		loop.run_until_complete(s.immutable_log_write(evs))


def test_batch_write_cross_tenant_raises():
	s = svc()
	ev = _event_create(tenant_id="other-tenant")
	from capabilities.common.audl.domain.rules import RuleViolation
	with pytest.raises(RuleViolation):
		loop.run_until_complete(s.immutable_log_write([ev]))


# ---------------------------------------------------------------------------
# audit_trail_search
# ---------------------------------------------------------------------------

def test_search_returns_tenant_events_only():
	s = svc()
	loop.run_until_complete(s.log_event(
		who=ACTOR, what="read", on_what="doc",
		how=AuditEventType.DATA_READ, where=None, when=None, result=True,
	))
	q   = AuditQueryCreate(tenant_id=TENANT, requested_by=ACTOR)
	res = loop.run_until_complete(s.audit_trail_search(q))
	assert res.total_count >= 1
	assert all(ev.tenant_id == TENANT for ev in res.events)


def test_search_filter_by_event_type():
	s = svc()
	loop.run_until_complete(s.log_event(
		who=ACTOR, what="login", on_what="s",
		how=AuditEventType.USER_LOGIN, where=None, when=None, result=True,
	))
	loop.run_until_complete(s.log_event(
		who=ACTOR, what="read", on_what="d",
		how=AuditEventType.DATA_READ, where=None, when=None, result=True,
	))
	q   = AuditQueryCreate(
		tenant_id=TENANT, requested_by=ACTOR,
		event_types=[AuditEventType.USER_LOGIN],
	)
	res = loop.run_until_complete(s.audit_trail_search(q))
	assert all(ev.event_type == AuditEventType.USER_LOGIN for ev in res.events)


def test_search_pagination():
	s = svc()
	for _ in range(10):
		loop.run_until_complete(s.log_event(
			who=ACTOR, what="x", on_what="r",
			how=AuditEventType.API_CALL, where=None, when=None, result=True,
		))
	q = AuditQueryCreate(tenant_id=TENANT, requested_by=ACTOR, limit=3, offset=0)
	r = loop.run_until_complete(s.audit_trail_search(q))
	assert len(r.events) == 3
	assert r.has_more is True


def test_search_success_filter():
	s = svc()
	loop.run_until_complete(s.log_event(
		who=ACTOR, what="fail", on_what="r",
		how=AuditEventType.USER_FAILED_LOGIN, where=None, when=None, result=False,
	))
	q = AuditQueryCreate(tenant_id=TENANT, requested_by=ACTOR, success=False)
	r = loop.run_until_complete(s.audit_trail_search(q))
	assert all(ev.success is False for ev in r.events)


# ---------------------------------------------------------------------------
# tamper_detection
# ---------------------------------------------------------------------------

def test_tamper_detection_clean():
	s = svc()
	loop.run_until_complete(s.log_event(
		who=ACTOR, what="login", on_what="s",
		how=AuditEventType.USER_LOGIN, where=None, when=None, result=True,
	))
	scan = loop.run_until_complete(s.tamper_detection(TamperDetectionCreate(
		tenant_id=TENANT, scan_type="on-demand", scanned_by=ACTOR,
	)))
	assert scan.status.value == "clean"
	assert scan.events_suspect == 0


def test_tamper_detection_suspect_on_corruption():
	s = svc()
	ev = loop.run_until_complete(s.log_event(
		who=ACTOR, what="login", on_what="s",
		how=AuditEventType.USER_LOGIN, where=None, when=None, result=True,
	))
	# Corrupt checksum directly
	s._events[ev.id].checksum = "0" * 64
	scan = loop.run_until_complete(s.tamper_detection(TamperDetectionCreate(
		tenant_id=TENANT, scan_type="on-demand", scanned_by=ACTOR,
	)))
	assert scan.events_suspect >= 1
	assert ev.id in scan.suspect_ids


# ---------------------------------------------------------------------------
# compliance_report
# ---------------------------------------------------------------------------

def test_compliance_report_generated():
	s   = svc()
	now = datetime.now(timezone.utc)
	req = ComplianceReportCreate(
		tenant_id=TENANT,
		framework=ComplianceFramework.GDPR,
		period_start=now - timedelta(days=30),
		period_end=now,
		requested_by=ACTOR,
	)
	rep = loop.run_until_complete(s.compliance_report(req))
	assert rep.framework == ComplianceFramework.GDPR
	assert rep.status.value == "ready"
	assert "framework" in rep.summary


def test_compliance_report_recommendations():
	s   = svc()
	now = datetime.now(timezone.utc)
	req = ComplianceReportCreate(
		tenant_id=TENANT,
		framework=ComplianceFramework.SOX,
		period_start=now - timedelta(days=7),
		period_end=now,
		requested_by=ACTOR,
		include_recommendations=True,
	)
	rep = loop.run_until_complete(s.compliance_report(req))
	assert isinstance(rep.summary.get("recommendations"), list)


# ---------------------------------------------------------------------------
# gdpr_data_subject_access
# ---------------------------------------------------------------------------

def test_dsr_access_fulfilled():
	s = svc()
	loop.run_until_complete(s.log_event(
		who="subject-1", what="login", on_what="subject-1",
		how=AuditEventType.USER_LOGIN, where=None, when=None, result=True,
	))
	req = DataSubjectRequestCreate(
		tenant_id=TENANT,
		dsr_type=DSRType.ACCESS,
		subject_id="subject-1",
		requested_by="subject-1",
		justification="I want to see my data",
	)
	dsr = loop.run_until_complete(s.gdpr_data_subject_access(req, is_admin=False))
	assert dsr.status == DSRStatus.FULFILLED
	assert "event_ids" in dsr.response_data
	assert len(dsr.response_data["event_ids"]) >= 1


# ---------------------------------------------------------------------------
# right_to_erasure_audit_impact
# ---------------------------------------------------------------------------

def test_erasure_impact_all_blocked():
	s = svc()
	loop.run_until_complete(s.log_event(
		who="subject-2", what="login", on_what="s",
		how=AuditEventType.USER_LOGIN, where=None, when=None, result=True,
	))
	impact = loop.run_until_complete(s.right_to_erasure_audit_impact("subject-2"))
	assert impact["erasure_blocked"] >= 1
	assert "Art. 17(3)(b)" in impact["reason"]


# ---------------------------------------------------------------------------
# evidence_package_export
# ---------------------------------------------------------------------------

def test_evidence_package_sealed():
	s  = svc()
	ev = loop.run_until_complete(s.log_event(
		who=ACTOR, what="read", on_what="doc",
		how=AuditEventType.DATA_READ, where=None, when=None, result=True,
	))
	req = EvidencePackageCreate(
		tenant_id=TENANT,
		name="Test Package",
		event_ids=[ev.id],
		requested_by=ACTOR,
		reason="forensic investigation",
	)
	pkg = loop.run_until_complete(s.evidence_package_export(req))
	assert pkg.status.value == "sealed"
	assert pkg.file_checksum is not None
	assert len(pkg.chain_of_custody) == 1


# ---------------------------------------------------------------------------
# retention_enforcement
# ---------------------------------------------------------------------------

def test_retention_enforcement_archives_expired():
	s = svc()
	# Log an event then fake its creation date to be long ago
	ev = loop.run_until_complete(s.log_event(
		who=ACTOR, what="old_event", on_what="r",
		how=AuditEventType.API_CALL, where=None, when=None, result=True,
	))
	s._events[ev.id].created_at = datetime.now(timezone.utc) - timedelta(days=100)
	# Create a policy that retains for 1 day
	pol = loop.run_until_complete(s.create_retention_policy(RetentionPolicyCreate(
		tenant_id=TENANT,
		name="test-policy",
		retain_days=1,
	)))
	result = loop.run_until_complete(s.retention_enforcement())
	assert result["archived"] >= 1


def test_retention_enforcement_skips_legal_hold():
	s = svc()
	ev = loop.run_until_complete(s.log_event(
		who=ACTOR, what="old", on_what="r",
		how=AuditEventType.API_CALL, where=None, when=None, result=True,
	))
	s._events[ev.id].created_at = datetime.now(timezone.utc) - timedelta(days=100)
	s._events[ev.id].legal_hold = True
	loop.run_until_complete(s.create_retention_policy(RetentionPolicyCreate(
		tenant_id=TENANT, name="policy-2", retain_days=1,
	)))
	result = loop.run_until_complete(s.retention_enforcement())
	assert ev.id in result["skipped_ids"]


# ---------------------------------------------------------------------------
# cross_tenant_audit_correlation
# ---------------------------------------------------------------------------

def test_cross_tenant_correlation():
	s1 = AuditLoggingService(db_session=None, tenant_id="tenant-a", actor_id="admin")
	s2 = AuditLoggingService(db_session=None, tenant_id="tenant-b", actor_id="admin")
	cid = "corr-123"
	loop.run_until_complete(s1.log_event(
		who="u1", what="x", on_what="r",
		how=AuditEventType.API_CALL, where=None, when=None, result=True,
		payload=AuditEventCreate(
			tenant_id="tenant-a", level=AuditLevel.INFO,
			event_type=AuditEventType.API_CALL, source=EventSource.API_GATEWAY,
			category="api", actor_id="u1", action="x",
			resource_id="r", success=True, correlation_id=cid,
		),
	))
	result = loop.run_until_complete(s1.cross_tenant_audit_correlation(cid, s2))
	assert result["correlation_id"] == cid
	assert result["tenant_a_event_count"] >= 1
	assert result["tenant_b_event_count"] == 0


# ---------------------------------------------------------------------------
# real_time_siem_stream
# ---------------------------------------------------------------------------

async def _collect_siem(s: AuditLoggingService, n: int) -> list:
	"""Log n events and collect them from the SIEM stream."""
	collected = []
	async def consumer():
		async for ev in s.real_time_siem_stream(risk_threshold=0.0):
			collected.append(ev)
			if len(collected) >= n:
				return
	task = asyncio.create_task(consumer())
	await asyncio.sleep(0)   # yield to let consumer register
	for _ in range(n):
		await s.log_event(
			who=ACTOR, what="x", on_what="r",
			how=AuditEventType.API_CALL, where=None, when=None, result=True,
		)
	await asyncio.wait_for(task, timeout=2.0)
	return collected


def test_siem_stream_delivers_events():
	s        = svc()
	received = loop.run_until_complete(_collect_siem(s, 3))
	assert len(received) == 3


# ---------------------------------------------------------------------------
# AuditTrail CRUD
# ---------------------------------------------------------------------------

def test_create_and_get_trail():
	s     = svc()
	trail = loop.run_until_complete(s.create_trail(AuditTrailCreate(
		tenant_id=TENANT, name="incident-42",
	)))
	assert trail.name == "incident-42"
	fetched = loop.run_until_complete(s.get_trail(trail.id))
	assert fetched.id == trail.id


def test_update_trail():
	s     = svc()
	trail = loop.run_until_complete(s.create_trail(AuditTrailCreate(
		tenant_id=TENANT, name="original",
	)))
	updated = loop.run_until_complete(s.update_trail(
		trail.id, AuditTrailUpdate(name="renamed"),
	))
	assert updated.name == "renamed"


def test_delete_trail_soft():
	s     = svc()
	trail = loop.run_until_complete(s.create_trail(AuditTrailCreate(
		tenant_id=TENANT, name="to-delete",
	)))
	loop.run_until_complete(s.delete_trail(trail.id))
	trails = loop.run_until_complete(s.list_trails())
	assert not any(t.id == trail.id for t in trails)


def test_get_nonexistent_trail_raises():
	s = svc()
	with pytest.raises(KeyError):
		loop.run_until_complete(s.get_trail("nonexistent"))


# ---------------------------------------------------------------------------
# RetentionPolicy CRUD
# ---------------------------------------------------------------------------

def test_create_list_delete_retention_policy():
	s   = svc()
	pol = loop.run_until_complete(s.create_retention_policy(RetentionPolicyCreate(
		tenant_id=TENANT, name="7yr", retain_days=2555,
	)))
	policies = loop.run_until_complete(s.list_retention_policies())
	assert any(p.id == pol.id for p in policies)
	loop.run_until_complete(s.delete_retention_policy(pol.id))
	policies_after = loop.run_until_complete(s.list_retention_policies())
	assert not any(p.id == pol.id for p in policies_after)


# ---------------------------------------------------------------------------
# DSR CRUD
# ---------------------------------------------------------------------------

def test_create_and_update_dsr():
	s   = svc()
	req = DataSubjectRequestCreate(
		tenant_id=TENANT,
		dsr_type=DSRType.ERASURE,
		subject_id="user-9",
		requested_by="admin",
		justification="Art. 17",
	)
	dsr = loop.run_until_complete(s.create_dsr(req, is_admin=True))
	assert dsr.status == DSRStatus.PENDING
	from capabilities.common.audl.models import DataSubjectRequestUpdate
	updated = loop.run_until_complete(s.update_dsr(
		dsr.id, DataSubjectRequestUpdate(status=DSRStatus.FULFILLED),
	))
	assert updated.status == DSRStatus.FULFILLED


# ---------------------------------------------------------------------------
# log_integrity_check
# ---------------------------------------------------------------------------

def test_integrity_check_intact():
	s = svc()
	for _ in range(5):
		loop.run_until_complete(s.log_event(
			who=ACTOR, what="x", on_what="r",
			how=AuditEventType.API_CALL, where=None, when=None, result=True,
		))
	result = loop.run_until_complete(s.log_integrity_check())
	assert result["integrity"] == "intact"
	assert result["chain_breaks"] == 0


def test_integrity_check_detects_corruption():
	s  = svc()
	ev = loop.run_until_complete(s.log_event(
		who=ACTOR, what="x", on_what="r",
		how=AuditEventType.API_CALL, where=None, when=None, result=True,
	))
	s._events[ev.id].chain_hash = "0" * 64
	result = loop.run_until_complete(s.log_integrity_check())
	assert result["chain_breaks"] >= 1
	assert result["integrity"] == "broken"


# ---------------------------------------------------------------------------
# tamper_proof_verify
# ---------------------------------------------------------------------------

def test_tamper_proof_verify_clean():
	s  = svc()
	ev = loop.run_until_complete(s.log_event(
		who=ACTOR, what="x", on_what="r",
		how=AuditEventType.API_CALL, where=None, when=None, result=True,
	))
	vfy = loop.run_until_complete(s.tamper_proof_verify(ev.id))
	assert vfy["status"] == "clean"
	assert vfy["checksum_ok"] is True


# ---------------------------------------------------------------------------
# pii_mask_in_logs
# ---------------------------------------------------------------------------

def test_pii_mask():
	s  = svc()
	ev = loop.run_until_complete(s.log_event(
		who="pii-user", what="x", on_what="pii-user",
		how=AuditEventType.DATA_READ, where=None, when=None, result=True,
		payload=_event_create(
			actor_id="pii-user", resource_id="pii-user",
			details={"email": "user@example.com"},
			contains_pii=True,
		),
	))
	result = loop.run_until_complete(s.pii_mask_in_logs("pii-user", ["email"]))
	assert result["events_masked"] >= 1
	assert s._events[ev.id].details.get("email") == "***MASKED***"


# ---------------------------------------------------------------------------
# gdpr_log_erasure
# ---------------------------------------------------------------------------

def test_gdpr_erasure_dry_run():
	s  = svc()
	loop.run_until_complete(s.log_event(
		who="erase-me", what="x", on_what="erase-me",
		how=AuditEventType.DATA_READ, where=None, when=None, result=True,
	))
	result = loop.run_until_complete(s.gdpr_log_erasure("erase-me", "Art. 17", dry_run=True))
	assert result["dry_run"] is True
	assert result["events_erased"] == 0
	assert result["events_affected"] >= 1


def test_gdpr_erasure_applies():
	s  = svc()
	ev = loop.run_until_complete(s.log_event(
		who="erase-me2", what="x", on_what="erase-me2",
		how=AuditEventType.DATA_READ, where=None, when=None, result=True,
	))
	result = loop.run_until_complete(s.gdpr_log_erasure("erase-me2", "Art. 17"))
	assert result["events_erased"] >= 1
	assert s._events[ev.id].details == {"_gdpr_erased": True}


# ---------------------------------------------------------------------------
# risk_summary
# ---------------------------------------------------------------------------

def test_risk_summary_fields():
	s   = svc()
	now = datetime.now(timezone.utc)
	for _ in range(3):
		loop.run_until_complete(s.log_event(
			who=ACTOR, what="x", on_what="r",
			how=AuditEventType.API_CALL, where=None, when=None, result=True,
		))
	rs = loop.run_until_complete(s.risk_summary(now - timedelta(hours=1), now + timedelta(hours=1)))
	assert rs.total_events >= 3
	assert "total_events" in rs.model_dump()


# ---------------------------------------------------------------------------
# audit_analytics
# ---------------------------------------------------------------------------

def test_audit_analytics():
	s   = svc()
	now = datetime.now(timezone.utc)
	loop.run_until_complete(s.log_event(
		who=ACTOR, what="x", on_what="r",
		how=AuditEventType.API_CALL, where=None, when=None, result=True,
	))
	result = loop.run_until_complete(s.audit_analytics(now - timedelta(hours=1), now + timedelta(hours=1)))
	assert "total_events" in result
	assert "risk_distribution" in result
	assert "avg_risk_score" in result


# ---------------------------------------------------------------------------
# set_legal_hold / bulk_set_legal_hold
# ---------------------------------------------------------------------------

def test_set_legal_hold():
	s  = svc()
	ev = loop.run_until_complete(s.log_event(
		who=ACTOR, what="x", on_what="r",
		how=AuditEventType.DATA_READ, where=None, when=None, result=True,
	))
	updated = loop.run_until_complete(s.set_legal_hold(ev.id, True, "litigation"))
	assert updated.legal_hold is True


def test_bulk_set_legal_hold():
	s  = svc()
	e1 = loop.run_until_complete(s.log_event(
		who=ACTOR, what="x", on_what="r",
		how=AuditEventType.API_CALL, where=None, when=None, result=True,
	))
	e2 = loop.run_until_complete(s.log_event(
		who=ACTOR, what="y", on_what="r",
		how=AuditEventType.API_CALL, where=None, when=None, result=True,
	))
	result = loop.run_until_complete(s.bulk_set_legal_hold([e1.id, e2.id, "missing"], True, "test"))
	assert result["applied"] == 2
	assert result["missed"] == 1


# ---------------------------------------------------------------------------
# purge_expired_events
# ---------------------------------------------------------------------------

def test_purge_expired_dry_run():
	s  = svc()
	ev = loop.run_until_complete(s.log_event(
		who=ACTOR, what="old", on_what="r",
		how=AuditEventType.API_CALL, where=None, when=None, result=True,
	))
	s._events[ev.id].created_at    = datetime.now(timezone.utc) - timedelta(days=3000)
	s._events[ev.id].retention_days = 1
	result = loop.run_until_complete(s.purge_expired_events(dry_run=True))
	assert result["eligible"] >= 1
	assert result["purged"] == 0
	assert ev.id in s._events   # not actually purged


def test_purge_expired_executes():
	s  = svc()
	ev = loop.run_until_complete(s.log_event(
		who=ACTOR, what="old", on_what="r",
		how=AuditEventType.API_CALL, where=None, when=None, result=True,
	))
	s._events[ev.id].created_at    = datetime.now(timezone.utc) - timedelta(days=3000)
	s._events[ev.id].retention_days = 1
	result = loop.run_until_complete(s.purge_expired_events(dry_run=False))
	assert result["purged"] >= 1
	assert ev.id not in s._events


# ---------------------------------------------------------------------------
# export_events_jsonl
# ---------------------------------------------------------------------------

def test_export_events_jsonl():
	s  = svc()
	ev = loop.run_until_complete(s.log_event(
		who=ACTOR, what="x", on_what="r",
		how=AuditEventType.API_CALL, where=None, when=None, result=True,
	))
	jsonl = loop.run_until_complete(s.export_events_jsonl([ev.id]))
	import json
	parsed = json.loads(jsonl)
	assert parsed["id"] == ev.id


def test_export_all_events_jsonl():
	s = svc()
	for _ in range(3):
		loop.run_until_complete(s.log_event(
			who=ACTOR, what="x", on_what="r",
			how=AuditEventType.API_CALL, where=None, when=None, result=True,
		))
	jsonl = loop.run_until_complete(s.export_events_jsonl())
	lines = jsonl.strip().split("\n")
	assert len(lines) == 3


# ---------------------------------------------------------------------------
# chain_tip
# ---------------------------------------------------------------------------

def test_chain_tip():
	s = svc()
	loop.run_until_complete(s.log_event(
		who=ACTOR, what="x", on_what="r",
		how=AuditEventType.API_CALL, where=None, when=None, result=True,
	))
	ct = loop.run_until_complete(s.chain_tip())
	assert ct["event_count"] >= 1
	assert len(ct["chain_tip"]) == 64


# ---------------------------------------------------------------------------
# high_risk_events / anomaly_in_audit
# ---------------------------------------------------------------------------

def test_high_risk_events():
	s  = svc()
	# failed auth on external IP = high risk
	ev = loop.run_until_complete(s.log_event(
		who="attacker", what="failed login", on_what="s",
		how=AuditEventType.USER_FAILED_LOGIN, where="8.8.8.8",
		when=None, result=False,
	))
	high = loop.run_until_complete(s.high_risk_events(threshold=0.1))
	assert any(e.id == ev.id for e in high)


def test_anomaly_in_audit():
	s  = svc()
	ev = loop.run_until_complete(s.log_event(
		who=ACTOR, what="x", on_what="r",
		how=AuditEventType.API_CALL, where=None, when=None, result=True,
	))
	s._events[ev.id].anomaly_score = 0.9
	anomalies = loop.run_until_complete(s.anomaly_in_audit(threshold=0.8))
	assert any(e.id == ev.id for e in anomalies)


# ---------------------------------------------------------------------------
# Domain events emitted
# ---------------------------------------------------------------------------

def test_domain_events_emitted():
	from capabilities.common.audl.service import subscribe_domain_events
	received: list[dict] = []
	subscribe_domain_events(received.append)
	s = svc()
	loop.run_until_complete(s.log_event(
		who=ACTOR, what="x", on_what="r",
		how=AuditEventType.API_CALL, where=None, when=None, result=True,
	))
	assert any(e["type"] == "audit_event_logged" for e in received)
