"""
Tests for domain/rules.py — all assertions, calculations, and predicates.

© 2025 Datacraft  www.datacraft.co.ke
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from capabilities.common.audl.domain.rules import (
	RuleViolation,
	assert_actor_present,
	assert_batch_size,
	assert_checksum_valid,
	assert_dsr_requester_authorised,
	assert_erasure_allowed,
	assert_event_immutable,
	assert_evidence_package_not_sealed,
	assert_no_cross_tenant_access,
	assert_no_legal_hold_deletion,
	assert_retention_not_expired,
	assert_risk_score_range,
	assert_tenant_context,
	calculate_chain_hash,
	calculate_event_checksum,
	calculate_retain_until,
	calculate_risk_score,
	is_external_ip,
	is_off_hours,
)


# ---------------------------------------------------------------------------
# assert_tenant_context
# ---------------------------------------------------------------------------

def test_tenant_context_passes():
	assert_tenant_context({"tenant_id": "acme"})  # no raise


def test_tenant_context_empty_raises():
	with pytest.raises(RuleViolation) as exc:
		assert_tenant_context({"tenant_id": ""})
	assert exc.value.rule_name == "tenant_context_required"


def test_tenant_context_missing_key_raises():
	with pytest.raises(RuleViolation):
		assert_tenant_context({})


# ---------------------------------------------------------------------------
# assert_actor_present
# ---------------------------------------------------------------------------

def test_actor_present_passes():
	assert_actor_present("user-123")


def test_actor_none_raises():
	with pytest.raises(RuleViolation) as exc:
		assert_actor_present(None)
	assert exc.value.rule_name == "actor_required"


def test_actor_whitespace_raises():
	with pytest.raises(RuleViolation):
		assert_actor_present("   ")


# ---------------------------------------------------------------------------
# assert_no_cross_tenant_access
# ---------------------------------------------------------------------------

def test_same_tenant_passes():
	assert_no_cross_tenant_access("t1", "t1")


def test_cross_tenant_raises():
	with pytest.raises(RuleViolation) as exc:
		assert_no_cross_tenant_access("t1", "t2")
	assert exc.value.rule_name == "cross_tenant_access_denied"


# ---------------------------------------------------------------------------
# assert_event_immutable
# ---------------------------------------------------------------------------

def test_immutable_true_raises():
	with pytest.raises(RuleViolation) as exc:
		assert_event_immutable(True)
	assert exc.value.rule_name == "event_immutable"


def test_immutable_false_passes():
	assert_event_immutable(False)  # mutable record — no raise


# ---------------------------------------------------------------------------
# assert_no_legal_hold_deletion
# ---------------------------------------------------------------------------

def test_legal_hold_blocks_deletion():
	with pytest.raises(RuleViolation) as exc:
		assert_no_legal_hold_deletion(True)
	assert exc.value.rule_name == "legal_hold_prevents_deletion"


def test_no_legal_hold_allows_deletion():
	assert_no_legal_hold_deletion(False)


# ---------------------------------------------------------------------------
# assert_checksum_valid
# ---------------------------------------------------------------------------

def test_checksum_valid_passes():
	digest = hashlib.sha256(b"test").hexdigest()
	assert_checksum_valid(digest, digest)  # no raise


def test_checksum_missing_raises():
	with pytest.raises(RuleViolation) as exc:
		assert_checksum_valid(None, "abc")
	assert exc.value.rule_name == "checksum_missing"


def test_checksum_mismatch_raises():
	with pytest.raises(RuleViolation) as exc:
		assert_checksum_valid("aaa" * 20 + "aaaa", "bbb" * 20 + "bbbb")
	assert exc.value.rule_name == "checksum_mismatch"


# ---------------------------------------------------------------------------
# assert_retention_not_expired
# ---------------------------------------------------------------------------

def test_retention_within_window_passes():
	created_at = datetime.now(timezone.utc) - timedelta(days=10)
	assert_retention_not_expired(created_at, retain_days=30)


def test_retention_expired_raises():
	created_at = datetime.now(timezone.utc) - timedelta(days=31)
	with pytest.raises(RuleViolation) as exc:
		assert_retention_not_expired(created_at, retain_days=30)
	assert exc.value.rule_name == "retention_period_expired"


# ---------------------------------------------------------------------------
# assert_dsr_requester_authorised
# ---------------------------------------------------------------------------

def test_dsr_subject_submits_own_request():
	assert_dsr_requester_authorised("user-1", "user-1", is_admin=False)


def test_dsr_admin_submits_for_other():
	assert_dsr_requester_authorised("admin-1", "user-2", is_admin=True)


def test_dsr_non_admin_submits_for_other_raises():
	with pytest.raises(RuleViolation) as exc:
		assert_dsr_requester_authorised("user-1", "user-2", is_admin=False)
	assert exc.value.rule_name == "dsr_requester_not_authorised"


# ---------------------------------------------------------------------------
# assert_erasure_allowed
# ---------------------------------------------------------------------------

def test_erasure_valid_passes():
	assert_erasure_allowed("user-1", "GDPR Art. 17 request")


def test_erasure_missing_subject_raises():
	with pytest.raises(RuleViolation) as exc:
		assert_erasure_allowed("", "some reason")
	assert exc.value.rule_name == "erasure_missing_subject"


def test_erasure_missing_justification_raises():
	with pytest.raises(RuleViolation) as exc:
		assert_erasure_allowed("user-1", "")
	assert exc.value.rule_name == "erasure_missing_justification"


# ---------------------------------------------------------------------------
# assert_evidence_package_not_sealed
# ---------------------------------------------------------------------------

def test_not_sealed_passes():
	assert_evidence_package_not_sealed("assembling")
	assert_evidence_package_not_sealed("ready")


def test_sealed_raises():
	with pytest.raises(RuleViolation) as exc:
		assert_evidence_package_not_sealed("sealed")
	assert exc.value.rule_name == "evidence_package_sealed"


# ---------------------------------------------------------------------------
# assert_batch_size
# ---------------------------------------------------------------------------

def test_batch_size_valid():
	assert_batch_size(1)
	assert_batch_size(10_000)


def test_batch_size_zero_raises():
	with pytest.raises(RuleViolation) as exc:
		assert_batch_size(0)
	assert exc.value.rule_name == "batch_empty"


def test_batch_size_too_large_raises():
	with pytest.raises(RuleViolation) as exc:
		assert_batch_size(10_001)
	assert exc.value.rule_name == "batch_too_large"


# ---------------------------------------------------------------------------
# assert_risk_score_range
# ---------------------------------------------------------------------------

def test_risk_score_valid():
	assert_risk_score_range(0.0)
	assert_risk_score_range(0.5)
	assert_risk_score_range(1.0)


def test_risk_score_negative_raises():
	with pytest.raises(RuleViolation) as exc:
		assert_risk_score_range(-0.01)
	assert exc.value.rule_name == "risk_score_out_of_range"


def test_risk_score_over_one_raises():
	with pytest.raises(RuleViolation) as exc:
		assert_risk_score_range(1.001)
	assert exc.value.rule_name == "risk_score_out_of_range"


# ---------------------------------------------------------------------------
# calculate_event_checksum
# ---------------------------------------------------------------------------

def test_checksum_is_deterministic():
	ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
	kwargs = dict(
		event_id="ev-1", tenant_id="t1", timestamp=ts,
		event_type="user_login", actor_id="u1",
		action="login", resource_type="session",
		resource_id="s1", success=True,
	)
	assert calculate_event_checksum(**kwargs) == calculate_event_checksum(**kwargs)


def test_checksum_changes_on_mutation():
	ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
	base = dict(
		event_id="ev-1", tenant_id="t1", timestamp=ts,
		event_type="user_login", actor_id="u1",
		action="login", resource_type="session",
		resource_id="s1", success=True,
	)
	cs1 = calculate_event_checksum(**base)
	cs2 = calculate_event_checksum(**{**base, "success": False})
	assert cs1 != cs2


def test_checksum_none_actor_handled():
	ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
	cs = calculate_event_checksum(
		event_id="ev-2", tenant_id="t1", timestamp=ts,
		event_type="system_start", actor_id=None,
		action="start", resource_type=None, resource_id=None, success=True,
	)
	assert len(cs) == 64  # SHA-256 hex digest


# ---------------------------------------------------------------------------
# calculate_chain_hash
# ---------------------------------------------------------------------------

def test_chain_hash_initial():
	tip = "0" * 64
	cs  = "a" * 64
	h   = calculate_chain_hash(tip, cs)
	assert len(h) == 64
	assert h != tip


def test_chain_hash_links_deterministically():
	h1 = calculate_chain_hash("0" * 64, "a" * 64)
	h2 = calculate_chain_hash("0" * 64, "a" * 64)
	assert h1 == h2


def test_chain_hash_differs_for_different_inputs():
	h1 = calculate_chain_hash("0" * 64, "a" * 64)
	h2 = calculate_chain_hash("0" * 64, "b" * 64)
	assert h1 != h2


# ---------------------------------------------------------------------------
# calculate_risk_score
# ---------------------------------------------------------------------------

def test_risk_score_zero_by_default():
	assert calculate_risk_score() == 0.0


def test_risk_score_failed_auth():
	score = calculate_risk_score(is_failed_auth=True)
	assert score == 0.35


def test_risk_score_caps_at_one():
	score = calculate_risk_score(
		is_failed_auth=True, is_privileged_actor=True,
		is_off_hours=True, is_external_ip=True,
		is_sensitive_data=True, is_error_event=True,
		anomaly_hint=1.0,
	)
	assert score == 1.0


def test_risk_score_anomaly_hint_contribution():
	score = calculate_risk_score(anomaly_hint=1.0)
	assert score == 0.20


def test_risk_score_in_range():
	for combo in [
		{}, {"is_failed_auth": True}, {"is_privileged_actor": True, "is_off_hours": True},
	]:
		s = calculate_risk_score(**combo)
		assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# calculate_retain_until
# ---------------------------------------------------------------------------

def test_retain_until_correct():
	created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
	until      = calculate_retain_until(created_at, 365)
	assert until == datetime(2026, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# is_off_hours
# ---------------------------------------------------------------------------

def test_off_hours_none():
	assert is_off_hours(None) is False


def test_off_hours_business_hours():
	ts = datetime(2025, 6, 1, 14, 0, 0, tzinfo=timezone.utc)  # 14:00 UTC
	assert is_off_hours(ts) is False


def test_off_hours_evening():
	ts = datetime(2025, 6, 1, 21, 0, 0, tzinfo=timezone.utc)  # 21:00 UTC
	assert is_off_hours(ts) is True


def test_off_hours_early_morning():
	ts = datetime(2025, 6, 1, 3, 0, 0, tzinfo=timezone.utc)  # 03:00 UTC
	assert is_off_hours(ts) is True


# ---------------------------------------------------------------------------
# is_external_ip
# ---------------------------------------------------------------------------

def test_private_ip_is_not_external():
	assert is_external_ip("10.0.0.1") is False
	assert is_external_ip("192.168.1.100") is False
	assert is_external_ip("172.16.5.5") is False
	assert is_external_ip("127.0.0.1") is False


def test_public_ip_is_external():
	assert is_external_ip("8.8.8.8") is True
	assert is_external_ip("203.0.113.5") is True


def test_none_ip_not_external():
	assert is_external_ip(None) is False


def test_invalid_ip_not_external():
	assert is_external_ip("not-an-ip") is False


# ---------------------------------------------------------------------------
# RuleViolation attributes
# ---------------------------------------------------------------------------

def test_rule_violation_attributes():
	exc = RuleViolation("my_rule", "bad thing happened", "fix_it")
	assert exc.rule_name       == "my_rule"
	assert exc.reason          == "bad thing happened"
	assert exc.required_action == "fix_it"
	assert "my_rule" in str(exc)
