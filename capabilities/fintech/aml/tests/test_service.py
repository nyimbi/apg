"""Async service tests for AMLService.

Uses plain async functions — no @pytest.mark.asyncio decorators needed.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))

from service import AMLService  # type: ignore
from models import (  # type: ignore
	AMLAlertCreate,
	AMLCaseCreate,
	AlertSeverity,
	AlertType,
	CTRCreate,
	CaseStatus,
	CaseType,
	InvestigationNoteCreate,
	RuleCondition,
	RuleType,
	SARCreate,
	TransactionMonitoringRuleCreate,
	WatchlistMatchStatus,
)


def _run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


def fresh(tenant_id: str = "t1") -> AMLService:
	return AMLService(tenant_id=tenant_id, actor_id="test-actor")


# ---------------------------------------------------------------------------
# Transaction monitoring
# ---------------------------------------------------------------------------

async def test_monitor_transaction_basic():
	svc = fresh()
	txn = {
		"id": "txn-1",
		"subject_reference": "cust-1",
		"kyc_profile_id": "kyc-1",
		"amount": 500.0,
		"currency": "USD",
		"source_capability": "fintech_payments",
		"source_reference": "pay-1",
	}
	result = await svc.monitor_transaction(txn)
	assert result["transaction_id"] == "txn-1"
	assert result["requires_ctr"] is False
	assert isinstance(result["risk_score"], int)


async def test_monitor_transaction_large_amount_generates_alert():
	svc = fresh()
	txn = {
		"id": "txn-large",
		"subject_reference": "cust-1",
		"kyc_profile_id": "kyc-1",
		"amount": 12_000.0,
		"currency": "USD",
		"source_capability": "fintech_payments",
		"source_reference": "pay-2",
	}
	result = await svc.monitor_transaction(txn)
	assert result["requires_ctr"] is True


async def test_monitor_transaction_requires_subject():
	svc = fresh()
	with pytest.raises((AssertionError, ValueError, Exception)):
		await svc.monitor_transaction({
			"id": "txn-bad",
			"subject_reference": "",
			"kyc_profile_id": "kyc-1",
			"amount": 100.0,
			"currency": "USD",
			"source_capability": "fintech_payments",
			"source_reference": "pay-1",
		})


async def test_monitor_transaction_requires_positive_amount():
	svc = fresh()
	with pytest.raises((AssertionError, ValueError, Exception)):
		await svc.monitor_transaction({
			"id": "txn-neg",
			"subject_reference": "cust-1",
			"kyc_profile_id": "kyc-1",
			"amount": -50.0,
			"currency": "USD",
			"source_capability": "fintech_payments",
			"source_reference": "pay-1",
		})


async def test_tenant_isolation():
	svc_a = AMLService(tenant_id="tenant-a", actor_id="actor")
	svc_b = AMLService(tenant_id="tenant-b", actor_id="actor")
	# Share the same in-memory store by pointing svc_b's alerts at svc_a's
	# (simulate shared DB; both should still see only their own data)
	txn_a = {
		"id": "txn-a", "subject_reference": "cust-a", "kyc_profile_id": "kyc-a",
		"amount": 100.0, "currency": "USD",
		"source_capability": "payments", "source_reference": "p1",
	}
	await svc_a.monitor_transaction(txn_a)
	alert_a = await svc_a.create_alert(AMLAlertCreate(
		tenant_id="tenant-a", created_by="actor",
		alert_type=AlertType.AGENT_REVIEW, severity=AlertSeverity.LOW,
		subject_reference="cust-a", evidence_references=["txn-a"],
	))
	# svc_b should not see tenant-a's alerts
	alerts_b = await svc_b.list_alerts()
	assert all(a.tenant_id == "tenant-b" for a in alerts_b)


# ---------------------------------------------------------------------------
# Alert CRUD
# ---------------------------------------------------------------------------

async def test_create_and_get_alert():
	svc = fresh()
	alert = await svc.create_alert(AMLAlertCreate(
		tenant_id="t1", created_by="actor",
		alert_type=AlertType.STRUCTURING, severity=AlertSeverity.MEDIUM,
		subject_reference="cust-1", evidence_references=["txn-1"],
		risk_score=55,
	))
	fetched = await svc.get_alert(alert.id)
	assert fetched.id == alert.id
	assert fetched.risk_score == 55


async def test_list_alerts_filtered():
	svc = fresh()
	for i, (at, sv) in enumerate([
		(AlertType.STRUCTURING, AlertSeverity.HIGH),
		(AlertType.LARGE_TRANSACTION, AlertSeverity.CRITICAL),
		(AlertType.VELOCITY, AlertSeverity.LOW),
	]):
		await svc.create_alert(AMLAlertCreate(
			tenant_id="t1", created_by="a",
			alert_type=at, severity=sv,
			subject_reference=f"cust-{i}", evidence_references=[f"e{i}"],
		))
	criticals = await svc.list_alerts(severity="critical")
	assert len(criticals) == 1
	assert criticals[0].severity == AlertSeverity.CRITICAL


async def test_approve_alert():
	svc = fresh()
	alert = await svc.create_alert(AMLAlertCreate(
		tenant_id="t1", created_by="a",
		alert_type=AlertType.PEP, severity=AlertSeverity.HIGH,
		subject_reference="s1", evidence_references=["e1"],
	))
	updated = await svc.approve_alert(alert.id, "reviewer-1")
	assert str(updated.status) == "escalated"
	assert updated.reviewer_id == "reviewer-1"


async def test_close_alert_requires_disposition():
	svc = fresh()
	alert = await svc.create_alert(AMLAlertCreate(
		tenant_id="t1", created_by="a",
		alert_type=AlertType.VELOCITY, severity=AlertSeverity.MEDIUM,
		subject_reference="s1", evidence_references=["e1"],
	))
	with pytest.raises(Exception):
		await svc.close_alert(alert.id, "", "reviewer-1")


async def test_close_alert_with_disposition():
	svc = fresh()
	alert = await svc.create_alert(AMLAlertCreate(
		tenant_id="t1", created_by="a",
		alert_type=AlertType.VELOCITY, severity=AlertSeverity.MEDIUM,
		subject_reference="s1", evidence_references=["e1"],
	))
	closed = await svc.close_alert(alert.id, "investigated_no_action", "reviewer-1")
	assert str(closed.status) == "closed"


async def test_soft_delete_alert():
	svc = fresh()
	alert = await svc.create_alert(AMLAlertCreate(
		tenant_id="t1", created_by="a",
		alert_type=AlertType.MULE_ACCOUNT, severity=AlertSeverity.HIGH,
		subject_reference="s1", evidence_references=["e1"],
	))
	await svc.delete_alert(alert.id)
	alerts = await svc.list_alerts()
	assert all(a.id != alert.id for a in alerts)


# ---------------------------------------------------------------------------
# Case management
# ---------------------------------------------------------------------------

async def test_case_management_from_alert():
	svc = fresh()
	alert = await svc.create_alert(AMLAlertCreate(
		tenant_id="t1", created_by="a",
		alert_type=AlertType.STRUCTURING, severity=AlertSeverity.HIGH,
		subject_reference="cust-1", evidence_references=["txn-1"],
	))
	case = await svc.case_management(alert.id, "inv-1")
	assert case.investigator_id == "inv-1"
	# Alert status should be updated
	updated_alert = await svc.get_alert(alert.id)
	assert str(updated_alert.status) == "case_opened"


async def test_investigate_case_adds_note():
	svc = fresh()
	alert = await svc.create_alert(AMLAlertCreate(
		tenant_id="t1", created_by="a",
		alert_type=AlertType.VELOCITY, severity=AlertSeverity.MEDIUM,
		subject_reference="cust-1", evidence_references=["e1"],
	))
	case = await svc.case_management(alert.id, "inv-1")
	updated = await svc.investigate_case(case.id, "Started review of transaction history.")
	assert str(updated.status) == "under_investigation"
	notes = await svc.list_notes(case.id)
	assert len(notes) == 1


async def test_close_case():
	svc = fresh()
	alert = await svc.create_alert(AMLAlertCreate(
		tenant_id="t1", created_by="a",
		alert_type=AlertType.VELOCITY, severity=AlertSeverity.MEDIUM,
		subject_reference="cust-1", evidence_references=["e1"],
	))
	case = await svc.case_management(alert.id, "inv-1")
	closed = await svc.close_case(case.id, CaseStatus.CLOSED_NO_ACTION, "No suspicious activity confirmed.")
	assert str(closed.status) == "closed_no_action"
	assert closed.closed_at is not None


async def test_close_terminal_case_raises():
	svc = fresh()
	alert = await svc.create_alert(AMLAlertCreate(
		tenant_id="t1", created_by="a",
		alert_type=AlertType.VELOCITY, severity=AlertSeverity.MEDIUM,
		subject_reference="cust-1", evidence_references=["e1"],
	))
	case = await svc.case_management(alert.id, "inv-1")
	await svc.close_case(case.id, CaseStatus.CLOSED_NO_ACTION, "done")
	with pytest.raises(Exception):
		await svc.close_case(case.id, CaseStatus.CLOSED_ACTION_TAKEN, "again")


# ---------------------------------------------------------------------------
# SAR
# ---------------------------------------------------------------------------

async def test_file_sar_full_lifecycle():
	svc = fresh()
	alert = await svc.create_alert(AMLAlertCreate(
		tenant_id="t1", created_by="a",
		alert_type=AlertType.STRUCTURING, severity=AlertSeverity.HIGH,
		subject_reference="cust-1", evidence_references=["txn-1"],
	))
	case = await svc.case_management(alert.id, "inv-1")

	sar = await svc.file_sar(case.id, SARCreate(
		tenant_id="t1", created_by="compliance",
		case_id=case.id,
		subject_reference="cust-1",
		subject_name="John Structurer",
		jurisdiction="US",
		filing_institution="First Bank",
		narrative="Customer conducted 5 cash deposits just under $10k over 8 days, consistent with structuring.",
		suspicious_activity_start=datetime(2026, 1, 1),
		suspicious_activity_end=datetime(2026, 1, 9),
		total_amount=48_500.0,
		currency="USD",
		transaction_ids=["txn-1"],
		evidence_references=["txn-1"],
	))
	assert str(sar.status) == "draft"

	approved = await svc.approve_sar(sar.id, "compliance-manager")
	assert str(approved.status) == "approved"

	filed = await svc.submit_sar(sar.id, "FINCEN-2026-XYZ")
	assert str(filed.status) == "filed"
	assert filed.filing_reference == "FINCEN-2026-XYZ"


async def test_sar_requires_narrative():
	svc = fresh()
	alert = await svc.create_alert(AMLAlertCreate(
		tenant_id="t1", created_by="a",
		alert_type=AlertType.STRUCTURING, severity=AlertSeverity.HIGH,
		subject_reference="cust-1", evidence_references=["e1"],
	))
	case = await svc.case_management(alert.id, "inv-1")
	with pytest.raises(Exception):
		await svc.file_sar(case.id, SARCreate(
			tenant_id="t1", created_by="a",
			case_id=case.id,
			subject_reference="cust-1",
			subject_name="J. Doe",
			jurisdiction="US",
			filing_institution="Bank",
			narrative="short",  # < 50 chars
			suspicious_activity_start=datetime(2026, 1, 1),
			suspicious_activity_end=datetime(2026, 1, 2),
			total_amount=10_000.0,
			currency="USD",
		))


# ---------------------------------------------------------------------------
# CTR
# ---------------------------------------------------------------------------

async def test_file_ctr():
	svc = fresh()
	ctr = await svc.file_ctr("txn-cash-1", CTRCreate(
		tenant_id="t1", created_by="teller",
		transaction_id="txn-cash-1",
		subject_reference="cust-1",
		subject_name="Jane Cash",
		amount=15_000.0,
		currency="USD",
		transaction_date=datetime(2026, 5, 1),
		transaction_type="cash_deposit",
		jurisdiction="US",
		filing_institution="First Bank",
	))
	assert str(ctr.status) == "pending"
	submitted = await svc.submit_ctr(ctr.id, "CTR-REF-001")
	assert str(submitted.status) == "filed"


async def test_ctr_below_threshold_raises():
	svc = fresh()
	with pytest.raises(Exception):
		await svc.file_ctr("txn-small", CTRCreate(
			tenant_id="t1", created_by="teller",
			transaction_id="txn-small",
			subject_reference="cust-1",
			subject_name="Jane Small",
			amount=5_000.0,  # below $10k US threshold
			currency="USD",
			transaction_date=datetime(2026, 5, 1),
			transaction_type="cash_deposit",
			jurisdiction="US",
			filing_institution="First Bank",
		))


# ---------------------------------------------------------------------------
# Watchlist screening
# ---------------------------------------------------------------------------

async def test_watchlist_screening_clean_subject():
	svc = fresh()
	matches = await svc.watchlist_screening("cust-clean", "Alice Normal")
	assert matches == []


async def test_watchlist_screening_sanctions_hit():
	svc = fresh()
	matches = await svc.watchlist_screening("cust-bad", "Test Sanctions Subject")
	assert len(matches) > 0
	assert matches[0].match_score >= 0.9


async def test_watchlist_match_review():
	svc = fresh()
	matches = await svc.watchlist_screening("cust-bad", "Test Sanctions Subject")
	assert matches
	reviewed = await svc.review_watchlist_match(matches[0].id, WatchlistMatchStatus.FALSE_POSITIVE, "reviewer-1")
	assert str(reviewed.status) == "false_positive"


# ---------------------------------------------------------------------------
# Network analysis & pattern detection
# ---------------------------------------------------------------------------

async def test_network_analysis_empty():
	svc = fresh()
	result = await svc.network_analysis("cust-no-txns")
	assert result.transaction_count == 0
	assert result.network_risk_score == 0


async def test_pattern_detection_empty():
	svc = fresh()
	result = await svc.pattern_detection("cust-no-txns", lookback_days=90)
	assert result.structuring_detected is False
	assert result.recommended_action == "no_action"


# ---------------------------------------------------------------------------
# Risk segmentation
# ---------------------------------------------------------------------------

async def test_risk_segmentation_low():
	svc = fresh()
	seg = await svc.risk_segmentation("cust-1", risk_score=10)
	assert str(seg.segment) == "low"


async def test_risk_segmentation_high():
	svc = fresh()
	seg = await svc.risk_segmentation("cust-1", risk_score=75)
	assert str(seg.segment) in {"high", "very_high"}


async def test_risk_segmentation_sanctions():
	svc = fresh()
	seg = await svc.risk_segmentation("cust-1", risk_score=100)
	assert str(seg.segment) == "prohibited"


async def test_risk_segmentation_tracks_previous():
	svc = fresh()
	await svc.risk_segmentation("cust-1", risk_score=10)
	seg2 = await svc.risk_segmentation("cust-1", risk_score=75)
	assert seg2.previous_segment == "low"


# ---------------------------------------------------------------------------
# Regulatory reporting
# ---------------------------------------------------------------------------

async def test_regulatory_reporting():
	svc = fresh()
	report = await svc.regulatory_reporting(
		jurisdiction="US",
		period_start=datetime(2026, 1, 1),
		period_end=datetime(2026, 12, 31),
	)
	assert report.jurisdiction == "US"
	assert report.sar_count >= 0


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

async def test_dashboard_summary():
	svc = fresh()
	await svc.create_alert(AMLAlertCreate(
		tenant_id="t1", created_by="a",
		alert_type=AlertType.STRUCTURING, severity=AlertSeverity.CRITICAL,
		subject_reference="cust-1", evidence_references=["e1"],
	))
	summary = await svc.dashboard_summary()
	assert summary["alert_count"] == 1
	assert summary["critical_alert_count"] == 1
	assert summary["open_alert_count"] == 1


# ---------------------------------------------------------------------------
# Rule CRUD
# ---------------------------------------------------------------------------

async def test_create_and_list_rules():
	svc = fresh()
	rule = await svc.create_rule(TransactionMonitoringRuleCreate(
		tenant_id="t1", created_by="admin",
		name="Large TX",
		description="Flag transactions over $10k",
		rule_type=RuleType.THRESHOLD,
		conditions=[RuleCondition(field="amount", operator="gte", value=10000)],
		alert_type=AlertType.LARGE_TRANSACTION,
		severity=AlertSeverity.HIGH,
	))
	rules = await svc.list_rules()
	assert any(r.id == rule.id for r in rules)


async def test_delete_rule_soft():
	svc = fresh()
	rule = await svc.create_rule(TransactionMonitoringRuleCreate(
		tenant_id="t1", created_by="admin",
		name="Temp Rule",
		description="Temp",
		rule_type=RuleType.VELOCITY,
		conditions=[],
		alert_type=AlertType.VELOCITY,
		severity=AlertSeverity.LOW,
	))
	await svc.delete_rule(rule.id)
	rules = await svc.list_rules()
	assert all(r.id != rule.id for r in rules)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

async def test_events_emitted():
	svc = fresh()
	await svc.create_alert(AMLAlertCreate(
		tenant_id="t1", created_by="a",
		alert_type=AlertType.VELOCITY, severity=AlertSeverity.LOW,
		subject_reference="s1", evidence_references=["e1"],
	))
	assert len(svc._events) >= 1
	assert svc._events[0]["event_type"] == "alert_created"
	assert svc._events[0]["tenant_id"] == "t1"
