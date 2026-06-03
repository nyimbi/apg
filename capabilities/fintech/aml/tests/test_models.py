"""Unit tests for AML Pydantic v2 models."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))

from models import (  # type: ignore
	AMLAlertCreate,
	AMLAlertUpdate,
	AMLCaseCreate,
	AlertSeverity,
	AlertStatus,
	AlertType,
	CTRCreate,
	CaseStatus,
	CaseType,
	InvestigationNoteCreate,
	RegulatoryFilingCreate,
	RiskSegmentCreate,
	RuleCondition,
	RuleType,
	SARCreate,
	TransactionMonitoringRuleCreate,
	WatchlistMatchCreate,
	uuid7str,
)
from datetime import datetime


def test_uuid7str_generates_unique_ids():
	ids = {uuid7str() for _ in range(100)}
	assert len(ids) == 100
	for i in ids:
		assert len(i) == 36


def test_alert_create_valid():
	alert = AMLAlertCreate(
		tenant_id="t1",
		created_by="user1",
		alert_type=AlertType.LARGE_TRANSACTION,
		severity=AlertSeverity.HIGH,
		subject_reference="customer-1",
		evidence_references=["txn-1"],
		risk_score=75,
	)
	assert alert.alert_type == AlertType.LARGE_TRANSACTION
	assert alert.risk_score == 75


def test_alert_create_rejects_invalid_risk_score():
	with pytest.raises(ValidationError):
		AMLAlertCreate(
			tenant_id="t1",
			created_by="user1",
			alert_type=AlertType.STRUCTURING,
			severity=AlertSeverity.MEDIUM,
			subject_reference="s",
			evidence_references=["e1"],
			risk_score=150,  # > 100
		)


def test_alert_update_partial():
	upd = AMLAlertUpdate(severity=AlertSeverity.CRITICAL)
	assert upd.severity == AlertSeverity.CRITICAL
	assert upd.status is None


def test_case_create_valid():
	case = AMLCaseCreate(
		tenant_id="t1",
		created_by="user1",
		alert_id="alert-1",
		case_type=CaseType.TRANSACTION_MONITORING,
		investigator_id="inv-1",
		subject_reference="customer-1",
		priority=2,
	)
	assert case.priority == 2


def test_case_create_rejects_invalid_priority():
	with pytest.raises(ValidationError):
		AMLCaseCreate(
			tenant_id="t1",
			created_by="user1",
			alert_id="a1",
			case_type=CaseType.SANCTIONS_ALERT,
			investigator_id="inv",
			subject_reference="s",
			priority=10,  # > 5
		)


def test_rule_condition_valid():
	cond = RuleCondition(field="amount", operator="gt", value=10000)
	assert cond.operator == "gt"


def test_monitoring_rule_create():
	rule = TransactionMonitoringRuleCreate(
		tenant_id="t1",
		created_by="user1",
		name="Large TX Rule",
		description="Flags transactions over $10k",
		rule_type=RuleType.THRESHOLD,
		conditions=[RuleCondition(field="amount", operator="gte", value=10000)],
		alert_type=AlertType.LARGE_TRANSACTION,
		severity=AlertSeverity.HIGH,
	)
	assert rule.score_weight == 1.0


def test_sar_create_valid():
	sar = SARCreate(
		tenant_id="t1",
		created_by="user1",
		case_id="case-1",
		subject_reference="cust-1",
		subject_name="John Doe",
		jurisdiction="US",
		filing_institution="First Bank",
		narrative="A" * 60,
		suspicious_activity_start=datetime(2026, 1, 1),
		suspicious_activity_end=datetime(2026, 3, 1),
		total_amount=50_000.0,
		currency="USD",
	)
	assert sar.total_amount == 50_000.0


def test_ctr_create_valid():
	ctr = CTRCreate(
		tenant_id="t1",
		created_by="user1",
		transaction_id="txn-1",
		subject_reference="cust-1",
		subject_name="Jane Doe",
		amount=15_000.0,
		currency="USD",
		transaction_date=datetime(2026, 5, 1),
		transaction_type="cash_deposit",
		jurisdiction="US",
		filing_institution="First Bank",
	)
	assert ctr.amount == 15_000.0


def test_watchlist_match_create():
	wm = WatchlistMatchCreate(
		tenant_id="t1",
		created_by="user1",
		subject_reference="s1",
		subject_name="Test Person",
		list_name="OFAC_SDN",
		list_entry_id="SDN-12345",
		match_score=0.95,
		match_fields=["name"],
	)
	assert wm.match_score == 0.95


def test_watchlist_match_rejects_invalid_score():
	with pytest.raises(ValidationError):
		WatchlistMatchCreate(
			tenant_id="t1",
			created_by="user1",
			subject_reference="s1",
			subject_name="Test",
			list_name="OFAC_SDN",
			list_entry_id="SDN-1",
			match_score=1.5,  # > 1.0
			match_fields=[],
		)


def test_risk_segment_create():
	seg = RiskSegmentCreate(
		tenant_id="t1",
		created_by="user1",
		subject_reference="cust-1",
		segment="high",
		risk_score=72,
		contributing_factors=["sanctions_proximity", "high_velocity"],
	)
	assert seg.risk_score == 72


def test_investigation_note_create():
	note = InvestigationNoteCreate(
		tenant_id="t1",
		created_by="inv1",
		case_id="case-1",
		body="Reviewed transaction history; found 3 structuring events.",
	)
	assert note.is_privileged is False


def test_regulatory_filing_create():
	filing = RegulatoryFilingCreate(
		tenant_id="t1",
		created_by="user1",
		filing_type="SAR",
		jurisdiction="US",
		regulator="FinCEN",
		reference_id="sar-1",
		period_start=datetime(2026, 1, 1),
		period_end=datetime(2026, 3, 31),
		filing_institution="First Bank",
	)
	assert filing.regulator == "FinCEN"
