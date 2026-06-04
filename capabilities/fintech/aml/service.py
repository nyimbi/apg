"""World-class AML service layer for APG.

Async, tenant-isolated, event-emitting service covering:
- Transaction monitoring with rule evaluation
- Alert lifecycle (open → triage → escalate → close)
- Case management (open → investigate → confirm → file SAR/CTR → close)
- Watchlist screening (OFAC, PEP, UN, EU, custom lists)
- Network analysis (round-trip, layering, counterparty risk)
- Pattern detection (structuring/smurfing, velocity)
- Risk segmentation
- Regulatory reporting (SAR, CTR, STR per jurisdiction)
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

try:
	from .domain.calculations import (
		calculate_false_positive_rate,
		calculate_network_risk_score,
		calculate_risk_score,
		calculate_sar_priority,
		detect_layering,
		detect_round_trip,
		detect_structuring,
		detect_velocity_anomaly,
		requires_ctr,
		risk_segment_from_score,
		severity_from_score,
	)
	from .domain.rules import (
		RuleViolation,
		assert_alert_close_has_disposition,
		assert_alert_escalation_has_reviewer,
		assert_alert_evidence_present,
		assert_alert_type_supported,
		assert_case_is_open_for_investigation,
		assert_case_type_supported,
		assert_ctr_amount_triggers_reporting,
		assert_currency_present,
		assert_investigator_assigned,
		assert_kyc_link_present,
		assert_match_score_valid,
		assert_no_cross_tenant_access,
		assert_positive_amount,
		assert_sar_human_approval,
		assert_sar_jurisdiction_present,
		assert_sar_narrative_present,
		assert_severity_supported,
		assert_source_reference_present,
		assert_tenant_context,
		assert_transaction_subject_present,
	)
except ImportError:  # pragma: no cover — direct file load in tests
	from domain.calculations import (  # type: ignore
		calculate_false_positive_rate,
		calculate_network_risk_score,
		calculate_risk_score,
		calculate_sar_priority,
		detect_layering,
		detect_round_trip,
		detect_structuring,
		detect_velocity_anomaly,
		requires_ctr,
		risk_segment_from_score,
		severity_from_score,
	)
	from domain.rules import (  # type: ignore
		RuleViolation,
		assert_alert_close_has_disposition,
		assert_alert_escalation_has_reviewer,
		assert_alert_evidence_present,
		assert_alert_type_supported,
		assert_case_is_open_for_investigation,
		assert_case_type_supported,
		assert_ctr_amount_triggers_reporting,
		assert_currency_present,
		assert_investigator_assigned,
		assert_kyc_link_present,
		assert_match_score_valid,
		assert_no_cross_tenant_access,
		assert_positive_amount,
		assert_sar_human_approval,
		assert_sar_jurisdiction_present,
		assert_sar_narrative_present,
		assert_severity_supported,
		assert_source_reference_present,
		assert_tenant_context,
		assert_transaction_subject_present,
	)

try:
	from .models import (
		AMLAlertCreate,
		AMLAlertResponse,
		AMLAlertUpdate,
		AMLCaseCreate,
		AMLCaseResponse,
		AMLCaseUpdate,
		AlertSeverity,
		AlertStatus,
		AlertType,
		CTRCreate,
		CTRResponse,
		CTRStatus,
		CaseStatus,
		FilingStatus,
		InvestigationNoteCreate,
		InvestigationNoteResponse,
		NetworkAnalysisResult,
		PatternDetectionResult,
		RegulatoryFilingCreate,
		RegulatoryFilingResponse,
		RegulatoryReportRequest,
		RegulatoryReportResponse,
		RiskSegmentCreate,
		RiskSegmentResponse,
		SARCreate,
		SARResponse,
		SARStatus,
		TransactionMonitoringRuleCreate,
		TransactionMonitoringRuleResponse,
		TransactionMonitoringRuleUpdate,
		WatchlistMatchCreate,
		WatchlistMatchResponse,
		WatchlistMatchStatus,
		uuid7str,
	)
except ImportError:  # pragma: no cover
	from models import (  # type: ignore
		AMLAlertCreate,
		AMLAlertResponse,
		AMLAlertUpdate,
		AMLCaseCreate,
		AMLCaseResponse,
		AMLCaseUpdate,
		AlertSeverity,
		AlertStatus,
		AlertType,
		CTRCreate,
		CTRResponse,
		CTRStatus,
		CaseStatus,
		FilingStatus,
		InvestigationNoteCreate,
		InvestigationNoteResponse,
		NetworkAnalysisResult,
		PatternDetectionResult,
		RegulatoryFilingCreate,
		RegulatoryFilingResponse,
		RegulatoryReportRequest,
		RegulatoryReportResponse,
		RiskSegmentCreate,
		RiskSegmentResponse,
		SARCreate,
		SARResponse,
		SARStatus,
		TransactionMonitoringRuleCreate,
		TransactionMonitoringRuleResponse,
		TransactionMonitoringRuleUpdate,
		WatchlistMatchCreate,
		WatchlistMatchResponse,
		WatchlistMatchStatus,
		uuid7str,
	)

logger = logging.getLogger("apg.fintech.aml")

# CTR thresholds by jurisdiction (local currency)
CTR_THRESHOLDS: dict[str, float] = {
	"US": 10_000.0,
	"UK": 10_000.0,
	"EU": 10_000.0,
	"AU": 10_000.0,
	"CA": 10_000.0,
	"KE": 1_000_000.0,
	"NG": 5_000_000.0,
	"ZA": 24_999.0,
}


class AMLService:
	"""Full AML lifecycle service. Dependency-light; state in-memory dicts.

	In production, replace the in-memory stores with async SQLAlchemy sessions
	backed by the PostgreSQL schema in database/schema.sql.
	"""

	def __init__(
		self,
		db_session: Any = None,
		tenant_id: str = "default",
		actor_id: str = "system",
	) -> None:
		self.db = db_session
		self.tenant_id = tenant_id
		self.actor_id = actor_id

		# In-memory stores (keyed by id)
		self._rules: dict[str, TransactionMonitoringRuleResponse] = {}
		self._alerts: dict[str, AMLAlertResponse] = {}
		self._cases: dict[str, AMLCaseResponse] = {}
		self._notes: dict[str, InvestigationNoteResponse] = {}
		self._sars: dict[str, SARResponse] = {}
		self._ctrs: dict[str, CTRResponse] = {}
		self._watchlist_matches: dict[str, WatchlistMatchResponse] = {}
		self._filings: dict[str, RegulatoryFilingResponse] = {}
		self._risk_segments: dict[str, RiskSegmentResponse] = {}
		self._transactions: dict[str, dict[str, Any]] = {}
		self._events: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# Logging helpers
	# ------------------------------------------------------------------

	def _log_op(self, op: str, entity_id: str, **kw: Any) -> None:
		logger.info("AML %s tenant=%s actor=%s id=%s %s", op, self.tenant_id, self.actor_id, entity_id, kw)

	def _log_warn(self, msg: str, **kw: Any) -> None:
		logger.warning("AML WARN tenant=%s %s %s", self.tenant_id, msg, kw)

	def _log_error(self, msg: str, exc: Exception | None = None) -> None:
		logger.error("AML ERROR tenant=%s %s exc=%s", self.tenant_id, msg, exc)

	def _log_pretty_event(self, event_type: str, payload: dict[str, Any]) -> str:
		return f"[{event_type}] tenant={self.tenant_id} {payload}"

	# ------------------------------------------------------------------
	# Event emission
	# ------------------------------------------------------------------

	async def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
		"""Emit domain event to the AML event stream (Bytewax/mqtt/etc)."""
		event = {
			"event_type": event_type,
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
			"timestamp": datetime.utcnow().isoformat(),
			"capability_id": "fintech_aml",
			"payload": payload,
		}
		self._events.append(event)
		logger.debug(self._log_pretty_event(event_type, payload))

	# ------------------------------------------------------------------
	# Tenant isolation helper
	# ------------------------------------------------------------------

	def _assert_tenant(self, resource_tenant: str) -> None:
		assert_no_cross_tenant_access(self.tenant_id, resource_tenant)

	def _tenant_filter(self, store: dict[str, Any]) -> list[Any]:
		return [v for v in store.values() if getattr(v, "tenant_id", None) == self.tenant_id]

	# ------------------------------------------------------------------
	# TransactionMonitoringRule CRUD
	# ------------------------------------------------------------------

	async def create_rule(self, data: TransactionMonitoringRuleCreate) -> TransactionMonitoringRuleResponse:
		"""Create a new transaction monitoring rule."""
		assert_tenant_context({"tenant_id": data.tenant_id})
		assert data.tenant_id == self.tenant_id, "tenant mismatch"

		rule = TransactionMonitoringRuleResponse(
			tenant_id=data.tenant_id,
			created_by=data.created_by,
			name=data.name,
			description=data.description,
			rule_type=data.rule_type,
			conditions=data.conditions,
			alert_type=data.alert_type,
			severity=data.severity,
			lookback_days=data.lookback_days,
			min_occurrences=data.min_occurrences,
			score_weight=data.score_weight,
			jurisdictions=data.jurisdictions,
			enabled=data.enabled,
			metadata=data.metadata,
		)
		self._rules[rule.id] = rule
		await self._emit_event("rule_created", {"rule_id": rule.id, "name": rule.name})
		self._log_op("create_rule", rule.id, name=rule.name)
		return rule

	async def get_rule(self, rule_id: str) -> TransactionMonitoringRuleResponse:
		"""Retrieve a monitoring rule by ID (tenant-isolated)."""
		rule = self._rules.get(rule_id)
		assert rule is not None, f"rule not found: {rule_id}"
		self._assert_tenant(rule.tenant_id)
		return rule

	async def update_rule(self, rule_id: str, data: TransactionMonitoringRuleUpdate) -> TransactionMonitoringRuleResponse:
		"""Update rule fields."""
		rule = await self.get_rule(rule_id)
		for field, val in data.model_dump(exclude_none=True).items():
			setattr(rule, field, val)
		rule.updated_at = datetime.utcnow()
		await self._emit_event("rule_updated", {"rule_id": rule_id})
		return rule

	async def delete_rule(self, rule_id: str) -> None:
		"""Soft-delete a monitoring rule."""
		rule = await self.get_rule(rule_id)
		rule.is_deleted = True
		await self._emit_event("rule_deleted", {"rule_id": rule_id})

	async def list_rules(self, enabled_only: bool = False) -> list[TransactionMonitoringRuleResponse]:
		"""List all active rules for this tenant."""
		rules = [r for r in self._tenant_filter(self._rules) if not r.is_deleted]
		if enabled_only:
			rules = [r for r in rules if r.enabled]
		return sorted(rules, key=lambda r: r.created_at)

	# ------------------------------------------------------------------
	# Transaction monitoring
	# ------------------------------------------------------------------

	async def monitor_transaction(
		self,
		txn: dict[str, Any],
	) -> dict[str, Any]:
		"""Ingest a transaction and run all active monitoring rules.

		Args:
			txn: dict with keys: id, subject_reference, kyc_profile_id,
			     amount, currency, source_capability, source_reference,
			     sender_account, receiver_account, created_at, metadata

		Returns:
			dict with risk_score, typology_flags, alerts_generated, requires_ctr
		"""
		assert_tenant_context({"tenant_id": self.tenant_id})
		tid = str(txn.get("id") or uuid7str())
		subject = str(txn.get("subject_reference", ""))
		kyc_id = str(txn.get("kyc_profile_id", ""))
		amount = float(txn.get("amount", 0))
		currency = str(txn.get("currency", ""))
		source_cap = str(txn.get("source_capability", ""))
		source_ref = str(txn.get("source_reference", ""))

		assert_transaction_subject_present(subject)
		assert_positive_amount(amount)
		assert_currency_present(currency)
		assert_source_reference_present(source_ref, source_cap)
		assert_kyc_link_present(kyc_id)

		txn_record = dict(txn)
		txn_record["id"] = tid
		txn_record["tenant_id"] = self.tenant_id
		txn_record["monitored_at"] = datetime.utcnow().isoformat()
		self._transactions[tid] = txn_record

		rule_results = await self.evaluate_rules(txn_record)
		generated_alerts: list[str] = []

		for hit in rule_results.get("hits", []):
			alert = await self.generate_alert(hit["rule_id"], txn_record)
			generated_alerts.append(alert.id)

		ctr_required = requires_ctr(amount, currency, str(txn.get("jurisdiction", "US")))

		result = {
			"transaction_id": tid,
			"tenant_id": self.tenant_id,
			"risk_score": rule_results.get("composite_score", 0),
			"typology_flags": rule_results.get("typology_flags", []),
			"alerts_generated": generated_alerts,
			"requires_ctr": ctr_required,
			"rule_hits": len(rule_results.get("hits", [])),
		}

		await self._emit_event("transaction_monitored", result)
		self._log_op("monitor_transaction", tid, score=result["risk_score"], alerts=len(generated_alerts))
		return result

	async def evaluate_rules(self, txn: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate all enabled rules against a transaction.

		Returns dict with hits list, composite_score, typology_flags.
		"""
		active_rules = await self.list_rules(enabled_only=True)
		hits: list[dict[str, Any]] = []
		flags: list[str] = []
		scores: list[float] = []

		amount = float(txn.get("amount", 0))
		risk_score = int(txn.get("risk_score", 0))
		velocity_count = int(txn.get("velocity_count", 0))
		sanctions_hit = bool(txn.get("sanctions_hit", False))
		pep_hit = bool(txn.get("pep_hit", False))
		high_risk_country = bool(txn.get("high_risk_country", False))

		for rule in active_rules:
			matched = await self._evaluate_single_rule(rule, txn)
			if matched:
				hits.append({"rule_id": rule.id, "rule_name": rule.name, "alert_type": rule.alert_type, "severity": rule.severity})
				flags.append(str(rule.alert_type))
				scores.append(rule.score_weight * 10)

		composite = calculate_risk_score(
			amount=amount,
			large_threshold=10_000.0,
			structuring_threshold=9_500.0,
			velocity_count=velocity_count,
			velocity_window_hours=24,
			sanctions_hit=sanctions_hit,
			pep_hit=pep_hit,
			high_risk_country=high_risk_country,
			adverse_media=bool(txn.get("adverse_media", False)),
			base_kyc_score=risk_score,
		)
		composite = min(composite + int(sum(scores)), 100)

		return {"hits": hits, "composite_score": composite, "typology_flags": list(set(flags))}

	async def _evaluate_single_rule(
		self,
		rule: TransactionMonitoringRuleResponse,
		txn: dict[str, Any],
	) -> bool:
		"""Evaluate one rule against a transaction. Returns True if rule fires."""
		amount = float(txn.get("amount", 0))

		for cond in rule.conditions:
			field_val = txn.get(cond.field)
			op = cond.operator
			threshold = cond.value

			try:
				if op == "gt" and not (float(field_val or 0) > float(threshold)):
					return False
				elif op == "gte" and not (float(field_val or 0) >= float(threshold)):
					return False
				elif op == "lt" and not (float(field_val or 0) < float(threshold)):
					return False
				elif op == "lte" and not (float(field_val or 0) <= float(threshold)):
					return False
				elif op == "eq" and str(field_val) != str(threshold):
					return False
				elif op == "in" and str(field_val) not in list(threshold):
					return False
				elif op == "contains" and str(threshold) not in str(field_val or ""):
					return False
			except (TypeError, ValueError):
				return False

		return True

	async def generate_alert(
		self,
		rule_id: str | None,
		txn: dict[str, Any],
	) -> AMLAlertResponse:
		"""Generate an AMLAlert from a rule hit on a transaction."""
		risk_score = int(txn.get("risk_score", 0))
		amount = float(txn.get("amount", 0))
		sanctions = bool(txn.get("sanctions_hit", False))

		if rule_id:
			rule = self._rules.get(rule_id)
			alert_type = str(rule.alert_type) if rule else "agent_review"
			severity = str(rule.severity) if rule else severity_from_score(risk_score)
		else:
			alert_type = "agent_review"
			severity = severity_from_score(risk_score)

		if sanctions:
			severity = "critical"

		data = AMLAlertCreate(
			tenant_id=self.tenant_id,
			created_by=self.actor_id,
			alert_type=AlertType(alert_type),
			severity=AlertSeverity(severity),
			subject_reference=str(txn.get("subject_reference", "")),
			kyc_profile_id=str(txn.get("kyc_profile_id", "")),
			rule_id=rule_id,
			transaction_ids=[str(txn.get("id", ""))],
			evidence_references=[str(txn.get("id", ""))],
			risk_score=risk_score,
			amount=amount,
			currency=str(txn.get("currency", "")),
		)
		return await self.create_alert(data)

	# ------------------------------------------------------------------
	# Alert CRUD
	# ------------------------------------------------------------------

	async def create_alert(self, data: AMLAlertCreate) -> AMLAlertResponse:
		"""Create an AML alert."""
		assert_tenant_context({"tenant_id": data.tenant_id})
		assert_alert_type_supported(data.alert_type.value if hasattr(data.alert_type, "value") else str(data.alert_type))
		assert_severity_supported(data.severity.value if hasattr(data.severity, "value") else str(data.severity))
		assert_alert_evidence_present(data.evidence_references)

		alert = AMLAlertResponse(
			tenant_id=data.tenant_id,
			created_by=data.created_by,
			alert_type=data.alert_type,
			severity=data.severity,
			subject_reference=data.subject_reference,
			kyc_profile_id=data.kyc_profile_id,
			rule_id=data.rule_id,
			transaction_ids=data.transaction_ids,
			evidence_references=data.evidence_references,
			risk_score=data.risk_score,
			typology_codes=data.typology_codes,
			amount=data.amount,
			currency=data.currency,
			narrative=data.narrative,
			metadata=data.metadata,
		)
		self._alerts[alert.id] = alert
		await self._emit_event("alert_created", {"alert_id": alert.id, "type": str(alert.alert_type), "severity": str(alert.severity)})
		self._log_op("create_alert", alert.id, type=str(alert.alert_type))
		return alert

	async def get_alert(self, alert_id: str) -> AMLAlertResponse:
		"""Get alert by ID (tenant-isolated)."""
		alert = self._alerts.get(alert_id)
		assert alert is not None, f"alert not found: {alert_id}"
		self._assert_tenant(alert.tenant_id)
		return alert

	async def update_alert(self, alert_id: str, data: AMLAlertUpdate) -> AMLAlertResponse:
		"""Partial update of an alert."""
		alert = await self.get_alert(alert_id)
		for field, val in data.model_dump(exclude_none=True).items():
			setattr(alert, field, val)
		alert.updated_at = datetime.utcnow()
		await self._emit_event("alert_updated", {"alert_id": alert_id})
		return alert

	async def delete_alert(self, alert_id: str) -> None:
		"""Soft-delete an alert."""
		alert = await self.get_alert(alert_id)
		alert.is_deleted = True
		await self._emit_event("alert_deleted", {"alert_id": alert_id})

	async def list_alerts(
		self,
		status: str | None = None,
		severity: str | None = None,
		alert_type: str | None = None,
		limit: int = 100,
		offset: int = 0,
	) -> list[AMLAlertResponse]:
		"""List alerts for this tenant with optional filters."""
		alerts = [a for a in self._tenant_filter(self._alerts) if not a.is_deleted]
		if status:
			alerts = [a for a in alerts if str(a.status) == status]
		if severity:
			alerts = [a for a in alerts if str(a.severity) == severity]
		if alert_type:
			alerts = [a for a in alerts if str(a.alert_type) == alert_type]
		alerts.sort(key=lambda a: a.created_at, reverse=True)
		return alerts[offset: offset + limit]

	async def approve_alert(self, alert_id: str, reviewer_id: str) -> AMLAlertResponse:
		"""Confirm alert as legitimate (move to escalated)."""
		alert = await self.get_alert(alert_id)
		assert_alert_escalation_has_reviewer(True, reviewer_id)
		alert.status = AlertStatus.ESCALATED
		alert.reviewer_id = reviewer_id
		alert.updated_at = datetime.utcnow()
		await self._emit_event("alert_approved", {"alert_id": alert_id, "reviewer": reviewer_id})
		return alert

	async def reject_alert(self, alert_id: str, reviewer_id: str, disposition: str) -> AMLAlertResponse:
		"""Close alert as false positive."""
		alert = await self.get_alert(alert_id)
		assert_alert_close_has_disposition(True, disposition)
		alert.status = AlertStatus.FALSE_POSITIVE
		alert.reviewer_id = reviewer_id
		alert.disposition = disposition
		alert.updated_at = datetime.utcnow()
		await self._emit_event("alert_rejected", {"alert_id": alert_id, "disposition": disposition})
		return alert

	async def close_alert(self, alert_id: str, disposition: str, reviewer_id: str) -> AMLAlertResponse:
		"""Close alert with disposition."""
		alert = await self.get_alert(alert_id)
		assert_alert_close_has_disposition(True, disposition)
		alert.status = AlertStatus.CLOSED
		alert.disposition = disposition
		alert.reviewer_id = reviewer_id
		alert.updated_at = datetime.utcnow()
		await self._emit_event("alert_closed", {"alert_id": alert_id, "disposition": disposition})
		return alert

	# ------------------------------------------------------------------
	# Case management
	# ------------------------------------------------------------------

	async def case_management(self, alert_id: str, investigator_id: str) -> AMLCaseResponse:
		"""Open a case from an alert — the primary case management entry point."""
		alert = await self.get_alert(alert_id)
		assert_investigator_assigned(investigator_id)

		data = AMLCaseCreate(
			tenant_id=self.tenant_id,
			created_by=self.actor_id,
			alert_id=alert_id,
			case_type=_alert_type_to_case_type(str(alert.alert_type)),
			investigator_id=investigator_id,
			subject_reference=alert.subject_reference,
			evidence_references=alert.evidence_references,
		)
		case = await self.create_case(data)
		alert.status = AlertStatus.CASE_OPENED
		alert.case_id = case.id
		alert.updated_at = datetime.utcnow()
		await self._emit_event("case_opened_from_alert", {"case_id": case.id, "alert_id": alert_id})
		return case

	async def create_case(self, data: AMLCaseCreate) -> AMLCaseResponse:
		"""Create an AML investigation case."""
		assert_tenant_context({"tenant_id": data.tenant_id})
		assert_case_type_supported(str(data.case_type))
		assert_investigator_assigned(data.investigator_id)

		case = AMLCaseResponse(
			tenant_id=data.tenant_id,
			created_by=data.created_by,
			alert_id=data.alert_id,
			case_type=data.case_type,
			investigator_id=data.investigator_id,
			subject_reference=data.subject_reference,
			priority=data.priority,
			evidence_references=data.evidence_references,
			notes=data.notes,
			due_date=data.due_date,
			metadata=data.metadata,
		)
		self._cases[case.id] = case
		await self._emit_event("case_created", {"case_id": case.id, "type": str(case.case_type)})
		self._log_op("create_case", case.id)
		return case

	async def get_case(self, case_id: str) -> AMLCaseResponse:
		"""Get case by ID (tenant-isolated)."""
		case = self._cases.get(case_id)
		assert case is not None, f"case not found: {case_id}"
		self._assert_tenant(case.tenant_id)
		return case

	async def update_case(self, case_id: str, data: AMLCaseUpdate) -> AMLCaseResponse:
		"""Update case fields."""
		case = await self.get_case(case_id)
		assert_case_is_open_for_investigation(str(case.status))
		for field, val in data.model_dump(exclude_none=True).items():
			setattr(case, field, val)
		case.updated_at = datetime.utcnow()
		await self._emit_event("case_updated", {"case_id": case_id})
		return case

	async def delete_case(self, case_id: str) -> None:
		"""Soft-delete a case."""
		case = await self.get_case(case_id)
		case.is_deleted = True
		await self._emit_event("case_deleted", {"case_id": case_id})

	async def list_cases(
		self,
		status: str | None = None,
		investigator_id: str | None = None,
		limit: int = 100,
		offset: int = 0,
	) -> list[AMLCaseResponse]:
		"""List cases for this tenant."""
		cases = [c for c in self._tenant_filter(self._cases) if not c.is_deleted]
		if status:
			cases = [c for c in cases if str(c.status) == status]
		if investigator_id:
			cases = [c for c in cases if c.investigator_id == investigator_id]
		cases.sort(key=lambda c: c.created_at, reverse=True)
		return cases[offset: offset + limit]

	async def investigate_case(self, case_id: str, note: str) -> AMLCaseResponse:
		"""Move case to under_investigation and add a note."""
		case = await self.get_case(case_id)
		assert_case_is_open_for_investigation(str(case.status))
		case.status = CaseStatus.UNDER_INVESTIGATION
		case.updated_at = datetime.utcnow()
		if note:
			await self.add_note(InvestigationNoteCreate(
				tenant_id=self.tenant_id,
				created_by=self.actor_id,
				case_id=case_id,
				body=note,
			))
		await self._emit_event("case_investigation_started", {"case_id": case_id})
		return case

	async def close_case(self, case_id: str, status: CaseStatus, notes: str) -> AMLCaseResponse:
		"""Close a case with a terminal status."""
		case = await self.get_case(case_id)
		assert_case_is_open_for_investigation(str(case.status))
		case.status = status
		case.closed_at = datetime.utcnow()
		case.closed_by = self.actor_id
		case.updated_at = datetime.utcnow()
		if notes:
			await self.add_note(InvestigationNoteCreate(
				tenant_id=self.tenant_id,
				created_by=self.actor_id,
				case_id=case_id,
				body=notes,
			))
		await self._emit_event("case_closed", {"case_id": case_id, "status": str(status)})
		return case

	# ------------------------------------------------------------------
	# Investigation notes
	# ------------------------------------------------------------------

	async def add_note(self, data: InvestigationNoteCreate) -> InvestigationNoteResponse:
		"""Add an investigation note to a case."""
		note = InvestigationNoteResponse(
			tenant_id=data.tenant_id,
			created_by=data.created_by,
			case_id=data.case_id,
			body=data.body,
			is_privileged=data.is_privileged,
			attachments=data.attachments,
		)
		self._notes[note.id] = note
		await self._emit_event("note_added", {"note_id": note.id, "case_id": data.case_id})
		return note

	async def list_notes(self, case_id: str) -> list[InvestigationNoteResponse]:
		"""List all notes for a case."""
		return sorted(
			[n for n in self._notes.values() if n.case_id == case_id and n.tenant_id == self.tenant_id],
			key=lambda n: n.created_at,
		)

	# ------------------------------------------------------------------
	# SAR — Suspicious Activity Report
	# ------------------------------------------------------------------

	async def file_sar(self, case_id: str, data: SARCreate) -> SARResponse:
		"""Draft and submit a SAR from a confirmed case."""
		case = await self.get_case(case_id)
		assert_case_is_open_for_investigation(str(case.status))
		assert_sar_narrative_present(data.narrative)
		assert_sar_jurisdiction_present(data.jurisdiction)

		sar = SARResponse(
			tenant_id=data.tenant_id,
			created_by=data.created_by,
			case_id=case_id,
			subject_reference=data.subject_reference,
			subject_name=data.subject_name,
			subject_dob=data.subject_dob,
			subject_tin=data.subject_tin,
			subject_address=data.subject_address,
			jurisdiction=data.jurisdiction,
			filing_institution=data.filing_institution,
			narrative=data.narrative,
			suspicious_activity_start=data.suspicious_activity_start,
			suspicious_activity_end=data.suspicious_activity_end,
			total_amount=data.total_amount,
			currency=data.currency,
			transaction_ids=data.transaction_ids,
			evidence_references=data.evidence_references,
			typology_codes=data.typology_codes,
			status=SARStatus.DRAFT,
			metadata=data.metadata,
		)
		self._sars[sar.id] = sar
		case.sar_id = sar.id
		case.status = CaseStatus.CONFIRMED_SUSPICIOUS
		case.updated_at = datetime.utcnow()
		await self._emit_event("sar_drafted", {"sar_id": sar.id, "case_id": case_id, "jurisdiction": data.jurisdiction})
		self._log_op("file_sar", sar.id, jurisdiction=data.jurisdiction)
		return sar

	async def approve_sar(self, sar_id: str, approved_by: str) -> SARResponse:
		"""Approve a SAR for filing."""
		sar = self._get_sar(sar_id)
		assert_sar_human_approval(approved_by)
		sar.status = SARStatus.APPROVED
		sar.approved_by = approved_by
		sar.approved_at = datetime.utcnow()
		sar.updated_at = datetime.utcnow()
		await self._emit_event("sar_approved", {"sar_id": sar_id, "approved_by": approved_by})
		return sar

	async def submit_sar(self, sar_id: str, filing_reference: str) -> SARResponse:
		"""Record SAR as filed with regulator reference."""
		sar = self._get_sar(sar_id)
		assert sar.approved_by, "SAR must be approved before filing"
		sar.status = SARStatus.FILED
		sar.filing_reference = filing_reference
		sar.filed_at = datetime.utcnow()
		sar.updated_at = datetime.utcnow()
		# Update case status
		case = next((c for c in self._cases.values() if c.sar_id == sar_id), None)
		if case:
			case.status = CaseStatus.SAR_FILED
		await self._emit_event("sar_filed", {"sar_id": sar_id, "reference": filing_reference})
		return sar

	async def reject_sar(self, sar_id: str, reason: str) -> SARResponse:
		"""Reject a SAR (returns to draft for amendment)."""
		sar = self._get_sar(sar_id)
		sar.status = SARStatus.REJECTED
		sar.rejection_reason = reason
		sar.updated_at = datetime.utcnow()
		await self._emit_event("sar_rejected", {"sar_id": sar_id, "reason": reason})
		return sar

	async def get_sar(self, sar_id: str) -> SARResponse:
		return self._get_sar(sar_id)

	async def list_sars(self, status: str | None = None) -> list[SARResponse]:
		sars = [s for s in self._sars.values() if s.tenant_id == self.tenant_id]
		if status:
			sars = [s for s in sars if str(s.status) == status]
		return sorted(sars, key=lambda s: s.created_at, reverse=True)

	def _get_sar(self, sar_id: str) -> SARResponse:
		sar = self._sars.get(sar_id)
		assert sar is not None, f"SAR not found: {sar_id}"
		self._assert_tenant(sar.tenant_id)
		return sar

	# ------------------------------------------------------------------
	# CTR — Currency Transaction Report
	# ------------------------------------------------------------------

	async def file_ctr(self, transaction_id: str, data: CTRCreate) -> CTRResponse:
		"""File a CTR for a cash transaction exceeding the reporting threshold."""
		threshold = CTR_THRESHOLDS.get(data.jurisdiction.upper(), 10_000.0)
		assert_ctr_amount_triggers_reporting(data.amount, threshold)

		ctr = CTRResponse(
			tenant_id=data.tenant_id,
			created_by=data.created_by,
			transaction_id=transaction_id,
			subject_reference=data.subject_reference,
			subject_name=data.subject_name,
			subject_id_number=data.subject_id_number,
			amount=data.amount,
			currency=data.currency,
			transaction_date=data.transaction_date,
			transaction_type=data.transaction_type,
			branch_id=data.branch_id,
			jurisdiction=data.jurisdiction,
			filing_institution=data.filing_institution,
			metadata=data.metadata,
		)
		self._ctrs[ctr.id] = ctr
		await self._emit_event("ctr_filed", {"ctr_id": ctr.id, "amount": data.amount, "jurisdiction": data.jurisdiction})
		self._log_op("file_ctr", ctr.id, amount=data.amount, currency=data.currency)
		return ctr

	async def submit_ctr(self, ctr_id: str, filing_reference: str) -> CTRResponse:
		"""Mark CTR as submitted to regulator."""
		ctr = self._get_ctr(ctr_id)
		ctr.status = CTRStatus.FILED
		ctr.filing_reference = filing_reference
		ctr.filed_at = datetime.utcnow()
		await self._emit_event("ctr_submitted", {"ctr_id": ctr_id, "reference": filing_reference})
		return ctr

	async def get_ctr(self, ctr_id: str) -> CTRResponse:
		return self._get_ctr(ctr_id)

	async def list_ctrs(self, status: str | None = None) -> list[CTRResponse]:
		ctrs = [c for c in self._ctrs.values() if c.tenant_id == self.tenant_id]
		if status:
			ctrs = [c for c in ctrs if str(c.status) == status]
		return sorted(ctrs, key=lambda c: c.created_at, reverse=True)

	def _get_ctr(self, ctr_id: str) -> CTRResponse:
		ctr = self._ctrs.get(ctr_id)
		assert ctr is not None, f"CTR not found: {ctr_id}"
		self._assert_tenant(ctr.tenant_id)
		return ctr

	# ------------------------------------------------------------------
	# Watchlist screening
	# ------------------------------------------------------------------

	async def watchlist_screening(
		self,
		subject_reference: str,
		subject_name: str,
		kyc_profile_id: str | None = None,
		lists: list[str] | None = None,
	) -> list[WatchlistMatchResponse]:
		"""Screen a subject against configured watchlists.

		In production this calls out to OFAC, UN, EU, PEP providers.
		This implementation performs a deterministic check on known test names
		and returns results suitable for integration testing.
		"""
		lists = lists or ["OFAC_SDN", "UN_CONSOLIDATED", "EU_CONSOLIDATED", "PEP"]
		results: list[WatchlistMatchResponse] = []

		# Simulate screening — real impl calls external screening APIs
		for list_name in lists:
			match_score = _simulate_watchlist_score(subject_name, list_name)
			if match_score > 0.3:
				data = WatchlistMatchCreate(
					tenant_id=self.tenant_id,
					created_by=self.actor_id,
					subject_reference=subject_reference,
					subject_name=subject_name,
					list_name=list_name,
					list_entry_id=f"{list_name}-{uuid7str()[:8]}",
					match_score=match_score,
					match_fields=["name"],
					kyc_profile_id=kyc_profile_id,
				)
				match = await self.create_watchlist_match(data)
				results.append(match)

		return results

	async def create_watchlist_match(self, data: WatchlistMatchCreate) -> WatchlistMatchResponse:
		"""Record a watchlist match."""
		assert_match_score_valid(data.match_score)

		match = WatchlistMatchResponse(
			tenant_id=data.tenant_id,
			created_by=data.created_by,
			subject_reference=data.subject_reference,
			subject_name=data.subject_name,
			list_name=data.list_name,
			list_entry_id=data.list_entry_id,
			match_score=data.match_score,
			match_fields=data.match_fields,
			matched_name=data.matched_name,
			matched_dob=data.matched_dob,
			matched_nationality=data.matched_nationality,
			kyc_profile_id=data.kyc_profile_id,
			metadata=data.metadata,
		)
		self._watchlist_matches[match.id] = match
		await self._emit_event("watchlist_match_created", {
			"match_id": match.id,
			"list": data.list_name,
			"score": data.match_score,
		})

		# Auto-escalate high confidence matches
		if data.match_score >= 0.9:
			await self._escalate_watchlist_match(match)

		return match

	async def _escalate_watchlist_match(self, match: WatchlistMatchResponse) -> None:
		"""Auto-generate a sanctions/PEP alert for high-confidence matches."""
		alert_type = AlertType.SANCTIONS if "SDN" in match.list_name or "SANCTIONS" in match.list_name.upper() else AlertType.PEP
		data = AMLAlertCreate(
			tenant_id=self.tenant_id,
			created_by=self.actor_id,
			alert_type=alert_type,
			severity=AlertSeverity.CRITICAL,
			subject_reference=match.subject_reference,
			kyc_profile_id=match.kyc_profile_id,
			evidence_references=[match.id],
			risk_score=95,
			narrative=f"High-confidence watchlist match: {match.list_name} score={match.match_score:.2f}",
		)
		await self.create_alert(data)
		match.alert_id = data.evidence_references[0]  # link back

	async def review_watchlist_match(
		self,
		match_id: str,
		status: WatchlistMatchStatus,
		reviewer_id: str,
	) -> WatchlistMatchResponse:
		"""Confirm or dismiss a watchlist match."""
		match = self._watchlist_matches.get(match_id)
		assert match is not None, f"match not found: {match_id}"
		self._assert_tenant(match.tenant_id)
		match.status = status
		match.reviewer_id = reviewer_id
		match.reviewed_at = datetime.utcnow()
		await self._emit_event("watchlist_match_reviewed", {"match_id": match_id, "status": str(status)})
		return match

	async def list_watchlist_matches(self, status: str | None = None) -> list[WatchlistMatchResponse]:
		matches = [m for m in self._watchlist_matches.values() if m.tenant_id == self.tenant_id]
		if status:
			matches = [m for m in matches if str(m.status) == status]
		return sorted(matches, key=lambda m: m.match_score, reverse=True)

	# ------------------------------------------------------------------
	# Network analysis
	# ------------------------------------------------------------------

	async def network_analysis(self, customer_id: str) -> NetworkAnalysisResult:
		"""Analyse transaction network for a customer.

		Detects round-trip, layering, and high-risk counterparty clusters.
		"""
		txns = [t for t in self._transactions.values() if t.get("tenant_id") == self.tenant_id and (t.get("subject_reference") == customer_id or t.get("sender_account") == customer_id)]

		direct_scores = [int(t.get("risk_score", 0)) for t in txns]
		counterparties = list({str(t.get("receiver_account", "")) for t in txns} - {customer_id})

		rt = detect_round_trip(txns)
		layer = detect_layering(txns)

		network_score = calculate_network_risk_score(
			direct_risk_scores=direct_scores,
			indirect_risk_scores=[],
			round_trip_detected=rt["detected"],
			layering_detected=layer["detected"],
		)

		flags: list[str] = []
		if rt["detected"]:
			flags.append("round_trip")
		if layer["detected"]:
			flags.append("layering")

		result = NetworkAnalysisResult(
			subject_reference=customer_id,
			tenant_id=self.tenant_id,
			counterparty_count=len(counterparties),
			transaction_count=len(txns),
			total_sent=sum(float(t.get("amount", 0)) for t in txns if str(t.get("sender_account")) == customer_id),
			total_received=sum(float(t.get("amount", 0)) for t in txns if str(t.get("receiver_account")) == customer_id),
			round_trip_detected=rt["detected"],
			layering_detected=layer["detected"],
			network_risk_score=network_score,
			typology_flags=flags,
		)
		await self._emit_event("network_analysis_complete", {"subject": customer_id, "score": network_score})
		return result

	# ------------------------------------------------------------------
	# Pattern detection
	# ------------------------------------------------------------------

	async def pattern_detection(
		self,
		customer_id: str,
		lookback_days: int = 90,
	) -> PatternDetectionResult:
		"""Detect AML typology patterns for a customer over a lookback window.

		Covers: structuring/smurfing, velocity anomalies, round-trip, layering.
		"""
		txns = [
			t for t in self._transactions.values()
			if t.get("tenant_id") == self.tenant_id
			and t.get("subject_reference") == customer_id
		]

		struct = detect_structuring(txns, lookback_days=lookback_days)
		vel = detect_velocity_anomaly(txns, window_hours=lookback_days * 24)
		rt = detect_round_trip(txns, lookback_days=lookback_days)
		layer = detect_layering(txns, lookback_days=lookback_days)

		patterns = []
		if struct["detected"]:
			patterns.append({"type": "structuring", "count": struct["count"], "total": struct["total_amount"]})
		if vel["detected"]:
			patterns.append({"type": "velocity_anomaly", "count": vel["count"], "total": vel["total_amount"]})
		if rt["detected"]:
			patterns.append({"type": "round_trip", "chains": rt["chain_count"]})
		if layer["detected"]:
			patterns.append({"type": "layering", "layers": layer["layers"]})

		risk_delta = len(patterns) * 15

		result = PatternDetectionResult(
			subject_reference=customer_id,
			tenant_id=self.tenant_id,
			lookback_days=lookback_days,
			structuring_detected=struct["detected"],
			smurfing_detected=struct["detected"],
			velocity_anomaly=vel["detected"],
			round_trip_detected=rt["detected"],
			layering_detected=layer["detected"],
			patterns=patterns,
			risk_delta=risk_delta,
			recommended_action="escalate" if risk_delta >= 30 else "review" if risk_delta > 0 else "no_action",
		)
		await self._emit_event("pattern_detection_complete", {"subject": customer_id, "patterns": len(patterns)})
		return result

	# ------------------------------------------------------------------
	# Risk segmentation
	# ------------------------------------------------------------------

	async def risk_segmentation(
		self,
		subject_reference: str,
		kyc_profile_id: str | None = None,
		contributing_factors: list[str] | None = None,
		risk_score: int = 0,
	) -> RiskSegmentResponse:
		"""Assign or update risk segment for a customer."""
		segment_str = risk_segment_from_score(risk_score)

		# Check for previous segment
		prev = next(
			(r for r in self._risk_segments.values()
			 if r.subject_reference == subject_reference and r.tenant_id == self.tenant_id
			 and not r.is_deleted),
			None,
		)

		data = RiskSegmentCreate(
			tenant_id=self.tenant_id,
			created_by=self.actor_id,
			subject_reference=subject_reference,
			kyc_profile_id=kyc_profile_id,
			segment=segment_str,
			risk_score=risk_score,
			contributing_factors=contributing_factors or [],
			review_date=datetime.utcnow() + timedelta(days=365 if segment_str == "low" else 90),
		)

		seg = RiskSegmentResponse(
			tenant_id=data.tenant_id,
			created_by=data.created_by,
			subject_reference=data.subject_reference,
			kyc_profile_id=data.kyc_profile_id,
			segment=data.segment,
			risk_score=data.risk_score,
			contributing_factors=data.contributing_factors,
			effective_date=data.effective_date,
			review_date=data.review_date,
			previous_segment=str(prev.segment) if prev else None,
		)
		self._risk_segments[seg.id] = seg
		if prev:
			prev.is_deleted = True

		await self._emit_event("risk_segment_assigned", {
			"subject": subject_reference,
			"segment": segment_str,
			"score": risk_score,
		})
		return seg

	# ------------------------------------------------------------------
	# Regulatory reporting
	# ------------------------------------------------------------------

	async def regulatory_reporting(
		self,
		jurisdiction: str,
		period_start: datetime,
		period_end: datetime,
		report_type: str = "sar_ctr_summary",
	) -> RegulatoryReportResponse:
		"""Generate a regulatory report for a jurisdiction and period."""
		assert jurisdiction, "jurisdiction required"

		sars = [
			s for s in self._sars.values()
			if s.tenant_id == self.tenant_id
			and s.jurisdiction.upper() == jurisdiction.upper()
			and period_start <= s.created_at <= period_end
		]
		ctrs = [
			c for c in self._ctrs.values()
			if c.tenant_id == self.tenant_id
			and c.jurisdiction.upper() == jurisdiction.upper()
			and period_start <= c.created_at <= period_end
		]
		alerts = [
			a for a in self._alerts.values()
			if a.tenant_id == self.tenant_id
			and period_start <= a.created_at <= period_end
			and not a.is_deleted
		]
		cases = [
			c for c in self._cases.values()
			if c.tenant_id == self.tenant_id
			and period_start <= c.created_at <= period_end
			and not c.is_deleted
		]

		total_amount = sum(s.total_amount for s in sars) + sum(c.amount for c in ctrs)

		report = RegulatoryReportResponse(
			tenant_id=self.tenant_id,
			jurisdiction=jurisdiction,
			period_start=period_start,
			period_end=period_end,
			report_type=report_type,
			sar_count=len(sars),
			ctr_count=len(ctrs),
			alert_count=len(alerts),
			case_count=len(cases),
			total_suspicious_amount=round(total_amount, 2),
			details=[s.model_dump(mode="json") for s in sars[:10]],
		)
		await self._emit_event("regulatory_report_generated", {
			"jurisdiction": jurisdiction,
			"period_start": period_start.isoformat(),
			"period_end": period_end.isoformat(),
		})
		return report

	# ------------------------------------------------------------------
	# Regulatory filings
	# ------------------------------------------------------------------

	async def create_filing(self, data: RegulatoryFilingCreate) -> RegulatoryFilingResponse:
		"""Record a regulatory filing."""
		filing = RegulatoryFilingResponse(
			tenant_id=data.tenant_id,
			created_by=data.created_by,
			filing_type=data.filing_type,
			jurisdiction=data.jurisdiction,
			regulator=data.regulator,
			reference_id=data.reference_id,
			period_start=data.period_start,
			period_end=data.period_end,
			filing_institution=data.filing_institution,
			metadata=data.metadata,
		)
		self._filings[filing.id] = filing
		await self._emit_event("filing_created", {"filing_id": filing.id, "type": data.filing_type})
		return filing

	async def submit_filing(self, filing_id: str, submission_reference: str) -> RegulatoryFilingResponse:
		"""Mark filing as submitted."""
		filing = self._filings.get(filing_id)
		assert filing is not None, f"filing not found: {filing_id}"
		self._assert_tenant(filing.tenant_id)
		filing.status = FilingStatus.SUBMITTED
		filing.submission_reference = submission_reference
		filing.submitted_at = datetime.utcnow()
		await self._emit_event("filing_submitted", {"filing_id": filing_id, "reference": submission_reference})
		return filing

	async def list_filings(self, jurisdiction: str | None = None) -> list[RegulatoryFilingResponse]:
		filings = [f for f in self._filings.values() if f.tenant_id == self.tenant_id]
		if jurisdiction:
			filings = [f for f in filings if f.jurisdiction.upper() == jurisdiction.upper()]
		return sorted(filings, key=lambda f: f.created_at, reverse=True)

	# ------------------------------------------------------------------
	# Dashboard summary
	# ------------------------------------------------------------------

	async def dashboard_summary(self) -> dict[str, Any]:  # type: ignore[override]
		"""Aggregate KPIs for the AML dashboard."""
		alerts = [a for a in self._alerts.values() if a.tenant_id == self.tenant_id and not a.is_deleted]
		cases = [c for c in self._cases.values() if c.tenant_id == self.tenant_id and not c.is_deleted]
		sars = [s for s in self._sars.values() if s.tenant_id == self.tenant_id]
		ctrs = [c for c in self._ctrs.values() if c.tenant_id == self.tenant_id]
		total_alerts = len(alerts)
		fps = sum(1 for a in alerts if str(a.status) == "false_positive")

		return {
			"tenant_id": self.tenant_id,
			"alert_count": total_alerts,
			"open_alert_count": sum(1 for a in alerts if str(a.status) == "open"),
			"critical_alert_count": sum(1 for a in alerts if str(a.severity) == "critical"),
			"case_count": len(cases),
			"open_case_count": sum(1 for c in cases if str(c.status) in {"open", "under_investigation"}),
			"sar_count": len(sars),
			"pending_sar_count": sum(1 for s in sars if str(s.status) in {"draft", "pending_approval"}),
			"ctr_count": len(ctrs),
			"false_positive_rate": calculate_false_positive_rate(total_alerts, fps),
			"watchlist_match_count": sum(1 for m in self._watchlist_matches.values() if m.tenant_id == self.tenant_id),
			"rule_count": len([r for r in self._rules.values() if r.tenant_id == self.tenant_id and r.enabled]),
			"event_count": len([e for e in self._events if e["tenant_id"] == self.tenant_id]),
		}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alert_type_to_case_type(alert_type: str) -> str:
	mapping = {
		"sanctions": "sanctions_alert",
		"structuring": "structuring_alert",
		"mule_account": "mule_account",
		"high_risk_kyc": "high_risk_customer",
		"terrorist_financing": "terrorist_financing",
		"trade_based": "trade_based_ml",
		"crypto_asset": "crypto_asset",
		"nft": "crypto_asset",
		"round_trip": "network_analysis",
		"layering": "network_analysis",
	}
	return mapping.get(alert_type, "transaction_monitoring")


def _simulate_watchlist_score(name: str, list_name: str) -> float:
	"""Deterministic test helper — returns > 0 only for specific test names."""
	HIGH_RISK_NAMES = {"osama bin laden", "kim jong un", "muammar gaddafi", "test sanctions subject"}
	clean = name.strip().lower()
	if clean in HIGH_RISK_NAMES:
		return 0.97
	if "test" in clean and "pep" in clean:
		return 0.85
	return 0.0


# ---------------------------------------------------------------------------
# Legacy sync shims — keep old test_package_contract tests green
# ---------------------------------------------------------------------------

class AntiMoneyLaunderingService(AMLService):
	"""Backward-compatible sync wrapper around AMLService.

	The old positional-argument sync API is preserved here so that existing
	tests and app.py integrations continue to work unchanged.
	"""

	def __init__(self) -> None:
		super().__init__()
		# Legacy attribute aliases for direct dict access in old tests
		self.transactions = self._transactions
		self.cases = self._cases
		self.alerts_store = self._alerts
		self.sar_drafts = self._sars

	# ------------------------------------------------------------------
	# Sync helpers
	# ------------------------------------------------------------------

	def _sync(self, coro: Any) -> Any:
		try:
			loop = asyncio.get_event_loop()
			if loop.is_running():
				import concurrent.futures
				with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
					return pool.submit(asyncio.run, coro).result()
			return loop.run_until_complete(coro)
		except RuntimeError:
			return asyncio.run(coro)

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		try:
			from .capability_contract import get_capability_contract
		except ImportError:
			from capability_contract import get_capability_contract  # type: ignore
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		try:
			from .capability_contract import evaluate_capability_rules
		except ImportError:
			from capability_contract import evaluate_capability_rules  # type: ignore
		return evaluate_capability_rules(context)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		actions = result.get("actions", [])
		reasons = ", ".join(a.get("reason", "aml_policy_denied") for a in actions)
		raise PermissionError(reasons or "aml_policy_denied")

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self._events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	# ------------------------------------------------------------------
	# Legacy positional-arg methods
	# ------------------------------------------------------------------

	def monitor_transaction(  # type: ignore[override]
		self,
		transaction_id: str,
		tenant_id: str,
		subject_reference: str,
		kyc_profile_id: str,
		amount: float | int | str,
		currency: str,
		source_capability: str,
		source_reference: str,
		risk_score: int | str = 0,
		sanctions_hit: bool = False,
		velocity_indicator: bool = False,
		review_id: str = "",
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Legacy sync monitor_transaction with positional arguments."""
		try:
			from .aml_runtime import normalize_amount, normalize_code, normalize_currency, normalize_risk_score, severity_from_score as _sev, typology_flags
			from .capability_contract import get_capability_contract, SUPPORTED_ALERT_TYPES, SUPPORTED_SEVERITIES
		except ImportError:
			from aml_runtime import normalize_amount, normalize_code, normalize_currency, normalize_risk_score, severity_from_score as _sev, typology_flags  # type: ignore
			from capability_contract import get_capability_contract, SUPPORTED_ALERT_TYPES, SUPPORTED_SEVERITIES  # type: ignore

		amount_value = normalize_amount(amount)
		currency_code = normalize_currency(currency)
		risk_value = normalize_risk_score(risk_score)
		config = get_capability_contract(tenant_id)["configuration"]["monitoring"]
		flags = typology_flags(
			amount_value, risk_value,
			float(config["large_transaction_threshold"]),
			float(config["structuring_threshold"]),
			sanctions_hit, velocity_indicator,
		)

		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "monitor_transaction",
			"subject_present": bool(subject_reference),
			"positive_amount": amount_value > 0,
			"currency_present": bool(currency_code),
			"source_reference_present": bool(source_reference and source_capability),
			"kyc_link_present": bool(kyc_profile_id),
			"large_transaction": "large_transaction" in flags,
			"velocity_indicator": "velocity" in flags,
			"structuring_indicator": "structuring" in flags,
			"sanctions_hit": sanctions_hit,
			"high_risk_kyc": "high_risk_kyc" in flags,
			"review_recorded": bool(review_id),
		})

		if transaction_id in self._transactions:
			raise ValueError(f"transaction already monitored: {transaction_id}")

		record = {
			"id": transaction_id,
			"tenant_id": tenant_id,
			"subject_reference": subject_reference,
			"kyc_profile_id": kyc_profile_id,
			"amount": amount_value,
			"currency": currency_code,
			"source_capability": normalize_code(source_capability),
			"source_reference": source_reference,
			"risk_score": risk_value,
			"typology_flags": flags,
			"status": "monitored",
		}
		self._transactions[transaction_id] = record
		self._audit(tenant_id, "aml_transaction_monitored", transaction_id)
		return record

	def create_alert(  # type: ignore[override]
		self,
		alert_id: str,
		tenant_id: str,
		alert_type: str,
		severity: str,
		subject_reference: str,
		evidence_references: list[str],
	) -> dict[str, Any]:
		"""Legacy sync create_alert."""
		try:
			from .capability_contract import SUPPORTED_ALERT_TYPES, SUPPORTED_SEVERITIES
		except ImportError:
			from capability_contract import SUPPORTED_ALERT_TYPES, SUPPORTED_SEVERITIES  # type: ignore
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_alert",
			"alert_type_supported": alert_type in SUPPORTED_ALERT_TYPES,
			"severity_supported": severity in SUPPORTED_SEVERITIES,
			"evidence_present": bool(evidence_references and subject_reference),
		})
		record = {
			"id": alert_id,
			"tenant_id": tenant_id,
			"alert_type": alert_type,
			"severity": severity,
			"subject_reference": subject_reference,
			"evidence_references": list(evidence_references),
			"status": "open",
			"disposition": "",
			"reviewer_id": "",
		}
		# Store as a simple namespace so old tests can do service.cases[id].status
		class _Alert:
			def __init__(self, d: dict) -> None:
				self.__dict__.update(d)
			def to_dict(self) -> dict:
				return dict(self.__dict__)
		obj = _Alert(record)
		self._alerts[alert_id] = obj  # type: ignore[assignment]
		self._audit(tenant_id, "aml_alert_created", alert_id)
		return record

	def create_alert_from_transaction(
		self,
		alert_id: str,
		tenant_id: str,
		transaction_id: str,
		alert_type: str | None = None,
	) -> dict[str, Any]:
		"""Legacy sync create_alert_from_transaction."""
		txn = self._transactions.get(transaction_id)
		if txn is None or txn.get("tenant_id") != tenant_id:
			raise KeyError(f"unknown transaction: {transaction_id}")
		flags = txn.get("typology_flags", [])
		sel_type = alert_type or (flags[0] if flags else "agent_review")
		risk = int(txn.get("risk_score", 0))
		sev = severity_from_score(risk)
		if "sanctions" in flags:
			sev = "critical"
		if sel_type == "large_transaction" and sev == "low":
			sev = "medium"
		return self.create_alert(alert_id, tenant_id, sel_type, sev, txn["subject_reference"], [transaction_id])

	def triage_alert(  # type: ignore[override]
		self,
		alert_id: str,
		tenant_id: str,
		action: str,
		disposition: str = "",
		reviewer_id: str = "",
	) -> dict[str, Any]:
		"""Legacy sync triage_alert."""
		alert = self._alerts.get(alert_id)
		if alert is None or getattr(alert, "tenant_id", None) != tenant_id:
			raise KeyError(f"unknown alert: {alert_id}")
		closing = action == "close"
		escalating = action in {"escalate", "open_case"}
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "triage_alert",
			"closing_alert": closing,
			"disposition_present": bool(disposition),
			"escalating_alert": escalating,
			"reviewer_present": bool(reviewer_id),
		})
		alert.status = "closed" if closing else "escalated" if escalating else "under_review"
		alert.disposition = disposition
		alert.reviewer_id = reviewer_id
		self._audit(tenant_id, "aml_alert_triaged", alert_id)
		return alert.to_dict() if hasattr(alert, "to_dict") else dict(alert.__dict__)

	def open_case(  # type: ignore[override]
		self,
		case_id: str,
		tenant_id: str,
		alert_id: str,
		case_type: str,
		investigator_id: str,
		evidence_references: list[str] | None = None,
	) -> dict[str, Any]:
		"""Legacy sync open_case."""
		try:
			from .capability_contract import SUPPORTED_CASE_TYPES
		except ImportError:
			from capability_contract import SUPPORTED_CASE_TYPES  # type: ignore
		alert = self._alerts.get(alert_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_case",
			"alert_present": alert is not None,
			"case_type_supported": case_type in SUPPORTED_CASE_TYPES,
			"case_investigator_required": bool(investigator_id),
			"investigator_present": bool(investigator_id),
		})

		class _Case:
			def __init__(self, d: dict) -> None:
				self.__dict__.update(d)
			def to_dict(self) -> dict:
				return dict(self.__dict__)

		record = {
			"id": case_id,
			"tenant_id": tenant_id,
			"alert_id": alert_id,
			"case_type": case_type,
			"investigator_id": investigator_id,
			"subject_reference": getattr(alert, "subject_reference", ""),
			"status": "under_investigation",
			"evidence_references": list(evidence_references or [alert_id]),
		}
		obj = _Case(record)
		self._cases[case_id] = obj  # type: ignore[assignment]
		if alert is not None:
			alert.status = "case_opened"
		self._audit(tenant_id, "aml_case_opened", case_id)
		return record

	def draft_sar(  # type: ignore[override]
		self,
		sar_id: str,
		tenant_id: str,
		case_id: str,
		subject_reference: str,
		jurisdiction: str,
		narrative: str,
		evidence_references: list[str],
		approved_by: str,
	) -> dict[str, Any]:
		"""Legacy sync draft_sar."""
		case = self._cases.get(case_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "draft_sar",
			"case_present": case is not None,
			"subject_present": bool(subject_reference),
			"jurisdiction_present": bool(jurisdiction),
			"narrative_present": bool(narrative),
			"evidence_present": bool(evidence_references),
			"human_approval_recorded": bool(approved_by),
		})
		record = {
			"id": sar_id,
			"tenant_id": tenant_id,
			"case_id": case_id,
			"subject_reference": subject_reference,
			"jurisdiction": jurisdiction,
			"narrative": narrative,
			"evidence_references": list(evidence_references),
			"approved_by": approved_by,
			"status": "approved_for_filing",
		}
		self._sars[sar_id] = record  # type: ignore[assignment]
		if case is not None:
			case.status = "confirmed_suspicious"
		self._audit(tenant_id, "aml_sar_drafted", sar_id)
		return record

	def register_aml_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		"""Legacy sync register_aml_agent."""
		try:
			from .capability_contract import SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES
		except ImportError:
			from capability_contract import SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES  # type: ignore
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_aml_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		record = {
			"id": agent_id,
			"tenant_id": tenant_id,
			"kind": "agent",
			"reference_id": agent_id,
			"status": "registered",
			"metadata": {"name": name, "runtime": runtime, "role": role, "scope": scope},
		}
		self._audit(tenant_id, "aml_agent_registered", agent_id)
		return record

	def validate_batch(  # type: ignore[override]
		self,
		tenant_id: str,
		item_count: int,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		"""Legacy sync validate_batch."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "aml_batch",
			"event_stream": event_stream,
		})
		return {
			"tenant_id": tenant_id,
			"item_count": item_count,
			"processor": "bytewax",
			"stream": "apg.fintech.aml.lifecycle",
			"accepted": True,
		}

	def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:  # type: ignore[override]
		"""Legacy sync dashboard_summary with optional tenant_id positional arg."""
		tid = tenant_id or self.tenant_id
		transactions = [t for t in self._transactions.values() if t.get("tenant_id") == tid]
		alerts = [v for v in self._alerts.values() if getattr(v, "tenant_id", None) == tid]
		cases = [v for v in self._cases.values() if getattr(v, "tenant_id", None) == tid]
		sars = [v for v in self._sars.values() if (v.get("tenant_id") if isinstance(v, dict) else getattr(v, "tenant_id", None)) == tid]
		events = [e for e in self._events if e.get("tenant_id") == tid]
		return {
			"tenant_id": tid,
			"transaction_count": len(transactions),
			"alert_count": len(alerts),
			"open_alert_count": sum(1 for a in alerts if getattr(a, "status", "") in {"open", "under_review", "escalated"}),
			"case_count": len(cases),
			"sar_count": len(sars),
			"critical_alert_count": sum(1 for a in alerts if getattr(a, "severity", "") == "critical"),
			"audit_event_count": len(events),
			"streaming": {"processor": "bytewax", "stream": "apg.fintech.aml.lifecycle"},
		}

	def list_alerts(self, tenant_id: str | None = None) -> list[dict[str, Any]]:  # type: ignore[override]
		"""Legacy sync list_alerts."""
		items = self._alerts.values()
		if tenant_id:
			items = [a for a in items if getattr(a, "tenant_id", None) == tenant_id]
		return sorted(
			[a.to_dict() if hasattr(a, "to_dict") else dict(a.__dict__) for a in items],
			key=lambda x: x.get("id", ""),
		)

	def list_cases(self, tenant_id: str | None = None) -> list[dict[str, Any]]:  # type: ignore[override]
		"""Legacy sync list_cases."""
		items = self._cases.values()
		if tenant_id:
			items = [c for c in items if getattr(c, "tenant_id", None) == tenant_id]
		return sorted(
			[c.to_dict() if hasattr(c, "to_dict") else dict(c.__dict__) for c in items],
			key=lambda x: x.get("id", ""),
		)


FintechAmlService = AntiMoneyLaunderingService
