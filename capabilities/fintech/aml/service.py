"""Executable service layer for APG Anti Money Laundering."""

from __future__ import annotations

from typing import Any

try:
	from .aml_runtime import normalize_amount, normalize_code, normalize_currency, normalize_risk_score, severity_from_score, typology_flags
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALERT_TYPES, SUPPORTED_CASE_TYPES, SUPPORTED_SEVERITIES, evaluate_capability_rules, get_capability_contract
	from .models import AmlAlert, AmlCase, AmlEvidence, AmlSarDraft, AmlTransaction
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from aml_runtime import normalize_amount, normalize_code, normalize_currency, normalize_risk_score, severity_from_score, typology_flags  # type: ignore
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALERT_TYPES, SUPPORTED_CASE_TYPES, SUPPORTED_SEVERITIES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import AmlAlert, AmlCase, AmlEvidence, AmlSarDraft, AmlTransaction  # type: ignore


class AntiMoneyLaunderingService:
	"""Dependency-light AML lifecycle runtime for generated applications."""

	def __init__(self) -> None:
		self.transactions: dict[str, AmlTransaction] = {}
		self.alerts: dict[str, AmlAlert] = {}
		self.cases: dict[str, AmlCase] = {}
		self.sar_drafts: dict[str, AmlSarDraft] = {}
		self.evidence: dict[str, AmlEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def monitor_transaction(self, transaction_id: str, tenant_id: str, subject_reference: str, kyc_profile_id: str, amount: float | int | str, currency: str, source_capability: str, source_reference: str, risk_score: int | str = 0, sanctions_hit: bool = False, velocity_indicator: bool = False, review_id: str = "", policy_attached: bool = True) -> dict[str, Any]:
		amount_value = normalize_amount(amount)
		currency_code = normalize_currency(currency)
		risk_value = normalize_risk_score(risk_score)
		configuration = get_capability_contract(tenant_id)["configuration"]["monitoring"]
		flags = typology_flags(amount_value, risk_value, float(configuration["large_transaction_threshold"]), float(configuration["structuring_threshold"]), sanctions_hit, velocity_indicator)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "monitor_transaction", "subject_present": bool(subject_reference), "positive_amount": amount_value > 0, "currency_present": bool(currency_code), "source_reference_present": bool(source_reference and source_capability), "kyc_link_present": bool(kyc_profile_id), "large_transaction": "large_transaction" in flags, "velocity_indicator": "velocity" in flags, "structuring_indicator": "structuring" in flags, "sanctions_hit": sanctions_hit, "high_risk_kyc": "high_risk_kyc" in flags, "review_recorded": bool(review_id)})
		if transaction_id in self.transactions:
			raise ValueError(f"transaction already monitored: {transaction_id}")
		transaction = AmlTransaction(transaction_id, tenant_id, subject_reference, kyc_profile_id, amount_value, currency_code, normalize_code(source_capability), source_reference, risk_value, flags)
		self.transactions[transaction_id] = transaction
		self._audit(tenant_id, "aml_transaction_monitored", transaction_id)
		return transaction.to_dict()

	def create_alert(self, alert_id: str, tenant_id: str, alert_type: str, severity: str, subject_reference: str, evidence_references: list[str]) -> dict[str, Any]:
		alert_type = normalize_code(alert_type)
		severity = normalize_code(severity)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "create_alert", "alert_type_supported": alert_type in SUPPORTED_ALERT_TYPES, "severity_supported": severity in SUPPORTED_SEVERITIES, "evidence_present": bool(evidence_references and subject_reference)})
		alert = AmlAlert(alert_id, tenant_id, alert_type, severity, subject_reference, list(evidence_references))
		self.alerts[alert_id] = alert
		self._audit(tenant_id, "aml_alert_created", alert_id)
		return alert.to_dict()

	def create_alert_from_transaction(self, alert_id: str, tenant_id: str, transaction_id: str, alert_type: str | None = None) -> dict[str, Any]:
		transaction = self._tenant_transaction(transaction_id, tenant_id)
		selected_type = normalize_code(alert_type or (transaction.typology_flags[0] if transaction.typology_flags else "agent_review"))
		severity = severity_from_score(transaction.risk_score)
		if "sanctions" in transaction.typology_flags:
			severity = "critical"
		if selected_type == "large_transaction" and severity == "low":
			severity = "medium"
		return self.create_alert(alert_id, tenant_id, selected_type, severity, transaction.subject_reference, [transaction_id])

	def triage_alert(self, alert_id: str, tenant_id: str, action: str, disposition: str = "", reviewer_id: str = "") -> dict[str, Any]:
		alert = self._tenant_alert(alert_id, tenant_id)
		action = normalize_code(action)
		closing = action == "close"
		escalating = action in {"escalate", "open_case"}
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "triage_alert", "closing_alert": closing, "disposition_present": bool(disposition), "escalating_alert": escalating, "reviewer_present": bool(reviewer_id)})
		alert.status = "closed" if closing else "escalated" if escalating else "under_review"
		alert.disposition = disposition
		alert.reviewer_id = reviewer_id
		self._audit(tenant_id, "aml_alert_triaged", alert_id)
		return alert.to_dict()

	def open_case(self, case_id: str, tenant_id: str, alert_id: str, case_type: str, investigator_id: str, evidence_references: list[str] | None = None) -> dict[str, Any]:
		alert = self._tenant_alert_or_none(alert_id, tenant_id)
		case_type = normalize_code(case_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "open_case", "alert_present": alert is not None, "case_type_supported": case_type in SUPPORTED_CASE_TYPES, "investigator_present": bool(investigator_id)})
		case = AmlCase(case_id, tenant_id, alert_id, case_type, investigator_id, alert.subject_reference if alert else "", "under_investigation", list(evidence_references or [alert_id]))
		self.cases[case_id] = case
		if alert is not None:
			alert.status = "case_opened"
		self._audit(tenant_id, "aml_case_opened", case_id)
		return case.to_dict()

	def draft_sar(self, sar_id: str, tenant_id: str, case_id: str, subject_reference: str, jurisdiction: str, narrative: str, evidence_references: list[str], approved_by: str) -> dict[str, Any]:
		case = self._tenant_case_or_none(case_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "draft_sar", "case_present": case is not None, "subject_present": bool(subject_reference), "jurisdiction_present": bool(jurisdiction), "narrative_present": bool(narrative), "evidence_present": bool(evidence_references), "human_approval_recorded": bool(approved_by)})
		draft = AmlSarDraft(sar_id, tenant_id, case_id, subject_reference, normalize_currency(jurisdiction), narrative, list(evidence_references), approved_by)
		self.sar_drafts[sar_id] = draft
		if case is not None:
			case.status = "confirmed_suspicious"
		self._audit(tenant_id, "aml_sar_drafted", sar_id)
		return draft.to_dict()

	def register_aml_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_aml_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		evidence = self._record_evidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self._audit(tenant_id, "aml_agent_registered", agent_id)
		return evidence

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "aml_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.aml.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		transactions = [item for item in self.transactions.values() if item.tenant_id == tenant_id]
		alerts = [item for item in self.alerts.values() if item.tenant_id == tenant_id]
		cases = [item for item in self.cases.values() if item.tenant_id == tenant_id]
		return {"tenant_id": tenant_id, "transaction_count": len(transactions), "alert_count": len(alerts), "open_alert_count": sum(1 for item in alerts if item.status in {"open", "under_review", "escalated"}), "case_count": len(cases), "sar_count": sum(1 for item in self.sar_drafts.values() if item.tenant_id == tenant_id), "critical_alert_count": sum(1 for item in alerts if item.severity == "critical"), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def list_alerts(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		alerts = self.alerts.values()
		if tenant_id is not None:
			alerts = [alert for alert in alerts if alert.tenant_id == tenant_id]
		return [alert.to_dict() for alert in sorted(alerts, key=lambda item: item.id)]

	def list_cases(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		cases = self.cases.values()
		if tenant_id is not None:
			cases = [case for case in cases if case.tenant_id == tenant_id]
		return [case.to_dict() for case in sorted(cases, key=lambda item: item.id)]

	def _tenant_transaction(self, transaction_id: str, tenant_id: str) -> AmlTransaction:
		transaction = self.transactions.get(transaction_id)
		if transaction is None or transaction.tenant_id != tenant_id:
			raise KeyError(f"unknown AML transaction: {transaction_id}")
		return transaction

	def _tenant_alert_or_none(self, alert_id: str, tenant_id: str) -> AmlAlert | None:
		alert = self.alerts.get(alert_id)
		if alert is None or alert.tenant_id != tenant_id:
			return None
		return alert

	def _tenant_alert(self, alert_id: str, tenant_id: str) -> AmlAlert:
		alert = self._tenant_alert_or_none(alert_id, tenant_id)
		if alert is None:
			raise KeyError(f"unknown AML alert: {alert_id}")
		return alert

	def _tenant_case_or_none(self, case_id: str, tenant_id: str) -> AmlCase | None:
		case = self.cases.get(case_id)
		if case is None or case.tenant_id != tenant_id:
			return None
		return case

	def _record_evidence(self, evidence_id: str, tenant_id: str, kind: str, reference_id: str, status: str, metadata: dict[str, Any]) -> dict[str, Any]:
		evidence = AmlEvidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
		self.evidence[evidence_id] = evidence
		return evidence.to_dict()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "aml_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "aml_policy_denied")


FintechAmlService = AntiMoneyLaunderingService
