"""Executable service layer for APG Fraud Detection."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CASE_TYPES, SUPPORTED_CHANNELS, SUPPORTED_DECISIONS, SUPPORTED_SIGNAL_TYPES, evaluate_capability_rules, get_capability_contract
	from .fraud_runtime import MONEY_SIGNAL_TYPES, collect_indicators, normalize_amount, normalize_code, normalize_currency, normalize_risk_score, recommended_decision, risk_band
	from .models import FraudCase, FraudDecision, FraudEvidence, FraudSignal
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CASE_TYPES, SUPPORTED_CHANNELS, SUPPORTED_DECISIONS, SUPPORTED_SIGNAL_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from fraud_runtime import MONEY_SIGNAL_TYPES, collect_indicators, normalize_amount, normalize_code, normalize_currency, normalize_risk_score, recommended_decision, risk_band  # type: ignore
	from models import FraudCase, FraudDecision, FraudEvidence, FraudSignal  # type: ignore


class FraudDetectionService:
	"""Dependency-light fraud runtime for generated applications."""

	def __init__(self) -> None:
		self.signals: dict[str, FraudSignal] = {}
		self.decisions: dict[str, FraudDecision] = {}
		self.cases: dict[str, FraudCase] = {}
		self.evidence: dict[str, FraudEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def score_signal(self, signal_id: str, tenant_id: str, subject_reference: str, kyc_profile_id: str, signal_type: str, channel: str, source_reference: str, amount: float | int | str | None = None, currency: str = "", risk_score: int | str = 0, velocity_indicator: bool = False, device_anomaly: bool = False, geo_anomaly: bool = False, aml_alert_present: bool = False, chargeback_signal: bool = False, account_takeover_indicator: bool = False, evidence_references: list[str] | None = None, review_id: str = "", policy_attached: bool = True) -> dict[str, Any]:
		signal_type = normalize_code(signal_type)
		channel = normalize_code(channel)
		amount_value = normalize_amount(amount)
		currency_code = normalize_currency(currency)
		score = normalize_risk_score(risk_score)
		money_signal = signal_type in MONEY_SIGNAL_TYPES
		chargeback = chargeback_signal or signal_type == "chargeback"
		indicators = collect_indicators(score, velocity_indicator, device_anomaly, geo_anomaly, aml_alert_present, chargeback, account_takeover_indicator)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "score_signal", "subject_present": bool(subject_reference), "signal_type_supported": signal_type in SUPPORTED_SIGNAL_TYPES, "channel_supported": channel in SUPPORTED_CHANNELS, "source_reference_present": bool(source_reference), "kyc_link_present": bool(kyc_profile_id), "money_signal": money_signal, "positive_amount": amount_value > 0, "currency_present": bool(currency_code), "risk_score_out_of_range": not 0 <= score <= 100, "high_risk_score": score >= 75, "velocity_indicator": velocity_indicator, "device_anomaly": device_anomaly, "geo_anomaly": geo_anomaly, "aml_alert_present": aml_alert_present, "account_takeover_indicator": account_takeover_indicator, "chargeback_signal": chargeback, "evidence_present": bool(evidence_references), "review_recorded": bool(review_id)})
		if signal_id in self.signals:
			raise ValueError(f"fraud signal already exists: {signal_id}")
		signal = FraudSignal(signal_id, tenant_id, subject_reference, kyc_profile_id, signal_type, channel, source_reference, amount_value, currency_code, score, indicators)
		self.signals[signal_id] = signal
		self._audit(tenant_id, "fraud_signal_scored", signal_id)
		return signal.to_dict() | {"risk_band": risk_band(score), "recommended_decision": recommended_decision(score)}

	def record_decision(self, decision_id: str, tenant_id: str, signal_id: str, decision: str, reason: str = "", reviewer_id: str = "", challenge_reference: str = "", human_approval: str = "") -> dict[str, Any]:
		signal = self._tenant_signal_or_none(signal_id, tenant_id)
		decision = normalize_code(decision)
		hold_or_block = decision in {"hold", "block"}
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_decision", "signal_present": signal is not None, "decision_supported": decision in SUPPORTED_DECISIONS, "step_up_decision": decision == "step_up", "challenge_present": bool(challenge_reference), "hold_or_block": hold_or_block, "reason_present": bool(reason), "human_approval_recorded": bool(human_approval)})
		record = FraudDecision(decision_id, tenant_id, signal_id, decision, reason, reviewer_id, challenge_reference, human_approval, "applied")
		self.decisions[decision_id] = record
		if signal is not None:
			signal.status = decision
		self._audit(tenant_id, "fraud_decision_recorded", decision_id)
		return record.to_dict()

	def open_case(self, case_id: str, tenant_id: str, signal_id: str, case_type: str, investigator_id: str, evidence_references: list[str]) -> dict[str, Any]:
		signal = self._tenant_signal_or_none(signal_id, tenant_id)
		case_type = normalize_code(case_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "open_case", "signal_present": signal is not None, "case_type_supported": case_type in SUPPORTED_CASE_TYPES, "investigator_present": bool(investigator_id), "evidence_present": bool(evidence_references)})
		case = FraudCase(case_id, tenant_id, signal_id, case_type, investigator_id, signal.subject_reference if signal else "", list(evidence_references), "open")
		self.cases[case_id] = case
		if signal is not None:
			signal.status = "case_opened"
		self._audit(tenant_id, "fraud_case_opened", case_id)
		return case.to_dict()

	def resolve_case(self, case_id: str, tenant_id: str, disposition: str, reviewer_id: str) -> dict[str, Any]:
		case = self._tenant_case(case_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "resolve_case", "disposition_present": bool(disposition), "reviewer_present": bool(reviewer_id)})
		case.status = "resolved"
		case.disposition = disposition
		case.reviewer_id = reviewer_id
		self._audit(tenant_id, "fraud_case_resolved", case_id)
		return case.to_dict()

	def register_fraud_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_fraud_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		evidence = self._record_evidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self._audit(tenant_id, "fraud_agent_registered", agent_id)
		return evidence

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "fraud_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.fraud.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		signals = [item for item in self.signals.values() if item.tenant_id == tenant_id]
		decisions = [item for item in self.decisions.values() if item.tenant_id == tenant_id]
		cases = [item for item in self.cases.values() if item.tenant_id == tenant_id]
		return {"tenant_id": tenant_id, "signal_count": len(signals), "decision_count": len(decisions), "case_count": len(cases), "open_case_count": sum(1 for item in cases if item.status != "resolved"), "high_risk_signal_count": sum(1 for item in signals if item.risk_score >= 75), "blocked_count": sum(1 for item in decisions if item.decision == "block"), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def list_signals(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		signals = self.signals.values()
		if tenant_id is not None:
			signals = [signal for signal in signals if signal.tenant_id == tenant_id]
		return [signal.to_dict() for signal in sorted(signals, key=lambda item: item.id)]

	def list_cases(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		cases = self.cases.values()
		if tenant_id is not None:
			cases = [case for case in cases if case.tenant_id == tenant_id]
		return [case.to_dict() for case in sorted(cases, key=lambda item: item.id)]

	def _tenant_signal_or_none(self, signal_id: str, tenant_id: str) -> FraudSignal | None:
		signal = self.signals.get(signal_id)
		if signal is None or signal.tenant_id != tenant_id:
			return None
		return signal

	def _tenant_case(self, case_id: str, tenant_id: str) -> FraudCase:
		case = self.cases.get(case_id)
		if case is None or case.tenant_id != tenant_id:
			raise KeyError(f"unknown fraud case: {case_id}")
		return case

	def _record_evidence(self, evidence_id: str, tenant_id: str, kind: str, reference_id: str, status: str, metadata: dict[str, Any]) -> dict[str, Any]:
		evidence = FraudEvidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
		self.evidence[evidence_id] = evidence
		return evidence.to_dict()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "fraud_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "fraud_policy_denied")


FintechFraudService = FraudDetectionService
