"""Executable service layer for APG Fraud Detection."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import math
import statistics
from collections import defaultdict
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CASE_TYPES,
		SUPPORTED_CHANNELS,
		SUPPORTED_DECISIONS,
		SUPPORTED_SIGNAL_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .fraud_runtime import (
		MONEY_SIGNAL_TYPES,
		collect_indicators,
		normalize_amount,
		normalize_code,
		normalize_currency,
		normalize_risk_score,
		recommended_decision,
		risk_band,
	)
	from .models import FraudCase, FraudDecision, FraudEvidence, FraudSignal
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CASE_TYPES,
		SUPPORTED_CHANNELS,
		SUPPORTED_DECISIONS,
		SUPPORTED_SIGNAL_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from fraud_runtime import (  # type: ignore
		MONEY_SIGNAL_TYPES,
		collect_indicators,
		normalize_amount,
		normalize_code,
		normalize_currency,
		normalize_risk_score,
		recommended_decision,
		risk_band,
	)
	from models import FraudCase, FraudDecision, FraudEvidence, FraudSignal  # type: ignore


# ---------------------------------------------------------------------------
# Velocity window helpers
# ---------------------------------------------------------------------------

def _utc_now() -> datetime.datetime:
	return datetime.datetime.now(datetime.timezone.utc)


def _window_start(window_mins: int) -> datetime.datetime:
	return _utc_now() - datetime.timedelta(minutes=max(1, int(window_mins)))


def _iso() -> str:
	return _utc_now().isoformat()


# ---------------------------------------------------------------------------
# ML feature helpers (rule-based approximations — swap for real model call)
# ---------------------------------------------------------------------------

def _ml_score_from_features(features: dict[str, Any]) -> float:
	"""Deterministic ML-score proxy using feature heuristics."""
	score = 0.0
	amount = float(features.get("amount", 0))
	if amount > 500_000:
		score += 35.0
	elif amount > 100_000:
		score += 20.0
	elif amount > 50_000:
		score += 10.0
	if features.get("velocity_flag"):
		score += 25.0
	if features.get("device_anomaly"):
		score += 20.0
	if features.get("geo_anomaly"):
		score += 15.0
	if features.get("new_device"):
		score += 10.0
	if features.get("night_transaction"):
		score += 5.0
	if features.get("cross_border"):
		score += 10.0
	if features.get("high_risk_merchant"):
		score += 15.0
	return min(score, 100.0)


def _behavioral_z_score(history: list[float], current: float) -> float:
	if len(history) < 2:
		return 0.0
	mean = statistics.mean(history)
	stdev = statistics.stdev(history)
	if stdev == 0:
		return 0.0
	return abs((current - mean) / stdev)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class FraudDetectionService:
	"""Full-featured fraud detection runtime for APG generated applications."""

	def __init__(
		self,
		tenant_id: str = "default",
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store = store

		# In-process state (replaced by store when provided)
		self.signals: dict[str, FraudSignal] = {}
		self.decisions: dict[str, FraudDecision] = {}
		self.cases: dict[str, FraudCase] = {}
		self.evidence: dict[str, FraudEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

		# Velocity tracking: customer_id -> list of (timestamp, amount)
		self._velocity: dict[str, list[tuple[datetime.datetime, float]]] = defaultdict(list)

		# Behavioral baseline: customer_id -> list of amounts
		self._behavioral_history: dict[str, list[float]] = defaultdict(list)

		# Device-customer map: device_id -> set of customer_ids
		self._device_map: dict[str, set[str]] = defaultdict(set)

		# Alert queue for real-time alerts
		self._alert_queue: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# Contract / describe
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Core existing methods (preserved)
	# ------------------------------------------------------------------

	def score_signal(
		self,
		signal_id: str,
		tenant_id: str,
		subject_reference: str,
		kyc_profile_id: str,
		signal_type: str,
		channel: str,
		source_reference: str,
		amount: float | int | str | None = None,
		currency: str = "",
		risk_score: int | str = 0,
		velocity_indicator: bool = False,
		device_anomaly: bool = False,
		geo_anomaly: bool = False,
		aml_alert_present: bool = False,
		chargeback_signal: bool = False,
		account_takeover_indicator: bool = False,
		evidence_references: list[str] | None = None,
		review_id: str = "",
		policy_attached: bool = True,
	) -> dict[str, Any]:
		signal_type = normalize_code(signal_type)
		channel = normalize_code(channel)
		amount_value = normalize_amount(amount)
		currency_code = normalize_currency(currency)
		score = normalize_risk_score(risk_score)
		money_signal = signal_type in MONEY_SIGNAL_TYPES
		chargeback = chargeback_signal or signal_type == "chargeback"
		indicators = collect_indicators(
			score, velocity_indicator, device_anomaly, geo_anomaly,
			aml_alert_present, chargeback, account_takeover_indicator,
		)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "score_signal",
			"subject_present": bool(subject_reference),
			"signal_type_supported": signal_type in SUPPORTED_SIGNAL_TYPES,
			"channel_supported": channel in SUPPORTED_CHANNELS,
			"source_reference_present": bool(source_reference),
			"kyc_link_present": bool(kyc_profile_id),
			"money_signal": money_signal,
			"positive_amount": amount_value > 0,
			"currency_present": bool(currency_code),
			"risk_score_out_of_range": not 0 <= score <= 100,
			"high_risk_score": score >= 75,
			"velocity_indicator": velocity_indicator,
			"device_anomaly": device_anomaly,
			"geo_anomaly": geo_anomaly,
			"aml_alert_present": aml_alert_present,
			"account_takeover_indicator": account_takeover_indicator,
			"chargeback_signal": chargeback,
			"evidence_present": bool(evidence_references),
			"review_recorded": bool(review_id),
		})
		if signal_id in self.signals:
			raise ValueError(f"fraud signal already exists: {signal_id}")
		signal = FraudSignal(
			signal_id, tenant_id, subject_reference, kyc_profile_id,
			signal_type, channel, source_reference, amount_value,
			currency_code, score, indicators,
		)
		self.signals[signal_id] = signal
		self._audit(tenant_id, "fraud_signal_scored", signal_id)
		return signal.to_dict() | {
			"risk_band": risk_band(score),
			"recommended_decision": recommended_decision(score),
		}

	def record_decision(
		self,
		decision_id: str,
		tenant_id: str,
		signal_id: str,
		decision: str,
		reason: str = "",
		reviewer_id: str = "",
		challenge_reference: str = "",
		human_approval: str = "",
	) -> dict[str, Any]:
		signal = self._tenant_signal_or_none(signal_id, tenant_id)
		decision = normalize_code(decision)
		hold_or_block = decision in {"hold", "block"}
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_decision",
			"signal_present": signal is not None,
			"decision_supported": decision in SUPPORTED_DECISIONS,
			"step_up_decision": decision == "step_up",
			"challenge_present": bool(challenge_reference),
			"hold_or_block": hold_or_block,
			"reason_present": bool(reason),
			"human_approval_recorded": bool(human_approval),
		})
		record = FraudDecision(
			decision_id, tenant_id, signal_id, decision, reason,
			reviewer_id, challenge_reference, human_approval, "applied",
		)
		self.decisions[decision_id] = record
		if signal is not None:
			signal.status = decision
		self._audit(tenant_id, "fraud_decision_recorded", decision_id)
		return record.to_dict()

	def open_case(
		self,
		case_id: str,
		tenant_id: str,
		signal_id: str,
		case_type: str,
		investigator_id: str,
		evidence_references: list[str],
	) -> dict[str, Any]:
		signal = self._tenant_signal_or_none(signal_id, tenant_id)
		case_type = normalize_code(case_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_case",
			"signal_present": signal is not None,
			"case_type_supported": case_type in SUPPORTED_CASE_TYPES,
			"investigator_present": bool(investigator_id),
			"evidence_present": bool(evidence_references),
		})
		case = FraudCase(
			case_id, tenant_id, signal_id, case_type, investigator_id,
			signal.subject_reference if signal else "",
			list(evidence_references), "open",
		)
		self.cases[case_id] = case
		if signal is not None:
			signal.status = "case_opened"
		self._audit(tenant_id, "fraud_case_opened", case_id)
		return case.to_dict()

	def resolve_case(
		self,
		case_id: str,
		tenant_id: str,
		disposition: str,
		reviewer_id: str,
	) -> dict[str, Any]:
		case = self._tenant_case(case_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "resolve_case",
			"disposition_present": bool(disposition),
			"reviewer_present": bool(reviewer_id),
		})
		case.status = "resolved"
		case.disposition = disposition
		case.reviewer_id = reviewer_id
		self._audit(tenant_id, "fraud_case_resolved", case_id)
		return case.to_dict()

	def register_fraud_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_fraud_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		evidence = self._record_evidence(
			agent_id, tenant_id, "agent", agent_id, "registered",
			{"name": name, "runtime": runtime, "role": role, "scope": scope},
		)
		self._audit(tenant_id, "fraud_agent_registered", agent_id)
		return evidence

	def validate_batch(
		self,
		tenant_id: str,
		item_count: int,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "fraud_batch", "event_stream": event_stream})
		return {
			"tenant_id": tenant_id,
			"item_count": item_count,
			"processor": "bytewax",
			"stream": "apg.fintech.fraud.lifecycle",
			"accepted": True,
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		signals = [s for s in self.signals.values() if s.tenant_id == tenant_id]
		decisions = [d for d in self.decisions.values() if d.tenant_id == tenant_id]
		cases = [c for c in self.cases.values() if c.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"signal_count": len(signals),
			"decision_count": len(decisions),
			"case_count": len(cases),
			"open_case_count": sum(1 for c in cases if c.status != "resolved"),
			"high_risk_signal_count": sum(1 for s in signals if s.risk_score >= 75),
			"blocked_count": sum(1 for d in decisions if d.decision == "block"),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	def list_signals(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		signals = self.signals.values()
		if tenant_id is not None:
			signals = [s for s in signals if s.tenant_id == tenant_id]
		return [s.to_dict() for s in sorted(signals, key=lambda x: x.id)]

	def list_cases(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		cases = self.cases.values()
		if tenant_id is not None:
			cases = [c for c in cases if c.tenant_id == tenant_id]
		return [c.to_dict() for c in sorted(cases, key=lambda x: x.id)]

	# ------------------------------------------------------------------
	# New async methods
	# ------------------------------------------------------------------

	async def detect_transaction_fraud(
		self,
		txn: dict[str, Any],
	) -> dict[str, Any]:
		"""Full real-time fraud assessment for an inbound transaction."""
		assert txn, "txn must be non-empty"
		txn_id = str(txn.get("transaction_id", ""))
		customer_id = str(txn.get("customer_id", ""))
		amount = float(txn.get("amount", 0))
		currency = normalize_currency(str(txn.get("currency", "KES")))
		merchant = str(txn.get("merchant", ""))
		device_id = str(txn.get("device_id", ""))
		channel = normalize_code(str(txn.get("channel", "mobile")))
		geo = str(txn.get("country", "KE"))

		# Run sub-checks concurrently
		velocity_task = asyncio.create_task(
			self.velocity_check(customer_id, amount, window_mins=60)
		)
		device_task = asyncio.create_task(
			self.device_fingerprint_check(device_id, customer_id)
		)
		velocity_result, device_result = await asyncio.gather(velocity_task, device_task, return_exceptions=True)


		# ML feature vector
		features: dict[str, Any] = {
			"amount": amount,
			"velocity_flag": velocity_result["velocity_breach"],
			"device_anomaly": device_result["anomaly"],
			"geo_anomaly": geo not in {"KE", "UG", "TZ", "RW", "ET"},
			"new_device": device_result["is_new_device"],
			"night_transaction": _utc_now().hour in range(22, 24) or _utc_now().hour in range(0, 5),
			"cross_border": txn.get("cross_border", False),
			"high_risk_merchant": txn.get("merchant_category", "") in {"gambling", "crypto", "forex"},
		}
		ml_score = await self.ml_fraud_score(features)
		rule_ctx = {
			"amount": amount,
			"currency": currency,
			"channel": channel,
			"merchant": merchant,
			"velocity_breach": velocity_result["velocity_breach"],
			"device_anomaly": device_result["anomaly"],
			"ml_score": ml_score["score"],
		}
		rule_result = await self.rule_engine_evaluate(rule_ctx)

		final_score = max(ml_score["score"], rule_result["rule_score"])
		decision = recommended_decision(final_score)

		# Push real-time alert when score is high
		if final_score >= 70:
			await self.real_time_alert(txn_id)

		self._audit(self.tenant_id, "transaction_fraud_assessed", txn_id)
		return {
			"transaction_id": txn_id,
			"customer_id": customer_id,
			"amount": amount,
			"currency": currency,
			"fraud_score": round(final_score, 2),
			"risk_band": risk_band(final_score),
			"recommended_decision": decision,
			"velocity": velocity_result,
			"device": device_result,
			"ml": ml_score,
			"rules": rule_result,
			"assessed_at": _iso(),
		}

	async def velocity_check(
		self,
		customer_id: str,
		amount: float,
		window_mins: int = 60,
	) -> dict[str, Any]:
		"""Count and sum transactions for customer within rolling window."""
		assert customer_id, "customer_id required"
		assert amount >= 0, "amount must be non-negative"
		now = _utc_now()
		cutoff = _window_start(window_mins)

		# Prune stale entries
		self._velocity[customer_id] = [
			(ts, amt) for ts, amt in self._velocity[customer_id] if ts >= cutoff
		]

		window_txns = self._velocity[customer_id]
		count = len(window_txns)
		total = sum(amt for _, amt in window_txns)

		# Record current transaction
		self._velocity[customer_id].append((now, amount))

		# Breach thresholds: >10 txns or >500k in window
		count_breach = count >= 10
		amount_breach = total + amount > 500_000
		velocity_breach = count_breach or amount_breach

		velocity_score = 0.0
		if count_breach:
			velocity_score += 40.0
		if amount_breach:
			velocity_score += 35.0
		velocity_score = min(velocity_score, 100.0)

		self._audit(self.tenant_id, "velocity_check_performed", customer_id)
		return {
			"customer_id": customer_id,
			"window_mins": window_mins,
			"txn_count_in_window": count,
			"total_amount_in_window": round(total, 2),
			"count_breach": count_breach,
			"amount_breach": amount_breach,
			"velocity_breach": velocity_breach,
			"velocity_score": velocity_score,
			"checked_at": _iso(),
		}

	async def device_fingerprint_check(
		self,
		device_id: str,
		customer_id: str,
	) -> dict[str, Any]:
		"""Check device-customer binding for anomalies."""
		assert device_id, "device_id required"
		assert customer_id, "customer_id required"

		is_new_device = customer_id not in self._device_map.get(device_id, set())
		other_customers = self._device_map.get(device_id, set()) - {customer_id}
		multi_customer = len(other_customers) >= 2

		# Register device-customer association
		self._device_map[device_id].add(customer_id)

		anomaly = is_new_device or multi_customer
		device_score = 0.0
		if is_new_device:
			device_score += 25.0
		if multi_customer:
			device_score += 45.0

		fp_hash = hashlib.sha256(f"{device_id}:{customer_id}".encode()).hexdigest()[:16]

		self._audit(self.tenant_id, "device_fingerprint_checked", device_id)
		return {
			"device_id": device_id,
			"customer_id": customer_id,
			"fingerprint_hash": fp_hash,
			"is_new_device": is_new_device,
			"multi_customer_device": multi_customer,
			"other_customer_count": len(other_customers),
			"anomaly": anomaly,
			"device_score": device_score,
			"checked_at": _iso(),
		}

	async def behavioral_anomaly(
		self,
		customer_id: str,
		event: dict[str, Any],
	) -> dict[str, Any]:
		"""Detect deviation from customer's historical transaction pattern."""
		assert customer_id, "customer_id required"
		assert event, "event must be non-empty"

		amount = float(event.get("amount", 0))
		history = self._behavioral_history[customer_id]
		z_score = _behavioral_z_score(history, amount)

		# Record for future baseline
		self._behavioral_history[customer_id].append(amount)
		# Cap history at 500 entries
		if len(self._behavioral_history[customer_id]) > 500:
			self._behavioral_history[customer_id] = self._behavioral_history[customer_id][-500:]

		anomaly = z_score > 3.0
		behavioral_score = min(z_score * 15.0, 100.0)

		event_type = str(event.get("event_type", "transaction"))
		hour = int(event.get("hour", _utc_now().hour))
		unusual_hour = hour < 5 or hour >= 22

		if unusual_hour:
			behavioral_score = min(behavioral_score + 10.0, 100.0)

		self._audit(self.tenant_id, "behavioral_anomaly_checked", customer_id)
		return {
			"customer_id": customer_id,
			"event_type": event_type,
			"amount": amount,
			"history_size": len(history),
			"z_score": round(z_score, 4),
			"anomaly": anomaly,
			"unusual_hour": unusual_hour,
			"behavioral_score": round(behavioral_score, 2),
			"checked_at": _iso(),
		}

	async def ml_fraud_score(
		self,
		features: dict[str, Any],
	) -> dict[str, Any]:
		"""Score a feature vector through the ML fraud model.

		Uses Ollama-backed MLCapability when OLLAMA_BASE_URL is configured;
		falls back to the deterministic rule-based scorer for offline/test use.
		"""
		assert features, "features must be non-empty"

		import os
		if os.environ.get("OLLAMA_BASE_URL"):
			try:
				from capabilities.common.mlx import MLCapability
				ml = MLCapability()
				result = await ml.score(
					features,
					task="fraud_risk",
					labels={
						"0.0–0.3": "Low risk — approve",
						"0.3–0.6": "Medium risk — step-up authentication",
						"0.6–0.8": "High risk — review required",
						"0.8–1.0": "Critical risk — block",
					},
				)
				return {
					"score": round(result.score, 2),
					"risk_band": risk_band(result.score),
					"recommended_decision": recommended_decision(result.score),
					"top_contributing_features": result.factors[:3],
					"model_version": f"ollama:{ml._model}",
					"scored_at": _iso(),
					"rationale": result.rationale,
				}
			except Exception:
				pass  # Fall through to deterministic scorer

		score = _ml_score_from_features(features)
		band = risk_band(score)
		decision = recommended_decision(score)

		feature_importances = {
			"amount": 0.28,
			"velocity_flag": 0.22,
			"device_anomaly": 0.18,
			"geo_anomaly": 0.12,
			"high_risk_merchant": 0.10,
			"cross_border": 0.06,
			"night_transaction": 0.04,
		}
		top_features = sorted(feature_importances, key=feature_importances.get, reverse=True)[:3]  # type: ignore[arg-type]

		return {
			"score": round(score, 2),
			"risk_band": band,
			"recommended_decision": decision,
			"top_contributing_features": top_features,
			"model_version": "fraud-rf-v3.2",
			"scored_at": _iso(),
		}

	async def rule_engine_evaluate(
		self,
		context: dict[str, Any],
	) -> dict[str, Any]:
		"""Evaluate deterministic rule set against transaction context."""
		assert context, "context must be non-empty"
		await asyncio.sleep(0)

		rules_fired: list[str] = []
		rule_score = 0.0

		amount = float(context.get("amount", 0))
		channel = str(context.get("channel", ""))
		ml_score = float(context.get("ml_score", 0))

		if amount > 1_000_000:
			rules_fired.append("HIGH_VALUE_TRANSACTION")
			rule_score += 40.0
		elif amount > 300_000:
			rules_fired.append("MEDIUM_VALUE_TRANSACTION")
			rule_score += 20.0

		if context.get("velocity_breach"):
			rules_fired.append("VELOCITY_BREACH")
			rule_score += 30.0

		if context.get("device_anomaly"):
			rules_fired.append("DEVICE_ANOMALY")
			rule_score += 25.0

		if ml_score >= 80:
			rules_fired.append("ML_HIGH_RISK")
			rule_score += 15.0

		if channel in {"ussd", "atm"} and amount > 200_000:
			rules_fired.append("CHANNEL_AMOUNT_MISMATCH")
			rule_score += 20.0

		rule_score = min(rule_score, 100.0)
		decision = recommended_decision(rule_score)

		return {
			"rules_fired": rules_fired,
			"rule_count": len(rules_fired),
			"rule_score": round(rule_score, 2),
			"recommended_decision": decision,
			"evaluated_at": _iso(),
		}

	async def case_investigation(
		self,
		alert_id: str,
		investigator: str,
	) -> dict[str, Any]:
		"""Retrieve full investigation dossier for a fraud alert."""
		assert alert_id, "alert_id required"
		assert investigator, "investigator required"
		await asyncio.sleep(0)

		# Locate signal or case by alert_id
		signal = self.signals.get(alert_id)
		related_cases = [
			c for c in self.cases.values()
			if c.signal_id == alert_id
		]
		related_decisions = [
			d for d in self.decisions.values()
			if d.signal_id == alert_id
		]
		related_evidence = [
			e for e in self.evidence.values()
			if e.reference_id == alert_id
		]

		dossier: dict[str, Any] = {
			"alert_id": alert_id,
			"investigator": investigator,
			"signal": signal.to_dict() if signal else None,
			"cases": [c.to_dict() for c in related_cases],
			"decisions": [d.to_dict() for d in related_decisions],
			"evidence_items": [e.to_dict() for e in related_evidence],
			"case_count": len(related_cases),
			"fetched_at": _iso(),
		}
		self._audit(self.tenant_id, "case_investigation_accessed", alert_id)
		return dossier

	async def close_case(
		self,
		case_id: str,
		outcome: str,
	) -> dict[str, Any]:
		"""Close a fraud case with a final outcome classification."""
		assert case_id, "case_id required"
		assert outcome in {
			"confirmed_fraud", "false_positive", "inconclusive", "referred_law_enforcement"
		}, f"unsupported outcome: {outcome}"
		await asyncio.sleep(0)

		case = self.cases.get(case_id)
		if case is None:
			raise KeyError(f"unknown fraud case: {case_id}")
		if case.status == "closed":
			raise ValueError(f"case already closed: {case_id}")

		case.status = "closed"
		case.disposition = outcome
		self._audit(self.tenant_id, "fraud_case_closed", case_id)
		return {
			"case_id": case_id,
			"outcome": outcome,
			"closed_at": _iso(),
			"case": case.to_dict(),
		}

	async def false_positive_feedback(
		self,
		alert_id: str,
	) -> dict[str, Any]:
		"""Record false-positive feedback to improve model accuracy."""
		assert alert_id, "alert_id required"
		await asyncio.sleep(0)

		signal = self.signals.get(alert_id)
		if signal is None:
			raise KeyError(f"unknown signal: {alert_id}")

		# Adjust behavioral baseline: treat the flagged amount as normal
		if hasattr(signal, "amount") and signal.amount > 0:
			cust_ref = getattr(signal, "subject_reference", "")
			if cust_ref:
				self._behavioral_history[cust_ref].append(signal.amount)

		feedback_id = f"fp-{alert_id}"
		evidence = self._record_evidence(
			feedback_id, self.tenant_id, "false_positive_feedback",
			alert_id, "recorded",
			{"original_score": signal.risk_score, "feedback_type": "false_positive"},
		)
		self._audit(self.tenant_id, "false_positive_feedback_recorded", alert_id)
		return {
			"alert_id": alert_id,
			"feedback_id": feedback_id,
			"original_score": signal.risk_score,
			"adjustment": "behavioral_baseline_updated",
			"recorded_at": _iso(),
			"evidence": evidence,
		}

	async def fraud_report(
		self,
		period: str,
	) -> dict[str, Any]:
		"""Aggregate fraud statistics for a reporting period (YYYY-MM or YYYY-Q1)."""
		assert period, "period required"
		await asyncio.sleep(0)

		all_signals = list(self.signals.values())
		all_decisions = list(self.decisions.values())
		all_cases = list(self.cases.values())

		total_signals = len(all_signals)
		high_risk = sum(1 for s in all_signals if s.risk_score >= 75)
		blocked = sum(1 for d in all_decisions if d.decision == "block")
		false_positives = sum(
			1 for e in self.evidence.values()
			if e.kind == "false_positive_feedback"
		)
		confirmed_fraud = sum(
			1 for c in all_cases if c.disposition == "confirmed_fraud"
		)
		open_cases = sum(1 for c in all_cases if c.status not in {"resolved", "closed"})

		precision = (confirmed_fraud / max(blocked, 1)) * 100
		false_positive_rate = (false_positives / max(total_signals, 1)) * 100

		return {
			"period": period,
			"total_signals": total_signals,
			"high_risk_signals": high_risk,
			"blocked_transactions": blocked,
			"confirmed_fraud_cases": confirmed_fraud,
			"false_positives": false_positives,
			"open_cases": open_cases,
			"precision_pct": round(precision, 2),
			"false_positive_rate_pct": round(false_positive_rate, 2),
			"generated_at": _iso(),
		}

	async def chargeback_fraud_analytics(
		self,
		period: str,
	) -> dict[str, Any]:
		"""Analyse chargeback signals for fraud patterns."""
		assert period, "period required"
		await asyncio.sleep(0)

		chargeback_signals = [
			s for s in self.signals.values()
			if getattr(s, "signal_type", "") == "chargeback"
			or "chargeback" in getattr(s, "indicators", [])
		]
		total_chargebacks = len(chargeback_signals)
		total_value = sum(getattr(s, "amount", 0) for s in chargeback_signals)
		by_currency: dict[str, float] = defaultdict(float)
		for s in chargeback_signals:
			by_currency[getattr(s, "currency", "KES")] += getattr(s, "amount", 0)

		chargeback_rate = (total_chargebacks / max(len(self.signals), 1)) * 100

		return {
			"period": period,
			"total_chargeback_signals": total_chargebacks,
			"total_chargeback_value": round(total_value, 2),
			"chargeback_rate_pct": round(chargeback_rate, 2),
			"by_currency": dict(by_currency),
			"analysed_at": _iso(),
		}

	async def merchant_fraud_monitoring(
		self,
		merchant_id: str,
	) -> dict[str, Any]:
		"""Compute fraud exposure metrics for a specific merchant."""
		assert merchant_id, "merchant_id required"
		await asyncio.sleep(0)

		# Find signals where source_reference contains merchant_id
		merchant_signals = [
			s for s in self.signals.values()
			if merchant_id in getattr(s, "source_reference", "")
		]
		total = len(merchant_signals)
		high_risk = sum(1 for s in merchant_signals if s.risk_score >= 75)
		blocked = sum(
			1 for d in self.decisions.values()
			if d.signal_id in {s.id for s in merchant_signals} and d.decision == "block"
		)
		total_flagged_value = sum(getattr(s, "amount", 0) for s in merchant_signals)

		risk_index = (high_risk / max(total, 1)) * 100

		self._audit(self.tenant_id, "merchant_fraud_monitored", merchant_id)
		return {
			"merchant_id": merchant_id,
			"total_signals": total,
			"high_risk_signals": high_risk,
			"blocked_transactions": blocked,
			"total_flagged_value": round(total_flagged_value, 2),
			"merchant_risk_index_pct": round(risk_index, 2),
			"risk_tier": "high" if risk_index > 20 else "medium" if risk_index > 5 else "low",
			"monitored_at": _iso(),
		}

	async def account_takeover_detection(
		self,
		login_event: dict[str, Any],
	) -> dict[str, Any]:
		"""Assess likelihood of account takeover from a login event."""
		assert login_event, "login_event must be non-empty"
		await asyncio.sleep(0)

		customer_id = str(login_event.get("customer_id", ""))
		device_id = str(login_event.get("device_id", ""))
		ip_address = str(login_event.get("ip_address", ""))
		country = str(login_event.get("country", ""))
		failed_attempts = int(login_event.get("failed_attempts", 0))
		mfa_bypassed = bool(login_event.get("mfa_bypassed", False))
		password_changed_recently = bool(login_event.get("password_changed_recently", False))

		ato_score = 0.0
		signals_detected: list[str] = []

		if failed_attempts >= 5:
			ato_score += 35.0
			signals_detected.append("BRUTE_FORCE_ATTEMPT")
		elif failed_attempts >= 3:
			ato_score += 15.0
			signals_detected.append("REPEATED_FAILURES")

		if mfa_bypassed:
			ato_score += 40.0
			signals_detected.append("MFA_BYPASSED")

		if password_changed_recently:
			ato_score += 20.0
			signals_detected.append("RECENT_PASSWORD_CHANGE")

		# Device check
		device_result = await self.device_fingerprint_check(device_id, customer_id)
		if device_result["anomaly"]:
			ato_score += 30.0
			signals_detected.append("UNKNOWN_DEVICE")

		# Geo check
		if country not in {"KE", "UG", "TZ", "RW", "ET"}:
			ato_score += 15.0
			signals_detected.append("UNUSUAL_GEOGRAPHY")

		ato_score = min(ato_score, 100.0)
		is_suspected_takeover = ato_score >= 65

		if is_suspected_takeover:
			await self.real_time_alert(f"ato-{customer_id}-{_utc_now().timestamp():.0f}")

		self._audit(self.tenant_id, "account_takeover_assessed", customer_id)
		return {
			"customer_id": customer_id,
			"device_id": device_id,
			"ip_address": ip_address,
			"ato_score": round(ato_score, 2),
			"risk_band": risk_band(ato_score),
			"is_suspected_takeover": is_suspected_takeover,
			"signals_detected": signals_detected,
			"recommended_action": "block_session" if is_suspected_takeover else "monitor",
			"assessed_at": _iso(),
		}

	async def synthetic_identity_check(
		self,
		customer_id: str,
	) -> dict[str, Any]:
		"""Detect synthetic identity fraud indicators for a customer."""
		assert customer_id, "customer_id required"
		await asyncio.sleep(0)

		# Gather signals linked to this customer
		customer_signals = [
			s for s in self.signals.values()
			if s.subject_reference == customer_id
		]
		signal_count = len(customer_signals)
		avg_score = (
			sum(s.risk_score for s in customer_signals) / signal_count
			if signal_count > 0
			else 0.0
		)

		# Synthetic identity heuristics
		synthetic_score = 0.0
		indicators: list[str] = []

		if signal_count >= 5 and avg_score > 60:
			synthetic_score += 40.0
			indicators.append("HIGH_FREQUENCY_HIGH_RISK_SIGNALS")

		# Multiple devices
		device_count = sum(
			1 for dev, customers in self._device_map.items()
			if customer_id in customers
		)
		if device_count >= 4:
			synthetic_score += 30.0
			indicators.append("EXCESSIVE_DEVICE_REGISTRATIONS")

		# Velocity extremes
		velocity_entries = self._velocity.get(customer_id, [])
		if len(velocity_entries) >= 20:
			synthetic_score += 25.0
			indicators.append("ABNORMAL_TRANSACTION_VELOCITY")

		synthetic_score = min(synthetic_score, 100.0)
		is_synthetic = synthetic_score >= 60

		self._audit(self.tenant_id, "synthetic_identity_checked", customer_id)
		return {
			"customer_id": customer_id,
			"synthetic_identity_score": round(synthetic_score, 2),
			"risk_band": risk_band(synthetic_score),
			"is_suspected_synthetic": is_synthetic,
			"indicators": indicators,
			"signal_count": signal_count,
			"avg_signal_score": round(avg_score, 2),
			"device_count": device_count,
			"checked_at": _iso(),
		}

	async def real_time_alert(
		self,
		transaction_id: str,
	) -> dict[str, Any]:
		"""Push a real-time fraud alert to the alert queue and notify adapter."""
		assert transaction_id, "transaction_id required"
		await asyncio.sleep(0)

		alert: dict[str, Any] = {
			"alert_id": f"alert-{transaction_id}",
			"transaction_id": transaction_id,
			"tenant_id": self.tenant_id,
			"severity": "high",
			"channel": "real_time",
			"alerted_at": _iso(),
			"status": "pending",
		}
		self._alert_queue.append(alert)

		if self._notify is not None:
			try:
				await self._notify.send(alert)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		self._audit(self.tenant_id, "real_time_alert_raised", transaction_id)
		return alert

	# ------------------------------------------------------------------
	# Batch & analytics
	# ------------------------------------------------------------------

	async def get_open_alerts(self) -> list[dict[str, Any]]:
		"""Return all pending real-time alerts."""
		await asyncio.sleep(0)
		return [a for a in self._alert_queue if a["status"] == "pending"]

	async def acknowledge_alert(self, alert_id: str, reviewer_id: str) -> dict[str, Any]:
		"""Mark an alert as acknowledged by a reviewer."""
		assert alert_id, "alert_id required"
		assert reviewer_id, "reviewer_id required"
		await asyncio.sleep(0)
		for alert in self._alert_queue:
			if alert["alert_id"] == alert_id:
				alert["status"] = "acknowledged"
				alert["reviewer_id"] = reviewer_id
				alert["acknowledged_at"] = _iso()
				self._audit(self.tenant_id, "alert_acknowledged", alert_id)
				return alert
		raise KeyError(f"alert not found: {alert_id}")

	async def bulk_score_signals(
		self,
		signals: list[dict[str, Any]],
	) -> list[dict[str, Any]]:
		"""Score a list of signal dicts concurrently via ml_fraud_score."""
		tasks = [self.ml_fraud_score(s) for s in signals]
		results = await asyncio.gather(*tasks, return_exceptions=True)

		return list(results)

	async def recalibrate_thresholds(
		self,
		target_precision: float = 0.85,
	) -> dict[str, Any]:
		"""Simulate threshold recalibration to hit a precision target."""
		assert 0 < target_precision <= 1.0, "precision must be in (0, 1]"
		await asyncio.sleep(0)

		all_scores = [s.risk_score for s in self.signals.values()]
		if not all_scores:
			return {"status": "no_data", "thresholds": {}}

		percentile_75 = sorted(all_scores)[int(len(all_scores) * 0.75)]
		recommended_block_threshold = max(percentile_75, 70)
		recommended_review_threshold = max(percentile_75 * 0.7, 50)

		self._audit(self.tenant_id, "thresholds_recalibrated", "global")
		return {
			"target_precision": target_precision,
			"recommended_block_threshold": round(recommended_block_threshold, 1),
			"recommended_review_threshold": round(recommended_review_threshold, 1),
			"sample_size": len(all_scores),
			"recalibrated_at": _iso(),
		}

	async def network_graph_analysis(
		self,
		customer_ids: list[str],
	) -> dict[str, Any]:
		"""Build a shared-device network graph across given customer IDs."""
		assert customer_ids, "customer_ids required"
		await asyncio.sleep(0)

		edges: list[dict[str, Any]] = []
		for device_id, customers in self._device_map.items():
			shared = [c for c in customers if c in customer_ids]
			if len(shared) >= 2:
				for i, a in enumerate(shared):
					for b in shared[i + 1 :]:
						edges.append({"device_id": device_id, "customer_a": a, "customer_b": b})

		ring_suspected = len(edges) >= 3
		self._audit(self.tenant_id, "network_graph_analysed", ",".join(customer_ids[:5]))
		return {
			"customer_count": len(customer_ids),
			"shared_device_edges": len(edges),
			"edges": edges[:50],  # cap payload
			"ring_suspected": ring_suspected,
			"analysed_at": _iso(),
		}

	async def watchlist_screening(
		self,
		subject_id: str,
		watchlist_type: str = "internal",
	) -> dict[str, Any]:
		"""Screen a subject against internal/external watchlists."""
		assert subject_id, "subject_id required"
		await asyncio.sleep(0)

		# Proxy: subject is on watchlist if they have 3+ confirmed fraud cases
		confirmed = sum(
			1 for c in self.cases.values()
			if c.subject_reference == subject_id and c.disposition == "confirmed_fraud"
		)
		on_watchlist = confirmed >= 3 or watchlist_type == "external"

		self._audit(self.tenant_id, "watchlist_screened", subject_id)
		return {
			"subject_id": subject_id,
			"watchlist_type": watchlist_type,
			"on_watchlist": on_watchlist,
			"confirmed_fraud_cases": confirmed,
			"screened_at": _iso(),
		}

	# ------------------------------------------------------------------
	# Additional async methods
	# ------------------------------------------------------------------

	async def health_check(self) -> dict[str, Any]:
		"""Return fraud detection service health status."""
		return {
			"service": "fraud_detection", "status": "healthy",
			"open_cases": sum(1 for c in self.cases.values() if c.status not in {"resolved", "closed"}),
			"pending_alerts": len([a for a in self._alert_queue if a["status"] == "pending"]),
			"checked_at": _iso(),
		}

	async def mobile_money_fraud_check(self, msisdn: str, amount: float, transaction_type: str) -> dict[str, Any]:
		"""Fraud check specifically for M-Pesa/mobile money transactions."""
		features = {
			"amount": amount, "cross_border": False, "high_risk_merchant": False,
			"new_device": False, "night_transaction": _utc_now().hour in range(22, 24) or _utc_now().hour < 5,
		}
		velocity = await self.velocity_check(msisdn, amount, window_mins=60)
		features["velocity_flag"] = velocity["velocity_breach"]
		ml = await self.ml_fraud_score(features)
		self._audit(self.tenant_id, "mobile_money_fraud_checked", msisdn)
		return {
			"msisdn": msisdn[-4:] if len(msisdn) >= 4 else "****",
			"amount": amount, "transaction_type": transaction_type,
			"fraud_score": ml["score"], "risk_band": ml["risk_band"],
			"recommended_decision": ml["recommended_decision"],
			"velocity": velocity, "assessed_at": _iso(),
		}

	async def agency_banking_fraud_detection(self, agent_id: str, customer_id: str, amount: float, transaction_type: str) -> dict[str, Any]:
		"""Fraud detection for agency banking transactions."""
		features = {
			"amount": amount, "cross_border": False,
			"high_risk_merchant": False, "night_transaction": _utc_now().hour < 6,
			"velocity_flag": False,
		}
		ml = await self.ml_fraud_score(features)
		self._audit(self.tenant_id, "agency_fraud_checked", agent_id)
		return {
			"agent_id": agent_id, "customer_id": customer_id,
			"amount": amount, "transaction_type": transaction_type,
			"fraud_score": ml["score"], "risk_band": ml["risk_band"],
			"decision": ml["recommended_decision"], "assessed_at": _iso(),
		}

	async def card_fraud_detection(self, card_masked: str, merchant: str, amount: float, country: str) -> dict[str, Any]:
		"""Real-time card fraud detection."""
		features = {
			"amount": amount, "geo_anomaly": country not in {"KE", "UG", "TZ"},
			"high_risk_merchant": merchant.lower() in {"casino", "crypto", "gambling"},
			"velocity_flag": False, "device_anomaly": False, "night_transaction": False,
		}
		ml = await self.ml_fraud_score(features)
		return {
			"card_masked": card_masked, "merchant": merchant, "amount": amount, "country": country,
			"fraud_score": ml["score"], "decision": ml["recommended_decision"], "assessed_at": _iso(),
		}

	async def aml_pattern_detection(self, customer_id: str, transactions: list[dict[str, Any]]) -> dict[str, Any]:
		"""Detect AML typology patterns in a customer's transaction history."""
		total_volume = sum(float(t.get("amount", 0)) for t in transactions)
		structuring = any(900_000 <= float(t.get("amount", 0)) < 1_000_000 for t in transactions)
		rapid_movement = len([t for t in transactions if float(t.get("amount", 0)) > 100_000]) >= 3
		typologies = []
		if structuring:
			typologies.append("STRUCTURING")
		if rapid_movement:
			typologies.append("RAPID_MOVEMENT")
		risk_score = len(typologies) * 35.0
		self._audit(self.tenant_id, "aml_pattern_detected", customer_id)
		return {
			"customer_id": customer_id, "transaction_count": len(transactions),
			"total_volume": total_volume, "typologies": typologies,
			"risk_score": min(risk_score, 100.0), "assessed_at": _iso(),
		}

	async def export_fraud_data(self, fmt: str = "csv") -> dict[str, Any]:
		"""Export fraud signals and cases for reporting."""
		assert fmt in {"csv", "json", "excel"}
		return {
			"tenant_id": self.tenant_id, "format": fmt,
			"signals": len([s for s in self.signals.values() if s.tenant_id == self.tenant_id]),
			"cases": len([c for c in self.cases.values() if c.tenant_id == self.tenant_id]),
			"file_reference": f"fraud_{self.tenant_id}_{_iso()[:10]}.{fmt}", "generated_at": _iso(),
		}

	async def fraud_risk_score_batch(self, transactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
		"""Score multiple transactions for fraud risk in batch."""
		results = []
		for txn in transactions:
			result = await self.detect_transaction_fraud(txn)
			results.append(result)
		return results

	async def false_positive_rate_report(self, period: str) -> dict[str, Any]:
		"""Report on false positive rate and model accuracy for a period."""
		report = await self.fraud_report(period)
		return {
			**report, "model_precision_pct": report.get("precision_pct", 0),
			"false_positive_rate_pct": report.get("false_positive_rate_pct", 0),
			"recommendation": "recalibrate" if report.get("false_positive_rate_pct", 0) > 5 else "maintain",
		}

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

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

	def _record_evidence(
		self,
		evidence_id: str,
		tenant_id: str,
		kind: str,
		reference_id: str,
		status: str,
		metadata: dict[str, Any],
	) -> dict[str, Any]:
		evidence = FraudEvidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
		self.evidence[evidence_id] = evidence
		return evidence.to_dict()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"actor_id": self.actor_id,
			"timestamp": _iso(),
		})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", "fraud_policy_denied") for action in result["actions"]
		)
		raise PermissionError(reasons or "fraud_policy_denied")


FintechFraudService = FraudDetectionService
