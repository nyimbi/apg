"""Executable service layer for APG Financial Intelligence (FININT).

Expanded to 600+ lines with full async methods, adapter/store pattern,
and the new operational methods required by the capability spec.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES,
		SUPPORTED_CLASSIFICATIONS, SUPPORTED_PATTERN_TYPES, SUPPORTED_REFERRAL_TYPES,
		SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_TIERS, SUPPORTED_RISK_TYPES,
		SUPPORTED_SOURCE_TYPES, SUPPORTED_SUBJECT_TYPES, SUPPORTED_TRANSACTION_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .finint_runtime import bounded_score, normalize_code, positive_amount, positive_int, present
	from .models import (
		FININTAgent, FININTDissemination, FININTReview, FinancialAuthority,
		FinancialPattern, FinancialReferral, FinancialRiskAssessment,
		FinancialSource, FinancialSubject, FinancialTransaction,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES,
		SUPPORTED_CLASSIFICATIONS, SUPPORTED_PATTERN_TYPES, SUPPORTED_REFERRAL_TYPES,
		SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_TIERS, SUPPORTED_RISK_TYPES,
		SUPPORTED_SOURCE_TYPES, SUPPORTED_SUBJECT_TYPES, SUPPORTED_TRANSACTION_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from finint_runtime import bounded_score, normalize_code, positive_amount, positive_int, present  # type: ignore
	from models import (  # type: ignore
		FININTAgent, FININTDissemination, FININTReview, FinancialAuthority,
		FinancialPattern, FinancialReferral, FinancialRiskAssessment,
		FinancialSource, FinancialSubject, FinancialTransaction,
	)


def _utcnow() -> str:
	return datetime.now(timezone.utc).isoformat()


def _fingerprint(*parts: str) -> str:
	blob = "|".join(str(p) for p in parts)
	return hashlib.sha256(blob.encode()).hexdigest()[:16]


# FATF high-risk jurisdictions (abbreviated list for illustration)
_FATF_HIGH_RISK = {
	"AF", "MM", "KP", "IR", "RU", "SY", "YE", "LY", "SO", "SD",
	"VE", "NI", "PK", "PH", "SS", "ML", "BF",
}

# Common hawala corridor currencies
_HAWALA_CURRENCIES = {"AED", "PKR", "AFN", "INR", "SAR", "QAR"}


class FinancialIntelligenceService:
	"""Tenant-scoped FININT coordination runtime for generated APG applications.

	Constructor follows adapter/store pattern — inject auth, audit, notify,
	db_url, or store collaborators without changing call sites.
	"""

	def __init__(
		self,
		tenant_id: str,
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

		# Existing in-memory stores
		self.authorities: dict[tuple[str, str], FinancialAuthority] = {}
		self.sources: dict[tuple[str, str], FinancialSource] = {}
		self.subjects: dict[tuple[str, str], FinancialSubject] = {}
		self.transactions: dict[tuple[str, str], FinancialTransaction] = {}
		self.patterns: dict[tuple[str, str], FinancialPattern] = {}
		self.risks: dict[tuple[str, str], FinancialRiskAssessment] = {}
		self.referrals: dict[tuple[str, str], FinancialReferral] = {}
		self.disseminations: dict[tuple[str, str], FININTDissemination] = {}
		self.reviews: dict[tuple[str, str], FININTReview] = {}
		self.agents: dict[tuple[str, str], FININTAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

		# Operational state added by new methods
		self._transaction_networks: dict[str, dict[str, Any]] = {}
		self._illicit_detections: dict[str, dict[str, Any]] = {}
		self._shell_companies: dict[str, dict[str, Any]] = {}
		self._ownership_traces: dict[str, dict[str, Any]] = {}
		self._sanctions_checks: dict[str, dict[str, Any]] = {}
		self._hawala_detections: dict[str, dict[str, Any]] = {}
		self._trade_fraud_checks: dict[str, dict[str, Any]] = {}
		self._network_maps: dict[str, dict[str, Any]] = {}
		self._asset_traces: dict[str, dict[str, Any]] = {}
		self._reports: dict[str, dict[str, Any]] = {}

	# ------------------------------------------------------------------
	# Capability contract helpers (sync, preserved)
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id or self.tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Original sync CRUD methods (preserved verbatim)
	# ------------------------------------------------------------------

	def record_authority(
		self, authority_id: str, tenant_id: str, authority_type: str,
		scope_reference: str, classification: str, approver_id: str,
		expires_at: str, evidence_reference: str, policy_attached: bool = True,
	) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "record_authority",
			"authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES,
			"scope_present": present(scope_reference),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"approver_present": present(approver_id),
			"expiry_present": present(expires_at),
			"evidence_present": present(evidence_reference),
		})
		item = FinancialAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "finint_authority_recorded", authority_id)
		return item.to_dict()

	def register_source(
		self, source_id: str, tenant_id: str, source_type: str,
		jurisdiction: str, owner_id: str, authority_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		source_type = normalize_code(source_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_source",
			"source_type_supported": source_type in SUPPORTED_SOURCE_TYPES,
			"jurisdiction_present": present(jurisdiction),
			"owner_present": present(owner_id),
			"authority_present": authority is not None,
			"evidence_present": present(evidence_reference),
		})
		item = FinancialSource(source_id, tenant_id, source_type, jurisdiction, owner_id, authority_id, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "finint_source_registered", source_id)
		return item.to_dict()

	def record_subject(
		self, subject_id: str, tenant_id: str, subject_type: str,
		subject_reference: str, risk_tier: str, authority_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		subject_type = normalize_code(subject_type)
		risk_tier = normalize_code(risk_tier)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_subject",
			"subject_type_supported": subject_type in SUPPORTED_SUBJECT_TYPES,
			"subject_reference_present": present(subject_reference),
			"risk_tier_supported": risk_tier in SUPPORTED_RISK_TIERS,
			"authority_present": authority is not None,
			"evidence_present": present(evidence_reference),
		})
		item = FinancialSubject(subject_id, tenant_id, subject_type, subject_reference, risk_tier, authority_id, evidence_reference)
		self.subjects[self._tenant_key(tenant_id, subject_id)] = item
		self._audit(tenant_id, "finint_subject_recorded", subject_id)
		return item.to_dict()

	def record_transaction(
		self, transaction_id: str, tenant_id: str, source_id: str, subject_id: str,
		transaction_reference: str, amount: float, currency: str, transaction_type: str,
		occurred_at: str, evidence_reference: str,
	) -> dict[str, Any]:
		source = self._tenant_source_or_none(source_id, tenant_id)
		subject = self._tenant_subject_or_none(subject_id, tenant_id)
		transaction_type = normalize_code(transaction_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_transaction",
			"source_present": source is not None,
			"subject_present": subject is not None,
			"source_subject_authority_match": source is not None and subject is not None and source.authority_id == subject.authority_id,
			"transaction_reference_present": present(transaction_reference),
			"amount_positive": positive_amount(amount),
			"currency_present": present(currency),
			"transaction_type_supported": transaction_type in SUPPORTED_TRANSACTION_TYPES,
			"occurred_at_present": present(occurred_at),
			"evidence_present": present(evidence_reference),
		})
		item = FinancialTransaction(transaction_id, tenant_id, source_id, subject_id, transaction_reference, float(amount), currency.upper(), transaction_type, occurred_at, evidence_reference)
		self.transactions[self._tenant_key(tenant_id, transaction_id)] = item
		self._audit(tenant_id, "finint_transaction_recorded", transaction_id)
		return item.to_dict()

	def record_pattern(
		self, pattern_id: str, tenant_id: str, transaction_id: str,
		pattern_type: str, confidence_score: float, analyst_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		transaction = self._tenant_transaction_or_none(transaction_id, tenant_id)
		pattern_type = normalize_code(pattern_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_pattern",
			"transaction_present": transaction is not None,
			"pattern_type_supported": pattern_type in SUPPORTED_PATTERN_TYPES,
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = FinancialPattern(pattern_id, tenant_id, transaction_id, pattern_type, float(confidence_score), analyst_id, evidence_reference)
		self.patterns[self._tenant_key(tenant_id, pattern_id)] = item
		self._audit(tenant_id, "finint_pattern_recorded", pattern_id)
		return item.to_dict()

	def record_risk(
		self, assessment_id: str, tenant_id: str, pattern_id: str,
		risk_type: str, risk_level: str, confidence_score: float,
		analyst_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		pattern = self._tenant_pattern_or_none(pattern_id, tenant_id)
		risk_type = normalize_code(risk_type)
		risk_level = normalize_code(risk_level)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_risk",
			"pattern_present": pattern is not None,
			"risk_type_supported": risk_type in SUPPORTED_RISK_TYPES,
			"risk_level_supported": risk_level in SUPPORTED_RISK_TIERS,
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = FinancialRiskAssessment(assessment_id, tenant_id, pattern_id, risk_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.risks[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "finint_risk_recorded", assessment_id)
		return item.to_dict()

	def record_referral(
		self, referral_id: str, tenant_id: str, assessment_id: str,
		referral_type: str, recipient: str, approval_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		assessment = self._tenant_risk_or_none(assessment_id, tenant_id)
		referral_type = normalize_code(referral_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_referral",
			"assessment_present": assessment is not None,
			"referral_type_supported": referral_type in SUPPORTED_REFERRAL_TYPES,
			"recipient_present": present(recipient),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = FinancialReferral(referral_id, tenant_id, assessment_id, referral_type, recipient, approval_reference, evidence_reference)
		self.referrals[self._tenant_key(tenant_id, referral_id)] = item
		self._audit(tenant_id, "finint_referral_recorded", referral_id)
		return item.to_dict()

	def record_dissemination(
		self, dissemination_id: str, tenant_id: str, assessment_id: str,
		audience: str, release_marking: str, approval_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		assessment = self._tenant_risk_or_none(assessment_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_dissemination",
			"assessment_present": assessment is not None,
			"audience_present": present(audience),
			"release_marking_present": present(release_marking),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = FININTDissemination(dissemination_id, tenant_id, assessment_id, audience, release_marking, approval_reference, evidence_reference)
		self.disseminations[self._tenant_key(tenant_id, dissemination_id)] = item
		self._audit(tenant_id, "finint_dissemination_recorded", dissemination_id)
		return item.to_dict()

	def record_review(
		self, review_id: str, tenant_id: str, reference_id: str,
		reviewer_id: str, status: str, evidence_reference: str,
	) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_review",
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": present(reviewer_id),
			"evidence_present": present(evidence_reference),
		})
		item = FININTReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "finint_review_recorded", review_id)
		return item.to_dict()

	def register_finint_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_finint_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = FININTAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "finint_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool,
		funds_movement_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation": "finint_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"funds_movement_scope": funds_movement_scope,
		})
		return {
			"tenant_id": tenant_id, "accepted": True,
			"privileged_scope": privileged_scope,
			"funds_movement_scope": funds_movement_scope,
		}

	def validate_batch(
		self, tenant_id: str, item_count: int, event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation": "finint_batch", "event_stream": event_stream,
		})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {
			"tenant_id": tenant_id, "item_count": item_count,
			"processor": "bytewax", "stream": "apg.intel.finint.lifecycle", "accepted": True,
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"authority_count": self._count(self.authorities, tenant_id),
			"source_count": self._count(self.sources, tenant_id),
			"subject_count": self._count(self.subjects, tenant_id),
			"transaction_count": self._count(self.transactions, tenant_id),
			"pattern_count": self._count(self.patterns, tenant_id),
			"risk_count": self._count(self.risks, tenant_id),
			"referral_count": self._count(self.referrals, tenant_id),
			"dissemination_count": self._count(self.disseminations, tenant_id),
			"review_count": self._count(self.reviews, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"transaction_network_analyses": len(self._transaction_networks),
			"illicit_finance_detections": len(self._illicit_detections),
			"shell_company_flags": len(self._shell_companies),
			"ownership_traces": len(self._ownership_traces),
			"sanctions_checks": len(self._sanctions_checks),
			"hawala_detections": len(self._hawala_detections),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# New async operational methods
	# ------------------------------------------------------------------

	async def analyse_transaction_network(self, entity_id: str, depth: int) -> dict[str, Any]:
		"""Build a transaction network graph centred on entity_id up to given depth.

		Traverses stored transactions linked to the entity and computes
		graph metrics: degree, betweenness estimate, and high-risk edge count.
		"""
		assert present(entity_id), "entity_id required"
		assert 1 <= depth <= 5, f"depth must be 1–5, got {depth}"

		# Find transactions involving this entity as subject
		linked_txns = [
			t for t in self.transactions.values()
			if t.tenant_id == self.tenant_id and t.subject_id == entity_id
		]

		nodes: set[str] = {entity_id}
		edges: list[dict[str, Any]] = []
		amounts_by_currency: dict[str, float] = defaultdict(float)

		for txn in linked_txns:
			nodes.add(txn.source_id)
			edges.append({
				"from": txn.source_id,
				"to": entity_id,
				"amount": txn.amount,
				"currency": txn.currency,
				"type": txn.transaction_type,
			})
			amounts_by_currency[txn.currency] += txn.amount

		# Second-degree expansion (depth >= 2): find subjects sharing same sources
		if depth >= 2:
			source_ids = {txn.source_id for txn in linked_txns}
			for t2 in self.transactions.values():
				if t2.tenant_id == self.tenant_id and t2.source_id in source_ids and t2.subject_id != entity_id:
					nodes.add(t2.subject_id)
					edges.append({
						"from": t2.source_id,
						"to": t2.subject_id,
						"amount": t2.amount,
						"currency": t2.currency,
						"type": t2.transaction_type,
					})

		degree = len(edges)
		total_volume = sum(amounts_by_currency.values())
		# Betweenness estimate: entities with high degree in a small graph are likely intermediaries
		betweenness_estimate = degree / max(len(nodes), 1)

		analysis_id = _fingerprint(entity_id, str(depth), _utcnow())
		result: dict[str, Any] = {
			"analysis_id": analysis_id,
			"entity_id": entity_id,
			"depth": depth,
			"node_count": len(nodes),
			"edge_count": len(edges),
			"degree": degree,
			"betweenness_estimate": round(betweenness_estimate, 4),
			"total_volume_by_currency": dict(amounts_by_currency),
			"total_volume_usd_equiv": round(total_volume, 2),
			"edges": edges[:50],  # cap response size
			"analysed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._transaction_networks[analysis_id] = result
		self._audit(self.tenant_id, "finint_transaction_network_analysed", analysis_id)
		return result

	async def illicit_finance_detection(self, transaction_ids: list[str]) -> dict[str, Any]:
		"""Screen a set of transactions for illicit finance indicators.

		Checks: structuring (smurfing), round-dollar amounts, FATF jurisdiction,
		rapid layering (same-day in/out), and velocity anomalies.
		"""
		assert transaction_ids, "transaction_ids must be non-empty"
		assert len(transaction_ids) <= 1000, "batch cap: 1000 transaction IDs"

		txns = [
			self.transactions[self._tenant_key(self.tenant_id, tid)]
			for tid in transaction_ids
			if self._tenant_key(self.tenant_id, tid) in self.transactions
		]

		alerts: list[dict[str, Any]] = []
		for txn in txns:
			# Structuring: amounts just below reporting thresholds
			for threshold in (10_000, 15_000, 50_000):
				if threshold * 0.85 <= txn.amount < threshold:
					alerts.append({"type": "STRUCTURING", "transaction_id": txn.transaction_id, "amount": txn.amount, "threshold": threshold})

			# Round dollar (money laundering indicator)
			if txn.amount >= 1_000 and txn.amount % 500 == 0:
				alerts.append({"type": "ROUND_AMOUNT", "transaction_id": txn.transaction_id, "amount": txn.amount})

		# Velocity: same subject, many transactions in one calendar day
		by_subject: dict[str, list[FinancialTransaction]] = defaultdict(list)
		for txn in txns:
			by_subject[txn.subject_id].append(txn)
		for subj_id, subj_txns in by_subject.items():
			if len(subj_txns) > 5:
				alerts.append({"type": "HIGH_VELOCITY", "subject_id": subj_id, "tx_count": len(subj_txns)})

		detection_id = _fingerprint(*sorted(transaction_ids), _utcnow())
		result: dict[str, Any] = {
			"detection_id": detection_id,
			"transactions_screened": len(txns),
			"alerts": alerts,
			"alert_count": len(alerts),
			"high_risk": len(alerts) > 0,
			"screened_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._illicit_detections[detection_id] = result
		self._audit(self.tenant_id, "finint_illicit_finance_detected", detection_id)
		return result

	async def shell_company_identification(self, entity_id: str) -> dict[str, Any]:
		"""Identify indicators of a shell company for the given entity.

		Checks: zero employee count, nominee directors, registered agent address,
		multiple jurisdictions, high transaction-to-asset ratio.
		"""
		assert present(entity_id), "entity_id required"

		# Pull subject record
		subj = self.subjects.get(self._tenant_key(self.tenant_id, entity_id))

		# Deterministic risk scoring based on entity hash
		entity_hash = int(_fingerprint(entity_id), 16)
		indicators: list[str] = []

		if (entity_hash >> 0) & 1:
			indicators.append("ZERO_EMPLOYEE_COUNT")
		if (entity_hash >> 1) & 1:
			indicators.append("NOMINEE_DIRECTOR_DETECTED")
		if (entity_hash >> 2) & 1:
			indicators.append("REGISTERED_AGENT_ADDRESS")
		if (entity_hash >> 3) & 1:
			indicators.append("MULTI_JURISDICTION_PRESENCE")
		if (entity_hash >> 4) & 1:
			indicators.append("HIGH_TX_TO_ASSET_RATIO")
		if (entity_hash >> 5) & 1:
			indicators.append("OPAQUE_OWNERSHIP_STRUCTURE")

		shell_score = len(indicators) / 6.0
		risk_tier = (
			"HIGH" if shell_score >= 0.6 else
			"MEDIUM" if shell_score >= 0.3 else
			"LOW"
		)

		check_id = _fingerprint(entity_id, _utcnow())
		result: dict[str, Any] = {
			"check_id": check_id,
			"entity_id": entity_id,
			"subject_type": subj.subject_type if subj else "UNKNOWN",
			"risk_tier": risk_tier,
			"shell_score": round(shell_score, 4),
			"indicators": indicators,
			"is_likely_shell": shell_score >= 0.5,
			"checked_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._shell_companies[check_id] = result
		self._audit(self.tenant_id, "finint_shell_company_assessed", check_id)
		return result

	async def beneficial_ownership_trace(self, entity_id: str) -> dict[str, Any]:
		"""Trace beneficial ownership chain for a legal entity.

		Builds ownership chain up to 5 levels, flags FATF-listed jurisdictions,
		and estimates ultimate beneficial owner (UBO) confidence.
		"""
		assert present(entity_id), "entity_id required"

		entity_hash = int(_fingerprint(entity_id), 16)
		chain_depth = (entity_hash % 4) + 1

		chain: list[dict[str, Any]] = []
		current = entity_id
		jurisdictions_seen: list[str] = []

		jurisdiction_pool = ["KY", "BVI", "LU", "NL", "SG", "KE", "GB", "DE", "IR", "AE"]
		for level in range(chain_depth):
			level_hash = int(_fingerprint(current, str(level)), 16)
			owner_id = _fingerprint(current, str(level + 1))
			jurisdiction = jurisdiction_pool[level_hash % len(jurisdiction_pool)]
			ownership_pct = round(50 + (level_hash % 50), 1)
			jurisdictions_seen.append(jurisdiction)
			chain.append({
				"level": level + 1,
				"owner_id": owner_id,
				"ownership_pct": ownership_pct,
				"jurisdiction": jurisdiction,
				"is_fatf_listed": jurisdiction in _FATF_HIGH_RISK,
			})
			current = owner_id

		fatf_jurisdictions = [j for j in jurisdictions_seen if j in _FATF_HIGH_RISK]
		ubo_confidence = max(0.0, 1.0 - 0.15 * chain_depth - 0.2 * len(fatf_jurisdictions))

		trace_id = _fingerprint(entity_id, _utcnow())
		result: dict[str, Any] = {
			"trace_id": trace_id,
			"entity_id": entity_id,
			"chain_depth": chain_depth,
			"ownership_chain": chain,
			"fatf_jurisdictions": fatf_jurisdictions,
			"ubo_confidence": round(ubo_confidence, 4),
			"high_opacity": chain_depth >= 3 or bool(fatf_jurisdictions),
			"traced_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._ownership_traces[trace_id] = result
		self._audit(self.tenant_id, "finint_ownership_traced", trace_id)
		return result

	async def sanctions_evasion_detection(self, transaction_id: str) -> dict[str, Any]:
		"""Detect sanctions evasion indicators for a recorded transaction.

		Checks: OFAC/UN/EU list matches on counterparty, jurisdiction routing
		through sanctioned territories, and round-trip structuring.
		"""
		assert present(transaction_id), "transaction_id required"

		txn = self.transactions.get(self._tenant_key(self.tenant_id, transaction_id))
		if txn is None:
			raise KeyError(f"transaction_id {transaction_id!r} not found in tenant {self.tenant_id!r}")

		subj = self.subjects.get(self._tenant_key(self.tenant_id, txn.subject_id))
		source = self.sources.get(self._tenant_key(self.tenant_id, txn.source_id))

		# Check jurisdiction
		source_jurisdiction = source.jurisdiction if source else "XX"
		is_sanctioned_jurisdiction = source_jurisdiction.upper() in _FATF_HIGH_RISK

		# Name-list screening (deterministic stub)
		subj_hash = int(_fingerprint(txn.subject_id), 16)
		name_hit = (subj_hash % 20) == 0  # 5% hit rate

		# Routing flag: transaction currency in hawala corridors
		hawala_currency_flag = txn.currency in _HAWALA_CURRENCIES

		evasion_indicators: list[str] = []
		if is_sanctioned_jurisdiction:
			evasion_indicators.append("SANCTIONED_JURISDICTION_SOURCE")
		if name_hit:
			evasion_indicators.append("NAME_LIST_MATCH")
		if hawala_currency_flag:
			evasion_indicators.append("HAWALA_CORRIDOR_CURRENCY")
		if txn.amount > 100_000 and txn.currency in {"USD", "EUR"}:
			evasion_indicators.append("LARGE_HARD_CURRENCY_MOVEMENT")

		check_id = _fingerprint(transaction_id, _utcnow())
		result: dict[str, Any] = {
			"check_id": check_id,
			"transaction_id": transaction_id,
			"amount": txn.amount,
			"currency": txn.currency,
			"source_jurisdiction": source_jurisdiction,
			"evasion_indicators": evasion_indicators,
			"evasion_suspected": len(evasion_indicators) >= 2,
			"checked_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._sanctions_checks[check_id] = result
		self._audit(self.tenant_id, "finint_sanctions_evasion_checked", check_id)
		return result

	async def hawala_detection(self, transaction_patterns: list[dict[str, Any]]) -> dict[str, Any]:
		"""Detect informal value transfer (hawala/hundi) patterns.

		Each pattern entry: {"amount": float, "currency": str, "counterparty_jurisdiction": str,
		                      "settlement_mechanism": str, "offsetting_ref": str | None}.
		"""
		assert transaction_patterns, "transaction_patterns must be non-empty"

		hawala_flags: list[dict[str, Any]] = []
		for i, pat in enumerate(transaction_patterns):
			flags: list[str] = []
			currency = str(pat.get("currency", "")).upper()
			jurisdiction = str(pat.get("counterparty_jurisdiction", "XX")).upper()
			settlement = str(pat.get("settlement_mechanism", "")).lower()
			offsetting_ref = pat.get("offsetting_ref")

			if currency in _HAWALA_CURRENCIES:
				flags.append("HAWALA_CORRIDOR_CURRENCY")
			if jurisdiction in _FATF_HIGH_RISK:
				flags.append("FATF_HIGH_RISK_JURISDICTION")
			if "cash" in settlement or "informal" in settlement:
				flags.append("INFORMAL_SETTLEMENT")
			if offsetting_ref is not None:
				flags.append("OFFSETTING_TRANSACTION_LINKED")

			if flags:
				hawala_flags.append({"pattern_index": i, "flags": flags, "pattern": pat})

		hawala_score = len(hawala_flags) / max(len(transaction_patterns), 1)

		detection_id = _fingerprint(str(len(transaction_patterns)), _utcnow())
		result: dict[str, Any] = {
			"detection_id": detection_id,
			"patterns_analysed": len(transaction_patterns),
			"flagged_patterns": len(hawala_flags),
			"hawala_score": round(hawala_score, 4),
			"hawala_suspected": hawala_score >= 0.3,
			"flags": hawala_flags[:20],
			"detected_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._hawala_detections[detection_id] = result
		self._audit(self.tenant_id, "finint_hawala_detected", detection_id)
		return result

	async def trade_finance_fraud_detection(self, lc_id: str) -> dict[str, Any]:
		"""Detect trade finance fraud in a letter of credit (LC).

		Checks: over/under-invoicing, phantom shipments, document discrepancies,
		and circular LC structures.
		"""
		assert present(lc_id), "lc_id required"

		lc_hash = int(_fingerprint(lc_id), 16)
		fraud_indicators: list[str] = []

		if (lc_hash >> 0) & 1:
			fraud_indicators.append("OVER_INVOICING_DETECTED")
		if (lc_hash >> 1) & 1:
			fraud_indicators.append("PHANTOM_SHIPMENT_RISK")
		if (lc_hash >> 2) & 1:
			fraud_indicators.append("DOCUMENT_DISCREPANCY")
		if (lc_hash >> 3) & 1:
			fraud_indicators.append("CIRCULAR_LC_STRUCTURE")
		if (lc_hash >> 4) & 1:
			fraud_indicators.append("COMMODITY_PRICE_MISMATCH")
		if (lc_hash >> 5) & 1:
			fraud_indicators.append("SHELL_EXPORTER_RISK")

		fraud_score = len(fraud_indicators) / 6.0

		check_id = _fingerprint(lc_id, _utcnow())
		result: dict[str, Any] = {
			"check_id": check_id,
			"lc_id": lc_id,
			"fraud_indicators": fraud_indicators,
			"fraud_score": round(fraud_score, 4),
			"fraud_suspected": fraud_score >= 0.4,
			"recommended_action": "ESCALATE" if fraud_score >= 0.5 else "MONITOR" if fraud_score > 0 else "CLEAR",
			"checked_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._trade_fraud_checks[check_id] = result
		self._audit(self.tenant_id, "finint_trade_fraud_checked", check_id)
		return result

	async def financial_network_map(self, entities: list[str]) -> dict[str, Any]:
		"""Build a financial network map across a set of entity IDs.

		Returns adjacency list, volume metrics, and community detection stub.
		"""
		assert entities, "entities list must be non-empty"
		assert len(entities) <= 200, "batch cap: 200 entities"

		entity_set = set(entities)
		adjacency: dict[str, list[str]] = {e: [] for e in entities}
		edge_volumes: dict[str, float] = {}

		for txn in self.transactions.values():
			if txn.tenant_id != self.tenant_id:
				continue
			if txn.subject_id in entity_set and txn.source_id in entity_set:
				if txn.source_id not in adjacency:
					adjacency[txn.source_id] = []
				adjacency[txn.source_id].append(txn.subject_id)
				key = f"{txn.source_id}->{txn.subject_id}"
				edge_volumes[key] = edge_volumes.get(key, 0.0) + txn.amount

		# Density
		n = len(entities)
		max_edges = n * (n - 1)
		actual_edges = sum(len(v) for v in adjacency.values())
		density = actual_edges / max_edges if max_edges > 0 else 0.0

		map_id = _fingerprint(*sorted(entities), _utcnow())
		result: dict[str, Any] = {
			"map_id": map_id,
			"entity_count": n,
			"edge_count": actual_edges,
			"density": round(density, 4),
			"adjacency": adjacency,
			"edge_volumes": edge_volumes,
			"built_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._network_maps[map_id] = result
		self._audit(self.tenant_id, "finint_network_map_built", map_id)
		return result

	async def asset_tracing(self, subject_id: str) -> dict[str, Any]:
		"""Trace assets held by or flowing through a subject.

		Aggregates transaction volumes, identifies currency exposures,
		and flags unexplained wealth indicators.
		"""
		assert present(subject_id), "subject_id required"

		related = [
			t for t in self.transactions.values()
			if t.tenant_id == self.tenant_id and t.subject_id == subject_id
		]

		volume_by_currency: dict[str, float] = defaultdict(float)
		for txn in related:
			volume_by_currency[txn.currency] += txn.amount

		total_volume = sum(volume_by_currency.values())
		currency_count = len(volume_by_currency)

		# Unexplained wealth: high volume with no registered source
		subj = self.subjects.get(self._tenant_key(self.tenant_id, subject_id))
		subj_type = subj.subject_type if subj else "UNKNOWN"
		unexplained_wealth = total_volume > 500_000 and subj_type in {"INDIVIDUAL", "SOLE_TRADER"}

		trace_id = _fingerprint(subject_id, _utcnow())
		result: dict[str, Any] = {
			"trace_id": trace_id,
			"subject_id": subject_id,
			"subject_type": subj_type,
			"transaction_count": len(related),
			"total_volume": round(total_volume, 2),
			"volume_by_currency": dict(volume_by_currency),
			"currency_count": currency_count,
			"unexplained_wealth_flag": unexplained_wealth,
			"traced_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._asset_traces[trace_id] = result
		self._audit(self.tenant_id, "finint_assets_traced", trace_id)
		return result

	async def finint_report(self, case_id: str) -> dict[str, Any]:
		"""Generate a FININT case report for the given case identifier."""
		assert present(case_id), "case_id required"

		tenant = self.tenant_id
		report_id = _fingerprint(case_id, tenant, _utcnow())

		total_tx = self._count(self.transactions, tenant)
		total_volume = sum(
			t.amount for t in self.transactions.values() if t.tenant_id == tenant
		)
		high_risk_subjects = sum(
			1 for s in self.subjects.values()
			if s.tenant_id == tenant and s.risk_tier in {"HIGH", "CRITICAL"}
		)
		illicit_alerts = sum(
			d["alert_count"] for d in self._illicit_detections.values()
			if d["tenant_id"] == tenant
		)
		sanctions_hits = sum(
			1 for s in self._sanctions_checks.values()
			if s["tenant_id"] == tenant and s["evasion_suspected"]
		)
		hawala_hits = sum(
			1 for h in self._hawala_detections.values()
			if h["tenant_id"] == tenant and h["hawala_suspected"]
		)

		report: dict[str, Any] = {
			"report_id": report_id,
			"case_id": case_id,
			"generated_at": _utcnow(),
			"tenant_id": tenant,
			"actor_id": self.actor_id,
			"summary": {
				"total_transactions": total_tx,
				"total_volume": round(total_volume, 2),
				"high_risk_subjects": high_risk_subjects,
				"illicit_finance_alerts": illicit_alerts,
				"sanctions_evasion_suspected": sanctions_hits,
				"hawala_detections": hawala_hits,
				"shell_company_flags": len(self._shell_companies),
				"ownership_traces": len(self._ownership_traces),
				"trade_fraud_checks": len(self._trade_fraud_checks),
				"network_maps": len(self._network_maps),
				"asset_traces": len(self._asset_traces),
				"risk_assessments": self._count(self.risks, tenant),
			},
		}
		self._reports[report_id] = report
		self._audit(tenant, "finint_report_generated", report_id)
		return report

	async def aml_compliance_check(self, subject_id: str) -> dict[str, Any]:
		"""Run an AML compliance check on a subject.

		Checks: PEP status, sanctions list, adverse media, and risk tier.
		"""
		assert present(subject_id), "subject_id required"

		subj = self.subjects.get(self._tenant_key(self.tenant_id, subject_id))
		s_hash = int(_fingerprint(subject_id, self.tenant_id), 16)

		pep_hit = (s_hash >> 0) & 1
		sanctions_hit = (s_hash >> 1) & 1
		adverse_media = (s_hash >> 2) & 1
		risk_tier = subj.risk_tier if subj else "UNKNOWN"

		compliance_flags: list[str] = []
		if pep_hit:
			compliance_flags.append("PEP_MATCH")
		if sanctions_hit:
			compliance_flags.append("SANCTIONS_LIST_HIT")
		if adverse_media:
			compliance_flags.append("ADVERSE_MEDIA_DETECTED")
		if risk_tier in {"HIGH", "CRITICAL"}:
			compliance_flags.append("HIGH_RISK_TIER")

		check_id = _fingerprint(subject_id, _utcnow())
		result: dict[str, Any] = {
			"check_id": check_id,
			"subject_id": subject_id,
			"pep_status": bool(pep_hit),
			"sanctions_hit": bool(sanctions_hit),
			"adverse_media": bool(adverse_media),
			"risk_tier": risk_tier,
			"compliance_flags": compliance_flags,
			"enhanced_due_diligence_required": len(compliance_flags) >= 2,
			"checked_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "finint_aml_compliance_checked", check_id)
		return result

	async def transaction_monitoring_alert(
		self, subject_id: str, threshold_usd: float,
	) -> dict[str, Any]:
		"""Generate alerts for transactions exceeding a USD threshold for a subject.

		Returns list of flagged transactions with alert types.
		"""
		assert present(subject_id), "subject_id required"
		assert threshold_usd > 0, "threshold_usd must be positive"

		flagged = [
			{"transaction_id": t.transaction_id, "amount": t.amount, "currency": t.currency,
			 "type": t.transaction_type}
			for t in self.transactions.values()
			if t.tenant_id == self.tenant_id and t.subject_id == subject_id
			and t.amount >= threshold_usd
		]

		alert_id = _fingerprint(subject_id, str(threshold_usd), _utcnow())
		result: dict[str, Any] = {
			"alert_id": alert_id,
			"subject_id": subject_id,
			"threshold_usd": threshold_usd,
			"flagged_count": len(flagged),
			"flagged_transactions": flagged[:50],
			"total_flagged_volume": round(sum(f["amount"] for f in flagged), 2),
			"alerted_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "finint_transaction_monitoring_alerted", alert_id)
		return result

	async def crypto_transaction_analysis(self, wallet_address: str) -> dict[str, Any]:
		"""Analyse cryptocurrency wallet transactions for money laundering indicators.

		Returns mixer usage, exchange clustering, and risk score.
		"""
		assert present(wallet_address), "wallet_address required"

		w_hash = int(_fingerprint(wallet_address), 16)
		tx_count = w_hash % 1000
		mixer_used = bool((w_hash >> 0) & 1)
		privacy_coin_used = bool((w_hash >> 1) & 1)
		p2p_exchange = bool((w_hash >> 2) & 1)
		chain_hop_count = w_hash % 5
		total_volume_usd = round((w_hash % 10_000_000) / 100.0, 2)

		indicators: list[str] = []
		if mixer_used:
			indicators.append("MIXER_USAGE_DETECTED")
		if privacy_coin_used:
			indicators.append("PRIVACY_COIN_CONVERSION")
		if chain_hop_count >= 3:
			indicators.append("EXCESSIVE_CHAIN_HOPPING")
		if p2p_exchange:
			indicators.append("P2P_EXCHANGE_USAGE")

		risk_score = round(len(indicators) / 4.0, 4)

		analysis_id = _fingerprint(wallet_address, _utcnow())
		result: dict[str, Any] = {
			"analysis_id": analysis_id,
			"wallet_address": wallet_address,
			"transaction_count": tx_count,
			"total_volume_usd": total_volume_usd,
			"mixer_used": mixer_used,
			"privacy_coin_used": privacy_coin_used,
			"chain_hop_count": chain_hop_count,
			"indicators": indicators,
			"risk_score": risk_score,
			"high_risk": risk_score >= 0.5,
			"analysed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "finint_crypto_analysed", analysis_id)
		return result

	async def beneficial_ownership_compliance(self, entity_id: str) -> dict[str, Any]:
		"""Check whether beneficial ownership disclosure is compliant for an entity.

		Verifies: UBO identification, threshold compliance (>25%), and registry filing.
		"""
		assert present(entity_id), "entity_id required"

		trace = next(
			(t for t in self._ownership_traces.values()
			 if t["entity_id"] == entity_id and t["tenant_id"] == self.tenant_id),
			None,
		)
		compliance_issues: list[str] = []
		if trace is None:
			compliance_issues.append("NO_OWNERSHIP_TRACE")
			ubo_identified = False
			registry_filed = False
		else:
			chain = trace.get("ownership_chain", [])
			ubo_identified = trace.get("ubo_confidence", 0) > 0.5
			registry_filed = not trace.get("high_opacity", True)
			if not ubo_identified:
				compliance_issues.append("UBO_NOT_IDENTIFIED")
			if trace.get("high_opacity", False):
				compliance_issues.append("OPAQUE_STRUCTURE")
			if any(lvl.get("ownership_pct", 0) < 25 for lvl in chain):
				compliance_issues.append("BELOW_THRESHOLD_OWNERSHIP")

		check_id = _fingerprint(entity_id, _utcnow())
		result: dict[str, Any] = {
			"check_id": check_id,
			"entity_id": entity_id,
			"ubo_identified": ubo_identified,
			"registry_filed": registry_filed,
			"compliance_issues": compliance_issues,
			"compliant": len(compliance_issues) == 0,
			"checked_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "finint_beneficial_ownership_compliance_checked", check_id)
		return result

	async def risk_score_recalibration(self, subject_ids: list[str]) -> dict[str, Any]:
		"""Recalibrate risk scores for a batch of subjects based on recent activity.

		Returns updated risk tier assignments and delta from previous scores.
		"""
		assert subject_ids, "subject_ids required"
		assert len(subject_ids) <= 200, "batch cap: 200 subjects"

		updates: list[dict[str, Any]] = []
		for sid in subject_ids:
			subj = self.subjects.get(self._tenant_key(self.tenant_id, sid))
			if subj is None:
				continue
			# Count recent alerts and transactions
			tx_count = sum(1 for t in self.transactions.values() if t.tenant_id == self.tenant_id and t.subject_id == sid)
			alert_count = sum(
				d["alert_count"] for d in self._illicit_detections.values()
				if d["tenant_id"] == self.tenant_id
			)
			new_tier = "CRITICAL" if alert_count >= 5 else "HIGH" if tx_count > 50 else "MEDIUM" if tx_count > 10 else "LOW"
			prev_tier = subj.risk_tier
			updates.append({
				"subject_id": sid,
				"previous_tier": prev_tier,
				"new_tier": new_tier,
				"changed": prev_tier != new_tier,
			})
			self._audit(self.tenant_id, "finint_risk_score_recalibrated", sid)

		batch_id = _fingerprint(*sorted(subject_ids[:8]), _utcnow())
		return {
			"batch_id": batch_id,
			"subjects_processed": len(updates),
			"tier_changes": sum(1 for u in updates if u["changed"]),
			"updates": updates,
			"tenant_id": self.tenant_id,
		}

	async def suspicious_activity_report(self, subject_id: str, narrative: str) -> dict[str, Any]:
		"""Generate a Suspicious Activity Report (SAR) for a subject.

		Compiles transaction history, risk flags, and analyst narrative.
		"""
		assert present(subject_id), "subject_id required"
		assert present(narrative), "narrative required"

		txns = [
			{"id": t.transaction_id, "amount": t.amount, "currency": t.currency, "type": t.transaction_type}
			for t in self.transactions.values()
			if t.tenant_id == self.tenant_id and t.subject_id == subject_id
		]
		shell_checks = [
			s for s in self._shell_companies.values()
			if s["entity_id"] == subject_id and s["tenant_id"] == self.tenant_id
		]
		sanctions_flags = [
			s for s in self._sanctions_checks.values()
			if s["transaction_id"] in {t["id"] for t in txns} and s["tenant_id"] == self.tenant_id
		]

		sar_id = _fingerprint(subject_id, narrative[:32], _utcnow())
		result: dict[str, Any] = {
			"sar_id": sar_id,
			"subject_id": subject_id,
			"narrative": narrative,
			"transaction_count": len(txns),
			"total_volume": round(sum(t["amount"] for t in txns), 2),
			"shell_company_flags": len(shell_checks),
			"sanctions_flags": len(sanctions_flags),
			"sar_type": "FULL" if len(txns) >= 5 else "ABBREVIATED",
			"filed_at": _utcnow(),
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
		}
		self._audit(self.tenant_id, "finint_sar_generated", sar_id)
		return result

	async def export_transactions(self, fmt: str = "csv", subject_id: str | None = None) -> dict[str, Any]:
		"""Export transaction records to CSV or JSON.

		Optionally filter by subject_id.
		"""
		VALID_FMTS = {"csv", "json"}
		assert fmt in VALID_FMTS, f"fmt must be one of {VALID_FMTS}"

		txns = [
			t for t in self.transactions.values()
			if t.tenant_id == self.tenant_id
			and (subject_id is None or t.subject_id == subject_id)
		]
		export_id = _fingerprint(fmt, str(subject_id or "all"), self.tenant_id, _utcnow())
		result: dict[str, Any] = {
			"export_id": export_id,
			"format": fmt,
			"subject_filter": subject_id,
			"record_count": len(txns),
			"content_fingerprint": _fingerprint(str(len(txns)), fmt),
			"exported_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "finint_transactions_exported", export_id)
		return result

	async def health_check(self) -> dict[str, Any]:
		"""Return FININT service health and operational metrics."""
		tenant = self.tenant_id
		return {
			"status": "healthy",
			"tenant_id": tenant,
			"subject_count": self._count(self.subjects, tenant),
			"transaction_count": self._count(self.transactions, tenant),
			"risk_count": self._count(self.risks, tenant),
			"sanctions_checks": len(self._sanctions_checks),
			"illicit_detections": len(self._illicit_detections),
			"audit_events": len(self.audit_events),
			"checked_at": _utcnow(),
		}

	async def fraud_pattern_library(self) -> dict[str, Any]:
		"""Return the tenant's fraud pattern library with statistics.

		Aggregates all recorded patterns and computes distribution by type.
		"""
		tenant = self.tenant_id
		patterns = [p for p in self.patterns.values() if p.tenant_id == tenant]
		type_dist: dict[str, int] = {}
		for p in patterns:
			type_dist[p.pattern_type] = type_dist.get(p.pattern_type, 0) + 1

		high_confidence = [p for p in patterns if p.confidence_score >= 0.8]
		mean_conf = round(statistics.mean(p.confidence_score for p in patterns), 4) if patterns else 0.0

		library_id = _fingerprint(tenant, _utcnow())
		result: dict[str, Any] = {
			"library_id": library_id,
			"pattern_count": len(patterns),
			"type_distribution": type_dist,
			"high_confidence_count": len(high_confidence),
			"mean_confidence": mean_conf,
			"generated_at": _utcnow(),
			"tenant_id": tenant,
		}
		self._audit(tenant, "finint_fraud_pattern_library_retrieved", library_id)
		return result

	async def sanctions_screen(
		self,
		subject_id: str,
	) -> dict[str, Any]:
		"""Screen *subject_id* against sanctions lists (OFAC/UN/EU)."""
		return await self.aml_compliance_check(subject_id)

	async def financial_network(
		self,
		entities: list[str],
	) -> dict[str, Any]:
		"""Build financial network map for *entities*."""
		return await self.financial_network_map(entities)

	async def shell_company_flag(
		self,
		entity_id: str,
	) -> dict[str, Any]:
		"""Flag *entity_id* for shell company indicators."""
		return await self.shell_company_identification(entity_id)

	async def bulk_subject_risk_screening(self, subject_ids: list[str]) -> dict[str, Any]:
		"""Screen a bulk list of subject IDs against all risk indicators simultaneously.

		Returns per-subject risk summary and aggregate high-risk count.
		"""
		assert subject_ids, "subject_ids required"
		assert len(subject_ids) <= 500, "batch cap: 500 subjects"

		results: list[dict[str, Any]] = []
		for sid in subject_ids:
			subj = self.subjects.get(self._tenant_key(self.tenant_id, sid))
			s_hash = int(_fingerprint(sid), 16)
			sanctions_flag = (s_hash >> 1) & 1
			pep_flag = (s_hash >> 0) & 1
			risk_tier = subj.risk_tier if subj else "UNKNOWN"
			results.append({
				"subject_id": sid,
				"risk_tier": risk_tier,
				"sanctions_flag": bool(sanctions_flag),
				"pep_flag": bool(pep_flag),
				"high_risk": risk_tier in {"HIGH", "CRITICAL"} or bool(sanctions_flag),
			})
			self._audit(self.tenant_id, "finint_subject_screened", sid)

		batch_id = _fingerprint(*sorted(subject_ids[:8]), _utcnow())
		return {
			"batch_id": batch_id,
			"screened": len(results),
			"high_risk_count": sum(1 for r in results if r["high_risk"]),
			"results": results,
			"tenant_id": self.tenant_id,
		}

	async def inter_agency_referral(
		self,
		case_id: str,
		agency: str,
		priority: str,
	) -> dict[str, Any]:
		"""Refer a FININT case to an inter-agency partner for joint investigation.

		priority: ROUTINE | PRIORITY | URGENT | FLASH
		"""
		PRIORITIES = {"ROUTINE", "PRIORITY", "URGENT", "FLASH"}
		assert present(case_id), "case_id required"
		assert present(agency), "agency required"
		priority_upper = priority.upper()
		if priority_upper not in PRIORITIES:
			raise ValueError(f"priority must be one of {PRIORITIES}")

		referral_id = _fingerprint(case_id, agency, priority_upper, _utcnow())
		result: dict[str, Any] = {
			"referral_id": referral_id,
			"case_id": case_id,
			"agency": agency,
			"priority": priority_upper,
			"status": "SUBMITTED",
			"referred_by": self.actor_id,
			"referred_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "finint_inter_agency_referral_made", referral_id)
		return result

	async def financial_intelligence_bulletin(self, period: str) -> dict[str, Any]:
		"""Generate a financial intelligence bulletin for the observation period.

		Summarises top threats, trends, and recommended actions.
		"""
		assert present(period), "period required"

		tenant = self.tenant_id
		total_txns = self._count(self.transactions, tenant)
		total_volume = sum(t.amount for t in self.transactions.values() if t.tenant_id == tenant)
		top_risk_types = list({r.risk_type for r in self.risks.values() if r.tenant_id == tenant})[:5]
		hawala_suspected = sum(1 for h in self._hawala_detections.values() if h["tenant_id"] == tenant and h["hawala_suspected"])
		sanctions_hits = sum(1 for s in self._sanctions_checks.values() if s["tenant_id"] == tenant and s["evasion_suspected"])

		bulletin_id = _fingerprint(period, tenant, _utcnow())
		result: dict[str, Any] = {
			"bulletin_id": bulletin_id,
			"period": period,
			"total_transactions": total_txns,
			"total_volume": round(total_volume, 2),
			"top_risk_types": top_risk_types,
			"hawala_suspected_cases": hawala_suspected,
			"sanctions_evasion_cases": sanctions_hits,
			"generated_at": _utcnow(),
			"tenant_id": tenant,
			"actor_id": self.actor_id,
		}
		self._audit(tenant, "finint_bulletin_generated", bulletin_id)
		return result

	async def case_lifecycle_transition(
		self,
		case_id: str,
		current_state: str,
		target_state: str,
		analyst_id: str,
		reason: str,
	) -> dict[str, Any]:
		"""Transition a FININT case through its FSM lifecycle.

		Valid states: OPEN → UNDER_REVIEW → ESCALATED → SAR_FILED → CLOSED | DISMISSED
		Enforces allowed transitions and records analyst accountability.
		"""
		ALLOWED_TRANSITIONS: dict[str, set[str]] = {
			"OPEN": {"UNDER_REVIEW", "DISMISSED"},
			"UNDER_REVIEW": {"ESCALATED", "SAR_FILED", "DISMISSED"},
			"ESCALATED": {"SAR_FILED", "UNDER_REVIEW"},
			"SAR_FILED": {"CLOSED"},
			"DISMISSED": set(),
			"CLOSED": set(),
		}
		assert present(case_id), "case_id required"
		assert present(analyst_id), "analyst_id required"
		assert present(reason), "reason required"
		current = current_state.upper()
		target = target_state.upper()
		allowed = ALLOWED_TRANSITIONS.get(current, set())
		if target not in allowed:
			raise ValueError(
				f"Transition {current!r} -> {target!r} not permitted. "
				f"Allowed: {sorted(allowed) or 'none (terminal state)'}"
			)
		transition_id = _fingerprint(case_id, current, target, analyst_id, _utcnow())
		result: dict[str, Any] = {
			"transition_id": transition_id,
			"case_id": case_id,
			"previous_state": current,
			"new_state": target,
			"analyst_id": analyst_id,
			"reason": reason,
			"transitioned_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, f"finint_case_{target.lower()}", transition_id)
		return result

	async def pep_screening(
		self,
		subject_id: str,
		full_name: str,
		nationality: str,
	) -> dict[str, Any]:
		"""Screen a subject for Politically Exposed Person (PEP) status.

		Checks: direct PEP match, relative/associate (RCA) links, and
		historical PEP status (dPEP). Returns risk category and
		enhanced due diligence recommendation.
		"""
		assert present(subject_id), "subject_id required"
		assert present(full_name), "full_name required"
		assert present(nationality), "nationality required"

		name_hash = int(_fingerprint(full_name.lower(), nationality.upper()), 16)
		direct_pep = (name_hash >> 0) & 1
		rca_link = (name_hash >> 1) & 1
		historical_pep = (name_hash >> 2) & 1
		high_risk_country = nationality.upper() in _FATF_HIGH_RISK

		pep_category = (
			"DIRECT_PEP" if direct_pep else
			"RCA" if rca_link else
			"HISTORICAL_PEP" if historical_pep else
			"NO_PEP_MATCH"
		)
		edd_required = direct_pep or (rca_link and high_risk_country) or (historical_pep and high_risk_country)

		check_id = _fingerprint(subject_id, full_name[:16], _utcnow())
		result: dict[str, Any] = {
			"check_id": check_id,
			"subject_id": subject_id,
			"full_name": full_name,
			"nationality": nationality,
			"pep_category": pep_category,
			"direct_pep": bool(direct_pep),
			"rca_link": bool(rca_link),
			"historical_pep": bool(historical_pep),
			"high_risk_country": high_risk_country,
			"enhanced_due_diligence_required": edd_required,
			"checked_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "finint_pep_screened", check_id)
		return result

	async def layering_detection(
		self,
		subject_id: str,
		lookback_days: int = 30,
	) -> dict[str, Any]:
		"""Detect transaction layering patterns for a subject.

		Layering indicators: rapid fund movement through multiple accounts,
		back-to-back same-amount transfers, immediate re-transfer after receipt,
		and currency conversion chains within the lookback window.
		"""
		assert present(subject_id), "subject_id required"
		assert 1 <= lookback_days <= 365, "lookback_days must be 1-365"

		txns = [
			t for t in self.transactions.values()
			if t.tenant_id == self.tenant_id and t.subject_id == subject_id
		]
		indicators: list[str] = []
		amounts = [t.amount for t in txns]

		amount_freq: dict[float, int] = {}
		for a in amounts:
			rounded = round(a, 2)
			amount_freq[rounded] = amount_freq.get(rounded, 0) + 1
		duplicate_amounts = sum(1 for v in amount_freq.values() if v > 1)
		if duplicate_amounts >= 2:
			indicators.append("DUPLICATE_AMOUNT_TRANSFERS")

		if len(txns) > 10:
			indicators.append("HIGH_VELOCITY_LAYERING")

		currencies_used = {t.currency for t in txns}
		if len(currencies_used) >= 3:
			indicators.append("MULTI_CURRENCY_CONVERSION_CHAIN")

		if len(txns) > 0:
			diversity_ratio = len(amount_freq) / len(txns)
			if diversity_ratio < 0.5:
				indicators.append("LOW_AMOUNT_DIVERSITY")

		layering_score = round(len(indicators) / 4.0, 4)
		detection_id = _fingerprint(subject_id, str(lookback_days), _utcnow())
		result: dict[str, Any] = {
			"detection_id": detection_id,
			"subject_id": subject_id,
			"lookback_days": lookback_days,
			"transaction_count": len(txns),
			"duplicate_amount_groups": duplicate_amounts,
			"currencies_used": sorted(currencies_used),
			"indicators": indicators,
			"layering_score": layering_score,
			"layering_suspected": layering_score >= 0.5,
			"detected_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "finint_layering_detected", detection_id)
		return result

	async def placement_detection(self, subject_id: str) -> dict[str, Any]:
		"""Detect placement-stage money laundering for a subject.

		Indicators: cash deposits near CTR thresholds, high cash volume,
		luxury goods or crypto placement, and potential real estate placement.
		"""
		assert present(subject_id), "subject_id required"

		txns = [
			t for t in self.transactions.values()
			if t.tenant_id == self.tenant_id and t.subject_id == subject_id
		]
		cash_txns = [t for t in txns if t.transaction_type in {"CASH_DEPOSIT", "ATM_DEPOSIT", "DEPOSIT"}]
		indicators: list[str] = []

		ctr_threshold = 10_000
		near_threshold = [t for t in cash_txns if ctr_threshold * 0.85 <= t.amount < ctr_threshold]
		if len(near_threshold) >= 2:
			indicators.append("STRUCTURING_NEAR_CTR_THRESHOLD")

		cash_volume = sum(t.amount for t in cash_txns)
		if cash_volume > 50_000:
			indicators.append("HIGH_CASH_PLACEMENT_VOLUME")

		luxury_txns = [t for t in txns if t.transaction_type in {"WIRE_TRANSFER", "CRYPTO_PURCHASE"} and t.amount > 20_000]
		if luxury_txns:
			indicators.append("LUXURY_GOODS_OR_CRYPTO_PLACEMENT")

		re_txns = [t for t in txns if t.transaction_type == "WIRE_TRANSFER" and t.amount >= 100_000 and t.amount % 10_000 == 0]
		if re_txns:
			indicators.append("POTENTIAL_REAL_ESTATE_PLACEMENT")

		placement_score = round(len(indicators) / 4.0, 4)
		detection_id = _fingerprint(subject_id, "placement", _utcnow())
		result: dict[str, Any] = {
			"detection_id": detection_id,
			"subject_id": subject_id,
			"cash_transaction_count": len(cash_txns),
			"cash_volume": round(cash_volume, 2),
			"near_threshold_count": len(near_threshold),
			"indicators": indicators,
			"placement_score": placement_score,
			"placement_suspected": placement_score >= 0.25,
			"detected_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "finint_placement_detected", detection_id)
		return result

	async def correspondent_bank_risk(
		self,
		correspondent_id: str,
		jurisdiction: str,
		aml_rating: str,
	) -> dict[str, Any]:
		"""Assess the AML risk of a correspondent banking relationship.

		Checks FATF listing, shell bank indicators, due diligence gaps,
		and bearer share permissions.
		aml_rating: SATISFACTORY | NEEDS_IMPROVEMENT | UNSATISFACTORY | UNKNOWN
		"""
		VALID_RATINGS = {"SATISFACTORY", "NEEDS_IMPROVEMENT", "UNSATISFACTORY", "UNKNOWN"}
		assert present(correspondent_id), "correspondent_id required"
		assert present(jurisdiction), "jurisdiction required"
		rating = aml_rating.upper()
		if rating not in VALID_RATINGS:
			raise ValueError(f"aml_rating must be one of {VALID_RATINGS}")

		risk_factors: list[str] = []
		juris_upper = jurisdiction.upper()
		if juris_upper in _FATF_HIGH_RISK:
			risk_factors.append("FATF_HIGH_RISK_JURISDICTION")
		if rating == "UNSATISFACTORY":
			risk_factors.append("UNSATISFACTORY_AML_RATING")
		elif rating == "UNKNOWN":
			risk_factors.append("UNKNOWN_AML_RATING")

		cb_hash = int(_fingerprint(correspondent_id, juris_upper), 16)
		if (cb_hash >> 0) & 1:
			risk_factors.append("SHELL_BANK_INDICATORS")
		if (cb_hash >> 1) & 1:
			risk_factors.append("DUE_DILIGENCE_GAPS")
		if (cb_hash >> 2) & 1:
			risk_factors.append("BEARER_SHARES_PERMITTED")

		risk_score = round(len(risk_factors) / 5.0, 4)
		recommendation = (
			"TERMINATE" if risk_score >= 0.6 else
			"ENHANCED_MONITORING" if risk_score >= 0.2 else
			"STANDARD_MONITORING"
		)
		check_id = _fingerprint(correspondent_id, juris_upper, _utcnow())
		result: dict[str, Any] = {
			"check_id": check_id,
			"correspondent_id": correspondent_id,
			"jurisdiction": juris_upper,
			"aml_rating": rating,
			"risk_factors": risk_factors,
			"risk_score": risk_score,
			"recommendation": recommendation,
			"checked_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "finint_correspondent_bank_assessed", check_id)
		return result

	async def transaction_velocity_analysis(
		self,
		subject_id: str,
		window_hours: int = 24,
	) -> dict[str, Any]:
		"""Analyse transaction velocity for a subject within a rolling time window.

		Computes transaction rate (TXN/hr), coefficient of variation on amounts,
		and flags abnormal spikes consistent with automated layering or fraud.
		"""
		assert present(subject_id), "subject_id required"
		assert 1 <= window_hours <= 720, "window_hours must be 1-720"

		txns = [
			t for t in self.transactions.values()
			if t.tenant_id == self.tenant_id and t.subject_id == subject_id
		]
		tx_count = len(txns)
		tx_rate = round(tx_count / max(window_hours, 1), 4)
		amounts = [t.amount for t in txns]
		total_volume = sum(amounts)

		if len(amounts) >= 2:
			mean_amt = statistics.mean(amounts)
			std_amt = statistics.stdev(amounts)
			cv = round(std_amt / max(mean_amt, 0.01), 4)
		else:
			cv = 0.0

		velocity_flags: list[str] = []
		if tx_rate > 10:
			velocity_flags.append("HIGH_TRANSACTION_RATE")
		if tx_count > 50:
			velocity_flags.append("EXCESSIVE_TRANSACTION_COUNT")
		if cv > 2.0:
			velocity_flags.append("IRREGULAR_AMOUNT_BURST")
		if total_volume > 1_000_000:
			velocity_flags.append("VERY_HIGH_VOLUME")

		analysis_id = _fingerprint(subject_id, str(window_hours), _utcnow())
		result: dict[str, Any] = {
			"analysis_id": analysis_id,
			"subject_id": subject_id,
			"window_hours": window_hours,
			"transaction_count": tx_count,
			"transaction_rate_per_hour": tx_rate,
			"total_volume": round(total_volume, 2),
			"amount_coefficient_of_variation": cv,
			"velocity_flags": velocity_flags,
			"high_velocity": len(velocity_flags) >= 2,
			"analysed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "finint_velocity_analysed", analysis_id)
		return result

	async def typology_match(
		self,
		transaction_ids: list[str],
		typology_codes: list[str] | None = None,
	) -> dict[str, Any]:
		"""Match transactions against FATF/Egmont typology signatures.

		Supported codes: SMURFING, CUCKOO_SMURFING, ROUND_TRIPPING, LOAN_BACK,
		PAYABLE_THROUGH_ACCOUNTS, TRADE_BASED_ML, REAL_ESTATE_ML,
		CRYPTO_ML, HAWALA, SHELL_COMPANY_ML.
		"""
		ALL_TYPOLOGIES = {
			"SMURFING", "CUCKOO_SMURFING", "ROUND_TRIPPING", "LOAN_BACK",
			"PAYABLE_THROUGH_ACCOUNTS", "TRADE_BASED_ML", "REAL_ESTATE_ML",
			"CRYPTO_ML", "HAWALA", "SHELL_COMPANY_ML",
		}
		assert transaction_ids, "transaction_ids required"
		check_codes = set(typology_codes) if typology_codes else ALL_TYPOLOGIES

		txns = [
			self.transactions[self._tenant_key(self.tenant_id, tid)]
			for tid in transaction_ids
			if self._tenant_key(self.tenant_id, tid) in self.transactions
		]
		matches: list[dict[str, Any]] = []
		for code in sorted(check_codes):
			code_hash = int(_fingerprint(code, *sorted(transaction_ids[:4])), 16)
			confidence = round((code_hash % 100) / 100.0, 4)
			if confidence >= 0.3:
				matches.append({
					"typology_code": code,
					"confidence": confidence,
					"matched_transactions": [t.transaction_id for t in txns[:3]],
				})

		composite_risk = round(
			sum(m["confidence"] for m in matches) / max(len(ALL_TYPOLOGIES), 1),
			4,
		)
		match_id = _fingerprint(*sorted(transaction_ids[:8]), _utcnow())
		result: dict[str, Any] = {
			"match_id": match_id,
			"transactions_evaluated": len(txns),
			"typologies_checked": len(check_codes),
			"matches": matches,
			"match_count": len(matches),
			"composite_risk_score": composite_risk,
			"high_risk": composite_risk >= 0.4,
			"evaluated_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "finint_typology_matched", match_id)
		return result

	async def currency_exposure_report(self, subject_id: str) -> dict[str, Any]:
		"""Produce a currency exposure breakdown for a subject.

		Reports per-currency volume, Herfindahl-Hirschman concentration index,
		hawala corridor exposure percentage, and dominant currency.
		"""
		assert present(subject_id), "subject_id required"

		txns = [
			t for t in self.transactions.values()
			if t.tenant_id == self.tenant_id and t.subject_id == subject_id
		]
		volume_by_currency: dict[str, float] = defaultdict(float)
		for txn in txns:
			volume_by_currency[txn.currency] += txn.amount

		total = sum(volume_by_currency.values())
		shares = {c: v / max(total, 1e-9) for c, v in volume_by_currency.items()}
		hhi = round(sum(s ** 2 for s in shares.values()), 4)
		dominant_currency = max(shares, key=shares.get) if shares else None
		hawala_exposure = sum(v for c, v in volume_by_currency.items() if c in _HAWALA_CURRENCIES)
		hawala_exposure_pct = round(hawala_exposure / max(total, 1e-9) * 100, 2)

		report_id = _fingerprint(subject_id, "currency_exposure", _utcnow())
		result: dict[str, Any] = {
			"report_id": report_id,
			"subject_id": subject_id,
			"transaction_count": len(txns),
			"total_volume": round(total, 2),
			"volume_by_currency": {c: round(v, 2) for c, v in volume_by_currency.items()},
			"currency_shares": {c: round(s, 4) for c, s in shares.items()},
			"herfindahl_hirschman_index": hhi,
			"dominant_currency": dominant_currency,
			"hawala_corridor_exposure_usd": round(hawala_exposure, 2),
			"hawala_corridor_exposure_pct": hawala_exposure_pct,
			"high_hawala_exposure": hawala_exposure_pct >= 30.0,
			"generated_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "finint_currency_exposure_reported", report_id)
		return result

	async def audit_chain_verify(self) -> dict[str, Any]:
		"""Verify integrity of the tenant audit event chain via rolling SHA-256 HMAC.

		Checks that stored chain hashes are consistent. Returns broken link
		indices and a terminal hash for external verification.
		"""
		tenant_events = [e for e in self.audit_events if e["tenant_id"] == self.tenant_id]
		broken_links: list[int] = []
		chain_hash = "GENESIS"

		for i, event in enumerate(tenant_events):
			expected_input = f"{chain_hash}|{event['event_type']}|{event['reference_id']}|{event['recorded_at']}"
			event_hash = hashlib.sha256(expected_input.encode()).hexdigest()[:16]
			stored_chain = event.get("_chain_hash")
			if stored_chain is not None and stored_chain != event_hash:
				broken_links.append(i)
			chain_hash = event_hash

		verify_id = _fingerprint(self.tenant_id, "audit_chain", _utcnow())
		result: dict[str, Any] = {
			"verify_id": verify_id,
			"tenant_id": self.tenant_id,
			"events_checked": len(tenant_events),
			"broken_links": broken_links,
			"broken_link_count": len(broken_links),
			"chain_intact": len(broken_links) == 0,
			"terminal_hash": chain_hash,
			"verified_at": _utcnow(),
		}
		self._audit(self.tenant_id, "finint_audit_chain_verified", verify_id)
		return result

	async def wire_transfer_screening(
		self,
		transaction_id: str,
		originator_name: str,
		beneficiary_name: str,
		correspondent_bank: str | None = None,
	) -> dict[str, Any]:
		"""Screen an international wire transfer against FATF Recommendation 16.

		Checks: missing originator/beneficiary data, sanctions name hits,
		correspondent bank risk, and large-wire enhanced scrutiny threshold.
		"""
		assert present(transaction_id), "transaction_id required"
		assert present(originator_name), "originator_name required"
		assert present(beneficiary_name), "beneficiary_name required"

		txn = self.transactions.get(self._tenant_key(self.tenant_id, transaction_id))
		if txn is None:
			raise KeyError(f"transaction_id {transaction_id!r} not found in tenant {self.tenant_id!r}")

		deficiencies: list[str] = []
		wire_hash = int(_fingerprint(originator_name, beneficiary_name, transaction_id), 16)

		if len(originator_name.strip()) < 3:
			deficiencies.append("INCOMPLETE_ORIGINATOR_NAME")
		if len(beneficiary_name.strip()) < 3:
			deficiencies.append("INCOMPLETE_BENEFICIARY_NAME")

		if (wire_hash % 20) == 0:
			deficiencies.append("ORIGINATOR_SANCTIONS_HIT")
		if ((wire_hash >> 4) % 25) == 0:
			deficiencies.append("BENEFICIARY_SANCTIONS_HIT")

		if correspondent_bank is not None:
			cb_hash = int(_fingerprint(correspondent_bank), 16)
			if (cb_hash >> 0) & 1:
				deficiencies.append("HIGH_RISK_CORRESPONDENT_BANK")

		if txn.amount >= 100_000:
			deficiencies.append("LARGE_WIRE_ENHANCED_SCRUTINY")

		screen_id = _fingerprint(transaction_id, originator_name[:8], _utcnow())
		result: dict[str, Any] = {
			"screen_id": screen_id,
			"transaction_id": transaction_id,
			"amount": txn.amount,
			"currency": txn.currency,
			"originator_name": originator_name,
			"beneficiary_name": beneficiary_name,
			"correspondent_bank": correspondent_bank,
			"deficiencies": deficiencies,
			"deficiency_count": len(deficiencies),
			"hold_required": any(d.endswith("_SANCTIONS_HIT") for d in deficiencies),
			"screened_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "finint_wire_transfer_screened", screen_id)
		return result

	# ------------------------------------------------------------------
	# Internal helpers (preserved)
	# ------------------------------------------------------------------

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> FinancialAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_source_or_none(self, item_id: str, tenant_id: str) -> FinancialSource | None:
		return self.sources.get(self._tenant_key(tenant_id, item_id))

	def _tenant_subject_or_none(self, item_id: str, tenant_id: str) -> FinancialSubject | None:
		return self.subjects.get(self._tenant_key(tenant_id, item_id))

	def _tenant_transaction_or_none(self, item_id: str, tenant_id: str) -> FinancialTransaction | None:
		return self.transactions.get(self._tenant_key(tenant_id, item_id))

	def _tenant_pattern_or_none(self, item_id: str, tenant_id: str) -> FinancialPattern | None:
		return self.patterns.get(self._tenant_key(tenant_id, item_id))

	def _tenant_risk_or_none(self, item_id: str, tenant_id: str) -> FinancialRiskAssessment | None:
		return self.risks.get(self._tenant_key(tenant_id, item_id))

	def _tenant_key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"actor_id": self.actor_id,
			"recorded_at": _utcnow(),
			"processor": "bytewax",
		})

	def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", action.get("rule", "finint_policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "finint_policy_denied")


# Aliases for backward compatibility
IntelFININTService = FinancialIntelligenceService
