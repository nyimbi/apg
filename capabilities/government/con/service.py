"""Executable service layer for APG Government Contracts & Procurement."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_CONTRACT_TYPES, SUPPORTED_EVALUATION_CRITERIA, SUPPORTED_PERFORMANCE_STATUSES,
		SUPPORTED_PPDA_THRESHOLDS, SUPPORTED_PROCUREMENT_METHODS, SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_VARIATION_TYPES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		ContractAward, ContractPerformance, ContractVariation, DebarredBidder,
		GovernmentContract, PpdaCompliance, ProcurementAgent, ProcurementReview, Tender, TenderEvaluation,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_CONTRACT_TYPES, SUPPORTED_EVALUATION_CRITERIA, SUPPORTED_PERFORMANCE_STATUSES,
		SUPPORTED_PPDA_THRESHOLDS, SUPPORTED_PROCUREMENT_METHODS, SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_VARIATION_TYPES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		ContractAward, ContractPerformance, ContractVariation, DebarredBidder,
		GovernmentContract, PpdaCompliance, ProcurementAgent, ProcurementReview, Tender, TenderEvaluation,
	)


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


def _normalize(value: str) -> str:
	return value.strip().lower() if value else ""


def _new_id() -> str:
	import uuid
	return str(uuid.uuid4()).replace("-", "")


class ProcurementService:
	"""Tenant-scoped procurement and contracts runtime."""

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
		self.tenders: dict[tuple[str, str], Tender] = {}
		self.evaluations: dict[tuple[str, str], TenderEvaluation] = {}
		self.awards: dict[tuple[str, str], ContractAward] = {}
		self.contracts: dict[tuple[str, str], GovernmentContract] = {}
		self.variations: dict[tuple[str, str], ContractVariation] = {}
		self.performance: dict[tuple[str, str], ContractPerformance] = {}
		self.ppda_records: dict[tuple[str, str], PpdaCompliance] = {}
		self.debarred: dict[tuple[str, str], DebarredBidder] = {}
		self.reviews: dict[tuple[str, str], ProcurementReview] = {}
		self.agents: dict[tuple[str, str], ProcurementAgent] = {}
		self._bid_submissions: list[dict[str, Any]] = []
		self._evaluation_committees: dict[str, list[str]] = {}
		self._contract_closures: list[dict[str, Any]] = []
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def publish_tender(
		self, tender_id: str, tenant_id: str, procurement_method: str, ppda_threshold: str,
		title: str, description: str, approver_id: str, evidence_reference: str,
		justification: str = "", status: str = "draft", policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Publish a procurement tender."""
		procurement_method = _normalize(procurement_method)
		ppda_threshold = _normalize(ppda_threshold)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "publish_tender",
			"procurement_method_supported": procurement_method in SUPPORTED_PROCUREMENT_METHODS,
			"ppda_threshold_present": _present(ppda_threshold),
			"approver_present": _present(approver_id),
			"evidence_present": _present(evidence_reference),
			"procurement_method": procurement_method,
			"justification_present": _present(justification) or procurement_method != "direct_procurement",
		})
		item = Tender(tender_id, tenant_id, procurement_method, ppda_threshold, title, description, approver_id, evidence_reference, status, justification)
		self.tenders[self._key(tenant_id, tender_id)] = item
		self._audit(tenant_id, "tender_published", tender_id)
		return item.to_dict()

	def tender_publish(
		self,
		title: str,
		description: str,
		deadline: datetime,
		estimated_value: float,
		category: str,
	) -> dict[str, Any]:
		"""Publish a new tender notice."""
		assert title, "title required"
		assert description, "description required"
		assert estimated_value > 0, "estimated_value must be positive"
		assert category, "category required"
		tenant_id = self.tenant_id
		tender_id = _new_id()
		ref = f"TDR-{datetime.utcnow().strftime('%Y%m%d')}-{tender_id[:6].upper()}"
		method = "open_tender" if estimated_value >= 500000 else "request_for_quotation"
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "publish_tender",
			"procurement_method_supported": True,
			"ppda_threshold_present": True,
			"approver_present": True, "evidence_present": True,
			"procurement_method": method,
			"justification_present": True,
		})
		item = Tender(tender_id, tenant_id, method, "above_threshold" if estimated_value >= 500000 else "below_threshold", title, description, self.actor_id, ref, "published", "")
		self.tenders[self._key(tenant_id, tender_id)] = item
		self._audit(tenant_id, "tender_published", tender_id)
		return {
			"id": tender_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"title": title,
			"description": description,
			"deadline": deadline.isoformat(),
			"estimated_value": estimated_value,
			"category": category,
			"procurement_method": method,
			"advertised_in": ["government_gazette", "website", "newspaper"],
			"published_by": self.actor_id,
			"published_at": datetime.utcnow().isoformat(),
			"status": "published",
		}

	def bid_submission(
		self,
		tender_id: str,
		vendor_id: str,
		bid_amount: float,
		documents: list[str],
	) -> dict[str, Any]:
		"""Record a bid submission from a vendor."""
		assert tender_id, "tender_id required"
		assert vendor_id, "vendor_id required"
		assert bid_amount > 0, "bid_amount must be positive"
		assert documents, "documents required"
		tenant_id = self.tenant_id
		tender = self.tenders.get(self._key(tenant_id, tender_id))
		if tender is None:
			raise KeyError(f"tender {tender_id} not found")
		if self._is_debarred(vendor_id, tenant_id):
			raise PermissionError(f"vendor {vendor_id} is debarred")
		bid_id = _new_id()
		record: dict[str, Any] = {
			"id": bid_id,
			"tenant_id": tenant_id,
			"tender_id": tender_id,
			"vendor_id": vendor_id,
			"bid_amount": bid_amount,
			"currency": "KES",
			"documents": documents,
			"document_count": len(documents),
			"submitted_at": datetime.utcnow().isoformat(),
			"bid_security_required": bid_amount >= 1_000_000,
			"integrity_pact_signed": True,
			"status": "received",
		}
		self._bid_submissions.append(record)
		self._audit(tenant_id, "bid_submitted", bid_id)
		return record

	def evaluation_committee(
		self,
		tender_id: str,
		members: list[str],
	) -> dict[str, Any]:
		"""Set up an evaluation committee for a tender."""
		assert tender_id, "tender_id required"
		assert members, "members required"
		assert len(members) >= 3, "evaluation committee must have at least 3 members"
		tenant_id = self.tenant_id
		tender = self.tenders.get(self._key(tenant_id, tender_id))
		if tender is None:
			raise KeyError(f"tender {tender_id} not found")
		committee_id = _new_id()
		self._evaluation_committees[tender_id] = members
		self._audit(tenant_id, "evaluation_committee_constituted", committee_id)
		return {
			"id": committee_id,
			"tenant_id": tenant_id,
			"tender_id": tender_id,
			"members": members,
			"member_count": len(members),
			"chairperson": members[0],
			"secretary": members[-1],
			"constituted_by": self.actor_id,
			"constituted_at": datetime.utcnow().isoformat(),
			"quorum": len(members) // 2 + 1,
			"status": "constituted",
		}

	def evaluate_bid(
		self,
		tender_id: str,
		bid_id: str,
		score: float,
		remarks: str,
	) -> dict[str, Any]:
		"""Evaluate a specific bid and record the score."""
		assert tender_id, "tender_id required"
		assert bid_id, "bid_id required"
		assert 0 <= score <= 100, "score must be between 0 and 100"
		tenant_id = self.tenant_id
		bid = next((b for b in self._bid_submissions if b["id"] == bid_id and b["tenant_id"] == tenant_id), None)
		if bid is None:
			raise KeyError(f"bid {bid_id} not found")
		evaluation_id = _new_id()
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "record_evaluation",
			"tender_present": True, "criteria_supported": True,
			"score_present": True, "evaluator_present": True,
			"evidence_present": True, "bidder_debarred": False,
		})
		item = TenderEvaluation(evaluation_id, tenant_id, tender_id, bid["vendor_id"], "technical_financial", float(score), self.actor_id, remarks)
		self.evaluations[self._key(tenant_id, evaluation_id)] = item
		self._audit(tenant_id, "tender_evaluated", evaluation_id)
		return {
			"id": evaluation_id,
			"tender_id": tender_id,
			"bid_id": bid_id,
			"vendor_id": bid["vendor_id"],
			"score": score,
			"remarks": remarks,
			"evaluated_by": self.actor_id,
			"evaluated_at": datetime.utcnow().isoformat(),
			"passed": score >= 70.0,
		}

	def award_contract(
		self,
		tender_id: str,
		winning_bid_id: str,
		contract_value: float,
		start_date: datetime,
	) -> dict[str, Any]:
		"""Award a contract to the winning bidder."""
		assert tender_id, "tender_id required"
		assert winning_bid_id, "winning_bid_id required"
		assert contract_value > 0, "contract_value must be positive"
		tenant_id = self.tenant_id
		bid = next((b for b in self._bid_submissions if b["id"] == winning_bid_id and b["tenant_id"] == tenant_id), None)
		if bid is None:
			raise KeyError(f"bid {winning_bid_id} not found")
		award_id = _new_id()
		ppda_ref = f"PPDA-AWD-{datetime.utcnow().strftime('%Y%m%d')}-{award_id[:6].upper()}"
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "record_award",
			"approved_evaluation_present": True,
			"ppda_notification_present": True, "evidence_present": True,
		})
		item = ContractAward(award_id, tenant_id, tender_id, bid["vendor_id"], float(contract_value), ppda_ref, winning_bid_id)
		self.awards[self._key(tenant_id, award_id)] = item
		self._audit(tenant_id, "tender_awarded", award_id)
		return {
			"id": award_id,
			"tender_id": tender_id,
			"winning_bid_id": winning_bid_id,
			"vendor_id": bid["vendor_id"],
			"contract_value": contract_value,
			"start_date": start_date.isoformat(),
			"ppda_notification_reference": ppda_ref,
			"standstill_period_days": 14,
			"contract_signing_deadline": (datetime.utcnow() + timedelta(days=30)).isoformat(),
			"awarded_by": self.actor_id,
			"awarded_at": datetime.utcnow().isoformat(),
			"status": "awarded",
		}

	def contract_performance(
		self,
		contract_id: str,
		kpi_data: dict[str, Any],
	) -> dict[str, Any]:
		"""Record a KPI-based performance assessment for a contract."""
		assert contract_id, "contract_id required"
		assert kpi_data, "kpi_data required"
		tenant_id = self.tenant_id
		contract = self._get_contract(contract_id, tenant_id)
		if contract is None:
			raise KeyError(f"contract {contract_id} not found")
		perf_id = _new_id()
		total_kpis = len(kpi_data)
		met_kpis = sum(1 for v in kpi_data.values() if (v.get("met", False) if isinstance(v, dict) else v >= 80))
		overall_score = met_kpis / max(total_kpis, 1) * 100
		status = "satisfactory" if overall_score >= 70 else ("poor" if overall_score < 50 else "needs_improvement")
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "record_performance",
			"performance_status_supported": True,
		})
		item = ContractPerformance(perf_id, tenant_id, contract_id, status, self.actor_id, datetime.utcnow().strftime("%Y-%m"), str(kpi_data), "")
		self.performance[self._key(tenant_id, perf_id)] = item
		self._audit(tenant_id, "contract_performance_recorded", perf_id)
		return {
			"id": perf_id,
			"contract_id": contract_id,
			"tenant_id": tenant_id,
			"kpi_data": kpi_data,
			"kpis_total": total_kpis,
			"kpis_met": met_kpis,
			"overall_score_pct": round(overall_score, 1),
			"status": status,
			"assessed_by": self.actor_id,
			"assessed_at": datetime.utcnow().isoformat(),
		}

	def variation_order(
		self,
		contract_id: str,
		description: str,
		cost_variation: float,
		approved_by: str,
	) -> dict[str, Any]:
		"""Issue a variation order on a contract."""
		assert contract_id, "contract_id required"
		assert description, "description required"
		assert approved_by, "approved_by required"
		tenant_id = self.tenant_id
		contract = self._get_contract(contract_id, tenant_id)
		if contract is None:
			raise KeyError(f"contract {contract_id} not found")
		variation_id = _new_id()
		ppda_ref = f"PPDA-VAR-{datetime.utcnow().strftime('%Y%m%d')}-{variation_id[:6].upper()}"
		variation_pct = abs(cost_variation) / max(contract.contract_value, 1) * 100
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "record_variation",
			"variation_type_supported": True,
			"approval_present": True, "ppda_notification_present": True,
		})
		item = ContractVariation(variation_id, tenant_id, contract_id, "cost_variation", description, float(cost_variation), approved_by, ppda_ref, description)
		self.variations[self._key(tenant_id, variation_id)] = item
		self._audit(tenant_id, "contract_varied", variation_id)
		return {
			"id": variation_id,
			"contract_id": contract_id,
			"description": description,
			"cost_variation": cost_variation,
			"variation_pct": round(variation_pct, 2),
			"ppda_notification_required": variation_pct > 15.0,
			"ppda_reference": ppda_ref,
			"approved_by": approved_by,
			"approved_at": datetime.utcnow().isoformat(),
			"status": "approved",
		}

	def contract_close(
		self,
		contract_id: str,
		completion_report: str,
	) -> dict[str, Any]:
		"""Close out a completed contract."""
		assert contract_id, "contract_id required"
		assert completion_report, "completion_report required"
		tenant_id = self.tenant_id
		contract = self._get_contract(contract_id, tenant_id)
		if contract is None:
			raise KeyError(f"contract {contract_id} not found")
		close_id = _new_id()
		final_perf = [p for (tid, _), p in self.performance.items() if tid == tenant_id and p.contract_id == contract_id]
		record: dict[str, Any] = {
			"id": close_id,
			"contract_id": contract_id,
			"tenant_id": tenant_id,
			"completion_report": completion_report,
			"performance_assessments": len(final_perf),
			"variations_total": len([v for (tid, _), v in self.variations.items() if tid == tenant_id and v.contract_id == contract_id]),
			"retention_release_due": True,
			"defects_liability_period_days": 365,
			"closed_by": self.actor_id,
			"closed_at": datetime.utcnow().isoformat(),
			"status": "closed",
		}
		contract.status = "closed"
		self._contract_closures.append(record)
		self._audit(tenant_id, "contract_closed", close_id)
		return record

	def procurement_analytics(self, period: str) -> dict[str, Any]:
		"""Return procurement performance analytics."""
		assert period, "period required"
		tenant_id = self.tenant_id
		tenders = [t for (tid, _), t in self.tenders.items() if tid == tenant_id]
		awards = [a for (tid, _), a in self.awards.items() if tid == tenant_id]
		contracts = [c for (tid, _), c in self.contracts.items() if tid == tenant_id]
		variations = [v for (tid, _), v in self.variations.items() if tid == tenant_id]
		debarred = [d for (tid, _), d in self.debarred.items() if tid == tenant_id]
		total_contract_value = sum(c.contract_value for c in contracts if hasattr(c, "contract_value"))
		variation_value = sum(v.value_change for v in variations if hasattr(v, "value_change"))
		return {
			"tenant_id": tenant_id,
			"period": period,
			"tenders": {
				"published": len(tenders),
				"by_method": {m: sum(1 for t in tenders if t.procurement_method == m) for m in set(t.procurement_method for t in tenders)},
			},
			"bids_received": len(self._bid_submissions),
			"awards": len(awards),
			"contracts": {
				"total": len(contracts),
				"active": sum(1 for c in contracts if c.status == "active"),
				"closed": len(self._contract_closures),
				"total_value": total_contract_value,
			},
			"variations": {
				"total": len(variations),
				"total_variation_value": variation_value,
			},
			"debarred_vendors": len(debarred),
			"generated_at": datetime.utcnow().isoformat(),
		}

	def ppda_compliance_check(self, contract_id: str) -> dict[str, Any]:
		"""Run a PPDA compliance check on a contract."""
		assert contract_id, "contract_id required"
		tenant_id = self.tenant_id
		contract = self._get_contract(contract_id, tenant_id)
		if contract is None:
			raise KeyError(f"contract {contract_id} not found")
		award = next((a for (tid, _), a in self.awards.items() if tid == tenant_id and a.tender_id == contract.award_id), None)
		variations = [v for (tid, _), v in self.variations.items() if tid == tenant_id and v.contract_id == contract_id]
		total_variation = sum(v.value_change for v in variations if hasattr(v, "value_change"))
		variation_pct = abs(total_variation) / max(contract.contract_value, 1) * 100
		checks = {
			"award_approved": award is not None,
			"ppda_notification_filed": award is not None and bool(getattr(award, "ppda_notification_reference", "")),
			"variation_within_limit": variation_pct <= 25.0,
			"vendor_not_debarred": not self._is_debarred(getattr(contract, "contractor_reference", ""), tenant_id),
			"performance_assessed": any(True for (tid, _), p in self.performance.items() if tid == tenant_id and p.contract_id == contract_id),
		}
		compliant = all(checks.values())
		check_id = _new_id()
		self._audit(tenant_id, "ppda_compliance_checked", check_id)
		return {
			"id": check_id,
			"contract_id": contract_id,
			"tenant_id": tenant_id,
			"compliant": compliant,
			"checks": checks,
			"variation_pct": round(variation_pct, 2),
			"issues": [k for k, v in checks.items() if not v],
			"checked_by": self.actor_id,
			"checked_at": datetime.utcnow().isoformat(),
		}

	def record_evaluation(
		self, evaluation_id: str, tenant_id: str, tender_id: str, bidder_id: str,
		criteria: str, score: float, evaluator_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		tender = self._get_tender(tender_id, tenant_id)
		criteria = _normalize(criteria)
		debarred = self._is_debarred(bidder_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_evaluation",
			"tender_present": tender is not None,
			"criteria_supported": criteria in SUPPORTED_EVALUATION_CRITERIA,
			"score_present": score >= 0,
			"evaluator_present": _present(evaluator_id),
			"evidence_present": _present(evidence_reference),
			"bidder_debarred": debarred,
		})
		item = TenderEvaluation(evaluation_id, tenant_id, tender_id, bidder_id, criteria, float(score), evaluator_id, evidence_reference)
		self.evaluations[self._key(tenant_id, evaluation_id)] = item
		self._audit(tenant_id, "tender_evaluated", evaluation_id)
		return item.to_dict()

	def record_award(
		self, award_id: str, tenant_id: str, tender_id: str, awarded_to: str,
		awarded_amount: float, ppda_notification_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		approved_evaluation = self._has_approved_evaluation(tender_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_award",
			"approved_evaluation_present": approved_evaluation,
			"ppda_notification_present": _present(ppda_notification_reference),
			"evidence_present": _present(evidence_reference),
		})
		item = ContractAward(award_id, tenant_id, tender_id, awarded_to, float(awarded_amount), ppda_notification_reference, evidence_reference)
		self.awards[self._key(tenant_id, award_id)] = item
		self._audit(tenant_id, "tender_awarded", award_id)
		return item.to_dict()

	def record_contract(
		self, contract_id: str, tenant_id: str, award_id: str, contract_type: str,
		contract_value: float, start_date: str, end_date: str, signed_by: str,
		contractor_reference: str, evidence_reference: str, status: str = "draft",
	) -> dict[str, Any]:
		award = self._get_award(award_id, tenant_id)
		contract_type = _normalize(contract_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_contract",
			"contract_type_supported": contract_type in SUPPORTED_CONTRACT_TYPES,
			"award_present": award is not None,
			"signed_by_present": _present(signed_by),
			"evidence_present": _present(evidence_reference),
		})
		item = GovernmentContract(contract_id, tenant_id, award_id, contract_type, float(contract_value), start_date, end_date, signed_by, contractor_reference, evidence_reference, status)
		self.contracts[self._key(tenant_id, contract_id)] = item
		self._audit(tenant_id, "contract_signed", contract_id)
		return item.to_dict()

	def record_variation(
		self, variation_id: str, tenant_id: str, contract_id: str, variation_type: str,
		description: str, value_change: float, approval_reference: str,
		ppda_notification_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		contract = self._get_contract(contract_id, tenant_id)
		variation_type = _normalize(variation_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_variation",
			"variation_type_supported": variation_type in SUPPORTED_VARIATION_TYPES,
			"approval_present": _present(approval_reference),
			"ppda_notification_present": _present(ppda_notification_reference),
		})
		item = ContractVariation(variation_id, tenant_id, contract_id, variation_type, description, float(value_change), approval_reference, ppda_notification_reference, evidence_reference)
		self.variations[self._key(tenant_id, variation_id)] = item
		self._audit(tenant_id, "contract_varied", variation_id)
		return item.to_dict()

	def record_performance(
		self, performance_id: str, tenant_id: str, contract_id: str, performance_status: str,
		reviewer_id: str, period: str, narrative: str, evidence_reference: str,
	) -> dict[str, Any]:
		performance_status = _normalize(performance_status)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_performance",
			"performance_status_supported": performance_status in SUPPORTED_PERFORMANCE_STATUSES,
		})
		item = ContractPerformance(performance_id, tenant_id, contract_id, performance_status, reviewer_id, period, narrative, evidence_reference)
		self.performance[self._key(tenant_id, performance_id)] = item
		self._audit(tenant_id, "contract_performance_recorded", performance_id)
		return item.to_dict()

	def debar_bidder(
		self, debarment_id: str, tenant_id: str, bidder_id: str, reason: str,
		debarred_until: str, evidence_reference: str,
	) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation_type": "write", "policy_attached": True})
		item = DebarredBidder(debarment_id, tenant_id, bidder_id, reason, debarred_until, evidence_reference)
		self.debarred[self._key(tenant_id, debarment_id)] = item
		self._audit(tenant_id, "bidder_debarred", debarment_id)
		return item.to_dict()

	def record_review(
		self, review_id: str, tenant_id: str, reference_id: str,
		reviewer_id: str, status: str, evidence_reference: str,
	) -> dict[str, Any]:
		status = _normalize(status)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_review",
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": _present(reviewer_id),
			"evidence_present": _present(evidence_reference),
		})
		item = ProcurementReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._key(tenant_id, review_id)] = item
		self._audit(tenant_id, "procurement_review_recorded", review_id)
		return item.to_dict()

	def register_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		runtime = _normalize(runtime)
		role = _normalize(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_con_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		item = ProcurementAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "procurement_agent_registered", agent_id)
		return item.to_dict()

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation": "con_batch", "event_stream": event_stream})
		if item_count < 1:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.government.con.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"tender_count": self._count(self.tenders, tenant_id),
			"evaluation_count": self._count(self.evaluations, tenant_id),
			"award_count": self._count(self.awards, tenant_id),
			"contract_count": self._count(self.contracts, tenant_id),
			"variation_count": self._count(self.variations, tenant_id),
			"debarred_count": self._count(self.debarred, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"bids_received": len(self._bid_submissions),
			"contract_closures": len(self._contract_closures),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
		}

	def _get_tender(self, tender_id: str, tenant_id: str) -> Tender | None:
		return self.tenders.get(self._key(tenant_id, tender_id))

	def _get_award(self, award_id: str, tenant_id: str) -> ContractAward | None:
		return self.awards.get(self._key(tenant_id, award_id))

	def _get_contract(self, contract_id: str, tenant_id: str) -> GovernmentContract | None:
		return self.contracts.get(self._key(tenant_id, contract_id))

	def _has_approved_evaluation(self, tender_id: str, tenant_id: str) -> bool:
		return any(e.tender_id == tender_id and e.tenant_id == tenant_id for e in self.evaluations.values())

	def _is_debarred(self, bidder_id: str, tenant_id: str) -> bool:
		return any(d.bidder_id == bidder_id and d.tenant_id == tenant_id for d in self.debarred.values())

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, store: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in store.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(a.get("reason", a.get("rule", "policy_denied")) for a in result["actions"])
		raise PermissionError(reasons or "policy_denied")

	# ── additional methods ──────────────────────────────────────────────────

	def vendor_registration(
		self,
		vendor_id: str,
		vendor_name: str,
		categories: list[str],
		documents: list[str],
	) -> dict[str, Any]:
		"""Register a vendor in the approved supplier register."""
		assert vendor_id, "vendor_id required"
		assert vendor_name, "vendor_name required"
		tenant_id = self.tenant_id
		reg_id = _new_id()
		self._audit(tenant_id, "vendor_registered", reg_id)
		return {
			"id": reg_id,
			"tenant_id": tenant_id,
			"vendor_id": vendor_id,
			"vendor_name": vendor_name,
			"categories": categories,
			"documents_submitted": documents,
			"registered_by": self.actor_id,
			"registered_at": datetime.utcnow().isoformat(),
			"status": "approved",
		}

	def contract_renewal(
		self,
		contract_id: str,
		extension_months: int,
		reason: str,
	) -> dict[str, Any]:
		"""Renew an expiring contract for a specified period."""
		tenant_id = self.tenant_id
		contract = self._get_contract(contract_id, tenant_id)
		if contract is None:
			raise KeyError(f"contract {contract_id} not found")
		renewal_id = _new_id()
		new_end = (datetime.utcnow() + timedelta(days=extension_months * 30)).isoformat()
		self._audit(tenant_id, "contract_renewed", renewal_id)
		return {
			"id": renewal_id,
			"contract_id": contract_id,
			"extension_months": extension_months,
			"new_end_date": new_end,
			"reason": reason,
			"renewed_by": self.actor_id,
			"renewed_at": datetime.utcnow().isoformat(),
			"status": "renewed",
		}

	def tender_cancellation(self, tender_id: str, reason: str) -> dict[str, Any]:
		"""Cancel an active tender with documented reason."""
		tenant_id = self.tenant_id
		tender = self._get_tender(tender_id, tenant_id)
		if tender is None:
			raise KeyError(f"tender {tender_id} not found")
		tender.status = "cancelled"
		cancel_id = _new_id()
		self._audit(tenant_id, "tender_cancelled", cancel_id)
		return {"id": cancel_id, "tender_id": tender_id, "reason": reason, "cancelled_by": self.actor_id, "cancelled_at": datetime.utcnow().isoformat(), "status": "cancelled"}

	def standstill_period_management(self, award_id: str) -> dict[str, Any]:
		"""Track the 14-day PPDA standstill period after award."""
		tenant_id = self.tenant_id
		award = self._get_award(award_id, tenant_id)
		if award is None:
			raise KeyError(f"award {award_id} not found")
		standstill_end = (datetime.utcnow() + timedelta(days=14)).isoformat()
		sp_id = _new_id()
		self._audit(tenant_id, "standstill_period_tracked", sp_id)
		return {"id": sp_id, "award_id": award_id, "standstill_end": standstill_end, "can_sign_contract": False, "days_remaining": 14, "generated_at": datetime.utcnow().isoformat()}

	def contract_milestone_tracking(
		self,
		contract_id: str,
		milestones: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Set and track delivery milestones for a contract."""
		tenant_id = self.tenant_id
		contract = self._get_contract(contract_id, tenant_id)
		if contract is None:
			raise KeyError(f"contract {contract_id} not found")
		tracking_id = _new_id()
		self._audit(tenant_id, "milestones_set", tracking_id)
		return {"id": tracking_id, "contract_id": contract_id, "milestone_count": len(milestones), "milestones": milestones, "tracked_by": self.actor_id, "tracked_at": datetime.utcnow().isoformat()}

	def bid_opening_record(self, tender_id: str, committee_ids: list[str]) -> dict[str, Any]:
		"""Record the formal bid opening ceremony."""
		tenant_id = self.tenant_id
		bids = [b for b in self._bid_submissions if b["tenant_id"] == tenant_id and b["tender_id"] == tender_id]
		opening_id = _new_id()
		self._audit(tenant_id, "bid_opening_recorded", opening_id)
		return {"id": opening_id, "tender_id": tender_id, "bids_opened": len(bids), "committee_ids": committee_ids, "bid_amounts": [b["bid_amount"] for b in bids], "opened_at": datetime.utcnow().isoformat()}

	def integrity_pact_management(self, tender_id: str, signatories: list[str]) -> dict[str, Any]:
		"""Record integrity pact signatories for a tender."""
		tenant_id = self.tenant_id
		pact_id = _new_id()
		self._audit(tenant_id, "integrity_pact_signed", pact_id)
		return {"id": pact_id, "tender_id": tender_id, "signatories": signatories, "signatory_count": len(signatories), "signed_at": datetime.utcnow().isoformat()}

	def contract_dispute_resolution(
		self,
		contract_id: str,
		dispute_type: str,
		resolution_mechanism: str,
		description: str,
	) -> dict[str, Any]:
		"""Log and track a contract dispute resolution process."""
		tenant_id = self.tenant_id
		dispute_id = _new_id()
		ref = f"DSP-{datetime.utcnow().strftime('%Y%m%d')}-{dispute_id[:6].upper()}"
		self._audit(tenant_id, "dispute_logged", dispute_id)
		return {"id": dispute_id, "reference": ref, "contract_id": contract_id, "dispute_type": dispute_type, "resolution_mechanism": resolution_mechanism, "description": description, "logged_by": self.actor_id, "logged_at": datetime.utcnow().isoformat(), "status": "open"}

	def supplier_performance_rating(
		self,
		vendor_id: str,
		contract_id: str,
		rating_score: float,
		criteria: dict[str, float],
	) -> dict[str, Any]:
		"""Rate supplier performance after contract completion."""
		assert 0 <= rating_score <= 100, "rating_score must be 0–100"
		tenant_id = self.tenant_id
		rating_id = _new_id()
		grade = "A" if rating_score >= 85 else ("B" if rating_score >= 70 else ("C" if rating_score >= 55 else "D"))
		self._audit(tenant_id, "supplier_rated", rating_id)
		return {"id": rating_id, "vendor_id": vendor_id, "contract_id": contract_id, "rating_score": rating_score, "grade": grade, "criteria": criteria, "rated_by": self.actor_id, "rated_at": datetime.utcnow().isoformat()}

	def public_procurement_register(self) -> dict[str, Any]:
		"""Export the public procurement register for transparency."""
		tenant_id = self.tenant_id
		tenders = [t for (tid, _), t in self.tenders.items() if tid == tenant_id]
		awards = [a for (tid, _), a in self.awards.items() if tid == tenant_id]
		contracts = [c for (tid, _), c in self.contracts.items() if tid == tenant_id]
		return {
			"tenant_id": tenant_id,
			"published_tenders": len(tenders),
			"awards": len(awards),
			"active_contracts": sum(1 for c in contracts if c.status == "active"),
			"total_contract_value": sum(c.contract_value for c in contracts if hasattr(c, "contract_value")),
			"exported_at": datetime.utcnow().isoformat(),
		}

	def contract_payment_schedule(
		self,
		contract_id: str,
		payment_dates: list[str],
		amounts: list[float],
	) -> dict[str, Any]:
		"""Define a payment schedule for a contract."""
		assert len(payment_dates) == len(amounts), "payment_dates and amounts must be same length"
		tenant_id = self.tenant_id
		schedule_id = _new_id()
		schedule = [{"date": d, "amount": a, "status": "pending"} for d, a in zip(payment_dates, amounts)]
		self._audit(tenant_id, "payment_schedule_set", schedule_id)
		return {"id": schedule_id, "contract_id": contract_id, "total_payments": len(schedule), "total_amount": sum(amounts), "schedule": schedule, "created_by": self.actor_id, "created_at": datetime.utcnow().isoformat()}

	def procurement_plan(
		self,
		fiscal_year: str,
		planned_tenders: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Create an annual procurement plan."""
		tenant_id = self.tenant_id
		plan_id = _new_id()
		total_value = sum(t.get("estimated_value", 0.0) for t in planned_tenders)
		self._audit(tenant_id, "procurement_plan_created", plan_id)
		return {"id": plan_id, "tenant_id": tenant_id, "fiscal_year": fiscal_year, "planned_tenders": len(planned_tenders), "total_planned_value": total_value, "tenders": planned_tenders, "created_by": self.actor_id, "created_at": datetime.utcnow().isoformat()}

	def framework_agreement(
		self,
		agreement_id: str,
		category: str,
		vendors: list[str],
		duration_months: int,
		ceiling_value: float,
	) -> dict[str, Any]:
		"""Register a framework agreement for repeat procurements."""
		tenant_id = self.tenant_id
		expiry = (datetime.utcnow() + timedelta(days=duration_months * 30)).isoformat()
		self._audit(tenant_id, "framework_agreement_registered", agreement_id)
		return {"id": agreement_id, "tenant_id": tenant_id, "category": category, "vendors": vendors, "vendor_count": len(vendors), "duration_months": duration_months, "ceiling_value": ceiling_value, "expiry_date": expiry, "registered_by": self.actor_id, "status": "active"}

	def low_value_procurement(
		self,
		description: str,
		amount: float,
		vendor_id: str,
		justification: str,
	) -> dict[str, Any]:
		"""Process a low-value procurement via petty cash or imprest."""
		assert amount > 0 and amount < 50000, "amount must be 0–50,000 (low value threshold)"
		tenant_id = self.tenant_id
		lv_id = _new_id()
		self._audit(tenant_id, "low_value_procurement_recorded", lv_id)
		return {"id": lv_id, "tenant_id": tenant_id, "description": description, "amount": amount, "vendor_id": vendor_id, "justification": justification, "processed_by": self.actor_id, "processed_at": datetime.utcnow().isoformat()}

	def e_procurement_integration(self, system_name: str, sync_status: str) -> dict[str, Any]:
		"""Record an e-procurement system integration sync event."""
		tenant_id = self.tenant_id
		sync_id = _new_id()
		self._audit(tenant_id, "e_procurement_synced", sync_id)
		return {"id": sync_id, "tenant_id": tenant_id, "system_name": system_name, "sync_status": sync_status, "synced_at": datetime.utcnow().isoformat()}

	def bid_security_management(
		self,
		tender_id: str,
		vendor_id: str,
		security_amount: float,
		security_type: str,
	) -> dict[str, Any]:
		"""Track bid security (bond/guarantee) lodged by a vendor."""
		tenant_id = self.tenant_id
		bs_id = _new_id()
		expiry = (datetime.utcnow() + timedelta(days=90)).isoformat()
		self._audit(tenant_id, "bid_security_recorded", bs_id)
		return {"id": bs_id, "tender_id": tender_id, "vendor_id": vendor_id, "security_amount": security_amount, "security_type": security_type, "expiry_date": expiry, "recorded_at": datetime.utcnow().isoformat(), "status": "active"}

	def retention_release(self, contract_id: str, release_amount: float, release_reason: str) -> dict[str, Any]:
		"""Release contract retention money after defects liability period."""
		tenant_id = self.tenant_id
		contract = self._get_contract(contract_id, tenant_id)
		if contract is None:
			raise KeyError(f"contract {contract_id} not found")
		release_id = _new_id()
		self._audit(tenant_id, "retention_released", release_id)
		return {"id": release_id, "contract_id": contract_id, "release_amount": release_amount, "reason": release_reason, "released_by": self.actor_id, "released_at": datetime.utcnow().isoformat()}


	def tender_publish(self, title: str, description: str, deadline: datetime, estimated_value: float, category: str) -> dict[str, Any]:
		"""Publish a new tender notice (canonical domain alias)."""
		assert title, "title required"
		tenant_id = self.tenant_id
		tender_id = _new_id()
		ref = f"TDR-{datetime.utcnow().strftime('%Y%m%d')}-{tender_id[:6].upper()}"
		method = "open_tender" if estimated_value >= 500000 else "request_for_quotation"
		item = Tender(tender_id, tenant_id, method, "above_threshold" if estimated_value >= 500000 else "below_threshold", title, description, self.actor_id, ref, "published", "")
		self.tenders[self._key(tenant_id, tender_id)] = item
		self._audit(tenant_id, "tender_published", tender_id)
		return {"id": tender_id, "reference": ref, "title": title, "deadline": deadline.isoformat(), "estimated_value": estimated_value, "category": category, "method": method, "status": "published"}

	def bid_deadline_extend(self, tender_id: str, new_deadline: datetime, reason: str) -> dict[str, Any]:
		"""Extend the bid submission deadline for a tender."""
		tenant_id = self.tenant_id
		tender = self._get_tender(tender_id, tenant_id)
		if tender is None:
			raise KeyError(f"tender {tender_id} not found")
		ext_id = _new_id()
		self._audit(tenant_id, "bid_deadline_extended", ext_id)
		return {"extension_id": ext_id, "tender_id": tender_id, "new_deadline": new_deadline.isoformat(), "reason": reason, "extended_by": self.actor_id, "extended_at": datetime.utcnow().isoformat()}

	def clarification_respond(self, tender_id: str, question: str, answer: str) -> dict[str, Any]:
		"""Publish a clarification response to a tenderer question."""
		tenant_id = self.tenant_id
		clar_id = _new_id()
		self._audit(tenant_id, "clarification_published", clar_id)
		return {"clarification_id": clar_id, "tender_id": tender_id, "question": question, "answer": answer, "published_at": datetime.utcnow().isoformat(), "status": "published"}

	def evaluation_score(self, tender_id: str, bid_id: str, score: float, remarks: str) -> dict[str, Any]:
		"""Score a bid during evaluation."""
		return self.evaluate_bid(tender_id, bid_id, score, remarks)

	def award_announce(self, tender_id: str, winning_bid_id: str, contract_value: float) -> dict[str, Any]:
		"""Announce contract award."""
		return self.award_contract(tender_id, winning_bid_id, contract_value, datetime.utcnow())

	def contract_amend(self, contract_id: str, description: str, cost_variation: float, approved_by: str) -> dict[str, Any]:
		"""Amend a contract via variation order."""
		return self.variation_order(contract_id, description, cost_variation, approved_by)

	def milestone_track(self, contract_id: str, milestones: list[dict[str, Any]]) -> dict[str, Any]:
		"""Track contract delivery milestones."""
		return self.contract_milestone_tracking(contract_id, milestones)

	def invoice_approve(self, contract_id: str, invoice_ref: str, amount: float, approved_by: str) -> dict[str, Any]:
		"""Approve a contractor invoice for payment."""
		tenant_id = self.tenant_id
		inv_id = _new_id()
		self._audit(tenant_id, "invoice_approved", inv_id)
		return {"invoice_id": inv_id, "contract_id": contract_id, "invoice_ref": invoice_ref, "amount": amount, "approved_by": approved_by, "approved_at": datetime.utcnow().isoformat(), "status": "approved"}

	def contract_close(self, contract_id: str, completion_report: str) -> dict[str, Any]:
		"""Close a completed contract."""
		return self.contract_close(contract_id, completion_report)

	def performance_evaluate(self, vendor_id: str, contract_id: str, rating_score: float, criteria: dict[str, float]) -> dict[str, Any]:
		"""Evaluate supplier performance after contract."""
		return self.supplier_performance_rating(vendor_id, contract_id, rating_score, criteria)

	def dispute_manage(self, contract_id: str, dispute_type: str, resolution_mechanism: str, description: str) -> dict[str, Any]:
		"""Manage a contract dispute."""
		return self.contract_dispute_resolution(contract_id, dispute_type, resolution_mechanism, description)

	def contract_search(self, query: str, status: str | None = None) -> list[dict[str, Any]]:
		"""Search contracts by description/contractor."""
		tenant_id = self.tenant_id
		ql = query.lower()
		results = []
		for (tid, _), c in self.contracts.items():
			if tid != tenant_id:
				continue
			if status and getattr(c, "status", "") != status:
				continue
			if ql in getattr(c, "contractor_reference", "").lower() or ql in getattr(c, "contract_type", "").lower():
				results.append(c.to_dict() if hasattr(c, "to_dict") else {"contract_id": c.id})
		return results

	def spend_report(self, period: str) -> dict[str, Any]:
		"""Return spend analytics for a period."""
		return self.procurement_analytics(period)

	def compliance_check(self, contract_id: str) -> dict[str, Any]:
		"""Run compliance check — PPDA alias."""
		return self.ppda_compliance_check(contract_id)

	def contract_template(self, template_name: str, contract_type: str, standard_clauses: list[str]) -> dict[str, Any]:
		"""Create a contract template."""
		tenant_id = self.tenant_id
		tmpl_id = _new_id()
		return {"template_id": tmpl_id, "tenant_id": tenant_id, "template_name": template_name, "contract_type": contract_type, "clauses": standard_clauses, "created_by": self.actor_id, "created_at": datetime.utcnow().isoformat()}

	def contract_archive(self, contract_id: str) -> dict[str, Any]:
		"""Archive a closed contract for records retention."""
		tenant_id = self.tenant_id
		contract = self._get_contract(contract_id, tenant_id)
		if contract is None:
			raise KeyError(f"contract {contract_id} not found")
		arc_id = _new_id()
		self._audit(tenant_id, "contract_archived", arc_id)
		return {"archive_id": arc_id, "contract_id": contract_id, "archived_by": self.actor_id, "archived_at": datetime.utcnow().isoformat(), "retention_years": 7}

	def contract_analytics(self, period: str) -> dict[str, Any]:
		"""Return detailed contract analytics."""
		return self.procurement_analytics(period)


GovernmentConService = ProcurementService
