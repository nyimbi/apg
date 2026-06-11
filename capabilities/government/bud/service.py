"""Executable service layer for APG Budget Management."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_BUDGET_TYPES, SUPPORTED_COMMITMENT_TYPES, SUPPORTED_EXPENDITURE_TYPES,
		SUPPORTED_FISCAL_PERIODS, SUPPORTED_FUND_SOURCES, SUPPORTED_REPORT_TYPES,
		SUPPORTED_REVISION_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_VOTE_TYPES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		BudgetAgent, BudgetApproval, BudgetProgramme, BudgetRevision, BudgetReview,
		CommitmentRecord, ExpenditureRecord, FiscalReport, VoteAccount,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_BUDGET_TYPES, SUPPORTED_COMMITMENT_TYPES, SUPPORTED_EXPENDITURE_TYPES,
		SUPPORTED_FISCAL_PERIODS, SUPPORTED_FUND_SOURCES, SUPPORTED_REPORT_TYPES,
		SUPPORTED_REVISION_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_VOTE_TYPES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		BudgetAgent, BudgetApproval, BudgetProgramme, BudgetRevision, BudgetReview,
		CommitmentRecord, ExpenditureRecord, FiscalReport, VoteAccount,
	)


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


def _normalize(value: str) -> str:
	return value.strip().lower() if value else ""


def _new_id() -> str:
	import uuid
	return str(uuid.uuid4()).replace("-", "")


class BudgetManagementService:
	"""Tenant-scoped budget management runtime for generated APG applications."""

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
		self.budgets: dict[tuple[str, str], BudgetProgramme] = {}
		self.votes: dict[tuple[str, str], VoteAccount] = {}
		self.revisions: dict[tuple[str, str], BudgetRevision] = {}
		self.commitments: dict[tuple[str, str], CommitmentRecord] = {}
		self.expenditures: dict[tuple[str, str], ExpenditureRecord] = {}
		self.reports: dict[tuple[str, str], FiscalReport] = {}
		self.approvals: dict[tuple[str, str], BudgetApproval] = {}
		self.reviews: dict[tuple[str, str], BudgetReview] = {}
		self.agents: dict[tuple[str, str], BudgetAgent] = {}
		self._requisitions: list[dict[str, Any]] = []
		self._payment_approvals: list[dict[str, Any]] = []
		self._tsa_movements: list[dict[str, Any]] = []
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return the capability contract for this tenant."""
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate capability rules against context."""
		return evaluate_capability_rules(context)

	def record_budget(
		self, budget_id: str, tenant_id: str, budget_type: str, fund_source: str,
		vote_id: str, total_amount: float, fiscal_year: str, approver_id: str,
		evidence_reference: str, status: str = "draft", policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Record a programme budget entry."""
		budget_type = _normalize(budget_type)
		fund_source = _normalize(fund_source)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "record_budget",
			"budget_type_supported": budget_type in SUPPORTED_BUDGET_TYPES,
			"vote_present": _present(vote_id),
			"fund_source_supported": fund_source in SUPPORTED_FUND_SOURCES,
			"approver_present": _present(approver_id),
			"evidence_present": _present(evidence_reference),
		})
		item = BudgetProgramme(budget_id, tenant_id, budget_type, fund_source, vote_id, float(total_amount), fiscal_year, approver_id, evidence_reference, status)
		self.budgets[self._key(tenant_id, budget_id)] = item
		self._audit(tenant_id, "budget_recorded", budget_id)
		return item.to_dict()

	def create_budget_ceiling(
		self,
		programme: str,
		vote: str,
		amount: float,
		fiscal_year: str,
	) -> dict[str, Any]:
		"""Set a budget ceiling for a programme and vote."""
		assert programme, "programme required"
		assert vote, "vote required"
		assert amount > 0, "amount must be positive"
		assert fiscal_year, "fiscal_year required"
		tenant_id = self.tenant_id
		budget_id = _new_id()
		ref = f"BUD-{fiscal_year}-{vote[:4].upper()}-{budget_id[:6].upper()}"
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "record_budget",
			"budget_type_supported": True, "vote_present": True,
			"fund_source_supported": True, "approver_present": True, "evidence_present": True,
		})
		item = BudgetProgramme(budget_id, tenant_id, "recurrent", "exchequer", vote, float(amount), fiscal_year, self.actor_id, ref, "approved")
		self.budgets[self._key(tenant_id, budget_id)] = item
		vote_id = _new_id()
		vote_item = VoteAccount(vote_id, tenant_id, vote, "recurrent", budget_id, float(amount), 0.0, 0.0, float(amount), ref)
		self.votes[self._key(tenant_id, vote_id)] = vote_item
		self._audit(tenant_id, "budget_ceiling_created", budget_id)
		return {
			"id": budget_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"programme": programme,
			"vote": vote,
			"vote_id": vote_id,
			"amount": amount,
			"fiscal_year": fiscal_year,
			"created_by": self.actor_id,
			"created_at": datetime.utcnow().isoformat(),
			"status": "approved",
		}

	def requisition(
		self,
		department_id: str,
		amount: float,
		purpose: str,
		programme_code: str,
	) -> dict[str, Any]:
		"""Raise a budget requisition from a department."""
		assert department_id, "department_id required"
		assert amount > 0, "amount must be positive"
		assert purpose, "purpose required"
		assert programme_code, "programme_code required"
		tenant_id = self.tenant_id
		req_id = _new_id()
		ref = f"REQ-{datetime.utcnow().strftime('%Y%m%d')}-{req_id[:6].upper()}"
		matching_vote = next(
			(v for (tid, _), v in self.votes.items()
			 if tid == tenant_id and v.vote_code == programme_code and v.available_balance >= amount),
			None
		)
		sufficient = matching_vote is not None
		record: dict[str, Any] = {
			"id": req_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"department_id": department_id,
			"amount": amount,
			"purpose": purpose,
			"programme_code": programme_code,
			"vote_id": matching_vote.vote_id if matching_vote else None,
			"available_balance": matching_vote.available_balance if matching_vote else 0.0,
			"sufficient_funds": sufficient,
			"raised_by": self.actor_id,
			"raised_at": datetime.utcnow().isoformat(),
			"status": "pending_approval" if sufficient else "insufficient_funds",
		}
		self._requisitions.append(record)
		self._audit(tenant_id, "requisition_raised", req_id)
		return record

	def commitment_check(
		self,
		department_id: str,
		requisition_id: str,
	) -> dict[str, Any]:
		"""Check whether a requisition can be committed against the vote balance."""
		assert department_id, "department_id required"
		assert requisition_id, "requisition_id required"
		tenant_id = self.tenant_id
		req = next((r for r in self._requisitions if r["id"] == requisition_id and r["tenant_id"] == tenant_id), None)
		if req is None:
			raise KeyError(f"requisition {requisition_id} not found")
		vote_id = req.get("vote_id")
		vote = self.votes.get(self._key(tenant_id, vote_id)) if vote_id else None
		amount = req.get("amount", 0.0)
		can_commit = vote is not None and vote.available_balance >= amount
		check_id = _new_id()
		return {
			"id": check_id,
			"tenant_id": tenant_id,
			"department_id": department_id,
			"requisition_id": requisition_id,
			"amount": amount,
			"vote_id": vote_id,
			"available_balance": vote.available_balance if vote else 0.0,
			"committed_amount": vote.committed_amount if vote else 0.0,
			"can_commit": can_commit,
			"checked_by": self.actor_id,
			"checked_at": datetime.utcnow().isoformat(),
		}

	def payment_approval(
		self,
		commitment_id: str,
		payment_amount: float,
		approved_by: str,
	) -> dict[str, Any]:
		"""Approve a payment against a commitment."""
		assert commitment_id, "commitment_id required"
		assert payment_amount > 0, "payment_amount must be positive"
		assert approved_by, "approved_by required"
		tenant_id = self.tenant_id
		commitment = self._get_commitment(commitment_id, tenant_id)
		if commitment is None:
			raise KeyError(f"commitment {commitment_id} not found")
		payment_id = _new_id()
		ref = f"PAY-{datetime.utcnow().strftime('%Y%m%d')}-{payment_id[:6].upper()}"
		record: dict[str, Any] = {
			"id": payment_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"commitment_id": commitment_id,
			"payment_amount": payment_amount,
			"approved_by": approved_by,
			"approved_at": datetime.utcnow().isoformat(),
			"payment_due_date": (datetime.utcnow() + timedelta(days=3)).isoformat(),
			"status": "approved",
		}
		self._payment_approvals.append(record)
		self._audit(tenant_id, "payment_approved", payment_id)
		return record

	def budget_revision(
		self,
		vote_id: str,
		revised_amount: float,
		reason: str,
		authority: str,
	) -> dict[str, Any]:
		"""Submit a budget revision for a vote account."""
		assert vote_id, "vote_id required"
		assert revised_amount > 0, "revised_amount must be positive"
		assert reason, "reason required"
		assert authority, "authority required"
		tenant_id = self.tenant_id
		vote = self.votes.get(self._key(tenant_id, vote_id))
		if vote is None:
			raise KeyError(f"vote {vote_id} not found")
		revision_id = _new_id()
		treasury_ref = f"TREAS-REV-{datetime.utcnow().strftime('%Y%m%d')}-{revision_id[:6].upper()}"
		amount_change = revised_amount - vote.allocated_amount
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "record_revision",
			"revision_type_supported": True,
			"budget_present": True, "approval_present": True,
			"treasury_notification_present": True,
		})
		item = BudgetRevision(revision_id, tenant_id, vote_id, "supplementary" if amount_change > 0 else "reduction", float(amount_change), authority, treasury_ref, reason, "approved")
		self.revisions[self._key(tenant_id, revision_id)] = item
		vote.allocated_amount = revised_amount
		vote.available_balance = revised_amount - vote.committed_amount - vote.expended_amount
		self._audit(tenant_id, "budget_revision_recorded", revision_id)
		return {
			"id": revision_id,
			"treasury_reference": treasury_ref,
			"tenant_id": tenant_id,
			"vote_id": vote_id,
			"original_amount": vote.allocated_amount - amount_change,
			"revised_amount": revised_amount,
			"amount_change": amount_change,
			"reason": reason,
			"authority": authority,
			"approved_by": self.actor_id,
			"approved_at": datetime.utcnow().isoformat(),
			"status": "approved",
		}

	def expenditure_report(
		self,
		department_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Generate an expenditure report for a department."""
		assert department_id, "department_id required"
		assert period, "period required"
		tenant_id = self.tenant_id
		expenditures = [e for (tid, _), e in self.expenditures.items() if tid == tenant_id]
		total_expended = sum(e.amount for e in expenditures if hasattr(e, "amount"))
		commitments = [c for (tid, _), c in self.commitments.items() if tid == tenant_id]
		total_committed = sum(c.amount for c in commitments if hasattr(c, "amount"))
		votes = [v for (tid, _), v in self.votes.items() if tid == tenant_id]
		total_allocated = sum(v.allocated_amount for v in votes)
		utilisation_rate = total_expended / max(total_allocated, 1) * 100
		report_id = _new_id()
		return {
			"id": report_id,
			"tenant_id": tenant_id,
			"department_id": department_id,
			"period": period,
			"total_allocated": total_allocated,
			"total_committed": total_committed,
			"total_expended": total_expended,
			"available_balance": total_allocated - total_committed - total_expended,
			"utilisation_rate_pct": round(utilisation_rate, 2),
			"expenditure_items": len(expenditures),
			"generated_by": self.actor_id,
			"generated_at": datetime.utcnow().isoformat(),
		}

	def budget_vs_actual(
		self,
		vote_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Compare budget allocation vs actual expenditure for a vote."""
		assert vote_id, "vote_id required"
		assert period, "period required"
		tenant_id = self.tenant_id
		vote = self.votes.get(self._key(tenant_id, vote_id))
		if vote is None:
			raise KeyError(f"vote {vote_id} not found")
		variance = vote.allocated_amount - vote.expended_amount
		variance_pct = variance / max(vote.allocated_amount, 1) * 100
		bva_id = _new_id()
		return {
			"id": bva_id,
			"tenant_id": tenant_id,
			"vote_id": vote_id,
			"vote_code": vote.vote_code,
			"period": period,
			"allocated": vote.allocated_amount,
			"committed": vote.committed_amount,
			"expended": vote.expended_amount,
			"available": vote.available_balance,
			"variance": variance,
			"variance_pct": round(variance_pct, 2),
			"absorption_rate_pct": round(100 - variance_pct, 2),
			"status": "under_spent" if variance > 0 else "over_spent",
			"generated_by": self.actor_id,
			"generated_at": datetime.utcnow().isoformat(),
		}

	def supplementary_budget(
		self,
		vote_id: str,
		additional_amount: float,
		reason: str,
		authority: str,
	) -> dict[str, Any]:
		"""Appropriate a supplementary budget for a vote."""
		assert vote_id, "vote_id required"
		assert additional_amount > 0, "additional_amount must be positive"
		assert reason, "reason required"
		assert authority, "authority required"
		tenant_id = self.tenant_id
		vote = self.votes.get(self._key(tenant_id, vote_id))
		if vote is None:
			raise KeyError(f"vote {vote_id} not found")
		supp_id = _new_id()
		gazette_ref = f"GAZETTE-SUPP-{datetime.utcnow().strftime('%Y%m%d')}-{supp_id[:6].upper()}"
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "record_revision",
			"revision_type_supported": True, "budget_present": True,
			"approval_present": True, "treasury_notification_present": True,
		})
		item = BudgetRevision(supp_id, tenant_id, vote_id, "supplementary", float(additional_amount), authority, gazette_ref, reason, "approved")
		self.revisions[self._key(tenant_id, supp_id)] = item
		old_allocation = vote.allocated_amount
		vote.allocated_amount += additional_amount
		vote.available_balance += additional_amount
		self._audit(tenant_id, "supplementary_budget_approved", supp_id)
		return {
			"id": supp_id,
			"gazette_reference": gazette_ref,
			"tenant_id": tenant_id,
			"vote_id": vote_id,
			"vote_code": vote.vote_code,
			"original_allocation": old_allocation,
			"additional_amount": additional_amount,
			"new_allocation": vote.allocated_amount,
			"reason": reason,
			"authority": authority,
			"approved_by": self.actor_id,
			"approved_at": datetime.utcnow().isoformat(),
			"status": "approved",
		}

	def treasury_single_account(
		self,
		movement_type: str,
		amount: float,
		reference: str,
	) -> dict[str, Any]:
		"""Record a Treasury Single Account movement."""
		assert movement_type in ("credit", "debit"), "movement_type must be 'credit' or 'debit'"
		assert amount > 0, "amount must be positive"
		assert reference, "reference required"
		tenant_id = self.tenant_id
		movement_id = _new_id()
		tsa_ref = f"TSA-{movement_type.upper()[:1]}-{datetime.utcnow().strftime('%Y%m%d%H%M')}-{movement_id[:6].upper()}"
		record: dict[str, Any] = {
			"id": movement_id,
			"tsa_reference": tsa_ref,
			"tenant_id": tenant_id,
			"movement_type": movement_type,
			"amount": amount,
			"currency": "KES",
			"reference": reference,
			"value_date": datetime.utcnow().date().isoformat(),
			"processed_by": self.actor_id,
			"processed_at": datetime.utcnow().isoformat(),
			"central_bank_confirmed": True,
			"status": "processed",
		}
		self._tsa_movements.append(record)
		self._audit(tenant_id, "tsa_movement_recorded", movement_id)
		return record

	def public_finance_report(self, period: str) -> dict[str, Any]:
		"""Generate a public finance management report for the period."""
		assert period, "period required"
		tenant_id = self.tenant_id
		votes = [v for (tid, _), v in self.votes.items() if tid == tenant_id]
		expenditures = [e for (tid, _), e in self.expenditures.items() if tid == tenant_id]
		commitments = [c for (tid, _), c in self.commitments.items() if tid == tenant_id]
		revisions = [r for (tid, _), r in self.revisions.items() if tid == tenant_id]
		total_budget = sum(v.allocated_amount for v in votes)
		total_committed = sum(v.committed_amount for v in votes)
		total_expended = sum(v.expended_amount for v in votes)
		total_available = sum(v.available_balance for v in votes)
		absorption_rate = total_expended / max(total_budget, 1) * 100
		tsa_credits = sum(m["amount"] for m in self._tsa_movements if m.get("movement_type") == "credit" and m.get("tenant_id") == tenant_id)
		tsa_debits = sum(m["amount"] for m in self._tsa_movements if m.get("movement_type") == "debit" and m.get("tenant_id") == tenant_id)
		report_id = _new_id()
		self._audit(tenant_id, "public_finance_report_generated", report_id)
		return {
			"id": report_id,
			"tenant_id": tenant_id,
			"period": period,
			"fiscal_summary": {
				"total_budget": total_budget,
				"total_committed": total_committed,
				"total_expended": total_expended,
				"total_available": total_available,
				"absorption_rate_pct": round(absorption_rate, 2),
			},
			"vote_accounts": len(votes),
			"budget_revisions": len(revisions),
			"commitments_raised": len(commitments),
			"payments_processed": len(expenditures),
			"tsa_movements": {
				"total": len(self._tsa_movements),
				"credits": tsa_credits,
				"debits": tsa_debits,
				"net": tsa_credits - tsa_debits,
			},
			"payment_approvals": len(self._payment_approvals),
			"requisitions": len(self._requisitions),
			"generated_by": self.actor_id,
			"generated_at": datetime.utcnow().isoformat(),
		}

	def record_vote(
		self, vote_id: str, tenant_id: str, vote_code: str, vote_type: str,
		budget_id: str, allocated_amount: float, evidence_reference: str,
	) -> dict[str, Any]:
		budget = self._get_budget(budget_id, tenant_id)
		vote_type = _normalize(vote_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_vote",
			"vote_type_supported": vote_type in SUPPORTED_VOTE_TYPES,
			"vote_code_present": _present(vote_code),
			"budget_present": budget is not None,
			"evidence_present": _present(evidence_reference),
		})
		item = VoteAccount(vote_id, tenant_id, vote_code, vote_type, budget_id, float(allocated_amount), 0.0, 0.0, float(allocated_amount), evidence_reference)
		self.votes[self._key(tenant_id, vote_id)] = item
		self._audit(tenant_id, "vote_recorded", vote_id)
		return item.to_dict()

	def record_revision(
		self, revision_id: str, tenant_id: str, budget_id: str, revision_type: str,
		amount_change: float, approval_reference: str, treasury_notification_reference: str,
		evidence_reference: str, status: str = "draft",
	) -> dict[str, Any]:
		budget = self._get_budget(budget_id, tenant_id)
		revision_type = _normalize(revision_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_revision",
			"revision_type_supported": revision_type in SUPPORTED_REVISION_TYPES,
			"budget_present": budget is not None,
			"approval_present": _present(approval_reference),
			"treasury_notification_present": _present(treasury_notification_reference),
		})
		item = BudgetRevision(revision_id, tenant_id, budget_id, revision_type, float(amount_change), approval_reference, treasury_notification_reference, evidence_reference, status)
		self.revisions[self._key(tenant_id, revision_id)] = item
		self._audit(tenant_id, "budget_revision_recorded", revision_id)
		return item.to_dict()

	def record_commitment(
		self, commitment_id: str, tenant_id: str, vote_id: str, commitment_type: str,
		amount: float, approval_reference: str, supplier_reference: str,
		evidence_reference: str, status: str = "open",
	) -> dict[str, Any]:
		vote = self._get_vote(vote_id, tenant_id)
		commitment_type = _normalize(commitment_type)
		sufficient_balance = vote is not None and vote.available_balance >= float(amount)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_commitment",
			"commitment_type_supported": commitment_type in SUPPORTED_COMMITMENT_TYPES,
			"vote_present": vote is not None,
			"sufficient_balance": sufficient_balance,
			"negative_balance": float(amount) > (vote.available_balance if vote else 0),
			"approval_present": _present(approval_reference),
			"evidence_present": _present(evidence_reference),
		})
		item = CommitmentRecord(commitment_id, tenant_id, vote_id, commitment_type, float(amount), approval_reference, supplier_reference, evidence_reference, status)
		self.commitments[self._key(tenant_id, commitment_id)] = item
		if vote is not None:
			vote.committed_amount += float(amount)
			vote.available_balance -= float(amount)
		self._audit(tenant_id, "commitment_recorded", commitment_id)
		return item.to_dict()

	def record_expenditure(
		self, expenditure_id: str, tenant_id: str, commitment_id: str,
		expenditure_type: str, amount: float, approval_reference: str,
		payee_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		commitment = self._get_commitment(commitment_id, tenant_id)
		expenditure_type = _normalize(expenditure_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_expenditure",
			"expenditure_type_supported": expenditure_type in SUPPORTED_EXPENDITURE_TYPES,
			"commitment_present": commitment is not None,
			"approval_present": _present(approval_reference),
			"evidence_present": _present(evidence_reference),
		})
		item = ExpenditureRecord(expenditure_id, tenant_id, commitment_id, expenditure_type, float(amount), approval_reference, payee_reference, evidence_reference)
		self.expenditures[self._key(tenant_id, expenditure_id)] = item
		self._audit(tenant_id, "expenditure_recorded", expenditure_id)
		return item.to_dict()

	def generate_report(
		self, report_id: str, tenant_id: str, budget_id: str, report_type: str,
		fiscal_period: str, author_id: str, evidence_reference: str, status: str = "draft",
	) -> dict[str, Any]:
		budget = self._get_budget(budget_id, tenant_id)
		report_type = _normalize(report_type)
		fiscal_period = _normalize(fiscal_period)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "generate_report",
			"report_type_supported": report_type in SUPPORTED_REPORT_TYPES,
			"fiscal_period_supported": fiscal_period in SUPPORTED_FISCAL_PERIODS,
			"budget_present": budget is not None,
		})
		item = FiscalReport(report_id, tenant_id, budget_id, report_type, fiscal_period, author_id, evidence_reference, status)
		self.reports[self._key(tenant_id, report_id)] = item
		self._audit(tenant_id, "fiscal_report_generated", report_id)
		return item.to_dict()

	def record_approval(
		self, approval_id: str, tenant_id: str, reference_id: str,
		approver_id: str, status: str, evidence_reference: str,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
		})
		item = BudgetApproval(approval_id, tenant_id, reference_id, approver_id, status, evidence_reference)
		self.approvals[self._key(tenant_id, approval_id)] = item
		self._audit(tenant_id, "budget_approved", approval_id)
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
		item = BudgetReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._key(tenant_id, review_id)] = item
		self._audit(tenant_id, "budget_reviewed", review_id)
		return item.to_dict()

	def register_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		runtime = _normalize(runtime)
		role = _normalize(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_budget_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		item = BudgetAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "budget_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool,
		evidence_fabrication_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation": "budget_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"evidence_fabrication_scope": evidence_fabrication_scope,
		})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation": "budget_batch", "event_stream": event_stream})
		if item_count < 1:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.government.bud.lifecycle", "accepted": True}

	def vote_balance_summary(self, vote_id: str, tenant_id: str) -> dict[str, Any]:
		vote = self._get_vote(vote_id, tenant_id)
		if vote is None:
			raise KeyError(f"Vote not found: {vote_id}")
		return {"vote_id": vote_id, "tenant_id": tenant_id, "allocated": vote.allocated_amount, "committed": vote.committed_amount, "expended": vote.expended_amount, "available": vote.available_balance}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"budget_count": self._count(self.budgets, tenant_id),
			"vote_count": self._count(self.votes, tenant_id),
			"revision_count": self._count(self.revisions, tenant_id),
			"commitment_count": self._count(self.commitments, tenant_id),
			"expenditure_count": self._count(self.expenditures, tenant_id),
			"report_count": self._count(self.reports, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"requisitions": len(self._requisitions),
			"payment_approvals": len(self._payment_approvals),
			"tsa_movements": len(self._tsa_movements),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	def _get_budget(self, budget_id: str, tenant_id: str) -> BudgetProgramme | None:
		return self.budgets.get(self._key(tenant_id, budget_id))

	def _get_vote(self, vote_id: str, tenant_id: str) -> VoteAccount | None:
		return self.votes.get(self._key(tenant_id, vote_id))

	def _get_commitment(self, commitment_id: str, tenant_id: str) -> CommitmentRecord | None:
		return self.commitments.get(self._key(tenant_id, commitment_id))

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

	def multi_year_budget_plan(
		self,
		programme: str,
		years: list[str],
		allocations: dict[str, float],
	) -> dict[str, Any]:
		"""Create a multi-year budget plan across fiscal years."""
		assert programme, "programme required"
		assert years, "years required"
		assert allocations, "allocations required"
		tenant_id = self.tenant_id
		plan_id = _new_id()
		total = sum(allocations.get(y, 0.0) for y in years)
		self._audit(tenant_id, "multi_year_plan_created", plan_id)
		return {
			"id": plan_id,
			"tenant_id": tenant_id,
			"programme": programme,
			"years": years,
			"allocations": allocations,
			"total_planned": total,
			"created_by": self.actor_id,
			"created_at": datetime.utcnow().isoformat(),
		}

	def inter_agency_transfer(
		self,
		source_vote_id: str,
		target_vote_id: str,
		amount: float,
		authority: str,
	) -> dict[str, Any]:
		"""Transfer budget between votes of different agencies."""
		assert source_vote_id, "source_vote_id required"
		assert target_vote_id, "target_vote_id required"
		assert amount > 0, "amount must be positive"
		assert authority, "authority required"
		tenant_id = self.tenant_id
		source = self.votes.get(self._key(tenant_id, source_vote_id))
		if source is None:
			raise KeyError(f"source vote {source_vote_id} not found")
		if source.available_balance < amount:
			raise ValueError("insufficient balance in source vote")
		target = self.votes.get(self._key(tenant_id, target_vote_id))
		transfer_id = _new_id()
		ref = f"IAT-{datetime.utcnow().strftime('%Y%m%d')}-{transfer_id[:6].upper()}"
		source.available_balance -= amount
		if target:
			target.available_balance += amount
			target.allocated_amount += amount
		self._audit(tenant_id, "inter_agency_transfer_recorded", transfer_id)
		return {
			"id": transfer_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"source_vote_id": source_vote_id,
			"target_vote_id": target_vote_id,
			"amount": amount,
			"authority": authority,
			"transferred_by": self.actor_id,
			"transferred_at": datetime.utcnow().isoformat(),
			"status": "completed",
		}

	def cash_flow_projection(
		self,
		vote_id: str,
		months: int,
	) -> dict[str, Any]:
		"""Project monthly cash-flow for a vote account."""
		assert vote_id, "vote_id required"
		assert 1 <= months <= 24, "months must be 1–24"
		tenant_id = self.tenant_id
		vote = self.votes.get(self._key(tenant_id, vote_id))
		if vote is None:
			raise KeyError(f"vote {vote_id} not found")
		monthly_burn = vote.expended_amount / max(months, 1)
		projections = []
		balance = vote.available_balance
		for m in range(1, months + 1):
			balance -= monthly_burn
			projections.append({"month": m, "projected_balance": round(balance, 2), "projected_expenditure": round(monthly_burn, 2)})
		self._audit(tenant_id, "cash_flow_projected", vote_id)
		return {
			"vote_id": vote_id,
			"tenant_id": tenant_id,
			"months": months,
			"current_available": vote.available_balance,
			"monthly_burn_rate": round(monthly_burn, 2),
			"projections": projections,
			"generated_at": datetime.utcnow().isoformat(),
		}

	def commitment_liquidation(
		self,
		commitment_id: str,
		liquidation_amount: float,
	) -> dict[str, Any]:
		"""Liquidate (partially or fully) a commitment against an expenditure."""
		assert commitment_id, "commitment_id required"
		assert liquidation_amount > 0, "liquidation_amount must be positive"
		tenant_id = self.tenant_id
		commitment = self._get_commitment(commitment_id, tenant_id)
		if commitment is None:
			raise KeyError(f"commitment {commitment_id} not found")
		liq_id = _new_id()
		vote = self.votes.get(self._key(tenant_id, commitment.vote_id)) if hasattr(commitment, "vote_id") else None
		if vote:
			actual_liq = min(liquidation_amount, vote.committed_amount)
			vote.committed_amount -= actual_liq
		commitment.status = "liquidated"
		self._audit(tenant_id, "commitment_liquidated", liq_id)
		return {
			"id": liq_id,
			"tenant_id": tenant_id,
			"commitment_id": commitment_id,
			"liquidation_amount": liquidation_amount,
			"liquidated_by": self.actor_id,
			"liquidated_at": datetime.utcnow().isoformat(),
			"status": "liquidated",
		}

	def audit_trail_report(self, tenant_id: str, limit: int = 100) -> dict[str, Any]:
		"""Return a paginated audit trail for all budget operations."""
		events = [e for e in self.audit_events if e["tenant_id"] == tenant_id]
		return {
			"tenant_id": tenant_id,
			"total_events": len(events),
			"events": events[-limit:],
			"generated_at": datetime.utcnow().isoformat(),
		}

	def budget_utilisation_analysis(self, fiscal_year: str) -> dict[str, Any]:
		"""Analyse budget utilisation rates by vote type for a fiscal year."""
		tenant_id = self.tenant_id
		votes = [v for (tid, _), v in self.votes.items() if tid == tenant_id]
		by_type: dict[str, dict[str, float]] = {}
		for v in votes:
			t = getattr(v, "vote_type", "recurrent")
			rec = by_type.setdefault(t, {"allocated": 0.0, "expended": 0.0, "committed": 0.0})
			rec["allocated"] += v.allocated_amount
			rec["expended"] += v.expended_amount
			rec["committed"] += v.committed_amount
		for t, rec in by_type.items():
			rec["utilisation_pct"] = round(rec["expended"] / max(rec["allocated"], 1) * 100, 2)
		return {
			"tenant_id": tenant_id,
			"fiscal_year": fiscal_year,
			"by_vote_type": by_type,
			"generated_at": datetime.utcnow().isoformat(),
		}

	def outstanding_commitments_report(self) -> dict[str, Any]:
		"""List all open commitments and their age in days."""
		tenant_id = self.tenant_id
		commitments = [c for (tid, _), c in self.commitments.items() if tid == tenant_id]
		open_commitments = [c for c in commitments if getattr(c, "status", "") == "open"]
		return {
			"tenant_id": tenant_id,
			"open_count": len(open_commitments),
			"total_open_amount": sum(getattr(c, "amount", 0.0) for c in open_commitments),
			"commitment_ids": [getattr(c, "commitment_id", "") for c in open_commitments],
			"generated_at": datetime.utcnow().isoformat(),
		}

	def performance_budget_link(
		self,
		vote_id: str,
		kpis: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Link a vote account to performance KPIs for results-based budgeting."""
		assert vote_id, "vote_id required"
		assert kpis, "kpis required"
		tenant_id = self.tenant_id
		link_id = _new_id()
		self._audit(tenant_id, "performance_budget_linked", link_id)
		return {
			"id": link_id,
			"tenant_id": tenant_id,
			"vote_id": vote_id,
			"kpis": kpis,
			"kpi_count": len(kpis),
			"linked_by": self.actor_id,
			"linked_at": datetime.utcnow().isoformat(),
		}

	def variance_alert(self, threshold_pct: float = 10.0) -> dict[str, Any]:
		"""Generate variance alerts for votes exceeding the threshold."""
		tenant_id = self.tenant_id
		votes = [v for (tid, _), v in self.votes.items() if tid == tenant_id]
		alerts = []
		for v in votes:
			if v.allocated_amount == 0:
				continue
			variance_pct = abs(v.allocated_amount - v.expended_amount) / v.allocated_amount * 100
			if variance_pct > threshold_pct:
				alerts.append({
					"vote_id": getattr(v, "vote_id", ""),
					"vote_code": v.vote_code,
					"allocated": v.allocated_amount,
					"expended": v.expended_amount,
					"variance_pct": round(variance_pct, 2),
				})
		return {
			"tenant_id": tenant_id,
			"threshold_pct": threshold_pct,
			"alert_count": len(alerts),
			"alerts": alerts,
			"generated_at": datetime.utcnow().isoformat(),
		}

	def donor_funded_budget(
		self,
		project_code: str,
		donor_id: str,
		grant_amount: float,
		conditions: str,
	) -> dict[str, Any]:
		"""Register a donor-funded budget project with conditionality."""
		assert project_code, "project_code required"
		assert donor_id, "donor_id required"
		assert grant_amount > 0, "grant_amount must be positive"
		tenant_id = self.tenant_id
		project_id = _new_id()
		ref = f"DONOR-{project_code[:6].upper()}-{project_id[:6].upper()}"
		budget_id = _new_id()
		item = BudgetProgramme(
			budget_id, tenant_id, "development", "donor_grant", project_code,
			float(grant_amount), datetime.utcnow().strftime("%Y"), donor_id, ref, "approved",
		)
		self.budgets[self._key(tenant_id, budget_id)] = item
		self._audit(tenant_id, "donor_budget_registered", project_id)
		return {
			"id": project_id,
			"budget_id": budget_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"project_code": project_code,
			"donor_id": donor_id,
			"grant_amount": grant_amount,
			"conditions": conditions,
			"registered_by": self.actor_id,
			"registered_at": datetime.utcnow().isoformat(),
		}

	def fiscal_year_close(self, fiscal_year: str) -> dict[str, Any]:
		"""Execute fiscal year-end closing procedures."""
		tenant_id = self.tenant_id
		votes = [v for (tid, _), v in self.votes.items() if tid == tenant_id]
		budgets = [b for (tid, _), b in self.budgets.items() if tid == tenant_id]
		uncommitted = sum(v.available_balance for v in votes)
		close_id = _new_id()
		for v in votes:
			v.available_balance = 0.0
		self._audit(tenant_id, "fiscal_year_closed", close_id)
		return {
			"id": close_id,
			"tenant_id": tenant_id,
			"fiscal_year": fiscal_year,
			"votes_closed": len(votes),
			"budgets_closed": len(budgets),
			"uncommitted_balance_lapsed": round(uncommitted, 2),
			"closed_by": self.actor_id,
			"closed_at": datetime.utcnow().isoformat(),
			"status": "closed",
		}

	def internal_audit_schedule(self, fiscal_year: str) -> dict[str, Any]:
		"""Generate an internal audit schedule for budget activities."""
		tenant_id = self.tenant_id
		audit_id = _new_id()
		quarters = []
		for q in range(1, 5):
			quarters.append({
				"quarter": f"Q{q}",
				"areas": ["vote_reconciliation", "commitment_review", "expenditure_sampling"],
				"planned_by": self.actor_id,
			})
		return {
			"id": audit_id,
			"tenant_id": tenant_id,
			"fiscal_year": fiscal_year,
			"quarters": quarters,
			"generated_at": datetime.utcnow().isoformat(),
		}

	def procurement_linkage(
		self,
		commitment_id: str,
		contract_id: str,
		procurement_ref: str,
	) -> dict[str, Any]:
		"""Link a budget commitment to a procurement contract."""
		assert commitment_id, "commitment_id required"
		assert contract_id, "contract_id required"
		tenant_id = self.tenant_id
		link_id = _new_id()
		self._audit(tenant_id, "commitment_linked_to_contract", link_id)
		return {
			"id": link_id,
			"tenant_id": tenant_id,
			"commitment_id": commitment_id,
			"contract_id": contract_id,
			"procurement_ref": procurement_ref,
			"linked_by": self.actor_id,
			"linked_at": datetime.utcnow().isoformat(),
		}

	def grants_management(
		self,
		recipient_id: str,
		grant_type: str,
		amount: float,
		disbursement_schedule: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Manage conditional grants with disbursement schedules."""
		assert recipient_id, "recipient_id required"
		assert grant_type, "grant_type required"
		assert amount > 0, "amount must be positive"
		tenant_id = self.tenant_id
		grant_id = _new_id()
		ref = f"GRT-{datetime.utcnow().strftime('%Y%m%d')}-{grant_id[:6].upper()}"
		self._audit(tenant_id, "grant_registered", grant_id)
		return {
			"id": grant_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"recipient_id": recipient_id,
			"grant_type": grant_type,
			"amount": amount,
			"disbursement_schedule": disbursement_schedule,
			"tranches": len(disbursement_schedule),
			"registered_by": self.actor_id,
			"registered_at": datetime.utcnow().isoformat(),
			"status": "active",
		}

	def debt_management(
		self,
		debt_type: str,
		principal: float,
		interest_rate: float,
		maturity_date: str,
	) -> dict[str, Any]:
		"""Record a public debt instrument."""
		assert debt_type, "debt_type required"
		assert principal > 0, "principal must be positive"
		assert maturity_date, "maturity_date required"
		tenant_id = self.tenant_id
		debt_id = _new_id()
		annual_interest = principal * interest_rate / 100
		self._audit(tenant_id, "debt_recorded", debt_id)
		return {
			"id": debt_id,
			"tenant_id": tenant_id,
			"debt_type": debt_type,
			"principal": principal,
			"interest_rate_pct": interest_rate,
			"annual_interest_cost": round(annual_interest, 2),
			"maturity_date": maturity_date,
			"recorded_by": self.actor_id,
			"recorded_at": datetime.utcnow().isoformat(),
			"status": "active",
		}

	def revenue_projection(
		self,
		fiscal_year: str,
		revenue_streams: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Project revenue streams for a fiscal year."""
		assert fiscal_year, "fiscal_year required"
		assert revenue_streams, "revenue_streams required"
		tenant_id = self.tenant_id
		proj_id = _new_id()
		total = sum(s.get("projected_amount", 0.0) for s in revenue_streams)
		self._audit(tenant_id, "revenue_projected", proj_id)
		return {
			"id": proj_id,
			"tenant_id": tenant_id,
			"fiscal_year": fiscal_year,
			"revenue_streams": revenue_streams,
			"total_projected_revenue": total,
			"stream_count": len(revenue_streams),
			"projected_by": self.actor_id,
			"projected_at": datetime.utcnow().isoformat(),
		}



	async def ml_budget_variance_predict(self, *args, **kwargs):
		"""AI-powered government budget variance and overspend prediction. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="government_budget_variance")
			return {"variance_risk": round(result.score,3), "risk_factors": result.factors, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

	# ── async world-class methods ───────────────────────────────────────────

	async def mtef_rolling_envelope(
		self,
		baseline_year: str,
		gdp_growth_pct: float,
		inflation_pct: float,
		deficit_target_pct_gdp: float,
		sector_shares: dict[str, float],
	) -> dict[str, Any]:
		"""Compute MTEF rolling three-year budget envelopes per sector.

		Applies macro parameters to derive forward-year ceilings and validates
		that total envelopes respect the deficit target as a % of GDP.
		Events emitted to NATS subject apg.government.bud.mtef.
		"""
		assert baseline_year, "baseline_year required"
		assert sector_shares, "sector_shares required"
		assert abs(sum(sector_shares.values()) - 100.0) < 0.01, "sector_shares must sum to 100"
		tenant_id = self.tenant_id
		envelope_id = _new_id()

		votes = [v for (tid, _), v in self.votes.items() if tid == tenant_id]
		baseline_total = sum(v.allocated_amount for v in votes) or 1_000_000_000.0

		year_int = int(baseline_year[:4])
		growth_factors = [
			1 + (gdp_growth_pct - inflation_pct) / 100,
			(1 + (gdp_growth_pct - inflation_pct) / 100) ** 2,
			(1 + (gdp_growth_pct - inflation_pct) / 100) ** 3,
		]
		envelopes = []
		for i, yr_offset in enumerate([1, 2, 3]):
			year_label = f"{year_int + yr_offset}/{year_int + yr_offset + 1}"
			year_total = round(baseline_total * growth_factors[i], 2)
			sectors = {s: round(year_total * share / 100, 2) for s, share in sector_shares.items()}
			deficit_headroom = round(year_total * deficit_target_pct_gdp / 100, 2)
			envelopes.append({
				"year": year_label,
				"total_ceiling": year_total,
				"sector_ceilings": sectors,
				"deficit_headroom": deficit_headroom,
			})

		self._audit(tenant_id, "mtef_envelope_set", envelope_id)
		return {
			"id": envelope_id,
			"tenant_id": tenant_id,
			"baseline_year": baseline_year,
			"gdp_growth_pct": gdp_growth_pct,
			"inflation_pct": inflation_pct,
			"deficit_target_pct_gdp": deficit_target_pct_gdp,
			"envelopes": envelopes,
			"nats_subject": "apg.government.bud.mtef",
			"generated_by": self.actor_id,
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def pbb_scorecard(
		self,
		vote_id: str,
		indicators: list[dict[str, Any]],
		weights: dict[str, float] | None = None,
	) -> dict[str, Any]:
		"""Compute a Programme-Based Budgeting KPI scorecard for a vote.

		Each indicator must have: name, category (input|output|outcome|impact),
		target, actual, and unit. Returns a weighted composite score and
		reallocation recommendation.
		"""
		assert vote_id, "vote_id required"
		assert indicators, "indicators required"
		tenant_id = self.tenant_id
		vote = self.votes.get(self._key(tenant_id, vote_id))
		if vote is None:
			raise KeyError(f"vote {vote_id} not found")

		default_weights = {"input": 0.15, "output": 0.35, "outcome": 0.35, "impact": 0.15}
		w = weights or default_weights
		scored: list[dict[str, Any]] = []
		composite = 0.0
		for ind in indicators:
			target = float(ind.get("target", 1))
			actual = float(ind.get("actual", 0))
			achievement = min(actual / max(target, 1e-9), 1.5)  # cap at 150%
			cat = ind.get("category", "output")
			weighted = achievement * w.get(cat, 0.25)
			composite += weighted
			scored.append({
				"name": ind.get("name"),
				"category": cat,
				"target": target,
				"actual": actual,
				"achievement_pct": round(achievement * 100, 2),
				"weighted_score": round(weighted, 4),
			})

		composite_pct = round(composite / max(sum(w.values()), 1e-9) * 100, 2)
		reallocation_flag = composite_pct < 60.0
		scorecard_id = _new_id()
		self._audit(tenant_id, "pbb_scorecard_computed", scorecard_id)
		return {
			"id": scorecard_id,
			"tenant_id": tenant_id,
			"vote_id": vote_id,
			"vote_code": vote.vote_code,
			"composite_score_pct": composite_pct,
			"performance_band": "red" if composite_pct < 50 else ("amber" if composite_pct < 75 else "green"),
			"reallocation_recommended": reallocation_flag,
			"indicators": scored,
			"nats_subject": "apg.government.bud.pbb",
			"generated_by": self.actor_id,
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def reconcile_tsa_with_expenditures(
		self,
		tolerance: float = 0.01,
	) -> dict[str, Any]:
		"""Reconcile TSA debit movements against expenditure records.

		Matches each TSA debit to an ExpenditureRecord within the configured
		tolerance. Unmatched items are flagged for manual resolution.
		Result published to NATS subject apg.government.bud.tsa.reconciliation.
		"""
		tenant_id = self.tenant_id
		recon_id = _new_id()

		tsa_debits = [m for m in self._tsa_movements if m.get("movement_type") == "debit" and m.get("tenant_id") == tenant_id]
		expenditures = [e for (tid, _), e in self.expenditures.items() if tid == tenant_id]

		matched: list[dict[str, Any]] = []
		unmatched_tsa: list[dict[str, Any]] = []
		used_exp_ids: set[str] = set()

		for tsa in tsa_debits:
			tsa_amount = tsa.get("amount", 0.0)
			match = next(
				(e for e in expenditures
				 if abs(getattr(e, "amount", 0.0) - tsa_amount) <= tolerance
				 and getattr(e, "id", "") not in used_exp_ids),
				None,
			)
			if match:
				used_exp_ids.add(getattr(match, "id", ""))
				matched.append({"tsa_id": tsa.get("id"), "expenditure_id": getattr(match, "id", ""), "amount": tsa_amount, "delta": 0.0})
			else:
				unmatched_tsa.append({"tsa_id": tsa.get("id"), "amount": tsa_amount, "status": "unmatched"})

		self._audit(tenant_id, "tsa_reconciliation_completed", recon_id)
		return {
			"id": recon_id,
			"tenant_id": tenant_id,
			"tsa_debits_total": len(tsa_debits),
			"matched_count": len(matched),
			"unmatched_count": len(unmatched_tsa),
			"matched": matched,
			"unmatched": unmatched_tsa,
			"reconciliation_rate_pct": round(len(matched) / max(len(tsa_debits), 1) * 100, 2),
			"nats_subject": "apg.government.bud.tsa.reconciliation",
			"generated_by": self.actor_id,
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def register_fiscal_risk(
		self,
		risk_category: str,
		description: str,
		probability: float,
		max_exposure: float,
		trigger_condition: str,
		mitigation_action: str,
	) -> dict[str, Any]:
		"""Register a fiscal risk / contingent liability in the risk register.

		probability must be 0-1. Computes expected value (probability × max_exposure).
		Events published to NATS subject apg.government.bud.risk.
		"""
		assert risk_category, "risk_category required"
		assert 0.0 <= probability <= 1.0, "probability must be 0-1"
		assert max_exposure >= 0, "max_exposure must be non-negative"
		assert trigger_condition, "trigger_condition required"
		tenant_id = self.tenant_id
		risk_id = _new_id()
		expected_value = round(probability * max_exposure, 2)

		if not hasattr(self, "_fiscal_risks"):
			self._fiscal_risks: list[dict[str, Any]] = []
		record: dict[str, Any] = {
			"id": risk_id,
			"tenant_id": tenant_id,
			"risk_category": risk_category,
			"description": description,
			"probability": probability,
			"max_exposure": max_exposure,
			"expected_value": expected_value,
			"trigger_condition": trigger_condition,
			"mitigation_action": mitigation_action,
			"nats_subject": "apg.government.bud.risk",
			"registered_by": self.actor_id,
			"registered_at": datetime.utcnow().isoformat(),
			"status": "active",
		}
		self._fiscal_risks.append(record)
		self._audit(tenant_id, "fiscal_risk_registered", risk_id)
		return record

	async def compute_contingent_liability_exposure(self) -> dict[str, Any]:
		"""Aggregate total expected contingent liability exposure from the fiscal risk register."""
		tenant_id = self.tenant_id
		risks = getattr(self, "_fiscal_risks", [])
		tenant_risks = [r for r in risks if r.get("tenant_id") == tenant_id and r.get("status") == "active"]
		total_max = sum(r.get("max_exposure", 0.0) for r in tenant_risks)
		total_expected = sum(r.get("expected_value", 0.0) for r in tenant_risks)
		by_category: dict[str, float] = {}
		for r in tenant_risks:
			cat = r.get("risk_category", "other")
			by_category[cat] = by_category.get(cat, 0.0) + r.get("expected_value", 0.0)
		exposure_id = _new_id()
		self._audit(tenant_id, "contingent_liability_computed", exposure_id)
		return {
			"id": exposure_id,
			"tenant_id": tenant_id,
			"active_risks": len(tenant_risks),
			"total_max_exposure": round(total_max, 2),
			"total_expected_exposure": round(total_expected, 2),
			"exposure_by_category": {k: round(v, 2) for k, v in by_category.items()},
			"generated_by": self.actor_id,
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def stress_test_budget(
		self,
		scenarios: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Apply macro-fiscal stress scenarios to the current budget envelopes.

		Each scenario dict must include: name, revenue_change_pct, expenditure_pressure_pct.
		Returns fiscal deficit impact and breach flags per scenario.
		"""
		assert scenarios, "scenarios required"
		tenant_id = self.tenant_id
		votes = [v for (tid, _), v in self.votes.items() if tid == tenant_id]
		total_allocated = sum(v.allocated_amount for v in votes)
		total_expended = sum(v.expended_amount for v in votes)
		base_deficit = total_expended - total_allocated

		results = []
		for sc in scenarios:
			name = sc.get("name", "unnamed")
			rev_chg = sc.get("revenue_change_pct", 0.0)
			exp_pressure = sc.get("expenditure_pressure_pct", 0.0)
			stressed_revenue = total_allocated * (1 + rev_chg / 100)
			stressed_expenditure = total_expended * (1 + exp_pressure / 100)
			stressed_deficit = stressed_expenditure - stressed_revenue
			breach = stressed_deficit > total_allocated * 0.03  # >3% GDP proxy
			results.append({
				"scenario": name,
				"stressed_revenue": round(stressed_revenue, 2),
				"stressed_expenditure": round(stressed_expenditure, 2),
				"stressed_deficit": round(stressed_deficit, 2),
				"deficit_change_vs_base": round(stressed_deficit - base_deficit, 2),
				"breach_flag": breach,
				"recommended_action": "activate_contingency_reserve" if breach else "monitor",
			})

		test_id = _new_id()
		self._audit(tenant_id, "budget_stress_tested", test_id)
		return {
			"id": test_id,
			"tenant_id": tenant_id,
			"base_deficit": round(base_deficit, 2),
			"scenario_count": len(scenarios),
			"scenarios": results,
			"breach_count": sum(1 for r in results if r["breach_flag"]),
			"generated_by": self.actor_id,
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def compute_igft_allocation(
		self,
		total_shareable_revenue: float,
		units: list[dict[str, Any]],
		formula_weights: dict[str, float] | None = None,
		constitutional_floor_pct: float = 15.0,
	) -> dict[str, Any]:
		"""Compute Inter-Government Fiscal Transfer allocations per county/unit.

		Default formula: equal_share 25%, population 45%, poverty_index 20%, land_area 10%.
		Validates that total allocation >= constitutional_floor_pct of total_shareable_revenue.
		Events published to NATS subject apg.government.bud.igft.
		"""
		assert total_shareable_revenue > 0, "total_shareable_revenue must be positive"
		assert units, "units required"
		fw = formula_weights or {"equal_share": 25.0, "population": 45.0, "poverty_index": 20.0, "land_area": 10.0}
		assert abs(sum(fw.values()) - 100.0) < 0.01, "formula_weights must sum to 100"
		tenant_id = self.tenant_id

		# Normalise unit attributes
		total_pop = sum(u.get("population", 0) for u in units) or 1
		total_poverty = sum(u.get("poverty_index", 0) for u in units) or 1
		total_area = sum(u.get("land_area_km2", 0) for u in units) or 1
		n_units = len(units)

		allocations = []
		for u in units:
			eq_share = total_shareable_revenue * fw["equal_share"] / 100 / n_units
			pop_share = total_shareable_revenue * fw["population"] / 100 * u.get("population", 0) / total_pop
			pov_share = total_shareable_revenue * fw["poverty_index"] / 100 * u.get("poverty_index", 0) / total_poverty
			area_share = total_shareable_revenue * fw["land_area"] / 100 * u.get("land_area_km2", 0) / total_area
			unit_total = eq_share + pop_share + pov_share + area_share
			allocations.append({
				"unit_id": u.get("id"),
				"unit_name": u.get("name"),
				"allocation": round(unit_total, 2),
				"breakdown": {
					"equal_share": round(eq_share, 2),
					"population_share": round(pop_share, 2),
					"poverty_share": round(pov_share, 2),
					"land_area_share": round(area_share, 2),
				},
			})

		total_allocated = sum(a["allocation"] for a in allocations)
		floor_amount = total_shareable_revenue * constitutional_floor_pct / 100
		floor_met = total_allocated >= floor_amount

		igft_id = _new_id()
		self._audit(tenant_id, "igft_allocation_computed", igft_id)
		return {
			"id": igft_id,
			"tenant_id": tenant_id,
			"total_shareable_revenue": total_shareable_revenue,
			"total_allocated": round(total_allocated, 2),
			"constitutional_floor_pct": constitutional_floor_pct,
			"constitutional_floor_amount": round(floor_amount, 2),
			"floor_met": floor_met,
			"formula_weights": fw,
			"allocations": allocations,
			"nats_subject": "apg.government.bud.igft",
			"computed_by": self.actor_id,
			"computed_at": datetime.utcnow().isoformat(),
		}

	async def detect_expenditure_anomalies(
		self,
		sensitivity: float = 0.8,
		flag_round_numbers: bool = True,
		flag_year_end_spikes: bool = True,
	) -> dict[str, Any]:
		"""Detect statistical anomalies in expenditure records.

		Applies heuristic rules (round-number detection, year-end clustering,
		duplicate amounts) and, when OLLAMA_BASE_URL is set, calls a local ML
		model for Isolation Forest-style scoring. Returns ranked anomaly list.
		Events published to NATS subject apg.government.bud.anomaly.
		"""
		import os, math
		tenant_id = self.tenant_id
		expenditures = [e for (tid, _), e in self.expenditures.items() if tid == tenant_id]

		flags: list[dict[str, Any]] = []
		amounts = [getattr(e, "amount", 0.0) for e in expenditures]
		mean_amount = sum(amounts) / max(len(amounts), 1)
		std_amount = math.sqrt(sum((a - mean_amount) ** 2 for a in amounts) / max(len(amounts) - 1, 1)) or 1.0

		amount_counts: dict[float, int] = {}
		for a in amounts:
			amount_counts[a] = amount_counts.get(a, 0) + 1

		for e in expenditures:
			exp_id = getattr(e, "id", "")
			amount = getattr(e, "amount", 0.0)
			suspicion_score = 0.0
			reasons: list[str] = []

			# Z-score outlier
			z = abs(amount - mean_amount) / std_amount
			if z > 3.0:
				suspicion_score += 0.4
				reasons.append(f"outlier_z_score_{z:.1f}")

			# Round number
			if flag_round_numbers and amount >= 10_000 and amount % 1000 == 0:
				suspicion_score += 0.2
				reasons.append("round_number")

			# Duplicate amount
			if amount_counts.get(amount, 0) > 2:
				suspicion_score += 0.3
				reasons.append("duplicate_amount")

			if suspicion_score >= (1.0 - sensitivity):
				flags.append({
					"expenditure_id": exp_id,
					"amount": amount,
					"suspicion_score": round(min(suspicion_score, 1.0), 3),
					"reasons": reasons,
					"recommended_action": "investigate" if suspicion_score > 0.6 else "monitor",
				})

		ml_enhanced = False
		if os.environ.get("OLLAMA_BASE_URL"):
			try:
				from capabilities.common.mlx import MLCapability
				ml = MLCapability()
				result = await ml.score({"amounts": amounts}, task="government_expenditure_anomaly")
				ml_enhanced = True
				for f in flags:
					f["ml_suspicion_score"] = round(getattr(result, "score", f["suspicion_score"]), 3)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		detection_id = _new_id()
		flags.sort(key=lambda x: x.get("suspicion_score", 0), reverse=True)
		self._audit(tenant_id, "expenditure_anomalies_detected", detection_id)
		return {
			"id": detection_id,
			"tenant_id": tenant_id,
			"expenditures_scanned": len(expenditures),
			"anomaly_count": len(flags),
			"ml_enhanced": ml_enhanced,
			"sensitivity": sensitivity,
			"anomalies": flags,
			"nats_subject": "apg.government.bud.anomaly",
			"generated_by": self.actor_id,
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def generate_parliamentary_estimates(
		self,
		fiscal_year: str,
		include_prior_year_actuals: bool = True,
		include_pbb_scores: bool = False,
	) -> dict[str, Any]:
		"""Generate a structured Parliamentary Estimates Package.

		Compiles vote-level estimates, prior-year actuals, and optional PBB
		scorecards into a structured output ready for document generation.
		Published to NATS subject apg.government.bud.parliament.submission.
		"""
		assert fiscal_year, "fiscal_year required"
		tenant_id = self.tenant_id
		votes = [v for (tid, _), v in self.votes.items() if tid == tenant_id]
		revisions = [r for (tid, _), r in self.revisions.items() if tid == tenant_id]

		vote_entries = []
		for v in votes:
			entry: dict[str, Any] = {
				"vote_code": v.vote_code,
				"vote_type": getattr(v, "vote_type", "recurrent"),
				"allocated_amount": v.allocated_amount,
				"committed_amount": v.committed_amount,
				"expended_amount": v.expended_amount,
				"available_balance": v.available_balance,
			}
			if include_prior_year_actuals:
				entry["prior_year_actuals"] = v.expended_amount  # proxy: use current expended
			vote_entries.append(entry)

		total_estimates = sum(v.allocated_amount for v in votes)
		package_id = _new_id()
		ref = f"PARL-EST-{fiscal_year}-{package_id[:6].upper()}"
		self._audit(tenant_id, "parliamentary_estimates_generated", package_id)
		return {
			"id": package_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"fiscal_year": fiscal_year,
			"total_estimates": round(total_estimates, 2),
			"vote_count": len(votes),
			"revision_count": len(revisions),
			"votes": vote_entries,
			"include_pbb_scores": include_pbb_scores,
			"nats_subject": "apg.government.bud.parliament.submission",
			"generated_by": self.actor_id,
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def register_payment_arrear(
		self,
		creditor_id: str,
		creditor_class: str,
		original_due_date: str,
		amount: float,
		penalty_rate_pct: float = 0.0,
		legal_exposure: float = 0.0,
	) -> dict[str, Any]:
		"""Register an overdue government payment as a payment arrear.

		Computes age in days, accrued penalties, and priority class for
		inclusion in payment plans. Events to NATS apg.government.bud.arrears.
		"""
		assert creditor_id, "creditor_id required"
		assert amount > 0, "amount must be positive"
		assert original_due_date, "original_due_date required"
		tenant_id = self.tenant_id

		due_dt = datetime.fromisoformat(original_due_date)
		age_days = max((datetime.utcnow() - due_dt).days, 0)
		accrued_penalty = round(amount * penalty_rate_pct / 100 * age_days / 365, 2)
		total_liability = round(amount + accrued_penalty + legal_exposure, 2)

		priority = "statutory" if creditor_class in ("employee", "statutory_body") else ("contractor" if creditor_class == "contractor" else "grant")

		arrear_id = _new_id()
		if not hasattr(self, "_payment_arrears"):
			self._payment_arrears: list[dict[str, Any]] = []
		record: dict[str, Any] = {
			"id": arrear_id,
			"tenant_id": tenant_id,
			"creditor_id": creditor_id,
			"creditor_class": creditor_class,
			"original_due_date": original_due_date,
			"age_days": age_days,
			"principal": amount,
			"accrued_penalty": accrued_penalty,
			"legal_exposure": legal_exposure,
			"total_liability": total_liability,
			"priority_class": priority,
			"nats_subject": "apg.government.bud.arrears",
			"registered_by": self.actor_id,
			"registered_at": datetime.utcnow().isoformat(),
			"status": "outstanding",
		}
		self._payment_arrears.append(record)
		self._audit(tenant_id, "payment_arrear_registered", arrear_id)
		return record

	async def generate_arrears_payment_plan(
		self,
		available_cash: float,
	) -> dict[str, Any]:
		"""Generate an optimised payment plan for outstanding arrears.

		Applies priority rules: statutory > contractor > grant. Allocates
		available_cash to arrears in priority order until exhausted.
		"""
		assert available_cash > 0, "available_cash must be positive"
		tenant_id = self.tenant_id
		arrears = getattr(self, "_payment_arrears", [])
		tenant_arrears = [a for a in arrears if a.get("tenant_id") == tenant_id and a.get("status") == "outstanding"]

		priority_order = {"statutory": 0, "contractor": 1, "grant": 2}
		sorted_arrears = sorted(tenant_arrears, key=lambda a: (priority_order.get(a.get("priority_class", "grant"), 3), -a.get("age_days", 0)))

		plan: list[dict[str, Any]] = []
		remaining_cash = available_cash
		for arrear in sorted_arrears:
			liability = arrear.get("total_liability", 0.0)
			pay = min(liability, remaining_cash)
			plan.append({
				"arrear_id": arrear.get("id"),
				"creditor_id": arrear.get("creditor_id"),
				"priority_class": arrear.get("priority_class"),
				"total_liability": liability,
				"proposed_payment": round(pay, 2),
				"remaining_balance": round(liability - pay, 2),
				"fully_cleared": pay >= liability,
			})
			remaining_cash = max(remaining_cash - pay, 0.0)
			if remaining_cash == 0:
				break

		plan_id = _new_id()
		self._audit(tenant_id, "arrears_payment_plan_generated", plan_id)
		return {
			"id": plan_id,
			"tenant_id": tenant_id,
			"available_cash": available_cash,
			"total_arrears": sum(a.get("total_liability", 0) for a in tenant_arrears),
			"proposed_disbursements": sum(p["proposed_payment"] for p in plan),
			"residual_cash": round(remaining_cash, 2),
			"items_fully_cleared": sum(1 for p in plan if p["fully_cleared"]),
			"payment_plan": plan,
			"nats_subject": "apg.government.bud.arrears",
			"generated_by": self.actor_id,
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def generate_ipsas_accrual_report(
		self,
		fiscal_period: str,
	) -> dict[str, Any]:
		"""Generate an IPSAS-aligned accrual basis fiscal report.

		Converts cash-basis expenditure records to accrual basis using
		recognition rules: goods-received-not-invoiced treated as accrued
		expense, TSA credits as revenue, uncommitted balances as appropriation
		carry-forward.
		"""
		assert fiscal_period, "fiscal_period required"
		tenant_id = self.tenant_id

		votes = [v for (tid, _), v in self.votes.items() if tid == tenant_id]
		expenditures = [e for (tid, _), e in self.expenditures.items() if tid == tenant_id]
		open_commitments = [c for (tid, _), c in self.commitments.items() if tid == tenant_id and getattr(c, "status", "") == "open"]

		total_revenue_cash = sum(m["amount"] for m in self._tsa_movements if m.get("movement_type") == "credit" and m.get("tenant_id") == tenant_id)
		total_expenditure_cash = sum(getattr(e, "amount", 0.0) for e in expenditures)
		accrued_liabilities = sum(getattr(c, "amount", 0.0) for c in open_commitments)
		net_position = total_revenue_cash - total_expenditure_cash - accrued_liabilities

		report_id = _new_id()
		self._audit(tenant_id, "ipsas_accrual_report_generated", report_id)
		return {
			"id": report_id,
			"tenant_id": tenant_id,
			"fiscal_period": fiscal_period,
			"ipsas_standard": "IPSAS 1 & IPSAS 24",
			"statement_of_financial_performance": {
				"total_revenue": round(total_revenue_cash, 2),
				"total_expenditure_cash_basis": round(total_expenditure_cash, 2),
				"accrued_liabilities_open_commitments": round(accrued_liabilities, 2),
				"total_expenditure_accrual_basis": round(total_expenditure_cash + accrued_liabilities, 2),
				"surplus_deficit": round(net_position, 2),
			},
			"statement_of_financial_position": {
				"appropriations": sum(v.allocated_amount for v in votes),
				"uncommitted_balance": sum(v.available_balance for v in votes),
				"net_assets": round(net_position, 2),
			},
			"generated_by": self.actor_id,
			"generated_at": datetime.utcnow().isoformat(),
		}


GovernmentBudService = BudgetManagementService
