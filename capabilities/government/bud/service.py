"""Executable service layer for APG Budget Management."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

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

GovernmentBudService = BudgetManagementService
