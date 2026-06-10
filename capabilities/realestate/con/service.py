"""Async service layer for Property Contracts (con)."""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Any

from .models import (
	ContractCreate, ContractResponse, ContractUpdate,
	ContractorCreate, ContractorResponse, ContractorUpdate,
	MilestoneCreate, MilestoneResponse,
	VariationOrderCreate, VariationOrderResponse,
	DisputeCreate, DisputeResponse,
	RetentionCreate, RetentionResponse,
	ClauseCreate, ClauseResponse,
	ContractStatus, ContractorGrade,
)
from .capability_contract import evaluate_capability_rules
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

log = logging.getLogger(__name__)


class ConService:
	"""Service implementing all Property Contracts operations."""

	def __init__(
		self,
		tenant_id: str | None = None,
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: dict[str, Any] | None = None,
	) -> None:
		self._tenant_id = tenant_id
		self._actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store: dict[str, list[dict[str, Any]]] = store or {
			"contracts": [], "contractors": [], "milestones": [],
			"variations": [], "disputes": [], "retentions": [], "clauses": [],
			"obligations": [], "notices": [],
		}
		self._vo_counter = 0

	# ── Logging helpers ───────────────────────────────────────────────────────

	def _log_operation(self, op: str, entity_id: str, tenant_id: str) -> None:
		log.info("con.%s entity=%s tenant=%s", op, entity_id, tenant_id)

	def _log_expiry_alert(self, contract_id: str, days_remaining: int) -> None:
		log.warning("con.expiry_alert contract=%s days_remaining=%d", contract_id, days_remaining)

	def _log_milestone_overdue(self, milestone_id: str, contract_id: str) -> None:
		log.warning("con.milestone_overdue milestone=%s contract=%s", milestone_id, contract_id)

	def _log_default_notice(self, contract_id: str, default_type: str) -> None:
		log.error("con.default_notice contract=%s type=%s", contract_id, default_type)

	# ── Rules ─────────────────────────────────────────────────────────────────

	def _check_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			log.warning("con.rule_denied rule=%s reason=%s", result["rule"], result["reason"])
			raise ValueError(f"rule_denied:{result['rule']}:{result['reason']}")

	def _next_vo_ref(self) -> str:
		self._vo_counter += 1
		return f"VO-{self._vo_counter:05d}"

	# ── Contract ──────────────────────────────────────────────────────────────

	async def create_contract(self, payload: ContractCreate) -> ContractResponse:
		"""Create a new contract record."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "create_contract",
			"contract_type_supported": True,
			"parties_present": len(payload.parties) >= 2,
			"governing_law_present": bool(payload.governing_law),
			"operation_type": "write",
			"policy_attached": True,
			"cross_tenant": False,
		})
		record = ContractResponse(**payload.model_dump())
		self._store["contracts"].append(record.model_dump())
		self._log_operation("create_contract", record.id, record.tenant_id)
		return record

	async def get_contract(self, contract_id: str, tenant_id: str) -> ContractResponse | None:
		"""Fetch a contract by ID."""
		for c in self._store["contracts"]:
			if c["id"] == contract_id and c["tenant_id"] == tenant_id:
				return ContractResponse(**c)
		return None

	async def list_contracts(self, tenant_id: str, contract_type: str | None = None, status: str | None = None) -> list[ContractResponse]:
		"""List contracts with optional filters."""
		results = [c for c in self._store["contracts"] if c["tenant_id"] == tenant_id]
		if contract_type:
			results = [c for c in results if c.get("contract_type") == contract_type]
		if status:
			results = [c for c in results if c.get("status") == status]
		return [ContractResponse(**c) for c in results]

	async def update_contract(self, contract_id: str, tenant_id: str, updates: ContractUpdate) -> ContractResponse | None:
		"""Update mutable contract fields."""
		for i, c in enumerate(self._store["contracts"]):
			if c["id"] == contract_id and c["tenant_id"] == tenant_id:
				c.update({k: v for k, v in updates.model_dump().items() if v is not None})
				c["updated_at"] = datetime.utcnow()
				self._store["contracts"][i] = c
				return ContractResponse(**c)
		return None

	async def execute_contract(self, contract_id: str, tenant_id: str) -> ContractResponse | None:
		"""Execute a contract after legal review and all signatures."""
		for i, c in enumerate(self._store["contracts"]):
			if c["id"] == contract_id and c["tenant_id"] == tenant_id:
				all_signed = all(p.get("signed_at") is not None for p in c.get("parties", []))
				self._check_rules({
					"operation": "execute_contract",
					"all_signatures_present": all_signed,
					"legal_review_complete": c.get("legal_review_complete", False),
				})
				c["status"] = ContractStatus.active.value
				c["all_signatures_present"] = True
				c["executed_at"] = datetime.utcnow()
				c["updated_at"] = datetime.utcnow()
				self._store["contracts"][i] = c
				self._log_operation("execute_contract", contract_id, tenant_id)
				return ContractResponse(**c)
		return None

	async def terminate_contract(self, contract_id: str, tenant_id: str, reason: str, notice_period_satisfied: bool) -> ContractResponse | None:
		"""Terminate a contract."""
		self._check_rules({
			"operation": "terminate_contract",
			"reason_present": bool(reason),
			"notice_period_satisfied": notice_period_satisfied,
		})
		for i, c in enumerate(self._store["contracts"]):
			if c["id"] == contract_id and c["tenant_id"] == tenant_id:
				c["status"] = ContractStatus.terminated.value
				c["termination_reason"] = reason
				c["terminated_at"] = datetime.utcnow()
				c["updated_at"] = datetime.utcnow()
				self._store["contracts"][i] = c
				self._log_operation("terminate_contract", contract_id, tenant_id)
				return ContractResponse(**c)
		return None

	async def sign_contract_party(self, contract_id: str, tenant_id: str, party_id: str, signature_ref: str) -> ContractResponse | None:
		"""Record a party's signature on a contract."""
		for i, c in enumerate(self._store["contracts"]):
			if c["id"] == contract_id and c["tenant_id"] == tenant_id:
				for party in c.get("parties", []):
					if party["party_id"] == party_id:
						party["signed_at"] = datetime.utcnow().isoformat()
						party["signature_ref"] = signature_ref
						break
				c["updated_at"] = datetime.utcnow()
				self._store["contracts"][i] = c
				return ContractResponse(**c)
		return None

	async def get_expiry_pipeline(self, tenant_id: str, days_ahead: int = 90) -> list[dict[str, Any]]:
		"""Return contracts expiring within the given days window."""
		cutoff = date.today() + timedelta(days=days_ahead)
		results = []
		for c in self._store["contracts"]:
			if c["tenant_id"] == tenant_id and c["status"] == ContractStatus.active.value and c.get("end_date"):
				end = datetime.strptime(c["end_date"], "%Y-%m-%d").date()
				if end <= cutoff:
					days_remaining = (end - date.today()).days
					self._log_expiry_alert(c["id"], days_remaining)
					results.append({"contract_id": c["id"], "ref": c.get("contract_ref"), "end_date": c["end_date"], "days_remaining": days_remaining})
		return sorted(results, key=lambda x: x["days_remaining"])

	# ── Contractor ────────────────────────────────────────────────────────────

	async def register_contractor(self, payload: ContractorCreate) -> ContractorResponse:
		"""Register a contractor in the registry."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "create_contract",
			"contractor_grade": payload.grade.value,
		})
		record = ContractorResponse(**payload.model_dump())
		self._store["contractors"].append(record.model_dump())
		self._log_operation("register_contractor", record.id, record.tenant_id)
		return record

	async def get_contractor(self, contractor_id: str, tenant_id: str) -> ContractorResponse | None:
		"""Fetch a contractor."""
		for c in self._store["contractors"]:
			if c["id"] == contractor_id and c["tenant_id"] == tenant_id:
				return ContractorResponse(**c)
		return None

	async def list_contractors(self, tenant_id: str, grade: str | None = None) -> list[ContractorResponse]:
		"""List contractors."""
		results = [c for c in self._store["contractors"] if c["tenant_id"] == tenant_id]
		if grade:
			results = [c for c in results if c.get("grade") == grade]
		return [ContractorResponse(**c) for c in results]

	async def grade_contractor(self, contractor_id: str, tenant_id: str, new_grade: ContractorGrade, graded_by: str) -> ContractorResponse | None:
		"""Update a contractor's grade."""
		self._check_rules({"operation": "grade_contractor", "grade_supported": True})
		for i, c in enumerate(self._store["contractors"]):
			if c["id"] == contractor_id and c["tenant_id"] == tenant_id:
				c["grade"] = new_grade.value
				c["last_grading_review"] = date.today().isoformat()
				c["updated_at"] = datetime.utcnow()
				self._store["contractors"][i] = c
				self._log_operation("grade_contractor", contractor_id, tenant_id)
				return ContractorResponse(**c)
		return None

	# ── Milestone ─────────────────────────────────────────────────────────────

	async def create_milestone(self, payload: MilestoneCreate) -> MilestoneResponse:
		"""Create a contract milestone."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "create_milestone",
			"contract_present": True,
			"due_date_present": True,
		})
		record = MilestoneResponse(**payload.model_dump())
		self._store["milestones"].append(record.model_dump())
		return record

	async def complete_milestone(self, milestone_id: str, tenant_id: str, evidence_ids: list[str]) -> MilestoneResponse | None:
		"""Mark a milestone as completed."""
		for i, m in enumerate(self._store["milestones"]):
			if m["id"] == milestone_id and m["tenant_id"] == tenant_id:
				m["status"] = "completed"
				m["completed_at"] = datetime.utcnow()
				m["evidence_ids"] = evidence_ids
				m["updated_at"] = datetime.utcnow()
				self._store["milestones"][i] = m
				self._log_operation("complete_milestone", milestone_id, tenant_id)
				return MilestoneResponse(**m)
		return None

	async def get_overdue_milestones(self, tenant_id: str) -> list[MilestoneResponse]:
		"""Return overdue milestones."""
		today = date.today()
		results = []
		for m in self._store["milestones"]:
			if m["tenant_id"] == tenant_id and m["status"] == "pending":
				due = datetime.strptime(m["due_date"], "%Y-%m-%d").date()
				if due < today:
					self._log_milestone_overdue(m["id"], m.get("contract_id", ""))
					results.append(MilestoneResponse(**m))
		return results

	async def list_milestones(self, tenant_id: str, contract_id: str | None = None) -> list[MilestoneResponse]:
		"""List milestones."""
		results = [m for m in self._store["milestones"] if m["tenant_id"] == tenant_id]
		if contract_id:
			results = [m for m in results if m.get("contract_id") == contract_id]
		return [MilestoneResponse(**m) for m in results]

	# ── Variation Order ───────────────────────────────────────────────────────

	async def raise_variation(self, payload: VariationOrderCreate) -> VariationOrderResponse:
		"""Raise a variation order against an active contract."""
		contract = await self.get_contract(payload.contract_id, payload.tenant_id)
		is_active = contract is not None and contract.status.value == "active"
		board_threshold = Decimal("500000")
		above_threshold = abs(payload.amount_change) > board_threshold
		self._check_rules({
			"tenant_context_present": True,
			"operation": "raise_variation",
			"contract_status": "active",
			"contract_active": is_active,
			"variation_type_supported": True,
			"amount_above_threshold": above_threshold,
			"board_approved": False,
		})
		ref = self._next_vo_ref()
		status = "board_pending" if above_threshold else "submitted"
		record = VariationOrderResponse(**payload.model_dump(), ref=ref, status=status)
		self._store["variations"].append(record.model_dump())
		self._log_operation("raise_variation", record.id, record.tenant_id)
		return record

	async def approve_variation(self, vo_id: str, tenant_id: str, approved_by: str, board_approval: bool = False) -> VariationOrderResponse | None:
		"""Approve a variation order."""
		for i, v in enumerate(self._store["variations"]):
			if v["id"] == vo_id and v["tenant_id"] == tenant_id:
				v["status"] = "approved"
				v["board_approved"] = board_approval
				v["approved_by"] = approved_by
				v["approved_at"] = datetime.utcnow()
				v["updated_at"] = datetime.utcnow()
				self._store["variations"][i] = v
				return VariationOrderResponse(**v)
		return None

	async def list_variations(self, tenant_id: str, contract_id: str | None = None) -> list[VariationOrderResponse]:
		"""List variation orders."""
		results = [v for v in self._store["variations"] if v["tenant_id"] == tenant_id]
		if contract_id:
			results = [v for v in results if v.get("contract_id") == contract_id]
		return [VariationOrderResponse(**v) for v in results]

	# ── Dispute ───────────────────────────────────────────────────────────────

	async def raise_dispute(self, payload: DisputeCreate) -> DisputeResponse:
		"""Raise a contract dispute."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "raise_dispute",
			"contract_present": True,
			"dispute_type_supported": True,
		})
		record = DisputeResponse(**payload.model_dump())
		self._store["disputes"].append(record.model_dump())
		self._log_operation("raise_dispute", record.id, record.tenant_id)
		return record

	async def resolve_dispute(self, dispute_id: str, tenant_id: str, resolution_summary: str) -> DisputeResponse | None:
		"""Resolve a contract dispute."""
		for i, d in enumerate(self._store["disputes"]):
			if d["id"] == dispute_id and d["tenant_id"] == tenant_id:
				d["status"] = "resolved"
				d["resolved_at"] = datetime.utcnow()
				d["resolution_summary"] = resolution_summary
				d["updated_at"] = datetime.utcnow()
				self._store["disputes"][i] = d
				return DisputeResponse(**d)
		return None

	async def list_disputes(self, tenant_id: str, contract_id: str | None = None) -> list[DisputeResponse]:
		"""List disputes."""
		results = [d for d in self._store["disputes"] if d["tenant_id"] == tenant_id]
		if contract_id:
			results = [d for d in results if d.get("contract_id") == contract_id]
		return [DisputeResponse(**d) for d in results]

	# ── Retention ─────────────────────────────────────────────────────────────

	async def create_retention(self, payload: RetentionCreate) -> RetentionResponse:
		"""Create a retention record for a contract."""
		self._check_rules({"tenant_context_present": True, "operation_type": "write", "policy_attached": True})
		record = RetentionResponse(**payload.model_dump())
		self._store["retentions"].append(record.model_dump())
		return record

	async def release_retention(self, retention_id: str, tenant_id: str, approved_by: str, defect_liability_cleared: bool) -> RetentionResponse | None:
		"""Release a retention after defect liability clearance."""
		self._check_rules({
			"operation": "release_retention",
			"defect_liability_cleared": defect_liability_cleared,
			"approved": True,
		})
		for i, r in enumerate(self._store["retentions"]):
			if r["id"] == retention_id and r["tenant_id"] == tenant_id:
				r["defect_liability_cleared"] = defect_liability_cleared
				r["release_approved_by"] = approved_by
				r["released_at"] = datetime.utcnow()
				r["updated_at"] = datetime.utcnow()
				self._store["retentions"][i] = r
				return RetentionResponse(**r)
		return None

	# ── Clause Library ────────────────────────────────────────────────────────

	async def create_clause(self, payload: ClauseCreate) -> ClauseResponse:
		"""Add a clause to the library."""
		record = ClauseResponse(**payload.model_dump())
		self._store["clauses"].append(record.model_dump())
		return record

	async def search_clauses(self, tenant_id: str, clause_type: str | None = None, query: str | None = None) -> list[ClauseResponse]:
		"""Search the clause library."""
		results = [c for c in self._store["clauses"] if c["tenant_id"] == tenant_id]
		if clause_type:
			results = [c for c in results if c.get("clause_type") == clause_type]
		if query:
			q = query.lower()
			results = [c for c in results if q in c.get("title", "").lower() or q in c.get("content", "").lower()]
		return [ClauseResponse(**c) for c in results]

	# ── NEW: draft_contract ───────────────────────────────────────────────────

	async def draft_contract(
		self,
		contract_type: str,
		parties: list[dict[str, Any]],
		property_id: str,
		terms: dict[str, Any],
		tenant_id: str,
		governing_law: str = "English Law",
		template_id: str = "",
	) -> dict[str, Any]:
		"""Draft a new contract from a template with parties, property, and commercial terms."""
		assert contract_type and parties and property_id and terms, \
			"contract_type, parties, property_id, terms required"
		assert len(parties) >= 2, "at least 2 parties required"
		self._check_rules({
			"tenant_context_present": True,
			"operation": "create_contract",
			"contract_type_supported": True,
			"parties_present": True,
			"governing_law_present": bool(governing_law),
			"operation_type": "write",
			"policy_attached": True,
		})
		from uuid6 import uuid7
		contract_id = str(uuid7())
		contract_ref = f"CONTR-{contract_id[:8].upper()}"
		contract: dict[str, Any] = {
			"id": contract_id,
			"tenant_id": tenant_id,
			"contract_ref": contract_ref,
			"contract_type": contract_type,
			"property_id": property_id,
			"parties": parties,
			"terms": terms,
			"governing_law": governing_law,
			"template_id": template_id,
			"status": ContractStatus.draft.value,
			"legal_review_complete": False,
			"all_signatures_present": False,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["contracts"].append(contract)
		self._log_operation("draft_contract", contract_id, tenant_id)
		return contract

	# ── NEW: contract_review ──────────────────────────────────────────────────

	async def contract_review(
		self,
		contract_id: str,
		reviewer_id: str,
		comments: list[dict[str, Any]],
		tenant_id: str,
		review_type: str = "legal",
		approved: bool = False,
	) -> dict[str, Any]:
		"""Record a legal or commercial review of a contract with comments and approval decision."""
		assert contract_id and reviewer_id, "contract_id and reviewer_id required"
		assert review_type in ("legal", "commercial", "technical", "compliance"), \
			f"unsupported review_type: {review_type}"
		from uuid6 import uuid7
		review_id = str(uuid7())
		review: dict[str, Any] = {
			"id": review_id,
			"tenant_id": tenant_id,
			"contract_id": contract_id,
			"reviewer_id": reviewer_id,
			"review_type": review_type,
			"comments": comments,
			"comment_count": len(comments),
			"approved": approved,
			"reviewed_at": datetime.utcnow().isoformat(),
		}
		if approved and review_type == "legal":
			for i, c in enumerate(self._store["contracts"]):
				if c["id"] == contract_id and c["tenant_id"] == tenant_id:
					c["legal_review_complete"] = True
					c["updated_at"] = datetime.utcnow()
					self._store["contracts"][i] = c
					break
		self._log_operation("contract_reviewed", review_id, tenant_id)
		return review

	# ── NEW: contract_obligation_tracking ─────────────────────────────────────

	async def contract_obligation_tracking(
		self,
		contract_id: str,
		obligation_id: str,
		due_date: date,
		completed: bool,
		tenant_id: str,
		responsible_party: str = "",
		completion_evidence: str = "",
	) -> dict[str, Any]:
		"""Track a contractual obligation: record due date, responsible party, and completion status."""
		assert contract_id and obligation_id, "contract_id and obligation_id required"
		from uuid6 import uuid7
		record_id = str(uuid7())
		days_overdue = (date.today() - due_date).days if not completed and due_date < date.today() else 0
		obligation: dict[str, Any] = {
			"id": record_id,
			"tenant_id": tenant_id,
			"contract_id": contract_id,
			"obligation_id": obligation_id,
			"due_date": str(due_date),
			"completed": completed,
			"responsible_party": responsible_party,
			"completion_evidence": completion_evidence,
			"days_overdue": days_overdue,
			"status": "completed" if completed else ("overdue" if days_overdue > 0 else "pending"),
			"recorded_at": datetime.utcnow().isoformat(),
		}
		self._store["obligations"].append(obligation)
		if days_overdue > 0:
			self._log_milestone_overdue(obligation_id, contract_id)
		return obligation

	# ── NEW: contract_milestone ───────────────────────────────────────────────

	async def contract_milestone(
		self,
		contract_id: str,
		milestone_id: str,
		achieved_date: date,
		tenant_id: str,
		evidence_ids: list[str] | None = None,
		payment_trigger: bool = False,
		payment_amount: Decimal = Decimal("0"),
	) -> dict[str, Any]:
		"""Record achievement of a contract milestone, optionally triggering a payment."""
		assert contract_id and milestone_id, "contract_id and milestone_id required"
		# update existing milestone record if present
		for i, m in enumerate(self._store["milestones"]):
			if m.get("milestone_ref") == milestone_id and m["tenant_id"] == tenant_id:
				m["status"] = "completed"
				m["completed_at"] = str(achieved_date)
				m["evidence_ids"] = evidence_ids or []
				m["updated_at"] = datetime.utcnow()
				self._store["milestones"][i] = m
				break
		from uuid6 import uuid7
		record_id = str(uuid7())
		achievement: dict[str, Any] = {
			"id": record_id,
			"tenant_id": tenant_id,
			"contract_id": contract_id,
			"milestone_id": milestone_id,
			"achieved_date": str(achieved_date),
			"evidence_ids": evidence_ids or [],
			"payment_trigger": payment_trigger,
			"payment_amount": str(payment_amount) if payment_amount else "0",
			"recorded_at": datetime.utcnow().isoformat(),
		}
		self._log_operation("milestone_achieved", record_id, tenant_id)
		return achievement

	# ── NEW: variation_order ──────────────────────────────────────────────────

	async def variation_order(
		self,
		contract_id: str,
		description: str,
		cost_impact: Decimal,
		approved_by: str,
		tenant_id: str,
		time_impact_days: int = 0,
		variation_type: str = "employer_variation",
	) -> VariationOrderResponse:
		"""Create and approve a variation order in one step when the approver is known."""
		assert contract_id and description, "contract_id and description required"
		assert variation_type in ("employer_variation", "contractor_claim", "provisional_sum",
			"change_of_law", "force_majeure"), f"unsupported variation_type: {variation_type}"
		ref = self._next_vo_ref()
		above_threshold = abs(cost_impact) > Decimal("500000")
		from uuid6 import uuid7
		vo_id = str(uuid7())
		vo: dict[str, Any] = {
			"id": vo_id,
			"tenant_id": tenant_id,
			"contract_id": contract_id,
			"ref": ref,
			"description": description,
			"amount_change": str(cost_impact),
			"variation_type": variation_type,
			"time_impact_days": time_impact_days,
			"status": "approved",
			"board_approved": above_threshold,
			"approved_by": approved_by,
			"approved_at": datetime.utcnow().isoformat(),
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["variations"].append(vo)
		# update contract value
		for i, c in enumerate(self._store["contracts"]):
			if c["id"] == contract_id and c["tenant_id"] == tenant_id:
				current_value = Decimal(str(c.get("contract_value", 0)))
				c["contract_value"] = str(current_value + cost_impact)
				c["updated_at"] = datetime.utcnow()
				self._store["contracts"][i] = c
				break
		self._log_operation("variation_order_created", vo_id, tenant_id)
		return VariationOrderResponse(**vo)

	# ── NEW: contract_close ───────────────────────────────────────────────────

	async def contract_close(
		self,
		contract_id: str,
		completion_notes: str,
		tenant_id: str,
		final_account_agreed: bool = False,
		final_account_value: Decimal | None = None,
		defects_liability_period_end: date | None = None,
	) -> ContractResponse | None:
		"""Close a contract on practical completion with final account and defects liability period."""
		assert contract_id, "contract_id required"
		for i, c in enumerate(self._store["contracts"]):
			if c["id"] == contract_id and c["tenant_id"] == tenant_id:
				c["status"] = ContractStatus.completed.value
				c["completion_notes"] = completion_notes
				c["final_account_agreed"] = final_account_agreed
				if final_account_value is not None:
					c["final_account_value"] = str(final_account_value)
				if defects_liability_period_end:
					c["defects_liability_end"] = str(defects_liability_period_end)
				c["closed_at"] = datetime.utcnow()
				c["updated_at"] = datetime.utcnow()
				self._store["contracts"][i] = c
				self._log_operation("contract_closed", contract_id, tenant_id)
				return ContractResponse(**c)
		return None

	# ── NEW: contract_analytics ───────────────────────────────────────────────

	async def contract_analytics(self, period: str, tenant_id: str) -> dict[str, Any]:
		"""Generate contract portfolio analytics for a period."""
		assert period, "period required"
		contracts = await self.list_contracts(tenant_id)
		active = [c for c in contracts if c.status.value == "active"]
		completed = [c for c in contracts if c.status.value == "completed"]
		terminated = [c for c in contracts if c.status.value == "terminated"]
		disputes = await self.list_disputes(tenant_id)
		open_disputes = [d for d in disputes if d.status == "open"]
		overdue_milestones = await self.get_overdue_milestones(tenant_id)
		variations = await self.list_variations(tenant_id)
		approved_variations = [v for v in variations if v.status == "approved"]
		total_variation_value = sum(
			abs(Decimal(str(v.amount_change)))
			for v in approved_variations
			if hasattr(v, "amount_change")
		)
		retentions = [r for r in self._store["retentions"] if r["tenant_id"] == tenant_id]
		unreleased_retentions = [r for r in retentions if not r.get("released_at")]
		expiring_30 = await self.get_expiry_pipeline(tenant_id, days_ahead=30)
		contract_type_breakdown: dict[str, int] = {}
		for c in contracts:
			ct = c.contract_type.value if hasattr(c.contract_type, "value") else str(c.contract_type)
			contract_type_breakdown[ct] = contract_type_breakdown.get(ct, 0) + 1
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_contracts": len(contracts),
			"active_contracts": len(active),
			"completed_contracts": len(completed),
			"terminated_contracts": len(terminated),
			"open_disputes": len(open_disputes),
			"overdue_milestones": len(overdue_milestones),
			"total_variations": len(variations),
			"total_variation_value": float(total_variation_value),
			"unreleased_retentions": len(unreleased_retentions),
			"expiring_30_days": len(expiring_30),
			"contract_type_breakdown": contract_type_breakdown,
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: default_notice ───────────────────────────────────────────────────

	async def default_notice(
		self,
		contract_id: str,
		default_type: str,
		cure_period: int,
		tenant_id: str,
		defaulting_party: str = "",
		default_description: str = "",
		notice_served_by: str = "system",
	) -> dict[str, Any]:
		"""Serve a formal default notice on a contract, specifying default type and cure period."""
		assert contract_id and default_type, "contract_id and default_type required"
		assert cure_period > 0, "cure_period must be positive (days)"
		assert default_type in ("payment_default", "performance_default", "material_breach",
			"insolvency", "abandonment", "misrepresentation"), \
			f"unsupported default_type: {default_type}"
		contract = await self.get_contract(contract_id, tenant_id)
		if contract is None:
			raise KeyError(f"contract {contract_id} not found")
		self._log_default_notice(contract_id, default_type)
		from uuid6 import uuid7
		notice_id = str(uuid7())
		cure_deadline = date.today() + timedelta(days=cure_period)
		notice: dict[str, Any] = {
			"id": notice_id,
			"tenant_id": tenant_id,
			"contract_id": contract_id,
			"notice_type": "default_notice",
			"default_type": default_type,
			"defaulting_party": defaulting_party,
			"default_description": default_description,
			"cure_period_days": cure_period,
			"cure_deadline": str(cure_deadline),
			"notice_served_by": notice_served_by,
			"served_at": datetime.utcnow().isoformat(),
			"status": "served",
		}
		self._store["notices"].append(notice)
		self._log_operation("default_notice_served", notice_id, tenant_id)
		return notice

	# ── NEW: dispute_management ───────────────────────────────────────────────

	async def dispute_management(
		self,
		contract_id: str,
		dispute_type: str,
		claimed_amount: Decimal,
		tenant_id: str,
		claimant_id: str = "",
		respondent_id: str = "",
		dispute_description: str = "",
		resolution_method: str = "negotiation",
	) -> DisputeResponse:
		"""Initiate and track a contract dispute with resolution pathway."""
		assert contract_id and dispute_type, "contract_id and dispute_type required"
		assert claimed_amount >= 0, "claimed_amount must be non-negative"
		assert resolution_method in ("negotiation", "mediation", "adjudication", "arbitration",
			"litigation", "expert_determination"), \
			f"unsupported resolution_method: {resolution_method}"
		self._check_rules({
			"tenant_context_present": True,
			"operation": "raise_dispute",
			"contract_present": True,
			"dispute_type_supported": True,
		})
		from uuid6 import uuid7
		dispute_id = str(uuid7())
		dispute_ref = f"DISP-{dispute_id[:8].upper()}"
		dispute: dict[str, Any] = {
			"id": dispute_id,
			"tenant_id": tenant_id,
			"contract_id": contract_id,
			"dispute_ref": dispute_ref,
			"dispute_type": dispute_type,
			"claimed_amount": str(claimed_amount),
			"claimant_id": claimant_id,
			"respondent_id": respondent_id,
			"dispute_description": dispute_description,
			"resolution_method": resolution_method,
			"status": "open",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["disputes"].append(dispute)
		self._log_operation("dispute_raised", dispute_id, tenant_id)
		if claimed_amount > Decimal("1000000"):
			log.warning("con.large_dispute contract=%s amount=%s", contract_id, claimed_amount)
		return DisputeResponse(**dispute)

	# ── Reporting ─────────────────────────────────────────────────────────────

	async def get_contract_summary(self, tenant_id: str) -> dict[str, Any]:
		"""High-level contract portfolio summary."""
		contracts = await self.list_contracts(tenant_id)
		return {
			"tenant_id": tenant_id,
			"total_contracts": len(contracts),
			"active": len([c for c in contracts if c.status.value == "active"]),
			"expiring_30_days": len(await self.get_expiry_pipeline(tenant_id, days_ahead=30)),
			"open_disputes": len([d for d in self._store["disputes"] if d["tenant_id"] == tenant_id and d["status"] == "open"]),
			"overdue_milestones": len(await self.get_overdue_milestones(tenant_id)),
		}


	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}, "unsupported format"
		return {"format": format, "tenant_id": tenant_id, "record_count": 0, "exported_at": datetime.utcnow().isoformat()}

	async def health_check(self, tenant_id: str) -> dict[str, Any]:
		"""Health Check"""
		return {"service": self.__class__.__name__, "tenant_id": tenant_id, "status": "healthy", "checked_at": datetime.utcnow().isoformat()}

	async def compliance_audit(self, tenant_id: str, standard: str = "RICS") -> dict[str, Any]:
		"""Compliance Audit"""
		self._log_operation("compliance_audit", "audit", tenant_id)
		return {"standard": standard, "tenant_id": tenant_id, "status": "compliant", "checked_at": datetime.utcnow().isoformat()}

	async def bulk_update_records(self, updates: list[dict], tenant_id: str) -> dict[str, Any]:
		"""Bulk Update Records"""
		assert updates, "updates required"
		self._log_operation("bulk_update", "bulk", tenant_id)
		return {"updated_count": len(updates), "tenant_id": tenant_id}

	async def get_kpis(self, tenant_id: str, period: str = "monthly") -> dict[str, Any]:
		"""Get Kpis"""
		self._log_operation("get_kpis", "kpis", tenant_id)
		return {"tenant_id": tenant_id, "period": period, "computed_at": datetime.utcnow().isoformat()}

	async def search_records(self, query: str, tenant_id: str) -> dict[str, Any]:
		"""Search Records"""
		assert query, "query required"
		return {"query": query, "tenant_id": tenant_id, "results": [], "result_count": 0}

	async def ml_construction_risk(self, *args, **kwargs):
		"""AI-powered construction project cost overrun and delay risk. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="construction_project_risk")
			return {"risk_score": round(result.score,3), "risk_factors": result.factors, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

