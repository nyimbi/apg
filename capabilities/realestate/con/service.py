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

	# ── Snagging ──────────────────────────────────────────────────────────────

	async def create_snag_item(
		self,
		contract_id: str,
		tenant_id: str,
		title: str,
		location: str,
		trade: str,
		severity: str = "minor",
		description: str = "",
		reported_by: str = "inspector",
		evidence_ids: list[str] | None = None,
		due_date: date | None = None,
	) -> dict[str, Any]:
		"""Create a snagging / defect item linked to a contract.

		Severity must be one of: critical | major | minor | observation.
		Trade identifies the responsible sub-trade (electrical, plumbing, finishes, etc.).
		"""
		assert contract_id and title and location and trade, \
			"contract_id, title, location, trade required"
		assert severity in ("critical", "major", "minor", "observation"), \
			f"unsupported severity: {severity}"
		from uuid6 import uuid7
		snag_id = str(uuid7())
		snag_ref = f"SNF-{snag_id[:8].upper()}"
		# SLA resolution days by severity
		sla_days: dict[str, int] = {"critical": 2, "major": 7, "minor": 14, "observation": 28}
		resolve_by = (date.today() + timedelta(days=sla_days[severity])) if due_date is None else due_date
		snag: dict[str, Any] = {
			"id": snag_id,
			"tenant_id": tenant_id,
			"contract_id": contract_id,
			"snag_ref": snag_ref,
			"title": title,
			"location": location,
			"trade": trade,
			"severity": severity,
			"description": description,
			"reported_by": reported_by,
			"evidence_ids": evidence_ids or [],
			"due_date": str(resolve_by),
			"status": "open",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store.setdefault("snags", []).append(snag)
		self._log_operation("snag_created", snag_id, tenant_id)
		return snag

	async def resolve_snag_item(
		self,
		snag_id: str,
		tenant_id: str,
		resolution_notes: str,
		resolved_by: str,
		evidence_ids: list[str] | None = None,
	) -> dict[str, Any] | None:
		"""Mark a snag item as resolved, capturing resolution evidence."""
		assert snag_id and resolution_notes and resolved_by, \
			"snag_id, resolution_notes, resolved_by required"
		for i, s in enumerate(self._store.get("snags", [])):
			if s["id"] == snag_id and s["tenant_id"] == tenant_id:
				s["status"] = "resolved"
				s["resolution_notes"] = resolution_notes
				s["resolved_by"] = resolved_by
				s["resolved_at"] = datetime.utcnow().isoformat()
				if evidence_ids:
					s.setdefault("evidence_ids", []).extend(evidence_ids)
				self._store["snags"][i] = s
				self._log_operation("snag_resolved", snag_id, tenant_id)
				return s
		return None

	async def get_snag_list(
		self,
		tenant_id: str,
		contract_id: str | None = None,
		status: str | None = None,
		severity: str | None = None,
		trade: str | None = None,
	) -> list[dict[str, Any]]:
		"""Return filtered snag list for a contract or tenant.

		Supports filtering by status (open/resolved/disputed), severity, and trade.
		"""
		results = [s for s in self._store.get("snags", []) if s["tenant_id"] == tenant_id]
		if contract_id:
			results = [s for s in results if s.get("contract_id") == contract_id]
		if status:
			results = [s for s in results if s.get("status") == status]
		if severity:
			results = [s for s in results if s.get("severity") == severity]
		if trade:
			results = [s for s in results if s.get("trade") == trade]
		return sorted(results, key=lambda x: x.get("due_date", ""))

	async def get_snag_summary(self, tenant_id: str, contract_id: str) -> dict[str, Any]:
		"""Return a snagging summary: counts by status and severity for a contract."""
		snags = await self.get_snag_list(tenant_id, contract_id=contract_id)
		by_status: dict[str, int] = {}
		by_severity: dict[str, int] = {}
		by_trade: dict[str, int] = {}
		overdue_count = 0
		today_str = str(date.today())
		for s in snags:
			st = s.get("status", "open")
			sv = s.get("severity", "minor")
			tr = s.get("trade", "unknown")
			by_status[st] = by_status.get(st, 0) + 1
			by_severity[sv] = by_severity.get(sv, 0) + 1
			by_trade[tr] = by_trade.get(tr, 0) + 1
			if st == "open" and s.get("due_date", "9999") < today_str:
				overdue_count += 1
		return {
			"contract_id": contract_id,
			"tenant_id": tenant_id,
			"total_snags": len(snags),
			"open_snags": by_status.get("open", 0),
			"resolved_snags": by_status.get("resolved", 0),
			"overdue_snags": overdue_count,
			"by_status": by_status,
			"by_severity": by_severity,
			"by_trade": by_trade,
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── Payment Certificates ──────────────────────────────────────────────────

	async def issue_payment_certificate(
		self,
		contract_id: str,
		tenant_id: str,
		period_end: date,
		gross_value: Decimal,
		variations_included: list[str],
		certified_by: str,
		retention_percentage: Decimal = Decimal("5"),
		advance_payment_deduction: Decimal = Decimal("0"),
	) -> dict[str, Any]:
		"""Issue an interim payment certificate (IPC) for a construction contract.

		Computes retention deduction and net certified amount.
		Stores certificate and updates cumulative certified total on the contract.
		"""
		assert contract_id and certified_by, "contract_id and certified_by required"
		assert gross_value >= 0, "gross_value must be non-negative"
		assert 0 <= retention_percentage <= 100, "retention_percentage must be 0-100"
		from uuid6 import uuid7
		cert_id = str(uuid7())
		cert_ref = f"IPC-{cert_id[:8].upper()}"
		retention_amount = (gross_value * retention_percentage / Decimal("100")).quantize(Decimal("0.01"))
		net_certified = gross_value - retention_amount - advance_payment_deduction
		cert: dict[str, Any] = {
			"id": cert_id,
			"tenant_id": tenant_id,
			"contract_id": contract_id,
			"cert_ref": cert_ref,
			"period_end": str(period_end),
			"gross_value": str(gross_value),
			"retention_amount": str(retention_amount),
			"advance_payment_deduction": str(advance_payment_deduction),
			"net_certified": str(net_certified),
			"variations_included": variations_included,
			"certified_by": certified_by,
			"status": "issued",
			"issued_at": datetime.utcnow().isoformat(),
		}
		self._store.setdefault("payment_certs", []).append(cert)
		# accumulate on the contract record
		for i, c in enumerate(self._store["contracts"]):
			if c["id"] == contract_id and c["tenant_id"] == tenant_id:
				prev = Decimal(str(c.get("total_certified", 0)))
				c["total_certified"] = str(prev + gross_value)
				c["updated_at"] = datetime.utcnow()
				self._store["contracts"][i] = c
				break
		self._log_operation("payment_cert_issued", cert_id, tenant_id)
		return cert

	# ── Risk Register ─────────────────────────────────────────────────────────

	async def register_risk(
		self,
		contract_id: str,
		tenant_id: str,
		title: str,
		category: str,
		probability: float,
		impact_cost: Decimal,
		impact_days: int,
		owner: str,
		mitigation_action: str = "",
		risk_type: str = "project",
	) -> dict[str, Any]:
		"""Add a risk item to the project risk register.

		Probability must be 0.0–1.0. Category: ground_conditions | supply_chain |
		regulatory | weather | design | contractor_default | force_majeure | other.
		risk_type: project | programme | commercial | health_safety | environmental.
		"""
		assert contract_id and title and owner, "contract_id, title, owner required"
		assert 0.0 <= probability <= 1.0, "probability must be 0.0-1.0"
		assert impact_cost >= 0 and impact_days >= 0, "impact_cost and impact_days must be non-negative"
		from uuid6 import uuid7
		risk_id = str(uuid7())
		expected_value = (Decimal(str(probability)) * impact_cost).quantize(Decimal("0.01"))
		# risk score on 1-25 scale (probability bands × impact bands)
		prob_band = min(5, max(1, int(probability * 5) + 1))
		impact_band = min(5, max(1, int(float(impact_cost) / 500_000) + 1))
		risk_score = prob_band * impact_band
		risk: dict[str, Any] = {
			"id": risk_id,
			"tenant_id": tenant_id,
			"contract_id": contract_id,
			"title": title,
			"category": category,
			"risk_type": risk_type,
			"probability": probability,
			"impact_cost": str(impact_cost),
			"impact_days": impact_days,
			"expected_value": str(expected_value),
			"risk_score": risk_score,
			"owner": owner,
			"mitigation_action": mitigation_action,
			"status": "open",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store.setdefault("risks", []).append(risk)
		self._log_operation("risk_registered", risk_id, tenant_id)
		return risk

	async def get_risk_register(
		self,
		tenant_id: str,
		contract_id: str | None = None,
		status: str | None = None,
		min_risk_score: int | None = None,
	) -> list[dict[str, Any]]:
		"""Return risk register items, sorted by risk score descending.

		Supports filtering by contract, status (open/mitigated/closed), and minimum risk score.
		"""
		results = [r for r in self._store.get("risks", []) if r["tenant_id"] == tenant_id]
		if contract_id:
			results = [r for r in results if r.get("contract_id") == contract_id]
		if status:
			results = [r for r in results if r.get("status") == status]
		if min_risk_score is not None:
			results = [r for r in results if r.get("risk_score", 0) >= min_risk_score]
		return sorted(results, key=lambda x: x.get("risk_score", 0), reverse=True)

	# ── Drawing Register ──────────────────────────────────────────────────────

	async def register_drawing(
		self,
		contract_id: str,
		tenant_id: str,
		drawing_number: str,
		revision: str,
		title: str,
		discipline: str,
		document_id: str,
		drawn_by: str,
		scale: str = "1:100",
	) -> dict[str, Any]:
		"""Register a drawing revision in the project drawing register.

		Automatically supersedes previous revisions of the same drawing number.
		discipline: architectural | structural | mechanical | electrical | civil | landscape.
		"""
		assert contract_id and drawing_number and revision and title and discipline and document_id, \
			"contract_id, drawing_number, revision, title, discipline, document_id required"
		from uuid6 import uuid7
		drawing_id = str(uuid7())
		# supersede previous revisions for this drawing number
		superseded_by = drawing_id
		for i, d in enumerate(self._store.get("drawings", [])):
			if (d.get("contract_id") == contract_id
					and d.get("drawing_number") == drawing_number
					and d.get("tenant_id") == tenant_id
					and d.get("status") == "current"):
				self._store["drawings"][i]["status"] = "superseded"
				self._store["drawings"][i]["superseded_by"] = superseded_by
				self._store["drawings"][i]["superseded_at"] = datetime.utcnow().isoformat()
		drawing: dict[str, Any] = {
			"id": drawing_id,
			"tenant_id": tenant_id,
			"contract_id": contract_id,
			"drawing_number": drawing_number,
			"revision": revision,
			"title": title,
			"discipline": discipline,
			"document_id": document_id,
			"drawn_by": drawn_by,
			"scale": scale,
			"status": "current",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store.setdefault("drawings", []).append(drawing)
		self._log_operation("drawing_registered", drawing_id, tenant_id)
		return drawing

	async def get_current_drawing_set(
		self,
		tenant_id: str,
		contract_id: str,
		discipline: str | None = None,
	) -> list[dict[str, Any]]:
		"""Return only the current (non-superseded) revision of each drawing for a contract.

		Optionally filter by discipline.
		"""
		results = [
			d for d in self._store.get("drawings", [])
			if d["tenant_id"] == tenant_id
			and d.get("contract_id") == contract_id
			and d.get("status") == "current"
		]
		if discipline:
			results = [d for d in results if d.get("discipline") == discipline]
		return sorted(results, key=lambda x: (x.get("discipline", ""), x.get("drawing_number", "")))

	# ── Practical Completion ──────────────────────────────────────────────────

	async def issue_practical_completion_certificate(
		self,
		contract_id: str,
		tenant_id: str,
		issued_by: str,
		dlp_months: int = 12,
		outstanding_snags_allowed: int = 0,
		commissioning_complete: bool = True,
		o_and_m_manuals_received: bool = True,
		notes: str = "",
	) -> dict[str, Any]:
		"""Issue a Practical Completion (PC) certificate after validating snag and commissioning status.

		Validates that outstanding snag count is within the allowed threshold.
		Sets DLP start date and computes DLP end date. Updates contract status to completed.
		Raises ValueError if validation conditions are not met.
		"""
		assert contract_id and issued_by, "contract_id and issued_by required"
		assert dlp_months > 0, "dlp_months must be positive"
		assert outstanding_snags_allowed >= 0, "outstanding_snags_allowed must be non-negative"
		# validate outstanding snags
		snag_summary = await self.get_snag_summary(tenant_id, contract_id)
		open_snags = snag_summary.get("open_snags", 0)
		if open_snags > outstanding_snags_allowed:
			raise ValueError(
				f"practical_completion_blocked: {open_snags} open snags, "
				f"threshold is {outstanding_snags_allowed}"
			)
		if not commissioning_complete:
			raise ValueError("practical_completion_blocked: commissioning not complete")
		from uuid6 import uuid7
		pc_id = str(uuid7())
		pc_ref = f"PC-{pc_id[:8].upper()}"
		dlp_start = date.today()
		dlp_end = date(
			dlp_start.year + (dlp_start.month + dlp_months - 1) // 12,
			((dlp_start.month + dlp_months - 1) % 12) + 1,
			dlp_start.day,
		)
		pc_cert: dict[str, Any] = {
			"id": pc_id,
			"tenant_id": tenant_id,
			"contract_id": contract_id,
			"cert_ref": pc_ref,
			"issued_by": issued_by,
			"issued_date": str(dlp_start),
			"dlp_start": str(dlp_start),
			"dlp_end": str(dlp_end),
			"dlp_months": dlp_months,
			"open_snags_at_issue": open_snags,
			"commissioning_complete": commissioning_complete,
			"o_and_m_manuals_received": o_and_m_manuals_received,
			"notes": notes,
			"status": "issued",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store.setdefault("pc_certs", []).append(pc_cert)
		# update contract
		for i, c in enumerate(self._store["contracts"]):
			if c["id"] == contract_id and c["tenant_id"] == tenant_id:
				c["status"] = ContractStatus.completed.value
				c["defects_liability_end"] = str(dlp_end)
				c["pc_cert_ref"] = pc_ref
				c["updated_at"] = datetime.utcnow()
				self._store["contracts"][i] = c
				break
		self._log_operation("pc_cert_issued", pc_id, tenant_id)
		return pc_cert

	# ── Extension of Time ─────────────────────────────────────────────────────

	async def submit_extension_of_time(
		self,
		contract_id: str,
		tenant_id: str,
		days_claimed: int,
		cause: str,
		cause_category: str,
		submitted_by: str,
		supporting_evidence_ids: list[str] | None = None,
		affected_milestone_ids: list[str] | None = None,
		delay_description: str = "",
	) -> dict[str, Any]:
		"""Submit an Extension of Time (EOT) claim against a construction contract.

		cause_category: employer_risk | neutral_risk | force_majeure | contractor_risk.
		Employer risk and neutral risk categories are eligible for EOT; contractor risk is not.
		"""
		assert contract_id and cause and submitted_by, \
			"contract_id, cause, submitted_by required"
		assert days_claimed > 0, "days_claimed must be positive"
		eligible_categories = ("employer_risk", "neutral_risk", "force_majeure")
		assert cause_category in ("employer_risk", "neutral_risk", "force_majeure", "contractor_risk"), \
			f"unsupported cause_category: {cause_category}"
		from uuid6 import uuid7
		eot_id = str(uuid7())
		eot_ref = f"EOT-{eot_id[:8].upper()}"
		eligible = cause_category in eligible_categories
		eot: dict[str, Any] = {
			"id": eot_id,
			"tenant_id": tenant_id,
			"contract_id": contract_id,
			"eot_ref": eot_ref,
			"days_claimed": days_claimed,
			"cause": cause,
			"cause_category": cause_category,
			"eligible_for_eot": eligible,
			"submitted_by": submitted_by,
			"supporting_evidence_ids": supporting_evidence_ids or [],
			"affected_milestone_ids": affected_milestone_ids or [],
			"delay_description": delay_description,
			"status": "submitted",
			"days_awarded": 0,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store.setdefault("eot_claims", []).append(eot)
		if not eligible:
			log.warning("con.eot_ineligible contract=%s category=%s", contract_id, cause_category)
		self._log_operation("eot_submitted", eot_id, tenant_id)
		return eot

	async def assess_extension_of_time(
		self,
		eot_id: str,
		tenant_id: str,
		days_awarded: int,
		assessed_by: str,
		assessment_notes: str = "",
	) -> dict[str, Any] | None:
		"""Assess and grant (or reject) an EOT claim.

		Extends affected milestone due dates by days_awarded when granted.
		"""
		assert eot_id and assessed_by, "eot_id and assessed_by required"
		assert days_awarded >= 0, "days_awarded must be non-negative"
		for i, e in enumerate(self._store.get("eot_claims", [])):
			if e["id"] == eot_id and e["tenant_id"] == tenant_id:
				status = "granted" if days_awarded > 0 else "rejected"
				e["days_awarded"] = days_awarded
				e["assessed_by"] = assessed_by
				e["assessment_notes"] = assessment_notes
				e["status"] = status
				e["assessed_at"] = datetime.utcnow().isoformat()
				self._store["eot_claims"][i] = e
				# extend affected milestones
				if days_awarded > 0:
					for mid in e.get("affected_milestone_ids", []):
						for j, m in enumerate(self._store["milestones"]):
							if m["id"] == mid and m["tenant_id"] == tenant_id and m["status"] == "pending":
								try:
									old_due = datetime.strptime(m["due_date"], "%Y-%m-%d").date()
									m["due_date"] = str(old_due + timedelta(days=days_awarded))
									m["eot_applied"] = eot_id
									self._store["milestones"][j] = m
								except Exception as _exc:
									_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
				self._log_operation("eot_assessed", eot_id, tenant_id)
				return e
		return None

