"""Results-Based Financing Service — DLIs, result claims, verification, payment triggers."""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

CAPABILITY_ID = "ngo_rbf"

SUPPORTED_PAYMENT_MODELS = {"output_based", "outcome_based", "impact_based", "hybrid"}
SUPPORTED_VERIFICATION_METHODS = {"third_party", "government", "self_report", "independent_audit", "beneficiary_survey"}
SUPPORTED_CONTRACT_STATUSES = {"draft", "active", "suspended", "closed", "cancelled"}


class ResultsBasedFinancingService:
	"""Async service for Results-Based Financing management."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self._contracts = WriteThruDict('contracts', tenant_id, _store)
		self._dlis = WriteThruDict('dlis', tenant_id, _store)
		self._claims = WriteThruDict('claims', tenant_id, _store)
		self._verifications = WriteThruDict('verifications', tenant_id, _store)
		self._payment_triggers = WriteThruDict('payment_triggers', tenant_id, _store)
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)

	# ── helpers ───────────────────────────────────────────────────────────────

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}-{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _tenant(self) -> str:
		if not self.tenant_id:
			raise PermissionError("tenant_context_required")
		return self.tenant_id

	def _emit(self, event_type: str, record_id: str, record_type: str, details: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": self._id("evt"),
			"tenant_id": self._tenant(),
			"event_type": event_type,
			"record_id": record_id,
			"record_type": record_type,
			"details": details or {},
			"emitted_at": self._now(),
		})

	def _guard_contract(self, contract_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		c = self._contracts.get(contract_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"contract_not_found:{contract_id}")
		return c

	def _guard_dli(self, dli_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		d = self._dlis.get(dli_id)
		if not d or d["tenant_id"] != tenant:
			raise KeyError(f"dli_not_found:{dli_id}")
		return d

	def _guard_claim(self, claim_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		cl = self._claims.get(claim_id)
		if not cl or cl["tenant_id"] != tenant:
			raise KeyError(f"claim_not_found:{claim_id}")
		return cl

	# ── health / describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"contract_count": len(self._contracts),
			"active_contracts": sum(1 for c in self._contracts.values() if c["status"] == "active"),
			"pending_claims": sum(1 for cl in self._claims.values() if cl["status"] == "submitted"),
			"pending_verifications": sum(1 for v in self._verifications.values() if v["status"] == "pending"),
			"total_payments_triggered": sum(
				pt["amount"] for pt in self._payment_triggers.values()
				if pt["status"] == "paid"
			),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"domain": "ngo",
			"version": "1.0.0",
			"description": "Result verification, payment triggers, disbursement-linked indicators, third-party verification",
			"payment_models": list(SUPPORTED_PAYMENT_MODELS),
			"verification_methods": list(SUPPORTED_VERIFICATION_METHODS),
			"tenant_id": self.tenant_id,
		}

	async def get_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
		tenant = self._tenant()
		events = [e for e in self._audit_events if e["tenant_id"] == tenant]
		return [deepcopy(e) for e in events[-limit:]]

	# ── contracts ─────────────────────────────────────────────────────────────

	async def list_contracts(self, status: str | None = None, programme_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(c) for c in self._contracts.values() if c["tenant_id"] == tenant]
		if status:
			items = [c for c in items if c["status"] == status]
		if programme_id:
			items = [c for c in items if c["programme_id"] == programme_id]
		return items

	async def get_contract(self, contract_id: str) -> dict[str, Any]:
		return deepcopy(self._guard_contract(contract_id))

	async def create_contract(
		self,
		programme_id: str,
		funder_reference: str,
		title: str,
		total_value: Decimal,
		start_date: str,
		end_date: str,
		description: str = "",
		currency: str = "KES",
		payment_model: str = "output_based",
		contract_manager: str = "",
	) -> dict[str, Any]:
		"""Create an RBF contract."""
		tenant = self._tenant()
		if not title:
			raise ValueError("title_required")
		if not funder_reference:
			raise ValueError("funder_reference_required")
		if payment_model not in SUPPORTED_PAYMENT_MODELS:
			raise ValueError(f"unsupported_payment_model:{payment_model}")
		if total_value <= 0:
			raise ValueError("total_value_must_be_positive")
		record: dict[str, Any] = {
			"id": self._id("rbfc"),
			"type": "ngo_rbf_contract",
			"tenant_id": tenant,
			"programme_id": programme_id,
			"funder_reference": funder_reference,
			"title": title,
			"description": description,
			"total_value": total_value,
			"paid_amount": Decimal("0"),
			"currency": currency,
			"start_date": start_date,
			"end_date": end_date,
			"payment_model": payment_model,
			"contract_manager": contract_manager,
			"status": "draft",
			"created_at": self._now(),
			"updated_at": None,
		}
		self._contracts[record["id"]] = record
		self._emit("rbf_contract_created", record["id"], "ngo_rbf_contract",
				   {"title": title, "total_value": str(total_value), "payment_model": payment_model})
		_log.info("RBF contract created: %s (%s)", record["id"], title)
		return deepcopy(record)

	async def update_contract(self, contract_id: str, **kwargs: Any) -> dict[str, Any]:
		c = self._guard_contract(contract_id)
		allowed = {"title", "description", "end_date", "status", "contract_manager"}
		if "status" in kwargs and kwargs["status"] not in SUPPORTED_CONTRACT_STATUSES:
			raise ValueError(f"invalid_status:{kwargs['status']}")
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				c[k] = v
		c["updated_at"] = self._now()
		self._emit("rbf_contract_updated", contract_id, "ngo_rbf_contract", kwargs)
		return deepcopy(c)

	async def activate_contract(self, contract_id: str, approved_by: str) -> dict[str, Any]:
		"""Activate a draft RBF contract."""
		c = self._guard_contract(contract_id)
		if c["status"] != "draft":
			raise ValueError(f"cannot_activate_{c['status']}_contract")
		if not approved_by:
			raise ValueError("approved_by_required")
		c["status"] = "active"
		c["approved_by"] = approved_by
		c["activated_at"] = self._now()
		c["updated_at"] = self._now()
		self._emit("rbf_contract_activated", contract_id, "ngo_rbf_contract", {"approved_by": approved_by})
		return deepcopy(c)

	async def close_contract(self, contract_id: str, closed_by: str) -> dict[str, Any]:
		"""Close an RBF contract."""
		c = self._guard_contract(contract_id)
		c["status"] = "closed"
		c["closed_by"] = closed_by
		c["closed_at"] = self._now()
		c["updated_at"] = self._now()
		self._emit("rbf_contract_closed", contract_id, "ngo_rbf_contract", {"closed_by": closed_by})
		return deepcopy(c)

	async def delete_contract(self, contract_id: str) -> dict[str, Any]:
		c = self._guard_contract(contract_id)
		if c["status"] not in {"draft", "cancelled"}:
			raise ValueError("only_draft_contracts_may_be_deleted")
		removed = self._contracts.pop(contract_id)
		self._emit("rbf_contract_deleted", contract_id, "ngo_rbf_contract")
		return deepcopy(removed)

	# ── DLIs ──────────────────────────────────────────────────────────────────

	async def list_dlis(self, contract_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(d) for d in self._dlis.values() if d["tenant_id"] == tenant]
		if contract_id:
			items = [d for d in items if d["contract_id"] == contract_id]
		return items

	async def get_dli(self, dli_id: str) -> dict[str, Any]:
		return deepcopy(self._guard_dli(dli_id))

	async def create_dli(
		self,
		contract_id: str,
		name: str,
		target_value: float,
		price_per_unit: Decimal,
		due_date: str,
		description: str = "",
		indicator_code: str = "",
		unit: str = "",
		currency: str = "KES",
		verification_method: str = "third_party",
	) -> dict[str, Any]:
		"""Create a Disbursement-Linked Indicator."""
		self._guard_contract(contract_id)
		if not name:
			raise ValueError("dli_name_required")
		if verification_method not in SUPPORTED_VERIFICATION_METHODS:
			raise ValueError(f"unsupported_verification_method:{verification_method}")
		if target_value <= 0:
			raise ValueError("target_value_must_be_positive")
		if price_per_unit <= 0:
			raise ValueError("price_per_unit_must_be_positive")
		record: dict[str, Any] = {
			"id": self._id("dli"),
			"type": "ngo_dli",
			"tenant_id": self._tenant(),
			"contract_id": contract_id,
			"name": name,
			"description": description,
			"indicator_code": indicator_code,
			"target_value": target_value,
			"achieved_value": 0.0,
			"unit": unit,
			"price_per_unit": price_per_unit,
			"currency": currency,
			"due_date": due_date,
			"verification_method": verification_method,
			"payment_earned": Decimal("0"),
			"status": "active",
			"created_at": self._now(),
		}
		self._dlis[record["id"]] = record
		self._emit("dli_created", record["id"], "ngo_dli",
				   {"contract_id": contract_id, "name": name, "target": target_value})
		return deepcopy(record)

	async def update_dli(self, dli_id: str, **kwargs: Any) -> dict[str, Any]:
		d = self._guard_dli(dli_id)
		allowed = {"name", "description", "target_value", "due_date", "price_per_unit",
				   "verification_method", "status"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				d[k] = v
		self._emit("dli_updated", dli_id, "ngo_dli", kwargs)
		return deepcopy(d)

	async def delete_dli(self, dli_id: str) -> dict[str, Any]:
		d = self._guard_dli(dli_id)
		removed = self._dlis.pop(dli_id)
		self._emit("dli_deleted", dli_id, "ngo_dli")
		return deepcopy(removed)

	# ── result claims ─────────────────────────────────────────────────────────

	async def list_claims(self, contract_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(cl) for cl in self._claims.values() if cl["tenant_id"] == tenant]
		if contract_id:
			items = [cl for cl in items if cl["contract_id"] == contract_id]
		if status:
			items = [cl for cl in items if cl["status"] == status]
		return items

	async def get_claim(self, claim_id: str) -> dict[str, Any]:
		return deepcopy(self._guard_claim(claim_id))

	async def submit_result_claim(
		self,
		contract_id: str,
		dli_id: str,
		claimed_value: float,
		claim_date: str,
		submitted_by: str,
		evidence_references: list[str] | None = None,
		notes: str = "",
	) -> dict[str, Any]:
		"""Submit a result claim against a DLI."""
		self._guard_contract(contract_id)
		dli = self._guard_dli(dli_id)
		if dli["contract_id"] != contract_id:
			raise ValueError("dli_does_not_belong_to_contract")
		if not submitted_by:
			raise ValueError("submitted_by_required")
		if claimed_value <= 0:
			raise ValueError("claimed_value_must_be_positive")
		# compute potential payment based on price per unit
		potential_payment = Decimal(str(claimed_value)) * dli["price_per_unit"]
		record: dict[str, Any] = {
			"id": self._id("clm"),
			"type": "ngo_result_claim",
			"tenant_id": self._tenant(),
			"contract_id": contract_id,
			"dli_id": dli_id,
			"claimed_value": claimed_value,
			"verified_value": 0.0,
			"claim_date": claim_date,
			"submitted_by": submitted_by,
			"evidence_references": evidence_references or [],
			"notes": notes,
			"potential_payment": potential_payment,
			"payment_triggered": Decimal("0"),
			"status": "submitted",
			"created_at": self._now(),
		}
		self._claims[record["id"]] = record
		self._emit("result_claim_submitted", record["id"], "ngo_result_claim",
				   {"contract_id": contract_id, "dli_id": dli_id, "claimed_value": claimed_value})
		_log.info("Result claim submitted: %s for DLI %s, value=%.2f", record["id"], dli_id, claimed_value)
		return deepcopy(record)

	async def update_claim(self, claim_id: str, **kwargs: Any) -> dict[str, Any]:
		cl = self._guard_claim(claim_id)
		allowed = {"notes", "evidence_references"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				cl[k] = v
		self._emit("result_claim_updated", claim_id, "ngo_result_claim", kwargs)
		return deepcopy(cl)

	async def withdraw_claim(self, claim_id: str, reason: str) -> dict[str, Any]:
		"""Withdraw a submitted claim."""
		cl = self._guard_claim(claim_id)
		if cl["status"] not in {"submitted"}:
			raise ValueError(f"cannot_withdraw_{cl['status']}_claim")
		cl["status"] = "withdrawn"
		cl["withdrawal_reason"] = reason
		cl["withdrawn_at"] = self._now()
		self._emit("result_claim_withdrawn", claim_id, "ngo_result_claim", {"reason": reason})
		return deepcopy(cl)

	# ── verifications ─────────────────────────────────────────────────────────

	async def list_verifications(self, claim_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(v) for v in self._verifications.values() if v["tenant_id"] == tenant]
		if claim_id:
			items = [v for v in items if v["claim_id"] == claim_id]
		return items

	async def get_verification(self, verification_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		v = self._verifications.get(verification_id)
		if not v or v["tenant_id"] != tenant:
			raise KeyError(f"verification_not_found:{verification_id}")
		return deepcopy(v)

	async def create_verification(
		self,
		claim_id: str,
		verifier: str,
		verification_date: str,
		verified_value: float = 0.0,
		accepted: bool = True,
		methodology: str = "",
		findings: str = "",
		adjustments: str = "",
	) -> dict[str, Any]:
		"""Record a third-party verification of a result claim."""
		cl = self._guard_claim(claim_id)
		if cl["status"] not in {"submitted"}:
			raise ValueError(f"cannot_verify_{cl['status']}_claim")
		if not verifier:
			raise ValueError("verifier_required")
		record: dict[str, Any] = {
			"id": self._id("ver"),
			"type": "ngo_verification",
			"tenant_id": self._tenant(),
			"claim_id": claim_id,
			"verifier": verifier,
			"verification_date": verification_date,
			"methodology": methodology,
			"verified_value": verified_value,
			"accepted": accepted,
			"findings": findings,
			"adjustments": adjustments,
			"status": "completed",
			"created_at": self._now(),
		}
		self._verifications[record["id"]] = record
		# update claim with verified value
		cl["verified_value"] = verified_value
		cl["status"] = "verified" if accepted else "rejected"
		cl["verified_at"] = self._now()
		# update DLI achieved value
		dli = self._dlis.get(cl["dli_id"])
		if dli and dli["tenant_id"] == self._tenant() and accepted:
			dli["achieved_value"] += verified_value
		self._emit("verification_completed", record["id"], "ngo_verification",
				   {"claim_id": claim_id, "accepted": accepted, "verified_value": verified_value})
		_log.info("Verification completed: %s accepted=%s value=%.2f", record["id"], accepted, verified_value)
		return deepcopy(record)

	async def update_verification(self, verification_id: str, **kwargs: Any) -> dict[str, Any]:
		tenant = self._tenant()
		v = self._verifications.get(verification_id)
		if not v or v["tenant_id"] != tenant:
			raise KeyError(f"verification_not_found:{verification_id}")
		allowed = {"findings", "adjustments", "methodology"}
		for k, val in kwargs.items():
			if k in allowed and val is not None:
				v[k] = val
		self._emit("verification_updated", verification_id, "ngo_verification", kwargs)
		return deepcopy(v)

	# ── payment triggers ──────────────────────────────────────────────────────

	async def list_payment_triggers(self, contract_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(pt) for pt in self._payment_triggers.values() if pt["tenant_id"] == tenant]
		if contract_id:
			items = [pt for pt in items if pt["contract_id"] == contract_id]
		return items

	async def get_payment_trigger(self, trigger_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		pt = self._payment_triggers.get(trigger_id)
		if not pt or pt["tenant_id"] != tenant:
			raise KeyError(f"payment_trigger_not_found:{trigger_id}")
		return deepcopy(pt)

	async def trigger_payment(
		self,
		contract_id: str,
		claim_id: str,
		verification_id: str,
		amount: Decimal,
		payment_date: str,
		approved_by: str,
		reference: str,
		currency: str = "KES",
		notes: str = "",
	) -> dict[str, Any]:
		"""Trigger a payment based on verified results."""
		contract = self._guard_contract(contract_id)
		cl = self._guard_claim(claim_id)
		if cl["status"] != "verified":
			raise ValueError(f"payment_requires_verified_claim:current_status={cl['status']}")
		if not approved_by:
			raise ValueError("approved_by_required")
		if amount <= 0:
			raise ValueError("amount_must_be_positive")
		remaining = contract["total_value"] - contract["paid_amount"]
		if amount > remaining:
			raise ValueError(f"payment_exceeds_remaining_contract_value:{remaining}")
		record: dict[str, Any] = {
			"id": self._id("pmt"),
			"type": "ngo_payment_trigger",
			"tenant_id": self._tenant(),
			"contract_id": contract_id,
			"claim_id": claim_id,
			"verification_id": verification_id,
			"amount": amount,
			"currency": currency,
			"payment_date": payment_date,
			"approved_by": approved_by,
			"reference": reference,
			"notes": notes,
			"status": "pending",
			"created_at": self._now(),
		}
		self._payment_triggers[record["id"]] = record
		cl["payment_triggered"] = amount
		cl["status"] = "payment_pending"
		self._emit("payment_triggered", record["id"], "ngo_payment_trigger",
				   {"contract_id": contract_id, "amount": str(amount), "approved_by": approved_by})
		_log.info("Payment triggered: %s amount=%s for claim %s", record["id"], amount, claim_id)
		return deepcopy(record)

	async def confirm_payment(self, trigger_id: str, confirmed_by: str) -> dict[str, Any]:
		"""Confirm a payment trigger as paid."""
		tenant = self._tenant()
		pt = self._payment_triggers.get(trigger_id)
		if not pt or pt["tenant_id"] != tenant:
			raise KeyError(f"payment_trigger_not_found:{trigger_id}")
		if pt["status"] != "pending":
			raise ValueError(f"cannot_confirm_{pt['status']}_payment")
		pt["status"] = "paid"
		pt["confirmed_by"] = confirmed_by
		pt["confirmed_at"] = self._now()
		# update contract paid amount
		contract = self._contracts.get(pt["contract_id"])
		if contract and contract["tenant_id"] == tenant:
			contract["paid_amount"] += pt["amount"]
		# update DLI payment earned
		claim = self._claims.get(pt["claim_id"])
		if claim:
			dli = self._dlis.get(claim["dli_id"])
			if dli and dli["tenant_id"] == tenant:
				dli["payment_earned"] += pt["amount"]
		# update claim status
		if claim:
			claim["status"] = "paid"
		self._emit("payment_confirmed", trigger_id, "ngo_payment_trigger", {"confirmed_by": confirmed_by})
		return deepcopy(pt)

	async def cancel_payment_trigger(self, trigger_id: str, reason: str) -> dict[str, Any]:
		"""Cancel a pending payment trigger."""
		tenant = self._tenant()
		pt = self._payment_triggers.get(trigger_id)
		if not pt or pt["tenant_id"] != tenant:
			raise KeyError(f"payment_trigger_not_found:{trigger_id}")
		if pt["status"] != "pending":
			raise ValueError(f"cannot_cancel_{pt['status']}_payment")
		pt["status"] = "cancelled"
		pt["cancellation_reason"] = reason
		pt["cancelled_at"] = self._now()
		self._emit("payment_cancelled", trigger_id, "ngo_payment_trigger", {"reason": reason})
		return deepcopy(pt)

	# ── analytics ─────────────────────────────────────────────────────────────

	async def contract_performance_summary(self, contract_id: str) -> dict[str, Any]:
		"""Return performance summary for an RBF contract."""
		contract = self._guard_contract(contract_id)
		tenant = self._tenant()
		dlis = [d for d in self._dlis.values() if d["contract_id"] == contract_id]
		claims = [cl for cl in self._claims.values() if cl["contract_id"] == contract_id and cl["tenant_id"] == tenant]
		payments = [pt for pt in self._payment_triggers.values() if pt["contract_id"] == contract_id and pt["status"] == "paid"]
		total_earned = sum(d["payment_earned"] for d in dlis)
		return {
			"contract_id": contract_id,
			"title": contract["title"],
			"status": contract["status"],
			"total_value": contract["total_value"],
			"paid_amount": contract["paid_amount"],
			"payment_pct": round(float(contract["paid_amount"] / contract["total_value"] * 100), 2) if contract["total_value"] else 0.0,
			"dli_count": len(dlis),
			"total_claims": len(claims),
			"verified_claims": len([cl for cl in claims if cl["status"] in {"verified", "paid"}]),
			"paid_payments": len(payments),
			"total_earned": total_earned,
			"generated_at": self._now(),
		}

	async def portfolio_rbf_summary(self) -> dict[str, Any]:
		"""Portfolio-level RBF summary across all contracts."""
		tenant = self._tenant()
		contracts = [c for c in self._contracts.values() if c["tenant_id"] == tenant]
		active = [c for c in contracts if c["status"] == "active"]
		total_value = sum(c["total_value"] for c in contracts)
		total_paid = sum(c["paid_amount"] for c in contracts)
		return {
			"tenant_id": tenant,
			"total_contracts": len(contracts),
			"active_contracts": len(active),
			"total_contract_value": total_value,
			"total_paid": total_paid,
			"payment_pct": round(float(total_paid / total_value * 100), 2) if total_value else 0.0,
			"pending_claims": sum(1 for cl in self._claims.values() if cl["tenant_id"] == tenant and cl["status"] == "submitted"),
			"pending_payments": sum(1 for pt in self._payment_triggers.values() if pt["tenant_id"] == tenant and pt["status"] == "pending"),
			"generated_at": self._now(),
		}

	async def dli_achievement_report(self, contract_id: str) -> list[dict[str, Any]]:
		"""Return DLI achievement status for all DLIs in a contract."""
		self._guard_contract(contract_id)
		dlis = [d for d in self._dlis.values() if d["contract_id"] == contract_id]
		return [
			{
				"dli_id": d["id"],
				"name": d["name"],
				"target_value": d["target_value"],
				"achieved_value": d["achieved_value"],
				"achievement_pct": round(d["achieved_value"] / d["target_value"] * 100, 2) if d["target_value"] else 0.0,
				"payment_earned": d["payment_earned"],
				"max_payment": d["price_per_unit"] * Decimal(str(d["target_value"])),
				"due_date": d["due_date"],
				"status": d["status"],
			}
			for d in dlis
		]

	async def bulk_create_dlis(self, contract_id: str, dlis: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-create DLIs for a contract."""
		tasks = [
			self.create_dli(
				contract_id=contract_id,
				name=d["name"],
				target_value=float(d["target_value"]),
				price_per_unit=Decimal(str(d["price_per_unit"])),
				due_date=d["due_date"],
				description=d.get("description", ""),
				indicator_code=d.get("indicator_code", ""),
				unit=d.get("unit", ""),
				currency=d.get("currency", "KES"),
				verification_method=d.get("verification_method", "third_party"),
			)
			for d in dlis
		]
		outcomes = await asyncio.gather(*tasks, return_exceptions=True)
		results, errors = [], []
		for dli_input, outcome in zip(dlis, outcomes):
			if isinstance(outcome, Exception):
				errors.append({"input": dli_input, "error": str(outcome)})
			else:
				results.append(outcome)
		return {"created": len(results), "failed": len(errors), "dlis": results, "errors": errors}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_contracts', '_dlis', '_claims', '_verifications', '_payment_triggers', '_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

