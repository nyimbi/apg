"""SACCO Guarantor Management — full async service.

Business rules:
- Savings cover ratio: guarantor free savings >= MIN_SAVINGS_COVER_PCT% of guarantee amount (default 100%).
- Exposure cap: total active guaranteed <= MAX_EXPOSURE_MULTIPLIER × share_capital (default 3×).
  Override per member via set_exposure_limit().
- Defaulter bar: any loan in arrears triggers ineligibility as guarantor.
- Savings freeze: on accept, frozen_amount = guaranteed_amount. Released on loan close/write-off/sub.
- GL posting: on call_guarantee, DR Guarantor Savings / CR Loan Recovery.
- Automatic release: nightly scan for fully repaid loans.
- At-risk: DPD > 30 on the guaranteed loan.
"""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import logging
from copy import deepcopy
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id

_log = logging.getLogger(__name__)

CAPABILITY_ID = "fintech_sacco_gua"

# ── Policy defaults (tunable per SACCO via config) ───────────────────────────
MIN_SAVINGS_COVER_PCT: Decimal = Decimal("100")   # free savings must be >= 100% of amount
MAX_EXPOSURE_MULTIPLIER: Decimal = Decimal("3")   # exposure <= 3× share capital
DEFAULT_MAX_EXPOSURE: Decimal = Decimal("500000") # absolute ceiling if share_capital unknown
AT_RISK_DPD_THRESHOLD: int = 30


class GuarantorService:
	"""Async service for SACCO guarantor management."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		# in-memory stores (replace with DB adapters in production)
		self._requests = WriteThruDict('requests', tenant_id, _store)
		self._guarantees = WriteThruDict('guarantees', tenant_id, _store)
		self._exposure_overrides = WriteThruDict('exposure_overrides', tenant_id, _store)   # member_id → override
		self._gl_entries = WriteThruList('gl_entries', tenant_id, _store)
		self._notices = WriteThruList('notices', tenant_id, _store)
		self._audit = WriteThruList('audit', tenant_id, _store)
		# injected member/loan context (populated by caller or APG composition)
		self._member_savings: dict[str, Decimal] = {}   # member_id → free savings
		self._member_shares: dict[str, Decimal] = {}    # member_id → share capital
		self._member_active: dict[str, bool] = {}       # member_id → is_active
		self._member_defaulter: dict[str, bool] = {}    # member_id → is_defaulter
		self._loan_status: dict[str, str] = {}          # loan_id → status
		self._loan_dpd: dict[str, int] = {}             # loan_id → days past due

	# ── Internal helpers ──────────────────────────────────────────────────────

	def _guard(self, tenant_id: str) -> str:
		value = guard_tenant_id(tenant_id or self.tenant_id)
		return value

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _uid(self, prefix: str) -> str:
		return f"{prefix}-{uuid4().hex[:12]}"

	def _emit(self, tenant_id: str, event: str, record: dict[str, Any]) -> None:
		self._audit.append({
			"tenant_id": tenant_id,
			"event": event,
			"record_id": record.get("id", ""),
			"emitted_at": self._now(),
		})

	def _log_operation(self, op: str, **kw: Any) -> None:
		_log.info("[gua] %s %s", op, " ".join(f"{k}={v}" for k, v in kw.items()))

	def _get_request(self, request_id: str, tenant_id: str) -> dict[str, Any]:
		r = self._requests.get(request_id)
		if not r or r["tenant_id"] != tenant_id:
			raise KeyError(f"guarantee_request_not_found: {request_id}")
		return r

	def _get_guarantee(self, guarantee_id: str, tenant_id: str) -> dict[str, Any]:
		g = self._guarantees.get(guarantee_id)
		if not g or g["tenant_id"] != tenant_id:
			raise KeyError(f"guarantee_not_found: {guarantee_id}")
		return g

	def _free_savings(self, member_id: str) -> Decimal:
		"""Return a member's unencumbered savings (total minus already frozen)."""
		total = self._member_savings.get(member_id, Decimal("0"))
		frozen = sum(
			Decimal(str(g.get("frozen_amount", 0)))
			for g in self._guarantees.values()
			if g.get("guarantor_member_id") == member_id and g.get("status") == "active"
		)
		return max(Decimal("0"), total - frozen)

	def _current_exposure(self, member_id: str, tenant_id: str) -> Decimal:
		return sum(
			Decimal(str(g.get("guaranteed_amount", 0)))
			for g in self._guarantees.values()
			if g["tenant_id"] == tenant_id
			and g.get("guarantor_member_id") == member_id
			and g.get("status") == "active"
		)

	def _max_exposure(self, member_id: str, tenant_id: str) -> Decimal:
		override = self._exposure_overrides.get(f"{tenant_id}:{member_id}")
		if override:
			return Decimal(str(override["limit"]))
		shares = self._member_shares.get(member_id, Decimal("0"))
		if shares > 0:
			return shares * MAX_EXPOSURE_MULTIPLIER
		return DEFAULT_MAX_EXPOSURE

	# ── Seed helpers (for testing / APG composition) ──────────────────────────

	def seed_member(
		self,
		member_id: str,
		savings: Decimal,
		shares: Decimal,
		is_active: bool = True,
		is_defaulter: bool = False,
	) -> None:
		"""Register member financial context (normally sourced from mem/dep/lnd caps)."""
		self._member_savings[member_id] = savings
		self._member_shares[member_id] = shares
		self._member_active[member_id] = is_active
		self._member_defaulter[member_id] = is_defaulter

	def seed_loan(self, loan_id: str, status: str, dpd: int = 0) -> None:
		"""Register loan context (normally sourced from lnd capability)."""
		self._loan_status[loan_id] = status
		self._loan_dpd[loan_id] = dpd

	# ── Eligibility & Exposure ────────────────────────────────────────────────

	async def check_guarantor_eligibility(
		self,
		tenant_id: str,
		member_id: str,
		amount_to_guarantee: Decimal,
	) -> dict[str, Any]:
		"""Evaluate whether a member can guarantee the requested amount."""
		t = self._guard(tenant_id)
		assert amount_to_guarantee > 0, "amount_to_guarantee must be positive"

		reasons: list[str] = []

		if not self._member_active.get(member_id, True):
			reasons.append("member_not_active")

		if self._member_defaulter.get(member_id, False):
			reasons.append("member_is_defaulter")

		free_savings = self._free_savings(member_id)
		required_cover = (amount_to_guarantee * MIN_SAVINGS_COVER_PCT / 100).quantize(Decimal("0.01"), ROUND_HALF_UP)
		if free_savings < required_cover:
			reasons.append(f"insufficient_savings: have {free_savings}, need {required_cover}")

		current_exposure = self._current_exposure(member_id, t)
		max_exp = self._max_exposure(member_id, t)
		if current_exposure + amount_to_guarantee > max_exp:
			reasons.append(f"exposure_limit_exceeded: current={current_exposure} limit={max_exp}")

		eligible = len(reasons) == 0
		headroom = max(Decimal("0"), max_exp - current_exposure)
		cover_ratio = (free_savings / amount_to_guarantee).quantize(Decimal("0.0001"), ROUND_HALF_UP) if amount_to_guarantee else Decimal("0")

		result: dict[str, Any] = {
			"member_id": member_id,
			"tenant_id": t,
			"amount_requested": str(amount_to_guarantee),
			"eligible": eligible,
			"reasons": reasons,
			"free_savings": str(free_savings),
			"current_exposure": str(current_exposure),
			"max_exposure_limit": str(max_exp),
			"headroom": str(headroom),
			"savings_cover_ratio": str(cover_ratio),
			"checked_at": self._now(),
		}
		self._log_operation("eligibility_check", member=member_id, eligible=eligible, amount=amount_to_guarantee)
		return result

	async def get_guarantor_exposure(self, tenant_id: str, member_id: str) -> dict[str, Any]:
		"""Return full exposure snapshot for a member."""
		t = self._guard(tenant_id)
		active_guarantees = [
			deepcopy(g) for g in self._guarantees.values()
			if g["tenant_id"] == t and g.get("guarantor_member_id") == member_id and g.get("status") == "active"
		]
		total_guaranteed = sum(Decimal(str(g["guaranteed_amount"])) for g in active_guarantees)
		frozen_savings = sum(Decimal(str(g["frozen_amount"])) for g in active_guarantees)
		at_risk = sum(
			Decimal(str(g["guaranteed_amount"])) for g in active_guarantees
			if self._loan_dpd.get(g.get("loan_id", ""), 0) > AT_RISK_DPD_THRESHOLD
		)
		free = self._free_savings(member_id)
		max_exp = self._max_exposure(member_id, t)
		available = max(Decimal("0"), min(free, max_exp - total_guaranteed))
		return {
			"member_id": member_id,
			"tenant_id": t,
			"total_guaranteed": str(total_guaranteed),
			"frozen_savings": str(frozen_savings),
			"active_guarantees": active_guarantees,
			"available_to_guarantee": str(available),
			"max_exposure_limit": str(max_exp),
			"at_risk_amount": str(at_risk),
			"computed_at": self._now(),
		}

	async def set_exposure_limit(
		self,
		tenant_id: str,
		member_id: str,
		limit: Decimal,
		set_by: str,
	) -> dict[str, Any]:
		"""Override the default exposure limit for a member."""
		t = self._guard(tenant_id)
		assert limit >= 0, "limit must be non-negative"
		key = f"{t}:{member_id}"
		record: dict[str, Any] = {
			"id": self._uid("exlim"),
			"tenant_id": t,
			"member_id": member_id,
			"limit": str(limit),
			"set_by": set_by,
			"created_at": self._now(),
		}
		self._exposure_overrides[key] = record
		self._emit(t, "exposure_limit_set", record)
		self._log_operation("set_exposure_limit", member=member_id, limit=limit, by=set_by)
		return deepcopy(record)

	# ── Request lifecycle ─────────────────────────────────────────────────────

	async def request_guarantee(
		self,
		tenant_id: str,
		loan_id: str,
		guarantor_member_id: str,
		amount_to_guarantee: Decimal,
		loan_applicant_message: str | None = None,
	) -> dict[str, Any]:
		"""Send a consent request to a prospective guarantor."""
		t = self._guard(tenant_id)
		assert amount_to_guarantee > 0, "amount_to_guarantee must be positive"

		# Guard: check eligibility
		elig = await self.check_guarantor_eligibility(t, guarantor_member_id, amount_to_guarantee)
		if not elig["eligible"]:
			raise ValueError(f"guarantor_ineligible: {'; '.join(elig['reasons'])}")

		req_id = self._uid("greq")
		record: dict[str, Any] = {
			"id": req_id,
			"type": "gua_request",
			"tenant_id": t,
			"loan_id": loan_id,
			"guarantor_member_id": guarantor_member_id,
			"amount_to_guarantee": str(amount_to_guarantee),
			"loan_applicant_message": loan_applicant_message,
			"status": "pending",
			"pin_verified": False,
			"acceptance_notes": None,
			"accepted_at": None,
			"decline_reason": None,
			"declined_at": None,
			"cancelled_by": None,
			"cancel_reason": None,
			"cancelled_at": None,
			"created_at": self._now(),
			"updated_at": self._now(),
		}
		self._requests[req_id] = record
		self._emit(t, "guarantee_requested", record)
		self._log_operation("request_guarantee", loan=loan_id, guarantor=guarantor_member_id, amount=amount_to_guarantee)
		return deepcopy(record)

	async def accept_guarantee(
		self,
		tenant_id: str,
		guarantee_request_id: str,
		guarantor_member_id: str,
		pin_verified: bool,
		acceptance_notes: str | None = None,
	) -> dict[str, Any]:
		"""Record informed consent and freeze guarantor savings."""
		t = self._guard(tenant_id)
		if not pin_verified:
			raise ValueError("pin_verification_required")

		req = self._get_request(guarantee_request_id, t)
		if req["guarantor_member_id"] != guarantor_member_id:
			raise PermissionError("guarantor_identity_mismatch")
		if req["status"] != "pending":
			raise ValueError(f"cannot_accept_request_in_status: {req['status']}")

		amount = Decimal(str(req["amount_to_guarantee"]))
		# Re-check eligibility at acceptance time (savings may have changed)
		elig = await self.check_guarantor_eligibility(t, guarantor_member_id, amount)
		if not elig["eligible"]:
			raise ValueError(f"guarantor_no_longer_eligible: {'; '.join(elig['reasons'])}")

		# Update request
		req.update({
			"status": "accepted",
			"pin_verified": True,
			"acceptance_notes": acceptance_notes,
			"accepted_at": self._now(),
			"updated_at": self._now(),
		})

		# Create active guarantee record with frozen savings
		gua_id = self._uid("gua")
		guarantee: dict[str, Any] = {
			"id": gua_id,
			"type": "gua_active",
			"tenant_id": t,
			"guarantee_request_id": guarantee_request_id,
			"loan_id": req["loan_id"],
			"guarantor_member_id": guarantor_member_id,
			"guaranteed_amount": str(amount),
			"frozen_amount": str(amount),
			"amount_called": str(Decimal("0")),
			"status": "active",
			"release_reason": None,
			"released_at": None,
			"released_by": None,
			"called_at": None,
			"call_reason": None,
			"substituted_by": None,
			"notices_sent": [],
			"created_at": self._now(),
			"updated_at": self._now(),
		}
		self._guarantees[gua_id] = guarantee
		self._emit(t, "guarantee_accepted", guarantee)
		self._log_operation("accept_guarantee", req=guarantee_request_id, guarantor=guarantor_member_id, frozen=amount)
		return deepcopy(guarantee)

	async def decline_guarantee(
		self,
		tenant_id: str,
		guarantee_request_id: str,
		guarantor_member_id: str,
		decline_reason: str,
	) -> dict[str, Any]:
		"""Guarantor refuses; no savings are touched."""
		t = self._guard(tenant_id)
		req = self._get_request(guarantee_request_id, t)
		if req["guarantor_member_id"] != guarantor_member_id:
			raise PermissionError("guarantor_identity_mismatch")
		if req["status"] != "pending":
			raise ValueError(f"cannot_decline_request_in_status: {req['status']}")
		req.update({
			"status": "declined",
			"decline_reason": decline_reason,
			"declined_at": self._now(),
			"updated_at": self._now(),
		})
		self._emit(t, "guarantee_declined", req)
		self._log_operation("decline_guarantee", req=guarantee_request_id, reason=decline_reason)
		return deepcopy(req)

	async def cancel_guarantee_request(
		self,
		tenant_id: str,
		guarantee_request_id: str,
		cancelled_by: str,
		reason: str,
	) -> dict[str, Any]:
		"""Cancel a pending request (loan officer / borrower initiated)."""
		t = self._guard(tenant_id)
		req = self._get_request(guarantee_request_id, t)
		if req["status"] not in {"pending", "declined"}:
			raise ValueError(f"cannot_cancel_request_in_status: {req['status']}")
		req.update({
			"status": "cancelled",
			"cancelled_by": cancelled_by,
			"cancel_reason": reason,
			"cancelled_at": self._now(),
			"updated_at": self._now(),
		})
		self._emit(t, "guarantee_request_cancelled", req)
		self._log_operation("cancel_request", req=guarantee_request_id, by=cancelled_by)
		return deepcopy(req)

	async def get_guarantee_request(self, tenant_id: str, guarantee_request_id: str) -> dict[str, Any]:
		t = self._guard(tenant_id)
		return deepcopy(self._get_request(guarantee_request_id, t))

	async def list_guarantee_requests(
		self,
		tenant_id: str,
		loan_id: str | None = None,
		guarantor_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		t = self._guard(tenant_id)
		items = [deepcopy(r) for r in self._requests.values() if r["tenant_id"] == t]
		if loan_id:
			items = [r for r in items if r["loan_id"] == loan_id]
		if guarantor_id:
			items = [r for r in items if r["guarantor_member_id"] == guarantor_id]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	# ── Active guarantee management ───────────────────────────────────────────

	async def get_loan_guarantors(self, tenant_id: str, loan_id: str) -> list[dict[str, Any]]:
		"""Return all active guarantors for a loan with their amounts."""
		t = self._guard(tenant_id)
		return [
			deepcopy(g) for g in self._guarantees.values()
			if g["tenant_id"] == t and g["loan_id"] == loan_id and g["status"] == "active"
		]

	async def release_guarantee(
		self,
		tenant_id: str,
		guarantee_id: str,
		release_reason: str,
		released_by: str = "system",
	) -> dict[str, Any]:
		"""Unfreeze guarantor savings and close the guarantee obligation."""
		t = self._guard(tenant_id)
		g = self._get_guarantee(guarantee_id, t)
		if g["status"] not in {"active", "accepted"}:
			raise ValueError(f"cannot_release_guarantee_in_status: {g['status']}")
		g.update({
			"status": "released",
			"release_reason": release_reason,
			"released_at": self._now(),
			"released_by": released_by,
			"frozen_amount": str(Decimal("0")),
			"updated_at": self._now(),
		})
		self._emit(t, "guarantee_released", g)
		self._log_operation("release_guarantee", gua=guarantee_id, reason=release_reason)
		# Send release notice
		await self.send_guarantee_notice(t, guarantee_id, "release")
		return deepcopy(g)

	async def substitute_guarantor(
		self,
		tenant_id: str,
		guarantee_id: str,
		new_guarantor_id: str,
		reason: str,
		approved_by: str,
	) -> dict[str, Any]:
		"""Replace guarantor: old released, new guarantee request created & must be accepted."""
		t = self._guard(tenant_id)
		old = self._get_guarantee(guarantee_id, t)
		if old["status"] != "active":
			raise ValueError(f"cannot_substitute_guarantee_in_status: {old['status']}")

		amount = Decimal(str(old["guaranteed_amount"]))
		# Verify new guarantor is eligible
		elig = await self.check_guarantor_eligibility(t, new_guarantor_id, amount)
		if not elig["eligible"]:
			raise ValueError(f"new_guarantor_ineligible: {'; '.join(elig['reasons'])}")

		# Release old guarantee
		old.update({
			"status": "substituted",
			"release_reason": "substitution",
			"released_at": self._now(),
			"released_by": approved_by,
			"substituted_by": new_guarantor_id,
			"frozen_amount": str(Decimal("0")),
			"updated_at": self._now(),
		})
		self._emit(t, "guarantee_substituted", old)

		# Create new guarantee request (caller must call accept_guarantee separately if workflow requires consent)
		new_req = await self.request_guarantee(
			tenant_id=t,
			loan_id=old["loan_id"],
			guarantor_member_id=new_guarantor_id,
			amount_to_guarantee=amount,
			loan_applicant_message=f"Substitution of guarantee {guarantee_id}. Reason: {reason}. Approved by: {approved_by}",
		)
		self._log_operation("substitute_guarantor", old=guarantee_id, new=new_guarantor_id, by=approved_by)
		return {
			"released_guarantee": deepcopy(old),
			"new_request": new_req,
		}

	async def call_guarantee(
		self,
		tenant_id: str,
		guarantee_id: str,
		amount_called: Decimal,
		reason: str,
	) -> dict[str, Any]:
		"""Default recovery: deduct from guarantor frozen savings and post GL."""
		t = self._guard(tenant_id)
		assert amount_called > 0, "amount_called must be positive"
		g = self._get_guarantee(guarantee_id, t)
		if g["status"] != "active":
			raise ValueError(f"cannot_call_guarantee_in_status: {g['status']}")

		frozen = Decimal(str(g["frozen_amount"]))
		if amount_called > frozen:
			raise ValueError(f"call_exceeds_frozen: call={amount_called} frozen={frozen}")

		g.update({
			"status": "called",
			"amount_called": str(Decimal(str(g["amount_called"])) + amount_called),
			"frozen_amount": str(frozen - amount_called),
			"called_at": self._now(),
			"call_reason": reason,
			"updated_at": self._now(),
		})

		# Post GL entry
		gl_id = self._uid("gl")
		gl_entry: dict[str, Any] = {
			"id": gl_id,
			"type": "gua_gl_entry",
			"tenant_id": t,
			"guarantee_id": guarantee_id,
			"loan_id": g["loan_id"],
			"guarantor_member_id": g["guarantor_member_id"],
			"amount": str(amount_called),
			"debit_account": "Guarantor Savings",
			"credit_account": "Loan Recovery",
			"narrative": f"Guarantee call: {reason}",
			"posted_at": self._now(),
		}
		self._gl_entries.append(gl_entry)
		self._emit(t, "guarantee_called", g)
		self._emit(t, "gl_entry_posted", gl_entry)
		self._log_operation("call_guarantee", gua=guarantee_id, amount=amount_called)
		# Send call notice
		await self.send_guarantee_notice(t, guarantee_id, "call_notice")
		return {
			"guarantee": deepcopy(g),
			"gl_entry": gl_entry,
		}

	async def get_called_guarantees(self, tenant_id: str, guarantor_id: str) -> list[dict[str, Any]]:
		"""Return all guarantees where money was taken from this guarantor."""
		t = self._guard(tenant_id)
		return [
			deepcopy(g) for g in self._guarantees.values()
			if g["tenant_id"] == t
			and g.get("guarantor_member_id") == guarantor_id
			and g.get("status") == "called"
		]

	async def get_guarantor_history(self, tenant_id: str, member_id: str) -> dict[str, Any]:
		"""Full historical record: all requests and guarantees for a member."""
		t = self._guard(tenant_id)
		requests = [
			deepcopy(r) for r in self._requests.values()
			if r["tenant_id"] == t and r["guarantor_member_id"] == member_id
		]
		guarantees = [
			deepcopy(g) for g in self._guarantees.values()
			if g["tenant_id"] == t and g.get("guarantor_member_id") == member_id
		]
		exposure = await self.get_guarantor_exposure(t, member_id)
		return {
			"member_id": member_id,
			"tenant_id": t,
			"requests": requests,
			"guarantees": guarantees,
			"current_exposure": exposure,
			"generated_at": self._now(),
		}

	async def get_at_risk_guarantees(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Active guarantees on loans with DPD > 30."""
		t = self._guard(tenant_id)
		return [
			deepcopy(g) for g in self._guarantees.values()
			if g["tenant_id"] == t
			and g.get("status") == "active"
			and self._loan_dpd.get(g.get("loan_id", ""), 0) > AT_RISK_DPD_THRESHOLD
		]

	# ── Notices ───────────────────────────────────────────────────────────────

	async def send_guarantee_notice(
		self,
		tenant_id: str,
		guarantee_id: str,
		notice_type: str,
	) -> dict[str, Any]:
		"""Dispatch a notice (warning / call / release) to guarantor."""
		t = self._guard(tenant_id)
		g = self._get_guarantee(guarantee_id, t)
		notice: dict[str, Any] = {
			"id": self._uid("ntc"),
			"type": "gua_notice",
			"tenant_id": t,
			"guarantee_id": guarantee_id,
			"guarantor_member_id": g["guarantor_member_id"],
			"notice_type": notice_type,
			"sent_at": self._now(),
			"channel": "sms",
			"delivered": False,
		}
		self._notices.append(notice)
		g.setdefault("notices_sent", []).append(notice_type)
		self._emit(t, f"notice_sent_{notice_type}", notice)
		self._log_operation("send_notice", gua=guarantee_id, type=notice_type)
		return notice

	# ── Portfolio & Reporting ─────────────────────────────────────────────────

	async def get_guarantee_portfolio_metrics(self, tenant_id: str) -> dict[str, Any]:
		"""Aggregate portfolio statistics."""
		t = self._guard(tenant_id)
		all_g = [g for g in self._guarantees.values() if g["tenant_id"] == t]
		active = [g for g in all_g if g["status"] == "active"]
		called = [g for g in all_g if g["status"] == "called"]
		released = [g for g in all_g if g["status"] == "released"]
		at_risk = [g for g in active if self._loan_dpd.get(g.get("loan_id", ""), 0) > AT_RISK_DPD_THRESHOLD]

		total_exposure = sum(Decimal(str(g["guaranteed_amount"])) for g in active)
		frozen = sum(Decimal(str(g["frozen_amount"])) for g in active)
		called_total = sum(Decimal(str(g["amount_called"])) for g in called)
		at_risk_exposure = sum(Decimal(str(g["guaranteed_amount"])) for g in at_risk)

		# Loans covered
		loan_ids = {g["loan_id"] for g in active}
		avg_per_loan = (Decimal(str(len(active))) / Decimal(str(len(loan_ids)))) if loan_ids else Decimal("0")

		ever_active = len(active) + len(called) + len(released)
		release_rate = (Decimal(str(len(released))) / Decimal(str(ever_active)) * 100).quantize(Decimal("0.01")) if ever_active else Decimal("0")
		call_rate = (Decimal(str(len(called))) / Decimal(str(ever_active)) * 100).quantize(Decimal("0.01")) if ever_active else Decimal("0")

		return {
			"tenant_id": t,
			"total_active_guarantees": len(active),
			"total_exposure": str(total_exposure),
			"total_frozen_savings": str(frozen),
			"total_called_amount": str(called_total),
			"at_risk_count": len(at_risk),
			"at_risk_exposure": str(at_risk_exposure),
			"avg_guarantees_per_loan": str(avg_per_loan.quantize(Decimal("0.01"))),
			"release_rate_pct": str(release_rate),
			"call_rate_pct": str(call_rate),
			"computed_at": self._now(),
		}

	async def process_automatic_releases(self, tenant_id: str) -> dict[str, Any]:
		"""Nightly: release guarantees for fully repaid or written-off loans."""
		t = self._guard(tenant_id)
		released_count = 0
		released_ids: list[str] = []

		for g in list(self._guarantees.values()):
			if g["tenant_id"] != t or g["status"] != "active":
				continue
			loan_status = self._loan_status.get(g["loan_id"], "")
			if loan_status == "closed":
				await self.release_guarantee(t, g["id"], "loan_repaid", released_by="system")
				released_count += 1
				released_ids.append(g["id"])
			elif loan_status == "written_off":
				await self.release_guarantee(t, g["id"], "loan_written_off", released_by="system")
				released_count += 1
				released_ids.append(g["id"])

		result = {
			"tenant_id": t,
			"released_count": released_count,
			"released_guarantee_ids": released_ids,
			"run_at": self._now(),
		}
		self._log_operation("auto_release_run", count=released_count)
		return result

	# ── Audit & GL ────────────────────────────────────────────────────────────

	async def get_audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		t = self._guard(tenant_id)
		return [deepcopy(e) for e in self._audit if e["tenant_id"] == t]

	async def get_gl_entries(self, tenant_id: str, guarantee_id: str | None = None) -> list[dict[str, Any]]:
		t = self._guard(tenant_id)
		entries = [deepcopy(e) for e in self._gl_entries if e["tenant_id"] == t]
		if guarantee_id:
			entries = [e for e in entries if e["guarantee_id"] == guarantee_id]
		return entries

	# ── Health ────────────────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		active = sum(1 for g in self._guarantees.values() if g.get("status") == "active")
		pending_reqs = sum(1 for r in self._requests.values() if r.get("status") == "pending")
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"active_guarantees": active,
			"pending_requests": pending_reqs,
			"total_gl_entries": len(self._gl_entries),
			"checked_at": self._now(),
		}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_requests', '_guarantees', '_exposure_overrides', '_gl_entries', '_notices', '_audit']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

