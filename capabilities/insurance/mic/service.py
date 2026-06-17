"""Micro-Insurance Platform Service (ins_mic).

Mobile-first product design, USSD enrolment, airtime premium deduction, instant payout via mobile money.
"""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import logging
from copy import deepcopy
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

ENROLMENT_CHANNELS = {"ussd", "sms", "whatsapp", "app", "agent_tablet", "stk_push"}
PAYMENT_METHODS = {"airtime", "mpesa", "airtel_money", "t_kash", "equitel"}
MOBILE_OPERATORS = {"safaricom", "airtel", "telkom", "faiba"}
PRODUCT_TYPES = {"life", "hospital", "accident", "crop", "livestock", "domestic"}
CLAIM_AUTO_PAY_THRESHOLD = Decimal("10000")


class MicroInsurancePlatformService:
	"""In-memory executable service for the Micro-Insurance Platform."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.products: dict[str, dict[str, Any]] = {}
		self.enrolments: dict[str, dict[str, Any]] = {}
		self.airtime_deductions: dict[str, dict[str, Any]] = {}
		self.mobile_payouts: dict[str, dict[str, Any]] = {}
		self.ussd_sessions: dict[str, dict[str, Any]] = {}
		self.claims: dict[str, dict[str, Any]] = {}
		self.renewals: dict[str, dict[str, Any]] = {}
		self._policy_seq: int = 0
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str) -> str:
		return f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _policy_number(self, tenant: str, product_code: str) -> str:
		self._policy_seq += 1
		return f"MIC/{product_code.upper()[:3]}/{self._policy_seq:07d}"

	def _emit(self, tenant_id: str, event_type: str, entity_id: str, entity_type: str, details: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": self._record_id("audit"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"entity_id": entity_id,
			"entity_type": entity_type,
			"details": details or {},
			"created_at": self._now(),
		})

	def _get_enrolment_by_policy(self, policy_number: str, tenant: str) -> dict[str, Any]:
		enr = next((e for e in self.enrolments.values() if e["policy_number"] == policy_number and e["tenant_id"] == tenant), None)
		if not enr:
			raise KeyError(f"enrolment_not_found:{policy_number}")
		return enr

	# ── Product Design ────────────────────────────────────────────────────────

	async def create_product(
		self,
		tenant_id: str,
		product_code: str,
		product_name: str,
		product_type: str,
		sum_insured: Decimal,
		premium: Decimal,
		coverage_days: int,
		ussd_menu_code: str,
		airtime_deduction: bool = False,
		mobile_money_payout: bool = True,
		currency: str = "KES",
		description: str = "",
	) -> dict[str, Any]:
		"""Create a micro-insurance product."""
		tenant = self._tenant(tenant_id)
		if product_type not in PRODUCT_TYPES:
			raise ValueError(f"unsupported_product_type:{product_type}")
		if any(p["product_code"] == product_code and p["tenant_id"] == tenant for p in self.products.values()):
			raise ValueError(f"product_code_duplicate:{product_code}")
		if coverage_days <= 0:
			raise ValueError("coverage_days_must_be_positive")
		record: dict[str, Any] = {
			"id": self._record_id("prod"),
			"type": "mic_product",
			"product_code": product_code,
			"product_name": product_name,
			"product_type": product_type,
			"sum_insured": Decimal(str(sum_insured)),
			"premium": Decimal(str(premium)),
			"coverage_days": coverage_days,
			"ussd_menu_code": ussd_menu_code,
			"airtime_deduction": airtime_deduction,
			"mobile_money_payout": mobile_money_payout,
			"currency": currency,
			"description": description,
			"status": "active",
			"enrolment_count": 0,
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.products[record["id"]] = record
		self._emit(tenant, "mic_product_created", record["id"], "mic_product", {"product_code": product_code})
		_log.info("Micro-insurance product created: %s tenant=%s", product_code, tenant)
		return deepcopy(record)

	async def get_product(self, tenant_id: str, product_id: str) -> dict[str, Any]:
		"""Retrieve a product."""
		tenant = self._tenant(tenant_id)
		prod = self.products.get(product_id)
		if not prod or prod["tenant_id"] != tenant:
			raise KeyError(f"product_not_found:{product_id}")
		return deepcopy(prod)

	async def list_products(self, tenant_id: str, product_type: str | None = None, active_only: bool = True) -> list[dict[str, Any]]:
		"""List micro-insurance products."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(p) for p in self.products.values() if p["tenant_id"] == tenant]
		if active_only:
			items = [p for p in items if p["status"] == "active"]
		if product_type:
			items = [p for p in items if p["product_type"] == product_type]
		return items

	async def update_product(self, tenant_id: str, product_id: str, updates: dict[str, Any]) -> dict[str, Any]:
		"""Update product fields."""
		tenant = self._tenant(tenant_id)
		prod = self.products.get(product_id)
		if not prod or prod["tenant_id"] != tenant:
			raise KeyError(f"product_not_found:{product_id}")
		allowed = {"product_name", "premium", "description", "status", "airtime_deduction"}
		for k, v in updates.items():
			if k in allowed:
				prod[k] = v
		prod["updated_at"] = self._now()
		self._emit(tenant, "mic_product_updated", product_id, "mic_product", {})
		return deepcopy(prod)

	async def delete_product(self, tenant_id: str, product_id: str) -> dict[str, Any]:
		"""Deactivate a product."""
		tenant = self._tenant(tenant_id)
		prod = self.products.get(product_id)
		if not prod or prod["tenant_id"] != tenant:
			raise KeyError(f"product_not_found:{product_id}")
		if prod["enrolment_count"] > 0:
			raise PermissionError("cannot_deactivate_product_with_active_enrolments")
		prod["status"] = "deactivated"
		prod["deactivated_at"] = self._now()
		self._emit(tenant, "mic_product_deactivated", product_id, "mic_product", {})
		return deepcopy(prod)

	# ── USSD Enrolment ────────────────────────────────────────────────────────

	async def process_ussd_session(
		self,
		tenant_id: str,
		session_id: str,
		msisdn: str,
		service_code: str,
		input_text: str,
		step: int = 0,
	) -> dict[str, Any]:
		"""Process a USSD session step for product enrolment."""
		tenant = self._tenant(tenant_id)
		if not msisdn.startswith(("07", "01", "+254", "254")):
			raise ValueError("invalid_msisdn")
		session: dict[str, Any] = {
			"id": self._record_id("ussd"),
			"type": "mic_ussd_session",
			"session_id": session_id,
			"msisdn": msisdn,
			"service_code": service_code,
			"input_text": input_text,
			"step": step,
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		# Build USSD menu response
		if step == 0:
			products = [p for p in self.products.values() if p["tenant_id"] == tenant and p["status"] == "active"]
			menu_lines = ["CON Welcome to Micro-Insurance"]
			for i, p in enumerate(products[:5], 1):
				menu_lines.append(f"{i}. {p['product_name']} - KES {p['premium']}")
			session["response"] = "\n".join(menu_lines)
		elif step == 1:
			session["response"] = "CON Enter your ID number:"
		elif step == 2:
			session["response"] = f"CON Confirm enrolment? Press 1 to confirm"
		else:
			session["response"] = "END Thank you. Your policy is being processed."
			session["completed"] = True
		self.ussd_sessions[session["id"]] = session
		self._emit(tenant, "ussd_session_processed", session["id"], "mic_ussd_session", {"msisdn": msisdn, "step": step})
		return deepcopy(session)

	async def enrol_subscriber(
		self,
		tenant_id: str,
		msisdn: str,
		product_code: str,
		name: str,
		id_number: str | None = None,
		enrolment_channel: str = "ussd",
		payment_method: str = "airtime",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Enrol a mobile subscriber in a micro-insurance product."""
		tenant = self._tenant(tenant_id)
		if enrolment_channel not in ENROLMENT_CHANNELS:
			raise ValueError(f"unsupported_enrolment_channel:{enrolment_channel}")
		if payment_method not in PAYMENT_METHODS:
			raise ValueError(f"unsupported_payment_method:{payment_method}")
		# Find product
		prod = next((p for p in self.products.values() if p["product_code"] == product_code and p["tenant_id"] == tenant and p["status"] == "active"), None)
		if not prod:
			raise KeyError(f"product_not_found:{product_code}")
		# Prevent duplicate active enrolments
		existing = next(
			(e for e in self.enrolments.values()
			 if e["msisdn"] == msisdn and e["product_code"] == product_code
			 and e["tenant_id"] == tenant and e["status"] == "active"),
			None,
		)
		if existing:
			raise ValueError(f"subscriber_already_enrolled:{msisdn}")
		coverage_start = date.today()
		coverage_end = coverage_start + timedelta(days=prod["coverage_days"])
		policy_number = self._policy_number(tenant, product_code)
		record: dict[str, Any] = {
			"id": self._record_id("enr"),
			"type": "mic_enrolment",
			"policy_number": policy_number,
			"msisdn": msisdn,
			"product_code": product_code,
			"product_id": prod["id"],
			"name": name,
			"id_number": id_number,
			"enrolment_channel": enrolment_channel,
			"payment_method": payment_method,
			"premium": prod["premium"],
			"sum_insured": prod["sum_insured"],
			"coverage_start": coverage_start.isoformat(),
			"coverage_end": coverage_end.isoformat(),
			"status": "active",
			"tenant_id": tenant,
			"created_at": self._now(),
			"metadata": deepcopy(metadata or {}),
		}
		self.enrolments[record["id"]] = record
		prod["enrolment_count"] = prod.get("enrolment_count", 0) + 1
		self._emit(tenant, "subscriber_enrolled", record["id"], "mic_enrolment", {"msisdn": msisdn, "product_code": product_code})
		_log.info("Subscriber enrolled: %s product=%s channel=%s tenant=%s", msisdn, product_code, enrolment_channel, tenant)
		return deepcopy(record)

	async def get_enrolment(self, tenant_id: str, enrolment_id: str) -> dict[str, Any]:
		"""Retrieve an enrolment."""
		tenant = self._tenant(tenant_id)
		enr = self.enrolments.get(enrolment_id)
		if not enr or enr["tenant_id"] != tenant:
			raise KeyError(f"enrolment_not_found:{enrolment_id}")
		return deepcopy(enr)

	async def list_enrolments(self, tenant_id: str, product_code: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		"""List enrolments."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(e) for e in self.enrolments.values() if e["tenant_id"] == tenant]
		if product_code:
			items = [e for e in items if e["product_code"] == product_code]
		if status:
			items = [e for e in items if e["status"] == status]
		return items

	async def update_enrolment(self, tenant_id: str, enrolment_id: str, updates: dict[str, Any]) -> dict[str, Any]:
		"""Update enrolment fields."""
		tenant = self._tenant(tenant_id)
		enr = self.enrolments.get(enrolment_id)
		if not enr or enr["tenant_id"] != tenant:
			raise KeyError(f"enrolment_not_found:{enrolment_id}")
		allowed = {"name", "id_number", "payment_method", "metadata"}
		for k, v in updates.items():
			if k in allowed:
				enr[k] = v
		enr["updated_at"] = self._now()
		self._emit(tenant, "mic_enrolment_updated", enrolment_id, "mic_enrolment", {})
		return deepcopy(enr)

	async def cancel_enrolment(self, tenant_id: str, enrolment_id: str, reason: str) -> dict[str, Any]:
		"""Cancel a subscriber's enrolment."""
		tenant = self._tenant(tenant_id)
		enr = self.enrolments.get(enrolment_id)
		if not enr or enr["tenant_id"] != tenant:
			raise KeyError(f"enrolment_not_found:{enrolment_id}")
		if enr["status"] != "active":
			raise PermissionError("only_active_enrolments_can_be_cancelled")
		enr["status"] = "cancelled"
		enr["cancellation_reason"] = reason
		enr["cancelled_at"] = self._now()
		prod = next((p for p in self.products.values() if p["product_code"] == enr["product_code"] and p["tenant_id"] == tenant), None)
		if prod and prod["enrolment_count"] > 0:
			prod["enrolment_count"] -= 1
		self._emit(tenant, "mic_enrolment_cancelled", enrolment_id, "mic_enrolment", {"reason": reason})
		return deepcopy(enr)

	# ── Airtime Deduction ─────────────────────────────────────────────────────

	async def deduct_airtime_premium(
		self,
		tenant_id: str,
		msisdn: str,
		product_code: str,
		amount: Decimal,
		operator: str,
		deduction_reference: str,
	) -> dict[str, Any]:
		"""Deduct micro-insurance premium via airtime."""
		tenant = self._tenant(tenant_id)
		if operator not in MOBILE_OPERATORS:
			raise ValueError(f"unsupported_operator:{operator}")
		enrolment = next(
			(e for e in self.enrolments.values()
			 if e["msisdn"] == msisdn and e["product_code"] == product_code
			 and e["tenant_id"] == tenant and e["status"] == "active"),
			None,
		)
		if not enrolment:
			raise KeyError(f"no_active_enrolment:{msisdn}:{product_code}")
		record: dict[str, Any] = {
			"id": self._record_id("air"),
			"type": "mic_airtime_deduction",
			"msisdn": msisdn,
			"product_code": product_code,
			"enrolment_id": enrolment["id"],
			"amount": Decimal(str(amount)),
			"operator": operator,
			"deduction_reference": deduction_reference,
			"status": "processed",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.airtime_deductions[record["id"]] = record
		self._emit(tenant, "airtime_deducted", record["id"], "mic_airtime_deduction", {"msisdn": msisdn, "amount": str(amount)})
		return deepcopy(record)

	async def list_airtime_deductions(self, tenant_id: str, msisdn: str | None = None) -> list[dict[str, Any]]:
		"""List airtime deduction records."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(d) for d in self.airtime_deductions.values() if d["tenant_id"] == tenant]
		if msisdn:
			items = [d for d in items if d["msisdn"] == msisdn]
		return items

	# ── Claims ────────────────────────────────────────────────────────────────

	async def register_claim(
		self,
		tenant_id: str,
		policy_number: str,
		msisdn: str,
		incident_description: str,
		claimed_amount: Decimal,
	) -> dict[str, Any]:
		"""Register a micro-insurance claim via mobile."""
		tenant = self._tenant(tenant_id)
		enrolment = self._get_enrolment_by_policy(policy_number, tenant)
		if enrolment["msisdn"] != msisdn:
			raise PermissionError("msisdn_mismatch")
		if enrolment["status"] != "active":
			raise PermissionError("policy_not_active")
		amount = Decimal(str(claimed_amount))
		if amount > enrolment["sum_insured"]:
			raise ValueError(f"claimed_amount_exceeds_sum_insured:{enrolment['sum_insured']}")
		record: dict[str, Any] = {
			"id": self._record_id("clm"),
			"type": "mic_claim",
			"policy_number": policy_number,
			"enrolment_id": enrolment["id"],
			"msisdn": msisdn,
			"product_code": enrolment["product_code"],
			"incident_description": incident_description,
			"claimed_amount": amount,
			"approved_amount": Decimal("0"),
			"auto_pay_eligible": amount <= CLAIM_AUTO_PAY_THRESHOLD,
			"status": "registered",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.claims[record["id"]] = record
		self._emit(tenant, "mic_claim_registered", record["id"], "mic_claim", {"policy_number": policy_number, "amount": str(amount)})
		# Auto-approve and payout for small claims
		if record["auto_pay_eligible"]:
			await self._auto_approve_claim(tenant, record["id"], amount)
		return deepcopy(self.claims[record["id"]])

	async def _auto_approve_claim(self, tenant: str, claim_id: str, amount: Decimal) -> None:
		"""Auto-approve and trigger mobile money payout for small claims."""
		clm = self.claims.get(claim_id)
		if not clm:
			return
		clm["status"] = "auto_approved"
		clm["approved_amount"] = amount
		clm["approved_at"] = self._now()
		self._emit(tenant, "mic_claim_auto_approved", claim_id, "mic_claim", {"amount": str(amount)})

	async def list_claims(self, tenant_id: str, msisdn: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		"""List micro-insurance claims."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(c) for c in self.claims.values() if c["tenant_id"] == tenant]
		if msisdn:
			items = [c for c in items if c["msisdn"] == msisdn]
		if status:
			items = [c for c in items if c["status"] == status]
		return items

	async def approve_claim(self, tenant_id: str, claim_id: str, approved_amount: Decimal, approved_by: str) -> dict[str, Any]:
		"""Manually approve a claim."""
		tenant = self._tenant(tenant_id)
		clm = self.claims.get(claim_id)
		if not clm or clm["tenant_id"] != tenant:
			raise KeyError(f"claim_not_found:{claim_id}")
		clm["status"] = "approved"
		clm["approved_amount"] = Decimal(str(approved_amount))
		clm["approved_by"] = approved_by
		clm["approved_at"] = self._now()
		self._emit(tenant, "mic_claim_approved", claim_id, "mic_claim", {"amount": str(approved_amount)})
		return deepcopy(clm)

	# ── Mobile Money Payout ───────────────────────────────────────────────────

	async def process_mobile_payout(
		self,
		tenant_id: str,
		claim_id: str,
		msisdn: str,
		amount: Decimal,
		operator: str,
		mobile_money_reference: str,
	) -> dict[str, Any]:
		"""Disburse claim payout via mobile money."""
		tenant = self._tenant(tenant_id)
		if operator not in MOBILE_OPERATORS:
			raise ValueError(f"unsupported_operator:{operator}")
		clm = self.claims.get(claim_id)
		if not clm or clm["tenant_id"] != tenant:
			raise KeyError(f"claim_not_found:{claim_id}")
		if clm["status"] not in {"approved", "auto_approved"}:
			raise PermissionError("claim_must_be_approved_for_payout")
		payout_amount = Decimal(str(amount))
		record: dict[str, Any] = {
			"id": self._record_id("pay"),
			"type": "mic_mobile_payout",
			"claim_id": claim_id,
			"msisdn": msisdn,
			"amount": payout_amount,
			"operator": operator,
			"mobile_money_reference": mobile_money_reference,
			"status": "disbursed",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.mobile_payouts[record["id"]] = record
		clm["status"] = "paid"
		clm["payout_reference"] = mobile_money_reference
		clm["paid_at"] = self._now()
		self._emit(tenant, "mic_payout_disbursed", record["id"], "mic_mobile_payout", {"msisdn": msisdn, "amount": str(payout_amount)})
		return deepcopy(record)

	async def list_mobile_payouts(self, tenant_id: str, msisdn: str | None = None) -> list[dict[str, Any]]:
		"""List mobile money payouts."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(p) for p in self.mobile_payouts.values() if p["tenant_id"] == tenant]
		if msisdn:
			items = [p for p in items if p["msisdn"] == msisdn]
		return items

	# ── Renewal ───────────────────────────────────────────────────────────────

	async def renew_enrolment(self, tenant_id: str, enrolment_id: str, payment_method: str | None = None) -> dict[str, Any]:
		"""Renew an expiring or lapsed micro-insurance enrolment."""
		tenant = self._tenant(tenant_id)
		enr = self.enrolments.get(enrolment_id)
		if not enr or enr["tenant_id"] != tenant:
			raise KeyError(f"enrolment_not_found:{enrolment_id}")
		prod = next((p for p in self.products.values() if p["product_code"] == enr["product_code"] and p["tenant_id"] == tenant), None)
		if not prod:
			raise KeyError("product_not_found")
		new_start = date.today()
		new_end = new_start + timedelta(days=prod["coverage_days"])
		renewal: dict[str, Any] = {
			"id": self._record_id("ren"),
			"type": "mic_renewal",
			"enrolment_id": enrolment_id,
			"policy_number": enr["policy_number"],
			"msisdn": enr["msisdn"],
			"new_coverage_start": new_start.isoformat(),
			"new_coverage_end": new_end.isoformat(),
			"payment_method": payment_method or enr["payment_method"],
			"premium": prod["premium"],
			"status": "renewed",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.renewals[renewal["id"]] = renewal
		enr["coverage_start"] = new_start.isoformat()
		enr["coverage_end"] = new_end.isoformat()
		enr["status"] = "active"
		enr["last_renewed_at"] = self._now()
		self._emit(tenant, "mic_enrolment_renewed", renewal["id"], "mic_renewal", {"msisdn": enr["msisdn"]})
		return deepcopy(renewal)

	async def expire_enrolments(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Batch-expire enrolments past coverage end date."""
		tenant = self._tenant(tenant_id)
		today = date.today().isoformat()
		expired = []
		for enr in self.enrolments.values():
			if enr["tenant_id"] == tenant and enr["status"] == "active" and enr["coverage_end"] < today:
				enr["status"] = "expired"
				enr["expired_at"] = self._now()
				self._emit(tenant, "mic_enrolment_expired", enr["id"], "mic_enrolment", {})
				expired.append(deepcopy(enr))
		return expired

	# ── Analytics ─────────────────────────────────────────────────────────────

	async def platform_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Platform-level summary metrics."""
		tenant = self._tenant(tenant_id)
		prods = [p for p in self.products.values() if p["tenant_id"] == tenant]
		enrs = [e for e in self.enrolments.values() if e["tenant_id"] == tenant]
		clms = [c for c in self.claims.values() if c["tenant_id"] == tenant]
		active_enrolments = [e for e in enrs if e["status"] == "active"]
		by_channel: dict[str, int] = {}
		for e in enrs:
			by_channel[e["enrolment_channel"]] = by_channel.get(e["enrolment_channel"], 0) + 1
		by_operator: dict[str, int] = {}
		for d in self.airtime_deductions.values():
			if d["tenant_id"] == tenant:
				by_operator[d["operator"]] = by_operator.get(d["operator"], 0) + 1
		return {
			"tenant_id": tenant,
			"active_products": sum(1 for p in prods if p["status"] == "active"),
			"total_enrolments": len(enrs),
			"active_enrolments": len(active_enrolments),
			"total_claims": len(clms),
			"paid_claims": sum(1 for c in clms if c["status"] == "paid"),
			"enrolments_by_channel": by_channel,
			"airtime_deductions_by_operator": by_operator,
			"total_premiums_collected": str(sum(d["amount"] for d in self.airtime_deductions.values() if d["tenant_id"] == tenant)),
			"total_claims_paid": str(sum(p["amount"] for p in self.mobile_payouts.values() if p["tenant_id"] == tenant)),
			"generated_at": self._now(),
		}

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "ins_mic",
			"status": "healthy",
			"product_count": len(self.products),
			"enrolment_count": len(self.enrolments),
			"claim_count": len(self.claims),
			"checked_at": self._now(),
		}

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return {
			"capability_id": "ins_mic",
			"name": "Micro-Insurance Platform",
			"version": "1.0.0",
			"domain": "insurance",
			"tenant_id": tenant_id,
			"enrolment_channels": list(ENROLMENT_CHANNELS),
			"payment_methods": list(PAYMENT_METHODS),
			"mobile_operators": list(MOBILE_OPERATORS),
			"product_types": list(PRODUCT_TYPES),
			"auto_pay_threshold": str(CLAIM_AUTO_PAY_THRESHOLD),
		}

	async def get_audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

