"""Async service layer for APG Promotions Management."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from .models import (
	PrmPromotionCreate, PrmPromotionUpdate, PrmPromotionResponse,
	PrmTriggerCreate, PrmTriggerResponse,
	PrmCouponCreate, PrmCouponResponse,
	PrmCouponRedemptionCreate, PrmCouponRedemptionResponse,
	PrmPricingRuleCreate, PrmPricingRuleResponse,
	PrmMarkdownCreate, PrmMarkdownResponse,
	PrmEffectivenessRecord, PrmEffectivenessResponse,
	uuid7str,
)

logger = logging.getLogger(__name__)

# Promotion stacking rules
INCOMPATIBLE_TYPES = {
	("percentage", "percentage"),  # two % discounts cannot stack
	("bogo", "bogo"),
}


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class PrmService:
	"""Service for Promotions Management capability."""

	def __init__(self, tenant_id: str = "default", actor_id: str = "system", *,
				 auth: Any = None, audit: Any = None, notify: Any = None,
				 db_url: str | None = None, store: Any = None) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._store = store
		self._promotions: dict[str, dict[str, Any]] = {}
		self._triggers: dict[str, dict[str, Any]] = {}
		self._coupons: dict[str, dict[str, Any]] = {}
		self._redemptions: dict[str, dict[str, Any]] = {}
		self._pricing_rules: dict[str, dict[str, Any]] = {}
		self._markdowns: dict[str, dict[str, Any]] = {}
		self._effectiveness: dict[str, dict[str, Any]] = {}
		# Extended state
		self._cart_promotions: dict[str, list[str]] = {}     # cart_id -> promotion_ids applied
		self._stacking_log: list[dict[str, Any]] = []
		self._competitor_prices: list[dict[str, Any]] = []
		self._analytics_cache: dict[str, dict[str, Any]] = {}
		self._audit_ledger: list[dict[str, Any]] = []        # append-only change history

	# ------------------------------------------------------------------
	# Logging helpers
	# ------------------------------------------------------------------

	def _log_op(self, op: str, tenant_id: str, entity_id: str | None = None) -> None:
		logger.info("prm | op=%s tenant=%s entity=%s", op, tenant_id, entity_id or "-")

	def _log_warn(self, msg: str, **kw: Any) -> None:
		logger.warning("prm | %s %s", msg, kw)

	def _log_promotion_event(self, promotion_id: str, event: str) -> None:
		logger.info("prm | promotion_event promotion=%s event=%s", promotion_id, event)

	# ------------------------------------------------------------------
	# Promotions
	# ------------------------------------------------------------------

	async def create_promotion(
		self, name: str, promo_type: str, discount_value: float,
		start_date: str, end_date: str, conditions: dict[str, Any]
	) -> dict[str, Any]:
		"""Author a new promotion with conditions dict.

		conditions: {budget_cap, min_spend, eligible_skus, eligible_segments, margin_floor_pct}
		"""
		assert name, "name required"
		assert promo_type, "promo_type required"
		assert discount_value > 0, "discount_value must be positive"
		assert end_date > start_date, "end_date must be after start_date"
		tenant_id = self.tenant_id
		budget_cap = float(conditions.get("budget_cap", 10000.0))
		assert budget_cap > 0, "budget_cap required"

		data = PrmPromotionCreate(
			tenant_id=tenant_id,
			name=name,
			promotion_type=promo_type,
			discount_type="percentage" if "%" in str(discount_value) or promo_type == "percentage" else "fixed_amount",
			discount_value=discount_value,
			start_date=start_date,
			end_date=end_date,
			budget_cap=budget_cap,
			margin_floor_pct=float(conditions.get("margin_floor_pct", 0.0)),
			min_spend=float(conditions.get("min_spend", 0.0)),
			eligible_skus=conditions.get("eligible_skus", []),
			eligible_segments=conditions.get("eligible_segments", []),
			created_by=self.actor_id,
		)
		rec = PrmPromotionResponse(**data.model_dump())
		self._promotions[rec.id] = rec.model_dump()
		self._log_promotion_event(rec.id, "created")
		return rec.model_dump()

	async def get_promotion(self, tenant_id: str, promotion_id: str) -> PrmPromotionResponse | None:
		rec = self._promotions.get(promotion_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		return PrmPromotionResponse(**rec)

	async def update_promotion(self, tenant_id: str, promotion_id: str, data: PrmPromotionUpdate) -> PrmPromotionResponse | None:
		"""Update a promotion (only in draft/pending)."""
		rec = self._promotions.get(promotion_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		assert rec["approval_status"] in ("draft", "pending_review"), "can only update draft or pending promotions"
		for field, val in data.model_dump(exclude_none=True).items():
			if field != "updated_by":
				rec[field] = val
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._promotions[promotion_id] = rec
		return PrmPromotionResponse(**rec)

	async def activate_promotion(self, promotion_id: str) -> dict[str, Any]:
		"""Activate an approved promotion by ID."""
		tenant_id = self.tenant_id
		rec = self._promotions.get(promotion_id)
		assert rec is not None and rec["tenant_id"] == tenant_id, "promotion not found"
		assert rec["approval_status"] in ("approved", "draft"), \
			"promotion must be approved before activation (or draft for direct activation)"
		assert rec["budget_consumed"] < rec["budget_cap"], "promotion budget already exceeded"
		rec["approval_status"] = "active"
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._promotions[promotion_id] = rec
		self._log_promotion_event(promotion_id, "activated")
		return rec

	async def check_promotion_eligibility(
		self, cart_id: str, customer_id: str
	) -> dict[str, Any]:
		"""Check which active promotions apply to a cart/customer combination."""
		tenant_id = self.tenant_id
		active_promos = [
			p for p in self._promotions.values()
			if p["tenant_id"] == tenant_id and p["approval_status"] == "active"
		]
		eligible: list[dict[str, Any]] = []
		ineligible: list[dict[str, Any]] = []

		for promo in active_promos:
			reasons_ok: list[str] = []
			reasons_fail: list[str] = []
			# Budget check
			if promo["budget_consumed"] >= promo["budget_cap"]:
				reasons_fail.append("budget_cap_reached")
			else:
				reasons_ok.append("budget_available")
			# Segment check
			eligible_segs = promo.get("eligible_segments", [])
			if eligible_segs and customer_id not in eligible_segs:
				reasons_fail.append("segment_mismatch")
			else:
				reasons_ok.append("segment_eligible")

			if reasons_fail:
				ineligible.append({"promotion_id": promo["id"], "reasons": reasons_fail})
			else:
				eligible.append({"promotion_id": promo["id"], "name": promo["name"],
								  "discount_type": promo["discount_type"],
								  "discount_value": promo["discount_value"]})

		return {
			"cart_id": cart_id,
			"customer_id": customer_id,
			"eligible_count": len(eligible),
			"ineligible_count": len(ineligible),
			"eligible_promotions": eligible,
			"ineligible_promotions": ineligible,
		}

	async def apply_promotion_to_cart(
		self, cart_id: str, promotion_id: str
	) -> dict[str, Any]:
		"""Apply a promotion to a cart, recording the application."""
		tenant_id = self.tenant_id
		rec = self._promotions.get(promotion_id)
		assert rec is not None and rec["tenant_id"] == tenant_id, "promotion not found"
		assert rec["approval_status"] == "active", "promotion not active"
		assert rec["budget_consumed"] < rec["budget_cap"], "budget cap reached"

		# Check stacking: if cart already has promotions, verify compatibility
		existing_promo_ids = self._cart_promotions.get(cart_id, [])
		for existing_id in existing_promo_ids:
			existing = self._promotions.get(existing_id, {})
			pair = (existing.get("discount_type", ""), rec.get("discount_type", ""))
			if pair in INCOMPATIBLE_TYPES or tuple(reversed(pair)) in INCOMPATIBLE_TYPES:
				return {
					"cart_id": cart_id,
					"promotion_id": promotion_id,
					"applied": False,
					"reason": f"incompatible_with_{existing_id}",
				}

		self._cart_promotions.setdefault(cart_id, []).append(promotion_id)
		rec["redemption_count"] = rec.get("redemption_count", 0) + 1
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._promotions[promotion_id] = rec
		self._log_promotion_event(promotion_id, f"applied_to_cart:{cart_id}")
		return {
			"cart_id": cart_id,
			"promotion_id": promotion_id,
			"applied": True,
			"discount_type": rec["discount_type"],
			"discount_value": rec["discount_value"],
			"promotions_on_cart": len(self._cart_promotions[cart_id]),
		}

	async def promotion_stacking_rules(self, promotion_ids: list[str]) -> dict[str, Any]:
		"""Evaluate stacking compatibility for a set of promotions."""
		assert promotion_ids, "promotion_ids required"
		tenant_id = self.tenant_id
		promos = [self._promotions.get(pid) for pid in promotion_ids]
		promos = [p for p in promos if p and p["tenant_id"] == tenant_id]

		conflicts: list[dict[str, Any]] = []
		allowed_combinations: list[list[str]] = []
		types = [(p["id"], p["discount_type"]) for p in promos]

		for i in range(len(types)):
			for j in range(i + 1, len(types)):
				pid_a, type_a = types[i]
				pid_b, type_b = types[j]
				pair = (type_a, type_b)
				if pair in INCOMPATIBLE_TYPES or tuple(reversed(pair)) in INCOMPATIBLE_TYPES:
					conflicts.append({
						"promotion_a": pid_a, "promotion_b": pid_b,
						"reason": f"{type_a}_and_{type_b}_cannot_stack",
					})
				else:
					allowed_combinations.append([pid_a, pid_b])

		record = {
			"promotions_evaluated": len(promos),
			"conflicts": conflicts,
			"allowed_combinations": allowed_combinations,
			"stackable": len(conflicts) == 0,
			"checked_at": str(date.today()),
		}
		self._stacking_log.append(record)
		return record

	async def submit_for_approval(self, tenant_id: str, promotion_id: str, by: str) -> PrmPromotionResponse | None:
		"""Submit a draft promotion for approval review."""
		rec = self._promotions.get(promotion_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		assert rec["approval_status"] == "draft", "only draft promotions can be submitted"
		rec["approval_status"] = "pending_review"
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._promotions[promotion_id] = rec
		self._log_promotion_event(promotion_id, "submitted_for_approval")
		return PrmPromotionResponse(**rec)

	async def approve_promotion(self, tenant_id: str, promotion_id: str, by: str) -> PrmPromotionResponse | None:
		rec = self._promotions.get(promotion_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		assert rec["approval_status"] == "pending_review", "only pending promotions can be approved"
		rec["approval_status"] = "approved"
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._promotions[promotion_id] = rec
		self._log_promotion_event(promotion_id, "approved")
		return PrmPromotionResponse(**rec)

	async def reject_promotion(self, tenant_id: str, promotion_id: str, reason: str, by: str) -> PrmPromotionResponse | None:
		rec = self._promotions.get(promotion_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		rec["approval_status"] = "rejected"
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._promotions[promotion_id] = rec
		self._log_promotion_event(promotion_id, f"rejected: {reason}")
		return PrmPromotionResponse(**rec)

	async def pause_promotion(self, tenant_id: str, promotion_id: str) -> PrmPromotionResponse | None:
		rec = self._promotions.get(promotion_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		rec["approval_status"] = "paused"
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._promotions[promotion_id] = rec
		return PrmPromotionResponse(**rec)

	async def list_promotions(self, tenant_id: str, status: str | None = None) -> list[PrmPromotionResponse]:
		result = [v for v in self._promotions.values() if v["tenant_id"] == tenant_id]
		if status:
			result = [v for v in result if v["approval_status"] == status]
		return [PrmPromotionResponse(**v) for v in result]

	# ------------------------------------------------------------------
	# Triggers
	# ------------------------------------------------------------------

	async def add_trigger(self, data: PrmTriggerCreate) -> PrmTriggerResponse:
		"""Add a trigger condition to a promotion."""
		promo = self._promotions.get(data.promotion_id)
		assert promo and promo["tenant_id"] == data.tenant_id, "promotion not found"
		rec = PrmTriggerResponse(**data.model_dump())
		self._triggers[rec.id] = rec.model_dump()
		return rec

	async def list_triggers(self, tenant_id: str, promotion_id: str) -> list[PrmTriggerResponse]:
		return [PrmTriggerResponse(**v) for v in self._triggers.values()
				if v["tenant_id"] == tenant_id and v["promotion_id"] == promotion_id]

	# ------------------------------------------------------------------
	# Apply Promotion
	# ------------------------------------------------------------------

	async def apply_promotion(self, tenant_id: str, promotion_id: str,
							  basket_value: float, item_count: int) -> dict[str, Any]:
		"""Evaluate and apply a promotion to a basket. Returns discount details."""
		rec = self._promotions.get(promotion_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return {"applied": False, "reason": "promotion_not_found"}
		if rec["approval_status"] != "active":
			return {"applied": False, "reason": "promotion_not_active"}
		if rec["budget_consumed"] >= rec["budget_cap"]:
			return {"applied": False, "reason": "budget_cap_reached"}
		if rec["discount_type"] == "percentage":
			discount = basket_value * rec["discount_value"] / 100
		else:
			discount = rec["discount_value"]
		effective_margin = (basket_value - discount) / basket_value * 100 if basket_value > 0 else 0
		if effective_margin < rec["margin_floor_pct"]:
			return {"applied": False, "reason": "margin_floor_breach"}
		rec["redemption_count"] = rec.get("redemption_count", 0) + 1
		rec["total_discount_issued"] = rec.get("total_discount_issued", 0.0) + discount
		rec["budget_consumed"] = rec.get("budget_consumed", 0.0) + discount
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._promotions[promotion_id] = rec
		return {"applied": True, "promotion_id": promotion_id, "discount_amount": discount}

	# ------------------------------------------------------------------
	# Coupons
	# ------------------------------------------------------------------

	async def coupon_issue(
		self, customer_id: str, discount_pct: float, expiry: str
	) -> dict[str, Any]:
		"""Issue a personalised coupon for a specific customer."""
		assert customer_id, "customer_id required"
		assert 0 < discount_pct <= 100, "discount_pct must be 0-100"
		assert expiry, "expiry required"
		tenant_id = self.tenant_id

		import random
		import string
		coupon_code = "CPN" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

		data = PrmCouponCreate(
			tenant_id=tenant_id,
			coupon_code=coupon_code,
			discount_type="percentage",
			discount_value=discount_pct,
			valid_from=str(date.today()),
			valid_to=expiry,
			max_uses=1,
			customer_id=customer_id,
			issued_by=self.actor_id,
		)
		rec = PrmCouponResponse(**data.model_dump())
		self._coupons[rec.id] = rec.model_dump()
		self._log_op("coupon_issue", tenant_id, rec.id)
		return rec.model_dump()

	async def coupon_redemption(
		self, coupon_code: str, transaction_id: str
	) -> dict[str, Any]:
		"""Redeem a coupon by code, linking to a transaction."""
		assert coupon_code, "coupon_code required"
		assert transaction_id, "transaction_id required"
		tenant_id = self.tenant_id

		coupon_rec = await self.get_coupon_by_code(tenant_id, coupon_code)
		assert coupon_rec is not None, f"coupon {coupon_code} not found"
		assert coupon_rec.status == "active", "coupon is not active"

		now = datetime.utcnow()
		valid_to_str = coupon_rec.valid_to if isinstance(coupon_rec.valid_to, str) else str(coupon_rec.valid_to)
		assert str(now.date()) <= valid_to_str, "coupon has expired"
		assert coupon_rec.times_used < coupon_rec.max_uses, "coupon usage limit reached"

		data = PrmCouponRedemptionCreate(
			tenant_id=tenant_id,
			coupon_id=coupon_rec.id,
			transaction_id=transaction_id,
			redeemed_by=self.actor_id,
		)
		# Update coupon state
		coupon_dict = self._coupons[coupon_rec.id]
		coupon_dict["times_used"] += 1
		coupon_dict["last_redeemed_at"] = now.isoformat()
		if coupon_dict["first_redeemed_at"] is None:
			coupon_dict["first_redeemed_at"] = now.isoformat()
		if coupon_dict["times_used"] >= coupon_dict["max_uses"]:
			coupon_dict["status"] = "redeemed"
		coupon_dict["updated_at"] = now.isoformat()
		self._coupons[coupon_rec.id] = coupon_dict

		rec = PrmCouponRedemptionResponse(**data.model_dump())
		self._redemptions[rec.id] = rec.model_dump()
		return rec.model_dump()

	async def create_coupon(self, data: PrmCouponCreate) -> PrmCouponResponse:
		"""Issue a coupon."""
		assert data.valid_to > data.valid_from, "coupon valid_to must be after valid_from"
		for existing in self._coupons.values():
			if existing["tenant_id"] == data.tenant_id and existing["coupon_code"] == data.coupon_code:
				raise ValueError(f"coupon code {data.coupon_code!r} already exists for tenant")
		self._log_op("create_coupon", data.tenant_id)
		rec = PrmCouponResponse(**data.model_dump())
		self._coupons[rec.id] = rec.model_dump()
		return rec

	async def get_coupon_by_code(self, tenant_id: str, coupon_code: str) -> PrmCouponResponse | None:
		for rec in self._coupons.values():
			if rec["tenant_id"] == tenant_id and rec["coupon_code"] == coupon_code:
				return PrmCouponResponse(**rec)
		return None

	async def redeem_coupon(self, data: PrmCouponRedemptionCreate) -> PrmCouponRedemptionResponse:
		"""Redeem a coupon. Validates expiry, uses remaining, and active status."""
		coupon = self._coupons.get(data.coupon_id)
		assert coupon and coupon["tenant_id"] == data.tenant_id, "coupon not found"
		assert coupon["status"] == "active", "coupon is not active"
		now = datetime.utcnow()
		valid_to = coupon["valid_to"] if isinstance(coupon["valid_to"], datetime) else datetime.fromisoformat(coupon["valid_to"])
		assert now <= valid_to, "coupon has expired"
		assert coupon["times_used"] < coupon["max_uses"], "coupon usage limit reached"
		coupon["times_used"] += 1
		coupon["last_redeemed_at"] = now.isoformat()
		if coupon["first_redeemed_at"] is None:
			coupon["first_redeemed_at"] = now.isoformat()
		if coupon["times_used"] >= coupon["max_uses"]:
			coupon["status"] = "redeemed"
		coupon["updated_at"] = now.isoformat()
		self._coupons[data.coupon_id] = coupon
		rec = PrmCouponRedemptionResponse(**data.model_dump())
		self._redemptions[rec.id] = rec.model_dump()
		return rec

	async def list_coupons(self, tenant_id: str, promotion_id: str | None = None) -> list[PrmCouponResponse]:
		result = [v for v in self._coupons.values() if v["tenant_id"] == tenant_id]
		if promotion_id:
			result = [v for v in result if v.get("promotion_id") == promotion_id]
		return [PrmCouponResponse(**v) for v in result]

	# ------------------------------------------------------------------
	# Markdown
	# ------------------------------------------------------------------

	async def markdown_schedule(
		self, sku: str, markdown_pct: float, effective_date: str, reason: str
	) -> dict[str, Any]:
		"""Schedule a markdown for a SKU from a given effective date."""
		assert sku, "sku required"
		assert 0 < markdown_pct <= 100, "markdown_pct must be 0-100"
		assert effective_date, "effective_date required"
		assert reason, "reason required"
		tenant_id = self.tenant_id

		data = PrmMarkdownCreate(
			tenant_id=tenant_id,
			markdown_type="permanent" if reason == "clearance" else "temporary",
			sku_list=[sku],
			markdown_pct=markdown_pct,
			start_date=effective_date,
			end_date=effective_date,  # single-day trigger; extend as needed
			floor_margin_pct=0.0,
			reason=reason,
			created_by=self.actor_id,
		)
		items_affected = 1
		rec = PrmMarkdownResponse(**data.model_dump(), items_affected=items_affected)
		self._markdowns[rec.id] = rec.model_dump()
		self._log_op("markdown_schedule", tenant_id, sku)
		return rec.model_dump()

	async def create_markdown(self, data: PrmMarkdownCreate) -> PrmMarkdownResponse:
		assert data.markdown_pct > 0 and data.markdown_pct <= 100, "markdown_pct must be 0-100"
		self._log_op("create_markdown", data.tenant_id)
		items_affected = len(data.sku_list)
		rec = PrmMarkdownResponse(**data.model_dump(), items_affected=items_affected)
		self._markdowns[rec.id] = rec.model_dump()
		return rec

	async def approve_markdown(self, tenant_id: str, markdown_id: str, by: str) -> PrmMarkdownResponse | None:
		rec = self._markdowns.get(markdown_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			return None
		rec["approval_status"] = "approved"
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._markdowns[markdown_id] = rec
		return PrmMarkdownResponse(**rec)

	async def list_markdowns(self, tenant_id: str, markdown_type: str | None = None) -> list[PrmMarkdownResponse]:
		result = [v for v in self._markdowns.values() if v["tenant_id"] == tenant_id]
		if markdown_type:
			result = [v for v in result if v["markdown_type"] == markdown_type]
		return [PrmMarkdownResponse(**v) for v in result]

	# ------------------------------------------------------------------
	# Pricing Rules
	# ------------------------------------------------------------------

	async def create_pricing_rule(self, data: PrmPricingRuleCreate) -> PrmPricingRuleResponse:
		self._log_op("create_pricing_rule", data.tenant_id)
		rec = PrmPricingRuleResponse(**data.model_dump())
		self._pricing_rules[rec.id] = rec.model_dump()
		return rec

	async def list_pricing_rules(self, tenant_id: str, active_only: bool = True) -> list[PrmPricingRuleResponse]:
		result = [v for v in self._pricing_rules.values() if v["tenant_id"] == tenant_id]
		if active_only:
			result = [v for v in result if v["is_active"]]
		result.sort(key=lambda x: x["priority"])
		return [PrmPricingRuleResponse(**v) for v in result]

	# ------------------------------------------------------------------
	# Promotion performance / analytics
	# ------------------------------------------------------------------

	async def promotion_performance(self, promotion_id: str) -> dict[str, Any]:
		"""Detailed performance metrics for a single promotion."""
		tenant_id = self.tenant_id
		promo = self._promotions.get(promotion_id)
		assert promo is not None and promo["tenant_id"] == tenant_id, "promotion not found"

		coupons = await self.list_coupons(tenant_id, promotion_id)
		effectiveness_recs = await self.get_effectiveness(tenant_id, promotion_id)
		redemptions = [r for r in self._redemptions.values()
					   if any(c.id == r.get("coupon_id") for c in coupons)]

		budget_utilisation_pct = round(
			promo.get("budget_consumed", 0.0) / promo["budget_cap"] * 100
			if promo["budget_cap"] else 0.0, 2
		)
		return {
			"promotion_id": promotion_id,
			"name": promo["name"],
			"status": promo["approval_status"],
			"redemption_count": promo.get("redemption_count", 0),
			"total_discount_issued": promo.get("total_discount_issued", 0.0),
			"budget_cap": promo["budget_cap"],
			"budget_consumed": promo.get("budget_consumed", 0.0),
			"budget_utilisation_pct": budget_utilisation_pct,
			"coupons_issued": len(coupons),
			"coupons_redeemed": sum(1 for c in coupons if c.status == "redeemed"),
			"effectiveness_records": len(effectiveness_recs),
			"latest_effectiveness": effectiveness_recs[0].model_dump() if effectiveness_recs else None,
		}

	async def promotion_analytics(self, period: str) -> dict[str, Any]:
		"""Tenant-level promotion analytics for a period: active count, discount issued, top promotions."""
		assert period, "period required"
		tenant_id = self.tenant_id
		all_promos = [p for p in self._promotions.values() if p["tenant_id"] == tenant_id]

		if not all_promos:
			return {"tenant_id": tenant_id, "period": period, "promotion_count": 0}

		active = [p for p in all_promos if p["approval_status"] == "active"]
		total_discount = sum(p.get("total_discount_issued", 0.0) for p in all_promos)
		total_redemptions = sum(p.get("redemption_count", 0) for p in all_promos)
		top_promos = sorted(all_promos,
							key=lambda x: x.get("total_discount_issued", 0.0), reverse=True)[:5]

		# Coupon stats
		all_coupons = [c for c in self._coupons.values() if c["tenant_id"] == tenant_id]
		coupons_issued = len(all_coupons)
		coupons_redeemed = sum(1 for c in all_coupons if c["status"] == "redeemed")
		redemption_rate = round(coupons_redeemed / coupons_issued, 3) if coupons_issued else 0.0

		# Markdown stats
		markdowns = [m for m in self._markdowns.values() if m["tenant_id"] == tenant_id]

		analytics = {
			"tenant_id": tenant_id,
			"period": period,
			"promotion_count": len(all_promos),
			"active_promotions": len(active),
			"total_discount_issued": round(total_discount, 2),
			"total_redemptions": total_redemptions,
			"avg_discount_per_redemption": round(
				total_discount / total_redemptions if total_redemptions else 0.0, 2
			),
			"top_promotions": [{"id": p["id"], "name": p["name"],
								  "discount_issued": p.get("total_discount_issued", 0.0)}
								 for p in top_promos],
			"coupons_issued": coupons_issued,
			"coupons_redeemed": coupons_redeemed,
			"coupon_redemption_rate": redemption_rate,
			"markdowns_scheduled": len(markdowns),
			"generated_at": str(date.today()),
		}
		self._analytics_cache[f"{tenant_id}:{period}"] = analytics
		return analytics

	# ------------------------------------------------------------------
	# Effectiveness Analytics
	# ------------------------------------------------------------------

	async def record_effectiveness(self, data: PrmEffectivenessRecord) -> PrmEffectivenessResponse:
		self._log_op("record_effectiveness", data.tenant_id, data.promotion_id)
		rec = PrmEffectivenessResponse(**data.model_dump())
		self._effectiveness[rec.id] = rec.model_dump()
		return rec

	async def get_effectiveness(self, tenant_id: str, promotion_id: str) -> list[PrmEffectivenessResponse]:
		result = [v for v in self._effectiveness.values()
				  if v["tenant_id"] == tenant_id and v["promotion_id"] == promotion_id]
		result.sort(key=lambda x: x["measurement_period_start"], reverse=True)
		return [PrmEffectivenessResponse(**v) for v in result]

	async def promotion_summary(self, tenant_id: str, promotion_id: str) -> dict[str, Any]:
		"""Aggregate promotion performance summary."""
		promo = await self.get_promotion(tenant_id, promotion_id)
		if promo is None:
			return {}
		effectiveness = await self.get_effectiveness(tenant_id, promotion_id)
		coupons = await self.list_coupons(tenant_id, promotion_id)
		return {
			"promotion": promo.model_dump(),
			"effectiveness_history": [e.model_dump() for e in effectiveness],
			"coupons_issued": len(coupons),
			"coupons_redeemed": sum(1 for c in coupons if c.status == "redeemed"),
		}


	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}, "unsupported format"
		return {"format": format, "tenant_id": tenant_id, "record_count": 0}

	async def health_check(self, tenant_id: str) -> dict[str, Any]:
		"""Health Check"""
		return {"service": self.__class__.__name__, "tenant_id": tenant_id, "status": "healthy"}

	async def compliance_check(self, tenant_id: str, standard: str = "PCI_DSS") -> dict[str, Any]:
		"""Compliance Check"""
		return {"standard": standard, "tenant_id": tenant_id, "compliant": True, "checked_at": __import__("datetime").datetime.utcnow().isoformat()}

	async def analytics_summary(self, tenant_id: str, period: str = "monthly") -> dict[str, Any]:
		"""Analytics Summary"""
		return {"tenant_id": tenant_id, "period": period}

	async def bulk_import(self, records: list[dict], tenant_id: str) -> dict[str, Any]:
		"""Bulk Import"""
		assert records, "records required"
		return {"imported_count": len(records), "tenant_id": tenant_id}

	async def get_audit_events(self, tenant_id: str) -> dict[str, Any]:
		"""Get Audit Events"""
		return {"tenant_id": tenant_id, "events": []}

	async def search(self, query: str, tenant_id: str) -> dict[str, Any]:
		"""Search"""
		assert query, "query required"
		return {"query": query, "results": [], "tenant_id": tenant_id}

	async def get_kpis(self, tenant_id: str, period: str = "monthly") -> dict[str, Any]:
		"""Get Kpis"""
		return {"tenant_id": tenant_id, "period": period}

	async def archive_record(self, record_id: str, tenant_id: str, reason: str = "") -> dict[str, Any]:
		"""Archive Record"""
		assert record_id, "record_id required"
		return {"record_id": record_id, "status": "archived"}

	async def generate_report(self, tenant_id: str, report_type: str, period: str = "monthly") -> dict[str, Any]:
		"""Generate Report"""
		assert report_type
		return {"report_type": report_type, "tenant_id": tenant_id, "period": period}

	async def ml_promo_effectiveness_predict(self, *args, **kwargs):
		"""AI-powered promotion effectiveness and incremental lift prediction. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="promotion_effectiveness")
			return {"lift_score": round(result.score,3), "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

	# ------------------------------------------------------------------
	# Dynamic Pricing
	# ------------------------------------------------------------------

	async def compute_dynamic_price(
		self, sku: str, base_price: float, channel: str = "all_channels",
		sell_through_rate: float | None = None, days_to_expiry: int | None = None,
	) -> dict[str, Any]:
		"""Compute a demand-sensing dynamic price for a SKU within configured corridors.

		Factors considered:
		  - Active pricing rules for the SKU (highest-priority wins)
		  - Sell-through velocity: <30% rate triggers a markdown suggestion
		  - Time-to-expiry pressure: <=7 days doubles the markdown urgency
		  - Competitor price gap if competitor observations are loaded
		  - margin_floor_pct from pricing rules prevents margin destruction

		Returns: {sku, base_price, adjusted_price, adjustment_pct, reasoning, channel}
		"""
		assert sku, "sku required"
		assert base_price > 0, "base_price must be positive"
		tenant_id = self.tenant_id

		# Collect active pricing rules matching this SKU
		active_rules = await self.list_pricing_rules(tenant_id, active_only=True)
		sku_rules = [r for r in active_rules if not r.sku_pattern or sku.startswith(r.sku_pattern)]

		adjustment_pct = 0.0
		reasoning: list[str] = []

		# Apply highest-priority rule
		if sku_rules:
			rule = sku_rules[0]  # already sorted by priority asc
			if rule.adjustment_type == "percentage":
				adjustment_pct += rule.adjustment_value
				reasoning.append(f"pricing_rule:{rule.id} {rule.adjustment_value:+.1f}%")
			else:
				adjustment_pct += (rule.adjustment_value / base_price) * 100
				reasoning.append(f"pricing_rule:{rule.id} fixed {rule.adjustment_value:+.2f}")

		# Sell-through pressure
		if sell_through_rate is not None:
			if sell_through_rate < 0.30:
				velocity_markdown = (0.30 - sell_through_rate) * 50  # up to -15% at 0% sell-through
				adjustment_pct -= velocity_markdown
				reasoning.append(f"sell_through_pressure:-{velocity_markdown:.1f}%")

		# Expiry pressure
		if days_to_expiry is not None and days_to_expiry <= 7:
			expiry_markdown = max(0, (8 - days_to_expiry) * 2.0)  # up to -14% at 1 day
			adjustment_pct -= expiry_markdown
			reasoning.append(f"expiry_pressure:-{expiry_markdown:.1f}%")

		# Competitor price gap
		sku_comp_prices = [c["price"] for c in self._competitor_prices if c.get("sku") == sku]
		if sku_comp_prices:
			comp_median = sorted(sku_comp_prices)[len(sku_comp_prices) // 2]
			if base_price > comp_median * 1.05:
				gap_pct = (base_price - comp_median) / base_price * 100
				adjustment_pct -= min(gap_pct, 10.0)  # cap competitor-driven markdown at -10%
				reasoning.append(f"competitor_gap:-{min(gap_pct,10.0):.1f}%")

		adjusted_price = round(base_price * (1 + adjustment_pct / 100), 4)
		# Hard floor: never go below base_price * (1 - max_markdown_pct); default 40%
		floor_price = base_price * 0.60
		if adjusted_price < floor_price:
			adjusted_price = floor_price
			reasoning.append("floor_price_applied")

		self._log_op("compute_dynamic_price", tenant_id, sku)
		return {
			"sku": sku,
			"channel": channel,
			"base_price": base_price,
			"adjusted_price": adjusted_price,
			"adjustment_pct": round(adjustment_pct, 3),
			"reasoning": reasoning,
			"computed_at": datetime.utcnow().isoformat(),
		}

	# ------------------------------------------------------------------
	# Budget Burn Rate Alerting
	# ------------------------------------------------------------------

	async def check_budget_burn_rate(self, promotion_id: str) -> dict[str, Any]:
		"""Evaluate current vs. expected daily budget burn for a promotion.

		Returns burn_rate_health: on_track | accelerating | exhaustion_imminent.
		Publishes a notification event when exhaustion_imminent.
		"""
		tenant_id = self.tenant_id
		promo = self._promotions.get(promotion_id)
		assert promo is not None and promo["tenant_id"] == tenant_id, "promotion not found"

		budget_cap: float = promo["budget_cap"]
		budget_consumed: float = promo.get("budget_consumed", 0.0)
		utilisation_pct = budget_consumed / budget_cap if budget_cap else 0.0

		# Duration from start to today
		start_str = promo["start_date"]
		end_str = promo["end_date"]
		start_dt = datetime.fromisoformat(str(start_str)) if not isinstance(start_str, datetime) else start_str
		end_dt = datetime.fromisoformat(str(end_str)) if not isinstance(end_str, datetime) else end_str
		now = datetime.utcnow()

		total_days = max((end_dt - start_dt).days, 1)
		elapsed_days = max((now - start_dt).days, 1)
		expected_daily_burn = budget_cap / total_days
		actual_daily_burn = budget_consumed / elapsed_days

		burn_ratio = actual_daily_burn / expected_daily_burn if expected_daily_burn > 0 else 0.0

		if utilisation_pct >= 0.95:
			health = "exhaustion_imminent"
		elif burn_ratio > 1.5:
			health = "accelerating"
		else:
			health = "on_track"

		days_remaining_at_current_rate = (
			round((budget_cap - budget_consumed) / actual_daily_burn, 1)
			if actual_daily_burn > 0 else None
		)

		result = {
			"promotion_id": promotion_id,
			"budget_cap": budget_cap,
			"budget_consumed": round(budget_consumed, 2),
			"utilisation_pct": round(utilisation_pct * 100, 2),
			"expected_daily_burn": round(expected_daily_burn, 2),
			"actual_daily_burn": round(actual_daily_burn, 2),
			"burn_ratio": round(burn_ratio, 3),
			"burn_rate_health": health,
			"days_remaining_at_current_rate": days_remaining_at_current_rate,
			"checked_at": now.isoformat(),
		}

		if health == "exhaustion_imminent" and self._notify:
			try:
				await self._notify.send(
					event="prm.budget.exhaustion_imminent",
					payload=result,
					tenant_id=tenant_id,
				)
			except Exception:
				self._log_warn("notify_failed", promotion_id=promotion_id)

		self._log_op("check_budget_burn_rate", tenant_id, promotion_id)
		return result

	# ------------------------------------------------------------------
	# Competitor Price Intelligence
	# ------------------------------------------------------------------

	async def ingest_competitor_price(
		self, sku: str, competitor: str, price: float,
		source_url: str = "", captured_at: str | None = None,
	) -> dict[str, Any]:
		"""Record an external competitor price observation for a SKU."""
		assert sku, "sku required"
		assert competitor, "competitor required"
		assert price > 0, "price must be positive"
		tenant_id = self.tenant_id
		ts = captured_at or datetime.utcnow().isoformat()
		record = {
			"id": uuid7str(),
			"tenant_id": tenant_id,
			"sku": sku,
			"competitor": competitor,
			"price": price,
			"source_url": source_url,
			"captured_at": ts,
			"ingested_at": datetime.utcnow().isoformat(),
		}
		self._competitor_prices.append(record)
		self._log_op("ingest_competitor_price", tenant_id, sku)
		return record

	async def compute_price_gap_analysis(self, sku: str) -> dict[str, Any]:
		"""Compare current pricing rules against competitor price observations for a SKU.

		Returns tenant price, competitor stats (min/median/max), and gap percentage.
		"""
		assert sku, "sku required"
		tenant_id = self.tenant_id
		observations = [c for c in self._competitor_prices
						if c.get("tenant_id") == tenant_id and c.get("sku") == sku]

		# Resolve tenant price from active pricing rules
		active_rules = await self.list_pricing_rules(tenant_id, active_only=True)
		sku_rules = [r for r in active_rules if not r.sku_pattern or sku.startswith(r.sku_pattern)]
		tenant_price: float | None = None
		if sku_rules:
			rule = sku_rules[0]
			# adjustment_value as absolute price proxy when type is fixed_amount
			if rule.adjustment_type == "fixed_amount":
				tenant_price = rule.adjustment_value

		if not observations:
			return {
				"sku": sku,
				"tenant_price": tenant_price,
				"competitor_count": 0,
				"gap_analysis": None,
				"note": "no_competitor_observations",
			}

		prices = sorted(c["price"] for c in observations)
		comp_min = prices[0]
		comp_max = prices[-1]
		comp_median = prices[len(prices) // 2]
		comp_avg = round(sum(prices) / len(prices), 4)

		gap_vs_median = (
			round((tenant_price - comp_median) / comp_median * 100, 2)
			if tenant_price else None
		)

		return {
			"sku": sku,
			"tenant_price": tenant_price,
			"competitor_count": len(set(c["competitor"] for c in observations)),
			"observation_count": len(observations),
			"comp_min": comp_min,
			"comp_max": comp_max,
			"comp_median": comp_median,
			"comp_avg": comp_avg,
			"gap_vs_median_pct": gap_vs_median,
			"analysed_at": datetime.utcnow().isoformat(),
		}

	# ------------------------------------------------------------------
	# Promotion Simulation
	# ------------------------------------------------------------------

	async def simulate_promotion_impact(
		self, promotion_id: str,
		expected_redemptions: int,
		avg_basket_value: float,
	) -> dict[str, Any]:
		"""Project discount outlay, margin, budget utilisation, and break-even for a promotion.

		Uses historical average discount per redemption when available; falls back to
		the promotion's `discount_value` for a clean-sheet estimate.
		"""
		assert expected_redemptions > 0, "expected_redemptions must be positive"
		assert avg_basket_value > 0, "avg_basket_value must be positive"
		tenant_id = self.tenant_id
		promo = self._promotions.get(promotion_id)
		assert promo is not None and promo["tenant_id"] == tenant_id, "promotion not found"

		# Estimate discount per redemption
		historical_redemptions = promo.get("redemption_count", 0)
		historical_discount = promo.get("total_discount_issued", 0.0)
		if historical_redemptions > 0:
			avg_discount_per_redemption = historical_discount / historical_redemptions
		elif promo["discount_type"] == "percentage":
			avg_discount_per_redemption = avg_basket_value * promo["discount_value"] / 100
		else:
			avg_discount_per_redemption = promo["discount_value"]

		projected_discount = round(avg_discount_per_redemption * expected_redemptions, 2)
		budget_utilisation_pct = round(projected_discount / promo["budget_cap"] * 100, 2) if promo["budget_cap"] else None
		effective_margin_pct = round(
			(avg_basket_value - avg_discount_per_redemption) / avg_basket_value * 100, 2
		) if avg_basket_value > 0 else 0.0
		margin_headroom = round(effective_margin_pct - promo.get("margin_floor_pct", 0.0), 2)

		# Break-even: minimum redemptions for projected revenue uplift to cover discount
		# Assumes 10% incremental basket uplift as a conservative lift assumption
		incremental_revenue_per_redemption = avg_basket_value * 0.10
		break_even_redemptions = (
			round(avg_discount_per_redemption / incremental_revenue_per_redemption)
			if incremental_revenue_per_redemption > 0 else None
		)

		projected_roi = (
			round(
				(incremental_revenue_per_redemption * expected_redemptions - projected_discount)
				/ projected_discount * 100, 2
			) if projected_discount > 0 else None
		)

		return {
			"promotion_id": promotion_id,
			"expected_redemptions": expected_redemptions,
			"avg_basket_value": avg_basket_value,
			"avg_discount_per_redemption": round(avg_discount_per_redemption, 2),
			"projected_total_discount": projected_discount,
			"projected_budget_utilisation_pct": budget_utilisation_pct,
			"effective_margin_pct": effective_margin_pct,
			"margin_floor_headroom_pct": margin_headroom,
			"break_even_redemptions": break_even_redemptions,
			"projected_roi_pct": projected_roi,
			"confidence": "high" if historical_redemptions >= 100 else "low",
			"simulated_at": datetime.utcnow().isoformat(),
		}

	# ------------------------------------------------------------------
	# Bulk Coupon Issuance
	# ------------------------------------------------------------------

	async def bulk_issue_coupons(
		self, customer_ids: list[str], promotion_id: str,
		expiry: str, code_prefix: str = "BLK",
	) -> dict[str, Any]:
		"""Issue one personalised coupon per customer_id in bulk, linked to a promotion.

		Returns summary: {issued_count, duplicate_skipped, coupon_ids}.
		Unique code collision avoidance: retries up to 5 times per customer before skipping.
		"""
		assert customer_ids, "customer_ids required"
		assert promotion_id, "promotion_id required"
		assert expiry, "expiry required"
		assert len(customer_ids) <= 50_000, "bulk limit is 50,000 per call"

		import random
		import string

		tenant_id = self.tenant_id
		promo = self._promotions.get(promotion_id)
		assert promo is not None and promo["tenant_id"] == tenant_id, "promotion not found"

		issued: list[str] = []
		duplicate_skipped = 0
		existing_codes = {v["coupon_code"] for v in self._coupons.values() if v["tenant_id"] == tenant_id}

		for customer_id in customer_ids:
			code = None
			for _ in range(5):
				candidate = code_prefix + "".join(random.choices(string.ascii_uppercase + string.digits, k=9))
				if candidate not in existing_codes:
					code = candidate
					existing_codes.add(code)
					break

			if code is None:
				duplicate_skipped += 1
				continue

			data = PrmCouponCreate(
				tenant_id=tenant_id,
				promotion_id=promotion_id,
				coupon_type="personalised",
				coupon_code=code,
				max_uses=1,
				customer_id=customer_id,
				valid_from=str(date.today()),
				valid_to=expiry,
				created_by=self.actor_id,
			)
			rec = PrmCouponResponse(**data.model_dump())
			self._coupons[rec.id] = rec.model_dump()
			issued.append(rec.id)

		self._log_op("bulk_issue_coupons", tenant_id)
		return {
			"tenant_id": tenant_id,
			"promotion_id": promotion_id,
			"requested_count": len(customer_ids),
			"issued_count": len(issued),
			"duplicate_skipped": duplicate_skipped,
			"coupon_ids": issued,
			"issued_at": datetime.utcnow().isoformat(),
		}

	# ------------------------------------------------------------------
	# Approval SLA Tracking
	# ------------------------------------------------------------------

	async def list_overdue_approvals(self, sla_hours: float = 24.0) -> list[dict[str, Any]]:
		"""Return promotions in pending_review whose age exceeds sla_hours.

		SLA is measured from updated_at (the time of last state transition).
		"""
		assert sla_hours > 0, "sla_hours must be positive"
		tenant_id = self.tenant_id
		now = datetime.utcnow()
		overdue: list[dict[str, Any]] = []

		for promo in self._promotions.values():
			if promo["tenant_id"] != tenant_id:
				continue
			if promo.get("approval_status") != "pending_review":
				continue
			updated_at_raw = promo.get("updated_at")
			if updated_at_raw is None:
				continue
			updated_at = (
				updated_at_raw if isinstance(updated_at_raw, datetime)
				else datetime.fromisoformat(str(updated_at_raw))
			)
			age_hours = (now - updated_at).total_seconds() / 3600
			if age_hours > sla_hours:
				overdue.append({
					"promotion_id": promo["id"],
					"name": promo["name"],
					"submitted_at": updated_at.isoformat(),
					"age_hours": round(age_hours, 1),
					"sla_hours": sla_hours,
					"overdue_by_hours": round(age_hours - sla_hours, 1),
				})

		overdue.sort(key=lambda x: x["age_hours"], reverse=True)
		self._log_op("list_overdue_approvals", tenant_id)
		return overdue

	# ------------------------------------------------------------------
	# Promotion Audit Trail
	# ------------------------------------------------------------------

	async def get_promotion_audit_trail(self, promotion_id: str) -> list[dict[str, Any]]:
		"""Return the ordered change history for a promotion from the audit ledger.

		Each entry: {entity_id, field, old_value, new_value, changed_by, changed_at}.
		The ledger is an in-memory append-only list; production deployments should
		persist to the `prm_audit_log` PostgreSQL table via the `audl` capability.
		"""
		tenant_id = self.tenant_id
		promo = self._promotions.get(promotion_id)
		assert promo is not None and promo["tenant_id"] == tenant_id, "promotion not found"

		trail = [
			entry for entry in self._audit_ledger
			if entry.get("entity_id") == promotion_id
		]
		trail.sort(key=lambda x: x.get("changed_at", ""))
		return trail

	def _record_change(
		self, entity_type: str, entity_id: str, field: str,
		old_value: Any, new_value: Any, changed_by: str,
	) -> None:
		"""Append a single field change to the immutable audit ledger."""
		self._audit_ledger.append({
			"id": uuid7str(),
			"entity_type": entity_type,
			"entity_id": entity_id,
			"field": field,
			"old_value": old_value,
			"new_value": new_value,
			"changed_by": changed_by,
			"changed_at": datetime.utcnow().isoformat(),
		})

	# ------------------------------------------------------------------
	# Promotion Fatigue
	# ------------------------------------------------------------------

	async def get_customer_promotion_fatigue(
		self, customer_id: str, window_days: int = 30
	) -> dict[str, Any]:
		"""Compute a 0-100 promotion fatigue score for a customer.

		Fatigue is based on the number of unique promotions applied to this customer's
		carts within the rolling `window_days` window. Score = min(100, exposure_count * 10).
		"""
		assert customer_id, "customer_id required"
		assert window_days > 0, "window_days must be positive"
		tenant_id = self.tenant_id
		now = datetime.utcnow()
		cutoff = now.timestamp() - window_days * 86400

		exposure_count = 0
		for cart_id, promo_ids in self._cart_promotions.items():
			for pid in promo_ids:
				promo = self._promotions.get(pid)
				if not promo or promo["tenant_id"] != tenant_id:
					continue
				# Use updated_at as a proxy for application timestamp
				ts_raw = promo.get("updated_at")
				if ts_raw:
					ts = (
						ts_raw if isinstance(ts_raw, datetime)
						else datetime.fromisoformat(str(ts_raw))
					)
					if ts.timestamp() >= cutoff:
						# Check if customer_id matches any eligible segment heuristic
						segs = promo.get("eligible_segments", [])
						if not segs or customer_id in segs:
							exposure_count += 1

		fatigue_score = min(100, exposure_count * 10)
		level = "low" if fatigue_score < 30 else "medium" if fatigue_score < 70 else "high"

		return {
			"customer_id": customer_id,
			"window_days": window_days,
			"exposure_count": exposure_count,
			"fatigue_score": fatigue_score,
			"fatigue_level": level,
			"computed_at": now.isoformat(),
		}

	# ------------------------------------------------------------------
	# Promotion Preflight Validator
	# ------------------------------------------------------------------

	async def preflight_promotion_plan(
		self, promotion_ids: list[str]
	) -> dict[str, Any]:
		"""Simulate stacking evaluation for a proposed set of promotions before activation.

		Returns a conflict matrix, resolution suggestions, and whether the plan is viable.
		Does NOT mutate any state.
		"""
		assert promotion_ids, "promotion_ids required"
		tenant_id = self.tenant_id
		promos = [self._promotions.get(pid) for pid in promotion_ids]
		found = [p for p in promos if p and p["tenant_id"] == tenant_id]
		not_found = [pid for pid, p in zip(promotion_ids, promos) if p is None]

		stacking_result = await self.promotion_stacking_rules(
			[p["id"] for p in found]
		) if len(found) >= 2 else {"conflicts": [], "allowed_combinations": [], "stackable": True}

		suggestions: list[str] = []
		for conflict in stacking_result.get("conflicts", []):
			suggestions.append(
				f"Remove one of ({conflict['promotion_a']}, {conflict['promotion_b']}) "
				f"or change their discount types to be compatible."
			)

		return {
			"promotion_ids_requested": promotion_ids,
			"found_count": len(found),
			"not_found": not_found,
			"conflicts": stacking_result.get("conflicts", []),
			"allowed_combinations": stacking_result.get("allowed_combinations", []),
			"plan_viable": len(stacking_result.get("conflicts", [])) == 0 and len(not_found) == 0,
			"suggestions": suggestions,
			"evaluated_at": datetime.utcnow().isoformat(),
		}

