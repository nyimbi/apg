"""Crop Insurance service — agr_ins."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)
_CAPABILITY_ID = "agr_ins"


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _new_id(prefix: str = "") -> str:
	suffix = uuid4().hex[:12]
	return f"{prefix}-{suffix}" if prefix else suffix


class CropInsuranceService:
	"""Async service for crop insurance: parametric index products, satellite verification,
	weather trigger claims, and premium calculation."""

	def __init__(self, tenant_id: str = "default") -> None:
		if not tenant_id:
			raise ValueError("tenant_id required")
		self.tenant_id = tenant_id
		self._products: dict[str, dict[str, Any]] = {}
		self._policies: dict[str, dict[str, Any]] = {}
		self._claims: dict[str, dict[str, Any]] = {}
		self._audit: list[dict[str, Any]] = []

	def _emit(self, event_type: str, entity_type: str, entity_id: str, payload: dict[str, Any]) -> None:
		self._audit.append({
			"id": _new_id("evt"),
			"tenant_id": self.tenant_id,
			"event_type": event_type,
			"entity_type": entity_type,
			"entity_id": entity_id,
			"payload": payload,
			"occurred_at": _now(),
		})

	# ------------------------------------------------------------------ health

	async def health_check(self) -> dict[str, Any]:
		return {
			"status": "ok",
			"capability": _CAPABILITY_ID,
			"tenant_id": self.tenant_id,
			"counts": {
				"products": len(self._products),
				"policies": len(self._policies),
				"claims": len(self._claims),
			},
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": _CAPABILITY_ID,
			"name": "Crop Insurance",
			"domain": "agriculture",
			"version": "1.0.0",
			"description": "Parametric index products, satellite verification, weather trigger claims, premium calculation.",
		}

	async def get_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
		return self._audit[-limit:]

	# ------------------------------------------------------------------ products

	async def list_products(self, trigger_type: str | None = None, active: bool | None = None) -> list[dict[str, Any]]:
		items = list(self._products.values())
		if trigger_type:
			items = [p for p in items if p.get("trigger_type") == trigger_type]
		if active is not None:
			items = [p for p in items if p.get("active") == active]
		return items

	async def get_product(self, product_id: str) -> dict[str, Any]:
		if product_id not in self._products:
			raise KeyError(f"product_not_found:{product_id}")
		return self._products[product_id]

	async def create_product(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			pid = _new_id("prd")
			ts = _now()
			record: dict[str, Any] = {
				"id": pid,
				"tenant_id": self.tenant_id,
				"name": payload["name"],
				"trigger_type": payload["trigger_type"],
				"trigger_threshold": float(payload["trigger_threshold"]),
				"trigger_unit": payload["trigger_unit"],
				"payout_per_unit": float(payload["payout_per_unit"]),
				"max_payout": float(payload["max_payout"]),
				"coverage_period_months": int(payload["coverage_period_months"]),
				"eligible_crops": list(payload.get("eligible_crops", [])),
				"eligible_regions": list(payload.get("eligible_regions", [])),
				"base_premium_rate_pct": float(payload["base_premium_rate_pct"]),
				"notes": payload.get("notes"),
				"active": True,
				"created_at": ts,
				"updated_at": ts,
			}
			self._products[pid] = record
			self._emit("product.created", "insurance_product", pid, record)
			return record
		except Exception as exc:
			_log.error("create_product failed: %s", exc)
			raise

	async def update_product(self, product_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if product_id not in self._products:
				raise KeyError(f"product_not_found:{product_id}")
			record = self._products[product_id]
			for field in ["name", "trigger_threshold", "payout_per_unit", "max_payout",
						"base_premium_rate_pct", "notes", "active"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("product.updated", "insurance_product", product_id, payload)
			return record
		except Exception as exc:
			_log.error("update_product failed: %s", exc)
			raise

	async def delete_product(self, product_id: str) -> dict[str, Any]:
		try:
			if product_id not in self._products:
				raise KeyError(f"product_not_found:{product_id}")
			self._products.pop(product_id)
			self._emit("product.deleted", "insurance_product", product_id, {"id": product_id})
			return {"deleted": True, "id": product_id}
		except Exception as exc:
			_log.error("delete_product failed: %s", exc)
			raise

	# ------------------------------------------------------------------ premium calculation

	async def calculate_premium(self, product_id: str, farmer_id: str,
								sum_insured: float, risk_modifier: float = 1.0) -> dict[str, Any]:
		"""Calculate premium for a given product and sum insured."""
		if product_id not in self._products:
			raise KeyError(f"product_not_found:{product_id}")
		product = self._products[product_id]
		base_rate = product["base_premium_rate_pct"] / 100
		base_premium = round(sum_insured * base_rate, 2)
		# Risk adjustment: risk_modifier > 1 = higher risk region/crop
		final_premium = round(base_premium * risk_modifier, 2)
		final_rate = round(final_premium / sum_insured * 100, 4) if sum_insured > 0 else base_rate * 100
		return {
			"product_id": product_id,
			"farmer_id": farmer_id,
			"sum_insured": sum_insured,
			"base_premium": base_premium,
			"risk_adjustment": round(risk_modifier, 3),
			"final_premium": final_premium,
			"currency": "KES",
			"rate_pct": final_rate,
		}

	# ------------------------------------------------------------------ policies

	async def list_policies(self, farmer_id: str | None = None, status: str | None = None,
							season: str | None = None, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		items = list(self._policies.values())
		if farmer_id:
			items = [p for p in items if p.get("farmer_id") == farmer_id]
		if status:
			items = [p for p in items if p.get("status") == status]
		if season:
			items = [p for p in items if p.get("season") == season]
		return items[offset: offset + limit]

	async def get_policy(self, policy_id: str) -> dict[str, Any]:
		if policy_id not in self._policies:
			raise KeyError(f"policy_not_found:{policy_id}")
		return self._policies[policy_id]

	async def create_policy(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Issue a new insurance policy, computing premium automatically."""
		try:
			product_id = payload["product_id"]
			sum_insured = float(payload["sum_insured"])
			premium_calc = await self.calculate_premium(product_id, payload["farmer_id"], sum_insured)
			pol_id = _new_id("pol")
			ts = _now()
			record: dict[str, Any] = {
				"id": pol_id,
				"tenant_id": self.tenant_id,
				"farmer_id": payload["farmer_id"],
				"product_id": product_id,
				"crop_id": payload["crop_id"],
				"farm_parcel_id": payload["farm_parcel_id"],
				"sum_insured": sum_insured,
				"premium_amount": premium_calc["final_premium"],
				"currency": payload.get("currency", "KES"),
				"coverage_start": payload["coverage_start"],
				"coverage_end": payload["coverage_end"],
				"season": payload["season"],
				"status": "quoted",
				"premium_paid_at": None,
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._policies[pol_id] = record
			self._emit("policy.created", "policy", pol_id, record)
			return record
		except Exception as exc:
			_log.error("create_policy failed: %s", exc)
			raise

	async def update_policy(self, policy_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if policy_id not in self._policies:
				raise KeyError(f"policy_not_found:{policy_id}")
			record = self._policies[policy_id]
			for field in ["status", "premium_paid_at", "notes"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			# Activate on premium payment
			if payload.get("premium_paid_at") and record.get("status") == "quoted":
				record["status"] = "active"
			record["updated_at"] = _now()
			self._emit("policy.updated", "policy", policy_id, payload)
			return record
		except Exception as exc:
			_log.error("update_policy failed: %s", exc)
			raise

	async def delete_policy(self, policy_id: str) -> dict[str, Any]:
		try:
			if policy_id not in self._policies:
				raise KeyError(f"policy_not_found:{policy_id}")
			self._policies.pop(policy_id)
			self._emit("policy.deleted", "policy", policy_id, {"id": policy_id})
			return {"deleted": True, "id": policy_id}
		except Exception as exc:
			_log.error("delete_policy failed: %s", exc)
			raise

	async def activate_policy(self, policy_id: str, payment_reference: str) -> dict[str, Any]:
		"""Mark premium as paid and activate the policy."""
		try:
			if policy_id not in self._policies:
				raise KeyError(f"policy_not_found:{policy_id}")
			self._policies[policy_id]["status"] = "active"
			self._policies[policy_id]["premium_paid_at"] = _now()
			self._policies[policy_id]["payment_reference"] = payment_reference
			self._policies[policy_id]["updated_at"] = _now()
			self._emit("policy.activated", "policy", policy_id, {"payment_reference": payment_reference})
			return self._policies[policy_id]
		except Exception as exc:
			_log.error("activate_policy failed: %s", exc)
			raise

	# ------------------------------------------------------------------ claims

	async def list_claims(self, policy_id: str | None = None, farmer_id: str | None = None,
						status: str | None = None, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		items = list(self._claims.values())
		if policy_id:
			items = [c for c in items if c.get("policy_id") == policy_id]
		if farmer_id:
			items = [c for c in items if c.get("farmer_id") == farmer_id]
		if status:
			items = [c for c in items if c.get("status") == status]
		return items[offset: offset + limit]

	async def get_claim(self, claim_id: str) -> dict[str, Any]:
		if claim_id not in self._claims:
			raise KeyError(f"claim_not_found:{claim_id}")
		return self._claims[claim_id]

	async def submit_claim(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Submit a parametric trigger claim."""
		try:
			policy_id = payload["policy_id"]
			if policy_id not in self._policies:
				raise KeyError(f"policy_not_found:{policy_id}")
			policy = self._policies[policy_id]
			if policy.get("status") != "active":
				raise ValueError("policy_not_active")
			claim_id = _new_id("clm")
			ts = _now()
			record: dict[str, Any] = {
				"id": claim_id,
				"tenant_id": self.tenant_id,
				"policy_id": policy_id,
				"farmer_id": policy["farmer_id"],
				"trigger_event": payload["trigger_event"],
				"trigger_value": float(payload["trigger_value"]),
				"observed_at": payload["observed_at"],
				"evidence_source": payload["evidence_source"],
				"evidence_reference": payload.get("evidence_reference"),
				"status": "submitted",
				"verified_trigger_value": None,
				"approved_payout": None,
				"rejection_reason": None,
				"paid_at": None,
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._claims[claim_id] = record
			self._emit("claim.submitted", "claim", claim_id, record)
			return record
		except Exception as exc:
			_log.error("submit_claim failed: %s", exc)
			raise

	async def update_claim(self, claim_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if claim_id not in self._claims:
				raise KeyError(f"claim_not_found:{claim_id}")
			record = self._claims[claim_id]
			for field in ["status", "verified_trigger_value", "approved_payout",
						"rejection_reason", "paid_at", "notes"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			# Mark policy as claimed when claim is paid
			if record.get("status") == "paid":
				policy_id = record.get("policy_id")
				if policy_id and policy_id in self._policies:
					self._policies[policy_id]["status"] = "claimed"
			self._emit("claim.updated", "claim", claim_id, payload)
			return record
		except Exception as exc:
			_log.error("update_claim failed: %s", exc)
			raise

	async def verify_trigger(self, claim_id: str, verified_value: float, source: str) -> dict[str, Any]:
		"""Verify trigger event against satellite/weather station data."""
		try:
			if claim_id not in self._claims:
				raise KeyError(f"claim_not_found:{claim_id}")
			claim = self._claims[claim_id]
			policy = self._policies.get(claim.get("policy_id", ""))
			if not policy:
				raise KeyError("policy_not_found_for_claim")
			product = self._products.get(policy.get("product_id", ""))
			if not product:
				raise KeyError("product_not_found_for_policy")

			claim["verified_trigger_value"] = verified_value
			claim["verification_source"] = source
			threshold = product["trigger_threshold"]
			trigger_type = product["trigger_type"]

			# Determine if trigger condition is met
			triggered = False
			if "deficit" in trigger_type or "decline" in trigger_type:
				triggered = verified_value < threshold
			else:
				triggered = verified_value > threshold

			if triggered:
				deficit = abs(verified_value - threshold)
				raw_payout = round(deficit * product["payout_per_unit"], 2)
				payout = min(raw_payout, product["max_payout"], policy["sum_insured"])
				claim["approved_payout"] = payout
				claim["status"] = "approved"
			else:
				claim["status"] = "rejected"
				claim["rejection_reason"] = f"trigger_not_met:verified={verified_value},threshold={threshold}"

			claim["updated_at"] = _now()
			self._emit("claim.verified", "claim", claim_id, {"triggered": triggered, "verified_value": verified_value})
			return claim
		except Exception as exc:
			_log.error("verify_trigger failed: %s", exc)
			raise

	async def get_portfolio_stats(self) -> dict[str, Any]:
		"""Insurance portfolio statistics."""
		policies = list(self._policies.values())
		claims = list(self._claims.values())
		active_policies = [p for p in policies if p.get("status") == "active"]
		total_sum_insured = sum(p.get("sum_insured", 0) for p in active_policies)
		total_premiums = sum(p.get("premium_amount", 0) for p in policies if p.get("premium_paid_at"))
		total_paid = sum(c.get("approved_payout", 0) for c in claims if c.get("status") == "paid")
		return {
			"total_policies": len(policies),
			"active_policies": len(active_policies),
			"total_sum_insured": round(total_sum_insured, 2),
			"total_premiums_collected": round(total_premiums, 2),
			"total_claims": len(claims),
			"paid_claims": len([c for c in claims if c.get("status") == "paid"]),
			"total_claims_paid": round(total_paid, 2),
			"loss_ratio_pct": round(total_paid / total_premiums * 100, 2) if total_premiums > 0 else 0,
		}

	async def get_farmer_coverage(self, farmer_id: str) -> dict[str, Any]:
		"""Return all active coverage for a farmer."""
		policies = [p for p in self._policies.values()
				if p.get("farmer_id") == farmer_id and p.get("status") == "active"]
		claims = [c for c in self._claims.values() if c.get("farmer_id") == farmer_id]
		return {
			"farmer_id": farmer_id,
			"active_policies": len(policies),
			"total_sum_insured": round(sum(p.get("sum_insured", 0) for p in policies), 2),
			"pending_claims": len([c for c in claims if c.get("status") in ("submitted", "under_review")]),
			"approved_claims": len([c for c in claims if c.get("status") == "approved"]),
			"policies": [{"id": p["id"], "product_id": p["product_id"], "sum_insured": p["sum_insured"], "coverage_end": p["coverage_end"]} for p in policies],
		}
