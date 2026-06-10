"""Guest Loyalty Programme service — points accrual, tier management, redemption, partner rewards, recognition preferences."""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)


def _uid() -> str:
	return uuid4().hex[:12]


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


# Default tier thresholds
_TIER_THRESHOLDS = {
	"bronze":   {"min_points": 0,      "min_nights": 0,  "multiplier": 1.0},
	"silver":   {"min_points": 5000,   "min_nights": 10, "multiplier": 1.25},
	"gold":     {"min_points": 15000,  "min_nights": 25, "multiplier": 1.50},
	"platinum": {"min_points": 50000,  "min_nights": 50, "multiplier": 2.00},
}

# Points earned per KES spent by tier
_EARN_RATE = {
	"bronze": 1.0,
	"silver": 1.25,
	"gold":   1.50,
	"platinum": 2.0,
}

_MEMBER_NUMBER_PREFIX = "HOT"


class LOYService:
	"""Guest Loyalty Programme service."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.members: dict[str, dict[str, Any]] = {}
		self.transactions: dict[str, dict[str, Any]] = {}
		self.tier_rules: dict[str, dict[str, Any]] = {}
		self.partners: dict[str, dict[str, Any]] = {}
		self.partner_transactions: dict[str, dict[str, Any]] = {}
		self.redemptions: dict[str, dict[str, Any]] = {}
		self.recognition_preferences: dict[str, dict[str, Any]] = {}
		self.bonus_campaigns: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []
		self._member_seq = 100000

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _emit(self, tenant_id: str, event_type: str, record_id: str, record_type: str, details: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": _uid(),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record_id,
			"record_type": record_type,
			"details": details or {},
			"created_at": _now(),
		})

	def _next_member_number(self) -> str:
		self._member_seq += 1
		return f"{_MEMBER_NUMBER_PREFIX}{self._member_seq}"

	def _compute_tier(self, lifetime_points: int, tier_qualifying_nights: int) -> str:
		"""Determine tier based on lifetime points and qualifying nights."""
		if lifetime_points >= _TIER_THRESHOLDS["platinum"]["min_points"] or tier_qualifying_nights >= _TIER_THRESHOLDS["platinum"]["min_nights"]:
			return "platinum"
		if lifetime_points >= _TIER_THRESHOLDS["gold"]["min_points"] or tier_qualifying_nights >= _TIER_THRESHOLDS["gold"]["min_nights"]:
			return "gold"
		if lifetime_points >= _TIER_THRESHOLDS["silver"]["min_points"] or tier_qualifying_nights >= _TIER_THRESHOLDS["silver"]["min_nights"]:
			return "silver"
		return "bronze"

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "hos_loy",
			"status": "healthy",
			"total_members": len(self.members),
			"active_members": sum(1 for m in self.members.values() if m["status"] == "active"),
			"checked_at": _now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": "hos_loy",
			"name": "Guest Loyalty Programme",
			"domain": "hospitality",
			"version": "1.0.0",
			"description": "Points accrual, tier management, redemption, partner rewards, recognition preferences",
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── Members ───────────────────────────────────────────────────────────────

	async def list_members(self, tenant_id: str | None = None, tier: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		members = [deepcopy(m) for m in self.members.values() if m["tenant_id"] == tenant]
		if tier:
			members = [m for m in members if m["tier"] == tier]
		if status:
			members = [m for m in members if m["status"] == status]
		return members

	async def get_member(self, member_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		member = self.members.get(member_id)
		if not member or member["tenant_id"] != tenant:
			raise KeyError(f"loyalty_member_not_found:{member_id}")
		return deepcopy(member)

	async def get_member_by_email(self, email: str, tenant_id: str | None = None) -> dict[str, Any] | None:
		tenant = self._tenant(tenant_id)
		for m in self.members.values():
			if m["tenant_id"] == tenant and m["email"] == email and m["status"] == "active":
				return deepcopy(m)
		return None

	async def enroll_member(self, guest_id: str, first_name: str, last_name: str, email: str,
	                         phone: str | None = None, enrollment_source: str = "front_desk",
	                         tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		# Prevent duplicate enrollments
		existing = await self.get_member_by_email(email, tenant_id=tenant)
		if existing:
			raise ValueError(f"member_already_enrolled:{email}")
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"guest_id": guest_id,
			"membership_number": self._next_member_number(),
			"first_name": first_name,
			"last_name": last_name,
			"email": email,
			"phone": phone,
			"tier": "bronze",
			"points_balance": 0,
			"lifetime_points": 0,
			"lifetime_spend": 0.0,
			"tier_qualifying_nights": 0,
			"tier_qualifying_spend": 0.0,
			"enrollment_source": enrollment_source,
			"preferences": {},
			"status": "active",
			"created_at": _now(),
		}
		# Award enrollment bonus points
		record["points_balance"] = 500
		record["lifetime_points"] = 500
		self.members[record["id"]] = record
		txn: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"member_id": record["id"],
			"transaction_type": "bonus",
			"points": 500,
			"running_balance": 500,
			"description": "Enrollment bonus",
			"reference_id": None,
			"spend_amount": 0.0,
			"created_at": _now(),
		}
		self.transactions[txn["id"]] = txn
		self._emit(tenant, "member_enrolled", record["id"], "loyalty_member", {"email": email})
		return deepcopy(record)

	async def update_member(self, member_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		member = self.members.get(member_id)
		if not member or member["tenant_id"] != tenant:
			raise KeyError(f"loyalty_member_not_found:{member_id}")
		allowed = {"phone", "email", "preferences", "status"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				member[k] = v
		self._emit(tenant, "member_updated", member_id, "loyalty_member")
		return deepcopy(member)

	async def delete_member(self, member_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		member = self.members.get(member_id)
		if not member or member["tenant_id"] != tenant:
			raise KeyError(f"loyalty_member_not_found:{member_id}")
		member["status"] = "deactivated"
		self._emit(tenant, "member_deactivated", member_id, "loyalty_member")
		return {"deactivated": True, "member_id": member_id}

	# ── Points ────────────────────────────────────────────────────────────────

	async def list_transactions(self, member_id: str, tenant_id: str | None = None, transaction_type: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		txns = [deepcopy(t) for t in self.transactions.values() if t["tenant_id"] == tenant and t["member_id"] == member_id]
		if transaction_type:
			txns = [t for t in txns if t["transaction_type"] == transaction_type]
		return sorted(txns, key=lambda x: x["created_at"], reverse=True)

	async def earn_points(self, member_id: str, spend_amount: float, description: str,
	                       reference_id: str | None = None, nights: int = 0,
	                       tenant_id: str | None = None) -> dict[str, Any]:
		"""Accrue points for a qualifying spend."""
		tenant = self._tenant(tenant_id)
		member = self.members.get(member_id)
		if not member or member["tenant_id"] != tenant:
			raise KeyError(f"loyalty_member_not_found:{member_id}")
		# Calculate points: earn_rate * spend_amount
		earn_rate = _EARN_RATE.get(member["tier"], 1.0)
		# Check for active bonus campaigns
		for campaign in self.bonus_campaigns.values():
			if campaign["tenant_id"] == tenant and campaign["status"] == "active":
				earn_rate *= campaign.get("multiplier", 1.0)
		points_earned = int(spend_amount * earn_rate)
		member["points_balance"] += points_earned
		member["lifetime_points"] += points_earned
		member["lifetime_spend"] += spend_amount
		member["tier_qualifying_spend"] += spend_amount
		if nights > 0:
			member["tier_qualifying_nights"] += nights
		# Re-evaluate tier
		new_tier = self._compute_tier(member["lifetime_points"], member["tier_qualifying_nights"])
		if new_tier != member["tier"]:
			old_tier = member["tier"]
			member["tier"] = new_tier
			self._emit(tenant, "tier_upgraded", member_id, "loyalty_member", {"from": old_tier, "to": new_tier})
		txn: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"member_id": member_id,
			"transaction_type": "earn",
			"points": points_earned,
			"running_balance": member["points_balance"],
			"description": description,
			"reference_id": reference_id,
			"spend_amount": spend_amount,
			"nights": nights,
			"created_at": _now(),
		}
		self.transactions[txn["id"]] = txn
		self._emit(tenant, "points_earned", txn["id"], "points_transaction", {"points": points_earned, "balance": member["points_balance"]})
		return deepcopy(txn)

	async def redeem_points(self, member_id: str, points: int, description: str,
	                         reference_id: str | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		"""Redeem points against a charge."""
		tenant = self._tenant(tenant_id)
		member = self.members.get(member_id)
		if not member or member["tenant_id"] != tenant:
			raise KeyError(f"loyalty_member_not_found:{member_id}")
		if member["points_balance"] < points:
			raise ValueError(f"insufficient_points:{member['points_balance']}<{points}")
		member["points_balance"] -= points
		txn: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"member_id": member_id,
			"transaction_type": "redeem",
			"points": -points,
			"running_balance": member["points_balance"],
			"description": description,
			"reference_id": reference_id,
			"spend_amount": 0.0,
			"cash_value": round(points * 0.05, 2),  # 1 point = 0.05 KES
			"created_at": _now(),
		}
		self.transactions[txn["id"]] = txn
		self.redemptions[txn["id"]] = txn
		self._emit(tenant, "points_redeemed", txn["id"], "points_transaction", {"points": points})
		return deepcopy(txn)

	async def adjust_points(self, member_id: str, points_delta: int, reason: str, adjusted_by: str,
	                         tenant_id: str | None = None) -> dict[str, Any]:
		"""Manual point adjustment (positive or negative)."""
		tenant = self._tenant(tenant_id)
		member = self.members.get(member_id)
		if not member or member["tenant_id"] != tenant:
			raise KeyError(f"loyalty_member_not_found:{member_id}")
		member["points_balance"] = max(0, member["points_balance"] + points_delta)
		if points_delta > 0:
			member["lifetime_points"] += points_delta
		txn: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"member_id": member_id,
			"transaction_type": "adjust",
			"points": points_delta,
			"running_balance": member["points_balance"],
			"description": reason,
			"reference_id": None,
			"adjusted_by": adjusted_by,
			"spend_amount": 0.0,
			"created_at": _now(),
		}
		self.transactions[txn["id"]] = txn
		self._emit(tenant, "points_adjusted", txn["id"], "points_transaction", {"delta": points_delta, "reason": reason})
		return deepcopy(txn)

	async def expire_points(self, member_id: str, points: int, reason: str = "points_expiry", tenant_id: str | None = None) -> dict[str, Any]:
		"""Expire a block of points."""
		tenant = self._tenant(tenant_id)
		member = self.members.get(member_id)
		if not member or member["tenant_id"] != tenant:
			raise KeyError(f"loyalty_member_not_found:{member_id}")
		expired = min(points, member["points_balance"])
		member["points_balance"] -= expired
		txn: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"member_id": member_id,
			"transaction_type": "expire",
			"points": -expired,
			"running_balance": member["points_balance"],
			"description": reason,
			"reference_id": None,
			"spend_amount": 0.0,
			"created_at": _now(),
		}
		self.transactions[txn["id"]] = txn
		self._emit(tenant, "points_expired", txn["id"], "points_transaction", {"expired": expired})
		return deepcopy(txn)

	# ── Tier Management ───────────────────────────────────────────────────────

	async def create_tier_rule(self, tier: str, min_points: int, min_lifetime_spend: float,
	                            min_nights: int, benefits: list[str], points_multiplier: float,
	                            base_earn_rate: float, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"tier": tier,
			"min_points": min_points,
			"min_lifetime_spend": min_lifetime_spend,
			"min_nights": min_nights,
			"benefits": benefits,
			"points_multiplier": points_multiplier,
			"base_earn_rate": base_earn_rate,
			"created_at": _now(),
		}
		self.tier_rules[record["id"]] = record
		self._emit(tenant, "tier_rule_created", record["id"], "tier_rule")
		return deepcopy(record)

	async def list_tier_rules(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(r) for r in self.tier_rules.values() if r["tenant_id"] == tenant]

	async def force_tier_upgrade(self, member_id: str, new_tier: str, reason: str, upgraded_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Manually override a member's tier."""
		tenant = self._tenant(tenant_id)
		member = self.members.get(member_id)
		if not member or member["tenant_id"] != tenant:
			raise KeyError(f"loyalty_member_not_found:{member_id}")
		old_tier = member["tier"]
		member["tier"] = new_tier
		member["tier_override"] = True
		member["tier_override_reason"] = reason
		self._emit(tenant, "tier_manually_upgraded", member_id, "loyalty_member", {"from": old_tier, "to": new_tier, "by": upgraded_by})
		return deepcopy(member)

	# ── Partners ──────────────────────────────────────────────────────────────

	async def list_partners(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(p) for p in self.partners.values() if p["tenant_id"] == tenant and p["is_active"]]

	async def create_partner(self, partner_name: str, partner_type: str, earn_rate: float = 1.0,
	                          redeem_rate: float = 1.0, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"partner_name": partner_name,
			"partner_type": partner_type,
			"earn_rate": earn_rate,
			"redeem_rate": redeem_rate,
			"is_active": True,
			"created_at": _now(),
		}
		self.partners[record["id"]] = record
		self._emit(tenant, "partner_created", record["id"], "loyalty_partner")
		return deepcopy(record)

	async def update_partner(self, partner_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		partner = self.partners.get(partner_id)
		if not partner or partner["tenant_id"] != tenant:
			raise KeyError(f"partner_not_found:{partner_id}")
		for k, v in updates.items():
			if v is not None:
				partner[k] = v
		return deepcopy(partner)

	async def delete_partner(self, partner_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		partner = self.partners.get(partner_id)
		if not partner or partner["tenant_id"] != tenant:
			raise KeyError(f"partner_not_found:{partner_id}")
		partner["is_active"] = False
		return {"deactivated": True, "partner_id": partner_id}

	async def earn_partner_points(self, member_id: str, partner_id: str, partner_spend: float,
	                               description: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Earn points from a partner transaction."""
		tenant = self._tenant(tenant_id)
		partner = self.partners.get(partner_id)
		if not partner or partner["tenant_id"] != tenant or not partner["is_active"]:
			raise KeyError(f"partner_not_found_or_inactive:{partner_id}")
		points = int(partner_spend * partner["earn_rate"])
		txn = await self.earn_points(member_id, partner_spend, f"Partner: {partner['partner_name']} - {description}", tenant_id=tenant)
		txn["partner_id"] = partner_id
		self.partner_transactions[txn["id"]] = txn
		return txn

	# ── Recognition Preferences ───────────────────────────────────────────────

	async def set_recognition_preferences(self, member_id: str, preferences: dict[str, Any],
	                                       tenant_id: str | None = None) -> dict[str, Any]:
		"""Store guest recognition preferences (pillow type, newspaper, allergies, etc.)."""
		tenant = self._tenant(tenant_id)
		member = self.members.get(member_id)
		if not member or member["tenant_id"] != tenant:
			raise KeyError(f"loyalty_member_not_found:{member_id}")
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"member_id": member_id,
			"preferences": deepcopy(preferences),
			"updated_at": _now(),
		}
		self.recognition_preferences[member_id] = record
		member["preferences"] = deepcopy(preferences)
		self._emit(tenant, "recognition_preferences_updated", member_id, "loyalty_member")
		return deepcopy(record)

	async def get_recognition_preferences(self, member_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		prefs = self.recognition_preferences.get(member_id)
		if not prefs or prefs["tenant_id"] != tenant:
			return {"member_id": member_id, "preferences": {}}
		return deepcopy(prefs)

	# ── Bonus Campaigns ───────────────────────────────────────────────────────

	async def create_bonus_campaign(self, name: str, date_from: str, date_to: str, multiplier: float,
	                                 description: str | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"name": name,
			"date_from": date_from,
			"date_to": date_to,
			"multiplier": multiplier,
			"description": description,
			"status": "active",
			"created_at": _now(),
		}
		self.bonus_campaigns[record["id"]] = record
		self._emit(tenant, "bonus_campaign_created", record["id"], "bonus_campaign")
		return deepcopy(record)

	async def list_bonus_campaigns(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(c) for c in self.bonus_campaigns.values() if c["tenant_id"] == tenant]

	# ── Analytics ─────────────────────────────────────────────────────────────

	async def tier_distribution(self, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		members = [m for m in self.members.values() if m["tenant_id"] == tenant and m["status"] == "active"]
		by_tier: dict[str, int] = {"bronze": 0, "silver": 0, "gold": 0, "platinum": 0}
		for m in members:
			by_tier[m["tier"]] = by_tier.get(m["tier"], 0) + 1
		return {
			"tenant_id": tenant,
			"total_active_members": len(members),
			"by_tier": by_tier,
			"total_points_outstanding": sum(m["points_balance"] for m in members),
			"generated_at": _now(),
		}

	async def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		members = [m for m in self.members.values() if m["tenant_id"] == tenant]
		active = [m for m in members if m["status"] == "active"]
		return {
			"tenant_id": tenant,
			"total_members": len(members),
			"active_members": len(active),
			"total_points_outstanding": sum(m["points_balance"] for m in active),
			"total_lifetime_spend": sum(m["lifetime_spend"] for m in active),
			"platinum_members": sum(1 for m in active if m["tier"] == "platinum"),
			"gold_members": sum(1 for m in active if m["tier"] == "gold"),
			"silver_members": sum(1 for m in active if m["tier"] == "silver"),
			"bronze_members": sum(1 for m in active if m["tier"] == "bronze"),
			"active_partners": sum(1 for p in self.partners.values() if p["tenant_id"] == tenant and p["is_active"]),
			"generated_at": _now(),
		}
