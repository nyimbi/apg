"""Agricultural Credit Scoring service — agr_crd."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

_log = logging.getLogger(__name__)
_CAPABILITY_ID = "agr_crd"


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _new_id(prefix: str = "") -> str:
	suffix = uuid4().hex[:12]
	return f"{prefix}-{suffix}" if prefix else suffix


def _score_to_rating(score: float) -> str:
	if score >= 90: return "AAA"
	if score >= 80: return "AA"
	if score >= 70: return "A"
	if score >= 60: return "BBB"
	if score >= 50: return "BB"
	if score >= 40: return "B"
	if score >= 30: return "CCC"
	return "D"


class AgriCreditService:
	"""Async service for agricultural credit: yield-based scoring, seasonal loans,
	group lending, and collateral registry."""

	def __init__(self, tenant_id: str = "default") -> None:
		if not tenant_id:
			raise ValueError("tenant_id required")
		self.tenant_id = tenant_id
		self._profiles: dict[str, dict[str, Any]] = {}
		self._loans: dict[str, dict[str, Any]] = {}
		self._collateral: dict[str, dict[str, Any]] = {}
		self._groups: dict[str, dict[str, Any]] = {}
		self._repayments: dict[str, list[dict[str, Any]]] = {}
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
				"credit_profiles": len(self._profiles),
				"loan_applications": len(self._loans),
				"collateral_items": len(self._collateral),
				"group_loans": len(self._groups),
			},
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": _CAPABILITY_ID,
			"name": "Agricultural Credit Scoring",
			"domain": "agriculture",
			"version": "1.0.0",
			"description": "Yield-based credit scoring, seasonal loan products, group lending, collateral registry.",
		}

	async def get_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
		return self._audit[-limit:]

	# ------------------------------------------------------------------ credit profiles

	async def list_profiles(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		return list(self._profiles.values())[offset: offset + limit]

	async def get_profile(self, profile_id: str) -> dict[str, Any]:
		if profile_id not in self._profiles:
			raise KeyError(f"credit_profile_not_found:{profile_id}")
		return self._profiles[profile_id]

	async def get_profile_by_farmer(self, farmer_id: str) -> dict[str, Any] | None:
		for p in self._profiles.values():
			if p.get("farmer_id") == farmer_id:
				return p
		return None

	async def create_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			pid = _new_id("cpr")
			ts = _now()
			record: dict[str, Any] = {
				"id": pid,
				"tenant_id": self.tenant_id,
				"farmer_id": payload["farmer_id"],
				"farm_parcel_ids": list(payload.get("farm_parcel_ids", [])),
				"years_farming": payload.get("years_farming"),
				"crop_types": list(payload.get("crop_types", [])),
				"avg_annual_yield_kg": payload.get("avg_annual_yield_kg"),
				"avg_annual_revenue": payload.get("avg_annual_revenue"),
				"mobile_money_account": payload.get("mobile_money_account"),
				"cooperative_member": bool(payload.get("cooperative_member", False)),
				"cooperative_id": payload.get("cooperative_id"),
				"credit_score": None,
				"rating": None,
				"last_scored_at": None,
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._profiles[pid] = record
			self._emit("profile.created", "credit_profile", pid, record)
			return record
		except Exception as exc:
			_log.error("create_profile failed: %s", exc)
			raise

	async def update_profile(self, profile_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if profile_id not in self._profiles:
				raise KeyError(f"credit_profile_not_found:{profile_id}")
			record = self._profiles[profile_id]
			for field in ["years_farming", "crop_types", "avg_annual_yield_kg",
						"avg_annual_revenue", "mobile_money_account", "cooperative_member",
						"cooperative_id", "notes"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("profile.updated", "credit_profile", profile_id, payload)
			return record
		except Exception as exc:
			_log.error("update_profile failed: %s", exc)
			raise

	async def delete_profile(self, profile_id: str) -> dict[str, Any]:
		try:
			if profile_id not in self._profiles:
				raise KeyError(f"credit_profile_not_found:{profile_id}")
			self._profiles.pop(profile_id)
			self._emit("profile.deleted", "credit_profile", profile_id, {"id": profile_id})
			return {"deleted": True, "id": profile_id}
		except Exception as exc:
			_log.error("delete_profile failed: %s", exc)
			raise

	# ------------------------------------------------------------------ scoring

	async def score_farmer(self, farmer_id: str) -> dict[str, Any]:
		"""Compute yield-based credit score for a farmer."""
		try:
			profile = None
			for p in self._profiles.values():
				if p.get("farmer_id") == farmer_id:
					profile = p
					break
			if not profile:
				raise KeyError(f"credit_profile_not_found_for_farmer:{farmer_id}")

			factors: dict[str, float] = {}

			# Factor 1: farming experience (0-25 pts)
			years = profile.get("years_farming") or 0
			factors["experience"] = min(25, years * 2.5)

			# Factor 2: yield consistency (0-25 pts)
			avg_yield = profile.get("avg_annual_yield_kg") or 0
			factors["yield_level"] = min(25, avg_yield / 500)

			# Factor 3: revenue (0-20 pts)
			revenue = profile.get("avg_annual_revenue") or 0
			factors["revenue"] = min(20, revenue / 10000)

			# Factor 4: crop diversity (0-10 pts)
			crop_count = len(profile.get("crop_types", []))
			factors["crop_diversity"] = min(10, crop_count * 3)

			# Factor 5: mobile money (5 pts)
			factors["mobile_money"] = 5.0 if profile.get("mobile_money_account") else 0.0

			# Factor 6: cooperative membership (10 pts)
			factors["cooperative"] = 10.0 if profile.get("cooperative_member") else 0.0

			# Factor 7: repayment history (0-5 pts)
			farmer_loans = [l for l in self._loans.values() if l.get("farmer_id") == farmer_id]
			settled = len([l for l in farmer_loans if l.get("status") == "settled"])
			defaulted = len([l for l in farmer_loans if l.get("status") == "defaulted"])
			if farmer_loans:
				repay_rate = settled / (settled + defaulted) if (settled + defaulted) > 0 else 0.5
				factors["repayment_history"] = round(repay_rate * 5, 2)
			else:
				factors["repayment_history"] = 2.5  # neutral for new borrowers

			score = min(100, round(sum(factors.values()), 2))
			rating = _score_to_rating(score)
			max_loan = round(score * 500, 2)  # KES 500 per score point
			rate = max(8.0, 25.0 - score * 0.15)  # 8-25% interest

			ts = _now()
			profile["credit_score"] = score
			profile["rating"] = rating
			profile["last_scored_at"] = ts

			result = {
				"farmer_id": farmer_id,
				"credit_score": score,
				"rating": rating,
				"max_loan_amount": max_loan,
				"recommended_rate_pct": round(rate, 2),
				"factors": {k: round(v, 2) for k, v in factors.items()},
				"scored_at": ts,
			}
			self._emit("farmer.scored", "credit_score", farmer_id, result)
			return result
		except Exception as exc:
			_log.error("score_farmer failed: %s", exc)
			raise

	# ------------------------------------------------------------------ loans

	async def list_loans(self, farmer_id: str | None = None, status: str | None = None,
						limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		items = list(self._loans.values())
		if farmer_id:
			items = [l for l in items if l.get("farmer_id") == farmer_id]
		if status:
			items = [l for l in items if l.get("status") == status]
		return items[offset: offset + limit]

	async def get_loan(self, loan_id: str) -> dict[str, Any]:
		if loan_id not in self._loans:
			raise KeyError(f"loan_not_found:{loan_id}")
		return self._loans[loan_id]

	async def apply_for_loan(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			lid = _new_id("lon")
			ts = _now()
			record: dict[str, Any] = {
				"id": lid,
				"tenant_id": self.tenant_id,
				"farmer_id": payload["farmer_id"],
				"amount": float(payload["amount"]),
				"currency": payload.get("currency", "KES"),
				"purpose": payload["purpose"],
				"season": payload["season"],
				"duration_months": int(payload["duration_months"]),
				"collateral_description": payload.get("collateral_description"),
				"guarantor_id": payload.get("guarantor_id"),
				"group_id": payload.get("group_id"),
				"status": "applied",
				"credit_score": None,
				"approved_amount": None,
				"interest_rate_pct": None,
				"disbursed_at": None,
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._loans[lid] = record
			self._emit("loan.applied", "loan_application", lid, record)
			return record
		except Exception as exc:
			_log.error("apply_for_loan failed: %s", exc)
			raise

	async def update_loan(self, loan_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if loan_id not in self._loans:
				raise KeyError(f"loan_not_found:{loan_id}")
			record = self._loans[loan_id]
			for field in ["status", "approved_amount", "interest_rate_pct", "disbursed_at", "notes"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("loan.updated", "loan_application", loan_id, payload)
			return record
		except Exception as exc:
			_log.error("update_loan failed: %s", exc)
			raise

	async def delete_loan(self, loan_id: str) -> dict[str, Any]:
		try:
			if loan_id not in self._loans:
				raise KeyError(f"loan_not_found:{loan_id}")
			self._loans.pop(loan_id)
			self._emit("loan.deleted", "loan_application", loan_id, {"id": loan_id})
			return {"deleted": True, "id": loan_id}
		except Exception as exc:
			_log.error("delete_loan failed: %s", exc)
			raise

	async def record_repayment(self, loan_id: str, amount: float) -> dict[str, Any]:
		"""Record a loan repayment and check if fully settled."""
		try:
			if loan_id not in self._loans:
				raise KeyError(f"loan_not_found:{loan_id}")
			if loan_id not in self._repayments:
				self._repayments[loan_id] = []
			repayment = {"amount": amount, "paid_at": _now()}
			self._repayments[loan_id].append(repayment)
			total_paid = sum(r["amount"] for r in self._repayments[loan_id])
			approved = self._loans[loan_id].get("approved_amount") or self._loans[loan_id]["amount"]
			if total_paid >= approved:
				self._loans[loan_id]["status"] = "settled"
			else:
				self._loans[loan_id]["status"] = "repaying"
			self._loans[loan_id]["updated_at"] = _now()
			self._emit("loan.repayment", "loan_application", loan_id, {"amount": amount, "total_paid": total_paid})
			return {"loan_id": loan_id, "amount_paid": amount, "total_paid": total_paid, "status": self._loans[loan_id]["status"]}
		except Exception as exc:
			_log.error("record_repayment failed: %s", exc)
			raise

	# ------------------------------------------------------------------ collateral

	async def list_collateral(self, farmer_id: str | None = None) -> list[dict[str, Any]]:
		items = list(self._collateral.values())
		if farmer_id:
			items = [c for c in items if c.get("farmer_id") == farmer_id]
		return items

	async def create_collateral(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			col_id = _new_id("col")
			ts = _now()
			record: dict[str, Any] = {
				"id": col_id,
				"tenant_id": self.tenant_id,
				"farmer_id": payload["farmer_id"],
				"description": payload["description"],
				"estimated_value": float(payload["estimated_value"]),
				"currency": payload.get("currency", "KES"),
				"asset_type": payload["asset_type"],
				"reference_number": payload.get("reference_number"),
				"pledged_to_loan": None,
				"notes": payload.get("notes"),
				"created_at": ts,
			}
			self._collateral[col_id] = record
			self._emit("collateral.created", "collateral", col_id, record)
			return record
		except Exception as exc:
			_log.error("create_collateral failed: %s", exc)
			raise

	async def delete_collateral(self, col_id: str) -> dict[str, Any]:
		try:
			if col_id not in self._collateral:
				raise KeyError(f"collateral_not_found:{col_id}")
			self._collateral.pop(col_id)
			self._emit("collateral.deleted", "collateral", col_id, {"id": col_id})
			return {"deleted": True, "id": col_id}
		except Exception as exc:
			_log.error("delete_collateral failed: %s", exc)
			raise

	# ------------------------------------------------------------------ group lending

	async def list_groups(self) -> list[dict[str, Any]]:
		return list(self._groups.values())

	async def create_group_loan(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			gid = _new_id("grp")
			ts = _now()
			members = list(payload["member_ids"])
			amount = float(payload["loan_amount"])
			record: dict[str, Any] = {
				"id": gid,
				"tenant_id": self.tenant_id,
				"group_name": payload["group_name"],
				"member_ids": members,
				"loan_amount": amount,
				"per_member_amount": round(amount / len(members), 2) if members else amount,
				"currency": payload.get("currency", "KES"),
				"season": payload["season"],
				"duration_months": int(payload["duration_months"]),
				"purpose": payload["purpose"],
				"status": "applied",
				"created_at": ts,
				"updated_at": ts,
			}
			self._groups[gid] = record
			self._emit("group_loan.created", "group_loan", gid, record)
			return record
		except Exception as exc:
			_log.error("create_group_loan failed: %s", exc)
			raise

	async def update_group_loan(self, group_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if group_id not in self._groups:
				raise KeyError(f"group_loan_not_found:{group_id}")
			record = self._groups[group_id]
			if "status" in payload:
				record["status"] = payload["status"]
			record["updated_at"] = _now()
			self._emit("group_loan.updated", "group_loan", group_id, payload)
			return record
		except Exception as exc:
			_log.error("update_group_loan failed: %s", exc)
			raise

	async def delete_group_loan(self, group_id: str) -> dict[str, Any]:
		try:
			if group_id not in self._groups:
				raise KeyError(f"group_loan_not_found:{group_id}")
			self._groups.pop(group_id)
			self._emit("group_loan.deleted", "group_loan", group_id, {"id": group_id})
			return {"deleted": True, "id": group_id}
		except Exception as exc:
			_log.error("delete_group_loan failed: %s", exc)
			raise

	async def get_portfolio_summary(self) -> dict[str, Any]:
		"""Aggregate loan portfolio statistics."""
		loans = list(self._loans.values())
		total_disbursed = sum(l.get("approved_amount") or 0 for l in loans if l.get("status") in ("disbursed", "repaying", "settled"))
		outstanding = sum(l.get("approved_amount") or 0 for l in loans if l.get("status") in ("disbursed", "repaying"))
		defaulted = sum(l.get("approved_amount") or 0 for l in loans if l.get("status") == "defaulted")
		return {
			"total_applications": len(loans),
			"approved": len([l for l in loans if l.get("status") not in ("applied", "rejected")]),
			"total_disbursed": round(total_disbursed, 2),
			"outstanding_balance": round(outstanding, 2),
			"defaulted_amount": round(defaulted, 2),
			"default_rate_pct": round(defaulted / total_disbursed * 100, 2) if total_disbursed > 0 else 0,
			"group_loans": len(self._groups),
		}
