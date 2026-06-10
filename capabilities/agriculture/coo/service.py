"""Cooperative Management service — agr_coo."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

_log = logging.getLogger(__name__)
_CAPABILITY_ID = "agr_coo"


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _new_id(prefix: str = "") -> str:
	suffix = uuid4().hex[:12]
	return f"{prefix}-{suffix}" if prefix else suffix


class CooperativeManagementService:
	"""Async service for cooperative management: member registry, share management,
	pooled inputs, dividend allocation, and annual returns."""

	def __init__(self, tenant_id: str = "default") -> None:
		if not tenant_id:
			raise ValueError("tenant_id required")
		self.tenant_id = tenant_id
		self._coops: dict[str, dict[str, Any]] = {}
		self._members: dict[str, dict[str, Any]] = {}
		self._input_pools: dict[str, dict[str, Any]] = {}
		self._dividends: dict[str, dict[str, Any]] = {}
		self._annual_returns: dict[str, dict[str, Any]] = {}
		self._share_ledger: list[dict[str, Any]] = []
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
				"cooperatives": len(self._coops),
				"members": len(self._members),
				"input_pools": len(self._input_pools),
				"dividend_allocations": len(self._dividends),
				"annual_returns": len(self._annual_returns),
			},
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": _CAPABILITY_ID,
			"name": "Cooperative Management",
			"domain": "agriculture",
			"version": "1.0.0",
			"description": "Member registry, share management, pooled inputs, dividend allocation, annual returns.",
		}

	async def get_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
		return self._audit[-limit:]

	# ------------------------------------------------------------------ cooperatives

	async def list_coops(self, region: str | None = None) -> list[dict[str, Any]]:
		items = list(self._coops.values())
		if region:
			items = [c for c in items if c.get("region") == region]
		return items

	async def get_coop(self, coop_id: str) -> dict[str, Any]:
		if coop_id not in self._coops:
			raise KeyError(f"coop_not_found:{coop_id}")
		return self._coops[coop_id]

	async def create_coop(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			cid = _new_id("coo")
			ts = _now()
			record: dict[str, Any] = {
				"id": cid,
				"tenant_id": self.tenant_id,
				"name": payload["name"],
				"registration_number": payload["registration_number"],
				"region": payload["region"],
				"crop_focus": list(payload.get("crop_focus", [])),
				"share_value": float(payload["share_value"]),
				"currency": payload.get("currency", "KES"),
				"total_shares_issued": 0,
				"total_members": 0,
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._coops[cid] = record
			self._emit("coop.created", "cooperative", cid, record)
			return record
		except Exception as exc:
			_log.error("create_coop failed: %s", exc)
			raise

	async def update_coop(self, coop_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if coop_id not in self._coops:
				raise KeyError(f"coop_not_found:{coop_id}")
			record = self._coops[coop_id]
			for field in ["name", "region", "crop_focus", "share_value", "notes"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("coop.updated", "cooperative", coop_id, payload)
			return record
		except Exception as exc:
			_log.error("update_coop failed: %s", exc)
			raise

	async def delete_coop(self, coop_id: str) -> dict[str, Any]:
		try:
			if coop_id not in self._coops:
				raise KeyError(f"coop_not_found:{coop_id}")
			self._coops.pop(coop_id)
			self._emit("coop.deleted", "cooperative", coop_id, {"id": coop_id})
			return {"deleted": True, "id": coop_id}
		except Exception as exc:
			_log.error("delete_coop failed: %s", exc)
			raise

	# ------------------------------------------------------------------ members

	async def list_members(self, coop_id: str | None = None, status: str | None = None,
						limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		items = list(self._members.values())
		if coop_id:
			items = [m for m in items if m.get("coop_id") == coop_id]
		if status:
			items = [m for m in items if m.get("status") == status]
		return items[offset: offset + limit]

	async def get_member(self, member_id: str) -> dict[str, Any]:
		if member_id not in self._members:
			raise KeyError(f"member_not_found:{member_id}")
		return self._members[member_id]

	async def create_member(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			coop_id = payload["coop_id"]
			if coop_id not in self._coops:
				raise KeyError(f"coop_not_found:{coop_id}")
			coop = self._coops[coop_id]
			mid = _new_id("mbr")
			ts = _now()
			shares = int(payload.get("shares_purchased", 1))
			record: dict[str, Any] = {
				"id": mid,
				"tenant_id": self.tenant_id,
				"coop_id": coop_id,
				"farmer_id": payload["farmer_id"],
				"name": payload["name"],
				"id_number": payload["id_number"],
				"shares_held": shares,
				"share_value": coop["share_value"],
				"total_share_value": round(shares * coop["share_value"], 2),
				"status": "active",
				"join_date": payload["join_date"],
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._members[mid] = record
			# Update coop totals
			coop["total_shares_issued"] = coop.get("total_shares_issued", 0) + shares
			coop["total_members"] = coop.get("total_members", 0) + 1
			coop["updated_at"] = ts
			# Record in share ledger
			self._share_ledger.append({
				"member_id": mid,
				"coop_id": coop_id,
				"type": "purchase",
				"shares": shares,
				"amount": record["total_share_value"],
				"occurred_at": ts,
			})
			self._emit("member.created", "member", mid, record)
			return record
		except Exception as exc:
			_log.error("create_member failed: %s", exc)
			raise

	async def update_member(self, member_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if member_id not in self._members:
				raise KeyError(f"member_not_found:{member_id}")
			record = self._members[member_id]
			for field in ["name", "status", "notes"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("member.updated", "member", member_id, payload)
			return record
		except Exception as exc:
			_log.error("update_member failed: %s", exc)
			raise

	async def delete_member(self, member_id: str) -> dict[str, Any]:
		try:
			if member_id not in self._members:
				raise KeyError(f"member_not_found:{member_id}")
			self._members.pop(member_id)
			self._emit("member.deleted", "member", member_id, {"id": member_id})
			return {"deleted": True, "id": member_id}
		except Exception as exc:
			_log.error("delete_member failed: %s", exc)
			raise

	async def transfer_shares(self, from_member_id: str, to_member_id: str, shares: int) -> dict[str, Any]:
		"""Transfer shares between two cooperative members."""
		try:
			if from_member_id not in self._members:
				raise KeyError(f"member_not_found:{from_member_id}")
			if to_member_id not in self._members:
				raise KeyError(f"member_not_found:{to_member_id}")
			source = self._members[from_member_id]
			dest = self._members[to_member_id]
			if source["coop_id"] != dest["coop_id"]:
				raise ValueError("members_must_be_in_same_coop")
			if source["shares_held"] < shares:
				raise ValueError(f"insufficient_shares:{source['shares_held']}<{shares}")
			ts = _now()
			source["shares_held"] -= shares
			source["total_share_value"] = round(source["shares_held"] * source["share_value"], 2)
			source["updated_at"] = ts
			dest["shares_held"] += shares
			dest["total_share_value"] = round(dest["shares_held"] * dest["share_value"], 2)
			dest["updated_at"] = ts
			self._share_ledger.append({
				"from_member_id": from_member_id,
				"to_member_id": to_member_id,
				"coop_id": source["coop_id"],
				"type": "transfer",
				"shares": shares,
				"occurred_at": ts,
			})
			self._emit("shares.transferred", "share_ledger", from_member_id, {"shares": shares})
			return {"from_member": from_member_id, "to_member": to_member_id, "shares_transferred": shares}
		except Exception as exc:
			_log.error("transfer_shares failed: %s", exc)
			raise

	# ------------------------------------------------------------------ input pools

	async def list_input_pools(self, coop_id: str | None = None, season: str | None = None) -> list[dict[str, Any]]:
		items = list(self._input_pools.values())
		if coop_id:
			items = [p for p in items if p.get("coop_id") == coop_id]
		if season:
			items = [p for p in items if p.get("season") == season]
		return items

	async def create_input_pool(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			pool_id = _new_id("pol")
			ts = _now()
			qty = float(payload["total_quantity"])
			unit_cost = float(payload["unit_cost"])
			record: dict[str, Any] = {
				"id": pool_id,
				"tenant_id": self.tenant_id,
				"coop_id": payload["coop_id"],
				"product_name": payload["product_name"],
				"total_quantity": qty,
				"unit": payload["unit"],
				"unit_cost": unit_cost,
				"total_cost": round(qty * unit_cost, 2),
				"season": payload["season"],
				"allocated_quantity": 0.0,
				"remaining_quantity": qty,
				"notes": payload.get("notes"),
				"created_at": ts,
			}
			self._input_pools[pool_id] = record
			self._emit("input_pool.created", "input_pool", pool_id, record)
			return record
		except Exception as exc:
			_log.error("create_input_pool failed: %s", exc)
			raise

	async def allocate_from_pool(self, pool_id: str, member_id: str, quantity: float) -> dict[str, Any]:
		"""Allocate a share of pooled input to a member."""
		try:
			if pool_id not in self._input_pools:
				raise KeyError(f"input_pool_not_found:{pool_id}")
			pool = self._input_pools[pool_id]
			if quantity > pool["remaining_quantity"]:
				raise ValueError(f"insufficient_pool_stock:{pool['remaining_quantity']}<{quantity}")
			pool["allocated_quantity"] = round(pool["allocated_quantity"] + quantity, 3)
			pool["remaining_quantity"] = round(pool["remaining_quantity"] - quantity, 3)
			self._emit("pool.allocated", "input_pool", pool_id, {"member_id": member_id, "quantity": quantity})
			return {"pool_id": pool_id, "member_id": member_id, "quantity_allocated": quantity,
					"remaining": pool["remaining_quantity"]}
		except Exception as exc:
			_log.error("allocate_from_pool failed: %s", exc)
			raise

	async def delete_input_pool(self, pool_id: str) -> dict[str, Any]:
		try:
			if pool_id not in self._input_pools:
				raise KeyError(f"input_pool_not_found:{pool_id}")
			self._input_pools.pop(pool_id)
			self._emit("input_pool.deleted", "input_pool", pool_id, {"id": pool_id})
			return {"deleted": True, "id": pool_id}
		except Exception as exc:
			_log.error("delete_input_pool failed: %s", exc)
			raise

	# ------------------------------------------------------------------ dividends

	async def list_dividends(self, coop_id: str | None = None) -> list[dict[str, Any]]:
		items = list(self._dividends.values())
		if coop_id:
			items = [d for d in items if d.get("coop_id") == coop_id]
		return items

	async def allocate_dividends(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Compute and record dividend allocations for all active members."""
		try:
			coop_id = payload["coop_id"]
			if coop_id not in self._coops:
				raise KeyError(f"coop_not_found:{coop_id}")
			did = _new_id("div")
			ts = _now()
			total_profit = float(payload["total_profit"])
			rate_pct = float(payload["dividend_rate_pct"])
			distributable = round(total_profit * rate_pct / 100, 2)
			active_members = [m for m in self._members.values()
							if m.get("coop_id") == coop_id and m.get("status") == "active"]
			total_shares = sum(m.get("shares_held", 0) for m in active_members)
			allocations = []
			for m in active_members:
				share_frac = m["shares_held"] / total_shares if total_shares > 0 else 0
				member_div = round(distributable * share_frac, 2)
				allocations.append({
					"member_id": m["id"],
					"farmer_id": m["farmer_id"],
					"shares_held": m["shares_held"],
					"dividend_amount": member_div,
				})
			record: dict[str, Any] = {
				"id": did,
				"tenant_id": self.tenant_id,
				"coop_id": coop_id,
				"financial_year": payload["financial_year"],
				"total_profit": total_profit,
				"dividend_rate_pct": rate_pct,
				"total_dividend_paid": distributable,
				"allocations": allocations,
				"notes": payload.get("notes"),
				"created_at": ts,
			}
			self._dividends[did] = record
			self._emit("dividends.allocated", "dividend_allocation", did, {"coop_id": coop_id, "total": distributable})
			return record
		except Exception as exc:
			_log.error("allocate_dividends failed: %s", exc)
			raise

	# ------------------------------------------------------------------ annual returns

	async def list_annual_returns(self, coop_id: str | None = None) -> list[dict[str, Any]]:
		items = list(self._annual_returns.values())
		if coop_id:
			items = [r for r in items if r.get("coop_id") == coop_id]
		return items

	async def file_annual_return(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			ret_id = _new_id("ret")
			ts = _now()
			coop_id = payload["coop_id"]
			revenue = float(payload["total_revenue"])
			expenses = float(payload["total_expenses"])
			profit = float(payload.get("net_profit", revenue - expenses))
			coop = self._coops.get(coop_id)
			total_equity = (coop.get("total_shares_issued", 0) * coop.get("share_value", 1)) if coop else 0
			roe = round(profit / total_equity * 100, 2) if total_equity > 0 else None
			record: dict[str, Any] = {
				"id": ret_id,
				"tenant_id": self.tenant_id,
				"coop_id": coop_id,
				"financial_year": payload["financial_year"],
				"total_revenue": revenue,
				"total_expenses": expenses,
				"net_profit": profit,
				"member_count": int(payload["member_count"]),
				"return_on_equity_pct": roe,
				"notes": payload.get("notes"),
				"created_at": ts,
			}
			self._annual_returns[ret_id] = record
			self._emit("annual_return.filed", "annual_return", ret_id, record)
			return record
		except Exception as exc:
			_log.error("file_annual_return failed: %s", exc)
			raise

	async def delete_annual_return(self, ret_id: str) -> dict[str, Any]:
		try:
			if ret_id not in self._annual_returns:
				raise KeyError(f"annual_return_not_found:{ret_id}")
			self._annual_returns.pop(ret_id)
			self._emit("annual_return.deleted", "annual_return", ret_id, {"id": ret_id})
			return {"deleted": True, "id": ret_id}
		except Exception as exc:
			_log.error("delete_annual_return failed: %s", exc)
			raise

	async def get_coop_summary(self, coop_id: str) -> dict[str, Any]:
		"""Consolidated cooperative summary."""
		if coop_id not in self._coops:
			raise KeyError(f"coop_not_found:{coop_id}")
		coop = self._coops[coop_id]
		members = [m for m in self._members.values() if m.get("coop_id") == coop_id]
		active = [m for m in members if m.get("status") == "active"]
		total_equity = sum(m.get("total_share_value", 0) for m in active)
		pools = [p for p in self._input_pools.values() if p.get("coop_id") == coop_id]
		return {
			"coop_id": coop_id,
			"name": coop["name"],
			"total_members": len(members),
			"active_members": len(active),
			"total_shares_issued": coop.get("total_shares_issued", 0),
			"total_equity": round(total_equity, 2),
			"currency": coop.get("currency", "KES"),
			"active_input_pools": len(pools),
		}

	async def get_member_statement(self, member_id: str) -> dict[str, Any]:
		"""Return share and dividend history for a member."""
		if member_id not in self._members:
			raise KeyError(f"member_not_found:{member_id}")
		member = self._members[member_id]
		coop_id = member["coop_id"]
		ledger = [t for t in self._share_ledger
				if t.get("member_id") == member_id or t.get("from_member_id") == member_id
				or t.get("to_member_id") == member_id]
		divs = [d for d in self._dividends.values() if d.get("coop_id") == coop_id]
		member_divs = []
		for d in divs:
			for alloc in d.get("allocations", []):
				if alloc.get("member_id") == member_id:
					member_divs.append({"year": d["financial_year"], "amount": alloc["dividend_amount"]})
		return {
			"member_id": member_id,
			"name": member["name"],
			"coop_id": coop_id,
			"shares_held": member["shares_held"],
			"total_share_value": member["total_share_value"],
			"share_transactions": ledger,
			"dividend_history": member_divs,
			"total_dividends": round(sum(d["amount"] for d in member_divs), 2),
		}
