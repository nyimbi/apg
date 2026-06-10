"""APG ESG/Carbon Tracking Service — expanded async runtime (42+ methods).

All state in _Store. Every mutation emits an audit event.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import statistics
from datetime import datetime, timezone
from typing import Any

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	import uuid
	def uuid7str() -> str:
		return str(uuid.uuid4())

import logging
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

logger = logging.getLogger(__name__)

VALID_SCOPES: set[str] = {"scope_1", "scope_2", "scope_3"}
KNOWN_FRAMEWORKS: set[str] = {"GRI", "SASB", "TCFD", "ISSB", "CDP", "CSRD"}
SUPPORTED_CHANNELS: set[str] = {"email", "sms", "webhook", "audit_log"}


def _utc_now() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize(v: str) -> str:
	return str(v or "").strip().lower().replace("-", "_").replace(" ", "_")


def _esg_rating(score: float) -> str:
	if score >= 80: return "AAA"
	if score >= 65: return "AA"
	if score >= 50: return "A"
	if score >= 35: return "BBB"
	if score >= 20: return "BB"
	return "B"


def _default_sections(framework: str) -> list[str]:
	return {
		"GRI": ["GRI 2 General Disclosures", "GRI 3 Material Topics", "GRI 302 Energy", "GRI 305 Emissions"],
		"SASB": ["Industry Classification", "Sustainability Accounting Metrics"],
		"TCFD": ["Governance", "Strategy", "Risk Management", "Metrics and Targets"],
		"ISSB": ["IFRS S1 General Requirements", "IFRS S2 Climate-related Disclosures"],
		"CDP": ["Climate Change Governance", "Targets", "Emissions Data"],
		"CSRD": ["General Disclosures", "Environmental", "Social", "Governance"],
	}.get(framework, ["General", "Environmental", "Social", "Governance"])


class _Store:
	def __init__(self) -> None:
		self._data: dict[str, dict[str, Any]] = {}

	async def put(self, col: str, rec: dict[str, Any]) -> dict[str, Any]:
		self._data.setdefault(col, {})[rec["id"]] = rec
		return rec

	async def get(self, col: str, rid: str) -> dict[str, Any] | None:
		return self._data.get(col, {}).get(rid)

	async def list(self, col: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(self._data.get(col, {}).values())
		if tenant_id is not None:
			items = [i for i in items if i.get("tenant_id") == tenant_id]
		return sorted(items, key=lambda i: i.get("id", ""))

	async def delete(self, col: str, rid: str) -> bool:
		bucket = self._data.get(col, {})
		if rid in bucket:
			del bucket[rid]
			return True
		return False


class _Audit:
	def __init__(self, store: _Store) -> None:
		self._store = store

	async def log_event(self, event_type: str, actor_id: str, tenant_id: str, subject_id: str,
						details: dict[str, Any] | None = None, severity: str = "info") -> dict[str, Any]:
		rec = {
			"id": uuid7str(), "tenant_id": tenant_id, "event_type": event_type,
			"actor_id": actor_id, "subject_id": subject_id, "severity": severity,
			"details": details or {}, "recorded_at": _utc_now(),
		}
		await self._store.put("esgc_audit", rec)
		return rec


class _Notify:
	async def send(self, recipient: str, channel: str, subject: str, body: str) -> dict[str, Any]:
		if channel not in SUPPORTED_CHANNELS:
			raise ValueError(f"unsupported_channel:{channel}")
		return {"id": uuid7str(), "recipient": recipient, "channel": channel, "subject": subject, "sent_at": _utc_now()}


class EsgcService:
	"""Async ESG/Carbon tracking service — 42+ methods."""

	def __init__(self, actor_id: str = "system", tenant_id: str = "default") -> None:
		self.actor_id = actor_id
		self.tenant_id = tenant_id
		self._store = _Store()
		self._audit = _Audit(self._store)
		self._notify = _Notify()

	# ------------------------------------------------------------------
	# 1. create_inventory
	# ------------------------------------------------------------------
	async def create_inventory(
		self,
		inventory_id: str,
		tenant_id: str,
		organization: str,
		owner: str,
		reporting_year: int,
		boundary_ref: str,
		geospatial_boundary: str,
		compliance_framework: str,
		status: str = "active",
	) -> dict[str, Any]:
		assert organization and owner, "organization and owner required"
		assert boundary_ref and geospatial_boundary, "boundary required"
		assert compliance_framework, "compliance_framework required"
		record = {
			"id": inventory_id, "tenant_id": tenant_id, "organization": organization, "owner": owner,
			"reporting_year": int(reporting_year), "boundary_ref": boundary_ref,
			"geospatial_boundary": geospatial_boundary, "compliance_framework": compliance_framework,
			"status": status, "created_at": _utc_now(),
		}
		await self._store.put("esgc_inventories", record)
		await self._audit.log_event("inventory_created", self.actor_id, tenant_id, inventory_id, {"reporting_year": reporting_year})
		return record

	# ------------------------------------------------------------------
	# 2. register_factor
	# ------------------------------------------------------------------
	async def register_factor(
		self,
		factor_id: str,
		tenant_id: str,
		name: str,
		scope: str,
		unit: str,
		co2e_per_unit: float,
		source: str,
		source_evidence: str,
		version: str,
		approved_source: bool,
		status: str = "active",
	) -> dict[str, Any]:
		if scope not in VALID_SCOPES:
			raise ValueError(f"invalid_scope:{scope}")
		record = {
			"id": factor_id, "tenant_id": tenant_id, "name": name, "scope": scope, "unit": unit,
			"co2e_per_unit": float(co2e_per_unit), "source": source, "source_evidence": source_evidence,
			"version": version, "approved_source": bool(approved_source), "status": status,
			"created_at": _utc_now(),
		}
		await self._store.put("esgc_factors", record)
		await self._audit.log_event("factor_registered", self.actor_id, tenant_id, factor_id, {"source": source, "version": version})
		return record

	# ------------------------------------------------------------------
	# 3. record_activity
	# ------------------------------------------------------------------
	async def record_activity(
		self,
		activity_id: str,
		tenant_id: str,
		inventory_id: str,
		factor_id: str,
		activity_type: str,
		quantity: float,
		unit: str,
		evidence_ref: str,
		anomaly_review_recorded: bool = False,
	) -> dict[str, Any]:
		inventory = await self._require_inventory(tenant_id, inventory_id)
		factor = await self._require_factor(tenant_id, factor_id)
		co2e = round(float(quantity) * factor["co2e_per_unit"], 6)
		record = {
			"id": activity_id, "tenant_id": tenant_id, "inventory_id": inventory_id,
			"factor_id": factor_id, "activity_type": activity_type, "scope": factor["scope"],
			"quantity": float(quantity), "unit": unit, "co2e_tonnes": co2e,
			"evidence_ref": evidence_ref, "anomaly_detected": False,
			"anomaly_review_recorded": anomaly_review_recorded, "status": "recorded", "created_at": _utc_now(),
		}
		await self._store.put("esgc_activities", record)
		await self._audit.log_event("activity_recorded", self.actor_id, tenant_id, activity_id, {"co2e_tonnes": co2e, "scope": factor["scope"]})
		return record

	# ------------------------------------------------------------------
	# 4. scope1_record
	# ------------------------------------------------------------------
	async def scope1_record(
		self,
		tenant_id: str,
		inventory_id: str,
		source_type: str,
		quantity: float,
		unit: str,
		emission_factor: float,
		period: str,
		evidence_ref: str = "",
	) -> dict[str, Any]:
		"""Record a Scope 1 (direct) emission."""
		factor_id = f"s1:{source_type}:{period}"
		if not await self._store.get("esgc_factors", factor_id):
			await self.register_factor(factor_id, tenant_id, f"{source_type} S1", "scope_1", unit, emission_factor, source_type, evidence_ref or f"inline:{source_type}", "v1", True)
		activity_id = f"act:s1:{source_type}:{period}:{uuid7str()[:8]}"
		return await self.record_activity(activity_id, tenant_id, inventory_id, factor_id, source_type, quantity, unit, evidence_ref or f"inline:{source_type}:{period}")

	# ------------------------------------------------------------------
	# 5. scope2_calculate
	# ------------------------------------------------------------------
	async def scope2_calculate(self, tenant_id: str, entity_id: str, period: str, grid_factor: float = 0.0004) -> dict[str, Any]:
		"""Aggregate Scope 2 emissions (market + location based)."""
		activities = [a for a in await self._store.list("esgc_activities", tenant_id)
					  if a["scope"] == "scope_2" and a["inventory_id"] == entity_id]
		market_total = sum(a["co2e_tonnes"] for a in activities)
		location_total = sum(a["quantity"] * grid_factor for a in activities)
		result = {
			"tenant_id": tenant_id, "entity_id": entity_id, "period": period, "scope": "scope_2",
			"market_based_co2e_tonnes": round(market_total, 4),
			"location_based_co2e_tonnes": round(location_total, 4),
			"grid_factor_used": grid_factor, "activity_count": len(activities),
			"calculated_at": _utc_now(),
		}
		await self._audit.log_event("scope2_calculated", self.actor_id, tenant_id, entity_id, result)
		return result

	# ------------------------------------------------------------------
	# 6. scope3_estimate
	# ------------------------------------------------------------------
	async def scope3_estimate(self, tenant_id: str, entity_id: str, category: str, period: str) -> dict[str, Any]:
		"""Estimate Scope 3 value-chain emissions for a category."""
		activities = [a for a in await self._store.list("esgc_activities", tenant_id)
					  if a["scope"] == "scope_3" and a["inventory_id"] == entity_id
					  and (category == "all" or a["activity_type"] == category)]
		total = sum(a["co2e_tonnes"] for a in activities)
		cats: dict[str, float] = {}
		for a in activities:
			cats[a["activity_type"]] = round(cats.get(a["activity_type"], 0.0) + a["co2e_tonnes"], 4)
		result = {
			"tenant_id": tenant_id, "entity_id": entity_id, "period": period, "scope": "scope_3",
			"category_filter": category, "total_co2e_tonnes": round(total, 4),
			"activity_count": len(activities), "breakdown_by_category": cats, "calculated_at": _utc_now(),
		}
		await self._audit.log_event("scope3_estimated", self.actor_id, tenant_id, entity_id, result)
		return result

	# ------------------------------------------------------------------
	# 7. carbon_offset_verify
	# ------------------------------------------------------------------
	async def carbon_offset_verify(self, tenant_id: str, offset_id: str) -> dict[str, Any]:
		"""Verify an existing carbon offset record."""
		offset = await self._store.get("esgc_offsets", offset_id)
		if offset is None or offset["tenant_id"] != tenant_id:
			raise KeyError(f"offset_not_found:{offset_id}")
		valid = offset.get("status") == "retired" and bool(offset.get("verification_standard"))
		result = {
			"offset_id": offset_id, "valid": valid, "registry": offset.get("registry"),
			"verification_standard": offset.get("verification_standard"),
			"verified_at": _utc_now(),
		}
		await self._audit.log_event("offset_verified", self.actor_id, tenant_id, offset_id, result)
		return result

	# ------------------------------------------------------------------
	# 8. green_certification
	# ------------------------------------------------------------------
	async def green_certification(
		self,
		tenant_id: str,
		certification_id: str,
		entity_id: str,
		standard: str,
		certifying_body: str,
		valid_until: str,
	) -> dict[str, Any]:
		"""Record a green certification (ISO 14001, LEED, etc.)."""
		assert standard and certifying_body, "standard and certifying_body required"
		record = {
			"id": certification_id, "tenant_id": tenant_id, "entity_id": entity_id,
			"standard": standard, "certifying_body": certifying_body,
			"valid_until": valid_until, "status": "active", "issued_at": _utc_now(),
		}
		await self._store.put("esgc_certifications", record)
		await self._audit.log_event("green_certification_issued", self.actor_id, tenant_id, certification_id, {"standard": standard})
		return record

	# ------------------------------------------------------------------
	# 9. supplier_esg_score
	# ------------------------------------------------------------------
	async def supplier_esg_score(
		self,
		tenant_id: str,
		supplier_id: str,
		metrics: dict[str, float],
		weights: dict[str, float] | None = None,
	) -> dict[str, Any]:
		"""Compute a weighted ESG score for a supplier."""
		assert metrics, "metrics required"
		w = weights or {k: 1.0 / len(metrics) for k in metrics}
		w_sum = sum(w.values()) or 1.0
		nw = {k: v / w_sum for k, v in w.items()}
		raw = sum(metrics.get(k, 0.0) * nw.get(k, 0.0) for k in metrics)
		final = max(0.0, min(100.0, round(raw, 2)))
		record = {
			"id": uuid7str(), "tenant_id": tenant_id, "supplier_id": supplier_id,
			"metrics": metrics, "weights": nw, "score": final, "rating": _esg_rating(final),
			"calculated_at": _utc_now(),
		}
		await self._store.put("esgc_supplier_scores", record)
		await self._audit.log_event("supplier_esg_scored", self.actor_id, tenant_id, record["id"], {"supplier_id": supplier_id, "score": final})
		return record

	# ------------------------------------------------------------------
	# 10. biodiversity_impact
	# ------------------------------------------------------------------
	async def biodiversity_impact(
		self,
		tenant_id: str,
		assessment_id: str,
		site_id: str,
		land_use_ha: float,
		ecosystem_type: str,
		impact_score: float,
	) -> dict[str, Any]:
		"""Record a biodiversity impact assessment."""
		assert 0 <= impact_score <= 100, "impact_score must be in [0,100]"
		record = {
			"id": assessment_id, "tenant_id": tenant_id, "site_id": site_id,
			"land_use_ha": land_use_ha, "ecosystem_type": ecosystem_type,
			"impact_score": impact_score,
			"risk_level": "high" if impact_score > 70 else "medium" if impact_score > 30 else "low",
			"assessed_at": _utc_now(),
		}
		await self._store.put("esgc_biodiversity", record)
		await self._audit.log_event("biodiversity_assessed", self.actor_id, tenant_id, assessment_id, {"site_id": site_id, "impact_score": impact_score})
		return record

	# ------------------------------------------------------------------
	# 11. water_usage
	# ------------------------------------------------------------------
	async def water_usage(
		self,
		tenant_id: str,
		record_id: str,
		entity_id: str,
		period: str,
		consumption_m3: float,
		source: str,
		water_stress_level: str = "low",
	) -> dict[str, Any]:
		"""Record water consumption data."""
		record = {
			"id": record_id, "tenant_id": tenant_id, "entity_id": entity_id,
			"period": period, "consumption_m3": consumption_m3, "source": source,
			"water_stress_level": water_stress_level, "recorded_at": _utc_now(),
		}
		await self._store.put("esgc_water", record)
		await self._audit.log_event("water_usage_recorded", self.actor_id, tenant_id, record_id, {"consumption_m3": consumption_m3, "period": period})
		return record

	# ------------------------------------------------------------------
	# 12. waste_tracking
	# ------------------------------------------------------------------
	async def waste_tracking(
		self,
		tenant_id: str,
		record_id: str,
		entity_id: str,
		period: str,
		waste_tonnes: float,
		waste_type: str,
		disposal_method: str,
		recycled_percent: float = 0.0,
	) -> dict[str, Any]:
		"""Record waste generation and disposal data."""
		assert 0 <= recycled_percent <= 100, "recycled_percent must be in [0,100]"
		record = {
			"id": record_id, "tenant_id": tenant_id, "entity_id": entity_id,
			"period": period, "waste_tonnes": waste_tonnes, "waste_type": waste_type,
			"disposal_method": disposal_method, "recycled_percent": recycled_percent,
			"recycled_tonnes": round(waste_tonnes * recycled_percent / 100, 4),
			"recorded_at": _utc_now(),
		}
		await self._store.put("esgc_waste", record)
		await self._audit.log_event("waste_recorded", self.actor_id, tenant_id, record_id, {"waste_tonnes": waste_tonnes, "period": period})
		return record

	# ------------------------------------------------------------------
	# 13. energy_audit
	# ------------------------------------------------------------------
	async def energy_audit(
		self,
		tenant_id: str,
		audit_id: str,
		entity_id: str,
		period: str,
		total_kwh: float,
		renewable_kwh: float,
		auditor: str,
	) -> dict[str, Any]:
		"""Record an energy audit."""
		assert renewable_kwh <= total_kwh, "renewable_kwh cannot exceed total_kwh"
		renewable_pct = round(renewable_kwh / max(total_kwh, 1) * 100, 2)
		record = {
			"id": audit_id, "tenant_id": tenant_id, "entity_id": entity_id,
			"period": period, "total_kwh": total_kwh, "renewable_kwh": renewable_kwh,
			"non_renewable_kwh": total_kwh - renewable_kwh,
			"renewable_percent": renewable_pct, "auditor": auditor, "audited_at": _utc_now(),
		}
		await self._store.put("esgc_energy_audits", record)
		await self._audit.log_event("energy_audited", self.actor_id, tenant_id, audit_id, {"period": period, "renewable_percent": renewable_pct})
		return record

	# ------------------------------------------------------------------
	# 14. esg_rating
	# ------------------------------------------------------------------
	async def esg_rating(
		self,
		tenant_id: str,
		entity_id: str,
		e_score: float,
		s_score: float,
		g_score: float,
	) -> dict[str, Any]:
		"""Compute and store a combined ESG rating."""
		combined = round((e_score + s_score + g_score) / 3, 2)
		record = {
			"id": uuid7str(), "tenant_id": tenant_id, "entity_id": entity_id,
			"environmental_score": e_score, "social_score": s_score, "governance_score": g_score,
			"combined_score": combined, "rating": _esg_rating(combined), "rated_at": _utc_now(),
		}
		await self._store.put("esgc_ratings", record)
		await self._audit.log_event("esg_rated", self.actor_id, tenant_id, record["id"], {"combined": combined, "rating": record["rating"]})
		return record

	# ------------------------------------------------------------------
	# 15. sdg_alignment
	# ------------------------------------------------------------------
	async def sdg_alignment(
		self,
		tenant_id: str,
		alignment_id: str,
		entity_id: str,
		sdg_goals: list[int],
		contribution_details: dict[int, str],
	) -> dict[str, Any]:
		"""Record SDG alignment mapping for an entity."""
		invalid = [g for g in sdg_goals if not (1 <= g <= 17)]
		if invalid:
			raise ValueError(f"invalid_sdg_goals:{invalid}")
		record = {
			"id": alignment_id, "tenant_id": tenant_id, "entity_id": entity_id,
			"sdg_goals": sdg_goals, "contribution_details": {str(k): v for k, v in contribution_details.items()},
			"goal_count": len(sdg_goals), "recorded_at": _utc_now(),
		}
		await self._store.put("esgc_sdg", record)
		await self._audit.log_event("sdg_aligned", self.actor_id, tenant_id, alignment_id, {"goals": sdg_goals})
		return record

	# ------------------------------------------------------------------
	# 16. climate_risk
	# ------------------------------------------------------------------
	async def climate_risk(
		self,
		tenant_id: str,
		assessment_id: str,
		entity_id: str,
		physical_risks: list[dict[str, Any]],
		transition_risks: list[dict[str, Any]],
		horizon: str = "medium_term",
	) -> dict[str, Any]:
		"""Assess and record TCFD-aligned climate risk."""
		record = {
			"id": assessment_id, "tenant_id": tenant_id, "entity_id": entity_id,
			"horizon": horizon, "physical_risks": physical_risks, "transition_risks": transition_risks,
			"physical_risk_count": len(physical_risks), "transition_risk_count": len(transition_risks),
			"overall_risk": "high" if len(physical_risks) + len(transition_risks) > 5 else "medium",
			"assessed_at": _utc_now(),
		}
		await self._store.put("esgc_climate_risk", record)
		await self._audit.log_event("climate_risk_assessed", self.actor_id, tenant_id, assessment_id, {"horizon": horizon})
		return record

	# ------------------------------------------------------------------
	# 17. carbon_credit_trade
	# ------------------------------------------------------------------
	async def carbon_credit_trade(
		self,
		tenant_id: str,
		trade_id: str,
		trade_type: str,
		credits_tco2e: float,
		price_per_credit: float,
		currency: str,
		counterparty: str,
		registry: str,
	) -> dict[str, Any]:
		"""Record a carbon credit trade (buy/sell)."""
		assert trade_type in {"buy", "sell"}, "trade_type must be buy or sell"
		assert credits_tco2e > 0, "credits_tco2e must be positive"
		total_value = round(credits_tco2e * price_per_credit, 4)
		record = {
			"id": trade_id, "tenant_id": tenant_id, "trade_type": trade_type,
			"credits_tco2e": credits_tco2e, "price_per_credit": price_per_credit,
			"total_value": total_value, "currency": currency, "counterparty": counterparty,
			"registry": registry, "status": "settled", "traded_at": _utc_now(),
		}
		await self._store.put("esgc_credit_trades", record)
		await self._audit.log_event("carbon_credit_traded", self.actor_id, tenant_id, trade_id, {"type": trade_type, "credits": credits_tco2e}, severity="medium")
		await self._notify.send(self.actor_id, "audit_log", "Carbon credit trade", f"{trade_type} {credits_tco2e} tCO2e @ {price_per_credit} {currency}")
		return record

	# ------------------------------------------------------------------
	# 18. esg_analytics
	# ------------------------------------------------------------------
	async def esg_analytics(self, tenant_id: str, period: str) -> dict[str, Any]:
		"""Aggregate ESG analytics across all entities."""
		activities = await self._store.list("esgc_activities", tenant_id)
		scores = await self._store.list("esgc_supplier_scores", tenant_id)
		offsets = await self._store.list("esgc_offsets", tenant_id)
		total_co2e = sum(a["co2e_tonnes"] for a in activities)
		score_vals = [s["score"] for s in scores]
		offsets_total = sum(o.get("credits_tco2e", 0.0) for o in offsets)
		return {
			"tenant_id": tenant_id, "period": period,
			"total_co2e_tonnes": round(total_co2e, 4),
			"net_co2e_tonnes": round(total_co2e - offsets_total, 4),
			"offset_credits_total": round(offsets_total, 4),
			"activity_count": len(activities),
			"scope_breakdown": {
				"scope_1": round(sum(a["co2e_tonnes"] for a in activities if a["scope"] == "scope_1"), 4),
				"scope_2": round(sum(a["co2e_tonnes"] for a in activities if a["scope"] == "scope_2"), 4),
				"scope_3": round(sum(a["co2e_tonnes"] for a in activities if a["scope"] == "scope_3"), 4),
			},
			"supplier_esg_score_count": len(scores),
			"avg_supplier_score": round(statistics.mean(score_vals), 2) if score_vals else None,
			"computed_at": _utc_now(),
		}

	# ------------------------------------------------------------------
	# 19. carbon_offset_purchase
	# ------------------------------------------------------------------
	async def carbon_offset_purchase(
		self,
		tenant_id: str,
		offset_id: str,
		registry: str,
		credits: float,
		project_name: str,
		amount: float,
		currency: str = "USD",
		verification_standard: str = "VCS",
	) -> dict[str, Any]:
		assert registry and project_name, "registry and project_name required"
		assert credits > 0 and amount > 0, "credits and amount must be positive"
		record = {
			"id": offset_id, "tenant_id": tenant_id, "registry": registry,
			"credits_tco2e": round(credits, 4), "project_name": project_name,
			"amount": amount, "currency": currency, "verification_standard": verification_standard,
			"status": "retired", "purchased_at": _utc_now(),
		}
		await self._store.put("esgc_offsets", record)
		await self._audit.log_event("carbon_offset_purchased", self.actor_id, tenant_id, offset_id, record)
		return record

	# ------------------------------------------------------------------
	# 20. net_zero_target_setting
	# ------------------------------------------------------------------
	async def net_zero_target_setting(
		self,
		tenant_id: str,
		target_id: str,
		entity_id: str,
		base_year: int,
		target_year: int,
		pathway: str,
		interim_milestones: list[dict[str, Any]] | None = None,
	) -> dict[str, Any]:
		assert base_year > 0 and target_year > base_year, "target_year must be after base_year"
		record = {
			"id": target_id, "tenant_id": tenant_id, "entity_id": entity_id,
			"base_year": base_year, "target_year": target_year, "pathway": pathway,
			"interim_milestones": list(interim_milestones or []),
			"status": "committed", "created_at": _utc_now(),
		}
		await self._store.put("esgc_net_zero", record)
		await self._audit.log_event("net_zero_target_set", self.actor_id, tenant_id, target_id, {"pathway": pathway, "target_year": target_year})
		return record

	# ------------------------------------------------------------------
	# 21. ghg_report
	# ------------------------------------------------------------------
	async def ghg_report(
		self,
		tenant_id: str,
		report_id: str,
		entity_id: str,
		standard: str,
		period: str,
		approved_by: str,
	) -> dict[str, Any]:
		assert standard and approved_by, "standard and approved_by required"
		s1 = await self.scope1_calculation(tenant_id, entity_id, period)
		s2 = await self.scope2_calculate(tenant_id, entity_id, period)
		s3 = await self.scope3_estimate(tenant_id, entity_id, "all", period)
		total = s1["total_co2e_tonnes"] + s2["market_based_co2e_tonnes"] + s3["total_co2e_tonnes"]
		offsets_total = sum(o.get("credits_tco2e", 0.0) for o in await self._store.list("esgc_offsets", tenant_id))
		report = {
			"id": report_id, "tenant_id": tenant_id, "entity_id": entity_id,
			"standard": standard, "period": period,
			"scope1_co2e_tonnes": s1["total_co2e_tonnes"],
			"scope2_co2e_tonnes": s2["market_based_co2e_tonnes"],
			"scope3_co2e_tonnes": s3["total_co2e_tonnes"],
			"gross_total_co2e_tonnes": round(total, 4),
			"offsets_co2e_tonnes": round(offsets_total, 4),
			"net_total_co2e_tonnes": round(total - offsets_total, 4),
			"approved_by": approved_by, "status": "published", "generated_at": _utc_now(),
		}
		await self._store.put("esgc_reports", report)
		await self._audit.log_event("ghg_report_generated", self.actor_id, tenant_id, report_id, {"period": period, "standard": standard})
		return report

	# ------------------------------------------------------------------
	# 22. esg_score_calculation
	# ------------------------------------------------------------------
	async def esg_score_calculation(
		self,
		tenant_id: str,
		score_id: str,
		entity_id: str,
		pillar: str,
		metrics: dict[str, float],
		weights: dict[str, float] | None = None,
	) -> dict[str, Any]:
		assert pillar in {"environmental", "social", "governance", "combined"}, f"invalid pillar:{pillar}"
		weights = weights or {k: 1.0 / len(metrics) for k in metrics}
		w_sum = sum(weights.values()) or 1.0
		nw = {k: v / w_sum for k, v in weights.items()}
		raw = sum(metrics.get(k, 0.0) * nw.get(k, 0.0) for k in metrics)
		final = max(0.0, min(100.0, round(raw, 2)))
		record = {
			"id": score_id, "tenant_id": tenant_id, "entity_id": entity_id,
			"pillar": pillar, "metrics": metrics, "weights": nw,
			"score": final, "rating": _esg_rating(final), "calculated_at": _utc_now(),
		}
		await self._store.put("esgc_scores", record)
		await self._audit.log_event("esg_score_calculated", self.actor_id, tenant_id, score_id, {"pillar": pillar, "score": final})
		return record

	# ------------------------------------------------------------------
	# 23. esg_disclosure_generation
	# ------------------------------------------------------------------
	async def esg_disclosure_generation(
		self,
		tenant_id: str,
		disclosure_id: str,
		entity_id: str,
		framework: str,
		period: str,
		prepared_by: str,
		sections: list[str] | None = None,
	) -> dict[str, Any]:
		if framework not in KNOWN_FRAMEWORKS:
			raise ValueError(f"unsupported_framework:{framework}")
		scores = [s for s in await self._store.list("esgc_scores", tenant_id) if s["entity_id"] == entity_id]
		record = {
			"id": disclosure_id, "tenant_id": tenant_id, "entity_id": entity_id,
			"framework": framework, "period": period, "prepared_by": prepared_by,
			"sections": sections or _default_sections(framework),
			"esg_scores_referenced": [s["id"] for s in scores],
			"status": "draft", "generated_at": _utc_now(),
		}
		await self._store.put("esgc_disclosures", record)
		await self._audit.log_event("disclosure_generated", self.actor_id, tenant_id, disclosure_id, {"framework": framework, "period": period})
		return record

	# ------------------------------------------------------------------
	# 24. publish_report
	# ------------------------------------------------------------------
	async def publish_report(
		self,
		report_id: str,
		tenant_id: str,
		inventory_id: str,
		report_type: str,
		period: str,
		compliance_mapping: str,
		audit_evidence_ref: str,
		approved_by: str,
		approval_recorded: bool,
	) -> dict[str, Any]:
		assert approval_recorded and approved_by, "approval required"
		inventory = await self._require_inventory(tenant_id, inventory_id)
		activities = [a for a in await self._store.list("esgc_activities", tenant_id) if a["inventory_id"] == inventory_id]
		total = round(sum(a["co2e_tonnes"] for a in activities), 4)
		record = {
			"id": report_id, "tenant_id": tenant_id, "inventory_id": inventory_id,
			"report_type": report_type, "period": period, "total_co2e_tonnes": total,
			"compliance_mapping": compliance_mapping, "audit_evidence_ref": audit_evidence_ref,
			"approved_by": approved_by, "status": "published", "published_at": _utc_now(),
		}
		await self._store.put("esgc_reports", record)
		await self._audit.log_event("report_published", self.actor_id, tenant_id, report_id, {"total_co2e_tonnes": total})
		return record

	# ------------------------------------------------------------------
	# 25. create_target
	# ------------------------------------------------------------------
	async def create_target(
		self,
		target_id: str,
		tenant_id: str,
		inventory_id: str,
		name: str,
		baseline_year: int,
		target_year: int,
		baseline_co2e_tonnes: float,
		target_reduction_percent: float,
	) -> dict[str, Any]:
		activities = [a for a in await self._store.list("esgc_activities", tenant_id) if a["inventory_id"] == inventory_id]
		current = round(sum(a["co2e_tonnes"] for a in activities), 4)
		target_abs = baseline_co2e_tonnes * (1 - target_reduction_percent / 100)
		progress = round((baseline_co2e_tonnes - current) / max(baseline_co2e_tonnes - target_abs, 0.0001) * 100, 2)
		record = {
			"id": target_id, "tenant_id": tenant_id, "inventory_id": inventory_id,
			"name": name, "baseline_year": baseline_year, "target_year": target_year,
			"baseline_co2e_tonnes": baseline_co2e_tonnes, "target_reduction_percent": target_reduction_percent,
			"current_co2e_tonnes": current, "progress_percent": min(100.0, max(0.0, progress)),
			"status": "on_track" if progress >= 0 else "off_track", "created_at": _utc_now(),
		}
		await self._store.put("esgc_targets", record)
		await self._audit.log_event("target_created", self.actor_id, tenant_id, target_id, {"progress_percent": record["progress_percent"]})
		return record

	# ------------------------------------------------------------------
	# 26. scope1_calculation (compat)
	# ------------------------------------------------------------------
	async def scope1_calculation(self, tenant_id: str, entity_id: str, period: str) -> dict[str, Any]:
		activities = [a for a in await self._store.list("esgc_activities", tenant_id)
					  if a["scope"] == "scope_1" and a["inventory_id"] == entity_id]
		total = sum(a["co2e_tonnes"] for a in activities)
		breakdown: dict[str, float] = {}
		for a in activities:
			breakdown[a["activity_type"]] = round(breakdown.get(a["activity_type"], 0.0) + a["co2e_tonnes"], 4)
		result = {
			"tenant_id": tenant_id, "entity_id": entity_id, "period": period, "scope": "scope_1",
			"total_co2e_tonnes": round(total, 4), "activity_count": len(activities),
			"breakdown_by_type": breakdown, "calculated_at": _utc_now(),
		}
		await self._audit.log_event("scope1_calculated", self.actor_id, tenant_id, entity_id, result)
		return result

	# ------------------------------------------------------------------
	# 27. bulk_record_activities
	# ------------------------------------------------------------------
	async def bulk_record_activities(self, tenant_id: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
		"""Bulk record emission activities in parallel."""
		tasks = [
			self.record_activity(
				item["activity_id"], tenant_id, item["inventory_id"], item["factor_id"],
				item["activity_type"], item["quantity"], item["unit"], item.get("evidence_ref", "")
			)
			for item in items
		]
		results = await asyncio.gather(*tasks, return_exceptions=True)
		out = []
		for item, res in zip(items, results):
			if isinstance(res, Exception):
				out.append({"activity_id": item["activity_id"], "status": "failed", "error": str(res)})
			else:
				out.append({**res, "status": "ok"})  # type: ignore[arg-type]
		await self._audit.log_event("bulk_activities_recorded", self.actor_id, tenant_id, "bulk", {"count": len(items)})
		return out

	# ------------------------------------------------------------------
	# 28. green_bond_reporting
	# ------------------------------------------------------------------
	async def green_bond_reporting(
		self,
		tenant_id: str,
		bond_id: str,
		use_of_proceeds: dict[str, float],
		impact_metrics: dict[str, Any],
		reporting_period: str,
		verifier: str = "",
	) -> dict[str, Any]:
		assert use_of_proceeds and impact_metrics and reporting_period, "all fields required"
		record = {
			"id": bond_id, "tenant_id": tenant_id, "reporting_period": reporting_period,
			"use_of_proceeds": use_of_proceeds, "total_proceeds": round(sum(use_of_proceeds.values()), 2),
			"impact_metrics": impact_metrics, "co2e_avoided_tonnes": float(impact_metrics.get("co2e_avoided_tonnes", 0.0)),
			"renewable_mwh": float(impact_metrics.get("renewable_mwh", 0.0)),
			"verifier": verifier, "standard": "ICMA_GBP", "status": "published", "generated_at": _utc_now(),
		}
		await self._store.put("esgc_green_bonds", record)
		await self._audit.log_event("green_bond_report_generated", self.actor_id, tenant_id, bond_id, {"period": reporting_period})
		return record

	# ------------------------------------------------------------------
	# 29. dashboard_summary
	# ------------------------------------------------------------------
	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		activities = await self._store.list("esgc_activities", tenant_id)
		return {
			"tenant_id": tenant_id,
			"inventory_count": len(await self._store.list("esgc_inventories", tenant_id)),
			"factor_count": len(await self._store.list("esgc_factors", tenant_id)),
			"activity_count": len(activities),
			"total_co2e_tonnes": round(sum(a["co2e_tonnes"] for a in activities), 4),
			"report_count": len(await self._store.list("esgc_reports", tenant_id)),
			"target_count": len(await self._store.list("esgc_targets", tenant_id)),
			"offset_count": len(await self._store.list("esgc_offsets", tenant_id)),
			"esg_score_count": len(await self._store.list("esgc_scores", tenant_id)),
			"disclosure_count": len(await self._store.list("esgc_disclosures", tenant_id)),
			"credit_trade_count": len(await self._store.list("esgc_credit_trades", tenant_id)),
			"audit_events": len(await self._store.list("esgc_audit", tenant_id)),
			"generated_at": _utc_now(),
		}

	# ------------------------------------------------------------------
	# 30. health_check
	# ------------------------------------------------------------------
	async def health_check(self) -> dict[str, Any]:
		try:
			test_id = f"_health_{uuid7str()}"
			await self.create_inventory(test_id, "_health", "HealthOrg", "system", 2026, "b", "g", "GHG Protocol")
			await self._store.delete("esgc_inventories", test_id)
			status = "healthy"
		except Exception as exc:
			status = f"degraded:{exc}"
		return {
			"service": "EsgcService", "status": status,
			"collections": {
				"inventories": len(await self._store.list("esgc_inventories")),
				"activities": len(await self._store.list("esgc_activities")),
				"audit_events": len(await self._store.list("esgc_audit")),
			},
			"checked_at": _utc_now(),
		}

	# ------------------------------------------------------------------
	# 31. export_csv
	# ------------------------------------------------------------------
	async def export_csv(self, tenant_id: str, collection: str = "esgc_activities") -> str:
		records = await self._store.list(collection, tenant_id)
		if not records:
			return ""
		buf = io.StringIO()
		writer = csv.DictWriter(buf, fieldnames=list(records[0].keys()))
		writer.writeheader()
		writer.writerows(records)
		return buf.getvalue()

	# ------------------------------------------------------------------
	# 32. export_json
	# ------------------------------------------------------------------
	async def export_json(self, tenant_id: str, collection: str = "esgc_activities") -> str:
		records = await self._store.list(collection, tenant_id)
		return json.dumps(records, indent=2, default=str)

	# ------------------------------------------------------------------
	# 33–42. list helpers
	# ------------------------------------------------------------------
	async def list_inventories(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("esgc_inventories", tenant_id)

	async def list_factors(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("esgc_factors", tenant_id)

	async def list_activities(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("esgc_activities", tenant_id)

	async def list_reports(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("esgc_reports", tenant_id)

	async def list_targets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("esgc_targets", tenant_id)

	async def list_offsets(self, tenant_id: str) -> list[dict[str, Any]]:
		return await self._store.list("esgc_offsets", tenant_id)

	async def list_esg_scores(self, tenant_id: str) -> list[dict[str, Any]]:
		return await self._store.list("esgc_scores", tenant_id)

	async def list_disclosures(self, tenant_id: str) -> list[dict[str, Any]]:
		return await self._store.list("esgc_disclosures", tenant_id)

	async def list_credit_trades(self, tenant_id: str) -> list[dict[str, Any]]:
		return await self._store.list("esgc_credit_trades", tenant_id)

	async def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("esgc_audit", tenant_id)

	# compat
	async def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self.list_activities(tenant_id)

	# ------------------------------------------------------------------
	# register_esgc_agent (compat)
	# ------------------------------------------------------------------
	async def register_esgc_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool = True,
		agent_id: str | None = None,
	) -> dict[str, Any]:
		record = {
			"id": agent_id or uuid7str(), "tenant_id": tenant_id, "name": name,
			"runtime": _normalize(runtime), "role": _normalize(role), "scope": scope,
			"contribution_disclosed": contribution_disclosed, "status": "active",
			"registered_at": _utc_now(),
		}
		await self._store.put("esgc_agents", record)
		await self._audit.log_event("agent_registered", self.actor_id, tenant_id, record["id"], {"role": role})
		return record

	async def list_esgc_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("esgc_agents", tenant_id)

	# ------------------------------------------------------------------
	# Internals
	# ------------------------------------------------------------------

	async def _require_inventory(self, tenant_id: str, inventory_id: str) -> dict[str, Any]:
		rec = await self._store.get("esgc_inventories", inventory_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			raise KeyError(f"inventory_not_found:{inventory_id}")
		return rec

	async def _require_factor(self, tenant_id: str, factor_id: str) -> dict[str, Any]:
		rec = await self._store.get("esgc_factors", factor_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			raise KeyError(f"factor_not_found:{factor_id}")
		return rec


__all__ = ["EsgcService"]
