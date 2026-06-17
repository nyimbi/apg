"""Actuarial Tools Service (ins_act).

Mortality tables, loss ratios, reserve calculations, IBNR, pricing models, experience analysis.
"""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import logging
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

RESERVE_METHODS = {"chain_ladder", "bornhuetter_ferguson", "cape_cod", "development", "average_cost_per_claim"}
TABLE_TYPES = {"life_1958_cso", "life_1980_cso", "life_2001_cso", "life_2017_cso", "general_industry", "population"}
DEVELOPMENT_METHODS = {"chain_ladder", "bornhuetter_ferguson", "frequency_severity", "cape_cod"}


class ActuarialToolsService:
	"""In-memory executable service for Actuarial Tools."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.mortality_tables: dict[str, dict[str, Any]] = {}
		self.loss_ratios: dict[str, dict[str, Any]] = {}
		self.reserve_calcs: dict[str, dict[str, Any]] = {}
		self.ibnr_estimates: dict[str, dict[str, Any]] = {}
		self.pricing_models: dict[str, dict[str, Any]] = {}
		self.experience_analyses: dict[str, dict[str, Any]] = {}
		self.claims_triangles: dict[str, dict[str, Any]] = {}
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

	# ── Mortality Tables ──────────────────────────────────────────────────────

	async def create_mortality_table(
		self,
		tenant_id: str,
		table_name: str,
		table_type: str,
		base_year: int,
		ages: list[int],
		qx_values: list[float],
		lx_values: list[float],
		source: str,
	) -> dict[str, Any]:
		"""Load a mortality (or morbidity) table."""
		tenant = self._tenant(tenant_id)
		if len(ages) != len(qx_values) or len(ages) != len(lx_values):
			raise ValueError("ages, qx_values and lx_values must have equal length")
		if any(q < 0 or q > 1 for q in qx_values):
			raise ValueError("qx_values must be between 0 and 1")
		record: dict[str, Any] = {
			"id": self._record_id("mort"),
			"type": "act_mortality_table",
			"table_name": table_name,
			"table_type": table_type,
			"base_year": base_year,
			"ages": list(ages),
			"qx_values": list(qx_values),
			"lx_values": list(lx_values),
			"source": source,
			"status": "active",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.mortality_tables[record["id"]] = record
		self._emit(tenant, "mortality_table_loaded", record["id"], "act_mortality_table", {"table_name": table_name})
		_log.info("Mortality table loaded: %s tenant=%s", table_name, tenant)
		return deepcopy(record)

	async def get_mortality_table(self, tenant_id: str, table_id: str) -> dict[str, Any]:
		"""Retrieve a mortality table."""
		tenant = self._tenant(tenant_id)
		tbl = self.mortality_tables.get(table_id)
		if not tbl or tbl["tenant_id"] != tenant:
			raise KeyError(f"mortality_table_not_found:{table_id}")
		return deepcopy(tbl)

	async def list_mortality_tables(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List mortality tables."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(t) for t in self.mortality_tables.values() if t["tenant_id"] == tenant]

	async def delete_mortality_table(self, tenant_id: str, table_id: str) -> dict[str, Any]:
		"""Retire a mortality table."""
		tenant = self._tenant(tenant_id)
		tbl = self.mortality_tables.get(table_id)
		if not tbl or tbl["tenant_id"] != tenant:
			raise KeyError(f"mortality_table_not_found:{table_id}")
		tbl["status"] = "retired"
		tbl["retired_at"] = self._now()
		self._emit(tenant, "mortality_table_retired", table_id, "act_mortality_table", {})
		return deepcopy(tbl)

	async def lookup_mortality_rate(self, tenant_id: str, table_id: str, age: int) -> dict[str, Any]:
		"""Look up mortality rate (qx) for a given age."""
		tenant = self._tenant(tenant_id)
		tbl = self.mortality_tables.get(table_id)
		if not tbl or tbl["tenant_id"] != tenant:
			raise KeyError(f"mortality_table_not_found:{table_id}")
		try:
			idx = tbl["ages"].index(age)
			return {
				"table_id": table_id,
				"table_name": tbl["table_name"],
				"age": age,
				"qx": tbl["qx_values"][idx],
				"lx": tbl["lx_values"][idx],
			}
		except ValueError:
			raise KeyError(f"age_not_in_table:{age}")

	# ── Loss Ratios ───────────────────────────────────────────────────────────

	async def calculate_loss_ratio(
		self,
		tenant_id: str,
		product_code: str,
		period_start: str,
		period_end: str,
		earned_premium: Decimal,
		incurred_losses: Decimal,
		expenses: Decimal = Decimal("0"),
	) -> dict[str, Any]:
		"""Calculate loss ratio and combined ratio."""
		tenant = self._tenant(tenant_id)
		ep = Decimal(str(earned_premium))
		il = Decimal(str(incurred_losses))
		exp = Decimal(str(expenses))
		if ep <= 0:
			raise ValueError("earned_premium_must_be_positive")
		loss_ratio = (il / ep * 100).quantize(Decimal("0.01"))
		expense_ratio = (exp / ep * 100).quantize(Decimal("0.01")) if exp > 0 else Decimal("0")
		combined_ratio = (loss_ratio + expense_ratio).quantize(Decimal("0.01"))
		record: dict[str, Any] = {
			"id": self._record_id("lr"),
			"type": "act_loss_ratio",
			"product_code": product_code,
			"period_start": period_start,
			"period_end": period_end,
			"earned_premium": ep,
			"incurred_losses": il,
			"expenses": exp,
			"loss_ratio_pct": loss_ratio,
			"expense_ratio_pct": expense_ratio,
			"combined_ratio_pct": combined_ratio,
			"profitable": combined_ratio < 100,
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.loss_ratios[record["id"]] = record
		self._emit(tenant, "loss_ratio_calculated", record["id"], "act_loss_ratio", {"product_code": product_code, "lr": str(loss_ratio)})
		return deepcopy(record)

	async def list_loss_ratios(self, tenant_id: str, product_code: str | None = None) -> list[dict[str, Any]]:
		"""List loss ratio reports."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.loss_ratios.values() if r["tenant_id"] == tenant]
		if product_code:
			items = [r for r in items if r["product_code"] == product_code]
		return items

	async def get_loss_ratio(self, tenant_id: str, report_id: str) -> dict[str, Any]:
		"""Get a loss ratio report."""
		tenant = self._tenant(tenant_id)
		rpt = self.loss_ratios.get(report_id)
		if not rpt or rpt["tenant_id"] != tenant:
			raise KeyError(f"loss_ratio_report_not_found:{report_id}")
		return deepcopy(rpt)

	# ── Reserve Calculations ──────────────────────────────────────────────────

	async def calculate_reserve(
		self,
		tenant_id: str,
		product_code: str,
		valuation_date: str,
		method: str,
		gross_claims_paid: Decimal,
		gross_claims_outstanding: Decimal,
		reinsurance_recoverable: Decimal = Decimal("0"),
		assumptions: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Calculate technical reserves (gross and net)."""
		tenant = self._tenant(tenant_id)
		if method not in RESERVE_METHODS:
			raise ValueError(f"unsupported_reserve_method:{method}")
		paid = Decimal(str(gross_claims_paid))
		outstanding = Decimal(str(gross_claims_outstanding))
		ri_rec = Decimal(str(reinsurance_recoverable))
		gross_reserve = paid + outstanding
		net_reserve = gross_reserve - ri_rec
		if net_reserve < 0:
			net_reserve = Decimal("0")
		record: dict[str, Any] = {
			"id": self._record_id("rsv"),
			"type": "act_reserve_calculation",
			"product_code": product_code,
			"valuation_date": valuation_date,
			"method": method,
			"gross_claims_paid": paid,
			"gross_claims_outstanding": outstanding,
			"gross_reserve": gross_reserve,
			"reinsurance_recoverable": ri_rec,
			"net_reserve": net_reserve,
			"assumptions": deepcopy(assumptions or {}),
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.reserve_calcs[record["id"]] = record
		self._emit(tenant, "reserve_calculated", record["id"], "act_reserve_calculation", {"product_code": product_code})
		return deepcopy(record)

	async def list_reserves(self, tenant_id: str, product_code: str | None = None) -> list[dict[str, Any]]:
		"""List reserve calculations."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.reserve_calcs.values() if r["tenant_id"] == tenant]
		if product_code:
			items = [r for r in items if r["product_code"] == product_code]
		return items

	async def get_reserve(self, tenant_id: str, reserve_id: str) -> dict[str, Any]:
		"""Retrieve a reserve calculation."""
		tenant = self._tenant(tenant_id)
		rsv = self.reserve_calcs.get(reserve_id)
		if not rsv or rsv["tenant_id"] != tenant:
			raise KeyError(f"reserve_not_found:{reserve_id}")
		return deepcopy(rsv)

	# ── IBNR ──────────────────────────────────────────────────────────────────

	async def estimate_ibnr(
		self,
		tenant_id: str,
		product_code: str,
		valuation_date: str,
		development_method: str,
		triangle_data: list[list[float]],
		confidence_level: float = 0.75,
	) -> dict[str, Any]:
		"""Estimate Incurred But Not Reported (IBNR) reserves."""
		tenant = self._tenant(tenant_id)
		if development_method not in DEVELOPMENT_METHODS:
			raise ValueError(f"unsupported_development_method:{development_method}")
		if not (0 < confidence_level < 1):
			raise ValueError("confidence_level must be between 0 and 1")
		periods = len(triangle_data)
		# Chain-ladder development factor approximation
		if triangle_data and triangle_data[0]:
			latest_diagonal = [row[-1] for row in triangle_data if row]
			total_latest = sum(latest_diagonal)
			ibnr_factor = Decimal(str(confidence_level * 0.15))
			ibnr_amount = (Decimal(str(total_latest)) * ibnr_factor).quantize(Decimal("0.01"))
		else:
			ibnr_amount = Decimal("0")
		record: dict[str, Any] = {
			"id": self._record_id("ibnr"),
			"type": "act_ibnr_estimate",
			"product_code": product_code,
			"valuation_date": valuation_date,
			"development_method": development_method,
			"triangle_periods": periods,
			"ibnr_amount": ibnr_amount,
			"confidence_level": confidence_level,
			"triangle_data_summary": {"rows": periods, "method": development_method},
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.ibnr_estimates[record["id"]] = record
		self._emit(tenant, "ibnr_estimated", record["id"], "act_ibnr_estimate", {"product_code": product_code, "amount": str(ibnr_amount)})
		return deepcopy(record)

	async def list_ibnr_estimates(self, tenant_id: str, product_code: str | None = None) -> list[dict[str, Any]]:
		"""List IBNR estimates."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(e) for e in self.ibnr_estimates.values() if e["tenant_id"] == tenant]
		if product_code:
			items = [e for e in items if e["product_code"] == product_code]
		return items

	async def delete_ibnr_estimate(self, tenant_id: str, ibnr_id: str) -> dict[str, Any]:
		"""Remove an IBNR estimate."""
		tenant = self._tenant(tenant_id)
		est = self.ibnr_estimates.get(ibnr_id)
		if not est or est["tenant_id"] != tenant:
			raise KeyError(f"ibnr_estimate_not_found:{ibnr_id}")
		del self.ibnr_estimates[ibnr_id]
		self._emit(tenant, "ibnr_estimate_deleted", ibnr_id, "act_ibnr_estimate", {})
		return {"id": ibnr_id, "status": "deleted"}

	# ── Pricing Models ────────────────────────────────────────────────────────

	async def create_pricing_model(
		self,
		tenant_id: str,
		model_name: str,
		product_code: str,
		risk_factors: list[str],
		base_rate: Decimal,
		parameters: dict[str, Any] | None = None,
		effective_date: str | None = None,
	) -> dict[str, Any]:
		"""Register a pricing model."""
		tenant = self._tenant(tenant_id)
		if any(m["model_name"] == model_name and m["tenant_id"] == tenant for m in self.pricing_models.values()):
			raise ValueError(f"pricing_model_name_duplicate:{model_name}")
		record: dict[str, Any] = {
			"id": self._record_id("pm"),
			"type": "act_pricing_model",
			"model_name": model_name,
			"product_code": product_code,
			"risk_factors": list(risk_factors),
			"base_rate": Decimal(str(base_rate)),
			"parameters": deepcopy(parameters or {}),
			"effective_date": effective_date or date.today().isoformat(),
			"status": "active",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.pricing_models[record["id"]] = record
		self._emit(tenant, "pricing_model_created", record["id"], "act_pricing_model", {"model_name": model_name})
		return deepcopy(record)

	async def list_pricing_models(self, tenant_id: str, product_code: str | None = None) -> list[dict[str, Any]]:
		"""List pricing models."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(m) for m in self.pricing_models.values() if m["tenant_id"] == tenant and m["status"] == "active"]
		if product_code:
			items = [m for m in items if m["product_code"] == product_code]
		return items

	async def get_pricing_model(self, tenant_id: str, model_id: str) -> dict[str, Any]:
		"""Retrieve a pricing model."""
		tenant = self._tenant(tenant_id)
		mdl = self.pricing_models.get(model_id)
		if not mdl or mdl["tenant_id"] != tenant:
			raise KeyError(f"pricing_model_not_found:{model_id}")
		return deepcopy(mdl)

	async def apply_pricing_model(self, tenant_id: str, model_id: str, risk_data: dict[str, Any]) -> dict[str, Any]:
		"""Apply a pricing model to compute a premium for given risk data."""
		tenant = self._tenant(tenant_id)
		mdl = self.pricing_models.get(model_id)
		if not mdl or mdl["tenant_id"] != tenant:
			raise KeyError(f"pricing_model_not_found:{model_id}")
		sum_insured = Decimal(str(risk_data.get("sum_insured", 1000000)))
		base_rate = mdl["base_rate"]
		# Apply factor adjustments based on risk_factors presence in risk_data
		factor = Decimal("1.0")
		applied_factors: dict[str, str] = {}
		for rf in mdl["risk_factors"]:
			if rf in risk_data:
				adj = Decimal(str(mdl["parameters"].get(rf, 0)))
				factor += adj
				applied_factors[rf] = str(adj)
		premium = (sum_insured * base_rate * factor).quantize(Decimal("0.01"))
		return {
			"model_id": model_id,
			"model_name": mdl["model_name"],
			"sum_insured": str(sum_insured),
			"base_rate": str(base_rate),
			"applied_factors": applied_factors,
			"total_factor": str(factor),
			"computed_premium": str(premium),
			"computed_at": self._now(),
		}

	# ── Experience Analysis ───────────────────────────────────────────────────

	async def run_experience_analysis(
		self,
		tenant_id: str,
		product_code: str,
		analysis_period_years: int,
		actual_claims: int,
		expected_claims: int,
		actual_loss_amount: Decimal,
		expected_loss_amount: Decimal,
	) -> dict[str, Any]:
		"""Perform actual vs expected experience analysis."""
		tenant = self._tenant(tenant_id)
		if expected_claims <= 0:
			raise ValueError("expected_claims_must_be_positive")
		ae_frequency = round(actual_claims / expected_claims, 4)
		exp_loss = Decimal(str(expected_loss_amount))
		act_loss = Decimal(str(actual_loss_amount))
		ae_severity = float((act_loss / exp_loss).quantize(Decimal("0.0001"))) if exp_loss > 0 else 0.0
		record: dict[str, Any] = {
			"id": self._record_id("exp"),
			"type": "act_experience_analysis",
			"product_code": product_code,
			"analysis_period_years": analysis_period_years,
			"actual_claims": actual_claims,
			"expected_claims": expected_claims,
			"ae_frequency_ratio": ae_frequency,
			"actual_loss_amount": act_loss,
			"expected_loss_amount": exp_loss,
			"ae_severity_ratio": ae_severity,
			"credibility": min(actual_claims / (actual_claims + 100), 1.0),
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.experience_analyses[record["id"]] = record
		self._emit(tenant, "experience_analysis_run", record["id"], "act_experience_analysis", {"product_code": product_code})
		return deepcopy(record)

	async def list_experience_analyses(self, tenant_id: str, product_code: str | None = None) -> list[dict[str, Any]]:
		"""List experience analyses."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(a) for a in self.experience_analyses.values() if a["tenant_id"] == tenant]
		if product_code:
			items = [a for a in items if a["product_code"] == product_code]
		return items

	async def get_experience_analysis(self, tenant_id: str, analysis_id: str) -> dict[str, Any]:
		"""Retrieve an experience analysis."""
		tenant = self._tenant(tenant_id)
		ana = self.experience_analyses.get(analysis_id)
		if not ana or ana["tenant_id"] != tenant:
			raise KeyError(f"experience_analysis_not_found:{analysis_id}")
		return deepcopy(ana)

	# ── Claims Triangle ───────────────────────────────────────────────────────

	async def upload_claims_triangle(
		self,
		tenant_id: str,
		product_code: str,
		valuation_date: str,
		accident_years: list[int],
		development_periods: list[int],
		cumulative_data: list[list[float]],
	) -> dict[str, Any]:
		"""Store a claims development triangle."""
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": self._record_id("tri"),
			"type": "act_claims_triangle",
			"product_code": product_code,
			"valuation_date": valuation_date,
			"accident_years": list(accident_years),
			"development_periods": list(development_periods),
			"cumulative_data": deepcopy(cumulative_data),
			"rows": len(accident_years),
			"cols": len(development_periods),
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.claims_triangles[record["id"]] = record
		self._emit(tenant, "claims_triangle_uploaded", record["id"], "act_claims_triangle", {"product_code": product_code})
		return deepcopy(record)

	async def list_claims_triangles(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List claims development triangles."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(t) for t in self.claims_triangles.values() if t["tenant_id"] == tenant]

	# ── Health & Summary ──────────────────────────────────────────────────────

	async def actuarial_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Summary of actuarial work product."""
		tenant = self._tenant(tenant_id)
		return {
			"tenant_id": tenant,
			"mortality_tables": len([t for t in self.mortality_tables.values() if t["tenant_id"] == tenant and t["status"] == "active"]),
			"loss_ratio_reports": len([r for r in self.loss_ratios.values() if r["tenant_id"] == tenant]),
			"reserve_calculations": len([r for r in self.reserve_calcs.values() if r["tenant_id"] == tenant]),
			"ibnr_estimates": len([e for e in self.ibnr_estimates.values() if e["tenant_id"] == tenant]),
			"pricing_models": len([m for m in self.pricing_models.values() if m["tenant_id"] == tenant and m["status"] == "active"]),
			"experience_analyses": len([a for a in self.experience_analyses.values() if a["tenant_id"] == tenant]),
			"generated_at": self._now(),
		}

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "ins_act",
			"status": "healthy",
			"mortality_table_count": len(self.mortality_tables),
			"pricing_model_count": len(self.pricing_models),
			"checked_at": self._now(),
		}

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return {
			"capability_id": "ins_act",
			"name": "Actuarial Tools",
			"version": "1.0.0",
			"domain": "insurance",
			"tenant_id": tenant_id,
			"reserve_methods": list(RESERVE_METHODS),
			"development_methods": list(DEVELOPMENT_METHODS),
			"table_types": list(TABLE_TYPES),
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

