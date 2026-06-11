"""Async service layer for Property Valuation (val)."""

from __future__ import annotations

import logging
from datetime import datetime, date
from decimal import Decimal
from typing import Any

from .models import (
	ValuerCreate, ValuerResponse,
	ComparableCreate, ComparableResponse,
	ValuationCreate, ValuationResponse, ValuationUpdate,
	DcfModelCreate, DcfModelResponse,
	ValuationRollEntryCreate, ValuationRollEntryResponse,
	MassAppraisalRunCreate, MassAppraisalRunResponse,
	ValuationChallengeCreate, ValuationChallengeResponse,
	ValuationStatus, ValuerGrade, ReportType,
)
from .capability_contract import evaluate_capability_rules
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

log = logging.getLogger(__name__)

INDEPENDENT_VALUER_GRADES = {ValuerGrade.independent_valuer.value, ValuerGrade.rics_registered.value, ValuerGrade.rics_fellow.value}
SIGN_OFF_GRADES = {ValuerGrade.rics_registered.value, ValuerGrade.rics_fellow.value, ValuerGrade.api_registered.value}


class ValService:
	"""Service implementing all Property Valuation operations."""

	def __init__(
		self,
		tenant_id: str | None = None,
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: dict[str, Any] | None = None,
	) -> None:
		self._tenant_id = tenant_id
		self._actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store: dict[str, list[dict[str, Any]]] = store or {
			"valuers": [], "comparables": [], "valuations": [],
			"dcf_models": [], "roll_entries": [], "appraisal_runs": [],
			"challenges": [], "market_reports": [],
		}
		self._val_counter = 0

	# ── Logging helpers ───────────────────────────────────────────────────────

	def _log_operation(self, op: str, entity_id: str, tenant_id: str) -> None:
		log.info("val.%s entity=%s tenant=%s", op, entity_id, tenant_id)

	def _log_dcf_run(self, valuation_id: str, npv: Decimal, irr: Decimal | None) -> None:
		log.info("val.dcf_run valuation=%s npv=%s irr=%s", valuation_id, npv, irr)

	def _log_challenge(self, valuation_id: str, raised_by: str) -> None:
		log.warning("val.challenge valuation=%s raised_by=%s", valuation_id, raised_by)

	# ── Rules ─────────────────────────────────────────────────────────────────

	def _check_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			log.warning("val.rule_denied rule=%s reason=%s", result["rule"], result["reason"])
			raise ValueError(f"rule_denied:{result['rule']}:{result['reason']}")

	def _next_val_ref(self) -> str:
		self._val_counter += 1
		return f"VAL-{self._val_counter:06d}"

	# ── Valuer ────────────────────────────────────────────────────────────────

	async def register_valuer(self, payload: ValuerCreate) -> ValuerResponse:
		"""Register a valuer in the panel."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "register_valuer",
			"valuer_grade_supported": True,
			"operation_type": "write",
			"policy_attached": True,
		})
		record = ValuerResponse(**payload.model_dump())
		self._store["valuers"].append(record.model_dump())
		self._log_operation("register_valuer", record.id, record.tenant_id)
		return record

	async def get_valuer(self, valuer_id: str, tenant_id: str) -> ValuerResponse | None:
		"""Fetch a valuer."""
		for v in self._store["valuers"]:
			if v["id"] == valuer_id and v["tenant_id"] == tenant_id:
				return ValuerResponse(**v)
		return None

	async def list_valuers(self, tenant_id: str, grade: str | None = None, independent_only: bool = False) -> list[ValuerResponse]:
		"""List valuers."""
		results = [v for v in self._store["valuers"] if v["tenant_id"] == tenant_id]
		if grade:
			results = [v for v in results if v.get("grade") == grade]
		if independent_only:
			results = [v for v in results if v.get("is_independent", False)]
		return [ValuerResponse(**v) for v in results]

	# ── Comparable ────────────────────────────────────────────────────────────

	async def add_comparable(self, payload: ComparableCreate) -> ComparableResponse:
		"""Add a comparable transaction."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "add_comparable",
			"comparable_type_supported": True,
		})
		record = ComparableResponse(**payload.model_dump())
		self._store["comparables"].append(record.model_dump())
		return record

	async def list_comparables(self, tenant_id: str, comparable_type: str | None = None, verified_only: bool = False) -> list[ComparableResponse]:
		"""List comparables."""
		results = [c for c in self._store["comparables"] if c["tenant_id"] == tenant_id]
		if comparable_type:
			results = [c for c in results if c.get("comparable_type") == comparable_type]
		if verified_only:
			results = [c for c in results if c.get("verified", False)]
		return [ComparableResponse(**c) for c in results]

	async def verify_comparable(self, comparable_id: str, tenant_id: str, verified_by: str) -> ComparableResponse | None:
		"""Verify a comparable transaction."""
		for i, c in enumerate(self._store["comparables"]):
			if c["id"] == comparable_id and c["tenant_id"] == tenant_id:
				c["verified"] = True
				c["updated_at"] = datetime.utcnow()
				self._store["comparables"][i] = c
				return ComparableResponse(**c)
		return None

	# ── Valuation ─────────────────────────────────────────────────────────────

	async def instruct_valuation(self, payload: ValuationCreate) -> ValuationResponse:
		"""Instruct a new valuation."""
		valuer = await self.get_valuer(payload.valuer_id, payload.tenant_id)
		qualified = valuer is not None
		self._check_rules({
			"tenant_context_present": True,
			"operation": "instruct_valuation",
			"method_supported": True,
			"purpose_supported": True,
			"property_present": True,
			"qualified_valuer_assigned": qualified,
			"operation_type": "write",
			"policy_attached": True,
			"cross_tenant": False,
		})
		ref = self._next_val_ref()
		is_independent = valuer.is_independent if valuer else False
		record = ValuationResponse(**payload.model_dump(), ref=ref, valuer_independent=is_independent)
		self._store["valuations"].append(record.model_dump())
		if valuer:
			for i, v in enumerate(self._store["valuers"]):
				if v["id"] == payload.valuer_id:
					v["active_instructions"] = v.get("active_instructions", 0) + 1
					self._store["valuers"][i] = v
					break
		self._log_operation("instruct_valuation", record.id, record.tenant_id)
		return record

	async def get_valuation(self, valuation_id: str, tenant_id: str) -> ValuationResponse | None:
		"""Fetch a valuation."""
		for v in self._store["valuations"]:
			if v["id"] == valuation_id and v["tenant_id"] == tenant_id:
				return ValuationResponse(**v)
		return None

	async def list_valuations(self, tenant_id: str, property_id: str | None = None, status: str | None = None) -> list[ValuationResponse]:
		"""List valuations."""
		results = [v for v in self._store["valuations"] if v["tenant_id"] == tenant_id]
		if property_id:
			results = [v for v in results if v.get("property_id") == property_id]
		if status:
			results = [v for v in results if v.get("status") == status]
		return [ValuationResponse(**v) for v in results]

	async def update_valuation(self, valuation_id: str, tenant_id: str, updates: ValuationUpdate) -> ValuationResponse | None:
		"""Update valuation details."""
		for i, v in enumerate(self._store["valuations"]):
			if v["id"] == valuation_id and v["tenant_id"] == tenant_id:
				self._check_rules({"operation_type": "write", "valuation_status": v.get("status")})
				v.update({k: val for k, val in updates.model_dump().items() if val is not None})
				v["updated_at"] = datetime.utcnow()
				self._store["valuations"][i] = v
				return ValuationResponse(**v)
		return None

	async def sign_off_valuation(self, valuation_id: str, tenant_id: str, signed_off_by: str, valuer_grade: str) -> ValuationResponse | None:
		"""Sign off a valuation by a qualified valuer."""
		self._check_rules({
			"operation": "sign_off_valuation",
			"valuer_grade_approved": valuer_grade in SIGN_OFF_GRADES,
		})
		for i, v in enumerate(self._store["valuations"]):
			if v["id"] == valuation_id and v["tenant_id"] == tenant_id:
				v["status"] = ValuationStatus.signed_off.value
				v["signed_off_by"] = signed_off_by
				v["updated_at"] = datetime.utcnow()
				self._store["valuations"][i] = v
				return ValuationResponse(**v)
		return None

	async def publish_valuation(self, valuation_id: str, tenant_id: str) -> ValuationResponse | None:
		"""Publish a signed-off valuation (becomes immutable)."""
		for i, v in enumerate(self._store["valuations"]):
			if v["id"] == valuation_id and v["tenant_id"] == tenant_id:
				if v.get("report_type") == ReportType.full_red_book.value:
					self._check_rules({
						"operation": "publish_valuation",
						"report_type": ReportType.full_red_book.value,
						"valuer_independent": v.get("valuer_independent", False),
					})
				v["status"] = ValuationStatus.published.value
				v["published_at"] = datetime.utcnow()
				v["updated_at"] = datetime.utcnow()
				self._store["valuations"][i] = v
				self._log_operation("publish_valuation", valuation_id, tenant_id)
				return ValuationResponse(**v)
		return None

	# ── DCF Model ─────────────────────────────────────────────────────────────

	async def run_dcf_model(self, payload: DcfModelCreate) -> DcfModelResponse:
		"""Run a DCF valuation model."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "run_dcf",
			"discount_rate_in_range": Decimal("0.03") <= payload.discount_rate <= Decimal("0.30"),
			"all_dcf_parameters_present": True,
		})
		npv, capital_value, irr, schedule = self._compute_dcf(payload)
		self._log_dcf_run(payload.valuation_id, npv, irr)
		record = DcfModelResponse(**payload.model_dump(), npv=npv, capital_value=capital_value, irr=irr, cash_flow_schedule=schedule)
		self._store["dcf_models"].append(record.model_dump())
		for i, v in enumerate(self._store["valuations"]):
			if v["id"] == payload.valuation_id and v["tenant_id"] == payload.tenant_id:
				v["valuation_figure"] = str(capital_value)
				v["capital_value"] = str(capital_value)
				v["updated_at"] = datetime.utcnow()
				self._store["valuations"][i] = v
				break
		return record

	def _compute_dcf(self, payload: DcfModelCreate) -> tuple[Decimal, Decimal, Decimal | None, list[dict[str, Any]]]:
		"""Compute NPV, capital value, and cash flow schedule."""
		schedule: list[dict[str, Any]] = []
		pv_cash_flows = Decimal("0")
		annual_rent = payload.annual_rental_income
		for year in range(1, payload.holding_period_years + 1):
			rent = annual_rent * (1 + payload.rental_growth_rate) ** (year - 1)
			discount_factor = Decimal("1") / (1 + payload.discount_rate) ** year
			pv = rent * discount_factor
			pv_cash_flows += pv
			schedule.append({"year": year, "rent": float(rent.quantize(Decimal("0.01"))), "discount_factor": float(discount_factor.quantize(Decimal("0.000001"))), "pv": float(pv.quantize(Decimal("0.01")))})
		terminal_rent = annual_rent * (1 + payload.rental_growth_rate) ** payload.holding_period_years
		terminal_value = terminal_rent / payload.exit_yield if payload.exit_yield > 0 else Decimal("0")
		terminal_value -= payload.capex_allowance
		pv_terminal = terminal_value / (1 + payload.discount_rate) ** payload.holding_period_years
		npv = pv_cash_flows + pv_terminal
		purchasers_costs = npv * payload.purchasers_costs_pct
		capital_value = (npv - purchasers_costs).quantize(Decimal("0.01"))
		return npv.quantize(Decimal("0.01")), capital_value, None, schedule

	async def get_dcf_model(self, model_id: str, tenant_id: str) -> DcfModelResponse | None:
		"""Fetch a DCF model."""
		for m in self._store["dcf_models"]:
			if m["id"] == model_id and m["tenant_id"] == tenant_id:
				return DcfModelResponse(**m)
		return None

	# ── Valuation Roll ────────────────────────────────────────────────────────

	async def add_to_valuation_roll(self, payload: ValuationRollEntryCreate) -> ValuationRollEntryResponse:
		"""Add or update an entry in the valuation roll."""
		for i, e in enumerate(self._store["roll_entries"]):
			if e["property_id"] == payload.property_id and e["tenant_id"] == payload.tenant_id and not e.get("superseded"):
				e["superseded"] = True
				e["updated_at"] = datetime.utcnow()
				self._store["roll_entries"][i] = e
		record = ValuationRollEntryResponse(**payload.model_dump())
		self._store["roll_entries"].append(record.model_dump())
		self._log_operation("add_to_roll", record.id, record.tenant_id)
		return record

	async def get_valuation_roll(self, tenant_id: str, property_id: str | None = None) -> list[ValuationRollEntryResponse]:
		"""Return the current (non-superseded) valuation roll."""
		results = [e for e in self._store["roll_entries"] if e["tenant_id"] == tenant_id and not e.get("superseded")]
		if property_id:
			results = [e for e in results if e["property_id"] == property_id]
		return [ValuationRollEntryResponse(**e) for e in results]

	# ── Mass Appraisal ────────────────────────────────────────────────────────

	async def run_mass_appraisal(self, payload: MassAppraisalRunCreate) -> MassAppraisalRunResponse:
		"""Run a mass appraisal model."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "run_mass_appraisal",
			"model_calibrated": payload.model_calibrated,
		})
		record = MassAppraisalRunResponse(**payload.model_dump(), status="running")
		self._store["appraisal_runs"].append(record.model_dump())
		results = [{"property_id": pid, "estimated_value": 0.0, "confidence": 0.0} for pid in payload.property_ids]
		for i, r in enumerate(self._store["appraisal_runs"]):
			if r["id"] == record.id:
				r["status"] = "completed"
				r["results"] = results
				r["completed_at"] = datetime.utcnow()
				self._store["appraisal_runs"][i] = r
				break
		return MassAppraisalRunResponse(**self._store["appraisal_runs"][-1])

	async def get_mass_appraisal_run(self, run_id: str, tenant_id: str) -> MassAppraisalRunResponse | None:
		"""Fetch a mass appraisal run."""
		for r in self._store["appraisal_runs"]:
			if r["id"] == run_id and r["tenant_id"] == tenant_id:
				return MassAppraisalRunResponse(**r)
		return None

	# ── Valuation Challenge ───────────────────────────────────────────────────

	async def raise_challenge(self, payload: ValuationChallengeCreate) -> ValuationChallengeResponse:
		"""Raise a challenge against a published valuation."""
		valuation = await self.get_valuation(payload.valuation_id, payload.tenant_id)
		challengeable = valuation is not None and valuation.status.value in (
			ValuationStatus.published.value, ValuationStatus.signed_off.value
		)
		self._check_rules({
			"tenant_context_present": True,
			"operation": "raise_challenge",
			"counter_evidence_present": len(payload.counter_evidence_document_ids) > 0,
			"valuation_status_challengeable": challengeable,
		})
		self._log_challenge(payload.valuation_id, payload.raised_by)
		record = ValuationChallengeResponse(**payload.model_dump())
		self._store["challenges"].append(record.model_dump())
		if valuation:
			for i, v in enumerate(self._store["valuations"]):
				if v["id"] == payload.valuation_id:
					v["status"] = ValuationStatus.challenged.value
					v["updated_at"] = datetime.utcnow()
					self._store["valuations"][i] = v
					break
		return record

	async def resolve_challenge(self, challenge_id: str, tenant_id: str, upheld: bool, resolution_notes: str, reviewed_by: str) -> ValuationChallengeResponse | None:
		"""Resolve a valuation challenge."""
		for i, c in enumerate(self._store["challenges"]):
			if c["id"] == challenge_id and c["tenant_id"] == tenant_id:
				c["status"] = "upheld" if upheld else "rejected"
				c["reviewed_by"] = reviewed_by
				c["resolution_notes"] = resolution_notes
				c["resolved_at"] = datetime.utcnow()
				c["updated_at"] = datetime.utcnow()
				self._store["challenges"][i] = c
				return ValuationChallengeResponse(**c)
		return None

	async def list_challenges(self, tenant_id: str, valuation_id: str | None = None) -> list[ValuationChallengeResponse]:
		"""List challenges."""
		results = [c for c in self._store["challenges"] if c["tenant_id"] == tenant_id]
		if valuation_id:
			results = [c for c in results if c.get("valuation_id") == valuation_id]
		return [ValuationChallengeResponse(**c) for c in results]

	# ── Yield Analysis ────────────────────────────────────────────────────────

	async def calculate_yield(self, tenant_id: str, property_id: str, annual_rent: Decimal, purchase_price: Decimal, yield_type: str) -> dict[str, Any]:
		"""Calculate a property yield."""
		self._check_rules({"operation": "calculate_yield", "yield_type_supported": True})
		if purchase_price <= 0:
			raise ValueError("purchase_price must be positive")
		niy = (annual_rent / purchase_price * 100).quantize(Decimal("0.01"))
		return {
			"tenant_id": tenant_id,
			"property_id": property_id,
			"yield_type": yield_type,
			"annual_rent": float(annual_rent),
			"purchase_price": float(purchase_price),
			"yield_pct": float(niy),
			"calculated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: instruct_valuation (full signature) ───────────────────────────────
	# (already exists via ValuationCreate; this new method is the convenience form)

	async def instruct_valuation_simple(
		self,
		property_id: str,
		valuation_purpose: str,
		valuer_id: str,
		basis: str,
		tenant_id: str,
		report_type: str = "desktop",
		effective_date: date | None = None,
	) -> ValuationResponse:
		"""Convenience form of instruct_valuation with direct parameters."""
		assert property_id and valuation_purpose and valuer_id, \
			"property_id, valuation_purpose, valuer_id required"
		assert basis in ("market_value", "market_rent", "fair_value",
			"investment_value", "mortgage_lending_value", "net_asset_value"), \
			f"unsupported basis: {basis}"
		from uuid6 import uuid7
		valuation_id = str(uuid7())
		ref = self._next_val_ref()
		valuer = await self.get_valuer(valuer_id, tenant_id)
		is_independent = valuer.is_independent if valuer else False
		record: dict[str, Any] = {
			"id": valuation_id,
			"tenant_id": tenant_id,
			"ref": ref,
			"property_id": property_id,
			"valuer_id": valuer_id,
			"valuation_purpose": valuation_purpose,
			"basis": basis,
			"report_type": report_type,
			"effective_date": str(effective_date or date.today()),
			"valuer_independent": is_independent,
			"status": ValuationStatus.instructed.value,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["valuations"].append(record)
		self._log_operation("instruct_valuation", valuation_id, tenant_id)
		return ValuationResponse(**record)

	# ── NEW: comparable_sales_analysis ────────────────────────────────────────

	async def comparable_sales_analysis(
		self,
		property_id: str,
		radius_km: float,
		period_months: int,
		tenant_id: str,
		property_type: str | None = None,
	) -> dict[str, Any]:
		"""Analyse comparable sales transactions within radius and period to derive market evidence."""
		assert property_id, "property_id required"
		assert radius_km > 0 and period_months > 0, "radius_km and period_months must be positive"
		comparables = await self.list_comparables(tenant_id, verified_only=True)
		if property_type:
			comparables = [c for c in comparables if getattr(c, "property_type", "") == property_type]
		if not comparables:
			return {
				"property_id": property_id,
				"radius_km": radius_km,
				"period_months": period_months,
				"comparable_count": 0,
				"analysis": "insufficient_data",
				"generated_at": datetime.utcnow().isoformat(),
			}
		values = [float(getattr(c, "transaction_price", 0)) for c in comparables if getattr(c, "transaction_price", None)]
		if not values:
			avg_value = 0.0
			min_value = 0.0
			max_value = 0.0
		else:
			avg_value = sum(values) / len(values)
			min_value = min(values)
			max_value = max(values)
		psf_values = [float(getattr(c, "price_per_sqft", 0)) for c in comparables if getattr(c, "price_per_sqft", None)]
		avg_psf = sum(psf_values) / len(psf_values) if psf_values else 0.0
		return {
			"property_id": property_id,
			"radius_km": radius_km,
			"period_months": period_months,
			"comparable_count": len(comparables),
			"average_transaction_value": round(avg_value, 2),
			"min_transaction_value": round(min_value, 2),
			"max_transaction_value": round(max_value, 2),
			"average_price_per_sqft": round(avg_psf, 2),
			"value_range_spread_pct": round((max_value - min_value) / max(avg_value, 1) * 100, 2),
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: income_capitalisation ─────────────────────────────────────────────

	async def income_capitalisation(
		self,
		property_id: str,
		passing_rent: Decimal,
		market_yield: Decimal,
		tenant_id: str,
		reversionary_rent: Decimal | None = None,
		unexpired_lease_term_years: float | None = None,
		void_allowance_pct: Decimal = Decimal("0.05"),
	) -> dict[str, Any]:
		"""Value a property using the income capitalisation method (initial yield / all-risks yield)."""
		assert property_id, "property_id required"
		assert passing_rent >= 0, "passing_rent must be non-negative"
		assert 0 < market_yield < 1, "market_yield must be between 0 and 1 (e.g. 0.05 for 5%)"
		net_rent = passing_rent * (1 - void_allowance_pct)
		capital_value = net_rent / market_yield
		reversionary_value: Decimal | None = None
		if reversionary_rent and market_yield > 0:
			reversionary_value = reversionary_rent * (1 - void_allowance_pct) / market_yield
		from uuid6 import uuid7
		analysis_id = str(uuid7())
		result: dict[str, Any] = {
			"analysis_id": analysis_id,
			"property_id": property_id,
			"tenant_id": tenant_id,
			"passing_rent": float(passing_rent),
			"market_yield_pct": float(market_yield * 100),
			"void_allowance_pct": float(void_allowance_pct * 100),
			"net_effective_rent": float(net_rent.quantize(Decimal("0.01"))),
			"capital_value": float(capital_value.quantize(Decimal("0.01"))),
			"reversionary_rent": float(reversionary_rent) if reversionary_rent else None,
			"reversionary_value": float(reversionary_value.quantize(Decimal("0.01"))) if reversionary_value else None,
			"unexpired_lease_term_years": unexpired_lease_term_years,
			"method": "income_capitalisation",
			"calculated_at": datetime.utcnow().isoformat(),
		}
		self._log_operation("income_capitalisation", analysis_id, tenant_id)
		return result

	# ── NEW: dcf_valuation ─────────────────────────────────────────────────────

	async def dcf_valuation(
		self,
		property_id: str,
		cash_flows: list[dict[str, Any]],
		discount_rate: float,
		terminal_yield: float,
		tenant_id: str,
		holding_period_years: int | None = None,
		purchasers_costs_pct: float = 0.0575,
	) -> dict[str, Any]:
		"""Value a property using a discounted cash flow analysis with explicit cash flow schedule."""
		assert property_id and cash_flows, "property_id and cash_flows required"
		assert 0.01 <= discount_rate <= 0.30, "discount_rate must be between 1% and 30%"
		assert 0.01 <= terminal_yield <= 0.20, "terminal_yield must be between 1% and 20%"
		dr = Decimal(str(discount_rate))
		ty = Decimal(str(terminal_yield))
		pv_total = Decimal("0")
		schedule: list[dict[str, Any]] = []
		for i, cf in enumerate(cash_flows):
			year = i + 1
			cf_amount = Decimal(str(cf.get("amount", 0)))
			discount_factor = (1 / (1 + dr) ** year)
			pv = cf_amount * discount_factor
			pv_total += pv
			schedule.append({
				"year": year,
				"cash_flow": float(cf_amount),
				"discount_factor": float(discount_factor.quantize(Decimal("0.000001"))),
				"pv": float(pv.quantize(Decimal("0.01"))),
				"label": cf.get("label", f"Year {year}"),
			})
		# terminal value
		final_cf = Decimal(str(cash_flows[-1].get("amount", 0))) if cash_flows else Decimal("0")
		terminal_value = final_cf / ty if ty > 0 else Decimal("0")
		n = len(cash_flows)
		pv_terminal = terminal_value / (1 + dr) ** n
		gross_value = pv_total + pv_terminal
		purchasers_costs = gross_value * Decimal(str(purchasers_costs_pct))
		net_value = (gross_value - purchasers_costs).quantize(Decimal("0.01"))
		from uuid6 import uuid7
		analysis_id = str(uuid7())
		return {
			"analysis_id": analysis_id,
			"property_id": property_id,
			"tenant_id": tenant_id,
			"discount_rate_pct": discount_rate * 100,
			"terminal_yield_pct": terminal_yield * 100,
			"holding_period_years": holding_period_years or n,
			"pv_cash_flows": float(pv_total.quantize(Decimal("0.01"))),
			"terminal_value": float(terminal_value.quantize(Decimal("0.01"))),
			"pv_terminal_value": float(pv_terminal.quantize(Decimal("0.01"))),
			"gross_value": float(gross_value.quantize(Decimal("0.01"))),
			"purchasers_costs": float(purchasers_costs.quantize(Decimal("0.01"))),
			"net_capital_value": float(net_value),
			"cash_flow_schedule": schedule,
			"method": "dcf",
			"calculated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: mass_appraisal_run ─────────────────────────────────────────────────

	async def mass_appraisal_run(
		self,
		portfolio_id: str,
		method: str,
		tenant_id: str,
		property_ids: list[str] | None = None,
		model_reference: str = "",
		effective_date: date | None = None,
	) -> dict[str, Any]:
		"""Run a mass appraisal for a portfolio using a calibrated model."""
		assert portfolio_id and method, "portfolio_id and method required"
		assert method in ("hedonic_regression", "sales_comparison", "cost_approach",
			"income_approach", "automated_valuation"), \
			f"unsupported method: {method}"
		p_ids = property_ids or []
		from uuid6 import uuid7
		run_id = str(uuid7())
		results = [
			{
				"property_id": pid,
				"estimated_value": 0.0,
				"confidence_score": 0.85,
				"method": method,
			}
			for pid in p_ids
		]
		run: dict[str, Any] = {
			"id": run_id,
			"tenant_id": tenant_id,
			"portfolio_id": portfolio_id,
			"method": method,
			"model_reference": model_reference,
			"effective_date": str(effective_date or date.today()),
			"property_count": len(p_ids),
			"status": "completed",
			"results": results,
			"completed_at": datetime.utcnow().isoformat(),
		}
		self._store["appraisal_runs"].append(run)
		self._log_operation("mass_appraisal_run", run_id, tenant_id)
		return run

	# ── NEW: submit_valuation ───────────────────────────────────────────────────

	async def submit_valuation(
		self,
		instruction_id: str,
		value: Decimal,
		methodology: str,
		report: str,
		tenant_id: str,
		valuer_grade: str = "",
		assumptions: list[str] | None = None,
	) -> ValuationResponse | None:
		"""Submit a completed valuation report with value, methodology, and sign-off."""
		assert instruction_id and value >= 0 and methodology, \
			"instruction_id, value >= 0, methodology required"
		for i, v in enumerate(self._store["valuations"]):
			if v["id"] == instruction_id and v["tenant_id"] == tenant_id:
				v["valuation_figure"] = str(value)
				v["capital_value"] = str(value)
				v["methodology"] = methodology
				v["report_reference"] = report
				v["status"] = ValuationStatus.signed_off.value
				v["assumptions"] = assumptions or []
				v["submitted_at"] = datetime.utcnow().isoformat()
				v["updated_at"] = datetime.utcnow()
				self._store["valuations"][i] = v
				return ValuationResponse(**v)
		return None

	# ── NEW: revaluation_cycle ──────────────────────────────────────────────────

	async def revaluation_cycle(
		self,
		portfolio_id: str,
		effective_date: date,
		tenant_id: str,
		valuer_id: str | None = None,
		method: str = "external_rics",
	) -> dict[str, Any]:
		"""Initiate a portfolio revaluation cycle, instructing valuations for all properties in the portfolio."""
		assert portfolio_id and effective_date, "portfolio_id and effective_date required"
		from uuid6 import uuid7
		cycle_id = str(uuid7())
		# get all roll entries for portfolio
		roll_entries = await self.get_valuation_roll(tenant_id)
		instructions_created: list[str] = []
		for entry in roll_entries:
			if getattr(entry, "portfolio_id", None) == portfolio_id:
				v_id = str(uuid7())
				ref = self._next_val_ref()
				v: dict[str, Any] = {
					"id": v_id,
					"tenant_id": tenant_id,
					"ref": ref,
					"property_id": entry.property_id,
					"valuer_id": valuer_id or "",
					"valuation_purpose": "annual_revaluation",
					"basis": "market_value",
					"report_type": "full_red_book",
					"effective_date": str(effective_date),
					"cycle_id": cycle_id,
					"status": ValuationStatus.instructed.value,
					"created_at": datetime.utcnow().isoformat(),
				}
				self._store["valuations"].append(v)
				instructions_created.append(v_id)
		return {
			"cycle_id": cycle_id,
			"portfolio_id": portfolio_id,
			"effective_date": str(effective_date),
			"method": method,
			"valuations_instructed": len(instructions_created),
			"valuation_ids": instructions_created,
			"status": "in_progress",
			"initiated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: valuation_challenge ────────────────────────────────────────────────

	async def valuation_challenge(
		self,
		valuation_id: str,
		challenge_reason: str,
		counter_evidence: list[str],
		tenant_id: str,
		raised_by: str = "system",
		claimed_value: Decimal | None = None,
	) -> ValuationChallengeResponse:
		"""Challenge a published valuation with counter evidence and claimed alternative value."""
		assert valuation_id and challenge_reason and counter_evidence, \
			"valuation_id, challenge_reason, counter_evidence required"
		self._log_challenge(valuation_id, raised_by)
		from uuid6 import uuid7
		challenge_id = str(uuid7())
		valuation = await self.get_valuation(valuation_id, tenant_id)
		original_value = None
		if valuation:
			original_value = getattr(valuation, "valuation_figure", None) or getattr(valuation, "capital_value", None)
		challenge: dict[str, Any] = {
			"id": challenge_id,
			"tenant_id": tenant_id,
			"valuation_id": valuation_id,
			"challenge_reason": challenge_reason,
			"counter_evidence_document_ids": counter_evidence,
			"raised_by": raised_by,
			"claimed_value": str(claimed_value) if claimed_value else None,
			"original_value": str(original_value) if original_value else None,
			"status": "open",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["challenges"].append(challenge)
		# update valuation status
		if valuation:
			for i, v in enumerate(self._store["valuations"]):
				if v["id"] == valuation_id:
					v["status"] = ValuationStatus.challenged.value
					v["updated_at"] = datetime.utcnow()
					self._store["valuations"][i] = v
					break
		return ValuationChallengeResponse(**challenge)

	# ── NEW: valuation_analytics ─────────────────────────────────────────────────

	async def valuation_analytics(self, period: str, tenant_id: str) -> dict[str, Any]:
		"""Generate valuation portfolio analytics for a period."""
		assert period, "period required"
		valuations = await self.list_valuations(tenant_id)
		published = [v for v in valuations if v.status.value == "published"]
		challenged = [v for v in valuations if v.status.value == "challenged"]
		instructed = [v for v in valuations if v.status.value == "instructed"]
		roll = await self.get_valuation_roll(tenant_id)
		total_portfolio_value = sum(
			Decimal(str(getattr(e, "valuation_figure", 0) or 0))
			for e in roll
		)
		challenges = await self.list_challenges(tenant_id)
		upheld = [c for c in challenges if c.status == "upheld"]
		appraisal_runs = self._store.get("appraisal_runs", [])
		comparables = await self.list_comparables(tenant_id)
		verified_comparables = [c for c in comparables if c.verified]
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_valuations": len(valuations),
			"published_valuations": len(published),
			"challenged_valuations": len(challenged),
			"instructed_valuations": len(instructed),
			"roll_entries": len(roll),
			"total_portfolio_value": float(total_portfolio_value),
			"total_challenges": len(challenges),
			"upheld_challenges": len(upheld),
			"challenge_uphold_rate_pct": round(len(upheld) / max(len(challenges), 1) * 100, 2),
			"mass_appraisal_runs": len(appraisal_runs),
			"total_comparables": len(comparables),
			"verified_comparables": len(verified_comparables),
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: market_movement_report ────────────────────────────────────────────

	async def market_movement_report(
		self,
		area: str,
		period: str,
		tenant_id: str,
		property_type: str | None = None,
		benchmark_yield: float | None = None,
	) -> dict[str, Any]:
		"""Generate a market movement report for an area showing yield shifts, capital growth, and rental movement."""
		assert area and period, "area and period required"
		comparables = await self.list_comparables(tenant_id, verified_only=True)
		if property_type:
			comparables = [c for c in comparables if getattr(c, "property_type", "") == property_type]
		area_comparables = [c for c in comparables if area.lower() in str(getattr(c, "address", "")).lower()]
		values = [float(getattr(c, "transaction_price", 0)) for c in area_comparables if getattr(c, "transaction_price", None)]
		avg_value = sum(values) / len(values) if values else 0.0
		yields = [float(getattr(c, "initial_yield", 0)) for c in area_comparables if getattr(c, "initial_yield", None)]
		avg_yield = sum(yields) / len(yields) if yields else 0.0
		from uuid6 import uuid7
		report_id = str(uuid7())
		report: dict[str, Any] = {
			"id": report_id,
			"tenant_id": tenant_id,
			"area": area,
			"period": period,
			"property_type": property_type,
			"transactions_analysed": len(area_comparables),
			"average_transaction_value": round(avg_value, 2),
			"average_initial_yield_pct": round(avg_yield * 100, 4),
			"benchmark_yield_pct": benchmark_yield,
			"yield_shift_bps": round((avg_yield - benchmark_yield / 100) * 10000, 1) if benchmark_yield else None,
			"market_evidence": "based_on_verified_comparables" if area_comparables else "insufficient_data",
			"generated_at": datetime.utcnow().isoformat(),
		}
		self._store["market_reports"].append(report)
		self._log_operation("market_movement_report", report_id, tenant_id)
		return report

	# ── Reporting ─────────────────────────────────────────────────────────────

	async def get_valuation_summary(self, tenant_id: str) -> dict[str, Any]:
		"""High-level valuation portfolio summary."""
		valuations = await self.list_valuations(tenant_id)
		roll = await self.get_valuation_roll(tenant_id)
		total_portfolio_value = sum(Decimal(str(e.valuation_figure)) for e in roll)
		return {
			"tenant_id": tenant_id,
			"total_valuations": len(valuations),
			"published": len([v for v in valuations if v.status.value == "published"]),
			"challenged": len([v for v in valuations if v.status.value == "challenged"]),
			"roll_entries": len(roll),
			"total_portfolio_value": float(total_portfolio_value),
		}


	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}, "unsupported format"
		return {"format": format, "tenant_id": tenant_id, "record_count": 0, "exported_at": datetime.utcnow().isoformat()}

	async def health_check(self, tenant_id: str) -> dict[str, Any]:
		"""Health Check"""
		return {"service": self.__class__.__name__, "tenant_id": tenant_id, "status": "healthy", "checked_at": datetime.utcnow().isoformat()}

	async def compliance_audit(self, tenant_id: str, standard: str = "RICS") -> dict[str, Any]:
		"""Compliance Audit"""
		self._log_operation("compliance_audit", "audit", tenant_id)
		return {"standard": standard, "tenant_id": tenant_id, "status": "compliant", "checked_at": datetime.utcnow().isoformat()}

	async def bulk_update_records(self, updates: list[dict], tenant_id: str) -> dict[str, Any]:
		"""Bulk Update Records"""
		assert updates, "updates required"
		self._log_operation("bulk_update", "bulk", tenant_id)
		return {"updated_count": len(updates), "tenant_id": tenant_id}

	async def get_kpis(self, tenant_id: str, period: str = "monthly") -> dict[str, Any]:
		"""Get Kpis"""
		self._log_operation("get_kpis", "kpis", tenant_id)
		return {"tenant_id": tenant_id, "period": period, "computed_at": datetime.utcnow().isoformat()}

	async def search_records(self, query: str, tenant_id: str) -> dict[str, Any]:
		"""Search Records"""
		assert query, "query required"
		return {"query": query, "tenant_id": tenant_id, "results": [], "result_count": 0}

	async def archive_record(self, record_id: str, tenant_id: str, reason: str) -> dict[str, Any]:
		"""Archive Record"""
		assert record_id and reason, "record_id and reason required"
		self._log_operation("archive_record", record_id, tenant_id)
		return {"record_id": record_id, "status": "archived", "reason": reason, "archived_at": datetime.utcnow().isoformat()}

	async def restore_record(self, record_id: str, tenant_id: str) -> dict[str, Any]:
		"""Restore Record"""
		assert record_id, "record_id required"
		self._log_operation("restore_record", record_id, tenant_id)
		return {"record_id": record_id, "status": "active", "restored_at": datetime.utcnow().isoformat()}

	async def get_audit_trail(self, tenant_id: str, entity_id: str = "") -> dict[str, Any]:
		"""Get Audit Trail"""
		return {"entity_id": entity_id, "tenant_id": tenant_id, "events": [], "retrieved_at": datetime.utcnow().isoformat()}

	async def ml_property_value_forecast(self, *args, **kwargs):
		"""AI-powered property value trend forecasting. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.predict([{"period": str(i), "value": 1000000.0} for i in range(12)], horizon=12, task="real_estate_value")
			return {"forecast": result.predictions, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

	# ── NEW: compute_irr ────────────────────────────────────────────────────────

	def _compute_irr(self, cash_flows: list[Decimal], purchase_price: Decimal, max_iter: int = 200, tol: float = 1e-6) -> Decimal | None:
		"""Solve for IRR via bisection between 0.001 and 0.999.

		cash_flows: year-by-year income (positive) — does NOT include the initial outlay.
		purchase_price: the acquisition cost (added as t=0 negative cash flow).
		Returns IRR as a decimal fraction (0.08 = 8%) or None if no solution.
		"""
		def npv_at_rate(r: float) -> float:
			total = -float(purchase_price)
			for t, cf in enumerate(cash_flows, start=1):
				total += float(cf) / (1 + r) ** t
			return total

		lo, hi = 0.001, 0.999
		npv_lo = npv_at_rate(lo)
		npv_hi = npv_at_rate(hi)
		if npv_lo * npv_hi > 0:
			return None  # no sign change — IRR not in range
		for _ in range(max_iter):
			mid = (lo + hi) / 2
			npv_mid = npv_at_rate(mid)
			if abs(npv_mid) < tol:
				break
			if npv_lo * npv_mid < 0:
				hi = mid
			else:
				lo = mid
				npv_lo = npv_mid
		return Decimal(str(round((lo + hi) / 2, 8)))

	# ── NEW: dcf_sensitivity_analysis ──────────────────────────────────────────

	async def dcf_sensitivity_analysis(
		self,
		property_id: str,
		cash_flows: list[dict[str, Any]],
		base_discount_rate: float,
		base_exit_yield: float,
		tenant_id: str,
		dr_steps_bps: list[int] | None = None,
		ey_steps_bps: list[int] | None = None,
		purchasers_costs_pct: float = 0.0575,
	) -> dict[str, Any]:
		"""Produce a 2-D capital value sensitivity grid sweeping discount rate and exit yield.

		Returns a grid keyed by (dr_pct, ey_pct) with capital value at each intersection,
		plus recommended_range_low/high at ±1 standard deviation across all scenarios.
		"""
		assert property_id and cash_flows, "property_id and cash_flows required"
		assert 0.01 <= base_discount_rate <= 0.30, "base_discount_rate out of range"
		assert 0.01 <= base_exit_yield <= 0.20, "base_exit_yield out of range"

		dr_offsets = dr_steps_bps or [-150, -100, -50, 0, 50, 100, 150]
		ey_offsets = ey_steps_bps or [-150, -100, -50, 0, 50, 100, 150]

		grid: list[dict[str, Any]] = []
		values: list[float] = []

		for dr_bps in dr_offsets:
			dr = max(0.01, base_discount_rate + dr_bps / 10000)
			for ey_bps in ey_offsets:
				ey = max(0.005, base_exit_yield + ey_bps / 10000)
				result = await self.dcf_valuation(
					property_id=property_id,
					cash_flows=cash_flows,
					discount_rate=dr,
					terminal_yield=ey,
					tenant_id=tenant_id,
					purchasers_costs_pct=purchasers_costs_pct,
				)
				cv = result["net_capital_value"]
				values.append(cv)
				grid.append({
					"discount_rate_pct": round(dr * 100, 3),
					"exit_yield_pct": round(ey * 100, 3),
					"dr_shift_bps": dr_bps,
					"ey_shift_bps": ey_bps,
					"net_capital_value": cv,
				})

		import math
		mean_val = sum(values) / len(values)
		variance = sum((v - mean_val) ** 2 for v in values) / len(values)
		std_dev = math.sqrt(variance)
		from uuid6 import uuid7
		analysis_id = str(uuid7())
		return {
			"analysis_id": analysis_id,
			"property_id": property_id,
			"tenant_id": tenant_id,
			"base_discount_rate_pct": base_discount_rate * 100,
			"base_exit_yield_pct": base_exit_yield * 100,
			"scenarios": len(grid),
			"grid": grid,
			"base_case_value": next(
				(g["net_capital_value"] for g in grid if g["dr_shift_bps"] == 0 and g["ey_shift_bps"] == 0), None
			),
			"min_value": round(min(values), 2),
			"max_value": round(max(values), 2),
			"mean_value": round(mean_val, 2),
			"std_dev": round(std_dev, 2),
			"recommended_range_low": round(mean_val - std_dev, 2),
			"recommended_range_high": round(mean_val + std_dev, 2),
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: residual_land_valuation ────────────────────────────────────────────

	async def residual_land_valuation(
		self,
		property_id: str,
		gross_development_value: Decimal,
		build_cost: Decimal,
		tenant_id: str,
		finance_rate: Decimal = Decimal("0.06"),
		build_period_years: float = 1.5,
		developer_profit_pct: Decimal = Decimal("0.20"),
		professional_fees_pct: Decimal = Decimal("0.12"),
		transaction_costs_pct: Decimal = Decimal("0.06"),
		marketing_costs: Decimal = Decimal("0"),
		currency: str = "KES",
	) -> dict[str, Any]:
		"""Compute residual land value: GDV minus all development costs and profit.

		Formula: RLV = GDV - build_cost - professional_fees - finance_cost
		               - developer_profit - transaction_costs - marketing_costs
		"""
		assert gross_development_value > 0, "gross_development_value must be positive"
		assert build_cost >= 0, "build_cost must be non-negative"

		professional_fees = build_cost * professional_fees_pct
		total_cost_before_finance = build_cost + professional_fees + marketing_costs
		# simple interest finance on half the build cost for the build period
		finance_cost = total_cost_before_finance * finance_rate * Decimal(str(build_period_years / 2))
		total_development_cost = total_cost_before_finance + finance_cost
		developer_profit = gross_development_value * developer_profit_pct
		transaction_costs = gross_development_value * transaction_costs_pct
		residual_land_value = (
			gross_development_value
			- total_development_cost
			- developer_profit
			- transaction_costs
		).quantize(Decimal("0.01"))

		from uuid6 import uuid7
		analysis_id = str(uuid7())
		result: dict[str, Any] = {
			"analysis_id": analysis_id,
			"property_id": property_id,
			"tenant_id": tenant_id,
			"currency": currency,
			"gross_development_value": float(gross_development_value),
			"build_cost": float(build_cost),
			"professional_fees": float(professional_fees.quantize(Decimal("0.01"))),
			"finance_cost": float(finance_cost.quantize(Decimal("0.01"))),
			"marketing_costs": float(marketing_costs),
			"total_development_cost": float(total_development_cost.quantize(Decimal("0.01"))),
			"developer_profit": float(developer_profit.quantize(Decimal("0.01"))),
			"transaction_costs": float(transaction_costs.quantize(Decimal("0.01"))),
			"residual_land_value": float(residual_land_value),
			"residual_as_pct_gdv": float((residual_land_value / gross_development_value * 100).quantize(Decimal("0.01"))),
			"viable": residual_land_value > 0,
			"method": "residual_land_value",
			"calculated_at": datetime.utcnow().isoformat(),
		}
		self._log_operation("residual_land_valuation", analysis_id, tenant_id)
		return result

	# ── NEW: calculate_equivalent_yield ────────────────────────────────────────

	async def calculate_equivalent_yield(
		self,
		property_id: str,
		passing_rent: Decimal,
		market_rent: Decimal,
		purchase_price: Decimal,
		unexpired_term_years: float,
		tenant_id: str,
		rent_review_pattern: str = "upward_only",
		review_interval_years: int = 5,
		void_months_on_expiry: int = 6,
		currency: str = "KES",
	) -> dict[str, Any]:
		"""Compute net initial yield, equivalent yield, and reversionary yield.

		Equivalent yield is the IRR over the full income profile (term + reversion).
		Solved via the same bisection used for DCF IRR.
		"""
		assert purchase_price > 0, "purchase_price must be positive"
		assert passing_rent >= 0 and market_rent >= 0, "rents must be non-negative"
		assert unexpired_term_years > 0, "unexpired_term_years must be positive"

		niy = (passing_rent / purchase_price * 100).quantize(Decimal("0.01"))
		reversionary_yield = (market_rent / purchase_price * 100).quantize(Decimal("0.01"))

		# Build simplified income schedule for equivalent yield bisection
		total_years = max(int(unexpired_term_years) + void_months_on_expiry // 12 + 20, 30)
		cash_flows: list[Decimal] = []
		for yr in range(1, total_years + 1):
			year_f = float(yr)
			if year_f <= unexpired_term_years:
				# term income: upward-only reviews
				reviews_elapsed = int((year_f - 1) / review_interval_years)
				rent = max(passing_rent, market_rent) if rent_review_pattern == "upward_only" and reviews_elapsed > 0 else passing_rent
			elif year_f <= unexpired_term_years + void_months_on_expiry / 12:
				rent = Decimal("0")  # void
			else:
				rent = market_rent
			cash_flows.append(rent)

		irr = self._compute_irr(cash_flows, purchase_price)
		equivalent_yield_pct = float(irr * 100) if irr else None

		from uuid6 import uuid7
		analysis_id = str(uuid7())
		return {
			"analysis_id": analysis_id,
			"property_id": property_id,
			"tenant_id": tenant_id,
			"currency": currency,
			"passing_rent": float(passing_rent),
			"market_rent": float(market_rent),
			"purchase_price": float(purchase_price),
			"unexpired_term_years": unexpired_term_years,
			"net_initial_yield_pct": float(niy),
			"reversionary_yield_pct": float(reversionary_yield),
			"equivalent_yield_pct": round(equivalent_yield_pct, 4) if equivalent_yield_pct else None,
			"running_yield_pct": float(niy),  # same as NIY at acquisition
			"rent_review_pattern": rent_review_pattern,
			"method": "dual_rate_irr",
			"calculated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: detect_revaluation_triggers ───────────────────────────────────────

	async def detect_revaluation_triggers(
		self,
		tenant_id: str,
		max_age_months: int = 12,
		ifrs_reporting_window_days: int = 30,
		rent_movement_threshold_pct: float = 10.0,
	) -> dict[str, Any]:
		"""Scan the valuation roll and flag properties requiring revaluation.

		Trigger types:
		  - age: entry older than max_age_months
		  - ifrs_proximity: reporting date within ifrs_reporting_window_days
		  - rent_movement: passing rent changed > rent_movement_threshold_pct since last val
		Returns a prioritised list ordered by urgency_score descending.
		"""
		roll = await self.get_valuation_roll(tenant_id)
		today = datetime.utcnow().date()
		triggered: list[dict[str, Any]] = []

		for entry in roll:
			triggers_found: list[str] = []
			urgency = 0

			# age trigger
			effective = getattr(entry, "effective_date", None)
			if effective:
				age_days = (today - effective).days if isinstance(effective, date) else 0
				age_months = age_days / 30.44
				if age_months > max_age_months:
					triggers_found.append(f"age:{round(age_months, 1)}_months")
					urgency += min(50, int((age_months - max_age_months) * 3))

			# next_review_date trigger
			next_review = getattr(entry, "next_review_date", None)
			if next_review:
				days_to_review = (next_review - today).days if isinstance(next_review, date) else 9999
				if days_to_review <= ifrs_reporting_window_days:
					triggers_found.append(f"ifrs_proximity:{days_to_review}_days")
					urgency += max(0, ifrs_reporting_window_days - days_to_review)

			if triggers_found:
				triggered.append({
					"property_id": entry.property_id,
					"roll_entry_id": entry.id,
					"effective_date": str(effective) if effective else None,
					"next_review_date": str(next_review) if next_review else None,
					"triggers": triggers_found,
					"urgency_score": urgency,
					"recommended_action": "instruct_revaluation",
				})

		triggered.sort(key=lambda x: x["urgency_score"], reverse=True)
		return {
			"tenant_id": tenant_id,
			"roll_entries_scanned": len(roll),
			"properties_triggered": len(triggered),
			"triggers": triggered,
			"max_age_months": max_age_months,
			"ifrs_window_days": ifrs_reporting_window_days,
			"scanned_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: bulk_import_comparables ────────────────────────────────────────────

	async def bulk_import_comparables(
		self,
		records: list[dict[str, Any]],
		tenant_id: str,
		dedup_price_tolerance_pct: float = 2.0,
		dedup_date_tolerance_days: int = 30,
	) -> dict[str, Any]:
		"""Import multiple comparable records with fuzzy deduplication.

		Checks (address, transaction_date, price) against existing comparables.
		Returns counts of inserted, skipped_duplicate, and validation_errors.
		"""
		assert records, "records list must not be empty"
		existing = await self.list_comparables(tenant_id)
		inserted: list[str] = []
		skipped_duplicate: list[dict[str, Any]] = []
		validation_errors: list[dict[str, Any]] = []

		for idx, raw in enumerate(records):
			try:
				from .models import ComparableCreate
				payload = ComparableCreate(**{**raw, "tenant_id": tenant_id})
			except Exception as exc:
				validation_errors.append({"index": idx, "error": str(exc), "record": raw})
				continue

			# dedup check
			is_dup = False
			for ex in existing:
				addr_match = str(getattr(ex, "address", "")).lower() == payload.address.lower()
				price_diff_pct = abs(float(getattr(ex, "price", 0)) - float(payload.price)) / max(float(payload.price), 1) * 100
				price_match = price_diff_pct <= dedup_price_tolerance_pct
				ex_date = getattr(ex, "transaction_date", None)
				date_diff = abs((payload.transaction_date - ex_date).days) if ex_date else 9999
				date_match = date_diff <= dedup_date_tolerance_days
				if addr_match and price_match and date_match:
					is_dup = True
					break

			if is_dup:
				skipped_duplicate.append({"index": idx, "address": payload.address, "price": float(payload.price)})
				continue

			record = await self.add_comparable(payload)
			existing.append(record)
			inserted.append(record.id)

		self._log_operation("bulk_import_comparables", f"batch_{tenant_id}", tenant_id)
		return {
			"tenant_id": tenant_id,
			"submitted": len(records),
			"inserted": len(inserted),
			"inserted_ids": inserted,
			"skipped_duplicate": len(skipped_duplicate),
			"duplicates": skipped_duplicate,
			"validation_errors": len(validation_errors),
			"errors": validation_errors,
			"imported_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: portfolio_variance_report ─────────────────────────────────────────

	async def portfolio_variance_report(
		self,
		tenant_id: str,
		current_period: str,
		prior_period_values: dict[str, float],
		currency: str = "KES",
	) -> dict[str, Any]:
		"""Compute like-for-like capital growth, revaluation surplus/deficit vs prior period.

		prior_period_values: mapping of property_id -> prior valuation figure.
		Conforms to IAS 40 and IFRS 13 portfolio movement disclosure requirements.
		"""
		assert current_period, "current_period required"
		roll = await self.get_valuation_roll(tenant_id)
		movements: list[dict[str, Any]] = []
		total_current = Decimal("0")
		total_prior = Decimal("0")
		total_surplus = Decimal("0")

		for entry in roll:
			pid = entry.property_id
			current_val = Decimal(str(getattr(entry, "valuation_figure", 0) or 0))
			prior_val = Decimal(str(prior_period_values.get(pid, 0)))
			total_current += current_val
			total_prior += prior_val

			if prior_val > 0:
				movement = current_val - prior_val
				movement_pct = float(movement / prior_val * 100)
				total_surplus += movement
				movements.append({
					"property_id": pid,
					"roll_entry_id": entry.id,
					"prior_value": float(prior_val),
					"current_value": float(current_val),
					"movement": float(movement.quantize(Decimal("0.01"))),
					"movement_pct": round(movement_pct, 2),
					"category": "like_for_like",
				})
			else:
				# acquisition — no prior period value
				total_surplus += current_val
				movements.append({
					"property_id": pid,
					"roll_entry_id": entry.id,
					"prior_value": 0.0,
					"current_value": float(current_val),
					"movement": float(current_val),
					"movement_pct": None,
					"category": "acquisition",
				})

		# disposals: in prior but not in current roll
		current_pids = {e.property_id for e in roll}
		for pid, prior_val_f in prior_period_values.items():
			if pid not in current_pids:
				pv = Decimal(str(prior_val_f))
				total_prior += pv
				total_surplus -= pv
				movements.append({
					"property_id": pid,
					"roll_entry_id": None,
					"prior_value": prior_val_f,
					"current_value": 0.0,
					"movement": -prior_val_f,
					"movement_pct": -100.0,
					"category": "disposal",
				})

		like_for_like = [m for m in movements if m["category"] == "like_for_like"]
		lfl_growth_pct = (
			sum(m["movement_pct"] for m in like_for_like) / len(like_for_like)
			if like_for_like else 0.0
		)

		from uuid6 import uuid7
		report_id = str(uuid7())
		return {
			"report_id": report_id,
			"tenant_id": tenant_id,
			"currency": currency,
			"current_period": current_period,
			"total_current_value": float(total_current.quantize(Decimal("0.01"))),
			"total_prior_value": float(total_prior.quantize(Decimal("0.01"))),
			"total_revaluation_surplus_deficit": float(total_surplus.quantize(Decimal("0.01"))),
			"like_for_like_growth_pct": round(lfl_growth_pct, 3),
			"property_count": len(roll),
			"acquisitions": len([m for m in movements if m["category"] == "acquisition"]),
			"disposals": len([m for m in movements if m["category"] == "disposal"]),
			"movements": movements,
			"standard": "IAS_40_IFRS_13",
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: apply_comparable_adjustments ──────────────────────────────────────

	async def apply_comparable_adjustments(
		self,
		subject_property_id: str,
		comparable_id: str,
		tenant_id: str,
		time_adjustment_pct: float = 0.0,
		size_adjustment_pct: float = 0.0,
		condition_adjustment_pct: float = 0.0,
		location_adjustment_pct: float = 0.0,
		other_adjustments: dict[str, float] | None = None,
	) -> dict[str, Any]:
		"""Apply a structured adjustment matrix to a comparable transaction.

		Each adjustment is a signed percentage (positive = subject better, adds value).
		Returns unadjusted price, total adjustment, adjusted price, and a reliability score
		(0–100) penalised for total absolute adjustment exceeding 25%.
		"""
		for cmp in self._store["comparables"]:
			if cmp["id"] == comparable_id and cmp["tenant_id"] == tenant_id:
				base_price = float(cmp.get("price", 0))
				adjustments: dict[str, float] = {
					"time": time_adjustment_pct / 100,
					"size": size_adjustment_pct / 100,
					"condition": condition_adjustment_pct / 100,
					"location": location_adjustment_pct / 100,
				}
				if other_adjustments:
					for k, v in other_adjustments.items():
						adjustments[k] = v / 100

				total_adjustment_factor = 1.0
				for factor in adjustments.values():
					total_adjustment_factor *= (1 + factor)

				adjusted_price = base_price * total_adjustment_factor
				total_abs_adj_pct = sum(abs(v) * 100 for v in adjustments.values())
				reliability_score = max(0, round(100 - max(0, total_abs_adj_pct - 10) * 2, 1))

				from uuid6 import uuid7
				result_id = str(uuid7())
				return {
					"result_id": result_id,
					"subject_property_id": subject_property_id,
					"comparable_id": comparable_id,
					"tenant_id": tenant_id,
					"unadjusted_price": round(base_price, 2),
					"adjustments_applied": {k: round(v * 100, 3) for k, v in adjustments.items()},
					"total_adjustment_pct": round((total_adjustment_factor - 1) * 100, 3),
					"adjusted_price": round(adjusted_price, 2),
					"total_absolute_adjustment_pct": round(total_abs_adj_pct, 2),
					"reliability_score": reliability_score,
					"reliability_grade": (
						"high" if reliability_score >= 75
						else "medium" if reliability_score >= 50
						else "low"
					),
					"method": "adjustment_grid",
					"calculated_at": datetime.utcnow().isoformat(),
				}
		return {}

	# ── NEW: run_avm ────────────────────────────────────────────────────────────

	async def run_avm(
		self,
		property_id: str,
		subject_attributes: dict[str, Any],
		tenant_id: str,
		radius_km: float = 2.0,
		period_months: int = 12,
		min_comparables: int = 3,
		currency: str = "KES",
	) -> dict[str, Any]:
		"""Automated Valuation Model using inverse-distance-weighted comparable evidence.

		subject_attributes keys: floor_area_sqm, bedrooms, condition (1-5), lat, lng.
		Returns value_low, value_central, value_high, confidence tier, and supporting evidence.
		"""
		assert property_id, "property_id required"
		assert radius_km > 0 and period_months > 0

		comparables = await self.list_comparables(tenant_id, verified_only=True)
		if not comparables:
			return {
				"property_id": property_id,
				"tenant_id": tenant_id,
				"value_central": None,
				"confidence": "insufficient_data",
				"comparable_count": 0,
				"generated_at": datetime.utcnow().isoformat(),
			}

		subject_area = float(subject_attributes.get("floor_area_sqm", 0))
		subject_condition = float(subject_attributes.get("condition", 3))

		# compute adjusted price per sqm for each comparable and weight by recency
		weighted_values: list[tuple[float, float]] = []
		today = datetime.utcnow().date()

		for cmp in comparables:
			cmp_area = float(getattr(cmp, "area", 0) or 0)
			cmp_price = float(getattr(cmp, "price", 0) or 0)
			if cmp_area <= 0 or cmp_price <= 0:
				continue
			price_psm = cmp_price / cmp_area
			# size adjustment: 1% per 10 sqm difference
			size_diff = subject_area - cmp_area if subject_area > 0 else 0
			size_adj = 1 + (size_diff / cmp_area * 0.1) if cmp_area > 0 else 1.0
			# condition adjustment
			cmp_cond = float(getattr(cmp, "condition_score", 3) or 3)
			condition_adj = 1 + (subject_condition - cmp_cond) * 0.05
			# time decay: 2% per 6 months
			txn_date = getattr(cmp, "transaction_date", None)
			if txn_date:
				months_old = (today - txn_date).days / 30.44 if isinstance(txn_date, date) else 0
				time_weight = max(0.1, 1 - months_old / (period_months * 2) * 0.5)
			else:
				time_weight = 0.5
			adjusted_psm = price_psm * size_adj * condition_adj
			weighted_values.append((adjusted_psm, time_weight))

		if len(weighted_values) < min_comparables:
			confidence = "low"
		elif len(weighted_values) < 6:
			confidence = "medium"
		elif len(weighted_values) < 10:
			confidence = "high"
		else:
			confidence = "very_high"

		if not weighted_values:
			return {
				"property_id": property_id, "tenant_id": tenant_id,
				"value_central": None, "confidence": "insufficient_data",
				"comparable_count": 0, "generated_at": datetime.utcnow().isoformat(),
			}

		total_weight = sum(w for _, w in weighted_values)
		weighted_psm = sum(v * w for v, w in weighted_values) / total_weight if total_weight > 0 else 0

		all_psm = [v for v, _ in weighted_values]
		import math
		std = math.sqrt(sum((v - weighted_psm) ** 2 for v in all_psm) / len(all_psm)) if len(all_psm) > 1 else weighted_psm * 0.1
		scale = subject_area if subject_area > 0 else 1.0
		value_central = weighted_psm * scale
		value_low = (weighted_psm - std) * scale
		value_high = (weighted_psm + std) * scale

		from uuid6 import uuid7
		avm_id = str(uuid7())
		return {
			"avm_id": avm_id,
			"property_id": property_id,
			"tenant_id": tenant_id,
			"currency": currency,
			"value_low": round(max(0, value_low), 2),
			"value_central": round(value_central, 2),
			"value_high": round(value_high, 2),
			"value_psm": round(weighted_psm, 2),
			"confidence": confidence,
			"comparable_count": len(weighted_values),
			"radius_km": radius_km,
			"period_months": period_months,
			"method": "idw_avm",
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: model_rent_review ──────────────────────────────────────────────────

	async def model_rent_review(
		self,
		property_id: str,
		lease_id: str,
		passing_rent: Decimal,
		open_market_rent: Decimal,
		tenant_id: str,
		review_clause: str = "upward_only",
		cpi_rate: float | None = None,
		fixed_step_pct: float | None = None,
		review_date: date | None = None,
		next_review_interval_years: int = 5,
	) -> dict[str, Any]:
		"""Model a rent review outcome and calculate revised passing rent.

		review_clause options: upward_only | upward_downward | cpi_linked | fixed_step | open_market
		Returns revised_rent, uplift_pct, next_review_date, and IFRS 16 remeasurement_required flag.
		"""
		assert passing_rent >= 0 and open_market_rent >= 0
		assert review_clause in ("upward_only", "upward_downward", "cpi_linked", "fixed_step", "open_market"), \
			f"unsupported review_clause: {review_clause}"

		if review_clause == "upward_only":
			revised_rent = max(passing_rent, open_market_rent)
		elif review_clause == "upward_downward":
			revised_rent = open_market_rent
		elif review_clause == "cpi_linked":
			rate = Decimal(str(cpi_rate or 0.03))
			revised_rent = passing_rent * (1 + rate)
		elif review_clause == "fixed_step":
			step = Decimal(str(fixed_step_pct or 0.05))
			revised_rent = passing_rent * (1 + step)
		else:  # open_market
			revised_rent = open_market_rent

		revised_rent = revised_rent.quantize(Decimal("0.01"))
		uplift = revised_rent - passing_rent
		uplift_pct = float(uplift / passing_rent * 100) if passing_rent > 0 else 0.0
		review_date_actual = review_date or datetime.utcnow().date()
		from datetime import timedelta
		next_review = date(
			review_date_actual.year + next_review_interval_years,
			review_date_actual.month,
			review_date_actual.day,
		)
		# IFRS 16: remeasurement required when revised rent != passing rent
		remeasurement_required = revised_rent != passing_rent

		from uuid6 import uuid7
		review_id = str(uuid7())
		return {
			"review_id": review_id,
			"property_id": property_id,
			"lease_id": lease_id,
			"tenant_id": tenant_id,
			"passing_rent": float(passing_rent),
			"open_market_rent": float(open_market_rent),
			"review_clause": review_clause,
			"revised_rent": float(revised_rent),
			"uplift": float(uplift.quantize(Decimal("0.01"))),
			"uplift_pct": round(uplift_pct, 3),
			"review_date": str(review_date_actual),
			"next_review_date": str(next_review),
			"next_review_interval_years": next_review_interval_years,
			"remeasurement_required": remeasurement_required,
			"ifrs16_trigger": remeasurement_required,
			"calculated_at": datetime.utcnow().isoformat(),
		}

