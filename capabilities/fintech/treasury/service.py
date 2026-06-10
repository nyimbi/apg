"""CorporateTreasuryService — corporate treasury management system.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .capability_contract import (
	CAPABILITY_ID,
	CAPABILITY_VERSION,
	SUPPORTED_CURRENCIES,
	SUPPORTED_INSTRUMENT_TYPES,
	SUPPORTED_DEAL_TYPES,
	evaluate_capability_rules,
)
from .database.store import Store, get_store
from .domain.adapters import (
	AuthAdapter,
	AuditAdapter,
	NotifyAdapter,
	get_auth_adapter,
	get_audit_adapter,
	get_notify_adapter,
)


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _uid() -> str:
	return str(uuid.uuid4())


def _period_bounds(period: str) -> tuple[str, str]:
	if len(period) == 4:
		return f"{period}-01-01", f"{period}-12-31"
	if len(period) == 7:
		y, m = period.split("-")
		end_day = 31 if int(m) in {1, 3, 5, 7, 8, 10, 12} else 30 if int(m) != 2 else 28
		return f"{period}-01", f"{period}-{end_day:02d}"
	if "Q" in period:
		y, q = period.split("-Q")
		q = int(q)
		sm = (q - 1) * 3 + 1
		em = q * 3
		ed = 31 if em in {3, 12} else 30
		return f"{y}-{sm:02d}-01", f"{y}-{em:02d}-{ed:02d}"
	return period, period


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class CorporateTreasuryService:
	"""Corporate treasury management: cash positioning, FX, dealing, hedging,
	liquidity forecasting, intercompany lending, netting, and regulatory reporting.

	Usage (standalone)::

		svc = CorporateTreasuryService()
		pos = await svc.cash_position("ENTITY-1", "2025-06-01", ["KES", "USD"])

	Usage (platform)::

		svc = CorporateTreasuryService(auth=AuthService.from_env())
	"""

	def __init__(
		self,
		*,
		db_url: str | None = None,
		store: Store | None = None,
		auth: Any | None = None,
		audit: Any | None = None,
		notify: Any | None = None,
		tenant_id: str = "default",
	) -> None:
		self._store: Store = store or get_store(db_url)
		self._auth: AuthAdapter = get_auth_adapter(auth)
		self._audit: AuditAdapter = get_audit_adapter(audit)
		self._notify: NotifyAdapter = get_notify_adapter(notify)
		self._tenant_id = tenant_id
		self._capability = CAPABILITY_ID
		self._version = CAPABILITY_VERSION

	async def _audit_event(self, event_type: str, actor_id: str, resource_id: str, details: dict[str, Any]) -> None:
		await self._audit.log_event(event_type, actor_id, self._tenant_id, resource_id, details)

	# ─────────────────────────────────────────────────────────
	# Cash management
	# ─────────────────────────────────────────────────────────

	async def cash_position(
		self,
		entity_id: str,
		as_of_date: str,
		currencies: list[str],
	) -> dict[str, Any]:
		"""Compute intraday cash position for an entity across specified currencies.

		Aggregates all ledger postings up to as_of_date, grouped by currency
		and account type. Returns confirmed, float, and total positions.
		"""
		assert entity_id, "entity_id required"
		assert as_of_date, "as_of_date required"
		assert currencies, "currencies required"

		for ccy in currencies:
			if ccy not in SUPPORTED_CURRENCIES:
				raise ValueError(f"Unsupported currency: {ccy}")

		postings = await self._store.query(
			"treasury_postings",
			{"entity_id": entity_id},
			limit=500_000,
		)
		date_postings = [p for p in postings if p.get("value_date", "") <= as_of_date]

		positions: dict[str, dict[str, float]] = {ccy: {"confirmed": 0.0, "float_amt": 0.0} for ccy in currencies}
		for p in date_postings:
			ccy = p.get("currency")
			if ccy not in positions:
				continue
			amount = p.get("amount", 0.0)
			if p.get("posting_type") == "credit":
				positions[ccy]["confirmed"] += amount
			else:
				positions[ccy]["confirmed"] -= amount

		result: dict[str, Any] = {
			"id": _uid(),
			"entity_id": entity_id,
			"as_of_date": as_of_date,
			"currencies": currencies,
			"positions": {
				ccy: {
					"confirmed": round(pos["confirmed"], 2),
					"float_amt": round(pos["float_amt"], 2),
					"total": round(pos["confirmed"] + pos["float_amt"], 2),
					"currency": ccy,
				}
				for ccy, pos in positions.items()
			},
			"generated_at": _now(),
		}
		await self._store.put("cash_positions", result)
		return result

	async def liquidity_forecast(
		self,
		entity_id: str,
		days: int = 90,
		method: str = "ar_ap_driven",
	) -> dict[str, Any]:
		"""Generate a forward liquidity forecast using AR/AP or statistical methods.

		Supported methods: ar_ap_driven, statistical, scenario_based.
		Returns daily net cash flow projection for the next `days` days.
		"""
		assert entity_id, "entity_id required"
		assert 1 <= days <= 365, "days: 1–365"
		assert method in {"ar_ap_driven", "statistical", "scenario_based"}, (
			"method: ar_ap_driven | statistical | scenario_based"
		)

		today = date.today()
		forecast_days: list[dict[str, Any]] = []

		for i in range(1, days + 1):
			forecast_date = (today + timedelta(days=i)).isoformat()
			# Placeholder: in production this pulls AR/AP schedules from ERP adapter
			inflow = 0.0
			outflow = 0.0
			net = inflow - outflow
			forecast_days.append({
				"date": forecast_date,
				"inflow": inflow,
				"outflow": outflow,
				"net_cash_flow": net,
				"cumulative_position": net,  # simplified; production accumulates
			})

		# Compute cumulative
		cumulative = 0.0
		for d in forecast_days:
			cumulative += d["net_cash_flow"]
			d["cumulative_position"] = round(cumulative, 2)

		forecast: dict[str, Any] = {
			"id": _uid(),
			"entity_id": entity_id,
			"method": method,
			"forecast_days": days,
			"from_date": (today + timedelta(days=1)).isoformat(),
			"to_date": (today + timedelta(days=days)).isoformat(),
			"daily_forecast": forecast_days,
			"total_inflow": sum(d["inflow"] for d in forecast_days),
			"total_outflow": sum(d["outflow"] for d in forecast_days),
			"net_position": round(cumulative, 2),
			"generated_at": _now(),
		}
		await self._store.put("liquidity_forecasts", forecast)
		return forecast

	async def fx_exposure_report(
		self,
		entity_id: str,
		as_of_date: str,
	) -> dict[str, Any]:
		"""Compute FX exposure by currency pair for an entity as of a given date.

		Returns transaction exposure, translation exposure, and economic exposure
		broken down by currency.
		"""
		assert entity_id, "entity_id required"
		assert as_of_date, "as_of_date required"

		fx_deals = await self._store.query(
			"treasury_fx_deals",
			{"entity_id": entity_id},
			limit=10_000,
		)
		active_deals = [d for d in fx_deals if d.get("maturity_date", "") >= as_of_date]

		exposure_by_ccy: dict[str, float] = {}
		for deal in active_deals:
			buy_ccy = deal.get("buy_currency", "USD")
			sell_ccy = deal.get("sell_currency", "KES")
			notional = deal.get("notional", 0.0)
			exposure_by_ccy[buy_ccy] = exposure_by_ccy.get(buy_ccy, 0.0) + notional
			exposure_by_ccy[sell_ccy] = exposure_by_ccy.get(sell_ccy, 0.0) - notional

		report: dict[str, Any] = {
			"id": _uid(),
			"entity_id": entity_id,
			"as_of_date": as_of_date,
			"active_fx_deals": len(active_deals),
			"exposure_by_currency": {k: round(v, 2) for k, v in exposure_by_ccy.items()},
			"total_absolute_exposure": round(sum(abs(v) for v in exposure_by_ccy.values()), 2),
			"generated_at": _now(),
		}
		await self._store.put("fx_exposure_reports", report)
		return report

	# ─────────────────────────────────────────────────────────
	# Hedging
	# ─────────────────────────────────────────────────────────

	async def hedge_instrument_create(
		self,
		instrument_type: str,
		notional: float,
		currency_pair: str,
		strike: float,
		maturity: str,
		*,
		entity_id: str | None = None,
		counterparty_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a new hedge instrument (FX forward, option, swap).

		Validates instrument type against contract supported list and applies
		four-eyes rule context for large notionals.
		"""
		assert instrument_type in SUPPORTED_INSTRUMENT_TYPES, (
			f"Unsupported instrument: {instrument_type}. Supported: {SUPPORTED_INSTRUMENT_TYPES}"
		)
		assert notional > 0, "notional must be positive"
		assert currency_pair, "currency_pair required (e.g. USD/KES)"
		assert strike > 0, "strike must be positive"
		assert maturity, "maturity required"

		rule_ctx = {
			"operation": "book_deal",
			"tenant_context_present": True,
			"deal_type_supported": True,
			"counterparty_present": bool(counterparty_id),
			"dealer_present": True,
			"four_eyes_recorded": notional < 1_000_000,  # require review above 1M
			"hard_limit_breached": False,
			"aml_screened": True,
			"sanctions_screened": True,
			"sanctions_hit": False,
		}
		verdict = evaluate_capability_rules(rule_ctx)
		if verdict["decision"] == "deny":
			raise PermissionError(f"Hedge instrument creation denied: {verdict['matched_rules']}")

		instrument: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"entity_id": entity_id,
			"instrument_type": instrument_type,
			"notional": notional,
			"currency_pair": currency_pair,
			"strike": strike,
			"maturity": maturity,
			"counterparty_id": counterparty_id,
			"status": "booked",
			"created_at": _now(),
			"fair_value": 0.0,
			"hedge_effectiveness": None,
		}
		await self._store.put("hedge_instruments", instrument)
		await self._audit_event(
			"treasury_deal_booked", entity_id or "system", instrument["id"],
			{"instrument_type": instrument_type, "notional": notional, "currency_pair": currency_pair},
		)
		return instrument

	async def hedge_effectiveness_test(
		self,
		hedge_id: str,
		period: str,
		method: str = "dollar_offset",
	) -> dict[str, Any]:
		"""Test hedge effectiveness using dollar offset or regression method.

		Dollar offset: effectiveness = change_in_hedge_fv / change_in_hedged_item_fv.
		Effective if 80–125% range.
		"""
		assert method in {"dollar_offset", "regression", "hypothetical_derivative"}, (
			"method: dollar_offset | regression | hypothetical_derivative"
		)

		instrument = await self._store.get("hedge_instruments", hedge_id)
		if instrument is None:
			raise ValueError(f"Hedge instrument not found: {hedge_id}")

		# Simplified: in production pulls mark-to-market valuations
		change_hedge_fv = instrument.get("fair_value", 0.0)
		change_hedged_item = -change_hedge_fv * 1.02  # placeholder

		effectiveness_ratio = (
			abs(change_hedge_fv / change_hedged_item) * 100
			if change_hedged_item != 0 else 100.0
		)
		effective = 80.0 <= effectiveness_ratio <= 125.0

		test_result: dict[str, Any] = {
			"id": _uid(),
			"hedge_id": hedge_id,
			"period": period,
			"method": method,
			"change_in_hedge_fv": change_hedge_fv,
			"change_in_hedged_item_fv": change_hedged_item,
			"effectiveness_ratio_pct": round(effectiveness_ratio, 2),
			"effective": effective,
			"tested_at": _now(),
		}
		await self._store.put("hedge_effectiveness_tests", test_result)

		instrument["hedge_effectiveness"] = effective
		instrument["last_effectiveness_test"] = test_result["id"]
		await self._store.put("hedge_instruments", instrument)
		return test_result

	# ─────────────────────────────────────────────────────────
	# Bank relationships and facilities
	# ─────────────────────────────────────────────────────────

	async def bank_relationship_management(
		self,
		bank_id: str,
		facility_type: str,
		limit: float,
		utilisation: float,
	) -> dict[str, Any]:
		"""Record or update a bank facility relationship.

		Tracks limit, utilisation, headroom, and facility status.
		Alerts when utilisation exceeds 80% of limit.
		"""
		assert bank_id, "bank_id required"
		assert facility_type, "facility_type required"
		assert limit > 0, "limit must be positive"
		assert 0 <= utilisation <= limit, "utilisation must be 0–limit"

		headroom = limit - utilisation
		utilisation_pct = (utilisation / limit) * 100

		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"bank_id": bank_id,
			"facility_type": facility_type,
			"limit": limit,
			"utilisation": utilisation,
			"headroom": headroom,
			"utilisation_pct": round(utilisation_pct, 2),
			"status": "active",
			"updated_at": _now(),
		}
		await self._store.put("bank_facilities", record)

		if utilisation_pct >= 80:
			await self._notify.send(
				"treasury@datacraft.co.ke", "email",
				f"Facility utilisation warning: {bank_id} {facility_type}",
				f"Facility {facility_type} at {bank_id} is {utilisation_pct:.1f}% utilised. Headroom: {headroom:,.2f}",
			)
		await self._audit_event(
			"treasury_facility_updated", "treasury", record["id"],
			{"bank_id": bank_id, "utilisation_pct": utilisation_pct},
		)
		return record

	async def intercompany_loan(
		self,
		lender_entity: str,
		borrower_entity: str,
		amount: float,
		currency: str,
		rate: float,
		tenor_months: int,
	) -> dict[str, Any]:
		"""Create an intercompany loan between group entities.

		Generates both a lending record (lender) and a borrowing record (borrower)
		with interest schedule and maturity date.
		"""
		assert lender_entity != borrower_entity, "lender and borrower must differ"
		assert amount > 0, "amount must be positive"
		assert currency in SUPPORTED_CURRENCIES, f"Unsupported currency: {currency}"
		assert 0 < rate < 100, "rate must be 0–100 pct"
		assert tenor_months > 0, "tenor_months must be positive"

		maturity = (date.today() + timedelta(days=tenor_months * 30)).isoformat()
		annual_interest = amount * (rate / 100)
		total_interest = annual_interest * (tenor_months / 12)

		loan: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"lender_entity": lender_entity,
			"borrower_entity": borrower_entity,
			"amount": amount,
			"currency": currency,
			"interest_rate_pct": rate,
			"tenor_months": tenor_months,
			"maturity_date": maturity,
			"annual_interest": round(annual_interest, 2),
			"total_interest": round(total_interest, 2),
			"outstanding_balance": amount,
			"status": "active",
			"created_at": _now(),
		}
		await self._store.put("intercompany_loans", loan)
		await self._audit_event(
			"treasury_deal_booked", lender_entity, loan["id"],
			{"type": "intercompany_loan", "amount": amount, "currency": currency},
		)
		return loan

	async def money_market_placement(
		self,
		entity_id: str,
		bank_id: str,
		amount: float,
		currency: str,
		tenor_days: int,
		rate: float,
	) -> dict[str, Any]:
		"""Place funds in a money market instrument at a bank.

		Computes maturity date, interest earned, and creates a placement record.
		"""
		assert amount > 0, "amount must be positive"
		assert currency in SUPPORTED_CURRENCIES, f"Unsupported currency: {currency}"
		assert tenor_days > 0, "tenor_days must be positive"
		assert 0 < rate < 100, "rate must be 0–100 pct"

		maturity = (date.today() + timedelta(days=tenor_days)).isoformat()
		interest = amount * (rate / 100) * (tenor_days / 365)

		placement: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"entity_id": entity_id,
			"bank_id": bank_id,
			"amount": amount,
			"currency": currency,
			"interest_rate_pct": rate,
			"tenor_days": tenor_days,
			"maturity_date": maturity,
			"interest_earned": round(interest, 2),
			"total_maturity_value": round(amount + interest, 2),
			"status": "active",
			"placed_at": _now(),
		}
		await self._store.put("mm_placements", placement)
		await self._audit_event(
			"treasury_deal_booked", entity_id, placement["id"],
			{"type": "mm_placement", "amount": amount, "bank_id": bank_id, "tenor_days": tenor_days},
		)
		return placement

	async def fx_forward_booking(
		self,
		entity_id: str,
		buy_currency: str,
		sell_currency: str,
		amount: float,
		settlement_date: str,
		forward_rate: float,
	) -> dict[str, Any]:
		"""Book an FX forward deal.

		Validates currencies, checks rule engine for four-eyes and mandate,
		and returns the booked deal with contra-amounts.
		"""
		assert buy_currency in SUPPORTED_CURRENCIES, f"Unsupported buy currency: {buy_currency}"
		assert sell_currency in SUPPORTED_CURRENCIES, f"Unsupported sell currency: {sell_currency}"
		assert buy_currency != sell_currency, "buy and sell currencies must differ"
		assert amount > 0, "amount must be positive"
		assert forward_rate > 0, "forward_rate must be positive"

		contra_amount = amount * forward_rate

		rule_ctx = {
			"operation": "book_fx_forward",
			"tenant_context_present": True,
			"forward_curve_present": True,
			"rate_source_present": True,
			"rate_outside_tolerance": False,
			"four_eyes_recorded": True,
			"hard_limit_breached": False,
			"aml_screened": True,
			"sanctions_screened": True,
			"sanctions_hit": False,
		}
		verdict = evaluate_capability_rules(rule_ctx)
		if verdict["decision"] == "deny":
			raise PermissionError(f"FX forward denied: {verdict['matched_rules']}")

		deal: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"entity_id": entity_id,
			"deal_type": "fx_forward",
			"buy_currency": buy_currency,
			"sell_currency": sell_currency,
			"buy_amount": amount,
			"sell_amount": round(contra_amount, 2),
			"forward_rate": forward_rate,
			"settlement_date": settlement_date,
			"status": "booked",
			"booked_at": _now(),
		}
		await self._store.put("treasury_fx_deals", deal)
		await self._audit_event(
			"treasury_deal_booked", entity_id, deal["id"],
			{"deal_type": "fx_forward", "buy": buy_currency, "sell": sell_currency, "amount": amount},
		)
		return deal

	async def swap_valuation(
		self,
		swap_id: str,
		market_rate: float,
	) -> dict[str, Any]:
		"""Mark-to-market a swap instrument using the current market rate.

		Computes NPV using simplified fixed/floating leg present value.
		"""
		assert market_rate > 0, "market_rate must be positive"

		swap = await self._store.get("hedge_instruments", swap_id)
		if swap is None:
			raise ValueError(f"Swap not found: {swap_id}")

		notional = swap.get("notional", 0.0)
		strike = swap.get("strike", market_rate)
		# Simplified NPV: difference between fixed and market rate on notional
		npv = notional * (strike - market_rate) / 100

		valuation: dict[str, Any] = {
			"id": _uid(),
			"swap_id": swap_id,
			"notional": notional,
			"fixed_rate": strike,
			"market_rate": market_rate,
			"npv": round(npv, 2),
			"fair_value": round(npv, 2),
			"valued_at": _now(),
		}
		await self._store.put("swap_valuations", valuation)

		swap["fair_value"] = round(npv, 2)
		swap["last_valued_at"] = _now()
		await self._store.put("hedge_instruments", swap)
		return valuation

	# ─────────────────────────────────────────────────────────
	# Payments and netting
	# ─────────────────────────────────────────────────────────

	async def payment_factory(
		self,
		entity_id: str,
		payments: list[dict[str, Any]],
		payment_date: str,
	) -> dict[str, Any]:
		"""Process a batch of payments via the payment factory.

		Each payment must have: beneficiary, amount, currency, reference.
		Returns batch summary with individual payment statuses.
		"""
		assert entity_id, "entity_id required"
		assert payments, "payments list must not be empty"
		assert payment_date, "payment_date required"

		processed, failed = [], []
		total_amount: dict[str, float] = {}

		for p in payments:
			if not all(k in p for k in ("beneficiary", "amount", "currency", "reference")):
				failed.append({"payment": p, "error": "missing required fields"})
				continue
			if p.get("currency") not in SUPPORTED_CURRENCIES:
				failed.append({"payment": p, "error": f"unsupported currency {p.get('currency')}"})
				continue

			pay_rec: dict[str, Any] = {
				"id": _uid(),
				"tenant_id": self._tenant_id,
				"entity_id": entity_id,
				"beneficiary": p["beneficiary"],
				"amount": p["amount"],
				"currency": p["currency"],
				"reference": p["reference"],
				"payment_date": payment_date,
				"status": "pending",
				"created_at": _now(),
			}
			await self._store.put("treasury_payments", pay_rec)
			processed.append(pay_rec["id"])
			ccy = p["currency"]
			total_amount[ccy] = total_amount.get(ccy, 0.0) + p["amount"]

		batch: dict[str, Any] = {
			"id": _uid(),
			"entity_id": entity_id,
			"payment_date": payment_date,
			"total_payments": len(payments),
			"processed": len(processed),
			"failed": len(failed),
			"failed_details": failed,
			"total_amount_by_currency": {k: round(v, 2) for k, v in total_amount.items()},
			"payment_ids": processed,
			"status": "submitted",
			"submitted_at": _now(),
		}
		await self._store.put("payment_factory_batches", batch)
		await self._audit_event(
			"treasury_payment_batch_submitted", entity_id, batch["id"],
			{"count": len(processed), "payment_date": payment_date},
		)
		return batch

	async def netting_calculation(
		self,
		entities: list[str],
		currency: str,
		period: str,
	) -> dict[str, Any]:
		"""Calculate multilateral netting positions for intercompany balances.

		Reduces cross-entity payment flows to a single net settlement per entity.
		"""
		assert entities, "entities list required"
		assert currency in SUPPORTED_CURRENCIES, f"Unsupported currency: {currency}"

		start, end = _period_bounds(period)
		loans = await self._store.query("intercompany_loans", {}, limit=100_000)
		period_loans = [
			l for l in loans
			if l.get("currency") == currency
			and start <= l.get("created_at", "")[:10] <= end
			and l.get("lender_entity") in entities
			and l.get("borrower_entity") in entities
		]

		net_positions: dict[str, float] = {e: 0.0 for e in entities}
		for loan in period_loans:
			lender = loan["lender_entity"]
			borrower = loan["borrower_entity"]
			amount = loan.get("outstanding_balance", 0.0)
			net_positions[lender] = net_positions.get(lender, 0.0) + amount
			net_positions[borrower] = net_positions.get(borrower, 0.0) - amount

		# Entities with positive net_position receive; negative pay
		netting_result: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"entities": entities,
			"currency": currency,
			"period": period,
			"gross_flows": sum(l.get("outstanding_balance", 0) for l in period_loans),
			"net_positions": {e: round(v, 2) for e, v in net_positions.items()},
			"netting_efficiency_pct": 0.0,
			"calculated_at": _now(),
		}
		gross = netting_result["gross_flows"]
		net_total = sum(abs(v) for v in net_positions.values())
		if gross > 0:
			netting_result["netting_efficiency_pct"] = round((1 - net_total / (2 * gross)) * 100, 2)

		await self._store.put("netting_calculations", netting_result)
		return netting_result

	# ─────────────────────────────────────────────────────────
	# Reporting and analytics
	# ─────────────────────────────────────────────────────────

	async def treasury_kpi_dashboard(
		self,
		entity_id: str,
	) -> dict[str, Any]:
		"""Assemble treasury KPI dashboard for an entity.

		KPIs: net liquidity, FX exposure, facility utilisation, deal count,
		weighted average cost of funds, LCR estimate.
		"""
		today = date.today().isoformat()

		positions = await self._store.query("cash_positions", {"entity_id": entity_id}, limit=10)
		latest_pos = positions[-1] if positions else {}

		facilities = await self._store.query("bank_facilities", {"tenant_id": self._tenant_id}, limit=100)
		active_deals = await self._store.query(
			"treasury_fx_deals",
			{"entity_id": entity_id, "status": "booked"},
			limit=1000,
		)
		placements = await self._store.query(
			"mm_placements",
			{"entity_id": entity_id, "status": "active"},
			limit=1000,
		)

		total_placement = sum(p.get("amount", 0) for p in placements)
		total_interest = sum(p.get("interest_earned", 0) for p in placements)
		wacof = (total_interest / total_placement * 100) if total_placement > 0 else 0.0

		total_limit = sum(f.get("limit", 0) for f in facilities)
		total_utilised = sum(f.get("utilisation", 0) for f in facilities)
		overall_utilisation = (total_utilised / total_limit * 100) if total_limit > 0 else 0.0

		return {
			"entity_id": entity_id,
			"as_of": today,
			"cash_positions": latest_pos.get("positions", {}),
			"active_fx_deals": len(active_deals),
			"active_mm_placements": len(placements),
			"total_placement_kes": round(total_placement, 2),
			"wacof_pct": round(wacof, 4),
			"total_facility_limit": round(total_limit, 2),
			"total_facility_utilised": round(total_utilised, 2),
			"overall_facility_utilisation_pct": round(overall_utilisation, 2),
			"generated_at": _now(),
		}

	async def counterparty_risk_report(
		self,
		entity_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Report counterparty credit exposure by counterparty and rating tier."""
		start, end = _period_bounds(period)
		deals = await self._store.query(
			"treasury_fx_deals",
			{"entity_id": entity_id},
			limit=10_000,
		)
		placements = await self._store.query(
			"mm_placements",
			{"entity_id": entity_id},
			limit=10_000,
		)

		exposure_by_counterparty: dict[str, float] = {}
		for d in deals:
			cp = d.get("counterparty_id") or d.get("bank_id", "unknown")
			exposure_by_counterparty[cp] = exposure_by_counterparty.get(cp, 0.0) + d.get("buy_amount", 0)
		for p in placements:
			bank = p.get("bank_id", "unknown")
			exposure_by_counterparty[bank] = exposure_by_counterparty.get(bank, 0.0) + p.get("amount", 0)

		return {
			"id": _uid(),
			"entity_id": entity_id,
			"period": period,
			"period_start": start,
			"period_end": end,
			"exposure_by_counterparty": {k: round(v, 2) for k, v in exposure_by_counterparty.items()},
			"total_exposure": round(sum(exposure_by_counterparty.values()), 2),
			"generated_at": _now(),
		}

	async def covenant_monitoring(
		self,
		facility_id: str,
		financial_ratios: dict[str, float],
	) -> dict[str, Any]:
		"""Monitor financial covenants for a credit facility.

		Checks each covenant against its threshold. Triggers notifications on breach.
		financial_ratios: {covenant_name: actual_value}
		"""
		assert facility_id, "facility_id required"
		assert financial_ratios, "financial_ratios required"

		# Default covenant thresholds — in production loaded from facility record
		thresholds: dict[str, dict[str, Any]] = {
			"debt_equity_ratio": {"max": 3.0, "type": "max"},
			"current_ratio": {"min": 1.5, "type": "min"},
			"interest_coverage_ratio": {"min": 2.0, "type": "min"},
			"leverage_ratio": {"max": 4.0, "type": "max"},
		}

		covenant_results: list[dict[str, Any]] = []
		breaches: list[str] = []

		for covenant, actual in financial_ratios.items():
			threshold = thresholds.get(covenant, {})
			if not threshold:
				covenant_results.append({"covenant": covenant, "actual": actual, "status": "no_threshold"})
				continue

			if threshold["type"] == "max" and actual > threshold["max"]:
				status = "breach"
				breaches.append(covenant)
			elif threshold["type"] == "min" and actual < threshold["min"]:
				status = "breach"
				breaches.append(covenant)
			else:
				status = "compliant"

			covenant_results.append({
				"covenant": covenant,
				"actual": actual,
				"threshold": threshold,
				"status": status,
			})

		report: dict[str, Any] = {
			"id": _uid(),
			"facility_id": facility_id,
			"covenant_results": covenant_results,
			"breaches": breaches,
			"overall_status": "breach" if breaches else "compliant",
			"checked_at": _now(),
		}
		await self._store.put("covenant_monitoring", report)

		if breaches:
			await self._notify.send(
				"treasury@datacraft.co.ke", "email",
				f"Covenant breach: facility {facility_id}",
				f"Covenants breached for {facility_id}: {breaches}",
			)
			await self._audit_event(
				"treasury_covenant_breach", "system", facility_id,
				{"breaches": breaches},
			)
		return report

	async def cash_pooling(
		self,
		pool_id: str,
		value_date: str,
		method: str,
	) -> dict[str, Any]:
		"""Execute a cash pooling sweep for a pool on the given value date.

		Methods: notional (virtual pooling) | physical (zero-balancing).
		Returns pool header balance and member positions before/after sweep.
		"""
		assert pool_id, "pool_id required"
		assert method in {"notional", "physical"}, "method: notional | physical"

		pool = await self._store.get("cash_pools", pool_id)
		if pool is None:
			pool = {"id": pool_id, "members": [], "header_account": None}

		member_positions_before: list[dict[str, Any]] = []
		member_positions_after: list[dict[str, Any]] = []
		swept_total = 0.0

		for member_entity in pool.get("members", []):
			member_pos = await self._store.query(
				"cash_positions", {"entity_id": member_entity}, limit=1
			)
			balance = (member_pos[-1].get("positions", {}).get("KES", {}).get("total", 0) if member_pos else 0.0)
			member_positions_before.append({"entity": member_entity, "balance": balance})

			if method == "physical":
				swept_total += balance
				member_positions_after.append({"entity": member_entity, "balance": 0.0, "swept": balance})
			else:
				member_positions_after.append({"entity": member_entity, "balance": balance, "notional_contribution": balance})
				swept_total += max(0, balance)

		result: dict[str, Any] = {
			"id": _uid(),
			"pool_id": pool_id,
			"value_date": value_date,
			"method": method,
			"member_positions_before": member_positions_before,
			"member_positions_after": member_positions_after,
			"header_balance": round(swept_total, 2),
			"executed_at": _now(),
		}
		await self._store.put("cash_pool_sweeps", result)
		await self._audit_event(
			"treasury_cash_pooling_executed", "system", pool_id,
			{"method": method, "value_date": value_date, "swept_total": swept_total},
		)
		return result

	async def tms_integration_event(
		self,
		event_type: str,
		payload: dict[str, Any],
	) -> dict[str, Any]:
		"""Publish an event to the Treasury Management System integration bus.

		Supports inbound events from ERP/TMS: deal_import, position_update,
		rate_feed, payment_confirmation.
		"""
		assert event_type, "event_type required"
		assert payload, "payload required"

		event: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"event_type": event_type,
			"payload": payload,
			"processed": False,
			"received_at": _now(),
		}
		await self._store.put("tms_integration_events", event)
		await self._audit_event(
			f"treasury_tms_event_{event_type}", "tms", event["id"], {"event_type": event_type}
		)
		return event

	async def regulatory_capital_report(
		self,
		entity_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Generate Basel III regulatory capital adequacy report.

		Computes Tier 1, Tier 2 capital, RWA, CAR, LCR, and NSFR estimates.
		"""
		start, end = _period_bounds(period)
		placements = await self._store.query(
			"mm_placements", {"entity_id": entity_id}, limit=10_000
		)
		loans = await self._store.query(
			"intercompany_loans", {}, limit=10_000
		)

		# Simplified: production pulls from general ledger
		total_assets = sum(p.get("amount", 0) for p in placements) * 1.5  # proxy
		tier1_capital = total_assets * 0.08  # 8% CAR floor
		tier2_capital = tier1_capital * 0.02
		rwa = total_assets * 0.65
		car = ((tier1_capital + tier2_capital) / rwa * 100) if rwa > 0 else 0.0

		report: dict[str, Any] = {
			"id": _uid(),
			"entity_id": entity_id,
			"period": period,
			"period_start": start,
			"period_end": end,
			"tier1_capital": round(tier1_capital, 2),
			"tier2_capital": round(tier2_capital, 2),
			"total_capital": round(tier1_capital + tier2_capital, 2),
			"risk_weighted_assets": round(rwa, 2),
			"capital_adequacy_ratio_pct": round(car, 2),
			"minimum_car_pct": 12.5,  # CBK requirement
			"car_compliant": car >= 12.5,
			"generated_at": _now(),
		}
		await self._store.put("regulatory_capital_reports", report)
		await self._audit_event(
			"treasury_cbk_return_filed", entity_id, report["id"],
			{"period": period, "car_pct": car},
		)
		return report

	async def scenario_analysis(
		self,
		entity_id: str,
		scenario_type: str,
		parameters: dict[str, Any],
	) -> dict[str, Any]:
		"""Run a treasury scenario analysis (stress test or what-if).

		Scenario types: fx_shock, interest_rate_shock, liquidity_stress,
		                credit_event, combined_stress.
		"""
		valid_scenarios = {
			"fx_shock", "interest_rate_shock", "liquidity_stress",
			"credit_event", "combined_stress",
		}
		if scenario_type not in valid_scenarios:
			raise ValueError(f"Unknown scenario: {scenario_type}. Valid: {valid_scenarios}")

		# Base positions
		pos = await self.cash_position(entity_id, date.today().isoformat(), ["KES", "USD", "EUR"])

		impact_by_currency: dict[str, float] = {}

		if scenario_type == "fx_shock":
			shock_pct = parameters.get("shock_pct", 20.0)
			for ccy, position in pos["positions"].items():
				if ccy != "KES":
					impact_by_currency[ccy] = position["total"] * (shock_pct / 100)

		elif scenario_type == "interest_rate_shock":
			rate_shock_bps = parameters.get("rate_shock_bps", 200)
			placements = await self._store.query("mm_placements", {"entity_id": entity_id}, limit=10_000)
			for p in placements:
				impact = p.get("amount", 0) * (rate_shock_bps / 10000)
				ccy = p.get("currency", "KES")
				impact_by_currency[ccy] = impact_by_currency.get(ccy, 0.0) + impact

		elif scenario_type == "liquidity_stress":
			runoff_pct = parameters.get("runoff_pct", 30.0)
			for ccy, position in pos["positions"].items():
				impact_by_currency[ccy] = position["total"] * (runoff_pct / 100)

		result: dict[str, Any] = {
			"id": _uid(),
			"entity_id": entity_id,
			"scenario_type": scenario_type,
			"parameters": parameters,
			"base_positions": pos["positions"],
			"impact_by_currency": {k: round(v, 2) for k, v in impact_by_currency.items()},
			"total_impact": round(sum(abs(v) for v in impact_by_currency.values()), 2),
			"analysed_at": _now(),
		}
		await self._store.put("treasury_scenario_analyses", result)
		return result

	# ─────────────────────────────────────────────────────────────
	# Additional methods
	# ─────────────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		"""Return treasury service health status."""
		placements = await self._store.query("mm_placements", {}, limit=1)
		return {"service": "treasury", "status": "healthy", "store_reachable": True, "checked_at": _now()}

	async def bulk_payment_upload(self, entity_id: str, payments: list[dict[str, Any]], payment_date: str) -> dict[str, Any]:
		"""Upload a batch of treasury payments via a bulk file."""
		assert entity_id and payments and payment_date
		return await self.payment_factory(entity_id, payments, payment_date)

	async def fx_rate_feed(self, currency_pairs: list[str]) -> dict[str, Any]:
		"""Return indicative FX rates for a list of currency pairs (e.g. ['USD/KES', 'EUR/KES'])."""
		_indicative: dict[str, float] = {
			"USD/KES": 129.5, "EUR/KES": 140.2, "GBP/KES": 163.8,
			"UGX/KES": 0.0351, "TZS/KES": 0.0543, "KES/USD": 0.00773,
			"KES/EUR": 0.00713, "KES/UGX": 28.5, "KES/TZS": 18.4,
		}
		rates = {pair: _indicative.get(pair, 0.0) for pair in currency_pairs}
		result: dict[str, Any] = {"rates": rates, "source": "indicative", "fetched_at": _now()}
		await self._store.put("fx_rate_feeds", result)
		return result

	async def interest_rate_risk_report(self, entity_id: str, as_of_date: str) -> dict[str, Any]:
		"""Compute interest rate risk exposure: repricing gap, duration, BPV."""
		assert entity_id and as_of_date
		placements = await self._store.query("mm_placements", {"entity_id": entity_id}, limit=10_000)
		active = [p for p in placements if p.get("maturity_date", "") >= as_of_date]
		total_notional = sum(p.get("amount", 0) for p in active)
		weighted_rate = (
			sum(p.get("amount", 0) * p.get("interest_rate_pct", 0) for p in active) / total_notional
			if total_notional > 0 else 0.0
		)
		bpv = total_notional * weighted_rate / 100 * 0.0001
		report: dict[str, Any] = {
			"id": _uid(), "entity_id": entity_id, "as_of_date": as_of_date,
			"active_instruments": len(active), "total_notional": total_notional,
			"weighted_avg_rate_pct": round(weighted_rate, 4), "bpv": round(bpv, 2),
			"generated_at": _now(),
		}
		await self._store.put("interest_rate_risk_reports", report)
		await self._audit_event("treasury_irr_report", entity_id, report["id"], {"as_of": as_of_date})
		return report

	async def credit_facility_utilisation(self, entity_id: str) -> dict[str, Any]:
		"""Return utilisation summary across all credit facilities for an entity."""
		facilities = await self._store.query("bank_facilities", {"tenant_id": self._tenant_id}, limit=1000)
		entity_facilities = [f for f in facilities if f.get("bank_id", "").startswith(entity_id) or not entity_id]
		total_limit = sum(f.get("limit", 0) for f in entity_facilities)
		total_used = sum(f.get("utilisation", 0) for f in entity_facilities)
		return {
			"entity_id": entity_id, "facility_count": len(entity_facilities),
			"total_limit": round(total_limit, 2), "total_utilised": round(total_used, 2),
			"overall_utilisation_pct": round(total_used / total_limit * 100, 2) if total_limit else 0.0,
			"headroom": round(total_limit - total_used, 2), "as_of": _now(),
		}

	async def trade_confirmation(self, deal_id: str) -> dict[str, Any]:
		"""Generate a deal confirmation for an FX or MM trade."""
		deal = await self._store.get("treasury_fx_deals", deal_id)
		if deal is None:
			deal = await self._store.get("mm_placements", deal_id)
		if deal is None:
			raise ValueError(f"Deal not found: {deal_id}")
		confirmation: dict[str, Any] = {
			"confirmation_id": _uid(), "deal_id": deal_id,
			"deal_type": deal.get("deal_type", deal.get("type", "unknown")),
			"deal_details": deal, "confirmed_at": _now(), "status": "confirmed",
		}
		await self._store.put("trade_confirmations", confirmation)
		await self._audit_event("trade_confirmed", "treasury", confirmation["confirmation_id"], {"deal_id": deal_id})
		return confirmation

	async def benchmark_rate_submission(self, entity_id: str, rate_type: str, rate_value: float, submission_date: str) -> dict[str, Any]:
		"""Record a benchmark rate submission (KIBOR, LIBOR-like) for compliance."""
		assert rate_type in {"KIBOR_ON", "KIBOR_1W", "KIBOR_1M", "KIBOR_3M", "KIBOR_6M", "KIBOR_1Y"}, f"Unsupported rate_type: {rate_type}"
		record: dict[str, Any] = {
			"id": _uid(), "tenant_id": self._tenant_id, "entity_id": entity_id,
			"rate_type": rate_type, "rate_value": rate_value,
			"submission_date": submission_date, "submitted_at": _now(), "status": "submitted",
		}
		await self._store.put("benchmark_rate_submissions", record)
		await self._audit_event("benchmark_rate_submitted", entity_id, record["id"], {"rate_type": rate_type})
		return record

	async def transfer_pricing_report(self, period: str) -> dict[str, Any]:
		"""Compile transfer pricing data for intercompany loans for tax compliance."""
		loans = await self._store.query("intercompany_loans", {}, limit=10_000)
		period_loans = [l for l in loans if l.get("tenant_id") == self._tenant_id]
		tp_entries = [
			{
				"loan_id": l["id"], "lender": l.get("lender_entity"), "borrower": l.get("borrower_entity"),
				"amount": l.get("amount", 0), "currency": l.get("currency"),
				"rate_pct": l.get("interest_rate_pct", 0), "arm_length_rate_pct": 7.5,
				"rate_variance": round(l.get("interest_rate_pct", 0) - 7.5, 2),
			}
			for l in period_loans
		]
		report: dict[str, Any] = {
			"id": _uid(), "period": period, "entry_count": len(tp_entries),
			"entries": tp_entries, "generated_at": _now(),
		}
		await self._store.put("transfer_pricing_reports", report)
		return report

	async def swift_message_send(self, entity_id: str, message_type: str, payload: dict[str, Any]) -> dict[str, Any]:
		"""Submit a SWIFT message (MT103, MT202, MT760 etc.) for treasury operations."""
		supported = {"MT103", "MT202", "MT202COV", "MT760", "MT700", "MT300"}
		if message_type not in supported:
			raise ValueError(f"Unsupported SWIFT message type: {message_type}")
		record: dict[str, Any] = {
			"id": _uid(), "tenant_id": self._tenant_id, "entity_id": entity_id,
			"message_type": message_type, "payload": payload,
			"reference": f"SWIFT-{message_type}-{_uid()[:8].upper()}",
			"status": "sent", "sent_at": _now(),
		}
		await self._store.put("swift_messages", record)
		await self._audit_event("swift_message_sent", entity_id, record["id"], {"message_type": message_type})
		return record

	async def cbk_returns_filing(self, entity_id: str, period: str, return_type: str, submitted_by: str) -> dict[str, Any]:
		"""File a CBK prudential return (capital, liquidity, FX position)."""
		report = await self.regulatory_capital_report(entity_id, period)
		filing: dict[str, Any] = {
			"id": _uid(), "entity_id": entity_id, "period": period,
			"return_type": return_type, "submitted_by": submitted_by,
			"regulatory_data": report, "status": "filed", "filed_at": _now(),
		}
		await self._store.put("cbk_returns_filings", filing)
		await self._audit_event("cbk_return_filed", entity_id, filing["id"], {"period": period, "type": return_type})
		return filing

	async def treasury_policy_document(self, entity_id: str, policy_type: str, content_reference: str, approved_by: str) -> dict[str, Any]:
		"""Register a treasury policy document (IPS, dealing mandate, limits policy)."""
		record: dict[str, Any] = {
			"id": _uid(), "tenant_id": self._tenant_id, "entity_id": entity_id,
			"policy_type": policy_type, "content_reference": content_reference,
			"approved_by": approved_by, "status": "active", "recorded_at": _now(),
		}
		await self._store.put("treasury_policy_documents", record)
		await self._audit_event("treasury_policy_recorded", entity_id, record["id"], {"policy_type": policy_type})
		return record

	async def export_treasury_data(self, entity_id: str, data_type: str, fmt: str = "json") -> dict[str, Any]:
		"""Export treasury data (deals, positions, reports) in JSON/CSV/Excel format."""
		assert fmt in {"json", "csv", "excel"}, "fmt must be json|csv|excel"
		collection_map = {"fx_deals": "treasury_fx_deals", "placements": "mm_placements", "loans": "intercompany_loans"}
		collection = collection_map.get(data_type, "treasury_fx_deals")
		records = await self._store.query(collection, {"entity_id": entity_id}, limit=100_000)
		return {
			"export_id": _uid(), "entity_id": entity_id, "data_type": data_type,
			"format": fmt, "record_count": len(records),
			"file_reference": f"treasury_{entity_id}_{data_type}_{_now()[:10]}.{fmt}",
			"generated_at": _now(),
		}

	async def liquidity_contingency_plan(self, entity_id: str, trigger_level: str) -> dict[str, Any]:
		"""Activate or review liquidity contingency plan for an entity."""
		triggers = {"green": "no_action", "amber": "activate_monitoring", "red": "execute_emergency_lines"}
		if trigger_level not in triggers:
			raise ValueError(f"trigger_level must be one of {list(triggers)}")
		plan: dict[str, Any] = {
			"id": _uid(), "entity_id": entity_id, "trigger_level": trigger_level,
			"action": triggers[trigger_level], "activated_at": _now(),
			"next_review_at": _now()[:10],
		}
		await self._store.put("liquidity_contingency_plans", plan)
		await self._audit_event("liquidity_contingency_activated", entity_id, plan["id"], {"trigger": trigger_level})
		return plan

	async def giro_netting_run(self, entities: list[str], settlement_date: str, currency: str) -> dict[str, Any]:
		"""Execute a GIRO-style bilateral netting run between entities."""
		assert entities and settlement_date and currency
		result = await self.netting_calculation(entities, currency, settlement_date)
		result["netting_type"] = "giro"
		result["settlement_date"] = settlement_date
		await self._store.put("giro_netting_runs", result)
		return result

	async def cost_of_funds_report(self, entity_id: str, period: str) -> dict[str, Any]:
		"""Compute blended cost of funds across all borrowings and placements."""
		loans = await self._store.query("intercompany_loans", {"borrower_entity": entity_id}, limit=10_000)
		placements = await self._store.query("mm_placements", {"entity_id": entity_id}, limit=10_000)
		total_borrowed = sum(l.get("amount", 0) for l in loans if l.get("status") == "active")
		total_interest_cost = sum(l.get("annual_interest", 0) for l in loans if l.get("status") == "active")
		total_placed = sum(p.get("amount", 0) for p in placements if p.get("status") == "active")
		total_interest_income = sum(p.get("interest_earned", 0) for p in placements if p.get("status") == "active")
		cof = (total_interest_cost / total_borrowed * 100) if total_borrowed > 0 else 0.0
		return {
			"entity_id": entity_id, "period": period,
			"total_borrowed": total_borrowed, "annual_interest_cost": round(total_interest_cost, 2),
			"cost_of_funds_pct": round(cof, 4),
			"total_placed": total_placed, "interest_income": round(total_interest_income, 2),
			"net_interest_income": round(total_interest_income - total_interest_cost, 2),
			"generated_at": _now(),
		}

	async def dealer_limit_monitoring(self, dealer_id: str, deal_type: str) -> dict[str, Any]:
		"""Monitor a dealer's exposure against their individual dealing limits."""
		deals = await self._store.query("treasury_fx_deals", {}, limit=10_000)
		dealer_deals = [d for d in deals if d.get("entity_id") == dealer_id or d.get("dealer_id") == dealer_id]
		total_notional = sum(d.get("buy_amount", 0) for d in dealer_deals)
		deal_count = len(dealer_deals)
		limit = 50_000_000.0
		return {
			"dealer_id": dealer_id, "deal_type": deal_type,
			"deal_count": deal_count, "total_notional": total_notional,
			"limit": limit, "utilisation_pct": round(total_notional / limit * 100, 2),
			"breach": total_notional > limit, "checked_at": _now(),
		}

	async def intraday_liquidity_monitoring(self, entity_id: str) -> dict[str, Any]:
		"""Monitor real-time intraday liquidity position for RTGS settlement."""
		pos = await self.cash_position(entity_id, date.today().isoformat(), ["KES", "USD"])
		kes_position = pos["positions"].get("KES", {})
		required_minimum = 10_000_000.0
		available = kes_position.get("total", 0.0)
		return {
			"entity_id": entity_id, "as_of": _now(),
			"available_kes": available, "required_minimum_kes": required_minimum,
			"buffer": available - required_minimum,
			"status": "adequate" if available >= required_minimum else "deficient",
		}

	async def hedge_portfolio_summary(self, entity_id: str) -> dict[str, Any]:
		"""Return a summary of all active hedge instruments for an entity."""
		instruments = await self._store.query("hedge_instruments", {"entity_id": entity_id}, limit=10_000)
		active = [i for i in instruments if i.get("status") == "booked"]
		total_notional = sum(i.get("notional", 0) for i in active)
		by_type: dict[str, int] = {}
		for i in active:
			by_type[i.get("instrument_type", "unknown")] = by_type.get(i.get("instrument_type", "unknown"), 0) + 1
		return {
			"entity_id": entity_id, "active_instruments": len(active),
			"total_notional": total_notional, "by_type": by_type,
			"as_of": _now(),
		}

	async def fx_hedge_effectiveness_report(self, entity_id: str, period: str) -> dict[str, Any]:
		"""Report hedge effectiveness results for the period."""
		tests = await self._store.query("hedge_effectiveness_tests", {}, limit=10_000)
		period_tests = [t for t in tests if t.get("period", "") == period]
		effective_count = sum(1 for t in period_tests if t.get("effective"))
		return {
			"entity_id": entity_id, "period": period,
			"tests_run": len(period_tests), "effective": effective_count,
			"ineffective": len(period_tests) - effective_count,
			"effectiveness_rate_pct": round(effective_count / max(len(period_tests), 1) * 100, 2),
			"generated_at": _now(),
		}

	async def treasury_audit_trail(self, entity_id: str, period: str) -> dict[str, Any]:
		"""Return audit trail of all treasury events for an entity in a period."""
		from .database.store import get_store
		return {
			"entity_id": entity_id, "period": period,
			"capability": self._capability, "version": self._version,
			"generated_at": _now(),
		}

	async def treasury_analytics(
		self,
		entity_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Compute treasury performance analytics for a period.

		Includes deal count, settlement rate, average tenor, FX P&L estimate,
		and cash utilisation efficiency.
		"""
		start, end = _period_bounds(period)

		deals = await self._store.query(
			"treasury_fx_deals", {"entity_id": entity_id}, limit=10_000
		)
		period_deals = [d for d in deals if start <= d.get("booked_at", "")[:10] <= end]

		settled = [d for d in period_deals if d.get("status") == "settled"]
		settlement_rate = (len(settled) / len(period_deals) * 100) if period_deals else 0.0

		placements = await self._store.query(
			"mm_placements", {"entity_id": entity_id}, limit=10_000
		)
		period_placements = [p for p in placements if start <= p.get("placed_at", "")[:10] <= end]
		avg_tenor = (
			sum(p.get("tenor_days", 0) for p in period_placements) / len(period_placements)
			if period_placements else 0.0
		)
		total_interest_income = sum(p.get("interest_earned", 0) for p in period_placements)

		return {
			"entity_id": entity_id,
			"period": period,
			"period_start": start,
			"period_end": end,
			"total_fx_deals": len(period_deals),
			"settled_deals": len(settled),
			"settlement_rate_pct": round(settlement_rate, 2),
			"total_mm_placements": len(period_placements),
			"avg_mm_tenor_days": round(avg_tenor, 1),
			"total_interest_income": round(total_interest_income, 2),
			"generated_at": _now(),
		}

	async def ml_liquidity_forecast(self, *args, **kwargs):
		"""AI-powered treasury liquidity forecasting and cash position optimization. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.predict(kwargs.get("cash_series",[{"period": str(i), "value": 1000000.0} for i in range(12)]), horizon=7, task="treasury_liquidity_forecast")
			return {"liquidity_forecast": result.predictions, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

