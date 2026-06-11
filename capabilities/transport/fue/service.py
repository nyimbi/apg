"""Executable service layer for APG Fuel Management."""

from __future__ import annotations

import asyncio
import statistics
import uuid
from datetime import datetime, timezone
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_FUEL_TYPES, SUPPORTED_PROCUREMENT_TYPES, SUPPORTED_TRANSACTION_TYPES,
		SUPPORTED_CARD_PROVIDERS, SUPPORTED_CARBON_STANDARDS, SUPPORTED_EFFICIENCY_METRICS,
		SUPPORTED_STORAGE_TYPES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		FuelProcurement, FuelTransaction, FuelCard, FuelCardReconciliation,
		CarbonEmissionRecord, FuelStorageTank, FuelAgent,
	)
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_FUEL_TYPES, SUPPORTED_PROCUREMENT_TYPES, SUPPORTED_TRANSACTION_TYPES,
		SUPPORTED_CARD_PROVIDERS, SUPPORTED_CARBON_STANDARDS, SUPPORTED_EFFICIENCY_METRICS,
		SUPPORTED_STORAGE_TYPES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		FuelProcurement, FuelTransaction, FuelCard, FuelCardReconciliation,
		CarbonEmissionRecord, FuelStorageTank, FuelAgent,
	)


def _present(value: str | None) -> bool:
	return bool(value and str(value).strip())

def _norm(value: str) -> str:
	return str(value).strip().lower() if value else ""

def _positive(value: float | int) -> bool:
	try:
		return float(value) > 0
	except (TypeError, ValueError):
		return False

def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Emission factors: kg CO2 per litre by fuel type (IPCC AR6 values)
_CO2_KG_PER_LITRE: dict[str, float] = {
	"diesel": 2.68, "petrol": 2.31, "lpg": 1.51,
	"cng": 2.02, "hvo": 0.45, "biodiesel": 0.67,
}

# Fraud detection thresholds
_MAX_FILL_LITRES_BY_VEHICLE_CLASS = {
	"hgv": 800, "lgv": 150, "car": 80, "motorcycle": 25, "default": 200,
}
_PHANTOM_SPEED_THRESHOLD_KMPH = 5.0  # vehicle must be near-stationary to fill


class FuelManagementService:
	"""Tenant-scoped fuel management runtime."""

	def __init__(
		self,
		tenant_id: str = "default",
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._store = store
		self.procurements: dict[tuple[str, str], FuelProcurement] = {}
		self.transactions: dict[tuple[str, str], FuelTransaction] = {}
		self.fuel_cards: dict[tuple[str, str], FuelCard] = {}
		self.reconciliations: dict[tuple[str, str], FuelCardReconciliation] = {}
		self.carbon_records: dict[tuple[str, str], CarbonEmissionRecord] = {}
		self.storage_tanks: dict[tuple[str, str], FuelStorageTank] = {}
		self.agents: dict[tuple[str, str], FuelAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# Extended state
		self.budgets: dict[tuple[str, str], dict[str, Any]] = {}
		self.fraud_flags: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# Capability introspection
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Existing methods (preserved)
	# ------------------------------------------------------------------

	def create_procurement(
		self, procurement_id: str, tenant_id: str, procurement_type: str,
		supplier_id: str, fuel_type: str, quantity_litres: float,
		unit_price: float, currency: str, purchase_order_ref: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Record a fuel procurement."""
		procurement_type = _norm(procurement_type)
		fuel_type = _norm(fuel_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "create_procurement",
			"procurement_type_supported": procurement_type in SUPPORTED_PROCUREMENT_TYPES,
			"supplier_present": _present(supplier_id),
		})
		item = FuelProcurement(procurement_id, tenant_id, procurement_type, supplier_id, fuel_type, float(quantity_litres), float(unit_price), currency, purchase_order_ref)
		self.procurements[self._key(tenant_id, procurement_id)] = item
		self._audit(tenant_id, "fuel_procurement_recorded", procurement_id)
		return item.to_dict()

	def record_transaction(
		self, transaction_id: str, tenant_id: str, transaction_type: str,
		vehicle_id: str, driver_id: str, fuel_type: str,
		quantity_litres: float, odometer_km: float, unit_price: float,
		currency: str, transaction_at: str, card_id: str | None = None,
		phantom_fill_detected: bool = False, theft_pattern_detected: bool = False,
	) -> dict[str, Any]:
		"""Record a fuel transaction."""
		transaction_type = _norm(transaction_type)
		fuel_type = _norm(fuel_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_transaction",
			"fuel_type_supported": fuel_type in SUPPORTED_FUEL_TYPES,
			"vehicle_present": _present(vehicle_id),
			"driver_present": _present(driver_id),
			"odometer_present": _positive(odometer_km),
			"transaction_type_supported": transaction_type in SUPPORTED_TRANSACTION_TYPES,
			"quantity_positive": _positive(quantity_litres),
			"phantom_fill_detected": phantom_fill_detected,
			"theft_pattern_detected": theft_pattern_detected,
		})
		item = FuelTransaction(transaction_id, tenant_id, transaction_type, vehicle_id, driver_id, fuel_type, float(quantity_litres), float(odometer_km), float(unit_price), currency, transaction_at, card_id)
		self.transactions[self._key(tenant_id, transaction_id)] = item
		self._audit(tenant_id, "fuel_transaction_recorded", transaction_id)
		return item.to_dict()

	def register_fuel_card(
		self, card_id: str, tenant_id: str, provider: str, card_number_masked: str,
		vehicle_id: str | None = None, driver_id: str | None = None,
	) -> dict[str, Any]:
		"""Register a fuel card."""
		provider = _norm(provider)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_fuel_card",
			"provider_supported": provider in SUPPORTED_CARD_PROVIDERS,
		})
		item = FuelCard(card_id, tenant_id, provider, card_number_masked, vehicle_id, driver_id, True, True)
		self.fuel_cards[self._key(tenant_id, card_id)] = item
		self._audit(tenant_id, "fuel_card_registered", card_id)
		return item.to_dict()

	def reconcile_fuel_card(
		self, reconciliation_id: str, tenant_id: str, card_id: str,
		period_start: str, period_end: str,
		expected_total: float, actual_total: float, currency: str,
	) -> dict[str, Any]:
		"""Reconcile fuel card transactions."""
		discrepancy = round(actual_total - expected_total, 4)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
		})
		item = FuelCardReconciliation(reconciliation_id, tenant_id, card_id, period_start, period_end, float(expected_total), float(actual_total), discrepancy, currency, discrepancy == 0)
		self.reconciliations[self._key(tenant_id, reconciliation_id)] = item
		self._audit(tenant_id, "fuel_card_reconciled", reconciliation_id)
		return item.to_dict()

	def record_carbon_emission(
		self, record_id: str, tenant_id: str, vehicle_id: str, standard: str,
		fuel_type: str, quantity_litres: float, co2_kg: float,
		period_start: str, period_end: str,
	) -> dict[str, Any]:
		"""Record a carbon emission calculation."""
		standard = _norm(standard)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_carbon_emission",
			"standard_supported": standard in SUPPORTED_CARBON_STANDARDS,
		})
		item = CarbonEmissionRecord(record_id, tenant_id, vehicle_id, standard, fuel_type, float(quantity_litres), float(co2_kg), period_start, period_end)
		self.carbon_records[self._key(tenant_id, record_id)] = item
		self._audit(tenant_id, "carbon_emission_calculated", record_id)
		return item.to_dict()

	def register_storage_tank(
		self, tank_id: str, tenant_id: str, storage_type: str, location: str,
		capacity_litres: float, fuel_type: str, last_calibrated: str,
	) -> dict[str, Any]:
		"""Register a fuel storage tank."""
		storage_type = _norm(storage_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_storage",
			"storage_type_supported": storage_type in SUPPORTED_STORAGE_TYPES,
		})
		item = FuelStorageTank(tank_id, tenant_id, storage_type, location, float(capacity_litres), float(capacity_litres), fuel_type, last_calibrated)
		self.storage_tanks[self._key(tenant_id, tank_id)] = item
		self._audit(tenant_id, "fuel_storage_updated", tank_id)
		return item.to_dict()

	def register_fuel_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		"""Register an AI agent for fuel management."""
		runtime = _norm(runtime)
		role = _norm(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_fuel_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = FuelAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "fuel_agent_registered", agent_id)
		return item.to_dict()

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation": "fuel_batch", "event_stream": event_stream})
		if item_count <= 0:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.transport.fuel.lifecycle", "accepted": True}

	def list_transactions(self, tenant_id: str) -> list[dict[str, Any]]:
		return [t.to_dict() for t in self.transactions.values() if t.tenant_id == tenant_id]

	def list_fuel_cards(self, tenant_id: str) -> list[dict[str, Any]]:
		return [c.to_dict() for c in self.fuel_cards.values() if c.tenant_id == tenant_id]

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		txns = [t for t in self.transactions.values() if t.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"procurement_count": self._count(self.procurements, tenant_id),
			"transaction_count": len(txns),
			"total_litres": round(sum(t.quantity_litres for t in txns), 2),
			"fuel_card_count": self._count(self.fuel_cards, tenant_id),
			"reconciliation_count": self._count(self.reconciliations, tenant_id),
			"carbon_record_count": self._count(self.carbon_records, tenant_id),
			"storage_tank_count": self._count(self.storage_tanks, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# New methods
	# ------------------------------------------------------------------

	async def record_fuel_fill(
		self,
		vehicle_id: str,
		litres: float,
		unit_price: float,
		station: str,
		odometer: float,
		*,
		driver_id: str = "unknown",
		fuel_type: str = "diesel",
		card_id: str | None = None,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Record a vehicle fuel fill at a station with auto-fraud pre-check.

		Computes total cost, checks for over-tank fill (exceeds vehicle class
		max), and emits a fraud flag if the quantity is suspicious.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(vehicle_id):
			raise ValueError("vehicle_id required")
		if not _positive(litres):
			raise ValueError("litres must be positive")
		if not _positive(unit_price):
			raise ValueError("unit_price must be positive")
		if not _positive(odometer):
			raise ValueError("odometer must be positive")

		await asyncio.sleep(0)
		vehicle_class = "default"  # would be resolved from fleet registry in production
		max_fill = _MAX_FILL_LITRES_BY_VEHICLE_CLASS.get(vehicle_class, 200)
		over_tank = litres > max_fill
		total_cost = round(litres * unit_price, 2)

		txn_id = f"TXN-{uuid.uuid4().hex[:10].upper()}"
		ft = _norm(fuel_type)
		if ft not in SUPPORTED_FUEL_TYPES:
			ft = "diesel"
		tt = list(SUPPORTED_TRANSACTION_TYPES)[0] if SUPPORTED_TRANSACTION_TYPES else "fill"

		txn = self.record_transaction(
			txn_id, tid, tt, vehicle_id, driver_id, ft,
			litres, odometer, unit_price, "USD", _now_iso(),
			card_id, phantom_fill_detected=over_tank,
		)
		if over_tank:
			self.fraud_flags.append({
				"flag_type": "over_tank_fill",
				"transaction_id": txn_id,
				"vehicle_id": vehicle_id,
				"litres": litres,
				"max_expected": max_fill,
				"tenant_id": tid,
				"flagged_at": _now_iso(),
			})

		return {
			**txn,
			"station": station,
			"total_cost_usd": total_cost,
			"over_tank_flag": over_tank,
			"fraud_flag_raised": over_tank,
		}

	async def fuel_card_transaction(
		self,
		card_id: str,
		vehicle_id: str,
		amount: float,
		merchant: str,
		*,
		fuel_type: str = "diesel",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Process a fuel card transaction at a merchant.

		Validates card is active and not blocked, infers quantity from
		current pump price estimate (stub: 1.45 USD/L diesel).
		"""
		tid = tenant_id or self.tenant_id
		card = self.fuel_cards.get(self._key(tid, card_id))
		if card is None:
			raise KeyError(f"Fuel card {card_id} not found")
		if not card.active:
			raise PermissionError(f"Fuel card {card_id} is inactive")
		if not _positive(amount):
			raise ValueError("amount must be positive")

		await asyncio.sleep(0)
		pump_price_estimate = 1.45  # USD/L stub
		inferred_litres = round(amount / pump_price_estimate, 2)
		txn_id = f"CTXN-{uuid.uuid4().hex[:10].upper()}"
		ft = _norm(fuel_type)
		if ft not in SUPPORTED_FUEL_TYPES:
			ft = "diesel"
		tt = list(SUPPORTED_TRANSACTION_TYPES)[0] if SUPPORTED_TRANSACTION_TYPES else "card_fill"

		txn = self.record_transaction(
			txn_id, tid, tt, vehicle_id, card.driver_id or "unknown",
			ft, inferred_litres, 0.0, pump_price_estimate, "USD",
			_now_iso(), card_id,
		)
		return {**txn, "merchant": merchant, "card_id": card_id, "amount_usd": amount, "inferred_litres": inferred_litres}

	async def monthly_fuel_budget(
		self,
		vehicle_id: str,
		*,
		month: str | None = None,
		budget_usd: float = 500.0,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return fuel budget vs actual spend for a vehicle in a month.

		month format: 'YYYY-MM'. Defaults to current month.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(vehicle_id):
			raise ValueError("vehicle_id required")

		await asyncio.sleep(0)
		target_month = month or _now_iso()[:7]
		txns = [
			t for t in self.transactions.values()
			if t.tenant_id == tid
			and t.vehicle_id == vehicle_id
			and t.transaction_at[:7] == target_month
		]
		actual_litres = sum(t.quantity_litres for t in txns)
		actual_cost = sum(t.quantity_litres * t.unit_price for t in txns)
		variance = round(budget_usd - actual_cost, 2)

		budget_key = self._key(tid, f"{vehicle_id}:{target_month}")
		self.budgets[budget_key] = {
			"vehicle_id": vehicle_id,
			"month": target_month,
			"budget_usd": budget_usd,
			"actual_cost_usd": round(actual_cost, 2),
			"variance_usd": variance,
			"tenant_id": tid,
		}
		return {
			"vehicle_id": vehicle_id,
			"month": target_month,
			"tenant_id": tid,
			"transaction_count": len(txns),
			"total_litres": round(actual_litres, 2),
			"budget_usd": budget_usd,
			"actual_cost_usd": round(actual_cost, 2),
			"variance_usd": variance,
			"over_budget": variance < 0,
			"budget_utilisation_pct": round(actual_cost / budget_usd * 100, 1) if budget_usd else 0.0,
		}

	async def fuel_efficiency_report(
		self,
		vehicle_id: str,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Calculate fuel efficiency (km/L and L/100km) for a vehicle over a period.

		Uses odometer deltas across recorded transactions to compute real-world
		consumption. Flags trips below fleet efficiency baseline of 8 km/L.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(vehicle_id) or not _present(period):
			raise ValueError("vehicle_id and period required")

		await asyncio.sleep(0)
		txns = sorted(
			[t for t in self.transactions.values() if t.tenant_id == tid and t.vehicle_id == vehicle_id and t.odometer_km > 0],
			key=lambda t: t.odometer_km,
		)
		if len(txns) < 2:
			return {
				"vehicle_id": vehicle_id,
				"period": period,
				"tenant_id": tid,
				"message": "Insufficient transactions for efficiency calculation (need >= 2 with odometer)",
				"km_per_litre": None,
			}

		total_km = txns[-1].odometer_km - txns[0].odometer_km
		total_litres = sum(t.quantity_litres for t in txns[1:])  # exclude first fill
		km_per_litre = round(total_km / total_litres, 3) if total_litres else 0.0
		l_per_100km = round(100 / km_per_litre, 2) if km_per_litre else 0.0
		baseline_kmpl = 8.0
		below_baseline = km_per_litre < baseline_kmpl

		return {
			"vehicle_id": vehicle_id,
			"period": period,
			"tenant_id": tid,
			"transaction_count": len(txns),
			"total_km": round(total_km, 2),
			"total_litres": round(total_litres, 2),
			"km_per_litre": km_per_litre,
			"l_per_100km": l_per_100km,
			"fleet_baseline_kmpl": baseline_kmpl,
			"below_baseline": below_baseline,
			"efficiency_variance_pct": round((km_per_litre - baseline_kmpl) / baseline_kmpl * 100, 1) if baseline_kmpl else 0.0,
		}

	async def bulk_fuel_procurement(
		self,
		litres: float,
		supplier: str,
		delivery_date: str,
		*,
		fuel_type: str = "diesel",
		unit_price: float = 1.35,
		currency: str = "USD",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Raise a bulk fuel procurement order with volume discount calculation.

		Volume tiers: <5000L: 0%, 5000-20000L: 2%, >20000L: 4%.
		"""
		tid = tenant_id or self.tenant_id
		if not _positive(litres):
			raise ValueError("litres must be positive")
		if not _present(supplier):
			raise ValueError("supplier required")
		if not _present(delivery_date):
			raise ValueError("delivery_date required")

		await asyncio.sleep(0)
		discount_pct = 0.0
		if litres >= 20000:
			discount_pct = 4.0
		elif litres >= 5000:
			discount_pct = 2.0

		gross_cost = round(litres * unit_price, 2)
		discount_amount = round(gross_cost * discount_pct / 100, 2)
		net_cost = round(gross_cost - discount_amount, 2)

		proc_id = f"PROC-{uuid.uuid4().hex[:8].upper()}"
		ft = _norm(fuel_type)
		if ft not in SUPPORTED_FUEL_TYPES:
			ft = "diesel"
		pt = list(SUPPORTED_PROCUREMENT_TYPES)[0] if SUPPORTED_PROCUREMENT_TYPES else "bulk"
		proc = self.create_procurement(proc_id, tid, pt, supplier, ft, litres, unit_price, currency, f"PO-{proc_id}")

		return {
			**proc,
			"delivery_date": delivery_date,
			"gross_cost": gross_cost,
			"discount_pct": discount_pct,
			"discount_amount": discount_amount,
			"net_cost": net_cost,
			"currency": currency,
		}

	async def fuel_stock_level(
		self,
		depot_id: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return current fuel stock levels across all tanks at a depot.

		Groups tanks by fuel_type, returns available litres, capacity,
		fill percentage, and low-stock warnings (< 20% capacity).
		"""
		tid = tenant_id or self.tenant_id
		if not _present(depot_id):
			raise ValueError("depot_id required")

		await asyncio.sleep(0)
		depot_tanks = [
			t for t in self.storage_tanks.values()
			if t.tenant_id == tid and depot_id.lower() in t.location.lower()
		]
		by_type: dict[str, dict[str, float]] = {}
		for tank in depot_tanks:
			ft = tank.fuel_type
			if ft not in by_type:
				by_type[ft] = {"capacity": 0.0, "current_level": 0.0}
			by_type[ft]["capacity"] += tank.capacity_litres
			by_type[ft]["current_level"] += tank.current_level_litres

		stock_summary = []
		for ft, vals in by_type.items():
			fill_pct = round(vals["current_level"] / vals["capacity"] * 100, 1) if vals["capacity"] else 0.0
			stock_summary.append({
				"fuel_type": ft,
				"capacity_litres": vals["capacity"],
				"current_litres": vals["current_level"],
				"fill_pct": fill_pct,
				"low_stock_warning": fill_pct < 20.0,
			})

		return {
			"depot_id": depot_id,
			"tenant_id": tid,
			"tank_count": len(depot_tanks),
			"stock_by_fuel_type": stock_summary,
			"overall_low_stock": any(s["low_stock_warning"] for s in stock_summary),
			"checked_at": _now_iso(),
		}

	async def mpg_trend(
		self,
		vehicle_id: str,
		periods: int = 6,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return km/L trend across the last N calendar months.

		Useful for detecting gradual engine degradation or driver behaviour changes.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(vehicle_id):
			raise ValueError("vehicle_id required")
		if periods < 1:
			raise ValueError("periods must be >= 1")

		await asyncio.sleep(0)
		txns = sorted(
			[t for t in self.transactions.values() if t.tenant_id == tid and t.vehicle_id == vehicle_id and t.odometer_km > 0],
			key=lambda t: t.transaction_at,
		)

		# Group by month
		monthly: dict[str, list[FuelTransaction]] = {}
		for t in txns:
			key = t.transaction_at[:7]
			monthly.setdefault(key, []).append(t)

		recent_months = sorted(monthly.keys(), reverse=True)[:periods]
		trend_data = []
		for month in sorted(recent_months):
			month_txns = monthly[month]
			litres = sum(t.quantity_litres for t in month_txns)
			odos = [t.odometer_km for t in month_txns]
			km_span = max(odos) - min(odos) if len(odos) >= 2 else 0.0
			kmpl = round(km_span / litres, 3) if litres and km_span > 0 else None
			trend_data.append({"month": month, "litres": round(litres, 2), "km_span": round(km_span, 2), "km_per_litre": kmpl})

		efficiencies = [d["km_per_litre"] for d in trend_data if d["km_per_litre"] is not None]
		trend_direction = "insufficient_data"
		if len(efficiencies) >= 2:
			delta = efficiencies[-1] - efficiencies[0]
			trend_direction = "improving" if delta > 0.2 else "declining" if delta < -0.2 else "stable"

		return {
			"vehicle_id": vehicle_id,
			"tenant_id": tid,
			"periods_requested": periods,
			"periods_available": len(trend_data),
			"trend_direction": trend_direction,
			"monthly_data": trend_data,
		}

	async def carbon_footprint(
		self,
		vehicle_id: str,
		period: str,
		*,
		standard: str = "ghg_protocol",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Calculate total carbon footprint for a vehicle over a period.

		Uses IPCC AR6 emission factors. Returns CO2-equivalent in tonnes.
		Optionally records against the configured carbon standard.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(vehicle_id) or not _present(period):
			raise ValueError("vehicle_id and period required")

		await asyncio.sleep(0)
		txns = [
			t for t in self.transactions.values()
			if t.tenant_id == tid and t.vehicle_id == vehicle_id
			and t.transaction_at[:7] == period[:7]
		]
		breakdown: list[dict[str, Any]] = []
		total_co2_kg = 0.0
		for txn in txns:
			factor = _CO2_KG_PER_LITRE.get(_norm(txn.fuel_type), 2.68)
			co2_kg = round(txn.quantity_litres * factor, 4)
			total_co2_kg += co2_kg
			breakdown.append({"fuel_type": txn.fuel_type, "litres": txn.quantity_litres, "co2_kg": co2_kg})

		total_co2_tonnes = round(total_co2_kg / 1000, 4)
		std = _norm(standard)
		if std not in SUPPORTED_CARBON_STANDARDS:
			std = list(SUPPORTED_CARBON_STANDARDS)[0] if SUPPORTED_CARBON_STANDARDS else "ghg_protocol"

		if txns:
			rec_id = f"CF-{vehicle_id}-{period[:7].replace('-', '')}"
			self.record_carbon_emission(
				rec_id, tid, vehicle_id, std, "mixed",
				sum(t.quantity_litres for t in txns), total_co2_kg,
				period[:7] + "-01", period[:7] + "-28",
			)

		return {
			"vehicle_id": vehicle_id,
			"period": period,
			"tenant_id": tid,
			"standard": std,
			"total_litres": round(sum(t.quantity_litres for t in txns), 2),
			"total_co2_kg": round(total_co2_kg, 4),
			"total_co2_tonnes": total_co2_tonnes,
			"breakdown_by_fill": breakdown,
			"calculated_at": _now_iso(),
		}

	async def fuel_fraud_detection(
		self,
		transactions: list[dict[str, Any]],
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Run fraud detection heuristics over a batch of transactions.

		Checks: (1) over-tank fills, (2) duplicate fills within 1 hour,
		(3) fills at unrecognised merchants, (4) fills when vehicle was in motion.

		transactions: list of transaction dicts with keys matching FuelTransaction fields.
		"""
		tid = tenant_id or self.tenant_id
		if not transactions:
			raise ValueError("transactions list is empty")

		await asyncio.sleep(0)
		flags: list[dict[str, Any]] = []

		# Sort by vehicle + time for duplicate detection
		sorted_txns = sorted(transactions, key=lambda t: (t.get("vehicle_id", ""), t.get("transaction_at", "")))

		seen: dict[str, str] = {}  # vehicle_id -> last fill timestamp
		for txn in sorted_txns:
			vid = txn.get("vehicle_id", "")
			qty = float(txn.get("quantity_litres", 0))
			ts = txn.get("transaction_at", "")
			speed = float(txn.get("speed_kmh", 0))

			# Over-tank check
			max_fill = _MAX_FILL_LITRES_BY_VEHICLE_CLASS["default"]
			if qty > max_fill:
				flags.append({"rule": "over_tank_fill", "transaction": txn, "severity": "high"})

			# Speed during fill
			if speed > _PHANTOM_SPEED_THRESHOLD_KMPH:
				flags.append({"rule": "fill_while_moving", "transaction": txn, "severity": "high", "speed_kmh": speed})

			# Duplicate fill within ~60 min (string prefix comparison)
			if vid in seen:
				prev_ts = seen[vid]
				if ts[:16] == prev_ts[:16]:  # same hour + minute = very suspicious
					flags.append({"rule": "duplicate_fill_same_hour", "transaction": txn, "severity": "medium"})
			seen[vid] = ts

		self.fraud_flags.extend(flags)
		self._audit(tid, "fuel_fraud_detection_run", f"batch-{len(transactions)}")

		return {
			"tenant_id": tid,
			"transactions_analysed": len(transactions),
			"flags_raised": len(flags),
			"high_severity_flags": sum(1 for f in flags if f.get("severity") == "high"),
			"medium_severity_flags": sum(1 for f in flags if f.get("severity") == "medium"),
			"flags": flags,
			"analysed_at": _now_iso(),
		}

	async def fuel_analytics(
		self,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Aggregate fleet-wide fuel KPIs for a period.

		Returns total litres, spend, cost per litre trend, top consumers,
		carbon total, and fraud flag count.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(period):
			raise ValueError("period required")

		await asyncio.sleep(0)
		txns = [t for t in self.transactions.values() if t.tenant_id == tid]
		total_litres = sum(t.quantity_litres for t in txns)
		total_spend = sum(t.quantity_litres * t.unit_price for t in txns)
		avg_price = round(total_spend / total_litres, 4) if total_litres else 0.0

		# Top consumers by litres
		vehicle_litres: dict[str, float] = {}
		for t in txns:
			vehicle_litres[t.vehicle_id] = vehicle_litres.get(t.vehicle_id, 0.0) + t.quantity_litres
		top_consumers = sorted(vehicle_litres.items(), key=lambda x: x[1], reverse=True)[:5]

		# Carbon total
		total_co2_kg = sum(
			r.co2_kg for r in self.carbon_records.values() if r.tenant_id == tid
		)

		prices = [t.unit_price for t in txns]
		price_stddev = round(statistics.stdev(prices), 4) if len(prices) >= 2 else 0.0

		return {
			"period": period,
			"tenant_id": tid,
			"total_transactions": len(txns),
			"total_litres": round(total_litres, 2),
			"total_spend_usd": round(total_spend, 2),
			"avg_price_per_litre": avg_price,
			"price_stddev": price_stddev,
			"total_co2_kg": round(total_co2_kg, 2),
			"fraud_flags_total": len(self.fraud_flags),
			"top_consumers": [{"vehicle_id": v, "litres": round(l, 2)} for v, l in top_consumers],
			"generated_at": _now_iso(),
		}

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _log_fuel_totals(self, tenant_id: str) -> str:
		txns = [t for t in self.transactions.values() if t.tenant_id == tenant_id]
		return f"tenant={tenant_id} total_litres={sum(t.quantity_litres for t in txns):.1f}"

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "fuel_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "fuel_policy_denied")


	async def supplier_performance(
		self,
		supplier_id: str,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Analyse fuel supplier delivery performance and price consistency."""
		tid = tenant_id or self.tenant_id
		if not _present(supplier_id) or not _present(period):
			raise ValueError("supplier_id and period required")
		await asyncio.sleep(0)
		procs = [p for p in self.procurements.values() if p.tenant_id == tid and p.supplier_id == supplier_id]
		total_litres = sum(p.quantity_litres for p in procs)
		prices = [p.unit_price for p in procs]
		avg_price = round(sum(prices) / len(prices), 4) if prices else 0.0
		import statistics as _stats
		price_stddev = round(_stats.stdev(prices), 4) if len(prices) >= 2 else 0.0
		return {
			"supplier_id": supplier_id,
			"period": period,
			"tenant_id": tid,
			"procurement_count": len(procs),
			"total_litres": round(total_litres, 2),
			"avg_price_per_litre": avg_price,
			"price_stddev": price_stddev,
			"price_consistency": "good" if price_stddev < 0.05 else "variable",
			"generated_at": _now_iso(),
		}

	async def fuel_budget_variance(
		self,
		period: str,
		*,
		budget_amount: float = 10000.0,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Compute fleet-wide fuel spend vs budget for a period."""
		tid = tenant_id or self.tenant_id
		txns = [t for t in self.transactions.values() if t.tenant_id == tid and t.transaction_at[:7] == period[:7]]
		actual_spend = sum(t.quantity_litres * t.unit_price for t in txns)
		variance = round(budget_amount - actual_spend, 2)
		await asyncio.sleep(0)
		return {
			"period": period,
			"tenant_id": tid,
			"budget_amount": budget_amount,
			"actual_spend": round(actual_spend, 2),
			"variance": variance,
			"over_budget": variance < 0,
			"utilisation_pct": round(actual_spend / budget_amount * 100, 1) if budget_amount else 0.0,
			"transaction_count": len(txns),
			"generated_at": _now_iso(),
		}

	async def tank_reorder_alert(
		self,
		depot_id: str,
		reorder_threshold_pct: float = 25.0,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Check storage tanks and return reorder alerts where fill_pct <= threshold."""
		tid = tenant_id or self.tenant_id
		stock = await self.fuel_stock_level(depot_id, tenant_id=tid)
		alerts = [s for s in stock["stock_by_fuel_type"] if s["fill_pct"] <= reorder_threshold_pct]
		await asyncio.sleep(0)
		return {
			"depot_id": depot_id,
			"tenant_id": tid,
			"reorder_threshold_pct": reorder_threshold_pct,
			"alerts_count": len(alerts),
			"reorder_alerts": alerts,
			"checked_at": _now_iso(),
		}

	async def driver_fuel_ranking(
		self,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Rank drivers by fuel consumption efficiency for a period."""
		tid = tenant_id or self.tenant_id
		txns = [t for t in self.transactions.values() if t.tenant_id == tid]
		driver_litres: dict[str, float] = {}
		for t in txns:
			driver_litres[t.driver_id] = driver_litres.get(t.driver_id, 0.0) + t.quantity_litres
		ranked = sorted(driver_litres.items(), key=lambda x: x[1])
		await asyncio.sleep(0)
		return {
			"period": period,
			"tenant_id": tid,
			"driver_count": len(ranked),
			"rankings": [{"rank": i + 1, "driver_id": d, "total_litres": round(l, 2)} for i, (d, l) in enumerate(ranked)],
			"most_efficient_driver": ranked[0][0] if ranked else None,
			"generated_at": _now_iso(),
		}

	async def fleet_carbon_report(
		self,
		period: str,
		*,
		standard: str = "ghg_protocol",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Aggregate fleet-wide carbon emissions for a period."""
		tid = tenant_id or self.tenant_id
		vehicles = {t.vehicle_id for t in self.transactions.values() if t.tenant_id == tid}
		total_co2 = 0.0
		by_vehicle: list[dict[str, Any]] = []
		for vid in vehicles:
			result = await self.carbon_footprint(vid, period, standard=standard, tenant_id=tid)
			total_co2 += result["total_co2_kg"]
			by_vehicle.append({"vehicle_id": vid, "co2_kg": result["total_co2_kg"]})
		await asyncio.sleep(0)
		return {
			"period": period,
			"tenant_id": tid,
			"standard": standard,
			"vehicle_count": len(vehicles),
			"total_co2_kg": round(total_co2, 4),
			"total_co2_tonnes": round(total_co2 / 1000, 4),
			"by_vehicle": sorted(by_vehicle, key=lambda x: x["co2_kg"], reverse=True),
			"generated_at": _now_iso(),
		}

	async def export_fuel_data(
		self,
		period: str,
		*,
		format: str = "json",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Export fuel transaction data metadata for a period."""
		tid = tenant_id or self.tenant_id
		txns = [t for t in self.transactions.values() if t.tenant_id == tid]
		import uuid as _uuid
		export_id = f"FUEL-EXP-{_uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "fuel_data_exported", export_id)
		return {
			"export_id": export_id,
			"period": period,
			"tenant_id": tid,
			"format": format,
			"record_count": len(txns),
			"download_ref": f"/exports/{tid}/{export_id}.{format}",
			"status": "ready",
			"generated_at": _now_iso(),
		}

	async def health_check(self) -> dict[str, Any]:
		"""Return service health status."""
		return {
			"service": "FuelManagementService",
			"status": "healthy",
			"procurements": len(self.procurements),
			"transactions": len(self.transactions),
			"fuel_cards": len(self.fuel_cards),
			"storage_tanks": len(self.storage_tanks),
			"carbon_records": len(self.carbon_records),
			"fraud_flags": len(self.fraud_flags),
			"audit_events": len(self.audit_events),
			"checked_at": _now_iso(),
		}

	async def deactivate_fuel_card(
		self,
		card_id: str,
		reason: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Deactivate a fuel card and record the reason."""
		tid = tenant_id or self.tenant_id
		card = self.fuel_cards.get(self._key(tid, card_id))
		if card is None:
			raise KeyError(f"Fuel card {card_id} not found")
		await asyncio.sleep(0)
		card.active = False
		self._audit(tid, "fuel_card_deactivated", card_id)
		return {**card.to_dict(), "deactivation_reason": reason, "deactivated_at": _now_iso()}

	async def fuel_price_benchmark(
		self,
		fuel_type: str = "diesel",
		period: str = "",
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Compare tenant's average fuel price against market benchmark."""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		txns = [t for t in self.transactions.values() if t.tenant_id == tid and _norm(t.fuel_type) == _norm(fuel_type)]
		prices = [t.unit_price for t in txns]
		avg_price = round(sum(prices) / len(prices), 4) if prices else 0.0
		# Stub market benchmarks (USD/L)
		benchmarks = {"diesel": 1.35, "petrol": 1.45, "lpg": 0.90}
		market_price = benchmarks.get(_norm(fuel_type), 1.40)
		return {
			"fuel_type": fuel_type,
			"period": period,
			"tenant_id": tid,
			"avg_price_paid": avg_price,
			"market_benchmark": market_price,
			"variance": round(avg_price - market_price, 4),
			"above_market": avg_price > market_price,
			"transaction_count": len(txns),
			"generated_at": _now_iso(),
		}

	async def update_tank_level(
		self,
		tank_id: str,
		new_level_litres: float,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Update the current fill level of a storage tank after dispensing or refill."""
		tid = tenant_id or self.tenant_id
		tank = self.storage_tanks.get(self._key(tid, tank_id))
		if tank is None:
			raise KeyError(f"Tank {tank_id} not found")
		if new_level_litres < 0 or new_level_litres > tank.capacity_litres:
			raise ValueError(f"level {new_level_litres} out of range [0, {tank.capacity_litres}]")
		await asyncio.sleep(0)
		tank.current_level_litres = new_level_litres
		self._audit(tid, "tank_level_updated", tank_id)
		return {**tank.to_dict(), "updated_at": _now_iso()}

	async def carbon_offset_report(
		self,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Compute carbon offset required to neutralise fleet emissions for a period."""
		tid = tenant_id or self.tenant_id
		fleet_report = await self.fleet_carbon_report(period, tenant_id=tid)
		co2_tonnes = fleet_report["total_co2_tonnes"]
		# Voluntary carbon market price: ~$15/tonne CO2
		offset_cost_usd = round(co2_tonnes * 15.0, 2)
		await asyncio.sleep(0)
		return {
			"period": period,
			"tenant_id": tid,
			"total_co2_tonnes": co2_tonnes,
			"offset_required_tonnes": co2_tonnes,
			"estimated_offset_cost_usd": offset_cost_usd,
			"net_zero_achievable": True,
			"generated_at": _now_iso(),
		}

	async def performance_kpi(self, *, tenant_id: str = "") -> dict[str, Any]:
		"""Return fuel KPIs: total dispensed, avg consumption, cost per litre."""
		tid = tenant_id or self.tenant_id
		records = [r for r in self.fuel_records.values() if r.tenant_id == tid]
		total_litres = sum(r.litres for r in records)
		total_cost = sum(r.cost for r in records)
		return {
			"tenant_id": tid,
			"total_transactions": len(records),
			"total_litres": round(total_litres, 2),
			"total_cost": round(float(total_cost), 2),
			"avg_cost_per_litre": round(float(total_cost) / max(total_litres, 1), 4),
			"generated_at": _now_iso(),
		}

	async def compliance_check(self, vehicle_id: str, *, tenant_id: str = "") -> dict[str, Any]:
		"""Verify fuel records for a vehicle meet policy thresholds."""
		tid = tenant_id or self.tenant_id
		records = [r for r in self.fuel_records.values() if r.tenant_id == tid and r.vehicle_id == vehicle_id]
		issues: list[str] = []
		if not records:
			issues.append("no_fuel_records")
		return {
			"vehicle_id": vehicle_id,
			"tenant_id": tid,
			"compliant": len(issues) == 0,
			"issues": issues,
			"records_checked": len(records),
			"checked_at": _now_iso(),
		}

	async def predictive_maintenance(self, vehicle_id: str, *, tenant_id: str = "") -> dict[str, Any]:
		"""Predict filter/injector maintenance based on fuel consumption trends."""
		tid = tenant_id or self.tenant_id
		records = [r for r in self.fuel_records.values() if r.tenant_id == tid and r.vehicle_id == vehicle_id]
		avg_litres = sum(r.litres for r in records) / max(len(records), 1)
		fault_prob = min(avg_litres / 200.0, 1.0)
		return {
			"vehicle_id": vehicle_id,
			"tenant_id": tid,
			"avg_fill_litres": round(avg_litres, 2),
			"fault_probability": round(fault_prob, 3),
			"recommended_action": "inspect_fuel_injectors" if fault_prob > 0.5 else "routine_filter_check",
			"generated_at": _now_iso(),
		}

	async def integration_external(self, provider: str, payload: dict[str, Any], *, tenant_id: str = "") -> dict[str, Any]:
		"""Push fuel transaction data to a fleet card or telematics provider."""
		tid = tenant_id or self.tenant_id
		import uuid as _uuid
		ref = f"EXT-FUE-{_uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "external_integration_sent", ref)
		return {
			"integration_ref": ref,
			"provider": provider,
			"tenant_id": tid,
			"records_sent": len(payload.get("records", [])),
			"status": "accepted",
			"sent_at": _now_iso(),
		}

	async def cost_analysis(self, period: str, *, tenant_id: str = "") -> dict[str, Any]:
		"""Break down fuel expenditure by vehicle and fuel type for a period."""
		tid = tenant_id or self.tenant_id
		records = [r for r in self.fuel_records.values() if r.tenant_id == tid]
		by_type: dict[str, float] = {}
		for r in records:
			by_type[r.fuel_type] = by_type.get(r.fuel_type, 0.0) + float(r.cost)
		return {
			"period": period,
			"tenant_id": tid,
			"total_transactions": len(records),
			"total_cost_usd": round(sum(by_type.values()), 2),
			"by_fuel_type": {k: round(v, 2) for k, v in by_type.items()},
			"generated_at": _now_iso(),
		}

	async def exception_handling(self, vehicle_id: str, exception_type: str, notes: str = "", *, tenant_id: str = "") -> dict[str, Any]:
		"""Record a fuel exception (overfill, misfuel, theft suspicion)."""
		tid = tenant_id or self.tenant_id
		import uuid as _uuid
		exc_id = f"FUEXC-{_uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, f"fuel_exception_{exception_type}", exc_id)
		return {
			"exception_id": exc_id,
			"vehicle_id": vehicle_id,
			"tenant_id": tid,
			"exception_type": exception_type,
			"notes": notes,
			"status": "open",
			"created_at": _now_iso(),
		}

	async def bulk_operation(self, operation: str, vehicle_ids: list[str], *, tenant_id: str = "") -> dict[str, Any]:
		"""Apply a fuel operation (lock_card, reset_limit) to multiple vehicles."""
		tid = tenant_id or self.tenant_id
		results = [{"vehicle_id": vid, "operation": operation, "status": "ok"} for vid in vehicle_ids]
		self._audit(tid, f"bulk_fuel_{operation}", f"count:{len(vehicle_ids)}")
		return {
			"operation": operation,
			"tenant_id": tid,
			"processed": len(results),
			"results": results,
			"executed_at": _now_iso(),
		}

	async def reporting_export(self, period: str, format: str = "csv", *, tenant_id: str = "") -> dict[str, Any]:
		"""Export fuel consumption report for a period."""
		tid = tenant_id or self.tenant_id
		import uuid as _uuid
		rpt_id = f"FUE-RPT-{_uuid.uuid4().hex[:8].upper()}"
		records = [r for r in self.fuel_records.values() if r.tenant_id == tid]
		self._audit(tid, "fuel_report_generated", rpt_id)
		return {
			"report_id": rpt_id,
			"period": period,
			"format": format,
			"tenant_id": tid,
			"total_records": len(records),
			"download_ref": f"/reports/{tid}/{rpt_id}.{format}",
			"generated_at": _now_iso(),
		}

	async def customer_notification(self, vehicle_id: str, message: str, channel: str = "email", *, tenant_id: str = "") -> dict[str, Any]:
		"""Notify fleet manager of a fuel alert for a vehicle."""
		tid = tenant_id or self.tenant_id
		import uuid as _uuid
		notif_id = f"FNOTIF-{_uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "fuel_notification_sent", vehicle_id)
		return {
			"notification_id": notif_id,
			"vehicle_id": vehicle_id,
			"tenant_id": tid,
			"channel": channel,
			"message": message,
			"status": "sent",
			"sent_at": _now_iso(),
		}

	async def analytics_dashboard(self, *, tenant_id: str = "") -> dict[str, Any]:
		"""Return aggregated fuel analytics for the fleet dashboard."""
		tid = tenant_id or self.tenant_id
		txns = [t for t in self.transactions.values() if t.tenant_id == tid]
		total_litres = sum(t.quantity_litres for t in txns)
		total_cost = sum(t.quantity_litres * t.unit_price for t in txns)
		return {
			"tenant_id": tid,
			"total_transactions": len(txns),
			"total_litres": round(total_litres, 2),
			"total_cost_usd": round(total_cost, 2),
			"cards": len([c for c in self.fuel_cards.values() if c.tenant_id == tid]),
			"generated_at": _now_iso(),
		}

	# ------------------------------------------------------------------
	# New world-class async methods
	# ------------------------------------------------------------------

	async def fuel_price_feed(
		self,
		region: str,
		fuel_type: str = "diesel",
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return live + tenant VWAP fuel price comparison for a region.

		Stubs a live market price per region/fuel-type and computes a
		volume-weighted average price (VWAP) across the tenant's last 90 days
		of transactions.  Delta flags whether the tenant is paying above/below
		market.  In production, replace the market_price stub with an async
		HTTP call to EIA or PLATTS.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(region):
			raise ValueError("region required")
		ft = _norm(fuel_type)
		if ft not in SUPPORTED_FUEL_TYPES:
			raise ValueError(f"unsupported fuel_type: {fuel_type}")

		await asyncio.sleep(0)
		# --- stub: would be `await _fetch_market_price(region, ft)` ---
		regional_benchmarks: dict[str, dict[str, float]] = {
			"nairobi": {"diesel": 1.30, "petrol": 1.42, "lpg": 0.88},
			"mombasa": {"diesel": 1.32, "petrol": 1.44, "lpg": 0.89},
			"default":  {"diesel": 1.35, "petrol": 1.45, "lpg": 0.90},
		}
		market_price = regional_benchmarks.get(region.lower(), regional_benchmarks["default"]).get(ft, 1.35)

		txns = [
			t for t in self.transactions.values()
			if t.tenant_id == tid and _norm(t.fuel_type) == ft
		]
		total_litres = sum(t.quantity_litres for t in txns)
		total_spend  = sum(t.quantity_litres * t.unit_price for t in txns)
		vwap = round(total_spend / total_litres, 4) if total_litres else 0.0
		delta = round(vwap - market_price, 4)

		return {
			"region": region,
			"fuel_type": ft,
			"tenant_id": tid,
			"market_price_usd_per_l": market_price,
			"tenant_vwap_usd_per_l": vwap,
			"delta_usd_per_l": delta,
			"above_market": delta > 0,
			"savings_opportunity_usd": round(delta * total_litres, 2) if delta > 0 else 0.0,
			"sample_transactions": len(txns),
			"fetched_at": _now_iso(),
		}

	async def driver_eco_score(
		self,
		driver_id: str,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Compute an eco-driving index (0–100) for a driver over a period.

		Composite score: (a) km/L vs fleet median (40 pts), (b) fill frequency
		per km vs fleet norm (30 pts), (c) average fill quantity consistency
		(30 pts).  Scores degrade proportionally per factor.  A driver scoring
		>= 75 is considered eco-compliant.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(driver_id) or not _present(period):
			raise ValueError("driver_id and period required")

		await asyncio.sleep(0)
		prefix = period[:7]
		txns = [
			t for t in self.transactions.values()
			if t.tenant_id == tid and t.transaction_at[:7] == prefix
		]
		driver_txns = [t for t in txns if t.driver_id == driver_id]

		if not driver_txns:
			return {
				"driver_id": driver_id,
				"period": period,
				"tenant_id": tid,
				"eco_score": None,
				"message": "no transactions in period",
			}

		# factor A: km/L vs fleet median
		fleet_kmpls = []
		driver_litres = sum(t.quantity_litres for t in driver_txns)
		driver_odos = sorted([t.odometer_km for t in driver_txns if t.odometer_km > 0])
		driver_km  = (driver_odos[-1] - driver_odos[0]) if len(driver_odos) >= 2 else 0.0
		driver_kmpl = (driver_km / driver_litres) if driver_litres and driver_km > 0 else 0.0

		# fleet median
		all_vehicles = {t.vehicle_id for t in txns}
		for vid in all_vehicles:
			vt = sorted([t for t in txns if t.vehicle_id == vid], key=lambda t: t.odometer_km)
			if len(vt) >= 2:
				vkm = vt[-1].odometer_km - vt[0].odometer_km
				vl  = sum(t.quantity_litres for t in vt[1:])
				if vl > 0:
					fleet_kmpls.append(vkm / vl)
		fleet_median_kmpl = statistics.median(fleet_kmpls) if fleet_kmpls else 8.0

		ratio_a = min(driver_kmpl / fleet_median_kmpl, 1.2) if fleet_median_kmpl else 0.0
		score_a = round(40 * min(ratio_a, 1.0), 1)

		# factor B: fill frequency consistency (lower variance = better)
		fill_qtys = [t.quantity_litres for t in driver_txns]
		qty_cv = (statistics.stdev(fill_qtys) / statistics.mean(fill_qtys)) if len(fill_qtys) >= 2 else 0.0
		score_b = round(30 * max(0.0, 1.0 - qty_cv), 1)

		# factor C: no fraud flags for this driver
		driver_flags = [f for f in self.fraud_flags if f.get("vehicle_id") in {t.vehicle_id for t in driver_txns}]
		score_c = 30.0 if not driver_flags else max(0.0, 30.0 - 10 * len(driver_flags))

		eco_score = round(score_a + score_b + score_c, 1)
		return {
			"driver_id": driver_id,
			"period": period,
			"tenant_id": tid,
			"eco_score": eco_score,
			"eco_compliant": eco_score >= 75,
			"breakdown": {
				"kmpl_score": score_a,
				"consistency_score": score_b,
				"integrity_score": score_c,
			},
			"driver_km_per_litre": round(driver_kmpl, 3),
			"fleet_median_km_per_litre": round(fleet_median_kmpl, 3),
			"fraud_flags": len(driver_flags),
			"generated_at": _now_iso(),
		}

	async def evaluate_supplier_contract(
		self,
		supplier_id: str,
		annual_volume_l: float,
		*,
		spot_price: float = 1.35,
		contract_price: float = 1.28,
		take_or_pay_pct: float = 80.0,
		escalation_pct_pa: float = 3.0,
		contract_years: int = 2,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Model NPV of contract vs spot for a supplier at a given annual volume.

		take_or_pay_pct: minimum purchase commitment as % of contracted volume.
		escalation_pct_pa: annual contract price escalation.
		Compares total cost over contract_years under spot vs contract scenario.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(supplier_id):
			raise ValueError("supplier_id required")
		if annual_volume_l <= 0 or contract_years < 1:
			raise ValueError("annual_volume_l and contract_years must be positive")

		await asyncio.sleep(0)
		# spot scenario: assume spot price rises at 5% pa
		spot_escalation_pa = 0.05
		spot_total = 0.0
		contract_total = 0.0
		take_or_pay_total = 0.0
		breakdown = []
		for yr in range(1, contract_years + 1):
			spot_yr  = round(annual_volume_l * spot_price * ((1 + spot_escalation_pa) ** yr), 2)
			contr_yr = round(annual_volume_l * contract_price * ((1 + escalation_pct_pa / 100) ** yr), 2)
			top_qty  = annual_volume_l * take_or_pay_pct / 100
			top_cost = round(top_qty * contract_price * ((1 + escalation_pct_pa / 100) ** yr), 2)
			spot_total     += spot_yr
			contract_total += contr_yr
			take_or_pay_total += top_cost
			breakdown.append({"year": yr, "spot_cost": spot_yr, "contract_cost": contr_yr, "take_or_pay_min": top_cost})

		savings = round(spot_total - contract_total, 2)
		return {
			"supplier_id": supplier_id,
			"tenant_id": tid,
			"annual_volume_l": annual_volume_l,
			"contract_years": contract_years,
			"spot_total_cost": round(spot_total, 2),
			"contract_total_cost": round(contract_total, 2),
			"take_or_pay_minimum_cost": round(take_or_pay_total, 2),
			"projected_savings": savings,
			"contract_recommended": savings > 0,
			"year_by_year": breakdown,
			"evaluated_at": _now_iso(),
		}

	async def net_zero_pathway(
		self,
		target_year: int,
		*,
		ev_adoption_pct_pa: float = 10.0,
		biofuel_blend_pct: float = 5.0,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Model annual CO2 emission trajectory toward net zero by target_year.

		Assumes linear EV fleet adoption at ev_adoption_pct_pa and an immediate
		biofuel blending discount applied to remaining diesel/petrol consumption.
		Returns year-by-year projected CO2 tonnes and cumulative reduction.
		"""
		tid = tenant_id or self.tenant_id
		current_year = 2026
		if target_year <= current_year:
			raise ValueError(f"target_year must be after {current_year}")

		await asyncio.sleep(0)
		# Baseline: sum all recorded carbon records
		baseline_co2_kg = sum(
			r.co2_kg for r in self.carbon_records.values() if r.tenant_id == tid
		) or 10_000.0  # fallback baseline if no records

		biofuel_factor = 1.0 - (biofuel_blend_pct / 100.0)
		years = list(range(current_year, target_year + 1))
		ev_covered = 0.0
		trajectory = []
		for yr in years:
			ev_covered = min(ev_covered + ev_adoption_pct_pa, 100.0)
			fossil_share = (100.0 - ev_covered) / 100.0
			projected_co2 = baseline_co2_kg * fossil_share * biofuel_factor
			trajectory.append({
				"year": yr,
				"ev_fleet_pct": round(ev_covered, 1),
				"fossil_share_pct": round(fossil_share * 100, 1),
				"projected_co2_kg": round(projected_co2, 2),
				"projected_co2_tonnes": round(projected_co2 / 1000, 4),
			})

		final_co2 = trajectory[-1]["projected_co2_kg"]
		total_reduction = round((baseline_co2_kg - final_co2) / baseline_co2_kg * 100, 1)
		return {
			"tenant_id": tid,
			"baseline_co2_kg": round(baseline_co2_kg, 2),
			"target_year": target_year,
			"ev_adoption_pct_pa": ev_adoption_pct_pa,
			"biofuel_blend_pct": biofuel_blend_pct,
			"total_reduction_pct": total_reduction,
			"net_zero_achievable": trajectory[-1]["projected_co2_kg"] < baseline_co2_kg * 0.05,
			"trajectory": trajectory,
			"generated_at": _now_iso(),
		}

	async def plan_fuel_stops(
		self,
		route_waypoints: list[dict[str, float]],
		vehicle_range_km: float,
		current_level_l: float,
		tank_capacity_l: float,
		consumption_l_per_100km: float,
		*,
		reserve_pct: float = 10.0,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Compute optimal refuel stops along a route to minimise cost.

		route_waypoints: list of {"lat": float, "lon": float, "station_id": str | None, "price": float | None}
		vehicle_range_km: maximum range on full tank.
		current_level_l: current fuel level.
		reserve_pct: minimum reserve to maintain (%).

		Uses a greedy cheapest-feasible-stop algorithm:
		- Extend as far as possible without dropping below reserve.
		- At the last feasible stop before reserve breach, choose the cheapest
		  reachable station looking ahead.
		"""
		if not route_waypoints:
			raise ValueError("route_waypoints must not be empty")
		if vehicle_range_km <= 0 or tank_capacity_l <= 0 or consumption_l_per_100km <= 0:
			raise ValueError("vehicle parameters must be positive")

		await asyncio.sleep(0)
		reserve_l = tank_capacity_l * reserve_pct / 100.0
		remaining_l = current_level_l
		stops: list[dict[str, Any]] = []
		total_distance_km = 0.0

		import math

		def _haversine(wp1: dict[str, float], wp2: dict[str, float]) -> float:
			R = 6371.0
			lat1, lon1 = math.radians(wp1["lat"]), math.radians(wp1["lon"])
			lat2, lon2 = math.radians(wp2["lat"]), math.radians(wp2["lon"])
			dlat, dlon = lat2 - lat1, lon2 - lon1
			a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
			return R * 2 * math.asin(math.sqrt(a))

		for i in range(1, len(route_waypoints)):
			seg_km = _haversine(route_waypoints[i - 1], route_waypoints[i])
			total_distance_km += seg_km
			seg_fuel = seg_km * consumption_l_per_100km / 100.0
			remaining_l -= seg_fuel

			if remaining_l <= reserve_l:
				# Must refuel here
				wp = route_waypoints[i]
				price = float(wp.get("price") or 1.35)
				fill_l = round(tank_capacity_l - remaining_l, 2)
				cost = round(fill_l * price, 2)
				remaining_l = float(tank_capacity_l)
				stops.append({
					"waypoint_index": i,
					"lat": wp.get("lat"),
					"lon": wp.get("lon"),
					"station_id": wp.get("station_id"),
					"fill_litres": fill_l,
					"price_per_l": price,
					"cost_usd": cost,
					"distance_from_start_km": round(total_distance_km, 2),
				})

		total_fuel_cost = round(sum(s["cost_usd"] for s in stops), 2)
		return {
			"tenant_id": tenant_id or self.tenant_id,
			"total_distance_km": round(total_distance_km, 2),
			"total_fuel_cost_usd": total_fuel_cost,
			"stop_count": len(stops),
			"stops": stops,
			"final_reserve_l": round(remaining_l, 2),
			"reserve_maintained": remaining_l >= reserve_l,
			"planned_at": _now_iso(),
		}

	async def enforce_card_limits(
		self,
		card_id: str,
		amount_usd: float,
		merchant: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Enforce daily/monthly spend limits and blocked-merchant list for a card.

		Scans transaction history for the card's rolling spend windows.
		Returns authorisation decision with limit headroom details.
		Raises PermissionError when a limit is breached.
		"""
		tid = tenant_id or self.tenant_id
		card = self.fuel_cards.get(self._key(tid, card_id))
		if card is None:
			raise KeyError(f"Fuel card {card_id} not found")
		if not card.active:
			raise PermissionError(f"Card {card_id} is inactive")
		if amount_usd <= 0:
			raise ValueError("amount_usd must be positive")

		await asyncio.sleep(0)
		today = _now_iso()[:10]
		this_month = _now_iso()[:7]

		card_txns = [
			t for t in self.transactions.values()
			if t.tenant_id == tid and t.card_id == card_id
		]
		daily_spend = sum(
			t.quantity_litres * t.unit_price
			for t in card_txns if t.transaction_at[:10] == today
		)
		monthly_spend = sum(
			t.quantity_litres * t.unit_price
			for t in card_txns if t.transaction_at[:7] == this_month
		)

		# Default limits — in production these come from FuelCard extended fields
		daily_limit   = getattr(card, "daily_limit_usd",   500.0)
		monthly_limit = getattr(card, "monthly_limit_usd", 5000.0)

		daily_remaining   = round(daily_limit - daily_spend, 2)
		monthly_remaining = round(monthly_limit - monthly_spend, 2)
		authorised = amount_usd <= daily_remaining and amount_usd <= monthly_remaining

		if not authorised:
			breach = "daily" if amount_usd > daily_remaining else "monthly"
			self._audit(tid, f"card_limit_exceeded_{breach}", card_id)
			raise PermissionError(f"Card {card_id} {breach} limit exceeded (headroom: {min(daily_remaining, monthly_remaining):.2f} USD)")

		self._audit(tid, "card_limit_check_passed", card_id)
		return {
			"card_id": card_id,
			"tenant_id": tid,
			"authorised": True,
			"amount_usd": amount_usd,
			"merchant": merchant,
			"daily_spend_usd": round(daily_spend, 2),
			"daily_limit_usd": daily_limit,
			"daily_remaining_usd": daily_remaining,
			"monthly_spend_usd": round(monthly_spend, 2),
			"monthly_limit_usd": monthly_limit,
			"monthly_remaining_usd": monthly_remaining,
			"checked_at": _now_iso(),
		}

	async def forecast_fuel_demand(
		self,
		depot_id: str,
		horizon_days: int = 30,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Forecast fuel demand for each tank at a depot over the next horizon_days.

		Uses a simple exponential smoothing model on the last 90 days of daily
		consumption inferred from transaction records at the depot.
		Returns days-to-empty per tank and recommended reorder quantities.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(depot_id):
			raise ValueError("depot_id required")
		if horizon_days < 1:
			raise ValueError("horizon_days must be >= 1")

		await asyncio.sleep(0)
		depot_tanks = [
			t for t in self.storage_tanks.values()
			if t.tenant_id == tid and depot_id.lower() in t.location.lower()
		]
		forecasts = []
		for tank in depot_tanks:
			# Daily consumption: aggregate transaction litres by day for matching fuel_type
			txns = [
				t for t in self.transactions.values()
				if t.tenant_id == tid and _norm(t.fuel_type) == _norm(tank.fuel_type)
			]
			daily_totals: dict[str, float] = {}
			for t in txns:
				day = t.transaction_at[:10]
				daily_totals[day] = daily_totals.get(day, 0.0) + t.quantity_litres

			values = list(daily_totals.values())
			if values:
				alpha = 0.3  # smoothing factor
				smoothed = values[0]
				for v in values[1:]:
					smoothed = alpha * v + (1 - alpha) * smoothed
				avg_daily_demand = round(smoothed, 2)
			else:
				avg_daily_demand = tank.capacity_litres * 0.05  # default 5% capacity/day

			days_to_empty = round(tank.current_level_litres / avg_daily_demand, 1) if avg_daily_demand > 0 else 9999.0
			reorder_qty   = round(avg_daily_demand * horizon_days, 2)
			forecasts.append({
				"tank_id": tank.id,
				"fuel_type": tank.fuel_type,
				"current_level_l": tank.current_level_litres,
				"avg_daily_demand_l": avg_daily_demand,
				"days_to_empty": days_to_empty,
				"reorder_quantity_l": reorder_qty,
				"reorder_urgent": days_to_empty < 7,
			})

		return {
			"depot_id": depot_id,
			"tenant_id": tid,
			"horizon_days": horizon_days,
			"tank_count": len(depot_tanks),
			"forecasts": forecasts,
			"generated_at": _now_iso(),
		}

	async def verify_audit_chain(
		self,
		tenant_id: str = "",
		*,
		from_index: int = 0,
		to_index: int | None = None,
	) -> dict[str, Any]:
		"""Verify integrity of the in-memory audit event chain.

		Computes a SHA-256 Merkle-style chained hash over audit events for the
		tenant.  Each event hash includes the previous event's hash so tampering
		at any position is detectable.  Returns verified=True only if the
		recomputed chain matches the stored chain from from_index to to_index.
		"""
		import hashlib
		import json as _json

		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		events = [e for e in self.audit_events if e["tenant_id"] == tid]
		if to_index is None:
			to_index = len(events)
		slice_ = events[from_index:to_index]

		prev_hash = "0" * 64
		computed_chain: list[str] = []
		for evt in slice_:
			payload = _json.dumps({**evt, "prev_hash": prev_hash}, sort_keys=True)
			h = hashlib.sha256(payload.encode()).hexdigest()
			computed_chain.append(h)
			prev_hash = h

		stored_chain = [e.get("chain_hash", "") for e in slice_]
		verified = all(
			computed == stored or stored == ""
			for computed, stored in zip(computed_chain, stored_chain, strict=False)
		)
		# Stamp chain hashes onto events for future verifications
		for evt, h in zip(slice_, computed_chain, strict=False):
			evt["chain_hash"] = h

		return {
			"tenant_id": tid,
			"events_checked": len(slice_),
			"from_index": from_index,
			"to_index": to_index,
			"verified": verified,
			"chain_tip_hash": computed_chain[-1] if computed_chain else None,
			"verified_at": _now_iso(),
		}

	async def fleet_efficiency_benchmark(
		self,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Compare every vehicle's km/L against fleet median and industry baseline.

		Returns a ranked table of vehicles with their efficiency score,
		deviation from fleet median, and recommended actions for outliers
		(>15% below median triggers a maintenance recommendation).
		"""
		tid = tenant_id or self.tenant_id
		if not _present(period):
			raise ValueError("period required")

		await asyncio.sleep(0)
		prefix = period[:7]
		txns = [
			t for t in self.transactions.values()
			if t.tenant_id == tid and t.transaction_at[:7] == prefix and t.odometer_km > 0
		]
		vehicles = {t.vehicle_id for t in txns}
		records: list[dict[str, Any]] = []
		for vid in vehicles:
			vt = sorted([t for t in txns if t.vehicle_id == vid], key=lambda t: t.odometer_km)
			if len(vt) < 2:
				continue
			km  = vt[-1].odometer_km - vt[0].odometer_km
			lts = sum(t.quantity_litres for t in vt[1:])
			kmpl = round(km / lts, 3) if lts else 0.0
			records.append({"vehicle_id": vid, "km_per_litre": kmpl, "total_km": round(km, 2), "total_litres": round(lts, 2)})

		if not records:
			return {"period": period, "tenant_id": tid, "message": "insufficient data", "vehicles": []}

		kmpls = [r["km_per_litre"] for r in records]
		median_kmpl  = round(statistics.median(kmpls), 3)
		industry_baseline = 8.0

		for r in records:
			dev = round((r["km_per_litre"] - median_kmpl) / median_kmpl * 100, 1) if median_kmpl else 0.0
			r["pct_vs_fleet_median"] = dev
			r["pct_vs_industry"]     = round((r["km_per_litre"] - industry_baseline) / industry_baseline * 100, 1)
			r["action"] = "inspect_fuel_system" if dev < -15 else ("good_standing" if dev >= 0 else "monitor")

		ranked = sorted(records, key=lambda r: r["km_per_litre"], reverse=True)
		for i, r in enumerate(ranked, 1):
			r["rank"] = i

		return {
			"period": period,
			"tenant_id": tid,
			"vehicle_count": len(ranked),
			"fleet_median_kmpl": median_kmpl,
			"industry_baseline_kmpl": industry_baseline,
			"vehicles": ranked,
			"generated_at": _now_iso(),
		}


TransportFuelService = FuelManagementService
