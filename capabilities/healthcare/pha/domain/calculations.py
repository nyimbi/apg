"""Financial and clinical calculations for Pharmacy Management.

All functions are pure (no side effects), type-safe, and handle edge cases.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any


# ── Expiry / Days-Remaining ────────────────────────────────────────────────────

def days_until_expiry(expiry_date: datetime, reference: datetime | None = None) -> int:
	"""Return whole days until expiry_date; negative if already expired."""
	ref = reference or datetime.utcnow()
	delta = expiry_date - ref
	return delta.days


def expiry_alert_level(days_remaining: int) -> str:
	"""Classify expiry urgency.

	Returns one of: "expired", "critical" (<7d), "warning" (<30d),
	"notice" (<90d), "ok".
	"""
	if days_remaining <= 0:
		return "expired"
	if days_remaining < 7:
		return "critical"
	if days_remaining < 30:
		return "warning"
	if days_remaining < 90:
		return "notice"
	return "ok"


def inventory_status_from_expiry(days_remaining: int) -> str:
	"""Map days-remaining to an InventoryStatus string."""
	if days_remaining <= 0:
		return "expired"
	if days_remaining <= 30:
		return "low_stock"
	return "in_stock"


# ── Cold Chain ─────────────────────────────────────────────────────────────────

def cold_chain_status(
	recorded_temp: float,
	min_acceptable: float,
	max_acceptable: float,
	excursion_minutes: int | None = None,
) -> str:
	"""Classify a temperature reading.

	Returns: "compliant", "excursion", or "critical".
	Critical = out-of-range AND excursion > 60 minutes.
	"""
	in_range = min_acceptable <= recorded_temp <= max_acceptable
	if in_range:
		return "compliant"
	if excursion_minutes is not None and excursion_minutes > 60:
		return "critical"
	return "excursion"


def temperature_deviation(
	recorded_temp: float,
	min_acceptable: float,
	max_acceptable: float,
) -> float:
	"""Return signed deviation from nearest bound (0 if in range)."""
	if recorded_temp < min_acceptable:
		return recorded_temp - min_acceptable  # negative
	if recorded_temp > max_acceptable:
		return recorded_temp - max_acceptable  # positive
	return 0.0


# ── Narcotics Register Balance ─────────────────────────────────────────────────

def narcotics_balance_after(
	balance_before: float,
	action: str,
	quantity: float,
	waste_amount: float = 0.0,
) -> float:
	"""Compute expected narcotics register balance after an action.

	Actions that increase balance: receive, transfer_in (not modelled separately).
	Actions that decrease balance: dispense, waste, destroy, transfer.
	Returns the new balance (may be negative — caller must validate).
	"""
	additive = {"receive"}
	subtractive = {"dispense", "waste", "destroy", "transfer"}

	effective_qty = quantity + waste_amount if action == "waste" else quantity

	if action in additive:
		return round(balance_before + effective_qty, 6)
	if action in subtractive:
		return round(balance_before - effective_qty, 6)
	# count / audit — no change
	return round(balance_before, 6)


def narcotics_discrepancy(
	expected_balance: float,
	physical_count: float,
) -> float:
	"""Return signed discrepancy: physical_count - expected_balance."""
	return round(physical_count - expected_balance, 6)


# ── Inventory Valuation ────────────────────────────────────────────────────────

def inventory_value(quantity: float, unit_price: float) -> float:
	"""Compute line-item inventory value."""
	if quantity < 0 or unit_price < 0:
		return 0.0
	return round(quantity * unit_price, 2)


def total_inventory_value(items: list[dict[str, Any]]) -> float:
	"""Sum inventory values across a list of item dicts with
	keys: quantity_on_hand, purchase_price (optional).
	"""
	total = 0.0
	for item in items:
		qty = item.get("quantity_on_hand", 0.0) or 0.0
		price = item.get("purchase_price") or 0.0
		total += inventory_value(qty, price)
	return round(total, 2)


# ── Reorder Logic ──────────────────────────────────────────────────────────────

def needs_reorder(quantity_on_hand: float, reorder_point: float) -> bool:
	"""True when stock is at or below the reorder point."""
	return quantity_on_hand <= reorder_point


def economic_order_quantity(
	annual_demand: float,
	ordering_cost: float,
	holding_cost_per_unit: float,
) -> float:
	"""Classic Wilson EOQ formula.

	Returns 0.0 if inputs are invalid / zero.
	"""
	if annual_demand <= 0 or ordering_cost <= 0 or holding_cost_per_unit <= 0:
		return 0.0
	return round(math.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit), 2)


def days_of_supply(quantity_on_hand: float, daily_dispensing_rate: float) -> float:
	"""Estimate days of supply remaining.

	Returns infinity if daily rate is zero (no consumption).
	"""
	if daily_dispensing_rate <= 0:
		return float("inf")
	return round(quantity_on_hand / daily_dispensing_rate, 1)


# ── Counselling Completion ─────────────────────────────────────────────────────

_COUNSELLING_FIELDS = [
	"indication_explained",
	"dosage_explained",
	"administration_explained",
	"side_effects_explained",
	"interactions_explained",
	"storage_explained",
	"missed_dose_explained",
	"patient_questions_addressed",
	"patient_understood",
]


def counselling_completion_score(checklist: dict[str, bool]) -> float:
	"""Return fraction of counselling items completed (0.0–1.0)."""
	if not _COUNSELLING_FIELDS:
		return 0.0
	completed = sum(1 for field in _COUNSELLING_FIELDS if checklist.get(field, False))
	return round(completed / len(_COUNSELLING_FIELDS), 4)


# ── Drug Substitution ──────────────────────────────────────────────────────────

def find_generic_substitute(
	brand_drug_id: str,
	formulary: list[dict[str, Any]],
) -> dict[str, Any] | None:
	"""Find a preferred generic substitute for a brand drug.

	Matches on generic_name and preferred formulary status.
	Returns the first matching generic, or None.
	"""
	brand = next((d for d in formulary if d.get("id") == brand_drug_id), None)
	if brand is None:
		return None
	generic_name = brand.get("generic_name", "").lower()
	for drug in formulary:
		if (
			drug.get("drug_type") == "generic"
			and drug.get("generic_name", "").lower() == generic_name
			and drug.get("formulary_status") == "preferred"
			and drug.get("id") != brand_drug_id
		):
			return drug
	return None


# ── Verification Timing ────────────────────────────────────────────────────────

def verification_turnaround_minutes(
	created_at: datetime,
	verified_at: datetime | None,
) -> float | None:
	"""Compute pharmacist verification turnaround in minutes."""
	if verified_at is None:
		return None
	delta = verified_at - created_at
	return round(delta.total_seconds() / 60, 2)


def average_verification_time(
	orders: list[dict[str, Any]],
) -> float | None:
	"""Mean pharmacist verification turnaround across a list of order dicts."""
	times: list[float] = []
	for o in orders:
		created = o.get("created_at")
		verified = o.get("verified_at")
		if created and verified:
			t = verification_turnaround_minutes(created, verified)
			if t is not None:
				times.append(t)
	if not times:
		return None
	return round(sum(times) / len(times), 2)


# ── Prescription Expiry ────────────────────────────────────────────────────────

def prescription_expiry_date(
	prescribed_at: datetime,
	controlled: bool = False,
	days_valid: int | None = None,
) -> datetime:
	"""Compute prescription expiry.

	Non-controlled: 1 year by default.
	Controlled (Schedule II): 6 months in most jurisdictions.
	Caller may override with explicit days_valid.
	"""
	from datetime import timedelta

	if days_valid is not None:
		return prescribed_at + timedelta(days=days_valid)
	if controlled:
		return prescribed_at + timedelta(days=180)
	return prescribed_at + timedelta(days=365)
