"""Deterministic carbon-accounting helpers for APG ESGC."""

from __future__ import annotations


class CarbonEngine:
	"""Pure carbon calculations used by the ESGC service."""

	def co2e_tonnes(self, quantity: float, co2e_per_unit: float) -> float:
		return round(max(quantity, 0.0) * max(co2e_per_unit, 0.0), 6)

	def inventory_total(self, activities: list[dict[str, float]]) -> float:
		return round(sum(float(item.get("co2e_tonnes", 0.0)) for item in activities), 6)

	def anomaly_detected(self, quantity: float, expected_max_quantity: float | None) -> bool:
		if expected_max_quantity is None:
			return False
		return quantity > expected_max_quantity

	def reduction_progress_percent(self, baseline: float, current: float, target_reduction_percent: float) -> float:
		if baseline <= 0 or target_reduction_percent <= 0:
			return 0.0
		achieved_reduction = max(baseline - current, 0.0)
		target_reduction = baseline * (target_reduction_percent / 100.0)
		if target_reduction <= 0:
			return 0.0
		return round(min((achieved_reduction / target_reduction) * 100.0, 100.0), 2)

	def target_status(self, progress_percent: float) -> str:
		if progress_percent >= 100:
			return "achieved"
		if progress_percent >= 75:
			return "on_track"
		if progress_percent > 0:
			return "behind"
		return "not_started"
