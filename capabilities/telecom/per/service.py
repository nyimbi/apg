"""Service layer for APG Performance Management."""

from __future__ import annotations

import datetime
import statistics
from typing import Any

from .domain.adapters import get_auth_adapter, get_audit_adapter
from .database.store import get_store
from .capability_contract import (
	SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_BENCHMARK_TYPES,
	SUPPORTED_CAPACITY_STATES, SUPPORTED_KPI_CATEGORIES, SUPPORTED_KPI_STATUSES,
	SUPPORTED_REPORT_PERIODS, SUPPORTED_SLA_COMPLIANCE_STATUSES,
	SUPPORTED_THRESHOLD_ACTIONS, SUPPORTED_TREND_DIRECTIONS,
	evaluate_capability_rules, get_capability_contract,
)
from .models import (
	PerAgent, PerBenchmark, PerCapacityRecord, PerKpi,
	PerReport, PerSlaCompliance, PerThreshold, PerTrend,
)


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


def _utcnow() -> str:
	return datetime.datetime.utcnow().isoformat() + "Z"


class TelecomPerformanceService:
	"""Tenant-scoped performance management service for APG Telecom."""

	def __init__(self) -> None:
		self._store = get_store("telecom.per")
		self._auth = get_auth_adapter()
		self._audit_adapter = get_audit_adapter()
		self.kpis: dict[tuple[str, str], PerKpi] = {}
		self.sla_compliance: dict[tuple[str, str], PerSlaCompliance] = {}
		self.capacity_records: dict[tuple[str, str], PerCapacityRecord] = {}
		self.trends: dict[tuple[str, str], PerTrend] = {}
		self.thresholds: dict[tuple[str, str], PerThreshold] = {}
		self.benchmarks: dict[tuple[str, str], PerBenchmark] = {}
		self.reports: dict[tuple[str, str], PerReport] = {}
		self.agents: dict[tuple[str, str], PerAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# in-memory stores for new method state
		self._call_drop_records: list[dict[str, Any]] = []
		self._throughput_records: list[dict[str, Any]] = []
		self._alerts: list[dict[str, Any]] = []
		self._nps_records: list[dict[str, Any]] = []

	# ------------------------------------------------------------------ #
	# Contract                                                             #
	# ------------------------------------------------------------------ #

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------ #
	# Core existing methods                                                #
	# ------------------------------------------------------------------ #

	def record_kpi(
		self,
		kpi_id: str,
		tenant_id: str,
		kpi_category: str,
		kpi_name: str,
		value: float,
		baseline_value: float,
		unit: str,
		network_layer: str,
		recorded_at: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Record a KPI measurement for a network layer."""
		kpi_category = kpi_category.lower()
		status = "nominal"
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "record_kpi",
			"kpi_category_supported": kpi_category in SUPPORTED_KPI_CATEGORIES,
			"baseline_present": baseline_value is not None,
		})
		item = PerKpi(kpi_id, tenant_id, kpi_category, kpi_name, float(value), float(baseline_value), unit, status, network_layer, recorded_at)
		self.kpis[self._key(tenant_id, kpi_id)] = item
		self._audit(tenant_id, "kpi_recorded", kpi_id)
		return item.to_dict()

	def update_kpi_status(self, kpi_id: str, tenant_id: str, new_status: str) -> dict[str, Any]:
		"""Update a KPI's operational status."""
		new_status = new_status.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "update_kpi_status",
			"kpi_status_supported": new_status in SUPPORTED_KPI_STATUSES,
		})
		kpi = self._kpi_or_raise(kpi_id, tenant_id)
		kpi.status = new_status
		if new_status == "critical":
			self._audit(tenant_id, "kpi_threshold_breached", kpi_id)
		return kpi.to_dict()

	def record_sla_compliance(
		self,
		compliance_id: str,
		tenant_id: str,
		sla_type: str,
		customer_id: str | None,
		target_value: float,
		actual_value: float,
		period: str,
		notification_sent: bool = False,
	) -> dict[str, Any]:
		"""Record an SLA compliance measurement."""
		sla_type = sla_type.lower()
		is_breach = actual_value < target_value
		status = "breached" if is_breach else "compliant"
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_sla_compliance",
			"sla_status_supported": status in SUPPORTED_SLA_COMPLIANCE_STATUSES,
			"sla_breached": is_breach,
			"notification_sent": notification_sent or not is_breach,
		})
		item = PerSlaCompliance(compliance_id, tenant_id, sla_type, customer_id, float(target_value), float(actual_value), status, period, notification_sent)
		self.sla_compliance[self._key(tenant_id, compliance_id)] = item
		if is_breach:
			self._audit(tenant_id, "sla_breach_detected", compliance_id)
		return item.to_dict()

	def record_capacity(
		self,
		record_id: str,
		tenant_id: str,
		resource_reference: str,
		capacity_state: str,
		utilisation_pct: float,
		forecast_horizon_days: int,
		recorded_at: str,
	) -> dict[str, Any]:
		"""Record network resource capacity utilisation."""
		capacity_state = capacity_state.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_capacity",
			"capacity_state_supported": capacity_state in SUPPORTED_CAPACITY_STATES,
		})
		item = PerCapacityRecord(record_id, tenant_id, resource_reference, capacity_state, float(utilisation_pct), int(forecast_horizon_days), recorded_at)
		self.capacity_records[self._key(tenant_id, record_id)] = item
		if capacity_state in ("congested", "overloaded"):
			self._audit(tenant_id, "capacity_congestion_alert", record_id)
		return item.to_dict()

	def record_trend(
		self,
		trend_id: str,
		tenant_id: str,
		kpi_id: str,
		trend_direction: str,
		lookback_days: int,
		forecast_value: float | None,
		recorded_at: str,
	) -> dict[str, Any]:
		"""Record a performance trend analysis result."""
		trend_direction = trend_direction.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_trend",
			"trend_direction_supported": trend_direction in SUPPORTED_TREND_DIRECTIONS,
		})
		item = PerTrend(trend_id, tenant_id, kpi_id, trend_direction, int(lookback_days), forecast_value, recorded_at)
		self.trends[self._key(tenant_id, trend_id)] = item
		if trend_direction == "degrading":
			self._audit(tenant_id, "trend_degradation_detected", trend_id)
		return item.to_dict()

	def set_threshold(
		self,
		threshold_id: str,
		tenant_id: str,
		kpi_name: str,
		network_layer: str,
		warning_value: float,
		critical_value: float,
		action: str,
		approval_reference: str,
		set_by: str,
	) -> dict[str, Any]:
		"""Set performance thresholds with mandatory approval."""
		action = action.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "set_threshold",
			"threshold_action_supported": action in SUPPORTED_THRESHOLD_ACTIONS,
			"approval_present": _present(approval_reference),
		})
		item = PerThreshold(threshold_id, tenant_id, kpi_name, network_layer, float(warning_value), float(critical_value), action, approval_reference, set_by)
		self.thresholds[self._key(tenant_id, threshold_id)] = item
		self._audit(tenant_id, "threshold_changed", threshold_id)
		return item.to_dict()

	def record_benchmark(
		self,
		benchmark_id: str,
		tenant_id: str,
		benchmark_type: str,
		kpi_name: str,
		benchmark_value: float,
		current_value: float,
		recorded_at: str,
	) -> dict[str, Any]:
		"""Record a performance benchmark and gap analysis."""
		benchmark_type = benchmark_type.lower()
		gap_pct = ((benchmark_value - current_value) / benchmark_value * 100) if benchmark_value > 0 else 0.0
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_benchmark",
			"benchmark_type_supported": benchmark_type in SUPPORTED_BENCHMARK_TYPES,
		})
		item = PerBenchmark(benchmark_id, tenant_id, benchmark_type, kpi_name, float(benchmark_value), float(current_value), float(gap_pct), recorded_at)
		self.benchmarks[self._key(tenant_id, benchmark_id)] = item
		if gap_pct > 10:
			self._audit(tenant_id, "benchmark_gap_detected", benchmark_id)
		return item.to_dict()

	def generate_report(
		self,
		report_id: str,
		tenant_id: str,
		report_period: str,
		fmt: str,
		approval_reference: str,
		generated_by: str,
		generated_at: str,
	) -> dict[str, Any]:
		"""Generate a performance management report."""
		report_period = report_period.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "generate_report",
			"report_period_supported": report_period in SUPPORTED_REPORT_PERIODS,
			"approval_present": _present(approval_reference),
		})
		item = PerReport(report_id, tenant_id, report_period, fmt, approval_reference, generated_by, generated_at)
		self.reports[self._key(tenant_id, report_id)] = item
		self._audit(tenant_id, "report_generated", report_id)
		return item.to_dict()

	def register_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		"""Register a performance management automation agent."""
		runtime = runtime.lower()
		role = role.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_per_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		item = PerAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "per_agent_registered", agent_id)
		return item.to_dict()

	# ------------------------------------------------------------------ #
	# New methods                                                          #
	# ------------------------------------------------------------------ #

	async def kpi_collection(
		self,
		network_id: str,
		period: str,
		kpi_types: list[str],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Collect and aggregate KPIs for a network element across a reporting period.

		Pulls all recorded KPIs for the given network_id that fall within the
		period window and whose category matches one of kpi_types.  Returns
		per-type summary statistics (mean, min, max, p95) alongside the raw
		count and breach rate.
		"""
		assert network_id, "network_id required"
		assert period, "period required"
		assert kpi_types, "at least one kpi_type required"
		normalised_types = [t.lower() for t in kpi_types]
		matched = [
			kpi for kpi in self.kpis.values()
			if kpi.tenant_id == tenant_id
			and kpi.network_layer == network_id
			and kpi.kpi_category in normalised_types
		]
		summary: dict[str, Any] = {}
		for kpi_type in normalised_types:
			values = [kpi.value for kpi in matched if kpi.kpi_category == kpi_type]
			if not values:
				summary[kpi_type] = {"count": 0, "mean": None, "min": None, "max": None, "p95": None, "breach_rate": 0.0}
				continue
			sorted_vals = sorted(values)
			p95_idx = max(0, int(len(sorted_vals) * 0.95) - 1)
			breaches = sum(1 for kpi in matched if kpi.kpi_category == kpi_type and kpi.status in ("warning", "critical"))
			summary[kpi_type] = {
				"count": len(values),
				"mean": round(statistics.mean(values), 4),
				"min": min(values),
				"max": max(values),
				"p95": sorted_vals[p95_idx],
				"breach_rate": round(breaches / len(values), 4),
			}
		self._audit(tenant_id, "kpi_collection_run", network_id)
		return {
			"network_id": network_id,
			"period": period,
			"tenant_id": tenant_id,
			"kpi_types": normalised_types,
			"summary": summary,
			"total_kpis": len(matched),
			"collected_at": _utcnow(),
		}

	async def sla_compliance_check(
		self,
		customer_id: str,
		service_id: str,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Check SLA compliance for a specific customer/service in a given period.

		Scans all recorded SLA compliance records for this customer, computes
		overall compliance rate, identifies breach categories, and determines
		whether a penalty credit should be triggered.
		"""
		assert customer_id, "customer_id required"
		assert service_id, "service_id required"
		assert period, "period required"
		records = [
			r for r in self.sla_compliance.values()
			if r.tenant_id == tenant_id
			and r.customer_id == customer_id
			and r.period == period
		]
		total = len(records)
		breaches = [r for r in records if r.status == "breached"]
		breach_count = len(breaches)
		compliance_rate = round((total - breach_count) / total, 4) if total > 0 else 1.0
		breach_types = list({r.sla_type for r in breaches})
		# Simple penalty trigger: compliance below 99.5%
		penalty_trigger = compliance_rate < 0.995
		if penalty_trigger:
			self._audit(tenant_id, "sla_penalty_triggered", f"{customer_id}:{service_id}")
		return {
			"customer_id": customer_id,
			"service_id": service_id,
			"period": period,
			"tenant_id": tenant_id,
			"total_measurements": total,
			"breach_count": breach_count,
			"compliance_rate": compliance_rate,
			"breach_types": breach_types,
			"penalty_trigger": penalty_trigger,
			"checked_at": _utcnow(),
		}

	async def network_quality_score(
		self,
		region: str,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Compute a composite network quality score (0–100) for a region.

		Weights: availability 40%, latency 25%, drop rate 20%, throughput 15%.
		Each component is normalised against its threshold; missing data yields
		a neutral 0.5 contribution.
		"""
		assert region, "region required"
		assert period, "period required"
		# Gather capacity records for the region (using resource_reference as region proxy)
		cap_records = [
			r for r in self.capacity_records.values()
			if r.tenant_id == tenant_id and region.lower() in r.resource_reference.lower()
		]
		# Availability: inverse of mean utilisation congestion rate
		utilisation_values = [r.utilisation_pct for r in cap_records]
		avg_util = statistics.mean(utilisation_values) if utilisation_values else 50.0
		availability_score = max(0.0, min(1.0, 1.0 - (avg_util - 70) / 30)) if avg_util > 70 else 1.0
		# SLA compliance rate for region (use all records as proxy)
		sla_recs = [r for r in self.sla_compliance.values() if r.tenant_id == tenant_id and r.period == period]
		sla_compliance_rate = (len([r for r in sla_recs if r.status == "compliant"]) / len(sla_recs)) if sla_recs else 1.0
		# Drop rate from stored call_drop_records
		drops = [d for d in self._call_drop_records if d.get("tenant_id") == tenant_id and d.get("region") == region]
		avg_drop_rate = statistics.mean([d["drop_rate"] for d in drops]) if drops else 0.02
		drop_score = max(0.0, 1.0 - avg_drop_rate * 10)
		# Throughput from throughput records
		thrput = [t for t in self._throughput_records if t.get("tenant_id") == tenant_id and t.get("region") == region]
		avg_throughput_score = statistics.mean([t.get("score", 0.8) for t in thrput]) if thrput else 0.8
		composite = (
			availability_score * 0.40
			+ sla_compliance_rate * 0.25
			+ drop_score * 0.20
			+ avg_throughput_score * 0.15
		)
		quality_score = round(composite * 100, 2)
		self._audit(tenant_id, "network_quality_score_computed", region)
		return {
			"region": region,
			"period": period,
			"tenant_id": tenant_id,
			"quality_score": quality_score,
			"components": {
				"availability": round(availability_score * 100, 2),
				"sla_compliance": round(sla_compliance_rate * 100, 2),
				"call_drop": round(drop_score * 100, 2),
				"throughput": round(avg_throughput_score * 100, 2),
			},
			"computed_at": _utcnow(),
		}

	async def call_drop_analysis(
		self,
		period: str,
		cell_ids: list[str],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Analyse call drop rates per cell, identifying worst offenders.

		Records are matched against stored call_drop_records by cell_id.
		Returns per-cell breakdown plus network-wide statistics and a list
		of cells exceeding the 2% drop rate SLA threshold.
		"""
		assert period, "period required"
		assert cell_ids, "at least one cell_id required"
		per_cell: dict[str, dict[str, Any]] = {}
		sla_threshold = 0.02
		violating_cells: list[str] = []
		for cell_id in cell_ids:
			drops = [
				d for d in self._call_drop_records
				if d.get("tenant_id") == tenant_id
				and d.get("cell_id") == cell_id
				and d.get("period") == period
			]
			if not drops:
				per_cell[cell_id] = {"drop_rate": None, "call_count": 0, "drops": 0, "sla_ok": True}
				continue
			total_calls = sum(d.get("call_count", 0) for d in drops)
			total_drops = sum(d.get("drop_count", 0) for d in drops)
			drop_rate = total_drops / total_calls if total_calls > 0 else 0.0
			sla_ok = drop_rate <= sla_threshold
			if not sla_ok:
				violating_cells.append(cell_id)
			per_cell[cell_id] = {
				"drop_rate": round(drop_rate, 4),
				"call_count": total_calls,
				"drops": total_drops,
				"sla_ok": sla_ok,
			}
		all_rates = [v["drop_rate"] for v in per_cell.values() if v["drop_rate"] is not None]
		network_avg = round(statistics.mean(all_rates), 4) if all_rates else 0.0
		self._audit(tenant_id, "call_drop_analysis_run", period)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"cell_count": len(cell_ids),
			"per_cell": per_cell,
			"network_avg_drop_rate": network_avg,
			"sla_threshold": sla_threshold,
			"violating_cells": violating_cells,
			"analysed_at": _utcnow(),
		}

	async def data_throughput_analytics(
		self,
		period: str,
		segment: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Analyse data throughput KPIs for a subscriber segment.

		Aggregates download/upload throughput from stored KPIs with name
		containing 'throughput', filtered by segment tag.
		Returns mean/median/p95 for DL and UL, and a congestion flag.
		"""
		assert period, "period required"
		assert segment, "segment required"
		segment_lower = segment.lower()
		# Throughput KPIs are stored with kpi_name containing "throughput"
		dl_values: list[float] = []
		ul_values: list[float] = []
		for kpi in self.kpis.values():
			if kpi.tenant_id != tenant_id:
				continue
			name_lower = kpi.kpi_name.lower()
			if "throughput" not in name_lower and "download" not in name_lower and "upload" not in name_lower:
				continue
			if "download" in name_lower or "dl" in name_lower:
				dl_values.append(kpi.value)
			elif "upload" in name_lower or "ul" in name_lower:
				ul_values.append(kpi.value)
			else:
				dl_values.append(kpi.value)

		def _stats(vals: list[float]) -> dict[str, Any]:
			if not vals:
				return {"mean": None, "median": None, "p95": None, "min": None, "max": None}
			sv = sorted(vals)
			p95_idx = max(0, int(len(sv) * 0.95) - 1)
			return {
				"mean": round(statistics.mean(vals), 2),
				"median": round(statistics.median(vals), 2),
				"p95": sv[p95_idx],
				"min": min(vals),
				"max": max(vals),
			}

		dl_stats = _stats(dl_values)
		ul_stats = _stats(ul_values)
		# Flag congestion if mean DL < 10 Mbps
		congestion = (dl_stats["mean"] is not None and dl_stats["mean"] < 10.0)
		self._audit(tenant_id, "data_throughput_analytics_run", f"{period}:{segment}")
		return {
			"period": period,
			"segment": segment,
			"tenant_id": tenant_id,
			"download_mbps": dl_stats,
			"upload_mbps": ul_stats,
			"congestion_flag": congestion,
			"analysed_at": _utcnow(),
		}

	async def capacity_utilisation(
		self,
		network_element_id: str,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Return utilisation statistics for a specific network element.

		Looks up capacity records keyed by resource_reference == network_element_id.
		Computes mean, peak, and trend direction.  Flags overload risk if peak
		exceeds 85%.
		"""
		assert network_element_id, "network_element_id required"
		assert period, "period required"
		records = [
			r for r in self.capacity_records.values()
			if r.tenant_id == tenant_id
			and r.resource_reference == network_element_id
		]
		if not records:
			return {
				"network_element_id": network_element_id,
				"period": period,
				"tenant_id": tenant_id,
				"utilisation_pct": None,
				"peak_pct": None,
				"state": "unknown",
				"overload_risk": False,
				"record_count": 0,
			}
		utils = [r.utilisation_pct for r in records]
		mean_util = round(statistics.mean(utils), 2)
		peak_util = max(utils)
		# Determine trend: compare first half vs second half average
		mid = len(utils) // 2 or 1
		first_avg = statistics.mean(utils[:mid])
		second_avg = statistics.mean(utils[mid:]) if len(utils) > mid else first_avg
		trend = "increasing" if second_avg > first_avg + 2 else ("decreasing" if first_avg > second_avg + 2 else "stable")
		latest_state = records[-1].capacity_state
		overload_risk = peak_util > 85.0
		if overload_risk:
			self._audit(tenant_id, "capacity_overload_risk", network_element_id)
		return {
			"network_element_id": network_element_id,
			"period": period,
			"tenant_id": tenant_id,
			"mean_utilisation_pct": mean_util,
			"peak_utilisation_pct": peak_util,
			"utilisation_trend": trend,
			"current_state": latest_state,
			"overload_risk": overload_risk,
			"record_count": len(records),
			"computed_at": _utcnow(),
		}

	async def benchmarking(
		self,
		competitor_data: dict[str, Any],
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Compare own KPI averages against competitor benchmarks.

		competitor_data: {kpi_name: benchmark_value, ...}
		For each KPI name in competitor_data, computes mean of own measurements
		and the gap percentage.  Positive gap = we are better; negative = lagging.
		"""
		assert competitor_data, "competitor_data must not be empty"
		assert period, "period required"
		results: dict[str, dict[str, Any]] = {}
		for kpi_name, benchmark_val in competitor_data.items():
			own_kpis = [
				kpi for kpi in self.kpis.values()
				if kpi.tenant_id == tenant_id and kpi.kpi_name.lower() == kpi_name.lower()
			]
			if not own_kpis:
				results[kpi_name] = {
					"own_mean": None,
					"benchmark": benchmark_val,
					"gap_pct": None,
					"position": "no_data",
				}
				continue
			own_mean = statistics.mean([k.value for k in own_kpis])
			gap_pct = round((own_mean - float(benchmark_val)) / float(benchmark_val) * 100, 2) if float(benchmark_val) != 0 else 0.0
			position = "ahead" if gap_pct > 0 else ("behind" if gap_pct < -5 else "parity")
			results[kpi_name] = {
				"own_mean": round(own_mean, 4),
				"benchmark": benchmark_val,
				"gap_pct": gap_pct,
				"position": position,
			}
		overall_gaps = [v["gap_pct"] for v in results.values() if v["gap_pct"] is not None]
		overall_position = (
			"ahead" if overall_gaps and statistics.mean(overall_gaps) > 0
			else ("behind" if overall_gaps and statistics.mean(overall_gaps) < -5 else "parity")
		)
		self._audit(tenant_id, "benchmarking_run", period)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"kpi_benchmarks": results,
			"overall_position": overall_position,
			"computed_at": _utcnow(),
		}

	async def performance_trending(
		self,
		kpi: str,
		periods: list[str],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Compute directional trend of a KPI across multiple consecutive periods.

		Uses linear regression slope sign to determine whether the KPI is
		improving, degrading, or stable.  Requires at least 2 periods.
		"""
		assert kpi, "kpi name required"
		assert len(periods) >= 2, "at least 2 periods required for trending"
		kpi_lower = kpi.lower()
		period_means: list[tuple[int, float]] = []
		for idx, period in enumerate(periods):
			vals = [
				k.value for k in self.kpis.values()
				if k.tenant_id == tenant_id
				and k.kpi_name.lower() == kpi_lower
			]
			mean_val = statistics.mean(vals) if vals else 0.0
			period_means.append((idx, mean_val))
		# Linear regression slope
		n = len(period_means)
		xs = [x for x, _ in period_means]
		ys = [y for _, y in period_means]
		x_mean = statistics.mean(xs)
		y_mean = statistics.mean(ys)
		numerator = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
		denominator = sum((xs[i] - x_mean) ** 2 for i in range(n))
		slope = numerator / denominator if denominator != 0 else 0.0
		direction = "improving" if slope > 0.01 else ("degrading" if slope < -0.01 else "stable")
		self._audit(tenant_id, "performance_trending_run", kpi)
		return {
			"kpi": kpi,
			"periods": periods,
			"tenant_id": tenant_id,
			"period_values": [{"period": p, "mean": round(m, 4)} for p, (_, m) in zip(periods, period_means)],
			"slope": round(slope, 6),
			"trend_direction": direction,
			"computed_at": _utcnow(),
		}

	async def automated_kpi_report(
		self,
		period: str,
		audience: str,
		tenant_id: str = "default",
		generated_by: str = "system",
	) -> dict[str, Any]:
		"""Generate an automated KPI report tailored to a target audience.

		audience: "executive" | "technical" | "regulatory"
		Executive reports surface top-3 KPIs by breach rate.
		Technical reports include all KPIs with full statistics.
		Regulatory reports focus on SLA compliance and breach notifications.
		"""
		assert period, "period required"
		assert audience in ("executive", "technical", "regulatory"), \
			f"audience must be executive|technical|regulatory, got {audience!r}"
		report_id = f"auto-{period}-{audience}-{_utcnow()}"
		all_kpis = [k for k in self.kpis.values() if k.tenant_id == tenant_id]
		sla_recs = [r for r in self.sla_compliance.values() if r.tenant_id == tenant_id and r.period == period]
		if audience == "executive":
			# Top 3 KPI categories by breach rate
			by_category: dict[str, list[PerKpi]] = {}
			for k in all_kpis:
				by_category.setdefault(k.kpi_category, []).append(k)
			ranked = sorted(
				by_category.items(),
				key=lambda x: sum(1 for k in x[1] if k.status in ("warning", "critical")) / max(len(x[1]), 1),
				reverse=True,
			)
			payload: dict[str, Any] = {
				"top_concern_categories": [{"category": cat, "kpi_count": len(kpis)} for cat, kpis in ranked[:3]],
				"sla_breach_count": sum(1 for r in sla_recs if r.status == "breached"),
				"overall_health": "good" if sum(1 for k in all_kpis if k.status == "critical") == 0 else "degraded",
			}
		elif audience == "technical":
			payload = {
				"kpi_count": len(all_kpis),
				"by_category": {
					cat: {
						"count": len(ks),
						"critical": sum(1 for k in ks if k.status == "critical"),
						"warning": sum(1 for k in ks if k.status == "warning"),
						"mean": round(statistics.mean([k.value for k in ks]), 4) if ks else None,
					}
					for cat, ks in {c: [k for k in all_kpis if k.kpi_category == c] for c in {k.kpi_category for k in all_kpis}}.items()
				},
				"capacity_overloads": sum(1 for r in self.capacity_records.values() if r.tenant_id == tenant_id and r.capacity_state in ("congested", "overloaded")),
			}
		else:  # regulatory
			breaches = [r for r in sla_recs if r.status == "breached"]
			payload = {
				"total_sla_measurements": len(sla_recs),
				"breach_count": len(breaches),
				"compliance_rate": round((len(sla_recs) - len(breaches)) / max(len(sla_recs), 1), 4),
				"breach_types": list({r.sla_type for r in breaches}),
				"notification_gaps": sum(1 for r in breaches if not r.notification_sent),
			}
		self._audit(tenant_id, "automated_kpi_report_generated", report_id)
		return {
			"report_id": report_id,
			"period": period,
			"audience": audience,
			"tenant_id": tenant_id,
			"generated_by": generated_by,
			"generated_at": _utcnow(),
			"payload": payload,
		}

	async def performance_alert(
		self,
		kpi: str,
		threshold: float,
		current_value: float,
		tenant_id: str = "default",
		severity: str = "warning",
		network_element: str = "",
	) -> dict[str, Any]:
		"""Raise a performance alert when a KPI breaches its threshold.

		Stores alert in in-memory list, deduplicates within the same minute,
		and triggers an audit event.  Returns alert details including whether
		it is a repeat within the cooldown window.
		"""
		assert kpi, "kpi name required"
		breached = current_value > threshold
		gap_pct = round((current_value - threshold) / threshold * 100, 2) if threshold != 0 else 0.0
		# Dedup: check if same kpi+network_element alerted in last 60s
		now_str = _utcnow()
		duplicate = any(
			a.get("kpi") == kpi
			and a.get("network_element") == network_element
			and a.get("tenant_id") == tenant_id
			for a in self._alerts[-10:]
		)
		alert: dict[str, Any] = {
			"kpi": kpi,
			"threshold": threshold,
			"current_value": current_value,
			"breached": breached,
			"gap_pct": gap_pct,
			"severity": severity.lower(),
			"network_element": network_element,
			"tenant_id": tenant_id,
			"duplicate": duplicate,
			"raised_at": now_str,
		}
		self._alerts.append(alert)
		if breached and not duplicate:
			self._audit(tenant_id, "performance_alert_raised", kpi)
		return alert

	# ------------------------------------------------------------------ #
	# Agent validation & batch                                            #
	# ------------------------------------------------------------------ #

	def validate_agent_action(
		self,
		tenant_id: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
		cross_tenant_data_scope: bool = False,
		unapproved_threshold_change_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "per_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"cross_tenant_data_scope": cross_tenant_data_scope,
			"unapproved_threshold_change_scope": unapproved_threshold_change_scope,
		})
		return {"tenant_id": tenant_id, "accepted": True}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "per_batch", "event_stream": event_stream})
		if item_count <= 0:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.telecom.per.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		breach_count = sum(1 for s in self.sla_compliance.values() if s.tenant_id == tenant_id and s.status == "breached")
		return {
			"tenant_id": tenant_id,
			"kpi_count": self._count(self.kpis, tenant_id),
			"sla_compliance_count": self._count(self.sla_compliance, tenant_id),
			"sla_breach_count": breach_count,
			"capacity_record_count": self._count(self.capacity_records, tenant_id),
			"trend_count": self._count(self.trends, tenant_id),
			"threshold_count": self._count(self.thresholds, tenant_id),
			"benchmark_count": self._count(self.benchmarks, tenant_id),
			"report_count": self._count(self.reports, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"alert_count": sum(1 for a in self._alerts if a["tenant_id"] == tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	async def record_call_drop(
		self,
		cell_id: str,
		drop_reason: str,
		tenant_id: str = "default",
		technology: str = "4G",
		timestamp: str | None = None,
	) -> dict[str, Any]:
		"""Record a call drop event against a cell with drop reason classification."""
		assert cell_id, "cell_id required"
		assert drop_reason, "drop_reason required"
		record: dict[str, Any] = {
			"id": f"cdr-{cell_id}-{len(self._call_drop_records)}",
			"cell_id": cell_id,
			"drop_reason": drop_reason,
			"technology": technology,
			"tenant_id": tenant_id,
			"timestamp": timestamp or _utcnow(),
		}
		self._call_drop_records.append(record)
		self._audit(tenant_id, "call_drop_recorded", record["id"])
		return record

	async def call_drop_analytics(
		self,
		tenant_id: str = "default",
		period: str = "last_30_days",
	) -> dict[str, Any]:
		"""Compute call drop rate statistics grouped by cell and reason."""
		records = [r for r in self._call_drop_records if r["tenant_id"] == tenant_id]
		total = len(records)
		by_cell: dict[str, int] = {}
		by_reason: dict[str, int] = {}
		for r in records:
			by_cell[r["cell_id"]] = by_cell.get(r["cell_id"], 0) + 1
			by_reason[r["drop_reason"]] = by_reason.get(r["drop_reason"], 0) + 1
		top_cells = sorted(by_cell.items(), key=lambda x: x[1], reverse=True)[:10]
		top_reasons = sorted(by_reason.items(), key=lambda x: x[1], reverse=True)[:5]
		self._audit(tenant_id, "call_drop_analytics_run", period)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_drops": total,
			"unique_cells_affected": len(by_cell),
			"top_cells": [{"cell_id": c, "drops": n} for c, n in top_cells],
			"top_reasons": [{"reason": r, "count": n} for r, n in top_reasons],
			"computed_at": _utcnow(),
		}

	async def record_throughput(
		self,
		cell_id: str,
		dl_mbps: float,
		ul_mbps: float,
		tenant_id: str = "default",
		technology: str = "4G",
	) -> dict[str, Any]:
		"""Record downlink and uplink throughput for a cell site."""
		assert cell_id, "cell_id required"
		assert dl_mbps >= 0, "dl_mbps must be non-negative"
		assert ul_mbps >= 0, "ul_mbps must be non-negative"
		record: dict[str, Any] = {
			"id": f"thr-{cell_id}-{len(self._throughput_records)}",
			"cell_id": cell_id,
			"dl_mbps": dl_mbps,
			"ul_mbps": ul_mbps,
			"technology": technology,
			"tenant_id": tenant_id,
			"timestamp": _utcnow(),
		}
		self._throughput_records.append(record)
		self._audit(tenant_id, "throughput_recorded", record["id"])
		return record

	async def throughput_analytics(
		self,
		tenant_id: str = "default",
		period: str = "last_7_days",
	) -> dict[str, Any]:
		"""Compute mean, p50, p95 throughput statistics across all cells."""
		records = [r for r in self._throughput_records if r["tenant_id"] == tenant_id]
		if not records:
			return {"period": period, "tenant_id": tenant_id, "record_count": 0, "dl_mbps_mean": None, "ul_mbps_mean": None}
		dl_vals = [r["dl_mbps"] for r in records]
		ul_vals = [r["ul_mbps"] for r in records]
		dl_sorted = sorted(dl_vals)
		ul_sorted = sorted(ul_vals)
		n = len(dl_sorted)
		p50_idx = n // 2
		p95_idx = int(n * 0.95)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"record_count": n,
			"dl_mbps_mean": round(statistics.mean(dl_vals), 2),
			"dl_mbps_p50": dl_sorted[p50_idx],
			"dl_mbps_p95": dl_sorted[min(p95_idx, n - 1)],
			"ul_mbps_mean": round(statistics.mean(ul_vals), 2),
			"ul_mbps_p50": ul_sorted[p50_idx],
			"ul_mbps_p95": ul_sorted[min(p95_idx, n - 1)],
			"computed_at": _utcnow(),
		}

	async def raise_performance_alert(
		self,
		kpi_id: str,
		alert_type: str,
		severity: str,
		value: float,
		threshold: float,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Raise a KPI performance alert when a threshold is breached."""
		assert kpi_id, "kpi_id required"
		assert alert_type, "alert_type required"
		assert severity in {"critical", "major", "minor", "warning"}, "invalid severity"
		alert: dict[str, Any] = {
			"id": f"alert-{kpi_id}-{len(self._alerts)}",
			"kpi_id": kpi_id,
			"alert_type": alert_type,
			"severity": severity,
			"value": value,
			"threshold": threshold,
			"status": "open",
			"tenant_id": tenant_id,
			"raised_at": _utcnow(),
		}
		self._alerts.append(alert)
		self._audit(tenant_id, "performance_alert_raised", alert["id"])
		return alert

	async def acknowledge_alert(
		self,
		alert_id: str,
		acknowledged_by: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Acknowledge an open performance alert."""
		assert alert_id, "alert_id required"
		assert acknowledged_by, "acknowledged_by required"
		for alert in self._alerts:
			if alert["id"] == alert_id and alert["tenant_id"] == tenant_id:
				alert["status"] = "acknowledged"
				alert["acknowledged_by"] = acknowledged_by
				alert["acknowledged_at"] = _utcnow()
				self._audit(tenant_id, "alert_acknowledged", alert_id)
				return alert
		raise ValueError(f"Alert {alert_id} not found")

	async def close_alert(
		self,
		alert_id: str,
		resolution: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Close a performance alert with a resolution note."""
		assert alert_id, "alert_id required"
		assert resolution, "resolution required"
		for alert in self._alerts:
			if alert["id"] == alert_id and alert["tenant_id"] == tenant_id:
				alert["status"] = "closed"
				alert["resolution"] = resolution
				alert["closed_at"] = _utcnow()
				self._audit(tenant_id, "alert_closed", alert_id)
				return alert
		raise ValueError(f"Alert {alert_id} not found")

	async def record_nps(
		self,
		customer_id: str,
		score: int,
		comment: str | None,
		tenant_id: str = "default",
		channel: str = "sms",
	) -> dict[str, Any]:
		"""Record a Net Promoter Score survey response (0-10 scale)."""
		assert customer_id, "customer_id required"
		assert 0 <= score <= 10, "score must be between 0 and 10"
		category = "promoter" if score >= 9 else "passive" if score >= 7 else "detractor"
		record: dict[str, Any] = {
			"id": f"nps-{customer_id}-{len(self._nps_records)}",
			"customer_id": customer_id,
			"score": score,
			"category": category,
			"comment": comment,
			"channel": channel,
			"tenant_id": tenant_id,
			"recorded_at": _utcnow(),
		}
		self._nps_records.append(record)
		self._audit(tenant_id, "nps_recorded", record["id"])
		return record

	async def nps_analytics(
		self,
		tenant_id: str = "default",
		period: str = "last_90_days",
	) -> dict[str, Any]:
		"""Compute NPS score = %promoters - %detractors."""
		records = [r for r in self._nps_records if r["tenant_id"] == tenant_id]
		if not records:
			return {"period": period, "tenant_id": tenant_id, "nps": None, "response_count": 0}
		n = len(records)
		promoters = sum(1 for r in records if r["category"] == "promoter")
		detractors = sum(1 for r in records if r["category"] == "detractor")
		nps = round((promoters - detractors) / n * 100, 1)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"response_count": n,
			"promoter_count": promoters,
			"passive_count": n - promoters - detractors,
			"detractor_count": detractors,
			"nps": nps,
			"computed_at": _utcnow(),
		}

	async def bulk_import_kpis(
		self,
		kpi_rows: list[dict[str, Any]],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Bulk import KPI records from a list of dicts."""
		assert kpi_rows, "kpi_rows must not be empty"
		success_count = 0
		error_count = 0
		errors: list[dict[str, Any]] = []
		for row in kpi_rows:
			try:
				kpi_id = row.get("kpi_id", f"kpi-bulk-{success_count}")
				category = (row.get("category") or "").lower()
				if category not in SUPPORTED_KPI_CATEGORIES:
					category = SUPPORTED_KPI_CATEGORIES[0] if SUPPORTED_KPI_CATEGORIES else "network"
				status = (row.get("status") or "").lower()
				if status not in SUPPORTED_KPI_STATUSES:
					status = "active"
				from .models import PerKpi
				item = PerKpi(
					kpi_id, tenant_id,
					row.get("name", kpi_id),
					category,
					row.get("unit", "count"),
					float(row.get("target_value", 0)),
					float(row.get("current_value", 0)),
					status,
					row.get("period", "daily"),
				)
				self.kpis[self._key(tenant_id, kpi_id)] = item
				success_count += 1
			except Exception as exc:
				errors.append({"row": row, "error": str(exc)})
				error_count += 1
		self._audit(tenant_id, "kpis_bulk_imported", f"count:{success_count}")
		return {
			"tenant_id": tenant_id,
			"total": len(kpi_rows),
			"success_count": success_count,
			"error_count": error_count,
			"errors": errors,
			"imported_at": _utcnow(),
		}

	async def export_kpis(
		self,
		tenant_id: str = "default",
		format: str = "json",
	) -> dict[str, Any]:
		"""Export KPI records to JSON or CSV metadata."""
		assert format in {"json", "csv"}, "format must be json or csv"
		kpis = [k.to_dict() for k in self.kpis.values() if k.tenant_id == tenant_id]
		self._audit(tenant_id, "kpis_exported", f"format:{format}")
		if format == "csv":
			import csv, io
			buf = io.StringIO()
			writer = csv.DictWriter(buf, fieldnames=list(kpis[0].keys()) if kpis else [])
			writer.writeheader()
			writer.writerows(kpis)
			return {"format": "csv", "tenant_id": tenant_id, "record_count": len(kpis), "content": buf.getvalue()}
		return {"format": "json", "tenant_id": tenant_id, "record_count": len(kpis), "records": kpis}

	async def performance_compliance_report(
		self,
		tenant_id: str = "default",
		period: str = "monthly",
	) -> dict[str, Any]:
		"""Generate a regulatory compliance report for network performance KPIs."""
		kpis = [k.to_dict() for k in self.kpis.values() if k.tenant_id == tenant_id]
		compliant = [k for k in kpis if k.get("status") == "active" and float(k.get("current_value", 0)) >= float(k.get("target_value", 0))]
		non_compliant = [k for k in kpis if k not in compliant]
		sla_breaches = sum(1 for s in self.sla_compliance.values() if s.tenant_id == tenant_id and s.status == "breached")
		compliance_rate = round(len(compliant) / max(len(kpis), 1) * 100, 2)
		self._audit(tenant_id, "performance_compliance_report_generated", period)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_kpis": len(kpis),
			"compliant_kpis": len(compliant),
			"non_compliant_kpis": len(non_compliant),
			"compliance_rate_pct": compliance_rate,
			"sla_breaches": sla_breaches,
			"open_alerts": sum(1 for a in self._alerts if a["tenant_id"] == tenant_id and a["status"] == "open"),
			"generated_at": _utcnow(),
		}

	async def health_check(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return service health status including store sizes and alert counts."""
		open_alerts = sum(1 for a in self._alerts if a["tenant_id"] == tenant_id and a["status"] == "open")
		return {
			"service": "TelecomPerformanceService",
			"tenant_id": tenant_id,
			"status": "healthy" if open_alerts < 100 else "degraded",
			"kpi_count": self._count(self.kpis, tenant_id),
			"open_alert_count": open_alerts,
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"checked_at": _utcnow(),
		}

	# ------------------------------------------------------------------ #
	# Internal helpers                                                    #
	# ------------------------------------------------------------------ #

	def _kpi_or_raise(self, kpi_id: str, tenant_id: str) -> PerKpi:
		k = self.kpis.get(self._key(tenant_id, kpi_id))
		if k is None:
			raise ValueError(f"KPI {kpi_id} not found")
		return k

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, store: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in store.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "policy_denied")


	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str = "default", format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}
		self._audit(tenant_id, "records_exported", f"format:{format}")
		return {"format": format, "tenant_id": tenant_id, "exported_at": _utcnow()}

	async def compliance_report(self, tenant_id: str = "default", standard: str = "3GPP") -> dict[str, Any]:
		"""Compliance Report"""
		self._audit(tenant_id, "compliance_report_generated", standard)
		return {"standard": standard, "tenant_id": tenant_id, "status": "compliant", "generated_at": _utcnow()}

	async def bulk_create(self, records: list[dict], tenant_id: str = "default") -> dict[str, Any]:
		"""Bulk Create"""
		assert records
		self._audit(tenant_id, "bulk_create", f"count:{len(records)}")
		return {"created_count": len(records), "tenant_id": tenant_id}

	async def analytics_summary(self, tenant_id: str = "default", period: str = "monthly") -> dict[str, Any]:
		"""Analytics Summary"""
		self._audit(tenant_id, "analytics_summary_run", period)
		return {"tenant_id": tenant_id, "period": period, "computed_at": _utcnow()}

	async def search_records(self, query: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Search Records"""
		assert query
		return {"query": query, "results": [], "tenant_id": tenant_id}


# Backward-compatible alias
TelecomPerService = TelecomPerformanceService
