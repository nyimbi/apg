"""Hospitality Analytics service — RevPAR, ADR, occupancy, GOP PAR, segment analysis, pace reporting, guest satisfaction."""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)


def _uid() -> str:
	return uuid4().hex[:12]


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


class ANAService:
	"""Hospitality Analytics service."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.kpi_snapshots: dict[str, dict[str, Any]] = {}
		self.segment_reports: dict[str, dict[str, Any]] = {}
		self.pace_reports: dict[str, dict[str, Any]] = {}
		self.satisfaction_surveys: dict[str, dict[str, Any]] = {}
		self.competitive_sets: dict[str, dict[str, Any]] = {}
		self.revenue_summaries: dict[str, dict[str, Any]] = {}
		self.channel_reports: dict[str, dict[str, Any]] = {}
		self.forecast_reports: dict[str, dict[str, Any]] = {}
		self.benchmarks: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _emit(self, tenant_id: str, event_type: str, record_id: str, record_type: str, details: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": _uid(),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record_id,
			"record_type": record_type,
			"details": details or {},
			"created_at": _now(),
		})

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "hos_ana",
			"status": "healthy",
			"kpi_snapshots": len(self.kpi_snapshots),
			"satisfaction_surveys": len(self.satisfaction_surveys),
			"checked_at": _now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": "hos_ana",
			"name": "Hospitality Analytics",
			"domain": "hospitality",
			"version": "1.0.0",
			"description": "RevPAR, ADR, occupancy, GOP PAR, segment analysis, pace reporting, guest satisfaction",
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── KPI Snapshots ─────────────────────────────────────────────────────────

	async def record_kpi_snapshot(self, date: str, total_rooms: int, occupied_rooms: int,
	                               total_revenue: float, room_revenue: float, ancillary_revenue: float,
	                               goppar: float | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		"""Record daily KPI metrics and compute derived indicators."""
		tenant = self._tenant(tenant_id)
		occupancy_rate = round(occupied_rooms / total_rooms * 100, 2) if total_rooms else 0.0
		adr = round(room_revenue / occupied_rooms, 2) if occupied_rooms else 0.0
		revpar = round(room_revenue / total_rooms, 2) if total_rooms else 0.0
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"date": date,
			"total_rooms": total_rooms,
			"occupied_rooms": occupied_rooms,
			"occupancy_rate": occupancy_rate,
			"adr": adr,
			"revpar": revpar,
			"goppar": goppar,
			"total_revenue": total_revenue,
			"room_revenue": room_revenue,
			"ancillary_revenue": ancillary_revenue,
			"ancillary_per_occupied_room": round(ancillary_revenue / occupied_rooms, 2) if occupied_rooms else 0.0,
			"total_revpar": round(total_revenue / total_rooms, 2) if total_rooms else 0.0,
			"generated_at": _now(),
		}
		self.kpi_snapshots[record["id"]] = record
		self._emit(tenant, "kpi_snapshot_recorded", record["id"], "kpi_snapshot", {"date": date, "revpar": revpar})
		return deepcopy(record)

	async def list_kpi_snapshots(self, tenant_id: str | None = None, date_from: str | None = None, date_to: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(s) for s in self.kpi_snapshots.values() if s["tenant_id"] == tenant]
		if date_from:
			items = [s for s in items if s["date"] >= date_from]
		if date_to:
			items = [s for s in items if s["date"] <= date_to]
		return sorted(items, key=lambda x: x["date"])

	async def get_kpi_snapshot(self, snapshot_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		snap = self.kpi_snapshots.get(snapshot_id)
		if not snap or snap["tenant_id"] != tenant:
			raise KeyError(f"kpi_snapshot_not_found:{snapshot_id}")
		return deepcopy(snap)

	async def update_kpi_snapshot(self, snapshot_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		snap = self.kpi_snapshots.get(snapshot_id)
		if not snap or snap["tenant_id"] != tenant:
			raise KeyError(f"kpi_snapshot_not_found:{snapshot_id}")
		for k, v in updates.items():
			if v is not None:
				snap[k] = v
		return deepcopy(snap)

	async def delete_kpi_snapshot(self, snapshot_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		snap = self.kpi_snapshots.get(snapshot_id)
		if not snap or snap["tenant_id"] != tenant:
			raise KeyError(f"kpi_snapshot_not_found:{snapshot_id}")
		del self.kpi_snapshots[snapshot_id]
		return {"deleted": True, "snapshot_id": snapshot_id}

	async def kpi_period_summary(self, date_from: str, date_to: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Aggregate KPI metrics over a date range."""
		tenant = self._tenant(tenant_id)
		snaps = [s for s in self.kpi_snapshots.values() if s["tenant_id"] == tenant and date_from <= s["date"] <= date_to]
		if not snaps:
			return {"tenant_id": tenant, "date_from": date_from, "date_to": date_to, "days": 0, "generated_at": _now()}
		total_rooms_days = sum(s["total_rooms"] for s in snaps)
		total_occupied = sum(s["occupied_rooms"] for s in snaps)
		avg_occupancy = round(total_occupied / total_rooms_days * 100, 2) if total_rooms_days else 0.0
		total_room_rev = sum(s["room_revenue"] for s in snaps)
		avg_adr = round(total_room_rev / total_occupied, 2) if total_occupied else 0.0
		avg_revpar = round(total_room_rev / total_rooms_days, 2) if total_rooms_days else 0.0
		total_rev = sum(s["total_revenue"] for s in snaps)
		return {
			"tenant_id": tenant,
			"date_from": date_from,
			"date_to": date_to,
			"days": len(snaps),
			"avg_occupancy_rate": avg_occupancy,
			"avg_adr": avg_adr,
			"avg_revpar": avg_revpar,
			"total_revenue": round(total_rev, 2),
			"total_room_revenue": round(total_room_rev, 2),
			"total_ancillary_revenue": round(sum(s["ancillary_revenue"] for s in snaps), 2),
			"best_revpar_date": max(snaps, key=lambda x: x["revpar"])["date"],
			"lowest_occupancy_date": min(snaps, key=lambda x: x["occupancy_rate"])["date"],
			"generated_at": _now(),
		}

	# ── Segment Analysis ──────────────────────────────────────────────────────

	async def record_segment_report(self, period: str, segment: str, room_nights: int,
	                                 revenue: float, total_room_nights: int,
	                                 tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		avg_rate = round(revenue / room_nights, 2) if room_nights else 0.0
		share_pct = round(room_nights / total_room_nights * 100, 2) if total_room_nights else 0.0
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"period": period,
			"segment": segment,
			"room_nights": room_nights,
			"revenue": revenue,
			"avg_rate": avg_rate,
			"share_pct": share_pct,
			"created_at": _now(),
		}
		self.segment_reports[record["id"]] = record
		self._emit(tenant, "segment_report_recorded", record["id"], "segment_report")
		return deepcopy(record)

	async def list_segment_reports(self, tenant_id: str | None = None, period: str | None = None, segment: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.segment_reports.values() if r["tenant_id"] == tenant]
		if period:
			items = [r for r in items if r["period"] == period]
		if segment:
			items = [r for r in items if r["segment"] == segment]
		return items

	async def segment_mix_report(self, period: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Generate a segment mix report for a period."""
		tenant = self._tenant(tenant_id)
		reports = [r for r in self.segment_reports.values() if r["tenant_id"] == tenant and r["period"] == period]
		total_revenue = sum(r["revenue"] for r in reports)
		total_nights = sum(r["room_nights"] for r in reports)
		return {
			"tenant_id": tenant,
			"period": period,
			"segments": [{
				"segment": r["segment"],
				"room_nights": r["room_nights"],
				"revenue": r["revenue"],
				"avg_rate": r["avg_rate"],
				"revenue_share_pct": round(r["revenue"] / total_revenue * 100, 2) if total_revenue else 0.0,
				"night_share_pct": round(r["room_nights"] / total_nights * 100, 2) if total_nights else 0.0,
			} for r in reports],
			"total_revenue": round(total_revenue, 2),
			"total_room_nights": total_nights,
			"generated_at": _now(),
		}

	# ── Pace Reporting ────────────────────────────────────────────────────────

	async def record_pace_report(self, report_date: str, future_date: str, booked_rooms: int,
	                              booked_revenue: float, on_the_books_adr: float,
	                              pickup_last_7_days: int = 0, vs_last_year_pct: float | None = None,
	                              tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"report_date": report_date,
			"future_date": future_date,
			"booked_rooms": booked_rooms,
			"booked_revenue": booked_revenue,
			"on_the_books_adr": on_the_books_adr,
			"pickup_last_7_days": pickup_last_7_days,
			"vs_same_time_last_year_pct": vs_last_year_pct,
			"created_at": _now(),
		}
		self.pace_reports[record["id"]] = record
		self._emit(tenant, "pace_report_recorded", record["id"], "pace_report")
		return deepcopy(record)

	async def list_pace_reports(self, tenant_id: str | None = None, future_date: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.pace_reports.values() if r["tenant_id"] == tenant]
		if future_date:
			items = [r for r in items if r["future_date"] == future_date]
		return sorted(items, key=lambda x: x["report_date"])

	async def pace_comparison(self, future_date: str, days_out: int = 30, tenant_id: str | None = None) -> dict[str, Any]:
		"""Compare current booking pace to historical."""
		tenant = self._tenant(tenant_id)
		reports = [r for r in self.pace_reports.values() if r["tenant_id"] == tenant and r["future_date"] == future_date]
		if not reports:
			return {"tenant_id": tenant, "future_date": future_date, "data_points": 0, "generated_at": _now()}
		latest = max(reports, key=lambda x: x["report_date"])
		return {
			"tenant_id": tenant,
			"future_date": future_date,
			"days_out": days_out,
			"current_booked_rooms": latest["booked_rooms"],
			"current_adr": latest["on_the_books_adr"],
			"pickup_last_7_days": latest["pickup_last_7_days"],
			"vs_last_year_pct": latest["vs_same_time_last_year_pct"],
			"data_points": len(reports),
			"generated_at": _now(),
		}

	# ── Guest Satisfaction ────────────────────────────────────────────────────

	async def record_satisfaction_survey(self, reservation_id: str, guest_name: str, overall_score: float,
	                                      room_score: float | None = None, service_score: float | None = None,
	                                      food_score: float | None = None, cleanliness_score: float | None = None,
	                                      comments: str | None = None, channel: str = "post_stay_email",
	                                      tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		if not 1.0 <= overall_score <= 10.0:
			raise ValueError(f"overall_score_must_be_1_to_10:{overall_score}")
		# NPS category: score 9-10 = promoter, 7-8 = passive, 1-6 = detractor
		if overall_score >= 9:
			nps_category = "promoter"
		elif overall_score >= 7:
			nps_category = "passive"
		else:
			nps_category = "detractor"
		avg_scores = [s for s in [room_score, service_score, food_score, cleanliness_score] if s is not None]
		composite_score = round(sum(avg_scores) / len(avg_scores), 2) if avg_scores else overall_score
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"reservation_id": reservation_id,
			"guest_name": guest_name,
			"overall_score": overall_score,
			"room_score": room_score,
			"service_score": service_score,
			"food_score": food_score,
			"cleanliness_score": cleanliness_score,
			"composite_score": composite_score,
			"comments": comments,
			"channel": channel,
			"nps_category": nps_category,
			"created_at": _now(),
		}
		self.satisfaction_surveys[record["id"]] = record
		self._emit(tenant, "satisfaction_survey_recorded", record["id"], "satisfaction_survey", {"nps": nps_category, "score": overall_score})
		return deepcopy(record)

	async def list_satisfaction_surveys(self, tenant_id: str | None = None, date_from: str | None = None, nps_category: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(s) for s in self.satisfaction_surveys.values() if s["tenant_id"] == tenant]
		if date_from:
			items = [s for s in items if s["created_at"][:10] >= date_from]
		if nps_category:
			items = [s for s in items if s["nps_category"] == nps_category]
		return items

	async def get_satisfaction_survey(self, survey_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		s = self.satisfaction_surveys.get(survey_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"satisfaction_survey_not_found:{survey_id}")
		return deepcopy(s)

	async def update_satisfaction_survey(self, survey_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		s = self.satisfaction_surveys.get(survey_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"satisfaction_survey_not_found:{survey_id}")
		for k, v in updates.items():
			if v is not None:
				s[k] = v
		return deepcopy(s)

	async def delete_satisfaction_survey(self, survey_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		s = self.satisfaction_surveys.get(survey_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"satisfaction_survey_not_found:{survey_id}")
		del self.satisfaction_surveys[survey_id]
		return {"deleted": True, "survey_id": survey_id}

	async def satisfaction_summary(self, date_from: str | None = None, date_to: str | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		surveys = [s for s in self.satisfaction_surveys.values() if s["tenant_id"] == tenant]
		if date_from:
			surveys = [s for s in surveys if s["created_at"][:10] >= date_from]
		if date_to:
			surveys = [s for s in surveys if s["created_at"][:10] <= date_to]
		if not surveys:
			return {"tenant_id": tenant, "survey_count": 0, "generated_at": _now()}
		promoters = sum(1 for s in surveys if s["nps_category"] == "promoter")
		detractors = sum(1 for s in surveys if s["nps_category"] == "detractor")
		nps = round((promoters - detractors) / len(surveys) * 100, 1)
		return {
			"tenant_id": tenant,
			"survey_count": len(surveys),
			"avg_overall_score": round(sum(s["overall_score"] for s in surveys) / len(surveys), 2),
			"avg_room_score": round(sum(s["room_score"] for s in surveys if s["room_score"]) / max(sum(1 for s in surveys if s["room_score"]), 1), 2),
			"avg_service_score": round(sum(s["service_score"] for s in surveys if s["service_score"]) / max(sum(1 for s in surveys if s["service_score"]), 1), 2),
			"promoters": promoters,
			"passives": sum(1 for s in surveys if s["nps_category"] == "passive"),
			"detractors": detractors,
			"nps": nps,
			"generated_at": _now(),
		}

	# ── Competitive Set ───────────────────────────────────────────────────────

	async def create_competitive_set(self, name: str, properties: list[dict[str, Any]], tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"name": name,
			"properties": deepcopy(properties),
			"status": "active",
			"created_at": _now(),
		}
		self.competitive_sets[record["id"]] = record
		self._emit(tenant, "competitive_set_created", record["id"], "competitive_set")
		return deepcopy(record)

	async def list_competitive_sets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(c) for c in self.competitive_sets.values() if c["tenant_id"] == tenant]

	# ── Benchmarking ──────────────────────────────────────────────────────────

	async def record_benchmark(self, period: str, metric: str, our_value: float, market_avg: float,
	                            competitive_set_avg: float | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		index = round(our_value / market_avg * 100, 1) if market_avg else 0.0
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"period": period,
			"metric": metric,
			"our_value": our_value,
			"market_avg": market_avg,
			"competitive_set_avg": competitive_set_avg,
			"index": index,
			"status": "above_market" if index > 100 else "below_market",
			"created_at": _now(),
		}
		self.benchmarks[record["id"]] = record
		self._emit(tenant, "benchmark_recorded", record["id"], "benchmark", {"metric": metric, "index": index})
		return deepcopy(record)

	async def list_benchmarks(self, tenant_id: str | None = None, metric: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(b) for b in self.benchmarks.values() if b["tenant_id"] == tenant]
		if metric:
			items = [b for b in items if b["metric"] == metric]
		return items

	# ── Channel Revenue ───────────────────────────────────────────────────────

	async def record_channel_revenue(self, period: str, channel: str, bookings: int, revenue: float,
	                                  commission: float, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		net_revenue = revenue - commission
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"period": period,
			"channel": channel,
			"bookings": bookings,
			"revenue": revenue,
			"commission": commission,
			"net_revenue": net_revenue,
			"commission_pct": round(commission / revenue * 100, 2) if revenue else 0.0,
			"created_at": _now(),
		}
		self.channel_reports[record["id"]] = record
		return deepcopy(record)

	async def channel_mix_report(self, period: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		reports = [r for r in self.channel_reports.values() if r["tenant_id"] == tenant and r["period"] == period]
		total_rev = sum(r["revenue"] for r in reports)
		return {
			"tenant_id": tenant,
			"period": period,
			"channels": [{
				"channel": r["channel"],
				"bookings": r["bookings"],
				"revenue": r["revenue"],
				"net_revenue": r["net_revenue"],
				"commission_pct": r["commission_pct"],
				"revenue_share": round(r["revenue"] / total_rev * 100, 2) if total_rev else 0.0,
			} for r in reports],
			"total_revenue": round(total_rev, 2),
			"total_commission": round(sum(r["commission"] for r in reports), 2),
			"generated_at": _now(),
		}

	# ── Executive Dashboard ───────────────────────────────────────────────────

	async def executive_dashboard(self, date_from: str, date_to: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Generate an executive-level analytics dashboard."""
		tenant = self._tenant(tenant_id)
		snaps = [s for s in self.kpi_snapshots.values() if s["tenant_id"] == tenant and date_from <= s["date"] <= date_to]
		surveys = [s for s in self.satisfaction_surveys.values() if s["tenant_id"] == tenant and date_from <= s["created_at"][:10] <= date_to]
		avg_occupancy = round(sum(s["occupancy_rate"] for s in snaps) / len(snaps), 2) if snaps else 0.0
		avg_revpar = round(sum(s["revpar"] for s in snaps) / len(snaps), 2) if snaps else 0.0
		avg_adr = round(sum(s["adr"] for s in snaps) / len(snaps), 2) if snaps else 0.0
		total_revenue = round(sum(s["total_revenue"] for s in snaps), 2)
		nps = 0.0
		if surveys:
			promoters = sum(1 for s in surveys if s["nps_category"] == "promoter")
			detractors = sum(1 for s in surveys if s["nps_category"] == "detractor")
			nps = round((promoters - detractors) / len(surveys) * 100, 1)
		return {
			"tenant_id": tenant,
			"date_from": date_from,
			"date_to": date_to,
			"days": len(snaps),
			"avg_occupancy_rate": avg_occupancy,
			"avg_adr": avg_adr,
			"avg_revpar": avg_revpar,
			"total_revenue": total_revenue,
			"guest_satisfaction_surveys": len(surveys),
			"nps_score": nps,
			"kpi_data_points": len(snaps),
			"segment_reports": len(self.segment_reports),
			"pace_reports": len(self.pace_reports),
			"generated_at": _now(),
		}

	async def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		return {
			"tenant_id": tenant,
			"kpi_snapshots": sum(1 for s in self.kpi_snapshots.values() if s["tenant_id"] == tenant),
			"satisfaction_surveys": sum(1 for s in self.satisfaction_surveys.values() if s["tenant_id"] == tenant),
			"segment_reports": sum(1 for r in self.segment_reports.values() if r["tenant_id"] == tenant),
			"pace_reports": sum(1 for r in self.pace_reports.values() if r["tenant_id"] == tenant),
			"benchmarks": sum(1 for b in self.benchmarks.values() if b["tenant_id"] == tenant),
			"generated_at": _now(),
		}
