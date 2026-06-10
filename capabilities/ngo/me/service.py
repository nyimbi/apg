"""M&E Service — indicator framework, data collection, reporting, impact assessment, learning cycles."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

_log = logging.getLogger(__name__)

CAPABILITY_ID = "ngo_me"

SUPPORTED_INDICATOR_TYPES = {"input", "output", "outcome", "impact", "process"}
SUPPORTED_EVAL_TYPES = {"baseline", "mid_term", "final", "real_time", "ex_post"}
SUPPORTED_RATINGS = {"highly_satisfactory", "satisfactory", "moderately_satisfactory",
					  "moderately_unsatisfactory", "unsatisfactory", "highly_unsatisfactory"}
SUPPORTED_REPORT_PERIODS = {"monthly", "quarterly", "semi_annual", "annual", "ad_hoc"}


class MEService:
	"""Async service for NGO monitoring and evaluation."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self._indicators: dict[str, dict[str, Any]] = {}
		self._data_collections: dict[str, dict[str, Any]] = {}
		self._progress_reports: dict[str, dict[str, Any]] = {}
		self._evaluations: dict[str, dict[str, Any]] = {}
		self._learning_cycles: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	# ── helpers ───────────────────────────────────────────────────────────────

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}-{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _tenant(self) -> str:
		if not self.tenant_id:
			raise PermissionError("tenant_context_required")
		return self.tenant_id

	def _emit(self, event_type: str, record_id: str, record_type: str, details: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": self._id("evt"),
			"tenant_id": self._tenant(),
			"event_type": event_type,
			"record_id": record_id,
			"record_type": record_type,
			"details": details or {},
			"emitted_at": self._now(),
		})

	def _guard_indicator(self, indicator_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		ind = self._indicators.get(indicator_id)
		if not ind or ind["tenant_id"] != tenant:
			raise KeyError(f"indicator_not_found:{indicator_id}")
		return ind

	# ── health / describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"indicator_count": len(self._indicators),
			"data_collection_count": len(self._data_collections),
			"progress_report_count": len(self._progress_reports),
			"evaluation_count": len(self._evaluations),
			"learning_cycle_count": len(self._learning_cycles),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"domain": "ngo",
			"version": "1.0.0",
			"description": "Indicator framework, data collection, progress reporting, impact assessment, learning cycles",
			"indicator_types": list(SUPPORTED_INDICATOR_TYPES),
			"evaluation_types": list(SUPPORTED_EVAL_TYPES),
			"tenant_id": self.tenant_id,
		}

	async def get_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
		tenant = self._tenant()
		events = [e for e in self._audit_events if e["tenant_id"] == tenant]
		return [deepcopy(e) for e in events[-limit:]]

	# ── indicators ────────────────────────────────────────────────────────────

	async def list_indicators(
		self,
		programme_id: str | None = None,
		indicator_type: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(i) for i in self._indicators.values() if i["tenant_id"] == tenant]
		if programme_id:
			items = [i for i in items if i["programme_id"] == programme_id]
		if indicator_type:
			items = [i for i in items if i["indicator_type"] == indicator_type]
		if status:
			items = [i for i in items if i["status"] == status]
		return items

	async def get_indicator(self, indicator_id: str) -> dict[str, Any]:
		return deepcopy(self._guard_indicator(indicator_id))

	async def create_indicator(
		self,
		programme_id: str,
		name: str,
		code: str,
		target_value: float,
		target_date: str,
		indicator_type: str = "output",
		description: str = "",
		unit: str = "",
		baseline_value: float = 0.0,
		baseline_date: str = "",
		disaggregation: list[str] | None = None,
		data_source: str = "",
		collection_method: str = "",
	) -> dict[str, Any]:
		"""Create an indicator in the M&E framework."""
		tenant = self._tenant()
		if not name or not code:
			raise ValueError("name_and_code_required")
		if indicator_type not in SUPPORTED_INDICATOR_TYPES:
			raise ValueError(f"unsupported_indicator_type:{indicator_type}")
		# enforce code uniqueness per programme
		if any(
			i["code"] == code and i["programme_id"] == programme_id and i["tenant_id"] == tenant
			for i in self._indicators.values()
		):
			raise ValueError(f"indicator_code_exists_in_programme:{code}")
		record: dict[str, Any] = {
			"id": self._id("ind"),
			"type": "ngo_indicator",
			"tenant_id": tenant,
			"programme_id": programme_id,
			"name": name,
			"code": code,
			"description": description,
			"indicator_type": indicator_type,
			"unit": unit,
			"baseline_value": baseline_value,
			"baseline_date": baseline_date,
			"target_value": target_value,
			"target_date": target_date,
			"current_value": baseline_value,
			"achievement_pct": 0.0,
			"disaggregation": disaggregation or [],
			"data_source": data_source,
			"collection_method": collection_method,
			"status": "active",
			"created_at": self._now(),
			"updated_at": None,
		}
		self._indicators[record["id"]] = record
		self._emit("indicator_created", record["id"], "ngo_indicator",
				   {"programme_id": programme_id, "code": code, "type": indicator_type})
		_log.info("Indicator created: %s (%s) for programme %s", record["id"], code, programme_id)
		return deepcopy(record)

	async def update_indicator(self, indicator_id: str, **kwargs: Any) -> dict[str, Any]:
		ind = self._guard_indicator(indicator_id)
		allowed = {"name", "description", "target_value", "target_date", "status",
				   "data_source", "collection_method", "disaggregation"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				ind[k] = v
		# recompute achievement pct if target changed
		if ind["target_value"]:
			ind["achievement_pct"] = round(ind["current_value"] / ind["target_value"] * 100, 2)
		ind["updated_at"] = self._now()
		self._emit("indicator_updated", indicator_id, "ngo_indicator", kwargs)
		return deepcopy(ind)

	async def delete_indicator(self, indicator_id: str) -> dict[str, Any]:
		ind = self._guard_indicator(indicator_id)
		removed = self._indicators.pop(indicator_id)
		self._emit("indicator_deleted", indicator_id, "ngo_indicator")
		return deepcopy(removed)

	async def set_indicator_baseline(
		self, indicator_id: str, baseline_value: float, baseline_date: str
	) -> dict[str, Any]:
		"""Set or update the baseline value for an indicator."""
		ind = self._guard_indicator(indicator_id)
		ind["baseline_value"] = baseline_value
		ind["baseline_date"] = baseline_date
		ind["updated_at"] = self._now()
		self._emit("indicator_baseline_set", indicator_id, "ngo_indicator",
				   {"baseline_value": baseline_value, "baseline_date": baseline_date})
		return deepcopy(ind)

	# ── data collection ───────────────────────────────────────────────────────

	async def list_data_collections(
		self, indicator_id: str | None = None, programme_id: str | None = None
	) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(dc) for dc in self._data_collections.values() if dc["tenant_id"] == tenant]
		if indicator_id:
			items = [dc for dc in items if dc["indicator_id"] == indicator_id]
		if programme_id:
			items = [dc for dc in items if dc["programme_id"] == programme_id]
		return items

	async def collect_data(
		self,
		indicator_id: str,
		programme_id: str,
		value: float,
		collection_date: str,
		collected_by: str,
		period: str = "",
		disaggregation_values: dict[str, Any] | None = None,
		notes: str = "",
	) -> dict[str, Any]:
		"""Submit a data point for an indicator."""
		ind = self._guard_indicator(indicator_id)
		if not collected_by:
			raise ValueError("collected_by_required")
		record: dict[str, Any] = {
			"id": self._id("dc"),
			"type": "ngo_data_collection",
			"tenant_id": self._tenant(),
			"indicator_id": indicator_id,
			"programme_id": programme_id,
			"value": value,
			"collection_date": collection_date,
			"collected_by": collected_by,
			"period": period,
			"disaggregation_values": disaggregation_values or {},
			"notes": notes,
			"verified": False,
			"created_at": self._now(),
		}
		self._data_collections[record["id"]] = record
		# update indicator current value to latest data point
		ind["current_value"] = value
		if ind["target_value"]:
			ind["achievement_pct"] = round(value / ind["target_value"] * 100, 2)
		ind["updated_at"] = self._now()
		self._emit("data_collected", record["id"], "ngo_data_collection",
				   {"indicator_id": indicator_id, "value": value})
		return deepcopy(record)

	async def verify_data_collection(self, collection_id: str, verified_by: str) -> dict[str, Any]:
		"""Mark a data collection record as verified."""
		tenant = self._tenant()
		dc = self._data_collections.get(collection_id)
		if not dc or dc["tenant_id"] != tenant:
			raise KeyError(f"data_collection_not_found:{collection_id}")
		dc["verified"] = True
		dc["verified_by"] = verified_by
		dc["verified_at"] = self._now()
		self._emit("data_collection_verified", collection_id, "ngo_data_collection",
				   {"verified_by": verified_by})
		return deepcopy(dc)

	async def delete_data_collection(self, collection_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		dc = self._data_collections.get(collection_id)
		if not dc or dc["tenant_id"] != tenant:
			raise KeyError(f"data_collection_not_found:{collection_id}")
		removed = self._data_collections.pop(collection_id)
		self._emit("data_collection_deleted", collection_id, "ngo_data_collection")
		return deepcopy(removed)

	# ── progress reports ──────────────────────────────────────────────────────

	async def list_progress_reports(self, programme_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(r) for r in self._progress_reports.values() if r["tenant_id"] == tenant]
		if programme_id:
			items = [r for r in items if r["programme_id"] == programme_id]
		return items

	async def get_progress_report(self, report_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		r = self._progress_reports.get(report_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"progress_report_not_found:{report_id}")
		return deepcopy(r)

	async def create_progress_report(
		self,
		programme_id: str,
		report_period: str,
		period_start: str,
		period_end: str,
		prepared_by: str,
		narrative: str = "",
		key_achievements: list[str] | None = None,
		challenges: list[str] | None = None,
		lessons_learned: list[str] | None = None,
	) -> dict[str, Any]:
		"""Create a progress report with automatic indicator snapshots."""
		tenant = self._tenant()
		if report_period not in SUPPORTED_REPORT_PERIODS:
			raise ValueError(f"unsupported_report_period:{report_period}")
		# snapshot current indicator values for this programme
		indicators = [i for i in self._indicators.values()
					  if i["programme_id"] == programme_id and i["tenant_id"] == tenant]
		snapshots = [
			{
				"indicator_id": i["id"],
				"code": i["code"],
				"name": i["name"],
				"target_value": i["target_value"],
				"current_value": i["current_value"],
				"achievement_pct": i["achievement_pct"],
				"unit": i["unit"],
			}
			for i in indicators
		]
		record: dict[str, Any] = {
			"id": self._id("pr"),
			"type": "ngo_progress_report",
			"tenant_id": tenant,
			"programme_id": programme_id,
			"report_period": report_period,
			"period_start": period_start,
			"period_end": period_end,
			"prepared_by": prepared_by,
			"narrative": narrative,
			"key_achievements": key_achievements or [],
			"challenges": challenges or [],
			"lessons_learned": lessons_learned or [],
			"indicator_snapshots": snapshots,
			"status": "draft",
			"created_at": self._now(),
		}
		self._progress_reports[record["id"]] = record
		self._emit("progress_report_created", record["id"], "ngo_progress_report",
				   {"programme_id": programme_id, "period": report_period})
		return deepcopy(record)

	async def submit_progress_report(self, report_id: str) -> dict[str, Any]:
		"""Submit a draft progress report."""
		tenant = self._tenant()
		r = self._progress_reports.get(report_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"progress_report_not_found:{report_id}")
		if r["status"] != "draft":
			raise ValueError(f"cannot_submit_{r['status']}_report")
		r["status"] = "submitted"
		r["submitted_at"] = self._now()
		self._emit("progress_report_submitted", report_id, "ngo_progress_report")
		return deepcopy(r)

	async def approve_progress_report(self, report_id: str, approved_by: str) -> dict[str, Any]:
		"""Approve a submitted progress report."""
		tenant = self._tenant()
		r = self._progress_reports.get(report_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"progress_report_not_found:{report_id}")
		r["status"] = "approved"
		r["approved_by"] = approved_by
		r["approved_at"] = self._now()
		self._emit("progress_report_approved", report_id, "ngo_progress_report",
				   {"approved_by": approved_by})
		return deepcopy(r)

	# ── evaluations ───────────────────────────────────────────────────────────

	async def list_evaluations(self, programme_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(e) for e in self._evaluations.values() if e["tenant_id"] == tenant]
		if programme_id:
			items = [e for e in items if e["programme_id"] == programme_id]
		return items

	async def get_evaluation(self, evaluation_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		ev = self._evaluations.get(evaluation_id)
		if not ev or ev["tenant_id"] != tenant:
			raise KeyError(f"evaluation_not_found:{evaluation_id}")
		return deepcopy(ev)

	async def create_evaluation(
		self,
		programme_id: str,
		evaluator: str,
		evaluation_date: str,
		evaluation_type: str = "mid_term",
		scope: str = "",
		methodology: str = "",
		findings: str = "",
		recommendations: str = "",
		rating: str = "satisfactory",
	) -> dict[str, Any]:
		"""Record a programme evaluation."""
		if evaluation_type not in SUPPORTED_EVAL_TYPES:
			raise ValueError(f"unsupported_evaluation_type:{evaluation_type}")
		if rating not in SUPPORTED_RATINGS:
			raise ValueError(f"unsupported_rating:{rating}")
		record: dict[str, Any] = {
			"id": self._id("eval"),
			"type": "ngo_evaluation",
			"tenant_id": self._tenant(),
			"programme_id": programme_id,
			"evaluation_type": evaluation_type,
			"evaluator": evaluator,
			"evaluation_date": evaluation_date,
			"scope": scope,
			"methodology": methodology,
			"findings": findings,
			"recommendations": recommendations,
			"rating": rating,
			"status": "completed",
			"created_at": self._now(),
		}
		self._evaluations[record["id"]] = record
		self._emit("evaluation_created", record["id"], "ngo_evaluation",
				   {"programme_id": programme_id, "type": evaluation_type, "rating": rating})
		_log.info("Evaluation created: %s type=%s rating=%s", record["id"], evaluation_type, rating)
		return deepcopy(record)

	async def update_evaluation(self, evaluation_id: str, **kwargs: Any) -> dict[str, Any]:
		tenant = self._tenant()
		ev = self._evaluations.get(evaluation_id)
		if not ev or ev["tenant_id"] != tenant:
			raise KeyError(f"evaluation_not_found:{evaluation_id}")
		allowed = {"findings", "recommendations", "rating", "methodology", "scope"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				ev[k] = v
		self._emit("evaluation_updated", evaluation_id, "ngo_evaluation", kwargs)
		return deepcopy(ev)

	# ── learning cycles ───────────────────────────────────────────────────────

	async def list_learning_cycles(self, programme_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(lc) for lc in self._learning_cycles.values() if lc["tenant_id"] == tenant]
		if programme_id:
			items = [lc for lc in items if lc["programme_id"] == programme_id]
		return items

	async def get_learning_cycle(self, cycle_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		lc = self._learning_cycles.get(cycle_id)
		if not lc or lc["tenant_id"] != tenant:
			raise KeyError(f"learning_cycle_not_found:{cycle_id}")
		return deepcopy(lc)

	async def create_learning_cycle(
		self,
		programme_id: str,
		cycle_name: str,
		start_date: str,
		end_date: str,
		facilitator: str,
		learning_questions: list[str] | None = None,
		notes: str = "",
	) -> dict[str, Any]:
		"""Initiate a learning cycle for a programme."""
		if not cycle_name:
			raise ValueError("cycle_name_required")
		record: dict[str, Any] = {
			"id": self._id("lc"),
			"type": "ngo_learning_cycle",
			"tenant_id": self._tenant(),
			"programme_id": programme_id,
			"cycle_name": cycle_name,
			"start_date": start_date,
			"end_date": end_date,
			"facilitator": facilitator,
			"learning_questions": learning_questions or [],
			"findings": [],
			"action_points": [],
			"notes": notes,
			"status": "active",
			"created_at": self._now(),
		}
		self._learning_cycles[record["id"]] = record
		self._emit("learning_cycle_created", record["id"], "ngo_learning_cycle",
				   {"programme_id": programme_id, "name": cycle_name})
		return deepcopy(record)

	async def add_learning_findings(
		self, cycle_id: str, findings: list[str], action_points: list[str] | None = None
	) -> dict[str, Any]:
		"""Add findings and action points to a learning cycle."""
		tenant = self._tenant()
		lc = self._learning_cycles.get(cycle_id)
		if not lc or lc["tenant_id"] != tenant:
			raise KeyError(f"learning_cycle_not_found:{cycle_id}")
		lc["findings"].extend(findings)
		if action_points:
			lc["action_points"].extend(action_points)
		self._emit("learning_findings_added", cycle_id, "ngo_learning_cycle",
				   {"finding_count": len(findings)})
		return deepcopy(lc)

	async def close_learning_cycle(self, cycle_id: str) -> dict[str, Any]:
		"""Close a completed learning cycle."""
		tenant = self._tenant()
		lc = self._learning_cycles.get(cycle_id)
		if not lc or lc["tenant_id"] != tenant:
			raise KeyError(f"learning_cycle_not_found:{cycle_id}")
		lc["status"] = "closed"
		lc["closed_at"] = self._now()
		self._emit("learning_cycle_closed", cycle_id, "ngo_learning_cycle")
		return deepcopy(lc)

	async def update_learning_cycle(self, cycle_id: str, **kwargs: Any) -> dict[str, Any]:
		tenant = self._tenant()
		lc = self._learning_cycles.get(cycle_id)
		if not lc or lc["tenant_id"] != tenant:
			raise KeyError(f"learning_cycle_not_found:{cycle_id}")
		allowed = {"cycle_name", "end_date", "facilitator", "notes", "status"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				lc[k] = v
		self._emit("learning_cycle_updated", cycle_id, "ngo_learning_cycle", kwargs)
		return deepcopy(lc)

	# ── analytics / impact ─────────────────────────────────────────────────────

	async def indicator_performance_dashboard(self, programme_id: str) -> dict[str, Any]:
		"""Dashboard of all indicator performance for a programme."""
		tenant = self._tenant()
		indicators = [i for i in self._indicators.values()
					  if i["programme_id"] == programme_id and i["tenant_id"] == tenant]
		on_track = [i for i in indicators if i["achievement_pct"] >= 75]
		at_risk = [i for i in indicators if 50 <= i["achievement_pct"] < 75]
		off_track = [i for i in indicators if i["achievement_pct"] < 50]
		return {
			"programme_id": programme_id,
			"total_indicators": len(indicators),
			"on_track": len(on_track),
			"at_risk": len(at_risk),
			"off_track": len(off_track),
			"avg_achievement_pct": round(
				sum(i["achievement_pct"] for i in indicators) / len(indicators), 2
			) if indicators else 0.0,
			"indicators": [
				{"id": i["id"], "code": i["code"], "name": i["name"],
				 "achievement_pct": i["achievement_pct"], "current_value": i["current_value"],
				 "target_value": i["target_value"], "unit": i["unit"]}
				for i in indicators
			],
			"generated_at": self._now(),
		}

	async def impact_assessment_summary(self, programme_id: str) -> dict[str, Any]:
		"""Summarise impact-level indicators for a programme."""
		tenant = self._tenant()
		impact_indicators = [
			i for i in self._indicators.values()
			if i["programme_id"] == programme_id
			and i["tenant_id"] == tenant
			and i["indicator_type"] == "impact"
		]
		evaluations = [
			e for e in self._evaluations.values()
			if e["programme_id"] == programme_id and e["tenant_id"] == tenant
		]
		return {
			"programme_id": programme_id,
			"impact_indicators": len(impact_indicators),
			"avg_impact_achievement": round(
				sum(i["achievement_pct"] for i in impact_indicators) / len(impact_indicators), 2
			) if impact_indicators else 0.0,
			"evaluation_count": len(evaluations),
			"latest_rating": evaluations[-1]["rating"] if evaluations else None,
			"generated_at": self._now(),
		}

	async def bulk_collect_data(self, data_points: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-submit multiple data collection points."""
		tasks = [
			self.collect_data(
				indicator_id=dp["indicator_id"],
				programme_id=dp["programme_id"],
				value=float(dp["value"]),
				collection_date=dp["collection_date"],
				collected_by=dp["collected_by"],
				period=dp.get("period", ""),
				disaggregation_values=dp.get("disaggregation_values", {}),
				notes=dp.get("notes", ""),
			)
			for dp in data_points
		]
		outcomes = await asyncio.gather(*tasks, return_exceptions=True)
		results, errors = [], []
		for dp, outcome in zip(data_points, outcomes):
			if isinstance(outcome, Exception):
				errors.append({"input": dp, "error": str(outcome)})
			else:
				results.append(outcome)
		return {"collected": len(results), "failed": len(errors), "records": results, "errors": errors}

	async def trend_analysis(self, indicator_id: str) -> dict[str, Any]:
		"""Return time-series trend for an indicator."""
		ind = self._guard_indicator(indicator_id)
		tenant = self._tenant()
		data_points = sorted(
			[dc for dc in self._data_collections.values()
			 if dc["indicator_id"] == indicator_id and dc["tenant_id"] == tenant],
			key=lambda x: x["collection_date"]
		)
		values = [{"date": dc["collection_date"], "value": dc["value"], "period": dc["period"]}
				  for dc in data_points]
		trend = "insufficient_data"
		if len(values) >= 2:
			delta = values[-1]["value"] - values[0]["value"]
			trend = "increasing" if delta > 0 else ("decreasing" if delta < 0 else "flat")
		return {
			"indicator_id": indicator_id,
			"code": ind["code"],
			"name": ind["name"],
			"target_value": ind["target_value"],
			"baseline_value": ind["baseline_value"],
			"current_value": ind["current_value"],
			"data_points": values,
			"trend": trend,
			"generated_at": self._now(),
		}
