"""View model builders for APG Clinical Analytics screens."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import ClinicalAnalyticsService


def dashboard_view_model(service: ClinicalAnalyticsService, tenant_id: str = "default") -> dict[str, Any]:
	"""Build the executive dashboard view payload (sync wrapper for template engines)."""
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		summary = loop.run_until_complete(service.dashboard_summary(tenant_id))
		contract = get_capability_contract(tenant_id)
		return {
			"title": "Clinical Analytics",
			"tenant_id": tenant_id,
			"summary": summary,
			"theme": contract["theme"],
			"routes": contract["ui"]["routes"],
		}
	finally:
		loop.close()


def cohort_list_view_model(service: ClinicalAnalyticsService, tenant_id: str, segment: str | None = None, status: str | None = None) -> dict[str, Any]:
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		cohorts = loop.run_until_complete(service.list_cohorts(tenant_id, segment=segment, status=status))
		return {
			"title": "Patient Cohorts",
			"tenant_id": tenant_id,
			"cohorts": [c.model_dump() for c in cohorts],
			"filter": {"segment": segment, "status": status},
		}
	finally:
		loop.close()


def cohort_detail_view_model(service: ClinicalAnalyticsService, tenant_id: str, cohort_id: str) -> dict[str, Any]:
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		cohort = loop.run_until_complete(service.get_cohort(tenant_id, cohort_id))
		if cohort is None:
			return {"error": "cohort_not_found", "cohort_id": cohort_id}
		metrics = loop.run_until_complete(service.list_metrics(tenant_id, cohort_id=cohort_id))
		return {
			"title": f"Cohort: {cohort.name}",
			"tenant_id": tenant_id,
			"cohort": cohort.model_dump(),
			"metrics": [m.model_dump() for m in metrics],
		}
	finally:
		loop.close()


def metric_list_view_model(service: ClinicalAnalyticsService, tenant_id: str, metric_type: str | None = None) -> dict[str, Any]:
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		metrics = loop.run_until_complete(service.list_metrics(tenant_id, metric_type=metric_type))
		return {
			"title": "Clinical Metrics",
			"tenant_id": tenant_id,
			"metrics": [m.model_dump() for m in metrics],
			"filter": {"metric_type": metric_type},
		}
	finally:
		loop.close()


def prediction_model_list_view_model(service: ClinicalAnalyticsService, tenant_id: str) -> dict[str, Any]:
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		models = loop.run_until_complete(service.list_prediction_models(tenant_id))
		return {
			"title": "Prediction Models",
			"tenant_id": tenant_id,
			"models": [m.model_dump() for m in models],
		}
	finally:
		loop.close()


def quality_indicator_view_model(service: ClinicalAnalyticsService, tenant_id: str, period: str | None = None) -> dict[str, Any]:
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		indicators = loop.run_until_complete(service.list_quality_indicators(tenant_id, period=period))
		below = [qi for qi in indicators if qi.performance_status == "below_target"]
		return {
			"title": "Quality Indicators",
			"tenant_id": tenant_id,
			"indicators": [qi.model_dump() for qi in indicators],
			"below_target_count": len(below),
			"filter": {"period": period},
		}
	finally:
		loop.close()


def care_gap_view_model(service: ClinicalAnalyticsService, tenant_id: str, patient_id: str | None = None, severity: str | None = None) -> dict[str, Any]:
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		gaps = loop.run_until_complete(service.list_care_gaps(tenant_id, patient_id=patient_id, severity=severity, status="open"))
		return {
			"title": "Care Gaps",
			"tenant_id": tenant_id,
			"gaps": [g.model_dump() for g in gaps],
			"filter": {"patient_id": patient_id, "severity": severity},
		}
	finally:
		loop.close()


def report_list_view_model(service: ClinicalAnalyticsService, tenant_id: str, report_type: str | None = None) -> dict[str, Any]:
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		reports = loop.run_until_complete(service.list_reports(tenant_id, report_type=report_type))
		return {
			"title": "Analytics Reports",
			"tenant_id": tenant_id,
			"reports": [r.model_dump() for r in reports],
			"filter": {"report_type": report_type},
		}
	finally:
		loop.close()
