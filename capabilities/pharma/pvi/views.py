"""View models for APG Pharma Pharmacovigilance screens."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import PharmacovigilanceService


def dashboard_model(service: PharmacovigilanceService, tenant_id: str = "default") -> dict[str, Any]:
	"""Compose the PVI dashboard view model."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Pharmacovigilance",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
		"pending_follow_ups": len(service.list_follow_ups(tenant_id, pending_only=True)),
	}


def case_queue_model(service: PharmacovigilanceService, tenant_id: str = "default",
					status: str | None = None, serious_only: bool = False) -> dict[str, Any]:
	"""Case queue view."""
	cases = service.list_cases(tenant_id, status=status, serious_only=serious_only)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Case Queue",
		"tenant_id": tenant_id,
		"status_filter": status,
		"serious_only": serious_only,
		"count": len(cases),
		"serious_count": sum(1 for c in cases if c.serious),
		"items": [c.model_dump() for c in cases],
		"supported_sources": contract["configuration"]["adverse_events"]["supported_sources"],
	}


def case_detail_model(service: PharmacovigilanceService, case_id: str,
					tenant_id: str = "default") -> dict[str, Any]:
	"""Detail view for a single case with submissions and follow-ups."""
	case = service.get_case(case_id, tenant_id)
	submissions = service.list_icsr_submissions(tenant_id, case_id=case_id)
	follow_ups = service.list_follow_ups(tenant_id, case_id=case_id)
	return {
		"title": f"Case: {case.case_number}",
		"tenant_id": tenant_id,
		"case": case.model_dump(),
		"submissions": [s.model_dump() for s in submissions],
		"follow_ups": [f.model_dump() for f in follow_ups],
		"pending_follow_up_count": sum(1 for f in follow_ups if f.status == "requested"),
	}


def follow_up_queue_model(service: PharmacovigilanceService, tenant_id: str = "default",
						pending_only: bool = True) -> dict[str, Any]:
	"""Follow-up queue view."""
	follow_ups = service.list_follow_ups(tenant_id, pending_only=pending_only)
	return {
		"title": "Follow-Up Queue",
		"tenant_id": tenant_id,
		"pending_only": pending_only,
		"count": len(follow_ups),
		"items": [f.model_dump() for f in follow_ups],
	}


def signal_management_model(service: PharmacovigilanceService, tenant_id: str = "default",
							product_id: str | None = None) -> dict[str, Any]:
	"""Signal management view."""
	signals = service.list_signals(tenant_id, product_id=product_id)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Signal Management",
		"tenant_id": tenant_id,
		"product_id": product_id,
		"count": len(signals),
		"open_count": sum(1 for s in signals if s.status not in ("closed",)),
		"items": [s.model_dump() for s in signals],
		"supported_types": contract["configuration"]["signals"]["supported_types"],
	}


def literature_screening_model(service: PharmacovigilanceService, tenant_id: str = "default",
								relevant_only: bool = False) -> dict[str, Any]:
	"""Literature screening view."""
	records = service.list_literature(tenant_id, relevant_only=relevant_only)
	return {
		"title": "Literature Screening",
		"tenant_id": tenant_id,
		"relevant_only": relevant_only,
		"count": len(records),
		"relevant_count": sum(1 for r in records if r.relevant),
		"items": [r.model_dump() for r in records],
	}


def psur_workbench_model(service: PharmacovigilanceService, tenant_id: str = "default",
						product_id: str | None = None) -> dict[str, Any]:
	"""PSUR/PBRER workbench view."""
	reports = service.list_psur_reports(tenant_id, product_id=product_id)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Periodic Safety Reports",
		"tenant_id": tenant_id,
		"product_id": product_id,
		"count": len(reports),
		"items": [r.model_dump() for r in reports],
		"supported_types": contract["configuration"]["psur"]["supported_types"],
	}


def regulatory_reporting_model(service: PharmacovigilanceService,
								tenant_id: str = "default") -> dict[str, Any]:
	"""Regulatory reporting console view."""
	submissions = service.list_icsr_submissions(tenant_id)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Regulatory Reporting",
		"tenant_id": tenant_id,
		"submission_count": len(submissions),
		"pending_count": sum(1 for s in submissions if s.status == "pending"),
		"items": [s.model_dump() for s in submissions],
		"supported_databases": contract["configuration"]["regulatory_reporting"]["supported_databases"],
		"reporting_timelines": contract["configuration"]["regulatory_reporting"]["supported_timelines"],
	}
