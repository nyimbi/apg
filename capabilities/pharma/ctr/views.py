"""View models for APG Pharma Clinical Trials Management screens."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import ClinicalTrialsService


def dashboard_model(service: ClinicalTrialsService, tenant_id: str = "default") -> dict[str, Any]:
	"""Compose the CTR dashboard view model."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Clinical Trials Management",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
		"streaming": contract["streaming"],
	}


def trial_registry_model(service: ClinicalTrialsService, tenant_id: str = "default",
						phase: str | None = None) -> dict[str, Any]:
	"""List view for clinical trials."""
	trials = service.list_trials(tenant_id, phase=phase)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Trial Registry",
		"tenant_id": tenant_id,
		"phase_filter": phase,
		"count": len(trials),
		"items": [t.model_dump() for t in trials],
		"supported_phases": contract["configuration"]["trials"]["supported_phases"],
	}


def trial_detail_model(service: ClinicalTrialsService, trial_id: str,
					tenant_id: str = "default") -> dict[str, Any]:
	"""Detail view for a single trial."""
	trial = service.get_trial(trial_id, tenant_id)
	sites = service.list_sites(tenant_id, trial_id=trial_id)
	patients = service.list_patients(tenant_id, trial_id=trial_id)
	protocols = service.list_protocols(tenant_id, trial_id=trial_id)
	ae_count = len(service.list_adverse_events(tenant_id, trial_id=trial_id))
	return {
		"title": f"Trial: {trial.trial_number}",
		"tenant_id": tenant_id,
		"trial": trial.model_dump(),
		"site_count": len(sites),
		"patient_count": len(patients),
		"protocol_count": len(protocols),
		"ae_count": ae_count,
		"sites": [s.model_dump() for s in sites],
	}


def protocol_workbench_model(service: ClinicalTrialsService, tenant_id: str = "default",
							trial_id: str | None = None) -> dict[str, Any]:
	"""Protocol workbench view."""
	protocols = service.list_protocols(tenant_id, trial_id=trial_id)
	return {
		"title": "Protocol Workbench",
		"tenant_id": tenant_id,
		"trial_id": trial_id,
		"count": len(protocols),
		"items": [p.model_dump() for p in protocols],
	}


def site_management_model(service: ClinicalTrialsService, tenant_id: str = "default",
						trial_id: str | None = None) -> dict[str, Any]:
	"""Site management view."""
	sites = service.list_sites(tenant_id, trial_id=trial_id)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Site Management",
		"tenant_id": tenant_id,
		"trial_id": trial_id,
		"count": len(sites),
		"items": [s.model_dump() for s in sites],
		"supported_statuses": contract["configuration"]["sites"]["supported_statuses"],
	}


def patient_tracker_model(service: ClinicalTrialsService, tenant_id: str = "default",
						trial_id: str | None = None, site_id: str | None = None) -> dict[str, Any]:
	"""Patient tracker view."""
	patients = service.list_patients(tenant_id, trial_id=trial_id, site_id=site_id)
	return {
		"title": "Patient Tracker",
		"tenant_id": tenant_id,
		"trial_id": trial_id,
		"site_id": site_id,
		"count": len(patients),
		"enrolled": sum(1 for p in patients if p.status == "enrolled"),
		"randomised": sum(1 for p in patients if p.status == "randomised"),
		"completed": sum(1 for p in patients if p.status == "completed"),
		"items": [p.model_dump() for p in patients],
	}


def ae_queue_model(service: ClinicalTrialsService, tenant_id: str = "default",
				trial_id: str | None = None, serious_only: bool = False) -> dict[str, Any]:
	"""Adverse event queue view."""
	events = service.list_adverse_events(tenant_id, trial_id=trial_id, serious_only=serious_only)
	return {
		"title": "Adverse Event Queue",
		"tenant_id": tenant_id,
		"trial_id": trial_id,
		"serious_only": serious_only,
		"count": len(events),
		"items": [ae.model_dump() for ae in events],
	}


def submission_tracker_model(service: ClinicalTrialsService, tenant_id: str = "default",
							trial_id: str | None = None) -> dict[str, Any]:
	"""Regulatory submission tracker view."""
	submissions = service.list_submissions(tenant_id, trial_id=trial_id)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Regulatory Submissions",
		"tenant_id": tenant_id,
		"trial_id": trial_id,
		"count": len(submissions),
		"items": [s.model_dump() for s in submissions],
		"supported_authorities": contract["configuration"]["submissions"]["supported_authorities"],
	}
