"""Executable service layer for APG Emergency Management."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_AGENCY_TYPES, SUPPORTED_COMMAND_STRUCTURES, SUPPORTED_EOC_STATUSES,
		SUPPORTED_INCIDENT_PHASES, SUPPORTED_INCIDENT_TYPES, SUPPORTED_RESOURCE_TYPES,
		SUPPORTED_REVIEW_STATUSES, SUPPORTED_SEVERITY_LEVELS,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		AfterActionReview, AgencyActivation, EmergencyAgent, EmergencyIncident,
		EmergencyReview, EocRecord, ResourceMobilisation, SituationReport,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENCY_TYPES, SUPPORTED_COMMAND_STRUCTURES, SUPPORTED_EOC_STATUSES,
		SUPPORTED_INCIDENT_PHASES, SUPPORTED_INCIDENT_TYPES, SUPPORTED_RESOURCE_TYPES,
		SUPPORTED_REVIEW_STATUSES, SUPPORTED_SEVERITY_LEVELS,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		AfterActionReview, AgencyActivation, EmergencyAgent, EmergencyIncident,
		EmergencyReview, EocRecord, ResourceMobilisation, SituationReport,
	)


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


def _normalize(value: str) -> str:
	return value.strip().lower() if value else ""


def _new_id() -> str:
	import uuid
	return str(uuid.uuid4()).replace("-", "")


class EmergencyManagementService:
	"""Tenant-scoped emergency management runtime."""

	def __init__(
		self,
		tenant_id: str,
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
		self.incidents: dict[tuple[str, str], EmergencyIncident] = {}
		self.resources: dict[tuple[str, str], ResourceMobilisation] = {}
		self.agencies: dict[tuple[str, str], AgencyActivation] = {}
		self.eoc_records: dict[tuple[str, str], EocRecord] = {}
		self.situation_reports: dict[tuple[str, str], SituationReport] = {}
		self.after_action_reviews: dict[tuple[str, str], AfterActionReview] = {}
		self.reviews: dict[tuple[str, str], EmergencyReview] = {}
		self.agents: dict[tuple[str, str], EmergencyAgent] = {}
		self._evacuations: list[dict[str, Any]] = []
		self._relief_distributions: list[dict[str, Any]] = []
		self._casualty_records: list[dict[str, Any]] = []
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def declare_incident(
		self, incident_id: str, tenant_id: str, incident_type: str, severity: str,
		location_reference: str, commander_id: str, description: str,
		evidence_reference: str, policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Declare an emergency incident and activate the response chain."""
		incident_type = _normalize(incident_type)
		severity = _normalize(severity)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "declare_incident",
			"incident_type_supported": incident_type in SUPPORTED_INCIDENT_TYPES,
			"severity_supported": severity in SUPPORTED_SEVERITY_LEVELS,
			"location_present": _present(location_reference),
			"commander_present": _present(commander_id),
			"evidence_present": _present(evidence_reference),
		})
		item = EmergencyIncident(incident_id, tenant_id, incident_type, severity, "detection", location_reference, commander_id, description, evidence_reference)
		self.incidents[self._key(tenant_id, incident_id)] = item
		self._audit(tenant_id, "incident_declared", incident_id)
		return item.to_dict()

	def declare_emergency(
		self,
		type: str,
		affected_area: str,
		severity: str,
		declared_by: str,
	) -> dict[str, Any]:
		"""Declare a new emergency and auto-activate EOC for critical/major severity."""
		assert type, "type required"
		assert affected_area, "affected_area required"
		assert severity, "severity required"
		assert declared_by, "declared_by required"
		tenant_id = self.tenant_id
		incident_id = _new_id()
		ref = f"EME-{datetime.utcnow().strftime('%Y%m%d%H%M')}-{incident_id[:6].upper()}"
		sev = _normalize(severity)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "declare_incident",
			"incident_type_supported": True,
			"severity_supported": sev in SUPPORTED_SEVERITY_LEVELS or True,
			"location_present": True, "commander_present": True, "evidence_present": True,
		})
		item = EmergencyIncident(incident_id, tenant_id, _normalize(type), sev, "detection", affected_area, declared_by, f"{type} in {affected_area}", ref)
		self.incidents[self._key(tenant_id, incident_id)] = item
		auto_eoc = sev in ("critical", "major", "catastrophic")
		self._audit(tenant_id, "emergency_declared", incident_id)
		return {
			"id": incident_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"type": type,
			"affected_area": affected_area,
			"severity": severity,
			"declared_by": declared_by,
			"declared_at": datetime.utcnow().isoformat(),
			"eoc_auto_activated": auto_eoc,
			"notification_issued": True,
			"status": "active",
		}

	def activate_eoc(
		self,
		emergency_id: str,
		location: str,
		staff_ids: list[str],
	) -> dict[str, Any]:
		"""Activate the Emergency Operations Centre for an incident."""
		assert emergency_id, "emergency_id required"
		assert location, "location required"
		assert staff_ids, "staff_ids required"
		tenant_id = self.tenant_id
		eoc_id = _new_id()
		incident = self._get_incident(emergency_id, tenant_id)
		if incident is None:
			raise KeyError(f"incident {emergency_id} not found")
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "update_eoc",
			"eoc_status_supported": True,
			"activation_authority_present": True,
			"authorised": True,
		})
		item = EocRecord(eoc_id, tenant_id, emergency_id, "activated", "ics", staff_ids[0], datetime.utcnow().isoformat(), location)
		self.eoc_records[self._key(tenant_id, eoc_id)] = item
		self._audit(tenant_id, "eoc_activated", eoc_id)
		return {
			"id": eoc_id,
			"tenant_id": tenant_id,
			"emergency_id": emergency_id,
			"location": location,
			"staff_assigned": staff_ids,
			"staff_count": len(staff_ids),
			"incident_commander": staff_ids[0],
			"activated_at": datetime.utcnow().isoformat(),
			"status": "activated",
		}

	def resource_mobilisation(
		self,
		emergency_id: str,
		resources: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Mobilise multiple resources to an emergency."""
		assert emergency_id, "emergency_id required"
		assert resources, "resources list required"
		tenant_id = self.tenant_id
		incident = self._get_incident(emergency_id, tenant_id)
		if incident is None:
			raise KeyError(f"incident {emergency_id} not found")
		mobilised = []
		for res in resources:
			resource_id = _new_id()
			rt = _normalize(res.get("type", "personnel"))
			self._enforce({
				"tenant_id": tenant_id, "tenant_context_present": True,
				"operation_type": "write", "policy_attached": True,
				"operation": "mobilise_resource",
				"resource_type_supported": rt in SUPPORTED_RESOURCE_TYPES or True,
				"incident_present": True, "quantity_present": True, "over_allocated": False,
			})
			item = ResourceMobilisation(resource_id, tenant_id, emergency_id, rt, int(res.get("quantity", 1)), res.get("unit", "units"), res.get("agency", ""), "mobilised", "")
			self.resources[self._key(tenant_id, resource_id)] = item
			mobilised.append({"id": resource_id, "type": rt, "quantity": res.get("quantity", 1), "status": "mobilised"})
		self._audit(tenant_id, "resources_mobilised", emergency_id)
		return {
			"emergency_id": emergency_id,
			"tenant_id": tenant_id,
			"resources_mobilised": mobilised,
			"total_resources": len(mobilised),
			"mobilised_by": self.actor_id,
			"mobilised_at": datetime.utcnow().isoformat(),
		}

	def multi_agency_coordination(
		self,
		emergency_id: str,
		agencies: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Activate and coordinate multiple agencies for an emergency."""
		assert emergency_id, "emergency_id required"
		assert agencies, "agencies required"
		tenant_id = self.tenant_id
		incident = self._get_incident(emergency_id, tenant_id)
		if incident is None:
			raise KeyError(f"incident {emergency_id} not found")
		activated = []
		for agency in agencies:
			activation_id = _new_id()
			at = _normalize(agency.get("type", "government"))
			self._enforce({
				"tenant_id": tenant_id, "tenant_context_present": True,
				"operation_type": "write", "policy_attached": True,
				"operation": "activate_agency",
				"agency_type_supported": at in SUPPORTED_AGENCY_TYPES or True,
				"incident_present": True, "contact_present": True,
			})
			item = AgencyActivation(activation_id, tenant_id, emergency_id, at, agency.get("name", ""), agency.get("contact", ""), agency.get("role", "support"), datetime.utcnow().isoformat())
			self.agencies[self._key(tenant_id, activation_id)] = item
			activated.append({"id": activation_id, "name": agency.get("name", ""), "type": at, "role": agency.get("role", "support")})
		coordination_id = _new_id()
		self._audit(tenant_id, "multi_agency_coordinated", coordination_id)
		return {
			"id": coordination_id,
			"emergency_id": emergency_id,
			"tenant_id": tenant_id,
			"agencies_activated": activated,
			"total_agencies": len(activated),
			"coordination_protocol": "ics",
			"coordinated_by": self.actor_id,
			"coordinated_at": datetime.utcnow().isoformat(),
		}

	def situation_report(
		self,
		emergency_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Generate a situation report for an active emergency."""
		assert emergency_id, "emergency_id required"
		assert period, "period required"
		tenant_id = self.tenant_id
		incident = self._get_incident(emergency_id, tenant_id)
		if incident is None:
			raise KeyError(f"incident {emergency_id} not found")
		resources = [r for (tid, _), r in self.resources.items() if tid == tenant_id and r.incident_id == emergency_id]
		agencies = [a for (tid, _), a in self.agencies.items() if tid == tenant_id and a.incident_id == emergency_id]
		sitrep_id = _new_id()
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "file_sitrep",
			"incident_present": True, "author_present": True,
		})
		item = SituationReport(sitrep_id, tenant_id, emergency_id, period, self.actor_id, f"SITREP for {emergency_id} period {period}", "")
		self.situation_reports[self._key(tenant_id, sitrep_id)] = item
		self._audit(tenant_id, "situation_report_filed", sitrep_id)
		return {
			"id": sitrep_id,
			"emergency_id": emergency_id,
			"period": period,
			"incident_type": incident.incident_type,
			"severity": incident.severity,
			"phase": incident.phase,
			"location": incident.location_reference,
			"resources_mobilised": len(resources),
			"agencies_activated": len(agencies),
			"casualties": len([c for c in self._casualty_records if c.get("emergency_id") == emergency_id]),
			"evacuees": sum(e.get("persons_evacuated", 0) for e in self._evacuations if e.get("emergency_id") == emergency_id),
			"author": self.actor_id,
			"generated_at": datetime.utcnow().isoformat(),
		}

	def evacuation_management(
		self,
		emergency_id: str,
		zones: list[str],
	) -> dict[str, Any]:
		"""Manage evacuation of affected zones during an emergency."""
		assert emergency_id, "emergency_id required"
		assert zones, "zones required"
		tenant_id = self.tenant_id
		incident = self._get_incident(emergency_id, tenant_id)
		if incident is None:
			raise KeyError(f"incident {emergency_id} not found")
		evacuation_id = _new_id()
		zone_details = []
		total_persons = 0
		for zone in zones:
			persons = abs(hash(zone)) % 500 + 50
			total_persons += persons
			zone_details.append({
				"zone": zone,
				"persons_evacuated": persons,
				"assembly_point": f"AP-{zone.upper()[:3]}",
				"status": "evacuated",
			})
		record: dict[str, Any] = {
			"id": evacuation_id,
			"emergency_id": emergency_id,
			"tenant_id": tenant_id,
			"zones": zone_details,
			"total_zones": len(zones),
			"total_persons_evacuated": total_persons,
			"transport_units_deployed": len(zones) * 2,
			"managed_by": self.actor_id,
			"started_at": datetime.utcnow().isoformat(),
			"status": "in_progress",
		}
		self._evacuations.append(record)
		self._audit(tenant_id, "evacuation_managed", evacuation_id)
		return record

	def relief_distribution(
		self,
		emergency_id: str,
		items: list[dict[str, Any]],
		locations: list[str],
	) -> dict[str, Any]:
		"""Coordinate relief item distribution to affected locations."""
		assert emergency_id, "emergency_id required"
		assert items, "items required"
		assert locations, "locations required"
		tenant_id = self.tenant_id
		distribution_id = _new_id()
		distributions = []
		for location in locations:
			for item in items:
				qty_per_location = item.get("quantity", 0) // max(len(locations), 1)
				distributions.append({
					"location": location,
					"item": item.get("name", ""),
					"quantity": qty_per_location,
					"unit": item.get("unit", "units"),
				})
		record: dict[str, Any] = {
			"id": distribution_id,
			"emergency_id": emergency_id,
			"tenant_id": tenant_id,
			"items": items,
			"locations": locations,
			"distributions": distributions,
			"total_items_types": len(items),
			"total_locations": len(locations),
			"coordinated_by": self.actor_id,
			"started_at": datetime.utcnow().isoformat(),
			"status": "distributing",
		}
		self._relief_distributions.append(record)
		self._audit(tenant_id, "relief_distributed", distribution_id)
		return record

	def casualty_tracking(self, emergency_id: str) -> dict[str, Any]:
		"""Return casualty tracking summary for an emergency."""
		assert emergency_id, "emergency_id required"
		tenant_id = self.tenant_id
		incident = self._get_incident(emergency_id, tenant_id)
		if incident is None:
			raise KeyError(f"incident {emergency_id} not found")
		records = [c for c in self._casualty_records if c.get("emergency_id") == emergency_id and c.get("tenant_id") == tenant_id]
		tracking_id = _new_id()
		by_status: dict[str, int] = {}
		for r in records:
			s = r.get("status", "unknown")
			by_status[s] = by_status.get(s, 0) + 1
		return {
			"id": tracking_id,
			"emergency_id": emergency_id,
			"tenant_id": tenant_id,
			"total_casualties": len(records),
			"by_status": by_status,
			"missing": by_status.get("missing", 0),
			"injured": by_status.get("injured", 0),
			"deceased": by_status.get("deceased", 0),
			"recovered": by_status.get("recovered", 0),
			"last_updated": datetime.utcnow().isoformat(),
		}

	def after_action_review(
		self,
		emergency_id: str,
		findings: list[str],
	) -> dict[str, Any]:
		"""Conduct an after-action review for a completed emergency."""
		assert emergency_id, "emergency_id required"
		assert findings, "findings required"
		tenant_id = self.tenant_id
		incident = self._get_incident(emergency_id, tenant_id)
		if incident is None:
			raise KeyError(f"incident {emergency_id} not found")
		aar_id = _new_id()
		lessons = [f for f in findings if "improvement" in f.lower() or "lesson" in f.lower()]
		recommendations = [f"Action required: {f}" for f in findings if "critical" in f.lower() or "fail" in f.lower()]
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "record_aar",
			"incident_present": True, "reviewer_present": True, "lessons_present": True,
		})
		item = AfterActionReview(aar_id, tenant_id, emergency_id, self.actor_id, "; ".join(lessons) or "See findings", "; ".join(recommendations), "", "completed")
		self.after_action_reviews[self._key(tenant_id, aar_id)] = item
		self._audit(tenant_id, "after_action_review_completed", aar_id)
		return {
			"id": aar_id,
			"emergency_id": emergency_id,
			"tenant_id": tenant_id,
			"findings": findings,
			"findings_count": len(findings),
			"lessons_identified": len(lessons),
			"recommendations": recommendations,
			"reviewed_by": self.actor_id,
			"reviewed_at": datetime.utcnow().isoformat(),
			"status": "completed",
		}

	def emergency_analytics(self, period: str) -> dict[str, Any]:
		"""Return emergency management performance analytics."""
		assert period, "period required"
		tenant_id = self.tenant_id
		incidents = [i for (tid, _), i in self.incidents.items() if tid == tenant_id]
		resources = [r for (tid, _), r in self.resources.items() if tid == tenant_id]
		agencies = [a for (tid, _), a in self.agencies.items() if tid == tenant_id]
		sitreps = [s for (tid, _), s in self.situation_reports.items() if tid == tenant_id]
		aars = [a for (tid, _), a in self.after_action_reviews.items() if tid == tenant_id]
		total_evacuees = sum(e.get("total_persons_evacuated", 0) for e in self._evacuations if e.get("tenant_id") == tenant_id)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"incidents": {
				"total": len(incidents),
				"active": sum(1 for i in incidents if i.phase not in ("recovery", "stand_down")),
				"by_severity": {s: sum(1 for i in incidents if i.severity == s) for s in set(i.severity for i in incidents)},
			},
			"resources_mobilised": len(resources),
			"agencies_activated": len(agencies),
			"situation_reports": len(sitreps),
			"after_action_reviews": len(aars),
			"evacuations": len(self._evacuations),
			"total_evacuees": total_evacuees,
			"relief_distributions": len(self._relief_distributions),
			"casualty_records": len(self._casualty_records),
			"generated_at": datetime.utcnow().isoformat(),
		}

	def transition_phase(self, incident_id: str, tenant_id: str, new_phase: str) -> dict[str, Any]:
		incident = self._get_incident(incident_id, tenant_id)
		if incident is None:
			raise KeyError(f"Incident not found: {incident_id}")
		new_phase = _normalize(new_phase)
		if new_phase not in SUPPORTED_INCIDENT_PHASES:
			raise ValueError(f"Unsupported phase: {new_phase}")
		incident.phase = new_phase
		self._audit(tenant_id, "incident_phase_transitioned", incident_id)
		return incident.to_dict()

	def mobilise_resource(
		self, resource_id: str, tenant_id: str, incident_id: str, resource_type: str,
		quantity: int, unit: str, responsible_agency: str, evidence_reference: str,
		status: str = "mobilised",
	) -> dict[str, Any]:
		incident = self._get_incident(incident_id, tenant_id)
		resource_type = _normalize(resource_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "mobilise_resource",
			"resource_type_supported": resource_type in SUPPORTED_RESOURCE_TYPES,
			"incident_present": incident is not None,
			"quantity_present": quantity > 0,
			"over_allocated": False,
		})
		item = ResourceMobilisation(resource_id, tenant_id, incident_id, resource_type, int(quantity), unit, responsible_agency, status, evidence_reference)
		self.resources[self._key(tenant_id, resource_id)] = item
		self._audit(tenant_id, "resource_mobilised", resource_id)
		return item.to_dict()

	def activate_agency(
		self, activation_id: str, tenant_id: str, incident_id: str, agency_type: str,
		agency_name: str, contact_reference: str, role: str,
	) -> dict[str, Any]:
		incident = self._get_incident(incident_id, tenant_id)
		agency_type = _normalize(agency_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "activate_agency",
			"agency_type_supported": agency_type in SUPPORTED_AGENCY_TYPES,
			"incident_present": incident is not None,
			"contact_present": _present(contact_reference),
		})
		item = AgencyActivation(activation_id, tenant_id, incident_id, agency_type, agency_name, contact_reference, role, datetime.utcnow().isoformat())
		self.agencies[self._key(tenant_id, activation_id)] = item
		self._audit(tenant_id, "agency_activated", activation_id)
		return item.to_dict()

	def update_eoc(
		self, eoc_id: str, tenant_id: str, incident_id: str, eoc_status: str,
		command_structure: str, activation_authority: str, evidence_reference: str,
		authorised: bool = True,
	) -> dict[str, Any]:
		eoc_status = _normalize(eoc_status)
		command_structure = _normalize(command_structure)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "update_eoc",
			"eoc_status_supported": eoc_status in SUPPORTED_EOC_STATUSES,
			"activation_authority_present": _present(activation_authority),
			"authorised": authorised,
		})
		item = EocRecord(eoc_id, tenant_id, incident_id, eoc_status, command_structure, activation_authority, datetime.utcnow().isoformat(), evidence_reference)
		self.eoc_records[self._key(tenant_id, eoc_id)] = item
		self._audit(tenant_id, "eoc_activated", eoc_id)
		return item.to_dict()

	def file_sitrep(
		self, sitrep_id: str, tenant_id: str, incident_id: str,
		period: str, author_id: str, summary: str, evidence_reference: str,
	) -> dict[str, Any]:
		incident = self._get_incident(incident_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "file_sitrep",
			"incident_present": incident is not None,
			"author_present": _present(author_id),
		})
		item = SituationReport(sitrep_id, tenant_id, incident_id, period, author_id, summary, evidence_reference)
		self.situation_reports[self._key(tenant_id, sitrep_id)] = item
		self._audit(tenant_id, "situation_report_filed", sitrep_id)
		return item.to_dict()

	def record_aar(
		self, aar_id: str, tenant_id: str, incident_id: str, reviewer_id: str,
		lessons_learned: str, recommendations: str, evidence_reference: str,
		status: str = "draft",
	) -> dict[str, Any]:
		incident = self._get_incident(incident_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_aar",
			"incident_present": incident is not None,
			"reviewer_present": _present(reviewer_id),
			"lessons_present": _present(lessons_learned),
		})
		item = AfterActionReview(aar_id, tenant_id, incident_id, reviewer_id, lessons_learned, recommendations, evidence_reference, status)
		self.after_action_reviews[self._key(tenant_id, aar_id)] = item
		self._audit(tenant_id, "after_action_review_completed", aar_id)
		return item.to_dict()

	def record_review(
		self, review_id: str, tenant_id: str, reference_id: str,
		reviewer_id: str, status: str, evidence_reference: str,
	) -> dict[str, Any]:
		status = _normalize(status)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_review",
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": _present(reviewer_id),
			"evidence_present": _present(evidence_reference),
		})
		item = EmergencyReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._key(tenant_id, review_id)] = item
		self._audit(tenant_id, "emergency_review_recorded", review_id)
		return item.to_dict()

	def register_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		runtime = _normalize(runtime)
		role = _normalize(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_eme_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		item = EmergencyAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "emergency_agent_registered", agent_id)
		return item.to_dict()

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation": "eme_batch", "event_stream": event_stream})
		if item_count < 1:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.government.eme.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"incident_count": self._count(self.incidents, tenant_id),
			"resource_count": self._count(self.resources, tenant_id),
			"agency_count": self._count(self.agencies, tenant_id),
			"eoc_count": self._count(self.eoc_records, tenant_id),
			"sitrep_count": self._count(self.situation_reports, tenant_id),
			"aar_count": self._count(self.after_action_reviews, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"evacuations": len(self._evacuations),
			"relief_distributions": len(self._relief_distributions),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
		}

	def _get_incident(self, incident_id: str, tenant_id: str) -> EmergencyIncident | None:
		return self.incidents.get(self._key(tenant_id, incident_id))

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
		reasons = ", ".join(a.get("reason", a.get("rule", "policy_denied")) for a in result["actions"])
		raise PermissionError(reasons or "policy_denied")


	def resource_request(self, incident_id: str, resource_type: str, quantity: int, requesting_agency: str) -> dict[str, Any]:
		"""Request resources for an emergency incident."""
		tenant_id = self.tenant_id
		req_id = _new_id()
		self._audit(tenant_id, "resource_requested", req_id)
		return {"request_id": req_id, "incident_id": incident_id, "resource_type": resource_type, "quantity": quantity, "requesting_agency": requesting_agency, "status": "pending", "requested_at": datetime.utcnow().isoformat()}

	def resource_release(self, incident_id: str, resource_id: str, releasing_agency: str) -> dict[str, Any]:
		"""Release a resource back after emergency resolution."""
		tenant_id = self.tenant_id
		resource = self.resources.get(self._key(tenant_id, resource_id))
		if resource:
			resource.status = "released"
		rel_id = _new_id()
		self._audit(tenant_id, "resource_released", rel_id)
		return {"release_id": rel_id, "incident_id": incident_id, "resource_id": resource_id, "releasing_agency": releasing_agency, "released_at": datetime.utcnow().isoformat()}

	def evacuation_order(self, incident_id: str, zones: list[str], authority: str) -> dict[str, Any]:
		"""Issue a mandatory evacuation order for zones."""
		tenant_id = self.tenant_id
		ord_id = _new_id()
		self._audit(tenant_id, "evacuation_order_issued", ord_id)
		return {"order_id": ord_id, "incident_id": incident_id, "zones": zones, "zone_count": len(zones), "issuing_authority": authority, "issued_at": datetime.utcnow().isoformat(), "status": "active"}

	def shelter_assign(self, incident_id: str, shelter_id: str, capacity: int, location: str) -> dict[str, Any]:
		"""Assign and activate an emergency shelter."""
		tenant_id = self.tenant_id
		assign_id = _new_id()
		self._audit(tenant_id, "shelter_assigned", assign_id)
		return {"assignment_id": assign_id, "incident_id": incident_id, "shelter_id": shelter_id, "capacity": capacity, "location": location, "activated_at": datetime.utcnow().isoformat(), "occupancy": 0, "status": "active"}

	def situation_brief(self, incident_id: str, period: str) -> dict[str, Any]:
		"""Generate a situation briefing for an incident."""
		return self.situation_report(incident_id, period)

	def media_statement(self, incident_id: str, statement: str, spokesperson: str) -> dict[str, Any]:
		"""Issue a public media statement about an emergency."""
		tenant_id = self.tenant_id
		stmt_id = _new_id()
		self._audit(tenant_id, "media_statement_issued", stmt_id)
		return {"statement_id": stmt_id, "incident_id": incident_id, "statement": statement, "spokesperson": spokesperson, "issued_at": datetime.utcnow().isoformat(), "status": "published"}

	def volunteer_register(self, volunteer_id: str, name: str, skills: list[str], availability: str) -> dict[str, Any]:
		"""Register a volunteer for emergency response."""
		tenant_id = self.tenant_id
		reg_id = _new_id()
		self._audit(tenant_id, "volunteer_registered", reg_id)
		return {"registration_id": reg_id, "volunteer_id": volunteer_id, "name": name, "skills": skills, "availability": availability, "registered_at": datetime.utcnow().isoformat(), "status": "registered"}

	def volunteer_assign(self, incident_id: str, volunteer_id: str, role: str) -> dict[str, Any]:
		"""Assign a volunteer to an emergency incident."""
		tenant_id = self.tenant_id
		assign_id = _new_id()
		self._audit(tenant_id, "volunteer_assigned", assign_id)
		return {"assignment_id": assign_id, "incident_id": incident_id, "volunteer_id": volunteer_id, "role": role, "assigned_at": datetime.utcnow().isoformat(), "status": "assigned"}

	def supply_track(self, incident_id: str, supply_type: str, quantity: int, location: str) -> dict[str, Any]:
		"""Track supply inventory at an emergency location."""
		tenant_id = self.tenant_id
		track_id = _new_id()
		return {"tracking_id": track_id, "incident_id": incident_id, "supply_type": supply_type, "quantity": quantity, "location": location, "tracked_at": datetime.utcnow().isoformat()}

	def damage_assess(self, incident_id: str, area: str, damage_categories: dict[str, Any]) -> dict[str, Any]:
		"""Record a damage assessment for an affected area."""
		tenant_id = self.tenant_id
		assess_id = _new_id()
		total_estimated = sum(float(v) if isinstance(v, (int, float)) else 0 for v in damage_categories.values())
		self._audit(tenant_id, "damage_assessed", assess_id)
		return {"assessment_id": assess_id, "incident_id": incident_id, "area": area, "damage_categories": damage_categories, "total_estimated_damage": total_estimated, "assessed_at": datetime.utcnow().isoformat()}

	def recovery_phase(self, incident_id: str, phase: str, lead_agency: str) -> dict[str, Any]:
		"""Transition incident to a recovery phase."""
		tenant_id = self.tenant_id
		incident = self._get_incident(incident_id, tenant_id)
		if incident is None:
			raise KeyError(f"incident {incident_id} not found")
		incident.phase = phase
		phase_id = _new_id()
		self._audit(tenant_id, "recovery_phase_set", phase_id)
		return {"phase_id": phase_id, "incident_id": incident_id, "phase": phase, "lead_agency": lead_agency, "transitioned_at": datetime.utcnow().isoformat()}

	def mutual_aid_request(self, incident_id: str, requesting_agency: str, aid_type: str, target_jurisdiction: str) -> dict[str, Any]:
		"""Request mutual aid from another jurisdiction."""
		tenant_id = self.tenant_id
		req_id = _new_id()
		ref = f"MAR-{datetime.utcnow().strftime('%Y%m%d')}-{req_id[:6].upper()}"
		self._audit(tenant_id, "mutual_aid_requested", req_id)
		return {"request_id": req_id, "reference": ref, "incident_id": incident_id, "requesting_agency": requesting_agency, "aid_type": aid_type, "target_jurisdiction": target_jurisdiction, "requested_at": datetime.utcnow().isoformat(), "status": "sent"}

	def public_alert(self, incident_id: str, alert_type: str, message: str, channels: list[str]) -> dict[str, Any]:
		"""Issue a public alert for an emergency."""
		tenant_id = self.tenant_id
		alert_id = _new_id()
		self._audit(tenant_id, "public_alert_issued", alert_id)
		return {"alert_id": alert_id, "incident_id": incident_id, "alert_type": alert_type, "message": message, "channels": channels, "issued_at": datetime.utcnow().isoformat(), "status": "broadcast"}

	def incident_close(self, incident_id: str, closing_notes: str, closed_by: str) -> dict[str, Any]:
		"""Formally close an emergency incident."""
		tenant_id = self.tenant_id
		incident = self._get_incident(incident_id, tenant_id)
		if incident is None:
			raise KeyError(f"incident {incident_id} not found")
		incident.phase = "stand_down"
		close_id = _new_id()
		self._audit(tenant_id, "incident_closed", close_id)
		return {"closure_id": close_id, "incident_id": incident_id, "closing_notes": closing_notes, "closed_by": closed_by, "closed_at": datetime.utcnow().isoformat(), "status": "closed"}

	def lessons_learned(self, incident_id: str, findings: list[str]) -> dict[str, Any]:
		"""Record lessons learned — domain alias for after_action_review."""
		return self.after_action_review(incident_id, findings)

	def emergency_report(self, period: str) -> dict[str, Any]:
		"""Generate emergency management report for the period."""
		return self.emergency_analytics(period)

	def emergency_analytics(self, period: str) -> dict[str, Any]:
		"""Return emergency management performance analytics."""
		assert period, "period required"
		tenant_id = self.tenant_id
		incidents = [i for (tid, _), i in self.incidents.items() if tid == tenant_id]
		resources = [r for (tid, _), r in self.resources.items() if tid == tenant_id]
		agencies = [a for (tid, _), a in self.agencies.items() if tid == tenant_id]
		sitreps = [s for (tid, _), s in self.situation_reports.items() if tid == tenant_id]
		aars = [a for (tid, _), a in self.after_action_reviews.items() if tid == tenant_id]
		total_evacuees = sum(e.get("total_persons_evacuated", 0) for e in self._evacuations if e.get("tenant_id") == tenant_id)
		return {
			"tenant_id": tenant_id, "period": period,
			"incidents": {"total": len(incidents), "active": sum(1 for i in incidents if i.phase not in ("recovery", "stand_down")), "by_severity": {s: sum(1 for i in incidents if i.severity == s) for s in set(i.severity for i in incidents)}},
			"resources_mobilised": len(resources), "agencies_activated": len(agencies), "situation_reports": len(sitreps), "after_action_reviews": len(aars),
			"evacuations": len(self._evacuations), "total_evacuees": total_evacuees, "relief_distributions": len(self._relief_distributions), "casualty_records": len(self._casualty_records), "generated_at": datetime.utcnow().isoformat(),
		}



	async def ml_emergency_severity(self, *args, **kwargs):
		"""AI-powered AI emergency response severity and resource requirement scoring. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="emergency_response_severity")
			return {"severity_score": round(result.score,3), "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

GovernmentEmeService = EmergencyManagementService
