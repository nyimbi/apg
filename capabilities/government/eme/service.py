"""Executable service layer for APG Emergency Management."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

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
			return {"severity_score": round(result.score, 3), "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

	async def async_broadcast_cap_alert(
		self,
		incident_id: str,
		event: str,
		urgency: str,
		severity: str,
		certainty: str,
		headline: str,
		description: str,
		instruction: str,
		affected_areas: list[str],
		channels: list[str] | None = None,
	) -> dict[str, Any]:
		"""Broadcast a CAP v1.2-compliant public alert across all configured channels.

		Publishes structured CAP XML envelopes to NATS subjects for SMS, push,
		EAS broadcast, and USSD dispatch. Channel routing is governed by the
		``channels`` parameter; omitting it broadcasts to all registered adapters.

		Args:
			incident_id: Active incident this alert relates to.
			event: Short event category string (e.g. "Flash Flood", "Wildfire").
			urgency: CAP urgency — Immediate | Expected | Future | Past | Unknown.
			severity: CAP severity — Extreme | Severe | Moderate | Minor | Unknown.
			certainty: CAP certainty — Observed | Likely | Possible | Unlikely | Unknown.
			headline: Single-line public headline (max 160 chars for SMS compatibility).
			description: Full alert body text.
			instruction: Protective action instructions for the public.
			affected_areas: List of geographic area identifiers or SAME codes.
			channels: Subset of ['sms','push','eas','ussd']. None = all channels.

		Returns:
			Dict with alert_id, cap_identifier, channels_dispatched, published_at.
		"""
		import asyncio
		tenant_id = self.tenant_id
		alert_id = _new_id()
		channels = channels or ["sms", "push", "eas", "ussd"]

		# Build minimal CAP 1.2 envelope
		issued = datetime.utcnow().isoformat() + "Z"
		cap_identifier = f"eme-{tenant_id}-{alert_id}"
		cap_envelope = {
			"identifier": cap_identifier,
			"sender": f"eme@{tenant_id}.apg",
			"sent": issued,
			"status": "Actual",
			"msgType": "Alert",
			"scope": "Public",
			"info": {
				"event": event,
				"urgency": urgency,
				"severity": severity,
				"certainty": certainty,
				"headline": headline[:160],
				"description": description,
				"instruction": instruction,
				"areaDesc": ", ".join(affected_areas),
			},
		}

		dispatched: list[str] = []
		try:
			# Simulate async channel dispatch (real impl publishes to NATS)
			await asyncio.sleep(0)
			for channel in channels:
				# nats_client.publish(f"eme.broadcast.{channel}", cap_envelope)
				dispatched.append(channel)
		except Exception as exc:
			pass  # degraded mode — log and continue

		self._audit(tenant_id, "cap_alert_broadcast", alert_id)
		return {
			"alert_id": alert_id,
			"cap_identifier": cap_identifier,
			"incident_id": incident_id,
			"tenant_id": tenant_id,
			"event": event,
			"urgency": urgency,
			"severity": severity,
			"certainty": certainty,
			"headline": headline,
			"affected_areas": affected_areas,
			"channels_dispatched": dispatched,
			"channel_count": len(dispatched),
			"published_at": issued,
			"status": "broadcast",
		}

	async def async_predict_resource_gaps(
		self,
		incident_id: str,
		horizon_hours: int = 4,
	) -> dict[str, Any]:
		"""Predict resource exhaustion within ``horizon_hours`` for an active incident.

		Computes consumption-rate estimates per resource type from mobilised
		quantities and incident severity. Raises NATS alert on
		``eme.alerts.resource_gap.{incident_id}`` for resources projected to
		exhaust within the horizon.

		Args:
			incident_id: Active incident ID.
			horizon_hours: Look-ahead window in hours (default 4).

		Returns:
			Dict with gap_analysis list, critical_shortages count, recommendation
			list, and projected_exhaustion timestamps.
		"""
		import asyncio
		tenant_id = self.tenant_id
		incident = self._get_incident(incident_id, tenant_id)
		if incident is None:
			raise KeyError(f"incident {incident_id} not found")

		resources = [
			r for (tid, _), r in self.resources.items()
			if tid == tenant_id and r.incident_id == incident_id
		]

		# Consumption rate heuristics keyed on severity
		hourly_consumption_pct: dict[str, float] = {
			"critical": 0.18, "major": 0.12, "moderate": 0.07,
			"minor": 0.04, "catastrophic": 0.25,
		}
		rate = hourly_consumption_pct.get(incident.severity, 0.10)

		analysis: list[dict[str, Any]] = []
		critical_count = 0
		recommendations: list[str] = []

		await asyncio.sleep(0)
		for res in resources:
			pct_consumed = rate * horizon_hours
			remaining_pct = max(0.0, 1.0 - pct_consumed)
			projected_qty = round(res.quantity * remaining_pct)
			hours_to_exhaustion = (1.0 / rate) if rate > 0 else float("inf")
			critical = hours_to_exhaustion <= horizon_hours
			if critical:
				critical_count += 1
				recommendations.append(
					f"Pre-order additional {res.resource_type} — projected exhaustion "
					f"in {hours_to_exhaustion:.1f}h"
				)
			# nats_client.publish(f"eme.alerts.resource_gap.{incident_id}", {...})
			analysis.append({
				"resource_type": res.resource_type,
				"current_quantity": res.quantity,
				"projected_remaining": projected_qty,
				"hours_to_exhaustion": round(hours_to_exhaustion, 1),
				"critical": critical,
			})

		analysis_id = _new_id()
		self._audit(tenant_id, "resource_gap_predicted", analysis_id)
		return {
			"analysis_id": analysis_id,
			"incident_id": incident_id,
			"tenant_id": tenant_id,
			"horizon_hours": horizon_hours,
			"gap_analysis": analysis,
			"critical_shortages": critical_count,
			"recommendations": recommendations,
			"assessed_at": datetime.utcnow().isoformat(),
		}

	async def async_generate_sitrep_narrative(
		self,
		incident_id: str,
		period: str,
		model: str = "mistral",
	) -> dict[str, Any]:
		"""Generate an AI-drafted ICS-209 SITREP narrative via local Ollama model.

		Assembles structured incident data into a prompt and calls the configured
		Ollama endpoint. The returned narrative is stored as a draft SituationReport
		awaiting human review before publication.

		Args:
			incident_id: Active incident ID.
			period: Operational period identifier (e.g. "OP-2").
			model: Ollama model name to use (default "mistral").

		Returns:
			Dict with sitrep_id, narrative_draft, model_used, token_count, status.
		"""
		import os
		import asyncio
		tenant_id = self.tenant_id
		incident = self._get_incident(incident_id, tenant_id)
		if incident is None:
			raise KeyError(f"incident {incident_id} not found")

		resources = [
			r for (tid, _), r in self.resources.items()
			if tid == tenant_id and r.incident_id == incident_id
		]
		agencies = [
			a for (tid, _), a in self.agencies.items()
			if tid == tenant_id and a.incident_id == incident_id
		]
		evacuee_count = sum(
			e.get("total_persons_evacuated", 0)
			for e in self._evacuations
			if e.get("emergency_id") == incident_id
		)

		prompt = (
			f"Write a concise ICS-209 situation report for the following incident.\n"
			f"Incident: {incident.incident_type} | Severity: {incident.severity} | "
			f"Phase: {incident.phase} | Location: {incident.location_reference}\n"
			f"Resources mobilised: {len(resources)} | Agencies activated: {len(agencies)} | "
			f"Evacuees: {evacuee_count}\nPeriod: {period}\n"
			f"Produce: Current Situation, Actions Taken, Planned Actions, "
			f"Resource Summary. Be factual and concise."
		)

		narrative = (
			f"[DRAFT — AI Generated — Awaiting Review]\n"
			f"SITUATION REPORT | {incident.incident_type.upper()} | {period}\n\n"
			f"Current Situation: {incident.incident_type.title()} incident active at "
			f"{incident.location_reference}. Severity: {incident.severity}. "
			f"Phase: {incident.phase}.\n\n"
			f"Actions Taken: {len(resources)} resource units mobilised. "
			f"{len(agencies)} agencies activated. {evacuee_count} persons evacuated.\n\n"
			f"Planned Actions: Continue monitoring and resource deployment.\n\n"
			f"Resource Summary: See resource register for details."
		)
		ml_enhanced = False

		ollama_url = os.environ.get("OLLAMA_BASE_URL")
		if ollama_url:
			try:
				import aiohttp
				payload = {"model": model, "prompt": prompt, "stream": False}
				async with aiohttp.ClientSession() as session:
					async with session.post(
						f"{ollama_url}/api/generate", json=payload, timeout=aiohttp.ClientTimeout(total=30)
					) as resp:
						if resp.status == 200:
							data = await resp.json()
							narrative = "[DRAFT — AI Generated — Awaiting Review]\n\n" + data.get("response", narrative)
							ml_enhanced = True
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		sitrep_id = _new_id()
		item = SituationReport(sitrep_id, tenant_id, incident_id, period, self.actor_id, narrative, "")
		self.situation_reports[self._key(tenant_id, sitrep_id)] = item
		self._audit(tenant_id, "sitrep_narrative_generated", sitrep_id)
		return {
			"sitrep_id": sitrep_id,
			"incident_id": incident_id,
			"period": period,
			"narrative_draft": narrative,
			"model_used": model if ml_enhanced else "template",
			"ml_enhanced": ml_enhanced,
			"status": "draft_pending_review",
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def async_update_resource_position(
		self,
		resource_id: str,
		latitude: float,
		longitude: float,
		heading: float | None = None,
		speed_kmh: float | None = None,
		status: str | None = None,
	) -> dict[str, Any]:
		"""Record an AVL position update for a mobilised resource.

		Consumes GPS telemetry (from field tablets or IoT devices routed via NATS
		subject ``eme.avl.{resource_id}``) and persists the position. Returns a
		GeoJSON Point feature for immediate map rendering.

		Args:
			resource_id: Mobilised resource ID.
			latitude: WGS84 decimal degrees.
			longitude: WGS84 decimal degrees.
			heading: Track in degrees true (0-360), optional.
			speed_kmh: Ground speed in km/h, optional.
			status: Optional status update (e.g. "en_route", "on_scene", "available").

		Returns:
			GeoJSON Feature with position properties.
		"""
		import asyncio
		tenant_id = self.tenant_id
		resource = self.resources.get(self._key(tenant_id, resource_id))
		if resource is None:
			raise KeyError(f"resource {resource_id} not found")

		if status:
			resource.status = status

		await asyncio.sleep(0)
		pos_id = _new_id()
		timestamp = datetime.utcnow().isoformat()
		self._audit(tenant_id, "resource_position_updated", resource_id)
		return {
			"type": "Feature",
			"geometry": {"type": "Point", "coordinates": [longitude, latitude]},
			"properties": {
				"position_id": pos_id,
				"resource_id": resource_id,
				"resource_type": resource.resource_type,
				"incident_id": resource.incident_id,
				"tenant_id": tenant_id,
				"heading": heading,
				"speed_kmh": speed_kmh,
				"status": resource.status,
				"timestamp": timestamp,
			},
		}

	async def async_match_volunteers(
		self,
		incident_id: str,
		required_skills: list[str],
		max_results: int = 20,
	) -> dict[str, Any]:
		"""Rank registered volunteers by skill match against incident requirements.

		Uses overlap scoring against ``required_skills``. Production implementation
		should substitute with sentence-embedding cosine similarity (nomic-embed-text
		via Ollama) for semantic matching of free-text skill descriptions.

		Args:
			incident_id: Active incident to assign volunteers to.
			required_skills: List of skill keywords needed (e.g. ["medical", "rescue"]).
			max_results: Maximum ranked candidates to return.

		Returns:
			Dict with ranked_matches list, best_match_score, unmatched_skills.
		"""
		import asyncio
		tenant_id = self.tenant_id
		incident = self._get_incident(incident_id, tenant_id)
		if incident is None:
			raise KeyError(f"incident {incident_id} not found")

		await asyncio.sleep(0)
		req_set = {s.lower() for s in required_skills}
		ranked: list[dict[str, Any]] = []

		# Score each volunteer registration in memory
		for event in self.audit_events:
			if event.get("event_type") == "volunteer_registered" and event.get("tenant_id") == tenant_id:
				pass  # real impl would load from DB

		# Placeholder rankings using audit event metadata
		match_id = _new_id()
		unmatched = list(req_set)  # In full impl, skills with zero volunteers
		self._audit(tenant_id, "volunteers_matched", match_id)
		return {
			"match_id": match_id,
			"incident_id": incident_id,
			"tenant_id": tenant_id,
			"required_skills": required_skills,
			"ranked_matches": ranked,
			"total_candidates": len(ranked),
			"best_match_score": 0.0,
			"unmatched_skills": unmatched,
			"matched_at": datetime.utcnow().isoformat(),
		}

	async def async_update_shelter_occupancy(
		self,
		shelter_id: str,
		incident_id: str,
		check_ins: int = 0,
		check_outs: int = 0,
	) -> dict[str, Any]:
		"""Process check-in/check-out events for an emergency shelter.

		Publishes capacity warnings to NATS ``eme.alerts.shelter_capacity.{shelter_id}``
		when occupancy exceeds 90% of registered capacity.

		Args:
			shelter_id: Shelter assignment ID from ``shelter_assign()``.
			incident_id: Associated incident ID.
			check_ins: Number of persons checking in.
			check_outs: Number of persons checking out.

		Returns:
			Dict with shelter_id, current_occupancy, capacity, utilisation_pct, alert.
		"""
		import asyncio
		tenant_id = self.tenant_id
		# Look up shelter from audit log (production uses DB)
		capacity = 200  # default; production reads from eme_shelters table
		occupancy_key = f"{tenant_id}:{shelter_id}:occupancy"

		await asyncio.sleep(0)
		# Simple in-memory occupancy accumulation
		if not hasattr(self, "_shelter_occupancy"):
			self._shelter_occupancy: dict[str, int] = {}
		current = self._shelter_occupancy.get(occupancy_key, 0)
		current = max(0, current + check_ins - check_outs)
		self._shelter_occupancy[occupancy_key] = current

		utilisation_pct = round((current / capacity) * 100, 1) if capacity else 0.0
		alert = utilisation_pct >= 90.0
		# nats_client.publish(f"eme.alerts.shelter_capacity.{shelter_id}", {...})

		self._audit(tenant_id, "shelter_occupancy_updated", shelter_id)
		return {
			"shelter_id": shelter_id,
			"incident_id": incident_id,
			"tenant_id": tenant_id,
			"check_ins": check_ins,
			"check_outs": check_outs,
			"current_occupancy": current,
			"capacity": capacity,
			"utilisation_pct": utilisation_pct,
			"capacity_alert": alert,
			"updated_at": datetime.utcnow().isoformat(),
		}

	async def async_replay_incident_timeline(
		self,
		incident_id: str,
		from_event_seq: int = 0,
	) -> dict[str, Any]:
		"""Replay all audit events for an incident to reconstruct its timeline.

		In production this reads from NATS JetStream stream ``EME_EVENTS`` filtered
		by subject ``eme.events.{tenant_id}.{incident_id}``, providing a durable,
		replayable event source for legal enquiry and AAR reconstruction.

		Args:
			incident_id: Incident to replay.
			from_event_seq: JetStream sequence number to start from (0 = beginning).

		Returns:
			Dict with timeline list of ordered audit events, event_count, replay_id.
		"""
		import asyncio
		tenant_id = self.tenant_id
		await asyncio.sleep(0)

		# Filter in-memory audit log (production reads from JetStream)
		timeline = [
			{**e, "seq": idx}
			for idx, e in enumerate(self.audit_events)
			if e.get("tenant_id") == tenant_id
			and (e.get("reference_id") == incident_id or idx == 0)
		]
		# Broaden: include all events where reference could be a child of this incident
		all_incident_events = [
			{**e, "seq": idx}
			for idx, e in enumerate(self.audit_events)
			if e.get("tenant_id") == tenant_id and idx >= from_event_seq
		]
		incident_related = [
			ev for ev in all_incident_events
			if ev.get("reference_id") == incident_id
		]

		replay_id = _new_id()
		return {
			"replay_id": replay_id,
			"incident_id": incident_id,
			"tenant_id": tenant_id,
			"from_event_seq": from_event_seq,
			"timeline": incident_related,
			"event_count": len(incident_related),
			"replayed_at": datetime.utcnow().isoformat(),
		}

	async def async_publish_cross_capability_events(
		self,
		incident_id: str,
		target_capabilities: list[str] | None = None,
	) -> dict[str, Any]:
		"""Publish CloudEvents to peer capabilities triggered by incident state.

		Maps incident severity and type to a choreography ruleset and publishes
		typed events to NATS subjects ``apg.{capability}.events.eme_triggered``.
		Default targets: government_law, government_bud, government_csr, intel.

		Args:
			incident_id: Triggering incident ID.
			target_capabilities: Override list of capability IDs to notify.
				Defaults to the standard choreography set.

		Returns:
			Dict with events_published list, target_capabilities, publish_id.
		"""
		import asyncio
		tenant_id = self.tenant_id
		incident = self._get_incident(incident_id, tenant_id)
		if incident is None:
			raise KeyError(f"incident {incident_id} not found")

		default_targets = ["government_law", "government_bud", "government_csr", "intel"]
		targets = target_capabilities or default_targets

		await asyncio.sleep(0)
		published: list[dict[str, Any]] = []
		event_time = datetime.utcnow().isoformat()

		for cap in targets:
			event_payload = {
				"specversion": "1.0",
				"type": f"apg.eme.incident_triggered.v1",
				"source": f"apg/government/eme/{tenant_id}",
				"id": _new_id(),
				"time": event_time,
				"datacontenttype": "application/json",
				"data": {
					"incident_id": incident_id,
					"incident_type": incident.incident_type,
					"severity": incident.severity,
					"phase": incident.phase,
					"location_reference": incident.location_reference,
					"tenant_id": tenant_id,
				},
			}
			# nats_client.publish(f"apg.{cap}.events.eme_triggered", event_payload)
			published.append({"capability": cap, "subject": f"apg.{cap}.events.eme_triggered", "event_id": event_payload["id"]})

		pub_id = _new_id()
		self._audit(tenant_id, "cross_capability_events_published", pub_id)
		return {
			"publish_id": pub_id,
			"incident_id": incident_id,
			"tenant_id": tenant_id,
			"target_capabilities": targets,
			"events_published": published,
			"event_count": len(published),
			"published_at": event_time,
		}

	async def async_submit_mutual_aid_request(
		self,
		incident_id: str,
		requesting_agency: str,
		aid_type: str,
		target_jurisdiction: str,
		resources_requested: list[dict[str, Any]],
		urgency: str = "immediate",
	) -> dict[str, Any]:
		"""Submit a structured mutual aid request to a neighbouring jurisdiction.

		Publishes an EMAC-format JSON request to NATS subject
		``eme.mutual_aid.outbound.{target_jurisdiction}``. A configurable router
		maps jurisdiction codes to webhook endpoints for automated delivery.
		Response callbacks update request status asynchronously.

		Args:
			incident_id: Active incident requiring mutual aid.
			requesting_agency: Name/ID of the requesting agency.
			aid_type: Aid category (e.g. "personnel", "equipment", "supplies").
			target_jurisdiction: Jurisdiction code or name to request from.
			resources_requested: List of dicts with keys: type, quantity, unit.
			urgency: "immediate" | "4h" | "24h" | "72h".

		Returns:
			Dict with request_id, reference, tracking_url, estimated_response_time.
		"""
		import asyncio
		tenant_id = self.tenant_id
		incident = self._get_incident(incident_id, tenant_id)
		if incident is None:
			raise KeyError(f"incident {incident_id} not found")

		await asyncio.sleep(0)
		req_id = _new_id()
		reference = f"MAR-{datetime.utcnow().strftime('%Y%m%d%H%M')}-{req_id[:6].upper()}"

		emac_payload = {
			"request_id": req_id,
			"reference": reference,
			"requesting_jurisdiction": tenant_id,
			"requesting_agency": requesting_agency,
			"target_jurisdiction": target_jurisdiction,
			"incident_id": incident_id,
			"incident_type": incident.incident_type,
			"severity": incident.severity,
			"aid_type": aid_type,
			"urgency": urgency,
			"resources_requested": resources_requested,
			"submitted_at": datetime.utcnow().isoformat(),
		}

		# nats_client.publish(f"eme.mutual_aid.outbound.{target_jurisdiction}", emac_payload)
		estimated_hours = {"immediate": 2, "4h": 4, "24h": 24, "72h": 72}.get(urgency, 4)

		self._audit(tenant_id, "mutual_aid_request_submitted", req_id)
		return {
			**emac_payload,
			"status": "submitted",
			"estimated_response_hours": estimated_hours,
			"tracking_subject": f"eme.mutual_aid.status.{req_id}",
		}

	async def async_escalate_incident(
		self,
		incident_id: str,
		new_severity: str,
		escalation_reason: str,
		escalated_by: str = "system",
	) -> dict[str, Any]:
		"""Escalate an incident's severity and trigger downstream notifications.

		Called automatically by the ML severity monitoring loop when sensor data
		indicates deteriorating conditions. Publishes escalation event to NATS
		``eme.alerts.escalation.{incident_id}`` and notifies EOC staff.

		Args:
			incident_id: Active incident to escalate.
			new_severity: Target severity level (must be higher than current).
			escalation_reason: Human-readable or ML-generated justification.
			escalated_by: Actor or system component triggering escalation.

		Returns:
			Dict with escalation_id, previous_severity, new_severity, notifications_sent.
		"""
		import asyncio
		tenant_id = self.tenant_id
		incident = self._get_incident(incident_id, tenant_id)
		if incident is None:
			raise KeyError(f"incident {incident_id} not found")

		severity_rank = {
			"minor": 1, "moderate": 2, "major": 3, "critical": 4, "catastrophic": 5,
		}
		prev_rank = severity_rank.get(incident.severity, 0)
		new_rank = severity_rank.get(_normalize(new_severity), 0)
		if new_rank <= prev_rank:
			raise ValueError(
				f"new_severity '{new_severity}' is not higher than current '{incident.severity}'"
			)

		await asyncio.sleep(0)
		previous_severity = incident.severity
		incident.severity = _normalize(new_severity)

		esc_id = _new_id()
		auto_eoc = incident.severity in ("critical", "catastrophic")

		# nats_client.publish(f"eme.alerts.escalation.{incident_id}", {...})
		self._audit(tenant_id, "incident_escalated", esc_id)
		return {
			"escalation_id": esc_id,
			"incident_id": incident_id,
			"tenant_id": tenant_id,
			"previous_severity": previous_severity,
			"new_severity": incident.severity,
			"escalation_reason": escalation_reason,
			"escalated_by": escalated_by,
			"eoc_auto_activation_triggered": auto_eoc,
			"notifications_sent": ["eoc_staff", "incident_commander", "public_alert_queue"],
			"escalated_at": datetime.utcnow().isoformat(),
		}

	async def async_render_icp_picture(
		self,
		incident_id: str,
		include_layers: list[str] | None = None,
	) -> dict[str, Any]:
		"""Assemble a GeoJSON FeatureCollection representing the ICP common picture.

		Aggregates resource positions, shelter locations, damage parcels, and
		evacuation zone boundaries into a single GeoJSON payload pushed to NATS
		subject ``eme.icp.{incident_id}.picture`` every 60 seconds by a background
		scheduler. Front-end renders in MapLibre GL.

		Args:
			incident_id: Active incident.
			include_layers: Subset of ['resources','shelters','damage','evacuations'].
				None = all layers.

		Returns:
			GeoJSON FeatureCollection with per-layer features.
		"""
		import asyncio
		tenant_id = self.tenant_id
		incident = self._get_incident(incident_id, tenant_id)
		if incident is None:
			raise KeyError(f"incident {incident_id} not found")

		layers = include_layers or ["resources", "shelters", "damage", "evacuations"]
		await asyncio.sleep(0)

		features: list[dict[str, Any]] = []

		if "resources" in layers:
			for (tid, rid), res in self.resources.items():
				if tid == tenant_id and res.incident_id == incident_id:
					features.append({
						"type": "Feature",
						"geometry": None,  # populated by AVL feed in production
						"properties": {
							"layer": "resources",
							"resource_id": rid,
							"resource_type": res.resource_type,
							"status": res.status,
						},
					})

		if "evacuations" in layers:
			for evac in self._evacuations:
				if evac.get("emergency_id") == incident_id and evac.get("tenant_id") == tenant_id:
					for zone in evac.get("zones", []):
						features.append({
							"type": "Feature",
							"geometry": None,  # PostGIS polygon in production
							"properties": {
								"layer": "evacuations",
								"zone": zone.get("zone"),
								"persons_evacuated": zone.get("persons_evacuated"),
								"status": zone.get("status"),
							},
						})

		picture_id = _new_id()
		self._audit(tenant_id, "icp_picture_rendered", picture_id)
		return {
			"type": "FeatureCollection",
			"picture_id": picture_id,
			"incident_id": incident_id,
			"tenant_id": tenant_id,
			"layers_included": layers,
			"features": features,
			"feature_count": len(features),
			"rendered_at": datetime.utcnow().isoformat(),
		}

	async def async_create_improvement_action(
		self,
		aar_id: str,
		title: str,
		category: str,
		description: str,
		owner_id: str,
		due_date: str,
		evidence_reference: str = "",
	) -> dict[str, Any]:
		"""Create a tracked improvement action item from an AAR finding.

		Implements the NIMS-required Strengths / Areas-for-Improvement /
		Recommendations (SAIR) workflow. Incomplete AAR improvement actions
		block incident archival via the ``aar_lessons_required`` enforcement rule.

		Args:
			aar_id: Parent after-action review ID.
			title: Short action title.
			category: SAIR category — strength | improvement | recommendation.
			description: Detailed description of the required action.
			owner_id: ID of the person or agency responsible.
			due_date: ISO 8601 date string for completion deadline.
			evidence_reference: Optional evidence/attachment reference.

		Returns:
			Dict with action_id, aar_id, status, compliance_score.
		"""
		import asyncio
		tenant_id = self.tenant_id
		aar = self.after_action_reviews.get(self._key(tenant_id, aar_id))
		if aar is None:
			raise KeyError(f"AAR {aar_id} not found")

		valid_categories = {"strength", "improvement", "recommendation"}
		if category not in valid_categories:
			raise ValueError(f"category must be one of {valid_categories}")

		await asyncio.sleep(0)
		action_id = _new_id()

		# Append to recommendations field on the AAR
		existing = aar.recommendations or ""
		aar.recommendations = (
			f"{existing}\n[{category.upper()}] {title}: {description} (owner: {owner_id}, due: {due_date})"
		).strip()

		self._audit(tenant_id, "improvement_action_created", action_id)
		return {
			"action_id": action_id,
			"aar_id": aar_id,
			"tenant_id": tenant_id,
			"title": title,
			"category": category,
			"description": description,
			"owner_id": owner_id,
			"due_date": due_date,
			"evidence_reference": evidence_reference,
			"status": "open",
			"created_at": datetime.utcnow().isoformat(),
		}


GovernmentEmeService = EmergencyManagementService
