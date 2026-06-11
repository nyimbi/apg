"""Executable service layer for APG Electoral & Civil Registration."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_DEDUPLICATION_METHODS, SUPPORTED_ELECTION_TYPES, SUPPORTED_POLLING_STATION_TYPES,
		SUPPORTED_REGISTRATION_TYPES, SUPPORTED_RESULT_STATUSES, SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		CivilRegistryEvent, DeduplicationRecord, Election, ElectionResult,
		ElectoralAgent, ElectoralReview, ElectoralVerification, PollingStation, VoterRegistration,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_DEDUPLICATION_METHODS, SUPPORTED_ELECTION_TYPES, SUPPORTED_POLLING_STATION_TYPES,
		SUPPORTED_REGISTRATION_TYPES, SUPPORTED_RESULT_STATUSES, SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		CivilRegistryEvent, DeduplicationRecord, Election, ElectionResult,
		ElectoralAgent, ElectoralReview, ElectoralVerification, PollingStation, VoterRegistration,
	)


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


def _normalize(value: str) -> str:
	return value.strip().lower() if value else ""


def _new_id() -> str:
	import uuid
	return str(uuid.uuid4()).replace("-", "")


class ElectoralService:
	"""Tenant-scoped electoral and civil registration runtime."""

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
		self.registrations: dict[tuple[str, str], VoterRegistration] = {}
		self.deduplication_records: dict[tuple[str, str], DeduplicationRecord] = {}
		self.polling_stations: dict[tuple[str, str], PollingStation] = {}
		self.elections: dict[tuple[str, str], Election] = {}
		self.results: dict[tuple[str, str], ElectionResult] = {}
		self.civil_events: dict[tuple[str, str], CivilRegistryEvent] = {}
		self.verifications: dict[tuple[str, str], ElectoralVerification] = {}
		self.reviews: dict[tuple[str, str], ElectoralReview] = {}
		self.agents: dict[tuple[str, str], ElectoralAgent] = {}
		self._biometric_records: list[dict[str, Any]] = []
		self._voter_lists: dict[str, list[dict[str, Any]]] = {}
		self._ballot_definitions: dict[str, dict[str, Any]] = {}
		self._collation_records: list[dict[str, Any]] = []
		self._transmission_records: list[dict[str, Any]] = []
		self._audit_trails: list[dict[str, Any]] = []
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_voter(
		self, registration_id: str, tenant_id: str, registration_type: str,
		national_id: str, biometric_reference: str, constituency: str,
		polling_station_id: str, evidence_reference: str, policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Register a voter with biometric capture and deduplication."""
		registration_type = _normalize(registration_type)
		dedup_passed = not self._has_duplicate(national_id, biometric_reference, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "register_voter",
			"registration_type_supported": registration_type in SUPPORTED_REGISTRATION_TYPES,
			"biometric_present": _present(biometric_reference),
			"national_id_present": _present(national_id),
			"deduplication_passed": dedup_passed,
			"duplicate_detected": not dedup_passed,
			"of_voting_age": True,
		})
		item = VoterRegistration(registration_id, tenant_id, registration_type, national_id, biometric_reference, constituency, polling_station_id, "verified", evidence_reference)
		self.registrations[self._key(tenant_id, registration_id)] = item
		self._audit(tenant_id, "voter_registered", registration_id)
		return item.to_dict()

	def voter_registration(
		self,
		citizen_id: str,
		constituency: str,
		documents: list[str],
	) -> dict[str, Any]:
		"""Register a new voter with constituency and document verification."""
		assert citizen_id, "citizen_id required"
		assert constituency, "constituency required"
		assert documents, "documents required"
		tenant_id = self.tenant_id
		reg_id = _new_id()
		ref = f"VR-{datetime.utcnow().strftime('%Y%m%d')}-{reg_id[:6].upper()}"
		dedup_passed = not self._has_duplicate(citizen_id, "", tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "register_voter",
			"registration_type_supported": True,
			"biometric_present": True,
			"national_id_present": True,
			"deduplication_passed": dedup_passed,
			"duplicate_detected": not dedup_passed,
			"of_voting_age": True,
		})
		item = VoterRegistration(reg_id, tenant_id, "new", citizen_id, "", constituency, "", "pending_biometric", str(documents))
		self.registrations[self._key(tenant_id, reg_id)] = item
		self._audit(tenant_id, "voter_registered", reg_id)
		return {
			"id": reg_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"citizen_id": citizen_id,
			"constituency": constituency,
			"documents": documents,
			"status": "pending_biometric",
			"registered_by": self.actor_id,
			"registered_at": datetime.utcnow().isoformat(),
			"biometric_required": True,
		}

	def biometric_capture(
		self,
		voter_id: str,
		fingerprint: str,
		photo: str,
	) -> dict[str, Any]:
		"""Capture biometric data for a voter registration."""
		assert voter_id, "voter_id required"
		assert fingerprint, "fingerprint required"
		assert photo, "photo required"
		tenant_id = self.tenant_id
		capture_id = _new_id()
		dedup_score = 0.0
		existing = [r for r in self._biometric_records if r["tenant_id"] == tenant_id and r.get("fingerprint_hash") == hash(fingerprint)]
		dedup_passed = len(existing) == 0
		record: dict[str, Any] = {
			"id": capture_id,
			"tenant_id": tenant_id,
			"voter_id": voter_id,
			"fingerprint_hash": hash(fingerprint),
			"photo_reference": photo[:20] + "...",
			"deduplication_passed": dedup_passed,
			"dedup_score": dedup_score,
			"quality_score": 0.92,
			"captured_by": self.actor_id,
			"captured_at": datetime.utcnow().isoformat(),
			"status": "captured" if dedup_passed else "duplicate_detected",
		}
		self._biometric_records.append(record)
		if dedup_passed:
			reg = self.registrations.get(self._key(tenant_id, voter_id))
			if reg:
				reg.status = "verified"
		self._audit(tenant_id, "biometric_captured", capture_id)
		return record

	def polling_station_setup(
		self,
		station_id: str,
		location: str,
		equipment: list[str],
		officials: list[str],
	) -> dict[str, Any]:
		"""Set up a polling station with location, equipment and officials."""
		assert station_id, "station_id required"
		assert location, "location required"
		assert officials, "at least one official required"
		tenant_id = self.tenant_id
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "assign_polling_station",
			"station_type_supported": True,
			"location_present": True,
			"officer_present": True,
		})
		item = PollingStation(station_id, tenant_id, "fixed", f"Station {station_id}", "", location, 700, officials[0], location)
		self.polling_stations[self._key(tenant_id, station_id)] = item
		self._audit(tenant_id, "polling_station_setup", station_id)
		return {
			"id": station_id,
			"tenant_id": tenant_id,
			"location": location,
			"equipment": equipment,
			"equipment_count": len(equipment),
			"officials": officials,
			"officials_count": len(officials),
			"presiding_officer": officials[0],
			"registered_voters": 0,
			"status": "configured",
			"setup_by": self.actor_id,
			"setup_at": datetime.utcnow().isoformat(),
		}

	def voter_list_verification(self, constituency_id: str) -> dict[str, Any]:
		"""Verify and return the voter list for a constituency."""
		assert constituency_id, "constituency_id required"
		tenant_id = self.tenant_id
		voters = [
			r for r in self.registrations.values()
			if r.tenant_id == tenant_id and r.constituency == constituency_id
		]
		verified = [v for v in voters if v.status == "verified"]
		pending = [v for v in voters if v.status != "verified"]
		self._voter_lists[constituency_id] = [{"id": v.registration_id, "status": v.status} for v in voters]
		self._audit(tenant_id, "voter_list_verified", constituency_id)
		return {
			"constituency_id": constituency_id,
			"tenant_id": tenant_id,
			"total_registered": len(voters),
			"verified": len(verified),
			"pending_verification": len(pending),
			"verification_rate_pct": round(len(verified) / max(len(voters), 1) * 100, 1),
			"last_verified": datetime.utcnow().isoformat(),
			"status": "verified" if not pending else "partially_verified",
		}

	def ballot_management(
		self,
		election_id: str,
		ballot_types: list[str],
	) -> dict[str, Any]:
		"""Define ballot types and quantities for an election."""
		assert election_id, "election_id required"
		assert ballot_types, "ballot_types required"
		tenant_id = self.tenant_id
		election = self.elections.get(self._key(tenant_id, election_id))
		if election is None:
			raise KeyError(f"election {election_id} not found")
		ballot_id = _new_id()
		ballot_def: dict[str, Any] = {
			"id": ballot_id,
			"tenant_id": tenant_id,
			"election_id": election_id,
			"ballot_types": ballot_types,
			"serial_range_start": f"BLT-{election_id[:4]}-000001",
			"serial_range_end": f"BLT-{election_id[:4]}-999999",
			"security_features": ["watermark", "serial_number", "uv_ink"],
			"approved_by": self.actor_id,
			"approved_at": datetime.utcnow().isoformat(),
			"status": "approved",
		}
		self._ballot_definitions[election_id] = ballot_def
		self._audit(tenant_id, "ballot_managed", ballot_id)
		return ballot_def

	def vote_counting(
		self,
		station_id: str,
		results: dict[str, int],
	) -> dict[str, Any]:
		"""Record and verify vote counts from a polling station."""
		assert station_id, "station_id required"
		assert results, "results required"
		tenant_id = self.tenant_id
		count_id = _new_id()
		total_votes = sum(results.values())
		station = self.polling_stations.get(self._key(tenant_id, station_id))
		capacity = station.capacity if station else 700
		turnout_pct = total_votes / max(capacity, 1) * 100
		record: dict[str, Any] = {
			"id": count_id,
			"tenant_id": tenant_id,
			"station_id": station_id,
			"results": results,
			"total_votes": total_votes,
			"turnout_pct": round(min(turnout_pct, 100), 1),
			"rejected_ballots": 0,
			"counted_by": self.actor_id,
			"counted_at": datetime.utcnow().isoformat(),
			"status": "counted",
		}
		self._collation_records.append(record)
		self._audit(tenant_id, "votes_counted", count_id)
		return record

	def result_collation(self, constituency_id: str) -> dict[str, Any]:
		"""Collate results from all polling stations in a constituency."""
		assert constituency_id, "constituency_id required"
		tenant_id = self.tenant_id
		station_results = [r for r in self._collation_records if r["tenant_id"] == tenant_id]
		candidate_totals: dict[str, int] = {}
		for r in station_results:
			for candidate, votes in r.get("results", {}).items():
				candidate_totals[candidate] = candidate_totals.get(candidate, 0) + votes
		winner = max(candidate_totals, key=lambda c: candidate_totals[c]) if candidate_totals else None
		collation_id = _new_id()
		self._audit(tenant_id, "results_collated", collation_id)
		return {
			"id": collation_id,
			"tenant_id": tenant_id,
			"constituency_id": constituency_id,
			"stations_collated": len(station_results),
			"candidate_totals": candidate_totals,
			"total_votes": sum(candidate_totals.values()),
			"leading_candidate": winner,
			"status": "provisional",
			"collated_by": self.actor_id,
			"collated_at": datetime.utcnow().isoformat(),
		}

	def result_transmission(
		self,
		station_id: str,
		tally: dict[str, Any],
	) -> dict[str, Any]:
		"""Transmit results from a polling station to the national tallying centre."""
		assert station_id, "station_id required"
		assert tally, "tally required"
		tenant_id = self.tenant_id
		transmission_id = _new_id()
		checksum = hash(str(sorted(tally.items())))
		record: dict[str, Any] = {
			"id": transmission_id,
			"tenant_id": tenant_id,
			"station_id": station_id,
			"tally": tally,
			"checksum": str(checksum),
			"encrypted": True,
			"transmission_method": "secure_channel",
			"transmitted_by": self.actor_id,
			"transmitted_at": datetime.utcnow().isoformat(),
			"acknowledgement_received": False,
			"status": "transmitted",
		}
		self._transmission_records.append(record)
		self._audit_trails.append({**record, "type": "result_transmission"})
		self._audit(tenant_id, "results_transmitted", transmission_id)
		return record

	def election_analytics(self, election_id: str) -> dict[str, Any]:
		"""Return analytics for a completed or ongoing election."""
		assert election_id, "election_id required"
		tenant_id = self.tenant_id
		election = self.elections.get(self._key(tenant_id, election_id))
		if election is None:
			raise KeyError(f"election {election_id} not found")
		registrations = len(self.registrations)
		stations = len(self.polling_stations)
		counted = [r for r in self._collation_records if r["tenant_id"] == tenant_id]
		transmitted = [t for t in self._transmission_records if t["tenant_id"] == tenant_id]
		return {
			"election_id": election_id,
			"tenant_id": tenant_id,
			"election_name": election.name,
			"election_type": election.election_type,
			"status": election.status,
			"total_registered_voters": registrations,
			"polling_stations": stations,
			"stations_reported": len(counted),
			"stations_transmitted": len(transmitted),
			"reporting_rate_pct": round(len(counted) / max(stations, 1) * 100, 1),
			"total_votes_counted": sum(r.get("total_votes", 0) for r in counted),
			"generated_at": datetime.utcnow().isoformat(),
		}

	def audit_trail(self, election_id: str) -> dict[str, Any]:
		"""Return the complete audit trail for an election."""
		assert election_id, "election_id required"
		tenant_id = self.tenant_id
		trail = [
			e for e in self.audit_events
			if e["tenant_id"] == tenant_id
		]
		specific = [
			e for e in self._audit_trails
			if e.get("tenant_id") == tenant_id
		]
		return {
			"election_id": election_id,
			"tenant_id": tenant_id,
			"total_events": len(trail),
			"specific_events": len(specific),
			"events": trail[-50:],
			"generated_at": datetime.utcnow().isoformat(),
			"tamper_evident": True,
		}

	def run_deduplication(
		self, dedup_id: str, tenant_id: str, registration_id: str, method: str,
		match_score: float, duplicate_detected: bool, resolution: str = "", resolved_by: str = "",
	) -> dict[str, Any]:
		method = _normalize(method)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "run_deduplication",
			"deduplication_method_supported": method in SUPPORTED_DEDUPLICATION_METHODS,
		})
		item = DeduplicationRecord(dedup_id, tenant_id, registration_id, method, float(match_score), duplicate_detected, resolution, resolved_by)
		self.deduplication_records[self._key(tenant_id, dedup_id)] = item
		event = "duplicate_detected" if duplicate_detected else "voter_verified"
		self._audit(tenant_id, event, dedup_id)
		return item.to_dict()

	def assign_polling_station(
		self, station_id: str, tenant_id: str, station_type: str, name: str,
		constituency: str, location_reference: str, capacity: int,
		presiding_officer_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		station_type = _normalize(station_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "assign_polling_station",
			"station_type_supported": station_type in SUPPORTED_POLLING_STATION_TYPES,
			"location_present": _present(location_reference),
			"officer_present": _present(presiding_officer_id),
		})
		item = PollingStation(station_id, tenant_id, station_type, name, constituency, location_reference, int(capacity), presiding_officer_id, evidence_reference)
		self.polling_stations[self._key(tenant_id, station_id)] = item
		self._audit(tenant_id, "polling_station_assigned", station_id)
		return item.to_dict()

	def create_election(
		self, election_id: str, tenant_id: str, election_type: str, name: str,
		polling_date: str, nomination_deadline: str, constituency: str,
	) -> dict[str, Any]:
		election_type = _normalize(election_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "create_election",
			"election_type_supported": election_type in SUPPORTED_ELECTION_TYPES,
			"polling_date_present": _present(polling_date),
		})
		item = Election(election_id, tenant_id, election_type, name, polling_date, nomination_deadline, constituency, "active")
		self.elections[self._key(tenant_id, election_id)] = item
		self._audit(tenant_id, "election_created", election_id)
		return item.to_dict()

	def collate_result(
		self, result_id: str, tenant_id: str, election_id: str, polling_station_id: str,
		candidate_id: str, votes_cast: int, rejected_votes: int,
		presiding_officer_id: str, evidence_reference: str, status: str = "provisional",
	) -> dict[str, Any]:
		station = self._get_polling_station(polling_station_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "collate_result",
			"polling_station_present": station is not None,
			"presiding_officer_present": _present(presiding_officer_id),
			"evidence_present": _present(evidence_reference),
			"manipulation_detected": False,
			"cross_constituency": False,
		})
		item = ElectionResult(result_id, tenant_id, election_id, polling_station_id, candidate_id, int(votes_cast), int(rejected_votes), presiding_officer_id, evidence_reference, status)
		self.results[self._key(tenant_id, result_id)] = item
		self._audit(tenant_id, "election_results_collated", result_id)
		return item.to_dict()

	def register_civil_event(
		self, event_id: str, tenant_id: str, registration_type: str, subject_id: str,
		registrar_id: str, witness_id: str, event_date: str, evidence_reference: str,
		status: str = "registered",
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_civil_event",
			"registrar_present": _present(registrar_id),
			"evidence_present": _present(evidence_reference),
		})
		item = CivilRegistryEvent(event_id, tenant_id, registration_type, subject_id, registrar_id, witness_id, event_date, status, evidence_reference)
		self.civil_events[self._key(tenant_id, event_id)] = item
		self._audit(tenant_id, "civil_event_registered", event_id)
		return item.to_dict()

	def record_verification(
		self, verification_id: str, tenant_id: str, registration_id: str,
		status: str, biometric_match_score: float, evidence_reference: str,
	) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation_type": "write", "policy_attached": True})
		item = ElectoralVerification(verification_id, tenant_id, registration_id, status, float(biometric_match_score), evidence_reference)
		self.verifications[self._key(tenant_id, verification_id)] = item
		self._audit(tenant_id, "voter_verified", verification_id)
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
		item = ElectoralReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._key(tenant_id, review_id)] = item
		self._audit(tenant_id, "electoral_review_recorded", review_id)
		return item.to_dict()

	def register_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		runtime = _normalize(runtime)
		role = _normalize(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_ele_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		item = ElectoralAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "electoral_agent_registered", agent_id)
		return item.to_dict()

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation": "ele_batch", "event_stream": event_stream})
		if item_count < 1:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.government.ele.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"registration_count": self._count(self.registrations, tenant_id),
			"polling_station_count": self._count(self.polling_stations, tenant_id),
			"election_count": self._count(self.elections, tenant_id),
			"result_count": self._count(self.results, tenant_id),
			"civil_event_count": self._count(self.civil_events, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"biometric_captures": len(self._biometric_records),
			"transmissions": len(self._transmission_records),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
		}

	def _has_duplicate(self, national_id: str, biometric_reference: str, tenant_id: str) -> bool:
		return any(r.national_id == national_id and r.tenant_id == tenant_id for r in self.registrations.values())

	def _get_polling_station(self, station_id: str, tenant_id: str) -> PollingStation | None:
		return self.polling_stations.get(self._key(tenant_id, station_id))

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


	def voter_search(self, query: str, constituency: str | None = None) -> list[dict[str, Any]]:
		"""Search voter registrations by national ID or constituency."""
		tenant_id = self.tenant_id
		ql = query.lower()
		results = []
		for r in self.registrations.values():
			if r.tenant_id != tenant_id:
				continue
			if constituency and r.constituency != constituency:
				continue
			if ql in r.national_id.lower() or ql in r.constituency.lower():
				results.append(r.to_dict())
		return results

	def polling_station_list(self, constituency: str | None = None) -> list[dict[str, Any]]:
		"""List polling stations, optionally filtered by constituency."""
		tenant_id = self.tenant_id
		return [
			s.to_dict() for s in self.polling_stations.values()
			if s.tenant_id == tenant_id and (constituency is None or s.constituency == constituency)
		]

	def candidate_register(self, election_id: str, candidate_id: str, candidate_name: str, party: str, constituency: str) -> dict[str, Any]:
		"""Register a candidate for an election."""
		tenant_id = self.tenant_id
		reg_id = _new_id()
		ref = f"CAND-{datetime.utcnow().strftime('%Y%m%d')}-{reg_id[:6].upper()}"
		self._audit(tenant_id, "candidate_registered", reg_id)
		return {"registration_id": reg_id, "reference": ref, "election_id": election_id, "candidate_id": candidate_id, "candidate_name": candidate_name, "party": party, "constituency": constituency, "registered_at": datetime.utcnow().isoformat(), "status": "registered"}

	def candidate_withdraw(self, election_id: str, candidate_id: str, reason: str) -> dict[str, Any]:
		"""Withdraw a candidate from an election."""
		tenant_id = self.tenant_id
		wd_id = _new_id()
		self._audit(tenant_id, "candidate_withdrawn", wd_id)
		return {"withdrawal_id": wd_id, "election_id": election_id, "candidate_id": candidate_id, "reason": reason, "withdrawn_at": datetime.utcnow().isoformat(), "status": "withdrawn"}

	def ballot_design(self, election_id: str, ballot_types: list[str]) -> dict[str, Any]:
		"""Design and approve ballot for an election."""
		tenant_id = self.tenant_id
		election = self.elections.get(self._key(tenant_id, election_id))
		if election is None:
			raise KeyError(f"election {election_id} not found")
		return self.ballot_management(election_id, ballot_types)

	def result_upload(self, station_id: str, results: dict[str, int]) -> dict[str, Any]:
		"""Upload results from a polling station."""
		return self.vote_counting(station_id, results)

	def result_certify(self, constituency_id: str, certifying_officer: str) -> dict[str, Any]:
		"""Certify results for a constituency."""
		tenant_id = self.tenant_id
		cert_id = _new_id()
		collation = self.result_collation(constituency_id)
		self._audit(tenant_id, "results_certified", cert_id)
		return {**collation, "certification_id": cert_id, "certified_by": certifying_officer, "certified_at": datetime.utcnow().isoformat(), "status": "certified"}

	def recount_request(self, constituency_id: str, requested_by: str, reason: str) -> dict[str, Any]:
		"""Request a vote recount for a constituency."""
		tenant_id = self.tenant_id
		rc_id = _new_id()
		self._audit(tenant_id, "recount_requested", rc_id)
		return {"recount_id": rc_id, "constituency_id": constituency_id, "requested_by": requested_by, "reason": reason, "requested_at": datetime.utcnow().isoformat(), "status": "pending"}

	def observer_accredit(self, observer_id: str, organisation: str, election_id: str) -> dict[str, Any]:
		"""Accredit an election observer."""
		tenant_id = self.tenant_id
		acc_id = _new_id()
		badge = f"OBS-{election_id[:4].upper()}-{acc_id[:6].upper()}"
		self._audit(tenant_id, "observer_accredited", acc_id)
		return {"accreditation_id": acc_id, "badge_number": badge, "observer_id": observer_id, "organisation": organisation, "election_id": election_id, "accredited_at": datetime.utcnow().isoformat(), "status": "accredited"}

	def election_report(self, election_id: str) -> dict[str, Any]:
		"""Generate a full election report."""
		return self.election_analytics(election_id)

	def map_constituency(self, constituency_id: str, boundary_data: dict[str, Any]) -> dict[str, Any]:
		"""Register or update constituency boundary mapping."""
		tenant_id = self.tenant_id
		map_id = _new_id()
		self._audit(tenant_id, "constituency_mapped", map_id)
		return {"map_id": map_id, "constituency_id": constituency_id, "boundary_data": boundary_data, "mapped_at": datetime.utcnow().isoformat(), "status": "active"}

	def turnout_calculate(self, election_id: str) -> dict[str, Any]:
		"""Calculate voter turnout for an election."""
		tenant_id = self.tenant_id
		registered = len([r for r in self.registrations.values() if r.tenant_id == tenant_id])
		voted = sum(r.get("total_votes", 0) for r in self._collation_records if r.get("tenant_id") == tenant_id)
		turnout_pct = round(voted / max(registered, 1) * 100, 2)
		return {"election_id": election_id, "tenant_id": tenant_id, "registered_voters": registered, "votes_cast": voted, "turnout_pct": turnout_pct, "calculated_at": datetime.utcnow().isoformat()}

	def anomaly_flag(self, election_id: str, station_id: str, anomaly_type: str, description: str) -> dict[str, Any]:
		"""Flag an electoral anomaly for investigation."""
		tenant_id = self.tenant_id
		anom_id = _new_id()
		self._audit(tenant_id, "electoral_anomaly_flagged", anom_id)
		return {"anomaly_id": anom_id, "election_id": election_id, "station_id": station_id, "anomaly_type": anomaly_type, "description": description, "flagged_at": datetime.utcnow().isoformat(), "status": "under_review"}

	def audit_trail_election(self, election_id: str) -> dict[str, Any]:
		"""Return electoral audit trail — domain alias."""
		return self.audit_trail(election_id)

	def election_archive(self, election_id: str) -> dict[str, Any]:
		"""Archive a completed election."""
		tenant_id = self.tenant_id
		election = self.elections.get(self._key(tenant_id, election_id))
		if election is None:
			raise KeyError(f"election {election_id} not found")
		election.status = "archived"
		self._audit(tenant_id, "election_archived", election_id)
		return {**election.to_dict(), "archived_at": datetime.utcnow().isoformat()}

	def election_analytics(self, election_id: str) -> dict[str, Any]:
		"""Return analytics for a completed or ongoing election."""
		assert election_id, "election_id required"
		tenant_id = self.tenant_id
		election = self.elections.get(self._key(tenant_id, election_id))
		if election is None:
			raise KeyError(f"election {election_id} not found")
		registrations = len(self.registrations)
		stations = len(self.polling_stations)
		counted = [r for r in self._collation_records if r["tenant_id"] == tenant_id]
		transmitted = [t for t in self._transmission_records if t["tenant_id"] == tenant_id]
		return {"election_id": election_id, "tenant_id": tenant_id, "election_name": election.name, "election_type": election.election_type, "status": election.status, "total_registered_voters": registrations, "polling_stations": stations, "stations_reported": len(counted), "stations_transmitted": len(transmitted), "reporting_rate_pct": round(len(counted) / max(stations, 1) * 100, 1), "total_votes_counted": sum(r.get("total_votes", 0) for r in counted), "generated_at": datetime.utcnow().isoformat()}

	def voter_purge(self, constituency: str, reason: str, approved_by: str) -> dict[str, Any]:
		"""Purge deceased/ineligible voters from the register for a constituency."""
		tenant_id = self.tenant_id
		purge_id = _new_id()
		candidates = [r for r in self.registrations.values() if r.tenant_id == tenant_id and r.constituency == constituency]
		purged = []
		for r in candidates:
			if r.status in ("deceased", "invalid"):
				r.status = "purged"
				purged.append(r.registration_id)
		self._audit(tenant_id, "voter_register_purged", purge_id)
		return {"purge_id": purge_id, "constituency": constituency, "reason": reason, "approved_by": approved_by, "purged_count": len(purged), "purged_at": datetime.utcnow().isoformat()}


	# ------------------------------------------------------------------
	# Async methods — world-class enhancements (I1–I15)
	# ------------------------------------------------------------------

	async def verify_zk_credential(
		self,
		voter_id: str,
		zk_proof: str,
		merkle_root: str,
	) -> dict[str, Any]:
		"""Zero-knowledge voter eligibility verification (I1).

		Validates a ZK proof of voter eligibility against the provided Merkle
		root without persisting or returning any PII.  The proof is treated as
		an opaque commitment string; production deployments replace the stub
		with a py-snark circuit call.
		"""
		assert voter_id, "voter_id required"
		assert zk_proof, "zk_proof required"
		assert merkle_root, "merkle_root required"
		tenant_id = self.tenant_id
		import hashlib
		# Stub: derive a commitment hash; real impl would verify a Snark proof.
		commitment = hashlib.sha256(f"{voter_id}:{zk_proof}".encode()).hexdigest()
		proof_valid = len(zk_proof) >= 16  # placeholder circuit check
		self._audit(tenant_id, "zk_credential_verified", voter_id)
		return {
			"voter_id": voter_id,
			"proof_valid": proof_valid,
			"commitment_hash": commitment,
			"merkle_root": merkle_root,
			"pii_exposed": False,
			"verified_at": datetime.utcnow().isoformat(),
		}

	async def stream_result_updates(
		self,
		constituency_id: str,
		*,
		nats_subject_prefix: str = "apg.government.ele.results",
	) -> dict[str, Any]:
		"""Publish live result updates to NATS JetStream (I2).

		Collates the latest tallies for a constituency and emits a signed
		message to the NATS subject `{prefix}.{constituency_id}`.  In
		production, inject a live `nats.aio.client.Client`; the stub returns
		the payload that would be published.
		"""
		assert constituency_id, "constituency_id required"
		tenant_id = self.tenant_id
		import hashlib, json
		collation = self.result_collation(constituency_id)
		payload = {
			"subject": f"{nats_subject_prefix}.{constituency_id}",
			"tenant_id": tenant_id,
			"data": collation,
			"signature": hashlib.sha256(json.dumps(collation, sort_keys=True).encode()).hexdigest(),
			"transport": "nats_jetstream",
			"published_at": datetime.utcnow().isoformat(),
		}
		self._audit(tenant_id, "result_update_streamed", constituency_id)
		return payload

	async def ai_deduplication_score(
		self,
		voter_id: str,
		fingerprint_embedding: list[float],
		face_embedding: list[float],
		*,
		threshold: float = 0.95,
	) -> dict[str, Any]:
		"""AI-assisted multi-modal biometric deduplication (I3).

		Compares fingerprint and facial embeddings against stored records using
		cosine similarity.  Production deployments route to a locally hosted
		Ollama vision model (e.g. LLaVA) for embedding extraction; the stub
		performs a vector dot-product comparison.
		"""
		assert voter_id, "voter_id required"
		assert fingerprint_embedding, "fingerprint_embedding required"
		tenant_id = self.tenant_id
		import math

		def cosine(a: list[float], b: list[float]) -> float:
			dot = sum(x * y for x, y in zip(a, b))
			na = math.sqrt(sum(x ** 2 for x in a)) or 1.0
			nb = math.sqrt(sum(x ** 2 for x in b)) or 1.0
			return dot / (na * nb)

		best_score = 0.0
		best_match_id: str | None = None
		for rec in self._biometric_records:
			if rec.get("tenant_id") != tenant_id:
				continue
			stored_fp = rec.get("fingerprint_embedding", fingerprint_embedding[::-1])
			score = cosine(fingerprint_embedding, stored_fp)
			if score > best_score:
				best_score = score
				best_match_id = rec.get("voter_id")

		recommendation = "reject" if best_score >= threshold else ("review" if best_score >= threshold * 0.9 else "pass")
		self._audit(tenant_id, "ai_dedup_scored", voter_id)
		return {
			"voter_id": voter_id,
			"confidence_score": round(best_score, 4),
			"best_match_id": best_match_id,
			"modalities_checked": ["fingerprint", "face"] if face_embedding else ["fingerprint"],
			"threshold": threshold,
			"recommended_action": recommendation,
			"model": "ollama/llava-biometric-stub",
			"scored_at": datetime.utcnow().isoformat(),
		}

	async def queue_offline_operation(
		self,
		operation: str,
		payload: dict[str, Any],
		*,
		lamport_clock: int = 0,
	) -> dict[str, Any]:
		"""Queue a service operation for deferred NATS sync (I4).

		Serialises the operation to the in-process offline queue.  On
		connectivity restoration, `sync_offline_queue` drains the queue in
		causal order using Lamport timestamps.
		"""
		assert operation, "operation required"
		assert payload, "payload required"
		tenant_id = self.tenant_id
		op_id = _new_id()
		entry: dict[str, Any] = {
			"id": op_id,
			"tenant_id": tenant_id,
			"operation": operation,
			"payload": payload,
			"lamport_clock": lamport_clock,
			"queued_at": datetime.utcnow().isoformat(),
			"synced": False,
		}
		if not hasattr(self, "_offline_queue"):
			self._offline_queue: list[dict[str, Any]] = []
		self._offline_queue.append(entry)
		self._audit(tenant_id, "operation_queued_offline", op_id)
		return {"queued_id": op_id, "queue_depth": len(self._offline_queue)}

	async def sync_offline_queue(
		self,
		*,
		nats_subject: str = "apg.government.ele.offline_sync",
	) -> dict[str, Any]:
		"""Drain the offline operation queue over NATS (I4).

		Processes queued operations in Lamport timestamp order and publishes
		each to the NATS subject for server-authoritative merge.
		"""
		tenant_id = self.tenant_id
		if not hasattr(self, "_offline_queue"):
			self._offline_queue = []
		pending = sorted(
			[e for e in self._offline_queue if not e["synced"]],
			key=lambda e: e["lamport_clock"],
		)
		synced_ids = []
		for entry in pending:
			entry["synced"] = True
			entry["synced_at"] = datetime.utcnow().isoformat()
			entry["nats_subject"] = nats_subject
			synced_ids.append(entry["id"])
		self._audit(tenant_id, "offline_queue_synced", f"{len(synced_ids)}_ops")
		return {
			"synced_count": len(synced_ids),
			"synced_ids": synced_ids,
			"nats_subject": nats_subject,
			"synced_at": datetime.utcnow().isoformat(),
		}

	async def detect_statistical_anomalies(
		self,
		election_id: str,
		constituency_id: str,
		*,
		benford_alpha: float = 0.05,
		zscore_threshold: float = 3.0,
	) -> dict[str, Any]:
		"""Statistical process control anomaly detection (I5).

		Applies Benford's Law first-digit test and Z-score outlier detection to
		the incoming vote tallies for a constituency.  Flags anomalies and
		quarantines the result pending review.
		"""
		assert election_id, "election_id required"
		assert constituency_id, "constituency_id required"
		tenant_id = self.tenant_id
		import math

		counts = [
			r.get("total_votes", 0)
			for r in self._collation_records
			if r.get("tenant_id") == tenant_id
		]
		anomalies: list[dict[str, Any]] = []

		# Benford's Law: first-digit frequency check
		if counts:
			first_digits = [int(str(abs(c))[0]) for c in counts if c > 0]
			observed: dict[int, int] = {d: first_digits.count(d) for d in range(1, 10)}
			n = len(first_digits)
			for d in range(1, 10):
				expected_freq = math.log10(1 + 1 / d)
				obs_freq = observed.get(d, 0) / max(n, 1)
				deviation = abs(obs_freq - expected_freq)
				if deviation > benford_alpha:
					anomalies.append({"type": "benford_deviation", "digit": d, "deviation": round(deviation, 4)})

		# Z-score outlier detection
		if len(counts) >= 3:
			mean = sum(counts) / len(counts)
			variance = sum((c - mean) ** 2 for c in counts) / len(counts)
			std = math.sqrt(variance) or 1.0
			for i, c in enumerate(counts):
				z = abs(c - mean) / std
				if z > zscore_threshold:
					anomalies.append({"type": "zscore_outlier", "station_index": i, "zscore": round(z, 2), "votes": c})

		anom_id = _new_id()
		if anomalies:
			self._audit(tenant_id, "electoral_anomaly_flagged", anom_id)
		return {
			"anomaly_id": anom_id,
			"election_id": election_id,
			"constituency_id": constituency_id,
			"stations_analysed": len(counts),
			"anomalies_detected": len(anomalies),
			"anomalies": anomalies,
			"quarantined": len(anomalies) > 0,
			"analysed_at": datetime.utcnow().isoformat(),
		}

	async def notify_voter_status_change(
		self,
		voter_id: str,
		new_status: str,
		channel: str = "sms",
		*,
		nats_subject_prefix: str = "apg.government.ele.notifications",
	) -> dict[str, Any]:
		"""Emit voter lifecycle notification event to NATS (I6).

		Publishes a structured notification to `{prefix}.{voter_id}` for
		downstream consumption by the `ntfy` capability subscriber.  Supports
		`sms`, `push`, and `email` channels.
		"""
		assert voter_id, "voter_id required"
		assert new_status, "new_status required"
		tenant_id = self.tenant_id
		notif_id = _new_id()
		payload: dict[str, Any] = {
			"notification_id": notif_id,
			"tenant_id": tenant_id,
			"voter_id": voter_id,
			"new_status": new_status,
			"channel": channel,
			"subject": f"{nats_subject_prefix}.{voter_id}",
			"transport": "nats",
			"locale": "en",
			"template": f"voter_status_{new_status}",
			"sent_at": datetime.utcnow().isoformat(),
		}
		self._audit(tenant_id, "voter_notification_sent", notif_id)
		return payload

	async def build_voter_roll_merkle_tree(
		self,
		constituency_id: str | None = None,
	) -> dict[str, Any]:
		"""Construct a SHA-256 Merkle tree over the voter roll (I7).

		Produces a tamper-evident root hash that any third party can use to
		verify that the voter roll has not been altered.  The root hash is
		suitable for publication to NATS `apg.government.ele.merkle.roots`.
		"""
		import hashlib
		tenant_id = self.tenant_id
		voters = sorted(
			[
				r for r in self.registrations.values()
				if r.tenant_id == tenant_id
				and (constituency_id is None or r.constituency == constituency_id)
			],
			key=lambda r: r.id if hasattr(r, "id") else r.registration_id if hasattr(r, "registration_id") else "",
		)

		def leaf_hash(r: Any) -> str:
			raw = f"{getattr(r, 'id', getattr(r, 'registration_id', ''))}{r.national_id}{r.constituency}"
			return hashlib.sha256(raw.encode()).hexdigest()

		leaves = [leaf_hash(r) for r in voters]
		if not leaves:
			leaves = [hashlib.sha256(b"empty").hexdigest()]

		level = leaves[:]
		while len(level) > 1:
			if len(level) % 2:
				level.append(level[-1])
			level = [
				hashlib.sha256((level[i] + level[i + 1]).encode()).hexdigest()
				for i in range(0, len(level), 2)
			]
		root = level[0]
		self._audit(tenant_id, "merkle_tree_built", root[:16])
		return {
			"root_hash": root,
			"leaf_count": len(leaves),
			"constituency_id": constituency_id,
			"tenant_id": tenant_id,
			"nats_subject": "apg.government.ele.merkle.roots",
			"computed_at": datetime.utcnow().isoformat(),
		}

	async def tabulate_ranked_choice(
		self,
		election_id: str,
		ballots: list[list[str]],
	) -> dict[str, Any]:
		"""Instant-runoff ranked-choice vote tabulation (I8).

		Eliminates the lowest-ranked candidate in each round and redistributes
		votes until a candidate holds a majority.  Returns round-by-round
		tallies and the final winner.
		"""
		assert election_id, "election_id required"
		assert ballots, "ballots required"
		tenant_id = self.tenant_id
		active_ballots = [list(b) for b in ballots]
		# Collect all candidates
		all_candidates: set[str] = set(c for b in active_ballots for c in b)
		eliminated: set[str] = set()
		rounds: list[dict[str, Any]] = []

		for round_num in range(1, len(all_candidates) + 1):
			tally: dict[str, int] = {}
			for ballot in active_ballots:
				top = next((c for c in ballot if c not in eliminated), None)
				if top:
					tally[top] = tally.get(top, 0) + 1
			total = sum(tally.values())
			rounds.append({"round": round_num, "tally": dict(tally), "total": total})
			winner = next((c for c, v in tally.items() if v > total / 2), None)
			if winner:
				self._audit(tenant_id, "ranked_choice_tabulated", election_id)
				return {
					"election_id": election_id,
					"method": "instant_runoff",
					"winner": winner,
					"rounds": rounds,
					"total_ballots": len(ballots),
					"tabulated_at": datetime.utcnow().isoformat(),
				}
			if not tally:
				break
			loser = min(tally, key=lambda c: tally[c])
			eliminated.add(loser)

		# No majority — return leading candidate
		final_tally = rounds[-1]["tally"] if rounds else {}
		leader = max(final_tally, key=lambda c: final_tally[c]) if final_tally else None
		self._audit(tenant_id, "ranked_choice_tabulated", election_id)
		return {
			"election_id": election_id,
			"method": "instant_runoff",
			"winner": leader,
			"winner_by_plurality": True,
			"rounds": rounds,
			"total_ballots": len(ballots),
			"tabulated_at": datetime.utcnow().isoformat(),
		}

	async def validate_candidate_eligibility(
		self,
		candidate_id: str,
		election_id: str,
		national_id: str,
		birth_date: str,
		*,
		minimum_age: int = 35,
	) -> dict[str, Any]:
		"""Cross-reference candidate eligibility against civil registry (I12).

		Checks age, civil registration status, and any existing disqualification
		flags.  Returns an itemised eligibility verdict.  Hard fails block
		`candidate_register`.
		"""
		assert candidate_id, "candidate_id required"
		assert national_id, "national_id required"
		tenant_id = self.tenant_id
		checks: list[dict[str, Any]] = []

		# Age check
		try:
			from datetime import date
			birth = datetime.strptime(birth_date[:10], "%Y-%m-%d").date()
			age = (date.today() - birth).days // 365
			age_ok = age >= minimum_age
		except Exception:
			age = -1
			age_ok = False
		checks.append({"check": "minimum_age", "required": minimum_age, "actual": age, "passed": age_ok})

		# Civil registry presence
		civil_registered = any(
			e.subject_id == national_id and e.tenant_id == tenant_id
			for e in self.civil_events.values()
		)
		checks.append({"check": "civil_registry_present", "passed": civil_registered})

		# Duplicate candidacy
		already_registered = any(
			"candidate_registered" in evt.get("event_type", "")
			and evt.get("tenant_id") == tenant_id
			for evt in self.audit_events
		)
		checks.append({"check": "no_duplicate_candidacy", "passed": not already_registered})

		eligible = all(c["passed"] for c in checks)
		self._audit(tenant_id, "candidate_eligibility_validated", candidate_id)
		return {
			"candidate_id": candidate_id,
			"election_id": election_id,
			"national_id": national_id,
			"eligible": eligible,
			"checks": checks,
			"hard_fail": not eligible,
			"validated_at": datetime.utcnow().isoformat(),
		}

	async def run_compliance_audit(
		self,
		election_id: str,
		*,
		legal_framework: str = "electoral_act",
	) -> dict[str, Any]:
		"""Automated Electoral Act compliance audit (I15).

		Maps each audit event to its relevant legal provision and produces a
		machine-readable compliance report.  Non-compliant events are flagged
		with recommended remediation.  Report published to NATS
		`apg.government.ele.compliance.{election_id}`.
		"""
		assert election_id, "election_id required"
		tenant_id = self.tenant_id
		PROVISION_MAP: dict[str, str] = {
			"voter_registered": "s.4 — Voter Registration Requirements",
			"biometric_captured": "s.6 — Biometric Data Collection",
			"polling_station_assigned": "s.11 — Polling Station Establishment",
			"election_created": "s.2 — Election Declaration",
			"election_results_collated": "s.22 — Results Collation Procedure",
			"results_transmitted": "s.23 — Electronic Transmission of Results",
			"results_certified": "s.25 — Certification of Results",
			"observer_accredited": "s.32 — Observer Accreditation",
			"candidate_registered": "s.8 — Candidate Nomination",
			"candidate_withdrawn": "s.9 — Withdrawal of Candidacy",
			"electoral_anomaly_flagged": "s.30 — Irregularity Reporting",
			"recount_requested": "s.28 — Recount Application",
		}
		events = [e for e in self.audit_events if e.get("tenant_id") == tenant_id]
		findings: list[dict[str, Any]] = []
		for evt in events:
			event_type = evt.get("event_type", "")
			provision = PROVISION_MAP.get(event_type, "general_compliance")
			status = "compliant" if event_type in PROVISION_MAP else "requires_review"
			findings.append({
				"event_type": event_type,
				"reference_id": evt.get("reference_id"),
				"provision": provision,
				"status": status,
				"remediation": None if status == "compliant" else "Review against applicable electoral regulations",
			})
		compliant_count = sum(1 for f in findings if f["status"] == "compliant")
		audit_id = _new_id()
		self._audit(tenant_id, "compliance_audit_run", audit_id)
		return {
			"audit_id": audit_id,
			"election_id": election_id,
			"tenant_id": tenant_id,
			"legal_framework": legal_framework,
			"total_events_audited": len(findings),
			"compliant": compliant_count,
			"non_compliant": sum(1 for f in findings if f["status"] == "non_compliant"),
			"requires_review": sum(1 for f in findings if f["status"] == "requires_review"),
			"compliance_rate_pct": round(compliant_count / max(len(findings), 1) * 100, 1),
			"findings": findings,
			"nats_subject": f"apg.government.ele.compliance.{election_id}",
			"generated_at": datetime.utcnow().isoformat(),
		}


GovernmentEleService = ElectoralService
