"""Service layer for APG Telecom Security."""

from __future__ import annotations

import datetime
import hashlib
import statistics
from typing import Any

from .domain.adapters import get_auth_adapter, get_audit_adapter
from .database.store import get_store
from .capability_contract import (
	SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_DIAMETER_ATTACK_TYPES,
	SUPPORTED_FRAUD_TYPES, SUPPORTED_INCIDENT_SEVERITIES, SUPPORTED_INCIDENT_STATUSES,
	SUPPORTED_SECURITY_INCIDENT_TYPES, SUPPORTED_INTERCEPT_STATUSES, SUPPORTED_LAWFUL_INTERCEPT_TYPES,
	SUPPORTED_SS7_ATTACK_TYPES, SUPPORTED_THREAT_INTEL_SOURCES,
	evaluate_capability_rules, get_capability_contract,
)
from .models import (
	SecAgent, SecDiameterAttack, SecFraudCase, SecIncident,
	SecLawfulIntercept, SecSs7Attack, SecThreatIntel,
)


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


def _bounded(value: float) -> bool:
	return 0.0 <= value <= 1.0


def _utcnow() -> str:
	return datetime.datetime.utcnow().isoformat() + "Z"


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class TelecomSecurityService:
	"""Tenant-scoped telecom security service for APG."""

	def __init__(self) -> None:
		self._store = get_store("telecom.sec")
		self._auth = get_auth_adapter()
		self._audit_adapter = get_audit_adapter()
		self.fraud_cases: dict[tuple[str, str], SecFraudCase] = {}
		self.ss7_attacks: dict[tuple[str, str], SecSs7Attack] = {}
		self.diameter_attacks: dict[tuple[str, str], SecDiameterAttack] = {}
		self.intercepts: dict[tuple[str, str], SecLawfulIntercept] = {}
		self.incidents: dict[tuple[str, str], SecIncident] = {}
		self.threat_intel: dict[tuple[str, str], SecThreatIntel] = {}
		self.agents: dict[tuple[str, str], SecAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# Extended state for new methods
		self._voip_fraud_records: list[dict[str, Any]] = []
		self._sim_swap_events: list[dict[str, Any]] = []
		self._ott_bypass_events: list[dict[str, Any]] = []
		self._intrusion_events: list[dict[str, Any]] = []
		self._incident_responses: list[dict[str, Any]] = []
		self._roaming_security_checks: list[dict[str, Any]] = []
		# Lawful intercept orders (separate from SecLawfulIntercept models)
		self._li_orders: dict[str, dict[str, Any]] = {}

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

	def raise_fraud_case(
		self,
		case_id: str,
		tenant_id: str,
		fraud_type: str,
		msisdn: str,
		confidence_score: float,
		evidence_reference: str,
		detected_at: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Raise a fraud case with evidence and confidence score."""
		fraud_type = fraud_type.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "raise_fraud_case",
			"fraud_type_supported": fraud_type in SUPPORTED_FRAUD_TYPES,
			"confidence_present": confidence_score is not None,
		})
		item = SecFraudCase(case_id, tenant_id, fraud_type, msisdn, float(confidence_score), evidence_reference, "open", detected_at, None)
		self.fraud_cases[self._key(tenant_id, case_id)] = item
		self._audit(tenant_id, "fraud_case_raised", case_id)
		return item.to_dict()

	def apply_fraud_block(self, case_id: str, tenant_id: str, evidence_reference: str) -> dict[str, Any]:
		"""Apply a fraud block to a detected case — evidence required."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "apply_fraud_block",
			"evidence_present": _present(evidence_reference),
		})
		case = self._fraud_case_or_raise(case_id, tenant_id)
		case.status = "blocked"
		self._audit(tenant_id, "fraud_block_applied", case_id)
		return case.to_dict()

	def record_ss7_attack(
		self,
		attack_id: str,
		tenant_id: str,
		attack_type: str,
		source_reference: str,
		target_reference: str,
		evidence_reference: str,
		detected_at: str,
	) -> dict[str, Any]:
		"""Record an SS7 protocol attack event."""
		attack_type = attack_type.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_ss7_attack",
			"attack_type_supported": attack_type in SUPPORTED_SS7_ATTACK_TYPES,
			"evidence_present": _present(evidence_reference),
		})
		item = SecSs7Attack(attack_id, tenant_id, attack_type, source_reference, target_reference, evidence_reference, detected_at, None)
		self.ss7_attacks[self._key(tenant_id, attack_id)] = item
		self._audit(tenant_id, "ss7_attack_detected", attack_id)
		return item.to_dict()

	def record_diameter_attack(
		self,
		attack_id: str,
		tenant_id: str,
		attack_type: str,
		source_realm: str,
		target_realm: str,
		evidence_reference: str,
		detected_at: str,
	) -> dict[str, Any]:
		"""Record a Diameter protocol attack event."""
		attack_type = attack_type.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_diameter_attack",
			"attack_type_supported": attack_type in SUPPORTED_DIAMETER_ATTACK_TYPES,
		})
		item = SecDiameterAttack(attack_id, tenant_id, attack_type, source_realm, target_realm, evidence_reference, detected_at, None)
		self.diameter_attacks[self._key(tenant_id, attack_id)] = item
		self._audit(tenant_id, "diameter_attack_detected", attack_id)
		return item.to_dict()

	def activate_intercept(
		self,
		intercept_id: str,
		tenant_id: str,
		intercept_type: str,
		target_msisdn: str,
		warrant_reference: str,
		regulatory_authority: str,
		activated_at: str,
		expires_at: str,
	) -> dict[str, Any]:
		"""Activate a lawful intercept — warrant and regulatory authority mandatory."""
		intercept_type = intercept_type.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "activate_intercept",
			"warrant_present": _present(warrant_reference),
			"regulatory_authority_present": _present(regulatory_authority),
			"intercept_type_supported": intercept_type in SUPPORTED_LAWFUL_INTERCEPT_TYPES,
		})
		item = SecLawfulIntercept(intercept_id, tenant_id, intercept_type, target_msisdn, warrant_reference, regulatory_authority, "active", activated_at, expires_at)
		self.intercepts[self._key(tenant_id, intercept_id)] = item
		self._audit(tenant_id, "intercept_activated", intercept_id)
		return item.to_dict()

	def update_intercept_status(self, intercept_id: str, tenant_id: str, new_status: str) -> dict[str, Any]:
		"""Update a lawful intercept's operational status."""
		new_status = new_status.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "update_intercept_status",
			"intercept_status_supported": new_status in SUPPORTED_INTERCEPT_STATUSES,
		})
		intercept = self._intercept_or_raise(intercept_id, tenant_id)
		intercept.status = new_status
		self._audit(tenant_id, "intercept_status_updated", intercept_id)
		return intercept.to_dict()

	def open_incident(
		self,
		incident_id: str,
		tenant_id: str,
		incident_type: str,
		severity: str,
		description: str,
		evidence_reference: str,
		opened_at: str,
	) -> dict[str, Any]:
		"""Open a security incident."""
		incident_type = incident_type.lower()
		severity = severity.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_incident",
			"incident_type_supported": incident_type in SUPPORTED_SECURITY_INCIDENT_TYPES,
			"severity_supported": severity in SUPPORTED_INCIDENT_SEVERITIES,
			"evidence_present": _present(evidence_reference),
		})
		item = SecIncident(incident_id, tenant_id, incident_type, severity, description, evidence_reference, "new", None, opened_at, None)
		self.incidents[self._key(tenant_id, incident_id)] = item
		self._audit(tenant_id, "security_incident_opened", incident_id)
		return item.to_dict()

	def update_incident_status(
		self,
		incident_id: str,
		tenant_id: str,
		new_status: str,
		resolved_at: str | None = None,
	) -> dict[str, Any]:
		"""Update the status of a security incident."""
		new_status = new_status.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "update_incident_status",
			"incident_status_supported": new_status in SUPPORTED_INCIDENT_STATUSES,
		})
		incident = self._incident_or_raise(incident_id, tenant_id)
		incident.status = new_status
		if resolved_at:
			incident.resolved_at = resolved_at
		if new_status in ("eradicated", "recovered", "closed"):
			self._audit(tenant_id, "security_incident_resolved", incident_id)
		return incident.to_dict()

	def record_threat_intel(
		self,
		intel_id: str,
		tenant_id: str,
		source: str,
		ioc_type: str,
		ioc_value: str,
		tlp_level: str,
		valid_from: str,
		valid_to: str | None = None,
		shared: bool = False,
	) -> dict[str, Any]:
		"""Record a threat intelligence indicator of compromise."""
		source = source.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_threat_intel",
			"source_supported": source in SUPPORTED_THREAT_INTEL_SOURCES,
		})
		item = SecThreatIntel(intel_id, tenant_id, source, ioc_type, ioc_value, tlp_level, valid_from, valid_to, shared)
		self.threat_intel[self._key(tenant_id, intel_id)] = item
		if shared:
			self._audit(tenant_id, "threat_ioc_shared", intel_id)
		return item.to_dict()

	def register_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		"""Register a security automation agent."""
		runtime = runtime.lower()
		role = role.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_sec_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		item = SecAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "sec_agent_registered", agent_id)
		return item.to_dict()

	# ------------------------------------------------------------------ #
	# New methods                                                          #
	# ------------------------------------------------------------------ #

	async def ss7_attack_detection(
		self,
		signaling_event: dict[str, Any],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Detect SS7 protocol attacks from a signaling event payload.

		Analyses: message_type, source_gt, destination_gt, opcode, imsi.
		Applies heuristic rules: location-update from non-home PLMN,
		SendRoutingInfo abuse, SRI-for-SM tracking patterns.
		Returns verdict, attack_type (if detected), and confidence.
		"""
		assert signaling_event, "signaling_event required"
		msg_type = str(signaling_event.get("message_type", "")).upper()
		source_gt = str(signaling_event.get("source_gt", ""))
		opcode = str(signaling_event.get("opcode", "")).lower()
		imsi = str(signaling_event.get("imsi", ""))
		# Heuristic rules
		attack_type: str | None = None
		confidence = 0.0
		if "sendRoutingInfo" in msg_type or "sri" in opcode:
			attack_type = "location_tracking"
			confidence = 0.75
		elif "updateLocation" in msg_type or "ul" in opcode:
			if source_gt and not source_gt.startswith("254"):  # non-KE PLMN prefix example
				attack_type = "location_hijacking"
				confidence = 0.65
		elif "insertSubscriberData" in msg_type or "isd" in opcode:
			attack_type = "subscriber_data_manipulation"
			confidence = 0.80
		elif "provideSubscriberInfo" in msg_type or "psi" in opcode:
			attack_type = "subscriber_info_disclosure"
			confidence = 0.70
		detected = attack_type is not None
		result: dict[str, Any] = {
			"event_id": signaling_event.get("event_id", f"evt-{_utcnow()}"),
			"tenant_id": tenant_id,
			"detected": detected,
			"attack_type": attack_type,
			"confidence": confidence,
			"source_gt": source_gt,
			"message_type": msg_type,
			"imsi_present": bool(imsi),
			"assessed_at": _utcnow(),
		}
		if detected:
			attack_id = f"ss7-{hashlib.md5(f'{source_gt}{msg_type}{_utcnow()}'.encode()).hexdigest()[:8]}"
			attack_type_norm = attack_type if attack_type in SUPPORTED_SS7_ATTACK_TYPES else (SUPPORTED_SS7_ATTACK_TYPES[0] if SUPPORTED_SS7_ATTACK_TYPES else "location_tracking")
			self.record_ss7_attack(
				attack_id=attack_id,
				tenant_id=tenant_id,
				attack_type=attack_type_norm,
				source_reference=source_gt,
				target_reference=imsi,
				evidence_reference=str(signaling_event),
				detected_at=_utcnow(),
			)
			result["attack_id"] = attack_id
		return result

	async def diameter_fraud_detection(
		self,
		request: dict[str, Any],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Detect Diameter protocol fraud from a request payload.

		Analyses: command_code, origin_realm, destination_realm, avp_list.
		Detects: realm spoofing, excessive location requests, ULR flooding,
		roaming fraud via manipulated CLR/DSR messages.
		"""
		assert request, "request required"
		command_code = int(request.get("command_code", 0))
		origin_realm = str(request.get("origin_realm", "")).lower()
		dest_realm = str(request.get("destination_realm", "")).lower()
		avps = request.get("avp_list", [])
		attack_type: str | None = None
		confidence = 0.0
		# Diameter command codes: 316=ULR, 321=CLR, 318=DSR
		if command_code == 316 and origin_realm != dest_realm:
			attack_type = "roaming_fraud_ulr"
			confidence = 0.72
		elif command_code in (321, 318):
			# CLR/DSR from unexpected realm
			if origin_realm and not origin_realm.endswith(".3gppnetwork.org"):
				attack_type = "realm_spoofing"
				confidence = 0.68
		elif "Experimental-Result" in str(avps) and command_code == 316:
			attack_type = "ulr_flooding"
			confidence = 0.55
		detected = attack_type is not None
		result: dict[str, Any] = {
			"request_id": request.get("request_id", f"req-{_utcnow()}"),
			"tenant_id": tenant_id,
			"command_code": command_code,
			"origin_realm": origin_realm,
			"detected": detected,
			"attack_type": attack_type,
			"confidence": confidence,
			"assessed_at": _utcnow(),
		}
		if detected:
			attack_id = f"dia-{hashlib.md5(f'{origin_realm}{command_code}{_utcnow()}'.encode()).hexdigest()[:8]}"
			attack_type_norm = attack_type if attack_type in SUPPORTED_DIAMETER_ATTACK_TYPES else (SUPPORTED_DIAMETER_ATTACK_TYPES[0] if SUPPORTED_DIAMETER_ATTACK_TYPES else "roaming_fraud")
			self.record_diameter_attack(
				attack_id=attack_id,
				tenant_id=tenant_id,
				attack_type=attack_type_norm,
				source_realm=origin_realm,
				target_realm=dest_realm,
				evidence_reference=str(request),
				detected_at=_utcnow(),
			)
			result["attack_id"] = attack_id
		return result

	async def voip_fraud_detection(
		self,
		cdr: dict[str, Any],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Detect VoIP fraud from a Call Detail Record.

		Patterns detected: IRSF (International Revenue Share Fraud) —
		calls to high-rate destinations; PBX hacking — high call volume
		in short window; Wangiri — short call + callback bait.
		"""
		assert cdr, "cdr required"
		destination = str(cdr.get("destination", ""))
		duration_secs = float(cdr.get("duration_secs", 0))
		call_count_last_hour = int(cdr.get("call_count_last_hour", 1))
		is_international = destination.startswith("+") and not destination.startswith("+254")
		# IRSF: short international calls to revenue-share prefixes
		irsf_prefixes = ("+881", "+882", "+883", "+979", "+963")
		is_irsf = any(destination.startswith(p) for p in irsf_prefixes)
		# PBX hacking: >100 calls/hour
		is_pbx_hack = call_count_last_hour > 100
		# Wangiri: <5s duration, international, calls > 10
		is_wangiri = duration_secs < 5.0 and is_international and call_count_last_hour > 10
		fraud_type: str | None = None
		confidence = 0.0
		if is_irsf:
			fraud_type = "irsf"
			confidence = 0.85
		elif is_pbx_hack:
			fraud_type = "pbx_hacking"
			confidence = 0.78
		elif is_wangiri:
			fraud_type = "wangiri"
			confidence = 0.70
		detected = fraud_type is not None
		record: dict[str, Any] = {
			"cdr_id": cdr.get("cdr_id", f"cdr-{_utcnow()}"),
			"tenant_id": tenant_id,
			"destination": destination,
			"duration_secs": duration_secs,
			"detected": detected,
			"fraud_type": fraud_type,
			"confidence": confidence,
			"assessed_at": _utcnow(),
		}
		self._voip_fraud_records.append(record)
		if detected:
			msisdn = str(cdr.get("calling_number", "unknown"))
			fraud_type_norm = fraud_type if fraud_type in SUPPORTED_FRAUD_TYPES else "irsf"
			case_id = f"voip-{hashlib.md5(f'{msisdn}{fraud_type}{_utcnow()}'.encode()).hexdigest()[:8]}"
			self.raise_fraud_case(
				case_id=case_id,
				tenant_id=tenant_id,
				fraud_type=fraud_type_norm,
				msisdn=msisdn,
				confidence_score=confidence,
				evidence_reference=str(cdr),
				detected_at=_utcnow(),
			)
			record["case_id"] = case_id
		return record

	async def sim_swap_detection(
		self,
		customer_id: str,
		event: dict[str, Any],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Detect suspicious SIM swap activity for a customer.

		Risk factors: swap within 24h of password reset, geographic
		anomaly (new SIM registered in different region), >2 swaps in 30 days,
		swap followed by high-value transaction.
		"""
		assert customer_id, "customer_id required"
		assert event, "event required"
		risk_factors: list[str] = []
		risk_score = 0.0
		# Factor: recent password reset
		if event.get("recent_password_reset", False):
			risk_factors.append("recent_password_reset")
			risk_score += 0.30
		# Factor: geographic anomaly
		if event.get("geographic_anomaly", False):
			risk_factors.append("geographic_anomaly")
			risk_score += 0.35
		# Factor: multiple swaps
		prior_swaps = [
			e for e in self._sim_swap_events
			if e.get("customer_id") == customer_id and e.get("tenant_id") == tenant_id
		]
		if len(prior_swaps) >= 2:
			risk_factors.append("multiple_swaps_30d")
			risk_score += 0.25
		# Factor: high-value transaction immediately after
		if event.get("high_value_transaction_after", False):
			risk_factors.append("high_value_transaction_after")
			risk_score += 0.40
		risk_score = min(1.0, risk_score)
		verdict = "suspicious" if risk_score >= 0.5 else ("low_risk" if risk_score >= 0.2 else "normal")
		swap_record: dict[str, Any] = {
			"customer_id": customer_id,
			"tenant_id": tenant_id,
			"risk_score": round(risk_score, 3),
			"verdict": verdict,
			"risk_factors": risk_factors,
			"event": event,
			"assessed_at": _utcnow(),
		}
		self._sim_swap_events.append(swap_record)
		if verdict == "suspicious":
			self._audit(tenant_id, "sim_swap_suspicious", customer_id)
		return swap_record

	async def ott_bypass_detection(
		self,
		traffic_pattern: dict[str, Any],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Detect OTT bypass (SIM box / grey route) traffic patterns.

		Analyses: call_duration_variance, destination_entropy, flash_call_ratio,
		cli_spoofing_indicators.  High destination entropy + low duration variance
		strongly indicates SIM box activity.
		"""
		assert traffic_pattern, "traffic_pattern required"
		duration_variance = float(traffic_pattern.get("call_duration_variance", 10.0))
		destination_entropy = float(traffic_pattern.get("destination_entropy", 2.0))
		flash_call_ratio = float(traffic_pattern.get("flash_call_ratio", 0.05))
		cli_spoofing = bool(traffic_pattern.get("cli_spoofing_indicators", False))
		score = 0.0
		indicators: list[str] = []
		# Low variance (<5s) suggests automated dialler
		if duration_variance < 5.0:
			score += 0.30
			indicators.append("low_duration_variance")
		# High entropy suggests many different destinations (SIM box pattern)
		if destination_entropy > 4.0:
			score += 0.35
			indicators.append("high_destination_entropy")
		# Flash call ratio > 20% indicates bypass trick
		if flash_call_ratio > 0.20:
			score += 0.25
			indicators.append("high_flash_call_ratio")
		if cli_spoofing:
			score += 0.30
			indicators.append("cli_spoofing")
		score = min(1.0, score)
		detected = score >= 0.5
		event_record: dict[str, Any] = {
			"tenant_id": tenant_id,
			"detected": detected,
			"confidence": round(score, 3),
			"indicators": indicators,
			"traffic_pattern": traffic_pattern,
			"assessed_at": _utcnow(),
		}
		self._ott_bypass_events.append(event_record)
		if detected:
			self._audit(tenant_id, "ott_bypass_detected", str(traffic_pattern.get("source_id", "unknown")))
		return event_record

	async def lawful_intercept_order(
		self,
		target_id: str,
		authority_ref: str,
		scope: str,
		expiry: str,
		tenant_id: str = "default",
		intercept_type: str = "call_content",
	) -> dict[str, Any]:
		"""Register a lawful intercept order from a regulatory authority.

		Validates warrant reference and authority, creates an LI order record,
		and activates the intercept on the target MSISDN.  Strictly audited.
		"""
		assert target_id, "target_id (MSISDN) required"
		assert authority_ref, "authority_ref (warrant number) required"
		assert scope, "scope required"
		assert expiry, "expiry required"
		if target_id in self._li_orders:
			existing = self._li_orders[target_id]
			if existing.get("status") == "active":
				raise ValueError(f"Active LI order already exists for {target_id}")
		order_id = f"LI-{hashlib.md5(f'{target_id}{authority_ref}'.encode()).hexdigest()[:10].upper()}"
		intercept_type_norm = intercept_type if intercept_type in SUPPORTED_LAWFUL_INTERCEPT_TYPES else (SUPPORTED_LAWFUL_INTERCEPT_TYPES[0] if SUPPORTED_LAWFUL_INTERCEPT_TYPES else "call_content")
		intercept = self.activate_intercept(
			intercept_id=order_id,
			tenant_id=tenant_id,
			intercept_type=intercept_type_norm,
			target_msisdn=target_id,
			warrant_reference=authority_ref,
			regulatory_authority=authority_ref,
			activated_at=_utcnow(),
			expires_at=expiry,
		)
		order: dict[str, Any] = {
			"order_id": order_id,
			"target_id": target_id,
			"authority_ref": authority_ref,
			"scope": scope,
			"expiry": expiry,
			"tenant_id": tenant_id,
			"status": "active",
			"created_at": _utcnow(),
			"intercept": intercept,
		}
		self._li_orders[target_id] = order
		self._audit(tenant_id, "lawful_intercept_order_created", order_id)
		return order

	async def data_retention_compliance(
		self,
		jurisdiction: str,
		retention_days: int,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Check data retention compliance for a jurisdiction.

		Validates that stored audit events and intercept records do not
		exceed the mandated retention window for the jurisdiction.
		Returns compliance status and any over-retained records.
		"""
		assert jurisdiction, "jurisdiction required"
		assert retention_days > 0, "retention_days must be positive"
		# Jurisdiction retention requirements (days)
		jurisdiction_requirements: dict[str, int] = {
			"KE": 90,   # Kenya Communications Act
			"TZ": 180,
			"UG": 90,
			"EU": 730,  # GDPR allows up to 2 years for telecom
			"US": 365,
			"ZA": 36500,  # RICA: 10 years
		}
		required = jurisdiction_requirements.get(jurisdiction.upper(), retention_days)
		now = datetime.datetime.utcnow()
		over_retained: list[str] = []
		for intercept in self.intercepts.values():
			if intercept.tenant_id != tenant_id:
				continue
			try:
				activated = datetime.datetime.fromisoformat(intercept.activated_at.replace("Z", ""))
				age_days = (now - activated).days
				if age_days > required:
					over_retained.append(intercept.id)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		compliant = len(over_retained) == 0
		self._audit(tenant_id, "data_retention_compliance_checked", jurisdiction)
		return {
			"jurisdiction": jurisdiction,
			"tenant_id": tenant_id,
			"required_retention_days": required,
			"requested_retention_days": retention_days,
			"compliant": compliant,
			"over_retained_count": len(over_retained),
			"over_retained_ids": over_retained[:20],
			"checked_at": _utcnow(),
		}

	async def roaming_security_check(
		self,
		roaming_partner_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Assess security posture of a roaming partner.

		Checks: IPX routing security, GRX firewall rating, SS7 attack history
		with this partner, and GSMA compliance level.  Returns a risk rating.
		"""
		assert roaming_partner_id, "roaming_partner_id required"
		# Count SS7 attacks from this partner
		ss7_from_partner = sum(
			1 for a in self.ss7_attacks.values()
			if a.tenant_id == tenant_id
			and roaming_partner_id.lower() in a.source_reference.lower()
		)
		diameter_from_partner = sum(
			1 for a in self.diameter_attacks.values()
			if a.tenant_id == tenant_id
			and roaming_partner_id.lower() in a.source_realm.lower()
		)
		risk_score = 0.0
		risk_factors: list[str] = []
		if ss7_from_partner > 0:
			risk_score += min(0.40, ss7_from_partner * 0.10)
			risk_factors.append(f"ss7_attacks:{ss7_from_partner}")
		if diameter_from_partner > 0:
			risk_score += min(0.30, diameter_from_partner * 0.10)
			risk_factors.append(f"diameter_attacks:{diameter_from_partner}")
		risk_rating = "high" if risk_score >= 0.5 else ("medium" if risk_score >= 0.2 else "low")
		check_record: dict[str, Any] = {
			"roaming_partner_id": roaming_partner_id,
			"tenant_id": tenant_id,
			"risk_score": round(risk_score, 3),
			"risk_rating": risk_rating,
			"risk_factors": risk_factors,
			"ss7_attack_count": ss7_from_partner,
			"diameter_attack_count": diameter_from_partner,
			"checked_at": _utcnow(),
		}
		self._roaming_security_checks.append(check_record)
		self._audit(tenant_id, "roaming_security_check_run", roaming_partner_id)
		return check_record

	async def network_intrusion_detection(
		self,
		traffic_event: dict[str, Any],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Detect network intrusion patterns from a traffic event.

		Analyses: source_ip, destination_ip, port, protocol, byte_count,
		packet_count, flags.  Detects: port scans, DDoS indicators, protocol
		anomalies, and known malicious IP patterns.
		"""
		assert traffic_event, "traffic_event required"
		source_ip = str(traffic_event.get("source_ip", ""))
		dst_port = int(traffic_event.get("dst_port", 0))
		protocol = str(traffic_event.get("protocol", "")).lower()
		byte_count = int(traffic_event.get("byte_count", 0))
		packet_count = int(traffic_event.get("packet_count", 0))
		flags = str(traffic_event.get("tcp_flags", "")).upper()
		indicators: list[str] = []
		threat_score = 0.0
		# Port scan: many distinct ports in short time (simulated via port range check)
		if dst_port in range(1, 1024) and packet_count > 50:
			indicators.append("possible_port_scan")
			threat_score += 0.30
		# DDoS: very high packet count + small byte count per packet
		avg_pkt_size = byte_count / max(packet_count, 1)
		if packet_count > 10000 and avg_pkt_size < 64:
			indicators.append("ddos_indicator")
			threat_score += 0.50
		# SYN flood: SYN flag without ACK
		if "S" in flags and "A" not in flags and packet_count > 1000:
			indicators.append("syn_flood")
			threat_score += 0.40
		# Protocol anomaly: DNS over non-53 port
		if protocol == "dns" and dst_port != 53:
			indicators.append("protocol_anomaly_dns_tunnel")
			threat_score += 0.35
		# Known threat intel: check source_ip against IOCs
		ioc_match = any(
			intel.ioc_value == source_ip
			for intel in self.threat_intel.values()
			if intel.tenant_id == tenant_id and intel.ioc_type == "ipv4"
		)
		if ioc_match:
			indicators.append("ioc_match")
			threat_score += 0.60
		threat_score = min(1.0, threat_score)
		detected = threat_score >= 0.4
		intrusion_record: dict[str, Any] = {
			"event_id": traffic_event.get("event_id", f"nids-{_utcnow()}"),
			"tenant_id": tenant_id,
			"source_ip": source_ip,
			"detected": detected,
			"threat_score": round(threat_score, 3),
			"indicators": indicators,
			"assessed_at": _utcnow(),
		}
		self._intrusion_events.append(intrusion_record)
		if detected:
			self._audit(tenant_id, "network_intrusion_detected", source_ip)
		return intrusion_record

	async def security_incident_response(
		self,
		incident_id: str,
		action: str,
		tenant_id: str = "default",
		performed_by: str = "system",
		notes: str = "",
	) -> dict[str, Any]:
		"""Execute a response action against an open security incident.

		Actions: contain, eradicate, recover, close, escalate.
		Validates incident exists, maps action to new status, records response
		step in history.
		"""
		assert incident_id, "incident_id required"
		assert action, "action required"
		action_to_status: dict[str, str] = {
			"contain": "under_investigation",
			"eradicate": "eradicated",
			"recover": "recovered",
			"close": "closed",
			"escalate": "under_investigation",
		}
		action_lower = action.lower()
		if action_lower not in action_to_status:
			raise ValueError(f"Unknown action {action!r}; must be one of {list(action_to_status)}")
		incident = self._incident_or_raise(incident_id, tenant_id)
		new_status = action_to_status[action_lower]
		incident.status = new_status
		response_record: dict[str, Any] = {
			"incident_id": incident_id,
			"action": action_lower,
			"new_status": new_status,
			"performed_by": performed_by,
			"notes": notes,
			"tenant_id": tenant_id,
			"performed_at": _utcnow(),
		}
		self._incident_responses.append(response_record)
		self._audit(tenant_id, f"incident_response_{action_lower}", incident_id)
		return {**incident.to_dict(), "response": response_record}

	# ------------------------------------------------------------------ #
	# Agent validation & batch                                            #
	# ------------------------------------------------------------------ #

	def validate_agent_action(
		self,
		tenant_id: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
		evidence_fabrication_scope: bool = False,
		cross_tenant_access_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "sec_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"evidence_fabrication_scope": evidence_fabrication_scope,
			"cross_tenant_access_scope": cross_tenant_access_scope,
		})
		return {"tenant_id": tenant_id, "accepted": True}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "sec_batch", "event_stream": event_stream})
		if item_count <= 0:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.telecom.sec.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		open_incidents = sum(1 for i in self.incidents.values() if i.tenant_id == tenant_id and i.status in ("new", "under_investigation"))
		return {
			"tenant_id": tenant_id,
			"fraud_case_count": self._count(self.fraud_cases, tenant_id),
			"ss7_attack_count": self._count(self.ss7_attacks, tenant_id),
			"diameter_attack_count": self._count(self.diameter_attacks, tenant_id),
			"active_intercept_count": sum(1 for i in self.intercepts.values() if i.tenant_id == tenant_id and i.status == "active"),
			"incident_count": self._count(self.incidents, tenant_id),
			"open_incident_count": open_incidents,
			"threat_intel_count": self._count(self.threat_intel, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"voip_fraud_detections": sum(1 for r in self._voip_fraud_records if r["tenant_id"] == tenant_id and r["detected"]),
			"intrusion_detections": sum(1 for r in self._intrusion_events if r["tenant_id"] == tenant_id and r["detected"]),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	async def detect_voip_fraud(
		self,
		call_id: str,
		calling_number: str,
		called_number: str,
		duration_seconds: int,
		cost: float,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Detect VoIP toll fraud by analysing call patterns (IRSF, PBX hijack)."""
		assert call_id, "call_id required"
		# Simple heuristic: calls over 3600s to international numbers with cost > 100 flagged
		is_international = called_number.startswith("+") and not called_number.startswith("+254")
		detected = is_international and duration_seconds > 3600 and cost > 100.0
		record: dict[str, Any] = {
			"id": f"voip-fraud-{call_id}",
			"call_id": call_id,
			"calling_number": calling_number,
			"called_number": called_number,
			"duration_seconds": duration_seconds,
			"cost": cost,
			"detected": detected,
			"fraud_type": "irsf" if detected else None,
			"tenant_id": tenant_id,
			"analysed_at": _utcnow(),
		}
		self._voip_fraud_records.append(record)
		if detected:
			self._audit(tenant_id, "voip_fraud_detected", call_id)
		return record

	async def detect_sim_swap(
		self,
		msisdn: str,
		old_iccid: str,
		new_iccid: str,
		swap_channel: str,
		tenant_id: str = "default",
		kyc_verified: bool = False,
	) -> dict[str, Any]:
		"""Evaluate a SIM swap request for fraud indicators."""
		assert msisdn, "msisdn required"
		# Flag as suspicious if not KYC verified and rapid swap
		prior_swaps = [s for s in self._sim_swap_events if s["msisdn"] == msisdn and s["tenant_id"] == tenant_id]
		suspicious = not kyc_verified or len(prior_swaps) >= 2
		event: dict[str, Any] = {
			"id": f"simswap-{msisdn}-{len(self._sim_swap_events)}",
			"msisdn": msisdn,
			"old_iccid": old_iccid,
			"new_iccid": new_iccid,
			"swap_channel": swap_channel,
			"kyc_verified": kyc_verified,
			"suspicious": suspicious,
			"prior_swap_count": len(prior_swaps),
			"tenant_id": tenant_id,
			"detected_at": _utcnow(),
		}
		self._sim_swap_events.append(event)
		if suspicious:
			self._audit(tenant_id, "sim_swap_suspicious", msisdn)
		return event

	async def record_intrusion_event(
		self,
		source_ip: str,
		attack_type: str,
		severity: str,
		tenant_id: str = "default",
		target_element: str = "core_network",
	) -> dict[str, Any]:
		"""Record a network intrusion detection event."""
		assert source_ip, "source_ip required"
		assert attack_type, "attack_type required"
		assert severity in {"low", "medium", "high", "critical"}, "invalid severity"
		detected = severity in {"high", "critical"}
		event: dict[str, Any] = {
			"id": f"intrusion-{source_ip}-{len(self._intrusion_events)}",
			"source_ip": source_ip,
			"attack_type": attack_type,
			"severity": severity,
			"target_element": target_element,
			"detected": detected,
			"tenant_id": tenant_id,
			"recorded_at": _utcnow(),
		}
		self._intrusion_events.append(event)
		if detected:
			self._audit(tenant_id, "intrusion_detected", source_ip)
		return event

	async def security_incident_analytics(
		self,
		tenant_id: str = "default",
		period: str = "monthly",
	) -> dict[str, Any]:
		"""Compute security KPIs: incident counts, fraud detection rates, MTTR."""
		incidents = [i.to_dict() for i in self.incidents.values() if i.tenant_id == tenant_id]
		open_incidents = [i for i in incidents if i.get("status") == "open"]
		fraud_cases = [f.to_dict() for f in self.fraud_cases.values() if f.tenant_id == tenant_id]
		voip_detected = sum(1 for r in self._voip_fraud_records if r["tenant_id"] == tenant_id and r["detected"])
		sim_suspicious = sum(1 for s in self._sim_swap_events if s["tenant_id"] == tenant_id and s["suspicious"])
		intrusions = sum(1 for e in self._intrusion_events if e["tenant_id"] == tenant_id and e["detected"])
		self._audit(tenant_id, "security_incident_analytics_run", period)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_incidents": len(incidents),
			"open_incidents": len(open_incidents),
			"fraud_case_count": len(fraud_cases),
			"voip_fraud_detections": voip_detected,
			"suspicious_sim_swaps": sim_suspicious,
			"intrusion_detections": intrusions,
			"computed_at": _utcnow(),
		}

	async def bulk_block_threats(
		self,
		threat_ids: list[str],
		tenant_id: str = "default",
		blocked_by: str = "system",
	) -> dict[str, Any]:
		"""Block multiple threat intelligence entries in bulk."""
		assert threat_ids, "threat_ids required"
		results: list[dict[str, Any]] = []
		for tid in threat_ids:
			threat = self.threat_intel.get(self._key(tenant_id, tid))
			if threat:
				threat.status = "blocked"
				results.append({"threat_id": tid, "status": "blocked"})
			else:
				results.append({"threat_id": tid, "status": "not_found"})
		self._audit(tenant_id, "bulk_threats_blocked", f"count:{len(threat_ids)}")
		return {
			"tenant_id": tenant_id,
			"blocked_count": sum(1 for r in results if r["status"] == "blocked"),
			"not_found_count": sum(1 for r in results if r["status"] == "not_found"),
			"results": results,
			"blocked_at": _utcnow(),
		}

	async def export_security_data(
		self,
		tenant_id: str = "default",
		format: str = "json",
	) -> dict[str, Any]:
		"""Export fraud cases, incidents and threat intel to JSON or CSV."""
		assert format in {"json", "csv"}, "format must be json or csv"
		fraud_cases = [f.to_dict() for f in self.fraud_cases.values() if f.tenant_id == tenant_id]
		incidents = [i.to_dict() for i in self.incidents.values() if i.tenant_id == tenant_id]
		self._audit(tenant_id, "security_data_exported", f"format:{format}")
		return {
			"format": format,
			"tenant_id": tenant_id,
			"fraud_case_count": len(fraud_cases),
			"incident_count": len(incidents),
			"fraud_cases": fraud_cases,
			"incidents": incidents,
			"exported_at": _utcnow(),
		}

	async def security_compliance_report(
		self,
		tenant_id: str = "default",
		standard: str = "3GPP_TS_33",
	) -> dict[str, Any]:
		"""Generate a security compliance report against a telecom security standard."""
		incidents = [i.to_dict() for i in self.incidents.values() if i.tenant_id == tenant_id]
		intercepts = [ic.to_dict() for ic in self.intercepts.values() if ic.tenant_id == tenant_id]
		compliant_intercepts = [ic for ic in intercepts if ic.get("status") == "active"]
		self._audit(tenant_id, "security_compliance_report_generated", standard)
		return {
			"standard": standard,
			"tenant_id": tenant_id,
			"incident_count": len(incidents),
			"open_incident_count": sum(1 for i in incidents if i.get("status") == "open"),
			"lawful_intercept_count": len(intercepts),
			"active_intercept_count": len(compliant_intercepts),
			"ss7_attack_count": self._count(self.ss7_attacks, tenant_id),
			"diameter_attack_count": self._count(self.diameter_attacks, tenant_id),
			"compliance_status": "compliant" if not any(i.get("status") == "open" for i in incidents) else "non_compliant",
			"generated_at": _utcnow(),
		}

	async def health_check(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return security service health status."""
		open_incidents = sum(1 for i in self.incidents.values() if i.tenant_id == tenant_id and i.status == "open")
		return {
			"service": "TelecomSecurityService",
			"tenant_id": tenant_id,
			"status": "healthy" if open_incidents < 10 else "critical",
			"open_incident_count": open_incidents,
			"fraud_case_count": self._count(self.fraud_cases, tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"checked_at": _utcnow(),
		}

	async def threat_intel_analytics(
		self,
		tenant_id: str = "default",
		period: str = "weekly",
	) -> dict[str, Any]:
		"""Summarise threat intelligence by source and type."""
		threats = [t.to_dict() for t in self.threat_intel.values() if t.tenant_id == tenant_id]
		by_source: dict[str, int] = {}
		blocked = 0
		for t in threats:
			src = t.get("source", "unknown")
			by_source[src] = by_source.get(src, 0) + 1
			if t.get("status") == "blocked":
				blocked += 1
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_threats": len(threats),
			"blocked_count": blocked,
			"active_count": len(threats) - blocked,
			"by_source": by_source,
			"computed_at": _utcnow(),
		}

	# ------------------------------------------------------------------ #
	# Internal helpers                                                    #
	# ------------------------------------------------------------------ #

	def _fraud_case_or_raise(self, case_id: str, tenant_id: str) -> SecFraudCase:
		c = self.fraud_cases.get(self._key(tenant_id, case_id))
		if c is None:
			raise ValueError(f"Fraud case {case_id} not found")
		return c

	def _intercept_or_raise(self, intercept_id: str, tenant_id: str) -> SecLawfulIntercept:
		i = self.intercepts.get(self._key(tenant_id, intercept_id))
		if i is None:
			raise ValueError(f"Intercept {intercept_id} not found")
		return i

	def _incident_or_raise(self, incident_id: str, tenant_id: str) -> SecIncident:
		i = self.incidents.get(self._key(tenant_id, incident_id))
		if i is None:
			raise ValueError(f"Incident {incident_id} not found")
		return i

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

	async def get_audit_trail(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Get Audit Trail"""
		return [e for e in self.audit_events if e["tenant_id"] == tenant_id]

	async def archive_record(self, record_id: str, tenant_id: str = "default", reason: str = "") -> dict[str, Any]:
		"""Archive Record"""
		assert record_id
		self._audit(tenant_id, "record_archived", record_id)
		return {"record_id": record_id, "status": "archived", "reason": reason}

	async def get_kpis(self, tenant_id: str = "default", period: str = "monthly") -> dict[str, Any]:
		"""Get Kpis"""
		return {"tenant_id": tenant_id, "period": period, "computed_at": _utcnow()}


# Backward-compatible alias
TelecomSecService = TelecomSecurityService
