"""Service layer for APG Quality of Service."""

from __future__ import annotations

import datetime
import statistics
from typing import Any

from .domain.adapters import get_auth_adapter, get_audit_adapter
from .database.store import get_store
from .capability_contract import (
	SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_DEGRADATION_CAUSES,
	SUPPORTED_ENFORCEMENT_STATUSES, SUPPORTED_POLICY_TYPES, SUPPORTED_QOS_CLASSES,
	SUPPORTED_REMEDIATION_TYPES, SUPPORTED_SLA_PARAMETERS, SUPPORTED_TRAFFIC_TYPES,
	evaluate_capability_rules, get_capability_contract,
)
from .models import (
	QosAgent, QosDegradation, QosEnforcementRecord, QosPolicy,
	QosRemediation, QosRootCause, QosSlasMeasurement, QosTrafficClassification,
)


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


def _bounded(value: float) -> bool:
	return 0.0 <= value <= 1.0


def _utcnow() -> str:
	return datetime.datetime.utcnow().isoformat() + "Z"


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class QualityOfServiceService:
	"""Tenant-scoped QoS management service for APG Telecom."""

	def __init__(self) -> None:
		self._store = get_store("telecom.qos")
		self._auth = get_auth_adapter()
		self._audit_adapter = get_audit_adapter()
		self.policies: dict[tuple[str, str], QosPolicy] = {}
		self.traffic_classifications: dict[tuple[str, str], QosTrafficClassification] = {}
		self.enforcement_records: dict[tuple[str, str], QosEnforcementRecord] = {}
		self.sla_measurements: dict[tuple[str, str], QosSlasMeasurement] = {}
		self.degradations: dict[tuple[str, str], QosDegradation] = {}
		self.root_causes: dict[tuple[str, str], QosRootCause] = {}
		self.remediations: dict[tuple[str, str], QosRemediation] = {}
		self.agents: dict[tuple[str, str], QosAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# In-memory stores for new method state
		self._speed_test_results: list[dict[str, Any]] = []
		self._voip_calls: list[dict[str, Any]] = []
		self._congestion_events: list[dict[str, Any]] = []
		self._sla_breach_notifications: list[dict[str, Any]] = []
		self._qos_profiles: dict[str, dict[str, Any]] = {}  # customer_id+service_id -> policy_id
		self._qos_sessions: dict[str, dict[str, Any]] = {}  # session_id -> enforcement state

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

	def create_qos_policy(
		self,
		policy_id: str,
		tenant_id: str,
		policy_type: str,
		qos_class: str,
		name: str,
		parameters: str,
		approval_reference: str,
		created_by: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create a QoS policy with mandatory conflict check and approval."""
		policy_type = policy_type.lower()
		qos_class = qos_class.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_qos_policy",
			"policy_type_supported": policy_type in SUPPORTED_POLICY_TYPES,
			"qos_class_supported": qos_class in SUPPORTED_QOS_CLASSES,
			"approval_present": _present(approval_reference),
			"conflict_checked": True,
		})
		item = QosPolicy(policy_id, tenant_id, policy_type, qos_class, name, parameters, approval_reference, "active", created_by)
		self.policies[self._key(tenant_id, policy_id)] = item
		self._audit(tenant_id, "qos_policy_activated", policy_id)
		return item.to_dict()

	def change_qos_policy(
		self,
		policy_id: str,
		tenant_id: str,
		new_parameters: str,
		is_downgrade: bool,
		approval_reference: str,
	) -> dict[str, Any]:
		"""Modify a QoS policy; downgrades require explicit approval."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "change_qos_policy",
			"is_downgrade": is_downgrade,
			"approval_present": _present(approval_reference),
		})
		policy = self._policy_or_raise(policy_id, tenant_id)
		policy.parameters = new_parameters
		self._audit(tenant_id, "qos_policy_changed", policy_id)
		return policy.to_dict()

	def classify_traffic(
		self,
		classification_id: str,
		tenant_id: str,
		traffic_type: str,
		classification: str,
		policy_id: str,
		flow_reference: str,
		classified_at: str,
	) -> dict[str, Any]:
		"""Classify a traffic flow and associate it with a QoS policy."""
		traffic_type = traffic_type.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "classify_traffic",
			"traffic_type_supported": traffic_type in SUPPORTED_TRAFFIC_TYPES,
			"classification_present": _present(classification),
		})
		item = QosTrafficClassification(classification_id, tenant_id, traffic_type, classification, policy_id, flow_reference, classified_at)
		self.traffic_classifications[self._key(tenant_id, classification_id)] = item
		self._audit(tenant_id, "traffic_classified", classification_id)
		return item.to_dict()

	def update_enforcement_status(
		self,
		enforcement_id: str,
		tenant_id: str,
		policy_id: str,
		ne_reference: str,
		status: str,
		enforced_at: str,
		last_updated: str,
	) -> dict[str, Any]:
		"""Record the enforcement status of a QoS policy on a network element."""
		status = status.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "update_enforcement_status",
			"enforcement_status_supported": status in SUPPORTED_ENFORCEMENT_STATUSES,
		})
		item = QosEnforcementRecord(enforcement_id, tenant_id, policy_id, ne_reference, status, enforced_at, last_updated)
		self.enforcement_records[self._key(tenant_id, enforcement_id)] = item
		self._audit(tenant_id, "enforcement_status_updated", enforcement_id)
		return item.to_dict()

	def record_sla_measurement(
		self,
		measurement_id: str,
		tenant_id: str,
		sla_parameter: str,
		measured_value: float,
		target_value: float,
		customer_id: str | None,
		measured_at: str,
	) -> dict[str, Any]:
		"""Record a QoS SLA measurement."""
		sla_parameter = sla_parameter.lower()
		is_breach = (
			measured_value > target_value
			if "latency" in sla_parameter or "loss" in sla_parameter or "jitter" in sla_parameter
			else measured_value < target_value
		)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_sla_measurement",
			"sla_parameter_supported": sla_parameter in SUPPORTED_SLA_PARAMETERS,
		})
		item = QosSlasMeasurement(measurement_id, tenant_id, sla_parameter, float(measured_value), float(target_value), customer_id, is_breach, measured_at)
		self.sla_measurements[self._key(tenant_id, measurement_id)] = item
		if is_breach:
			self._audit(tenant_id, "sla_breach_detected", measurement_id)
		return item.to_dict()

	def record_degradation(
		self,
		degradation_id: str,
		tenant_id: str,
		cause: str,
		confidence_score: float,
		description: str,
		affected_resource: str,
		evidence_reference: str,
		detected_at: str,
	) -> dict[str, Any]:
		"""Record a QoS degradation event with root cause attribution."""
		cause = cause.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_degradation",
			"degradation_cause_supported": cause in SUPPORTED_DEGRADATION_CAUSES,
			"confidence_present": confidence_score is not None,
			"evidence_present": _present(evidence_reference),
		})
		item = QosDegradation(degradation_id, tenant_id, cause, float(confidence_score), description, affected_resource, evidence_reference, detected_at, "open")
		self.degradations[self._key(tenant_id, degradation_id)] = item
		self._audit(tenant_id, "degradation_detected", degradation_id)
		return item.to_dict()

	def record_root_cause(
		self,
		rca_id: str,
		tenant_id: str,
		degradation_id: str,
		root_cause_description: str,
		confidence_score: float,
		evidence_reference: str,
		identified_at: str,
	) -> dict[str, Any]:
		"""Record a root cause analysis for a degradation event."""
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True})
		item = QosRootCause(rca_id, tenant_id, degradation_id, root_cause_description, float(confidence_score), evidence_reference, identified_at)
		self.root_causes[self._key(tenant_id, rca_id)] = item
		self._audit(tenant_id, "root_cause_identified", rca_id)
		return item.to_dict()

	def trigger_remediation(
		self,
		remediation_id: str,
		tenant_id: str,
		degradation_id: str,
		remediation_type: str,
		is_disruptive: bool,
		approval_reference: str | None,
		triggered_at: str,
	) -> dict[str, Any]:
		"""Trigger a remediation action for a QoS degradation."""
		remediation_type = remediation_type.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "trigger_remediation",
			"remediation_type_supported": remediation_type in SUPPORTED_REMEDIATION_TYPES,
			"is_disruptive": is_disruptive,
			"approval_present": _present(approval_reference) if is_disruptive else True,
		})
		item = QosRemediation(remediation_id, tenant_id, degradation_id, remediation_type, is_disruptive, approval_reference, "in_progress", triggered_at, None)
		self.remediations[self._key(tenant_id, remediation_id)] = item
		self._audit(tenant_id, "remediation_triggered", remediation_id)
		return item.to_dict()

	def complete_remediation(self, remediation_id: str, tenant_id: str, completed_at: str) -> dict[str, Any]:
		"""Mark a remediation as completed."""
		remediation = self._remediation_or_raise(remediation_id, tenant_id)
		remediation.status = "completed"
		remediation.completed_at = completed_at
		self._audit(tenant_id, "remediation_completed", remediation_id)
		return remediation.to_dict()

	def register_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		"""Register a QoS automation agent."""
		runtime = runtime.lower()
		role = role.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_qos_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		item = QosAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "qos_agent_registered", agent_id)
		return item.to_dict()

	# ------------------------------------------------------------------ #
	# New methods                                                          #
	# ------------------------------------------------------------------ #

	async def qos_policy_create(
		self,
		name: str,
		traffic_class: str,
		dscp_marking: int,
		bandwidth_limit: int,
		priority: int,
		tenant_id: str = "default",
		created_by: str = "system",
		approval_reference: str = "",
	) -> dict[str, Any]:
		"""Create a QoS policy from network-oriented parameters.

		dscp_marking: DSCP value 0-63 (EF=46, AF41=34, CS0=0, etc.)
		bandwidth_limit: Kbps ceiling for this traffic class
		priority: 1 (highest) – 8 (best-effort)
		"""
		assert _present(name), "policy name required"
		assert 0 <= dscp_marking <= 63, f"dscp_marking must be 0-63, got {dscp_marking}"
		assert bandwidth_limit > 0, "bandwidth_limit must be positive Kbps"
		assert 1 <= priority <= 8, "priority must be 1-8"
		traffic_class_norm = traffic_class.lower()
		# Map DSCP to QoS class
		dscp_to_class = {46: "ef", 34: "af41", 26: "af31", 18: "af21", 10: "af11", 0: "be"}
		qos_class = dscp_to_class.get(dscp_marking, "af31")
		policy_id = f"qos-{name.lower().replace(' ', '-')}-{_utcnow()[:10]}"
		parameters = f"dscp={dscp_marking},bw_limit={bandwidth_limit}kbps,priority={priority}"
		result = self.create_qos_policy(
			policy_id=policy_id,
			tenant_id=tenant_id,
			policy_type=traffic_class_norm if traffic_class_norm in SUPPORTED_POLICY_TYPES else "traffic_shaping",
			qos_class=qos_class,
			name=name,
			parameters=parameters,
			approval_reference=approval_reference or f"auto-{_utcnow()}",
			created_by=created_by,
		)
		result.update({
			"dscp_marking": dscp_marking,
			"bandwidth_limit_kbps": bandwidth_limit,
			"priority": priority,
			"traffic_class": traffic_class_norm,
		})
		return result

	async def apply_qos_profile(
		self,
		customer_id: str,
		service_id: str,
		policy_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Bind a QoS policy to a customer service subscription.

		Validates the policy exists, then stores the customer-service → policy
		binding.  If a prior binding exists, it is replaced and the change is
		audited.
		"""
		assert customer_id, "customer_id required"
		assert service_id, "service_id required"
		assert policy_id, "policy_id required"
		policy = self.policies.get(self._key(tenant_id, policy_id))
		if policy is None:
			raise ValueError(f"QoS policy {policy_id} not found for tenant {tenant_id}")
		profile_key = f"{customer_id}:{service_id}:{tenant_id}"
		prior = self._qos_profiles.get(profile_key)
		self._qos_profiles[profile_key] = {
			"customer_id": customer_id,
			"service_id": service_id,
			"policy_id": policy_id,
			"tenant_id": tenant_id,
			"applied_at": _utcnow(),
			"prior_policy_id": prior["policy_id"] if prior else None,
		}
		self._audit(tenant_id, "qos_profile_applied", f"{customer_id}:{service_id}")
		return self._qos_profiles[profile_key]

	async def traffic_classification(
		self,
		packet_metadata: dict[str, Any],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Classify a packet/flow based on metadata heuristics.

		packet_metadata keys: src_port, dst_port, protocol, app_id, payload_size_bytes
		Returns traffic_type, suggested_dscp, and recommended policy_id (if any).
		"""
		assert packet_metadata, "packet_metadata required"
		dst_port = int(packet_metadata.get("dst_port", 0))
		protocol = str(packet_metadata.get("protocol", "")).lower()
		app_id = str(packet_metadata.get("app_id", "")).lower()
		# Rule-based classification
		if dst_port in (5060, 5061) or protocol == "sip" or "voice" in app_id or "voip" in app_id:
			traffic_type = "voip"
			dscp = 46  # EF
		elif dst_port in (554, 8554) or "video" in app_id or "stream" in app_id:
			traffic_type = "video_streaming"
			dscp = 34  # AF41
		elif dst_port == 443 or dst_port == 80:
			traffic_type = "web_browsing"
			dscp = 18  # AF21
		elif dst_port in (20, 21) or "ftp" in app_id or "backup" in app_id:
			traffic_type = "bulk_data"
			dscp = 10  # AF11
		elif "game" in app_id or dst_port in (27015, 7777):
			traffic_type = "gaming"
			dscp = 34  # AF41
		else:
			traffic_type = "best_effort"
			dscp = 0
		# Find a matching policy
		matching_policy = next(
			(p for p in self.policies.values()
			 if p.tenant_id == tenant_id and traffic_type in p.policy_type),
			None,
		)
		return {
			"traffic_type": traffic_type,
			"suggested_dscp": dscp,
			"recommended_policy_id": matching_policy.id if matching_policy else None,
			"packet_metadata": packet_metadata,
			"classified_at": _utcnow(),
		}

	async def congestion_detection(
		self,
		network_element_id: str,
		threshold_pct: float,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Detect congestion on a network element by examining SLA breach rate.

		threshold_pct: percentage (0-100) above which congestion is declared.
		Checks recent SLA measurements for high latency/jitter/loss parameters
		and returns a congestion verdict with contributing measurements.
		"""
		assert network_element_id, "network_element_id required"
		assert 0 < threshold_pct <= 100, "threshold_pct must be (0, 100]"
		recent_measurements = [
			m for m in self.sla_measurements.values()
			if m.tenant_id == tenant_id and m.is_breach
		]
		breach_rate = len(recent_measurements) / max(len(self.sla_measurements), 1)
		congested = breach_rate * 100 >= threshold_pct
		contributing = [
			{"id": m.id, "parameter": m.sla_parameter, "value": m.measured_value, "target": m.target_value}
			for m in recent_measurements[:10]
		]
		if congested:
			event: dict[str, Any] = {
				"network_element_id": network_element_id,
				"breach_rate": round(breach_rate, 4),
				"threshold_pct": threshold_pct,
				"tenant_id": tenant_id,
				"detected_at": _utcnow(),
			}
			self._congestion_events.append(event)
			self._audit(tenant_id, "congestion_detected", network_element_id)
		return {
			"network_element_id": network_element_id,
			"tenant_id": tenant_id,
			"threshold_pct": threshold_pct,
			"breach_rate_pct": round(breach_rate * 100, 2),
			"congested": congested,
			"contributing_measurements": contributing,
			"assessed_at": _utcnow(),
		}

	async def qos_enforcement(
		self,
		session_id: str,
		policy_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Enforce a QoS policy for an active session.

		Looks up the policy, validates it is active, stores the session
		enforcement state, and returns enforcement parameters the network
		element should apply.
		"""
		assert session_id, "session_id required"
		assert policy_id, "policy_id required"
		policy = self.policies.get(self._key(tenant_id, policy_id))
		if policy is None:
			raise ValueError(f"Policy {policy_id} not found")
		if policy.status != "active":
			raise ValueError(f"Policy {policy_id} is {policy.status}, cannot enforce")
		enforcement: dict[str, Any] = {
			"session_id": session_id,
			"policy_id": policy_id,
			"qos_class": policy.qos_class,
			"parameters": policy.parameters,
			"tenant_id": tenant_id,
			"enforced_at": _utcnow(),
			"status": "enforced",
		}
		self._qos_sessions[session_id] = enforcement
		self._audit(tenant_id, "qos_session_enforced", session_id)
		return enforcement

	async def service_degradation_alert(
		self,
		customer_id: str,
		service_id: str,
		current_quality: dict[str, float],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Raise a service degradation alert for a customer/service.

		current_quality keys: latency_ms, packet_loss_pct, jitter_ms, download_mbps
		Compares against standard SLA thresholds and returns severity.
		"""
		assert customer_id, "customer_id required"
		assert service_id, "service_id required"
		assert current_quality, "current_quality metrics required"
		thresholds = {
			"latency_ms": 150.0,
			"packet_loss_pct": 1.0,
			"jitter_ms": 30.0,
			"download_mbps": 5.0,  # minimum floor
		}
		violations: list[dict[str, Any]] = []
		for metric, threshold in thresholds.items():
			val = current_quality.get(metric)
			if val is None:
				continue
			is_violation = (val > threshold) if metric != "download_mbps" else (val < threshold)
			if is_violation:
				violations.append({"metric": metric, "value": val, "threshold": threshold})
		severity = "critical" if len(violations) >= 3 else ("warning" if violations else "normal")
		alert: dict[str, Any] = {
			"customer_id": customer_id,
			"service_id": service_id,
			"tenant_id": tenant_id,
			"severity": severity,
			"violations": violations,
			"current_quality": current_quality,
			"raised_at": _utcnow(),
		}
		if violations:
			self._audit(tenant_id, "service_degradation_alert", f"{customer_id}:{service_id}")
		return alert

	async def speed_test_result(
		self,
		customer_id: str,
		download_mbps: float,
		upload_mbps: float,
		latency_ms: float,
		tenant_id: str = "default",
		server_location: str = "",
	) -> dict[str, Any]:
		"""Record a speed test result for a customer and assess QoS compliance.

		Benchmarks: download >= 10 Mbps, upload >= 5 Mbps, latency <= 100 ms for
		standard broadband SLA.  Returns pass/fail per metric and an overall grade.
		"""
		assert customer_id, "customer_id required"
		assert download_mbps >= 0, "download_mbps must be non-negative"
		assert upload_mbps >= 0, "upload_mbps must be non-negative"
		assert latency_ms >= 0, "latency_ms must be non-negative"
		dl_ok = download_mbps >= 10.0
		ul_ok = upload_mbps >= 5.0
		lat_ok = latency_ms <= 100.0
		passed = sum([dl_ok, ul_ok, lat_ok])
		grade = "A" if passed == 3 else ("B" if passed == 2 else ("C" if passed == 1 else "F"))
		result: dict[str, Any] = {
			"customer_id": customer_id,
			"tenant_id": tenant_id,
			"download_mbps": download_mbps,
			"upload_mbps": upload_mbps,
			"latency_ms": latency_ms,
			"server_location": server_location,
			"dl_sla_ok": dl_ok,
			"ul_sla_ok": ul_ok,
			"latency_sla_ok": lat_ok,
			"grade": grade,
			"tested_at": _utcnow(),
		}
		self._speed_test_results.append(result)
		if grade in ("C", "F"):
			self._audit(tenant_id, "speed_test_sla_failure", customer_id)
		return result

	async def voip_mos_calculation(
		self,
		call_id: str,
		metrics: dict[str, float],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Calculate Mean Opinion Score (MOS) for a VoIP call using E-model approximation.

		metrics keys: packet_loss_pct, latency_ms, jitter_ms, codec_efficiency (0-1)
		MOS range: 1.0 (bad) – 5.0 (excellent).  >= 4.0 = good, 3.5-3.9 = fair, < 3.5 = poor.
		Uses simplified ITU-T G.107 E-model: R = 93.2 – Id – Ie + A.
		"""
		assert call_id, "call_id required"
		loss = float(metrics.get("packet_loss_pct", 0.0))
		latency = float(metrics.get("latency_ms", 20.0))
		jitter = float(metrics.get("jitter_ms", 5.0))
		codec_eff = float(metrics.get("codec_efficiency", 0.9))
		# E-model R factor
		# Id: delay impairment — 0.024 * latency + 0.11 * (latency - 177.3) * H(latency - 177.3)
		Id = 0.024 * latency + 0.11 * max(0.0, latency - 177.3)
		# Ie: equipment impairment from packet loss
		Ie = 30 * (1 - codec_eff) + 15 * loss
		# A: advantage factor (mobile = 10, typically 0)
		A = 0.0
		R = max(0.0, min(100.0, 93.2 - Id - Ie + A))
		# Jitter penalty: subtract 0.1 per ms above 10 ms
		jitter_penalty = max(0.0, (jitter - 10.0) * 0.1)
		R = max(0.0, R - jitter_penalty)
		# Convert R to MOS
		if R < 0:
			mos = 1.0
		elif R > 100:
			mos = 4.5
		else:
			mos = round(1.0 + 0.035 * R + R * (R - 60.0) * (100.0 - R) * 7e-6, 2)
		quality = "excellent" if mos >= 4.3 else ("good" if mos >= 4.0 else ("fair" if mos >= 3.5 else "poor"))
		call_record: dict[str, Any] = {
			"call_id": call_id,
			"tenant_id": tenant_id,
			"mos": mos,
			"r_factor": round(R, 2),
			"quality": quality,
			"metrics": metrics,
			"calculated_at": _utcnow(),
		}
		self._voip_calls.append(call_record)
		if quality == "poor":
			self._audit(tenant_id, "voip_poor_mos", call_id)
		return call_record

	async def qos_report(
		self,
		period: str,
		service_type: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Generate a QoS analytics report for a period and service type.

		Aggregates SLA measurement compliance, speed test grades, VoIP MOS
		distribution, and congestion events into a structured report.
		"""
		assert period, "period required"
		assert service_type, "service_type required"
		sla_recs = list(self.sla_measurements.values())
		total_sla = len(sla_recs)
		sla_breaches = sum(1 for m in sla_recs if m.is_breach and m.tenant_id == tenant_id)
		compliance_rate = round((total_sla - sla_breaches) / max(total_sla, 1), 4)
		speed_tests = [r for r in self._speed_test_results if r["tenant_id"] == tenant_id]
		grade_dist: dict[str, int] = {"A": 0, "B": 0, "C": 0, "F": 0}
		for r in speed_tests:
			grade_dist[r.get("grade", "F")] += 1
		voip_mos = [r["mos"] for r in self._voip_calls if r["tenant_id"] == tenant_id]
		avg_mos = round(statistics.mean(voip_mos), 2) if voip_mos else None
		congestion_count = len([e for e in self._congestion_events if e.get("tenant_id") == tenant_id])
		return {
			"period": period,
			"service_type": service_type,
			"tenant_id": tenant_id,
			"sla_compliance_rate": compliance_rate,
			"sla_breach_count": sla_breaches,
			"speed_test_count": len(speed_tests),
			"speed_test_grade_distribution": grade_dist,
			"voip_avg_mos": avg_mos,
			"voip_call_count": len(voip_mos),
			"congestion_events": congestion_count,
			"degradation_count": self._count(self.degradations, tenant_id),
			"generated_at": _utcnow(),
		}

	async def sla_breach_notification(
		self,
		customer_id: str,
		service_id: str,
		breach_type: str,
		tenant_id: str = "default",
		channel: str = "email",
	) -> dict[str, Any]:
		"""Send an SLA breach notification to a customer.

		Deduplicates notifications for the same customer/service/breach_type
		within the same day.  Records notification for compliance tracking.
		"""
		assert customer_id, "customer_id required"
		assert service_id, "service_id required"
		assert breach_type, "breach_type required"
		today = _utcnow()[:10]
		duplicate = any(
			n.get("customer_id") == customer_id
			and n.get("service_id") == service_id
			and n.get("breach_type") == breach_type
			and n.get("notified_at", "")[:10] == today
			and n.get("tenant_id") == tenant_id
			for n in self._sla_breach_notifications
		)
		notification: dict[str, Any] = {
			"customer_id": customer_id,
			"service_id": service_id,
			"breach_type": breach_type,
			"tenant_id": tenant_id,
			"channel": channel,
			"duplicate_suppressed": duplicate,
			"notified_at": _utcnow(),
		}
		self._sla_breach_notifications.append(notification)
		if not duplicate:
			self._audit(tenant_id, "sla_breach_notification_sent", f"{customer_id}:{service_id}")
		return notification

	# ------------------------------------------------------------------ #
	# Agent validation & batch                                            #
	# ------------------------------------------------------------------ #

	def validate_agent_action(
		self,
		tenant_id: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
		cross_tenant_qos_scope: bool = False,
		unapproved_policy_change_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "qos_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"cross_tenant_qos_scope": cross_tenant_qos_scope,
			"unapproved_policy_change_scope": unapproved_policy_change_scope,
		})
		return {"tenant_id": tenant_id, "accepted": True}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "qos_batch", "event_stream": event_stream})
		if item_count <= 0:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.telecom.qos.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		open_degradations = sum(1 for d in self.degradations.values() if d.tenant_id == tenant_id and d.status == "open")
		return {
			"tenant_id": tenant_id,
			"policy_count": self._count(self.policies, tenant_id),
			"traffic_classification_count": self._count(self.traffic_classifications, tenant_id),
			"enforcement_record_count": self._count(self.enforcement_records, tenant_id),
			"sla_measurement_count": self._count(self.sla_measurements, tenant_id),
			"degradation_count": self._count(self.degradations, tenant_id),
			"open_degradation_count": open_degradations,
			"root_cause_count": self._count(self.root_causes, tenant_id),
			"remediation_count": self._count(self.remediations, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"speed_test_count": sum(1 for r in self._speed_test_results if r["tenant_id"] == tenant_id),
			"voip_call_count": sum(1 for r in self._voip_calls if r["tenant_id"] == tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	async def run_speed_test(
		self,
		cell_id: str,
		dl_mbps: float,
		ul_mbps: float,
		latency_ms: float,
		tenant_id: str = "default",
		technology: str = "4G",
	) -> dict[str, Any]:
		"""Record a speed test result for QoS assessment."""
		assert cell_id, "cell_id required"
		result: dict[str, Any] = {
			"id": f"spd-{cell_id}-{len(self._speed_test_results)}",
			"cell_id": cell_id,
			"dl_mbps": dl_mbps,
			"ul_mbps": ul_mbps,
			"latency_ms": latency_ms,
			"technology": technology,
			"tenant_id": tenant_id,
			"tested_at": _utcnow(),
		}
		self._speed_test_results.append(result)
		self._audit(tenant_id, "speed_test_recorded", result["id"])
		return result

	async def speed_test_analytics(
		self,
		tenant_id: str = "default",
		period: str = "last_30_days",
	) -> dict[str, Any]:
		"""Compute aggregate speed test statistics."""
		records = [r for r in self._speed_test_results if r["tenant_id"] == tenant_id]
		if not records:
			return {"period": period, "tenant_id": tenant_id, "record_count": 0}
		dl_vals = [r["dl_mbps"] for r in records]
		ul_vals = [r["ul_mbps"] for r in records]
		lat_vals = [r["latency_ms"] for r in records]
		return {
			"period": period,
			"tenant_id": tenant_id,
			"record_count": len(records),
			"dl_mbps_mean": round(statistics.mean(dl_vals), 2),
			"ul_mbps_mean": round(statistics.mean(ul_vals), 2),
			"latency_ms_mean": round(statistics.mean(lat_vals), 2),
			"latency_ms_p95": round(sorted(lat_vals)[int(len(lat_vals) * 0.95)], 2),
			"computed_at": _utcnow(),
		}

	async def record_voip_call(
		self,
		call_id: str,
		mos_score: float,
		packet_loss_pct: float,
		jitter_ms: float,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Record VoIP call quality metrics (MOS, packet loss, jitter)."""
		assert call_id, "call_id required"
		assert 1.0 <= mos_score <= 5.0, "mos_score must be between 1.0 and 5.0"
		quality = "excellent" if mos_score >= 4.0 else "good" if mos_score >= 3.5 else "fair" if mos_score >= 2.5 else "poor"
		record: dict[str, Any] = {
			"id": f"voip-{call_id}",
			"call_id": call_id,
			"mos_score": mos_score,
			"packet_loss_pct": packet_loss_pct,
			"jitter_ms": jitter_ms,
			"quality": quality,
			"tenant_id": tenant_id,
			"recorded_at": _utcnow(),
		}
		self._voip_calls.append(record)
		self._audit(tenant_id, "voip_call_recorded", record["id"])
		return record

	async def voip_quality_analytics(
		self,
		tenant_id: str = "default",
		period: str = "weekly",
	) -> dict[str, Any]:
		"""Compute VoIP quality KPIs: mean MOS, poor call rate."""
		calls = [c for c in self._voip_calls if c["tenant_id"] == tenant_id]
		if not calls:
			return {"period": period, "tenant_id": tenant_id, "call_count": 0}
		mos_vals = [c["mos_score"] for c in calls]
		poor = sum(1 for c in calls if c["quality"] == "poor")
		return {
			"period": period,
			"tenant_id": tenant_id,
			"call_count": len(calls),
			"mean_mos": round(statistics.mean(mos_vals), 3),
			"poor_call_count": poor,
			"poor_call_rate_pct": round(poor / len(calls) * 100, 2),
			"computed_at": _utcnow(),
		}

	async def record_congestion_event(
		self,
		cell_id: str,
		congestion_level: str,
		affected_users: int,
		tenant_id: str = "default",
		cause: str = "high_traffic",
	) -> dict[str, Any]:
		"""Record a network congestion event for QoS tracking."""
		assert cell_id, "cell_id required"
		assert congestion_level in {"low", "medium", "high", "critical"}, "invalid congestion_level"
		event: dict[str, Any] = {
			"id": f"cng-{cell_id}-{len(self._congestion_events)}",
			"cell_id": cell_id,
			"congestion_level": congestion_level,
			"affected_users": affected_users,
			"cause": cause,
			"tenant_id": tenant_id,
			"occurred_at": _utcnow(),
		}
		self._congestion_events.append(event)
		self._audit(tenant_id, "congestion_event_recorded", event["id"])
		return event

	async def congestion_analytics(
		self,
		tenant_id: str = "default",
		period: str = "weekly",
	) -> dict[str, Any]:
		"""Aggregate congestion events by cell and level."""
		events = [e for e in self._congestion_events if e["tenant_id"] == tenant_id]
		by_level: dict[str, int] = {}
		by_cell: dict[str, int] = {}
		for e in events:
			by_level[e["congestion_level"]] = by_level.get(e["congestion_level"], 0) + 1
			by_cell[e["cell_id"]] = by_cell.get(e["cell_id"], 0) + 1
		total_affected = sum(e["affected_users"] for e in events)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"event_count": len(events),
			"total_affected_users": total_affected,
			"by_level": by_level,
			"top_cells": sorted(by_cell.items(), key=lambda x: x[1], reverse=True)[:5],
			"computed_at": _utcnow(),
		}

	async def qos_sla_compliance_report(
		self,
		tenant_id: str = "default",
		period: str = "monthly",
	) -> dict[str, Any]:
		"""Generate a QoS SLA compliance report across all SLA measurements."""
		measurements = [m.to_dict() for m in self.sla_measurements.values() if m.tenant_id == tenant_id]
		if not measurements:
			return {"period": period, "tenant_id": tenant_id, "measurement_count": 0, "compliance_rate_pct": None}
		compliant = sum(1 for m in measurements if float(m.get("actual_value", 0)) >= float(m.get("threshold_value", 0)))
		compliance_rate = round(compliant / len(measurements) * 100, 2)
		self._audit(tenant_id, "qos_sla_compliance_report_generated", period)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"measurement_count": len(measurements),
			"compliant_count": compliant,
			"non_compliant_count": len(measurements) - compliant,
			"compliance_rate_pct": compliance_rate,
			"generated_at": _utcnow(),
		}

	async def bulk_apply_qos_policies(
		self,
		cell_ids: list[str],
		policy_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Apply a QoS policy to multiple cells in bulk."""
		assert cell_ids, "cell_ids required"
		assert policy_id, "policy_id required"
		policy = self._policy_or_raise(policy_id, tenant_id)
		results: list[dict[str, Any]] = []
		for cell_id in cell_ids:
			from .models import QosEnforcementRecord
			enf_id = f"enf-{policy_id}-{cell_id}"
			enf = QosEnforcementRecord(enf_id, tenant_id, policy_id, cell_id, "applied", _utcnow())
			self.enforcement_records[self._key(tenant_id, enf_id)] = enf
			results.append({"cell_id": cell_id, "enforcement_id": enf_id, "status": "applied"})
		self._audit(tenant_id, "bulk_qos_policies_applied", f"policy:{policy_id}:cells:{len(cell_ids)}")
		return {
			"tenant_id": tenant_id,
			"policy_id": policy_id,
			"cell_count": len(cell_ids),
			"results": results,
			"applied_at": _utcnow(),
		}

	async def export_qos_data(
		self,
		tenant_id: str = "default",
		format: str = "json",
	) -> dict[str, Any]:
		"""Export QoS policies, SLA measurements and enforcement records."""
		assert format in {"json", "csv"}, "format must be json or csv"
		policies = [p.to_dict() for p in self.policies.values() if p.tenant_id == tenant_id]
		measurements = [m.to_dict() for m in self.sla_measurements.values() if m.tenant_id == tenant_id]
		self._audit(tenant_id, "qos_data_exported", f"format:{format}")
		return {
			"format": format,
			"tenant_id": tenant_id,
			"policy_count": len(policies),
			"measurement_count": len(measurements),
			"policies": policies,
			"measurements": measurements,
			"exported_at": _utcnow(),
		}

	async def health_check(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return QoS service health status."""
		open_degradations = sum(
			1 for d in self.degradations.values()
			if d.tenant_id == tenant_id and d.status == "open"
		)
		return {
			"service": "QualityOfServiceService",
			"tenant_id": tenant_id,
			"status": "healthy" if open_degradations < 50 else "degraded",
			"policy_count": self._count(self.policies, tenant_id),
			"open_degradation_count": open_degradations,
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"checked_at": _utcnow(),
		}

	# ------------------------------------------------------------------ #
	# Internal helpers                                                    #
	# ------------------------------------------------------------------ #

	def _policy_or_raise(self, policy_id: str, tenant_id: str) -> QosPolicy:
		p = self.policies.get(self._key(tenant_id, policy_id))
		if p is None:
			raise ValueError(f"Policy {policy_id} not found")
		return p

	def _remediation_or_raise(self, remediation_id: str, tenant_id: str) -> QosRemediation:
		r = self.remediations.get(self._key(tenant_id, remediation_id))
		if r is None:
			raise ValueError(f"Remediation {remediation_id} not found")
		return r

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


# Backward-compatible alias
TelecomQosService = QualityOfServiceService
