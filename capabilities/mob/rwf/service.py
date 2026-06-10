"""Async service layer for APG Remote Workforce."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from uuid6 import uuid7
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache


def uuid7str() -> str:
	return str(uuid7())


try:
	from .capability_contract import (
		SUPPORTED_COMPLIANCE_CHECK_TYPES,
		SUPPORTED_EQUIPMENT_STATES,
		SUPPORTED_EQUIPMENT_TYPES,
		SUPPORTED_INCIDENT_TYPES,
		SUPPORTED_ONBOARDING_STATES,
		SUPPORTED_ONBOARDING_STEP_TYPES,
		SUPPORTED_PRODUCTIVITY_METRICS,
		SUPPORTED_VPN_PROTOCOLS,
		SUPPORTED_VPN_STATES,
		SUPPORTED_WORK_POLICY_STATES,
		SUPPORTED_WORK_POLICY_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import (
		ComplianceCheckCreate,
		ComplianceCheckResponse,
		EquipmentRequisitionCreate,
		EquipmentRequisitionResponse,
		OnboardingRecordCreate,
		OnboardingRecordResponse,
		OnboardingStepCreate,
		OnboardingStepResponse,
		PolicyAcknowledgmentCreate,
		PolicyAcknowledgmentResponse,
		ProductivityMetricCreate,
		ProductivityMetricResponse,
		RemoteIncidentCreate,
		RemoteIncidentResponse,
		VpnAccessCreate,
		VpnAccessResponse,
		VpnSessionResponse,
		WorkPolicyCreate,
		WorkPolicyResponse,
		WorkPolicyUpdate,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_COMPLIANCE_CHECK_TYPES,
		SUPPORTED_EQUIPMENT_STATES,
		SUPPORTED_EQUIPMENT_TYPES,
		SUPPORTED_INCIDENT_TYPES,
		SUPPORTED_ONBOARDING_STATES,
		SUPPORTED_ONBOARDING_STEP_TYPES,
		SUPPORTED_PRODUCTIVITY_METRICS,
		SUPPORTED_VPN_PROTOCOLS,
		SUPPORTED_VPN_STATES,
		SUPPORTED_WORK_POLICY_STATES,
		SUPPORTED_WORK_POLICY_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from models import (  # type: ignore
		ComplianceCheckCreate,
		ComplianceCheckResponse,
		EquipmentRequisitionCreate,
		EquipmentRequisitionResponse,
		OnboardingRecordCreate,
		OnboardingRecordResponse,
		OnboardingStepCreate,
		OnboardingStepResponse,
		PolicyAcknowledgmentCreate,
		PolicyAcknowledgmentResponse,
		ProductivityMetricCreate,
		ProductivityMetricResponse,
		RemoteIncidentCreate,
		RemoteIncidentResponse,
		VpnAccessCreate,
		VpnAccessResponse,
		VpnSessionResponse,
		WorkPolicyCreate,
		WorkPolicyResponse,
		WorkPolicyUpdate,
	)


def _present(v: str | None) -> bool:
	return bool(v and v.strip())


class RemoteWorkforceService:
	"""Tenant-scoped runtime for the Remote Workforce capability."""

	def __init__(self) -> None:
		self._work_policies: dict[tuple[str, str], WorkPolicyResponse] = {}
		self._acknowledgments: dict[tuple[str, str], PolicyAcknowledgmentResponse] = {}
		self._vpn_access: dict[tuple[str, str], VpnAccessResponse] = {}
		self._vpn_sessions: dict[tuple[str, str], VpnSessionResponse] = {}
		self._productivity_metrics: dict[tuple[str, str], ProductivityMetricResponse] = {}
		self._equipment: dict[tuple[str, str], EquipmentRequisitionResponse] = {}
		self._onboarding_records: dict[tuple[str, str], OnboardingRecordResponse] = {}
		self._onboarding_steps: dict[tuple[str, str], OnboardingStepResponse] = {}
		self._compliance_checks: dict[tuple[str, str], ComplianceCheckResponse] = {}
		self._incidents: dict[tuple[str, str], RemoteIncidentResponse] = {}
		self._audit_events: list[dict[str, Any]] = []
		# track per-employee equipment count: (tenant_id, employee_id) -> count
		self._employee_equipment_count: dict[tuple[str, str], int] = {}

	# -------------------------------------------------------------------------
	# Contract
	# -------------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# -------------------------------------------------------------------------
	# Work Policies
	# -------------------------------------------------------------------------

	async def create_work_policy(self, payload: WorkPolicyCreate) -> WorkPolicyResponse:
		"""Create a remote work policy."""
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_work_policy",
			"policy_type_supported": payload.policy_type in SUPPORTED_WORK_POLICY_TYPES,
		})
		policy = WorkPolicyResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			name=payload.name,
			policy_type=payload.policy_type,
			description=payload.description,
			content=payload.content,
			effective_date=payload.effective_date,
			expiry_date=payload.expiry_date,
			applicable_roles=payload.applicable_roles,
			geographic_scope=payload.geographic_scope,
			state="draft",
			version=1,
			created_by=payload.created_by,
		)
		self._work_policies[self._key(payload.tenant_id, policy.id)] = policy
		self._audit(payload.tenant_id, "work_policy_created", policy.id)
		return policy

	async def activate_work_policy(self, tenant_id: str, policy_id: str, approval_reference: str, activated_by: str) -> WorkPolicyResponse:
		"""Activate a work policy after approval."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "activate_work_policy",
			"approval_present": _present(approval_reference),
		})
		policy = self._require_policy(tenant_id, policy_id)
		policy.state = "active"
		policy.approval_reference = approval_reference
		policy.updated_at = datetime.utcnow()
		self._audit(tenant_id, "work_policy_activated", policy_id)
		return policy

	async def update_work_policy(self, tenant_id: str, policy_id: str, payload: WorkPolicyUpdate) -> WorkPolicyResponse:
		"""Update a work policy (increments version)."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
		})
		policy = self._require_policy(tenant_id, policy_id)
		if payload.name:
			policy.name = payload.name
		if payload.description is not None:
			policy.description = payload.description
		if payload.content is not None:
			policy.content = payload.content
		if payload.applicable_roles is not None:
			policy.applicable_roles = payload.applicable_roles
		if payload.geographic_scope is not None:
			policy.geographic_scope = payload.geographic_scope
		if payload.expiry_date is not None:
			policy.expiry_date = payload.expiry_date
		if payload.state:
			assert payload.state in SUPPORTED_WORK_POLICY_STATES
			policy.state = payload.state
		policy.version += 1
		policy.updated_at = datetime.utcnow()
		self._audit(tenant_id, "work_policy_updated", policy_id)
		return policy

	async def list_work_policies(self, tenant_id: str, policy_type: str | None = None, state: str | None = None) -> list[WorkPolicyResponse]:
		"""List work policies."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		policies = [p for p in self._work_policies.values() if p.tenant_id == tenant_id]
		if policy_type:
			policies = [p for p in policies if p.policy_type == policy_type]
		if state:
			policies = [p for p in policies if p.state == state]
		return sorted(policies, key=lambda p: p.created_at)

	async def get_work_policy(self, tenant_id: str, policy_id: str) -> WorkPolicyResponse:
		"""Get a work policy by ID."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		return self._require_policy(tenant_id, policy_id)

	async def acknowledge_policy(self, payload: PolicyAcknowledgmentCreate) -> PolicyAcknowledgmentResponse:
		"""Record an employee's acknowledgment of a work policy."""
		policy = self._require_policy(payload.tenant_id, payload.policy_id)
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "acknowledge_policy",
			"policy_state": policy.state,
		})
		ack = PolicyAcknowledgmentResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			policy_id=payload.policy_id,
			employee_id=payload.employee_id,
			acknowledged_at=payload.acknowledged_at,
			ip_address=payload.ip_address,
			device_id=payload.device_id,
			created_by=payload.created_by,
		)
		self._acknowledgments[self._key(payload.tenant_id, ack.id)] = ack
		policy.acknowledgment_count += 1
		policy.updated_at = datetime.utcnow()
		self._audit(payload.tenant_id, "work_policy_acknowledged", ack.id)
		return ack

	async def list_acknowledgments(self, tenant_id: str, policy_id: str | None = None, employee_id: str | None = None) -> list[PolicyAcknowledgmentResponse]:
		"""List policy acknowledgments."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		acks = [a for a in self._acknowledgments.values() if a.tenant_id == tenant_id]
		if policy_id:
			acks = [a for a in acks if a.policy_id == policy_id]
		if employee_id:
			acks = [a for a in acks if a.employee_id == employee_id]
		return sorted(acks, key=lambda a: a.acknowledged_at)

	# -------------------------------------------------------------------------
	# VPN Access
	# -------------------------------------------------------------------------

	async def provision_vpn(self, payload: VpnAccessCreate) -> VpnAccessResponse:
		"""Provision VPN access for an employee."""
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "provision_vpn",
			"vpn_protocol_supported": payload.vpn_protocol in SUPPORTED_VPN_PROTOCOLS,
			"approval_present": _present(payload.approval_reference),
			"mfa_verified": payload.mfa_verified,
			"split_tunneling_requested": payload.split_tunneling_requested,
		})
		access = VpnAccessResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			employee_id=payload.employee_id,
			vpn_protocol=payload.vpn_protocol,
			approval_reference=payload.approval_reference,
			mfa_verified=payload.mfa_verified,
			split_tunneling_enabled=False,
			allowed_networks=payload.allowed_networks,
			state="active",
			provisioned_at=datetime.utcnow(),
			expires_at=datetime.utcnow() + timedelta(hours=12),
			created_by=payload.created_by,
		)
		self._vpn_access[self._key(payload.tenant_id, access.id)] = access
		self._audit(payload.tenant_id, "vpn_access_provisioned", access.id)
		return access

	async def revoke_vpn(self, tenant_id: str, access_id: str, reason: str, revoked_by: str) -> VpnAccessResponse:
		"""Revoke VPN access."""
		access = self._require_vpn(tenant_id, access_id)
		access.state = "revoked"
		access.revoked_at = datetime.utcnow()
		access.revocation_reason = reason
		access.updated_at = datetime.utcnow()
		self._audit(tenant_id, "vpn_access_revoked", access_id)
		return access

	async def start_vpn_session(self, tenant_id: str, access_id: str, client_ip: str | None = None) -> VpnSessionResponse:
		"""Start a VPN session for active access."""
		access = self._require_vpn(tenant_id, access_id)
		self._enforce({"vpn_state": access.state} if access.state in ("revoked", "suspended") else {})
		session = VpnSessionResponse(
			id=uuid7str(),
			tenant_id=tenant_id,
			vpn_access_id=access_id,
			employee_id=access.employee_id,
			started_at=datetime.utcnow(),
			client_ip=client_ip,
		)
		self._vpn_sessions[self._key(tenant_id, session.id)] = session
		self._audit(tenant_id, "vpn_session_started", session.id)
		return session

	async def end_vpn_session(self, tenant_id: str, session_id: str, bytes_in: int = 0, bytes_out: int = 0) -> VpnSessionResponse:
		"""End a VPN session."""
		session = self._require_vpn_session(tenant_id, session_id)
		session.ended_at = datetime.utcnow()
		session.bytes_in = bytes_in
		session.bytes_out = bytes_out
		session.duration_seconds = int((session.ended_at - session.started_at).total_seconds())
		session.updated_at = datetime.utcnow()
		self._audit(tenant_id, "vpn_session_ended", session_id)
		return session

	async def list_vpn_access(self, tenant_id: str, employee_id: str | None = None, state: str | None = None) -> list[VpnAccessResponse]:
		"""List VPN access records."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		records = [a for a in self._vpn_access.values() if a.tenant_id == tenant_id]
		if employee_id:
			records = [a for a in records if a.employee_id == employee_id]
		if state:
			records = [a for a in records if a.state == state]
		return sorted(records, key=lambda a: a.provisioned_at)

	# -------------------------------------------------------------------------
	# Productivity
	# -------------------------------------------------------------------------

	async def record_productivity_metric(self, payload: ProductivityMetricCreate) -> ProductivityMetricResponse:
		"""Record a productivity metric (requires consent)."""
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_productivity",
			"consent_given": payload.consent_given,
			"metric_type_supported": payload.metric_type in SUPPORTED_PRODUCTIVITY_METRICS,
		})
		metric = ProductivityMetricResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			employee_id=payload.employee_id,
			metric_type=payload.metric_type,
			value=payload.value,
			period_start=payload.period_start,
			period_end=payload.period_end,
			consent_given=payload.consent_given,
			notes=payload.notes,
			created_by=payload.created_by,
		)
		self._productivity_metrics[self._key(payload.tenant_id, metric.id)] = metric
		self._audit(payload.tenant_id, "productivity_metric_recorded", metric.id)
		return metric

	async def get_productivity_summary(self, tenant_id: str, employee_id: str) -> dict[str, Any]:
		"""Aggregate productivity metrics for an employee."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		metrics = [m for m in self._productivity_metrics.values() if m.tenant_id == tenant_id and m.employee_id == employee_id]
		by_type: dict[str, list[float]] = {}
		for m in metrics:
			by_type.setdefault(m.metric_type, []).append(m.value)
		averages = {k: sum(v) / len(v) for k, v in by_type.items()}
		return {
			"employee_id": employee_id,
			"tenant_id": tenant_id,
			"total_records": len(metrics),
			"metric_averages": averages,
		}

	async def list_productivity_metrics(self, tenant_id: str, employee_id: str | None = None, metric_type: str | None = None) -> list[ProductivityMetricResponse]:
		"""List productivity metrics."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		metrics = [m for m in self._productivity_metrics.values() if m.tenant_id == tenant_id]
		if employee_id:
			metrics = [m for m in metrics if m.employee_id == employee_id]
		if metric_type:
			metrics = [m for m in metrics if m.metric_type == metric_type]
		return sorted(metrics, key=lambda m: m.created_at)

	# -------------------------------------------------------------------------
	# Equipment Requisition
	# -------------------------------------------------------------------------

	async def request_equipment(self, payload: EquipmentRequisitionCreate) -> EquipmentRequisitionResponse:
		"""Submit an equipment requisition."""
		emp_key = (payload.tenant_id, payload.employee_id)
		current_count = self._employee_equipment_count.get(emp_key, 0)
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "request_equipment",
			"equipment_type_supported": payload.equipment_type in SUPPORTED_EQUIPMENT_TYPES,
			"equipment_limit_exceeded": (current_count + payload.quantity) > 5,
		})
		req = EquipmentRequisitionResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			employee_id=payload.employee_id,
			equipment_type=payload.equipment_type,
			quantity=payload.quantity,
			justification=payload.justification,
			delivery_address=payload.delivery_address,
			state="requested",
			created_by=payload.created_by,
		)
		self._equipment[self._key(payload.tenant_id, req.id)] = req
		self._audit(payload.tenant_id, "equipment_requested", req.id)
		return req

	async def approve_equipment(self, tenant_id: str, requisition_id: str, approval_reference: str, approved_by: str) -> EquipmentRequisitionResponse:
		"""Approve an equipment requisition."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "approve_equipment",
			"approval_present": _present(approval_reference),
		})
		req = self._require_equipment(tenant_id, requisition_id)
		req.state = "approved"
		req.approval_reference = approval_reference
		req.updated_at = datetime.utcnow()
		emp_key = (tenant_id, req.employee_id)
		self._employee_equipment_count[emp_key] = self._employee_equipment_count.get(emp_key, 0) + req.quantity
		self._audit(tenant_id, "equipment_approved", requisition_id)
		return req

	async def ship_equipment(self, tenant_id: str, requisition_id: str, asset_tag: str) -> EquipmentRequisitionResponse:
		"""Mark equipment as shipped."""
		req = self._require_equipment(tenant_id, requisition_id)
		req.state = "shipped"
		req.asset_tag = asset_tag
		req.shipped_at = datetime.utcnow()
		req.updated_at = datetime.utcnow()
		return req

	async def deliver_equipment(self, tenant_id: str, requisition_id: str) -> EquipmentRequisitionResponse:
		"""Mark equipment as delivered."""
		req = self._require_equipment(tenant_id, requisition_id)
		req.state = "delivered"
		req.delivered_at = datetime.utcnow()
		req.updated_at = datetime.utcnow()
		self._audit(tenant_id, "equipment_delivered", requisition_id)
		return req

	async def return_equipment(self, tenant_id: str, requisition_id: str, returned_by: str) -> EquipmentRequisitionResponse:
		"""Mark equipment as returned."""
		req = self._require_equipment(tenant_id, requisition_id)
		req.state = "returned"
		req.returned_at = datetime.utcnow()
		req.updated_at = datetime.utcnow()
		emp_key = (tenant_id, req.employee_id)
		current = self._employee_equipment_count.get(emp_key, 0)
		self._employee_equipment_count[emp_key] = max(0, current - req.quantity)
		self._audit(tenant_id, "equipment_returned", requisition_id)
		return req

	async def list_equipment(self, tenant_id: str, employee_id: str | None = None, state: str | None = None) -> list[EquipmentRequisitionResponse]:
		"""List equipment requisitions."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		items = [e for e in self._equipment.values() if e.tenant_id == tenant_id]
		if employee_id:
			items = [e for e in items if e.employee_id == employee_id]
		if state:
			items = [e for e in items if e.state == state]
		return sorted(items, key=lambda e: e.created_at)

	# -------------------------------------------------------------------------
	# Digital Onboarding
	# -------------------------------------------------------------------------

	async def start_onboarding(self, payload: OnboardingRecordCreate) -> OnboardingRecordResponse:
		"""Initiate digital onboarding for a remote employee."""
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "start_onboarding",
			"manager_approval_present": _present(payload.manager_approval_reference),
		})
		all_steps = list(SUPPORTED_ONBOARDING_STEP_TYPES)
		record = OnboardingRecordResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			employee_id=payload.employee_id,
			manager_id=payload.manager_id,
			manager_approval_reference=payload.manager_approval_reference,
			start_date=payload.start_date,
			timezone=payload.timezone,
			collaboration_tools=payload.collaboration_tools,
			state="in_progress",
			pending_steps=all_steps,
			completed_steps=[],
			created_by=payload.created_by,
		)
		self._onboarding_records[self._key(payload.tenant_id, record.id)] = record
		self._audit(payload.tenant_id, "onboarding_started", record.id)
		return record

	async def complete_onboarding_step(self, payload: OnboardingStepCreate) -> OnboardingStepResponse:
		"""Complete a single onboarding step."""
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "complete_onboarding_step",
			"step_type_supported": payload.step_type in SUPPORTED_ONBOARDING_STEP_TYPES,
		})
		record = self._require_onboarding(payload.tenant_id, payload.onboarding_id)
		step = OnboardingStepResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			onboarding_id=payload.onboarding_id,
			step_type=payload.step_type,
			notes=payload.notes,
			completed_by=payload.completed_by,
			completed_at=datetime.utcnow(),
			created_by=payload.created_by,
		)
		self._onboarding_steps[self._key(payload.tenant_id, step.id)] = step
		if payload.step_type not in record.completed_steps:
			record.completed_steps.append(payload.step_type)
		if payload.step_type in record.pending_steps:
			record.pending_steps.remove(payload.step_type)
		if not record.pending_steps:
			record.state = "completed"
			record.completed_at = datetime.utcnow()
			self._audit(payload.tenant_id, "onboarding_completed", record.id)
		else:
			self._audit(payload.tenant_id, "onboarding_step_completed", step.id)
		record.updated_at = datetime.utcnow()
		return step

	async def list_onboarding_records(self, tenant_id: str, state: str | None = None, employee_id: str | None = None) -> list[OnboardingRecordResponse]:
		"""List onboarding records."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		records = [r for r in self._onboarding_records.values() if r.tenant_id == tenant_id]
		if state:
			records = [r for r in records if r.state == state]
		if employee_id:
			records = [r for r in records if r.employee_id == employee_id]
		return sorted(records, key=lambda r: r.created_at)

	async def get_onboarding_record(self, tenant_id: str, record_id: str) -> OnboardingRecordResponse:
		"""Get an onboarding record."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		return self._require_onboarding(tenant_id, record_id)

	# -------------------------------------------------------------------------
	# Compliance Checks
	# -------------------------------------------------------------------------

	async def record_compliance_check(self, payload: ComplianceCheckCreate) -> ComplianceCheckResponse:
		"""Record a remote compliance check result."""
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_compliance_check",
			"check_type_supported": payload.check_type in SUPPORTED_COMPLIANCE_CHECK_TYPES,
		})
		check = ComplianceCheckResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			employee_id=payload.employee_id,
			check_type=payload.check_type,
			result=payload.result,
			evidence_reference=payload.evidence_reference,
			notes=payload.notes,
			next_due_at=datetime.utcnow() + timedelta(days=30),
			created_by=payload.created_by,
		)
		self._compliance_checks[self._key(payload.tenant_id, check.id)] = check
		self._audit(payload.tenant_id, "compliance_check_completed", check.id)
		return check

	async def list_compliance_checks(self, tenant_id: str, employee_id: str | None = None, check_type: str | None = None, result: str | None = None) -> list[ComplianceCheckResponse]:
		"""List compliance checks."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		checks = [c for c in self._compliance_checks.values() if c.tenant_id == tenant_id]
		if employee_id:
			checks = [c for c in checks if c.employee_id == employee_id]
		if check_type:
			checks = [c for c in checks if c.check_type == check_type]
		if result:
			checks = [c for c in checks if c.result == result]
		return sorted(checks, key=lambda c: c.created_at)

	# -------------------------------------------------------------------------
	# Remote Incidents
	# -------------------------------------------------------------------------

	async def raise_incident(self, payload: RemoteIncidentCreate) -> RemoteIncidentResponse:
		"""Raise a remote workforce incident."""
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "raise_incident",
			"incident_type_supported": payload.incident_type in SUPPORTED_INCIDENT_TYPES,
		})
		incident = RemoteIncidentResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			employee_id=payload.employee_id,
			incident_type=payload.incident_type,
			description=payload.description,
			severity=payload.severity,
			reported_by=payload.reported_by,
			state="open",
			created_by=payload.created_by,
		)
		self._incidents[self._key(payload.tenant_id, incident.id)] = incident
		self._audit(payload.tenant_id, "remote_incident_raised", incident.id)
		return incident

	async def resolve_incident(self, tenant_id: str, incident_id: str, resolution_notes: str, resolved_by: str) -> RemoteIncidentResponse:
		"""Resolve a remote incident."""
		incident = self._require_incident(tenant_id, incident_id)
		incident.state = "resolved"
		incident.resolution_notes = resolution_notes
		incident.resolved_at = datetime.utcnow()
		incident.resolved_by = resolved_by
		incident.updated_at = datetime.utcnow()
		self._audit(tenant_id, "remote_incident_resolved", incident_id)
		return incident

	async def list_incidents(self, tenant_id: str, employee_id: str | None = None, incident_type: str | None = None, state: str | None = None) -> list[RemoteIncidentResponse]:
		"""List remote incidents."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		incidents = [i for i in self._incidents.values() if i.tenant_id == tenant_id]
		if employee_id:
			incidents = [i for i in incidents if i.employee_id == employee_id]
		if incident_type:
			incidents = [i for i in incidents if i.incident_type == incident_type]
		if state:
			incidents = [i for i in incidents if i.state == state]
		return sorted(incidents, key=lambda i: i.created_at)

	# -------------------------------------------------------------------------
	# Dashboard
	# -------------------------------------------------------------------------

	async def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		"""High-level remote workforce dashboard."""
		policies = [p for p in self._work_policies.values() if p.tenant_id == tenant_id]
		vpn = [v for v in self._vpn_access.values() if v.tenant_id == tenant_id]
		equipment = [e for e in self._equipment.values() if e.tenant_id == tenant_id]
		onboarding = [o for o in self._onboarding_records.values() if o.tenant_id == tenant_id]
		incidents = [i for i in self._incidents.values() if i.tenant_id == tenant_id]
		compliance = [c for c in self._compliance_checks.values() if c.tenant_id == tenant_id]
		return {
			"total_work_policies": len(policies),
			"active_policies": sum(1 for p in policies if p.state == "active"),
			"active_vpn_access": sum(1 for v in vpn if v.state == "active"),
			"equipment_requests": len(equipment),
			"pending_equipment": sum(1 for e in equipment if e.state == "requested"),
			"onboarding_in_progress": sum(1 for o in onboarding if o.state == "in_progress"),
			"open_incidents": sum(1 for i in incidents if i.state == "open"),
			"compliance_failures": sum(1 for c in compliance if c.result == "fail"),
		}

	# -------------------------------------------------------------------------
	# Private helpers
	# -------------------------------------------------------------------------

	def _log_policy_summary(self, tenant_id: str) -> str:
		count = sum(1 for p in self._work_policies.values() if p.tenant_id == tenant_id and p.state == "active")
		return f"tenant={tenant_id} active_policies={count}"

	def _log_vpn_summary(self, tenant_id: str) -> str:
		count = sum(1 for v in self._vpn_access.values() if v.tenant_id == tenant_id and v.state == "active")
		return f"tenant={tenant_id} active_vpn={count}"

	def _key(self, tenant_id: str, entity_id: str) -> tuple[str, str]:
		return (tenant_id, entity_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			raise ValueError(f"{result['reason']}: {result['required_action']}")

	def _audit(self, tenant_id: str, event_type: str, entity_id: str) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"entity_id": entity_id,
			"timestamp": datetime.utcnow().isoformat(),
		})

	def _require_policy(self, tenant_id: str, policy_id: str) -> WorkPolicyResponse:
		p = self._work_policies.get((tenant_id, policy_id))
		assert p is not None, f"work_policy_not_found: {policy_id}"
		return p

	def _require_vpn(self, tenant_id: str, access_id: str) -> VpnAccessResponse:
		v = self._vpn_access.get((tenant_id, access_id))
		assert v is not None, f"vpn_access_not_found: {access_id}"
		return v

	def _require_vpn_session(self, tenant_id: str, session_id: str) -> VpnSessionResponse:
		s = self._vpn_sessions.get((tenant_id, session_id))
		assert s is not None, f"vpn_session_not_found: {session_id}"
		return s

	def _require_equipment(self, tenant_id: str, req_id: str) -> EquipmentRequisitionResponse:
		e = self._equipment.get((tenant_id, req_id))
		assert e is not None, f"equipment_requisition_not_found: {req_id}"
		return e

	def _require_onboarding(self, tenant_id: str, record_id: str) -> OnboardingRecordResponse:
		r = self._onboarding_records.get((tenant_id, record_id))
		assert r is not None, f"onboarding_record_not_found: {record_id}"
		return r

	# ── 6 new methods ───────────────────────────────────────────────────────

	async def equipment_request(
		self,
		tenant_id: str,
		employee_id: str,
		items_needed: list[str],
		urgency: str = "normal",
		requested_by: str | None = None,
	) -> dict[str, Any]:
		"""Submit an equipment requisition for a remote worker."""
		from .models import EquipmentRequisitionCreate
		payload = EquipmentRequisitionCreate(
			tenant_id=tenant_id,
			employee_id=employee_id,
			items=items_needed,
			urgency=urgency,
			created_by=requested_by or employee_id,
		)
		return await self.request_equipment(payload)

	async def virtual_onboarding(
		self,
		tenant_id: str,
		employee_id: str,
		onboarding_checklist: list[str],
		assigned_buddy: str | None = None,
		created_by: str = "hr",
	) -> dict[str, Any]:
		"""Initiate a virtual onboarding sequence for a new remote employee."""
		from .models import OnboardingRecordCreate
		payload = OnboardingRecordCreate(
			tenant_id=tenant_id,
			employee_id=employee_id,
			checklist=onboarding_checklist,
			assigned_buddy=assigned_buddy,
			created_by=created_by,
		)
		return await self.create_onboarding_record(payload)

	async def async_collaboration_session(
		self,
		tenant_id: str,
		participants: list[str],
		project_id: str,
		session_type: str = "async_standup",
		created_by: str = "team_lead",
	) -> dict[str, Any]:
		"""Create an async collaboration session for distributed teams."""
		session_id = f"collab-{project_id[:6]}-{len(self._audit_events)+1}"
		self._audit(tenant_id, "async_collaboration_session_created", session_id)
		return {
			"session_id": session_id,
			"tenant_id": tenant_id,
			"project_id": project_id,
			"participants": participants,
			"session_type": session_type,
			"status": "active",
			"created_by": created_by,
			"created_at": datetime.utcnow().isoformat(),
		}

	async def productivity_report(
		self,
		tenant_id: str,
		employee_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return a remote worker productivity summary for a period."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		incidents = [i for i in self._incidents.values() if i.tenant_id == tenant_id and i.employee_id == employee_id]
		onboarding = [o for o in self._onboarding_records.values() if o.tenant_id == tenant_id and o.employee_id == employee_id]
		completed_checklist = sum(
			sum(1 for item in getattr(o, "checklist_progress", {}).values() if item)
			for o in onboarding
		)
		return {
			"employee_id": employee_id,
			"tenant_id": tenant_id,
			"period": period,
			"remote_incidents": len(incidents),
			"onboarding_items_completed": completed_checklist,
			"vpn_sessions": len([s for s in self._vpn_sessions.values() if s.tenant_id == tenant_id and getattr(s, "employee_id", None) == employee_id]),
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def security_compliance_remote(
		self,
		tenant_id: str,
		employee_id: str,
	) -> dict[str, Any]:
		"""Check remote worker security compliance status."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		policies = [p for p in self._work_policies.values() if p.tenant_id == tenant_id and p.state == "active"]
		vpn_active = any(
			s for s in self._vpn_sessions.values()
			if s.tenant_id == tenant_id and s.status == "active"
		)
		equipment = [e for e in self._equipment.values() if e.tenant_id == tenant_id and e.employee_id == employee_id]
		issues: list[str] = []
		if not vpn_active:
			issues.append("no_active_vpn_session")
		if not equipment:
			issues.append("no_equipment_on_record")
		if not policies:
			issues.append("no_active_work_policy")
		return {
			"employee_id": employee_id,
			"tenant_id": tenant_id,
			"compliant": len(issues) == 0,
			"issues": issues,
			"active_policies": len(policies),
			"vpn_connected": vpn_active,
			"equipment_on_record": len(equipment),
			"checked_at": datetime.utcnow().isoformat(),
		}

	async def rwf_analytics(
		self,
		tenant_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return remote workforce analytics for a period."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		policies = [p for p in self._work_policies.values() if p.tenant_id == tenant_id]
		vpn = [v for v in self._vpn_access.values() if v.tenant_id == tenant_id]
		equipment = [e for e in self._equipment.values() if e.tenant_id == tenant_id]
		onboarding = [o for o in self._onboarding_records.values() if o.tenant_id == tenant_id]
		incidents = [i for i in self._incidents.values() if i.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"period": period,
			"active_work_policies": sum(1 for p in policies if p.state == "active"),
			"vpn_access_records": len(vpn),
			"equipment_requests": len(equipment),
			"onboarding_records": len(onboarding),
			"remote_incidents": len(incidents),
			"audit_events": sum(1 for e in self._audit_events if e.get("tenant_id") == tenant_id),
		}

	def _require_incident(self, tenant_id: str, incident_id: str) -> RemoteIncidentResponse:
		i = self._incidents.get((tenant_id, incident_id))
		assert i is not None, f"incident_not_found: {incident_id}"
		return i

	async def ml_field_worker_route_optimize(self, *args, **kwargs):
		"""AI-powered AI field worker route and schedule optimization. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="field_worker_route_optimization")
			return {"route_score": round(result.score,3), "rationale": result.rationale, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

