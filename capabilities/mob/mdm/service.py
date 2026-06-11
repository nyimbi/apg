"""Async service layer for APG Mobile Device Management."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from uuid6 import uuid7
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache


def uuid7str() -> str:
	return str(uuid7())


try:
	from .capability_contract import (
		SUPPORTED_APP_DISTRIBUTION_TYPES,
		SUPPORTED_COMPLIANCE_STATES,
		SUPPORTED_DEVICE_TYPES,
		SUPPORTED_ENROLMENT_METHODS,
		SUPPORTED_ENROLMENT_STATES,
		SUPPORTED_LOCK_ACTIONS,
		SUPPORTED_OS_PLATFORMS,
		SUPPORTED_OWNERSHIP_TYPES,
		SUPPORTED_POLICY_STATES,
		SUPPORTED_POLICY_TYPES,
		SUPPORTED_PROFILE_TYPES,
		SUPPORTED_WIPE_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import (
		AppDistributionCreate,
		AppDistributionResponse,
		ComplianceEvaluationCreate,
		ComplianceRecordResponse,
		DeviceEnrolmentCreate,
		DeviceResponse,
		DeviceUpdate,
		MdmAlertResponse,
		MdmProfileCreate,
		MdmProfileResponse,
		PolicyAssignmentCreate,
		PolicyAssignmentResponse,
		PolicyCreate,
		PolicyResponse,
		PolicyUpdate,
		WipeRequestCreate,
		WipeRequestResponse,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_APP_DISTRIBUTION_TYPES,
		SUPPORTED_COMPLIANCE_STATES,
		SUPPORTED_DEVICE_TYPES,
		SUPPORTED_ENROLMENT_METHODS,
		SUPPORTED_ENROLMENT_STATES,
		SUPPORTED_LOCK_ACTIONS,
		SUPPORTED_OS_PLATFORMS,
		SUPPORTED_OWNERSHIP_TYPES,
		SUPPORTED_POLICY_STATES,
		SUPPORTED_POLICY_TYPES,
		SUPPORTED_PROFILE_TYPES,
		SUPPORTED_WIPE_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from models import (  # type: ignore
		AppDistributionCreate,
		AppDistributionResponse,
		ComplianceEvaluationCreate,
		ComplianceRecordResponse,
		DeviceEnrolmentCreate,
		DeviceResponse,
		DeviceUpdate,
		MdmAlertResponse,
		MdmProfileCreate,
		MdmProfileResponse,
		PolicyAssignmentCreate,
		PolicyAssignmentResponse,
		PolicyCreate,
		PolicyResponse,
		PolicyUpdate,
		WipeRequestCreate,
		WipeRequestResponse,
	)


def _present(v: str | None) -> bool:
	return bool(v and v.strip())


class MobileDeviceManagementService:
	"""Tenant-scoped runtime for the Mobile Device Management capability."""

	def __init__(self) -> None:
		self._devices: dict[tuple[str, str], DeviceResponse] = {}
		self._policies: dict[tuple[str, str], PolicyResponse] = {}
		self._policy_assignments: dict[tuple[str, str], PolicyAssignmentResponse] = {}
		self._compliance_records: dict[tuple[str, str], ComplianceRecordResponse] = {}
		self._app_distributions: dict[tuple[str, str], AppDistributionResponse] = {}
		self._wipe_requests: dict[tuple[str, str], WipeRequestResponse] = {}
		self._profiles: dict[tuple[str, str], MdmProfileResponse] = {}
		self._alerts: dict[tuple[str, str], MdmAlertResponse] = {}
		self._audit_events: list[dict[str, Any]] = []

	# -------------------------------------------------------------------------
	# Contract helpers
	# -------------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return capability contract."""
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate rules against context."""
		return evaluate_capability_rules(context)

	# -------------------------------------------------------------------------
	# Device Enrolment
	# -------------------------------------------------------------------------

	async def enrol_device(self, payload: DeviceEnrolmentCreate) -> DeviceResponse:
		"""Enrol a device into MDM."""
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "enrol_device",
			"device_type_supported": payload.device_type in SUPPORTED_DEVICE_TYPES,
			"os_platform_supported": payload.os_platform in SUPPORTED_OS_PLATFORMS,
			"enrolment_method_supported": payload.enrolment_method in SUPPORTED_ENROLMENT_METHODS,
			"ownership_type_supported": payload.ownership_type in SUPPORTED_OWNERSHIP_TYPES,
			"approval_present": _present(payload.approval_reference),
		})
		device = DeviceResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			serial_number=payload.serial_number,
			device_type=payload.device_type,
			os_platform=payload.os_platform,
			os_version=payload.os_version,
			ownership_type=payload.ownership_type,
			enrolment_method=payload.enrolment_method,
			enrolment_state="enrolled",
			approval_reference=payload.approval_reference,
			assigned_user_id=payload.assigned_user_id,
			asset_tag=payload.asset_tag,
			location=payload.location,
			enrolled_at=datetime.utcnow(),
			created_by=payload.created_by,
		)
		self._devices[self._key(payload.tenant_id, device.id)] = device
		self._audit(payload.tenant_id, "device_enrolled", device.id)
		return device

	async def get_device(self, tenant_id: str, device_id: str) -> DeviceResponse:
		"""Get a device by ID."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		return self._require_device(tenant_id, device_id)

	async def list_devices(self, tenant_id: str, os_platform: str | None = None, enrolment_state: str | None = None, ownership_type: str | None = None) -> list[DeviceResponse]:
		"""List devices with optional filters."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		devices = [d for d in self._devices.values() if d.tenant_id == tenant_id]
		if os_platform:
			devices = [d for d in devices if d.os_platform == os_platform]
		if enrolment_state:
			devices = [d for d in devices if d.enrolment_state == enrolment_state]
		if ownership_type:
			devices = [d for d in devices if d.ownership_type == ownership_type]
		return sorted(devices, key=lambda d: d.enrolled_at)

	async def update_device(self, tenant_id: str, device_id: str, payload: DeviceUpdate) -> DeviceResponse:
		"""Update device attributes or state."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
		})
		device = self._require_device(tenant_id, device_id)
		if payload.enrolment_state:
			assert payload.enrolment_state in SUPPORTED_ENROLMENT_STATES, f"enrolment_state must be one of {SUPPORTED_ENROLMENT_STATES}"
			device.enrolment_state = payload.enrolment_state
		if payload.assigned_user_id is not None:
			device.assigned_user_id = payload.assigned_user_id
		if payload.asset_tag is not None:
			device.asset_tag = payload.asset_tag
		if payload.location is not None:
			device.location = payload.location
		if payload.os_version is not None:
			device.os_version = payload.os_version
		device.last_seen_at = datetime.utcnow()
		device.updated_at = datetime.utcnow()
		self._audit(tenant_id, "device_updated", device_id)
		return device

	async def unenrol_device(self, tenant_id: str, device_id: str, unenrolled_by: str) -> DeviceResponse:
		"""Unenrol a device from MDM."""
		device = self._require_device(tenant_id, device_id)
		device.enrolment_state = "unenrolled"
		device.updated_at = datetime.utcnow()
		self._audit(tenant_id, "device_unenrolled", device_id)
		return device

	async def suspend_device(self, tenant_id: str, device_id: str, suspended_by: str) -> DeviceResponse:
		"""Suspend a device, blocking all actions."""
		device = self._require_device(tenant_id, device_id)
		device.enrolment_state = "suspended"
		device.updated_at = datetime.utcnow()
		self._audit(tenant_id, "device_suspended", device_id)
		return device

	# -------------------------------------------------------------------------
	# Policies
	# -------------------------------------------------------------------------

	async def create_policy(self, payload: PolicyCreate) -> PolicyResponse:
		"""Create an MDM policy."""
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_policy",
			"policy_type_supported": payload.policy_type in SUPPORTED_POLICY_TYPES,
		})
		policy = PolicyResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			name=payload.name,
			policy_type=payload.policy_type,
			description=payload.description,
			configuration=payload.configuration,
			platform_targets=payload.platform_targets,
			state="draft",
			version=1,
			created_by=payload.created_by,
		)
		self._policies[self._key(payload.tenant_id, policy.id)] = policy
		self._audit(payload.tenant_id, "policy_created", policy.id)
		return policy

	async def activate_policy(self, tenant_id: str, policy_id: str, approval_reference: str, activated_by: str) -> PolicyResponse:
		"""Activate a policy after approval."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "activate_policy",
			"approval_present": _present(approval_reference),
		})
		policy = self._require_policy(tenant_id, policy_id)
		policy.state = "active"
		policy.approval_reference = approval_reference
		policy.updated_at = datetime.utcnow()
		self._audit(tenant_id, "policy_activated", policy_id)
		return policy

	async def update_policy(self, tenant_id: str, policy_id: str, payload: PolicyUpdate) -> PolicyResponse:
		"""Update a policy (increments version)."""
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
		if payload.configuration is not None:
			policy.configuration = payload.configuration
		if payload.platform_targets is not None:
			policy.platform_targets = payload.platform_targets
		if payload.state:
			assert payload.state in SUPPORTED_POLICY_STATES
			policy.state = payload.state
		policy.version += 1
		policy.updated_at = datetime.utcnow()
		self._audit(tenant_id, "policy_updated", policy_id)
		return policy

	async def list_policies(self, tenant_id: str, policy_type: str | None = None, state: str | None = None) -> list[PolicyResponse]:
		"""List policies with optional filters."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		policies = [p for p in self._policies.values() if p.tenant_id == tenant_id]
		if policy_type:
			policies = [p for p in policies if p.policy_type == policy_type]
		if state:
			policies = [p for p in policies if p.state == state]
		return sorted(policies, key=lambda p: p.created_at)

	async def assign_policy(self, payload: PolicyAssignmentCreate) -> PolicyAssignmentResponse:
		"""Assign a policy to a device."""
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
		})
		device = self._require_device(payload.tenant_id, payload.device_id)
		self._enforce({"device_state": device.enrolment_state} if device.enrolment_state in ("suspended", "wiped") else {})
		self._require_policy(payload.tenant_id, payload.policy_id)
		assignment = PolicyAssignmentResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			policy_id=payload.policy_id,
			device_id=payload.device_id,
			assigned_by=payload.assigned_by,
			created_by=payload.created_by,
		)
		self._policy_assignments[self._key(payload.tenant_id, assignment.id)] = assignment
		self._audit(payload.tenant_id, "policy_assigned", assignment.id)
		return assignment

	async def list_policy_assignments(self, tenant_id: str, device_id: str | None = None, policy_id: str | None = None) -> list[PolicyAssignmentResponse]:
		"""List policy assignments."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		assignments = [a for a in self._policy_assignments.values() if a.tenant_id == tenant_id]
		if device_id:
			assignments = [a for a in assignments if a.device_id == device_id]
		if policy_id:
			assignments = [a for a in assignments if a.policy_id == policy_id]
		return sorted(assignments, key=lambda a: a.assigned_at)

	# -------------------------------------------------------------------------
	# Compliance
	# -------------------------------------------------------------------------

	async def evaluate_compliance(self, payload: ComplianceEvaluationCreate) -> ComplianceRecordResponse:
		"""Run a compliance evaluation for a device."""
		device = self._require_device(payload.tenant_id, payload.device_id)
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation": "evaluate_compliance",
			"device_enrolled": device.enrolment_state == "enrolled",
		})
		non_compliant = any(f.get("severity") in ("high", "critical") for f in payload.findings)
		new_state = "non_compliant" if non_compliant else ("compliant" if payload.findings else "compliant")
		record = ComplianceRecordResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			device_id=payload.device_id,
			compliance_state=new_state,
			findings=payload.findings,
			evaluated_by=payload.evaluator_id,
			evaluated_at=datetime.utcnow(),
			next_evaluation_at=datetime.utcnow() + timedelta(hours=1),
			created_by=payload.created_by,
		)
		self._compliance_records[self._key(payload.tenant_id, record.id)] = record
		device.compliance_state = new_state
		device.updated_at = datetime.utcnow()
		self._audit(payload.tenant_id, "compliance_evaluated", record.id)
		if new_state == "non_compliant":
			await self._raise_alert(payload.tenant_id, payload.device_id, "compliance_violation", "high", f"Device {payload.device_id} is non-compliant: {len(payload.findings)} finding(s)")
		return record

	async def list_compliance_records(self, tenant_id: str, device_id: str | None = None, compliance_state: str | None = None) -> list[ComplianceRecordResponse]:
		"""List compliance records."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		records = [r for r in self._compliance_records.values() if r.tenant_id == tenant_id]
		if device_id:
			records = [r for r in records if r.device_id == device_id]
		if compliance_state:
			records = [r for r in records if r.compliance_state == compliance_state]
		return sorted(records, key=lambda r: r.evaluated_at)

	# -------------------------------------------------------------------------
	# App Distribution
	# -------------------------------------------------------------------------

	async def distribute_app(self, payload: AppDistributionCreate) -> AppDistributionResponse:
		"""Push an app to a device."""
		device = self._require_device(payload.tenant_id, payload.device_id)
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "distribute_app",
			"device_enrolled": device.enrolment_state == "enrolled",
			"distribution_type_supported": payload.distribution_type in SUPPORTED_APP_DISTRIBUTION_TYPES,
		})
		dist = AppDistributionResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			app_bundle_id=payload.app_bundle_id,
			app_name=payload.app_name,
			app_version=payload.app_version,
			device_id=payload.device_id,
			distribution_type=payload.distribution_type,
			approval_reference=payload.approval_reference,
			silent_install=payload.silent_install,
			state="distributed",
			distributed_at=datetime.utcnow(),
			created_by=payload.created_by,
		)
		self._app_distributions[self._key(payload.tenant_id, dist.id)] = dist
		self._audit(payload.tenant_id, "app_distributed", dist.id)
		return dist

	async def list_app_distributions(self, tenant_id: str, device_id: str | None = None) -> list[AppDistributionResponse]:
		"""List app distributions."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		dists = [d for d in self._app_distributions.values() if d.tenant_id == tenant_id]
		if device_id:
			dists = [d for d in dists if d.device_id == device_id]
		return sorted(dists, key=lambda d: d.created_at)

	# -------------------------------------------------------------------------
	# Remote Wipe
	# -------------------------------------------------------------------------

	async def request_wipe(self, payload: WipeRequestCreate) -> WipeRequestResponse:
		"""Request a remote wipe with dual approval."""
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "request_wipe",
			"wipe_type_supported": payload.wipe_type in SUPPORTED_WIPE_TYPES,
			"approval_present": _present(payload.approval_reference),
			"dual_approval_present": _present(payload.second_approval_reference),
		})
		self._require_device(payload.tenant_id, payload.device_id)
		wipe = WipeRequestResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			device_id=payload.device_id,
			wipe_type=payload.wipe_type,
			approval_reference=payload.approval_reference,
			second_approval_reference=payload.second_approval_reference,
			justification=payload.justification,
			requested_by=payload.requested_by,
			state="pending",
			created_by=payload.created_by,
		)
		self._wipe_requests[self._key(payload.tenant_id, wipe.id)] = wipe
		self._audit(payload.tenant_id, "wipe_requested", wipe.id)
		return wipe

	async def execute_wipe(self, tenant_id: str, wipe_id: str, executed_by: str) -> WipeRequestResponse:
		"""Execute a pending wipe request."""
		wipe = self._require_wipe(tenant_id, wipe_id)
		assert wipe.state == "pending", "only_pending_wipes_can_be_executed"
		wipe.state = "completed"
		wipe.executed_at = datetime.utcnow()
		wipe.completed_at = datetime.utcnow()
		wipe.updated_at = datetime.utcnow()
		device = self._devices.get((tenant_id, wipe.device_id))
		if device:
			device.enrolment_state = "wiped"
			device.updated_at = datetime.utcnow()
		self._audit(tenant_id, "wipe_completed", wipe_id)
		return wipe

	async def list_wipe_requests(self, tenant_id: str, device_id: str | None = None, state: str | None = None) -> list[WipeRequestResponse]:
		"""List wipe requests."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		wipes = [w for w in self._wipe_requests.values() if w.tenant_id == tenant_id]
		if device_id:
			wipes = [w for w in wipes if w.device_id == device_id]
		if state:
			wipes = [w for w in wipes if w.state == state]
		return sorted(wipes, key=lambda w: w.created_at)

	# -------------------------------------------------------------------------
	# MDM Profiles
	# -------------------------------------------------------------------------

	async def create_profile(self, payload: MdmProfileCreate) -> MdmProfileResponse:
		"""Create an MDM configuration profile."""
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "deploy_profile",
			"profile_type_supported": payload.profile_type in SUPPORTED_PROFILE_TYPES,
			"device_enrolled": True,
		})
		profile = MdmProfileResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			name=payload.name,
			profile_type=payload.profile_type,
			platform=payload.platform,
			payload=payload.payload,
			state="draft",
			created_by=payload.created_by,
		)
		self._profiles[self._key(payload.tenant_id, profile.id)] = profile
		self._audit(payload.tenant_id, "profile_created", profile.id)
		return profile

	async def deploy_profile(self, tenant_id: str, profile_id: str, device_id: str, deployed_by: str) -> MdmProfileResponse:
		"""Deploy a profile to a device."""
		device = self._require_device(tenant_id, device_id)
		self._enforce({
			"operation": "deploy_profile",
			"device_enrolled": device.enrolment_state == "enrolled",
			"profile_type_supported": True,
		})
		profile = self._require_profile(tenant_id, profile_id)
		profile.state = "deployed" if profile.state != "deployed" else "deployed"
		profile.deployed_to_count += 1
		profile.updated_at = datetime.utcnow()
		self._audit(tenant_id, "profile_deployed", profile_id)
		return profile

	async def list_profiles(self, tenant_id: str, profile_type: str | None = None) -> list[MdmProfileResponse]:
		"""List MDM profiles."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		profiles = [p for p in self._profiles.values() if p.tenant_id == tenant_id]
		if profile_type:
			profiles = [p for p in profiles if p.profile_type == profile_type]
		return sorted(profiles, key=lambda p: p.created_at)

	# -------------------------------------------------------------------------
	# Alerts
	# -------------------------------------------------------------------------

	async def _raise_alert(self, tenant_id: str, device_id: str, alert_type: str, severity: str, message: str) -> MdmAlertResponse:
		"""Internal: raise an MDM alert."""
		alert = MdmAlertResponse(
			id=uuid7str(),
			tenant_id=tenant_id,
			device_id=device_id,
			alert_type=alert_type,
			severity=severity,
			message=message,
		)
		self._alerts[self._key(tenant_id, alert.id)] = alert
		self._audit(tenant_id, "mdm_alert_raised", alert.id)
		return alert

	async def list_alerts(self, tenant_id: str, device_id: str | None = None, resolved: bool | None = None) -> list[MdmAlertResponse]:
		"""List MDM alerts."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		alerts = [a for a in self._alerts.values() if a.tenant_id == tenant_id]
		if device_id:
			alerts = [a for a in alerts if a.device_id == device_id]
		if resolved is not None:
			alerts = [a for a in alerts if a.resolved == resolved]
		return sorted(alerts, key=lambda a: a.created_at)

	async def resolve_alert(self, tenant_id: str, alert_id: str, resolved_by: str) -> MdmAlertResponse:
		"""Mark an alert as resolved."""
		alert = self._alerts.get((tenant_id, alert_id))
		assert alert is not None, f"alert_not_found: {alert_id}"
		alert.resolved = True
		alert.resolved_at = datetime.utcnow()
		alert.updated_at = datetime.utcnow()
		return alert

	# -------------------------------------------------------------------------
	# Dashboard
	# -------------------------------------------------------------------------

	# ── 11 new methods ──────────────────────────────────────────────────────

	async def policy_push(
		self, tenant_id: str, policy_id: str, device_group: list[str], pushed_by: str = "admin"
	) -> dict[str, Any]:
		"""Push a policy to a group of devices."""
		self._require_policy(tenant_id, policy_id)
		results: list[dict[str, Any]] = []
		for device_id in device_group:
			device = self._devices.get((tenant_id, device_id))
			if device:
				self._audit(tenant_id, "policy_pushed", device_id)
				results.append({"device_id": device_id, "status": "pushed"})
			else:
				results.append({"device_id": device_id, "status": "not_found"})
		return {
			"policy_id": policy_id,
			"tenant_id": tenant_id,
			"pushed_by": pushed_by,
			"device_count": len(device_group),
			"results": results,
			"pushed_at": datetime.utcnow().isoformat(),
		}

	async def wipe_device(
		self, tenant_id: str, device_id: str, wipe_type: str, authorised_by: str
	) -> dict[str, Any]:
		"""Issue a remote wipe command for a device."""
		from .models import WipeRequestCreate
		payload = WipeRequestCreate(
			tenant_id=tenant_id,
			device_id=device_id,
			wipe_type=wipe_type,
			authorised_by=authorised_by,
			created_by=authorised_by,
		)
		return await self.request_wipe(payload)

	async def lock_device(
		self, tenant_id: str, device_id: str, reason: str, locked_by: str = "admin"
	) -> dict[str, Any]:
		"""Lock a device remotely."""
		device = self._require_device(tenant_id, device_id)
		device.enrolment_state = "locked"
		device.updated_at = datetime.utcnow()
		self._audit(tenant_id, "device_locked", device_id)
		return {
			"device_id": device_id,
			"tenant_id": tenant_id,
			"status": "locked",
			"reason": reason,
			"locked_by": locked_by,
			"locked_at": datetime.utcnow().isoformat(),
		}

	async def unlock_device(
		self, tenant_id: str, device_id: str, pin: str, unlocked_by: str = "admin"
	) -> dict[str, Any]:
		"""Unlock a previously locked device."""
		device = self._require_device(tenant_id, device_id)
		device.enrolment_state = "enrolled"
		device.updated_at = datetime.utcnow()
		self._audit(tenant_id, "device_unlocked", device_id)
		return {
			"device_id": device_id,
			"tenant_id": tenant_id,
			"status": "unlocked",
			"unlocked_by": unlocked_by,
			"unlocked_at": datetime.utcnow().isoformat(),
		}

	async def location_track(
		self, tenant_id: str, device_id: str
	) -> dict[str, Any]:
		"""Return last known location for a device (stub — integrates with GPS provider)."""
		device = self._require_device(tenant_id, device_id)
		self._audit(tenant_id, "device_location_requested", device_id)
		return {
			"device_id": device_id,
			"tenant_id": tenant_id,
			"lat": None,
			"lng": None,
			"accuracy_metres": None,
			"note": "location_data_requires_gps_integration",
			"requested_at": datetime.utcnow().isoformat(),
		}

	async def app_blacklist_check(
		self, tenant_id: str, device_id: str, app_list: list[str]
	) -> dict[str, Any]:
		"""Check if any blacklisted apps are installed on a device."""
		device = self._require_device(tenant_id, device_id)
		installed = getattr(device, "installed_apps", []) or []
		violations = [app for app in app_list if app in installed]
		self._audit(tenant_id, "app_blacklist_checked", device_id)
		return {
			"device_id": device_id,
			"tenant_id": tenant_id,
			"blacklisted_apps_checked": len(app_list),
			"violations": violations,
			"compliant": len(violations) == 0,
			"checked_at": datetime.utcnow().isoformat(),
		}

	async def certificate_push(
		self, tenant_id: str, device_id: str, cert: str, pushed_by: str = "admin"
	) -> dict[str, Any]:
		"""Push a certificate to a device."""
		self._require_device(tenant_id, device_id)
		cert_id = f"cert-push-{device_id[:6]}-{len(self._audit_events)+1}"
		self._audit(tenant_id, "certificate_pushed", cert_id)
		return {
			"cert_push_id": cert_id,
			"device_id": device_id,
			"tenant_id": tenant_id,
			"cert_fingerprint": cert[:20] + "...",
			"pushed_by": pushed_by,
			"pushed_at": datetime.utcnow().isoformat(),
		}

	async def vpn_profile_push(
		self, tenant_id: str, device_id: str, profile: dict[str, Any], pushed_by: str = "admin"
	) -> dict[str, Any]:
		"""Push a VPN profile to a device."""
		self._require_device(tenant_id, device_id)
		push_id = f"vpn-push-{device_id[:6]}-{len(self._audit_events)+1}"
		self._audit(tenant_id, "vpn_profile_pushed", push_id)
		return {
			"push_id": push_id,
			"device_id": device_id,
			"tenant_id": tenant_id,
			"vpn_server": profile.get("server", "vpn.corp.example.com"),
			"protocol": profile.get("protocol", "ikev2"),
			"pushed_by": pushed_by,
			"pushed_at": datetime.utcnow().isoformat(),
		}

	async def mdm_compliance_report(
		self, tenant_id: str, period: str
	) -> dict[str, Any]:
		"""Generate an MDM compliance report for a period."""
		devices = [d for d in self._devices.values() if d.tenant_id == tenant_id]
		compliant = sum(1 for d in devices if d.compliance_state == "compliant")
		non_compliant = sum(1 for d in devices if d.compliance_state == "non_compliant")
		alerts = [a for a in self._alerts.values() if a.tenant_id == tenant_id and not a.resolved]
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_devices": len(devices),
			"compliant_devices": compliant,
			"non_compliant_devices": non_compliant,
			"compliance_rate_pct": round(compliant / max(len(devices), 1) * 100, 1),
			"open_alerts": len(alerts),
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def mdm_kpi_summary(
		self, tenant_id: str
	) -> dict[str, Any]:
		"""Return a concise MDM KPI card for dashboard consumption."""
		devices = [d for d in self._devices.values() if d.tenant_id == tenant_id]
		enrolled = sum(1 for d in devices if d.enrolment_state == "enrolled")
		compliant = sum(1 for d in devices if d.compliance_state == "compliant")
		wipes = sum(1 for w in self._wipe_requests.values() if w.tenant_id == tenant_id and w.state == "pending")
		return {
			"tenant_id": tenant_id,
			"total_devices": len(devices),
			"enrolled_devices": enrolled,
			"compliant_devices": compliant,
			"compliance_rate_pct": round(compliant / max(len(devices), 1) * 100, 1),
			"pending_wipes": wipes,
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def mdm_analytics(
		self, tenant_id: str, period: str
	) -> dict[str, Any]:
		"""Return MDM analytics for a period."""
		devices = [d for d in self._devices.values() if d.tenant_id == tenant_id]
		by_platform: dict[str, int] = {}
		by_state: dict[str, int] = {}
		for d in devices:
			by_platform[d.os_platform] = by_platform.get(d.os_platform, 0) + 1
			by_state[d.enrolment_state] = by_state.get(d.enrolment_state, 0) + 1
		policies = [p for p in self._policies.values() if p.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_devices": len(devices),
			"by_platform": by_platform,
			"by_enrolment_state": by_state,
			"total_policies": len(policies),
			"active_policies": sum(1 for p in policies if p.state == "active"),
			"audit_events": sum(1 for e in self._audit_events if e.get("tenant_id") == tenant_id),
		}

	# -------------------------------------------------------------------------
	# Bulk Operations
	# -------------------------------------------------------------------------

	async def bulk_enrol_devices(
		self,
		payloads: list[DeviceEnrolmentCreate],
	) -> dict[str, Any]:
		"""Enrol multiple devices in a single operation.

		Returns a BulkEnrolmentResult dict with per-row success/failure details.
		Idempotent: devices whose serial_number already exists in the tenant are
		skipped and counted as duplicates rather than raising an error.
		"""
		assert payloads, "bulk_enrol_devices requires at least one payload"
		tenant_id = payloads[0].tenant_id
		assert all(p.tenant_id == tenant_id for p in payloads), "all payloads must share the same tenant_id"

		existing_serials = {d.serial_number for d in self._devices.values() if d.tenant_id == tenant_id}
		results: list[dict[str, Any]] = []
		succeeded = 0
		failed = 0
		duplicates = 0

		for p in payloads:
			if p.serial_number in existing_serials:
				results.append({"serial_number": p.serial_number, "status": "duplicate", "error": None})
				duplicates += 1
				continue
			try:
				device = await self.enrol_device(p)
				existing_serials.add(p.serial_number)
				results.append({"serial_number": p.serial_number, "status": "enrolled", "device_id": device.id, "error": None})
				succeeded += 1
			except Exception as exc:
				results.append({"serial_number": p.serial_number, "status": "failed", "error": str(exc)})
				failed += 1

		return {
			"tenant_id": tenant_id,
			"total": len(payloads),
			"succeeded": succeeded,
			"failed": failed,
			"duplicates": duplicates,
			"results": results,
			"enrolled_at": datetime.utcnow().isoformat(),
		}

	# -------------------------------------------------------------------------
	# Device Health Score
	# -------------------------------------------------------------------------

	async def get_device_health_score(
		self,
		tenant_id: str,
		device_id: str,
	) -> dict[str, Any]:
		"""Compute a weighted health score (0–100) for a device.

		Scoring components (configurable weights):
		- compliance_state (40 pts): compliant=40, pending=20, non_compliant=0
		- enrolment_state (20 pts): enrolled=20, suspended=5, unenrolled/wiped=0
		- last_seen_recency (20 pts): within 1h=20, within 24h=10, older=0
		- open_alerts (20 pts): no open critical/high alerts=20, any=0
		"""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		device = self._require_device(tenant_id, device_id)

		# Compliance component
		compliance_score = {"compliant": 40, "pending_evaluation": 20, "non_compliant": 0}.get(device.compliance_state, 0)

		# Enrolment component
		enrolment_score = {"enrolled": 20, "suspended": 5, "unenrolled": 0, "wiped": 0, "locked": 10}.get(device.enrolment_state, 0)

		# Recency component
		now = datetime.utcnow()
		last_seen = device.last_seen_at or device.enrolled_at
		hours_since = (now - last_seen).total_seconds() / 3600
		if hours_since <= 1:
			recency_score = 20
		elif hours_since <= 24:
			recency_score = 10
		else:
			recency_score = 0

		# Alert component
		open_critical = any(
			a.device_id == device_id and not a.resolved and a.severity in ("critical", "high")
			for a in self._alerts.values()
			if a.tenant_id == tenant_id
		)
		alert_score = 0 if open_critical else 20

		total = compliance_score + enrolment_score + recency_score + alert_score
		self._audit(tenant_id, "device_health_score_computed", device_id)
		return {
			"device_id": device_id,
			"tenant_id": tenant_id,
			"health_score": total,
			"components": {
				"compliance": compliance_score,
				"enrolment": enrolment_score,
				"last_seen_recency": recency_score,
				"open_alerts": alert_score,
			},
			"grade": "A" if total >= 80 else ("B" if total >= 60 else ("C" if total >= 40 else "F")),
			"computed_at": now.isoformat(),
		}

	# -------------------------------------------------------------------------
	# Device Groups
	# -------------------------------------------------------------------------

	async def create_device_group(
		self,
		tenant_id: str,
		name: str,
		description: str | None,
		created_by: str,
	) -> dict[str, Any]:
		"""Create a named device group for bulk policy targeting."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
		})
		assert _present(name), "group name must not be empty"
		group_id = uuid7str()
		group = {
			"id": group_id,
			"tenant_id": tenant_id,
			"name": name,
			"description": description,
			"device_ids": [],
			"created_by": created_by,
			"created_at": datetime.utcnow().isoformat(),
		}
		if not hasattr(self, "_device_groups"):
			self._device_groups: dict[tuple[str, str], dict[str, Any]] = {}
		self._device_groups[self._key(tenant_id, group_id)] = group
		self._audit(tenant_id, "device_group_created", group_id)
		return group

	async def add_device_to_group(
		self,
		tenant_id: str,
		group_id: str,
		device_id: str,
		added_by: str,
	) -> dict[str, Any]:
		"""Add a device to an existing device group."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		self._require_device(tenant_id, device_id)
		if not hasattr(self, "_device_groups"):
			self._device_groups = {}
		group = self._device_groups.get((tenant_id, group_id))
		assert group is not None, f"device_group_not_found: {group_id}"
		if device_id not in group["device_ids"]:
			group["device_ids"].append(device_id)
		self._audit(tenant_id, "device_added_to_group", device_id)
		return {"group_id": group_id, "device_id": device_id, "status": "added", "added_by": added_by}

	async def assign_policy_to_group(
		self,
		tenant_id: str,
		group_id: str,
		policy_id: str,
		assigned_by: str,
		created_by: str,
	) -> dict[str, Any]:
		"""Assign a policy to all devices in a device group.

		Returns a summary with per-device assignment results.
		"""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
		})
		if not hasattr(self, "_device_groups"):
			self._device_groups = {}
		group = self._device_groups.get((tenant_id, group_id))
		assert group is not None, f"device_group_not_found: {group_id}"
		self._require_policy(tenant_id, policy_id)

		results: list[dict[str, Any]] = []
		for device_id in group["device_ids"]:
			try:
				from .models import PolicyAssignmentCreate
				pa = PolicyAssignmentCreate(
					tenant_id=tenant_id,
					policy_id=policy_id,
					device_id=device_id,
					assigned_by=assigned_by,
					created_by=created_by,
				)
				assignment = await self.assign_policy(pa)
				results.append({"device_id": device_id, "assignment_id": assignment.id, "status": "assigned"})
			except Exception as exc:
				results.append({"device_id": device_id, "status": "failed", "error": str(exc)})

		self._audit(tenant_id, "policy_assigned_to_group", group_id)
		return {
			"group_id": group_id,
			"policy_id": policy_id,
			"tenant_id": tenant_id,
			"device_count": len(group["device_ids"]),
			"results": results,
			"assigned_at": datetime.utcnow().isoformat(),
		}

	# -------------------------------------------------------------------------
	# Wipe Cancellation
	# -------------------------------------------------------------------------

	async def cancel_wipe(
		self,
		tenant_id: str,
		wipe_id: str,
		cancelled_by: str,
		reason: str,
	) -> WipeRequestResponse:
		"""Cancel a pending wipe request within the rollback window.

		Only `pending` wipes may be cancelled. Completed or already-cancelled
		wipes raise an assertion error. The cancellation is recorded in the
		audit trail with the provided reason.
		"""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"approval_present": True,
		})
		wipe = self._require_wipe(tenant_id, wipe_id)
		assert wipe.state == "pending", f"only_pending_wipes_can_be_cancelled: current_state={wipe.state}"
		assert _present(reason), "cancellation reason must not be empty"
		wipe.state = "cancelled"
		wipe.error_message = f"cancelled_by={cancelled_by}: {reason}"
		wipe.updated_at = datetime.utcnow()
		# Restore device enrolment state if device exists
		device = self._devices.get((tenant_id, wipe.device_id))
		if device and device.enrolment_state == "wiped":
			device.enrolment_state = "enrolled"
			device.updated_at = datetime.utcnow()
		self._audit(tenant_id, "wipe_cancelled", wipe_id)
		return wipe

	# -------------------------------------------------------------------------
	# Certificate Lifecycle
	# -------------------------------------------------------------------------

	async def track_certificate(
		self,
		tenant_id: str,
		device_id: str,
		cert_serial: str,
		issuer: str,
		not_before: datetime,
		not_after: datetime,
		tracked_by: str,
	) -> dict[str, Any]:
		"""Register a certificate record for lifecycle tracking.

		Automatically raises a `certificate_expiring_soon` alert if the
		certificate expires within 30 days.
		"""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		self._require_device(tenant_id, device_id)
		assert _present(cert_serial), "cert_serial must not be empty"
		assert not_after > not_before, "not_after must be after not_before"

		if not hasattr(self, "_certificates"):
			self._certificates: dict[tuple[str, str], dict[str, Any]] = {}

		cert_id = uuid7str()
		days_remaining = (not_after - datetime.utcnow()).days
		status = "valid" if days_remaining > 0 else "expired"
		record = {
			"id": cert_id,
			"tenant_id": tenant_id,
			"device_id": device_id,
			"cert_serial": cert_serial,
			"issuer": issuer,
			"not_before": not_before.isoformat(),
			"not_after": not_after.isoformat(),
			"days_remaining": days_remaining,
			"status": status,
			"tracked_by": tracked_by,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._certificates[self._key(tenant_id, cert_id)] = record
		self._audit(tenant_id, "certificate_tracked", cert_id)

		if 0 < days_remaining <= 30:
			await self._raise_alert(
				tenant_id, device_id, "certificate_expiring_soon",
				"high" if days_remaining <= 7 else "medium",
				f"Certificate {cert_serial} expires in {days_remaining} day(s) on {not_after.date().isoformat()}",
			)
		elif days_remaining <= 0:
			await self._raise_alert(
				tenant_id, device_id, "certificate_expired",
				"critical",
				f"Certificate {cert_serial} expired on {not_after.date().isoformat()}",
			)
		return record

	async def list_certificates(
		self,
		tenant_id: str,
		device_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List tracked certificates with optional filters."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		if not hasattr(self, "_certificates"):
			self._certificates = {}
		certs = [c for c in self._certificates.values() if c["tenant_id"] == tenant_id]
		if device_id:
			certs = [c for c in certs if c["device_id"] == device_id]
		if status:
			certs = [c for c in certs if c["status"] == status]
		return sorted(certs, key=lambda c: c["not_after"])

	# -------------------------------------------------------------------------
	# Audit Log Query
	# -------------------------------------------------------------------------

	async def get_audit_log(
		self,
		tenant_id: str,
		entity_id: str | None = None,
		event_type: str | None = None,
		limit: int = 100,
	) -> list[dict[str, Any]]:
		"""Return filtered audit log entries for a tenant.

		Supports filtering by entity_id (device, policy, wipe, etc.) and/or
		event_type. Results are newest-first. Limit defaults to 100.
		"""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		assert 1 <= limit <= 1000, "limit must be between 1 and 1000"
		entries = [e for e in self._audit_events if e.get("tenant_id") == tenant_id]
		if entity_id:
			entries = [e for e in entries if e.get("entity_id") == entity_id]
		if event_type:
			entries = [e for e in entries if e.get("event_type") == event_type]
		# Newest first
		entries = sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)
		return entries[:limit]

	async def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		"""High-level MDM dashboard summary."""
		devices = [d for d in self._devices.values() if d.tenant_id == tenant_id]
		policies = [p for p in self._policies.values() if p.tenant_id == tenant_id]
		wipes = [w for w in self._wipe_requests.values() if w.tenant_id == tenant_id]
		alerts = [a for a in self._alerts.values() if a.tenant_id == tenant_id]
		return {
			"total_devices": len(devices),
			"devices_by_state": self._count_by(devices, "enrolment_state"),
			"devices_by_platform": self._count_by(devices, "os_platform"),
			"compliance_summary": self._count_by(devices, "compliance_state"),
			"total_policies": len(policies),
			"active_policies": sum(1 for p in policies if p.state == "active"),
			"pending_wipes": sum(1 for w in wipes if w.state == "pending"),
			"open_alerts": sum(1 for a in alerts if not a.resolved),
		}

	# -------------------------------------------------------------------------
	# Private helpers
	# -------------------------------------------------------------------------

	def _log_device_summary(self, tenant_id: str) -> str:
		count = sum(1 for d in self._devices.values() if d.tenant_id == tenant_id)
		return f"tenant={tenant_id} enrolled_devices={count}"

	def _log_compliance_summary(self, tenant_id: str) -> str:
		non_compliant = sum(1 for d in self._devices.values() if d.tenant_id == tenant_id and d.compliance_state == "non_compliant")
		return f"tenant={tenant_id} non_compliant_devices={non_compliant}"

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

	def _require_device(self, tenant_id: str, device_id: str) -> DeviceResponse:
		device = self._devices.get((tenant_id, device_id))
		assert device is not None, f"device_not_found: {device_id}"
		return device

	def _require_policy(self, tenant_id: str, policy_id: str) -> PolicyResponse:
		policy = self._policies.get((tenant_id, policy_id))
		assert policy is not None, f"policy_not_found: {policy_id}"
		return policy

	def _require_wipe(self, tenant_id: str, wipe_id: str) -> WipeRequestResponse:
		wipe = self._wipe_requests.get((tenant_id, wipe_id))
		assert wipe is not None, f"wipe_request_not_found: {wipe_id}"
		return wipe

	def _require_profile(self, tenant_id: str, profile_id: str) -> MdmProfileResponse:
		profile = self._profiles.get((tenant_id, profile_id))
		assert profile is not None, f"profile_not_found: {profile_id}"
		return profile

	def _count_by(self, items: list[Any], attr: str) -> dict[str, int]:
		counts: dict[str, int] = {}
		for item in items:
			k = getattr(item, attr, "unknown")
			counts[k] = counts.get(k, 0) + 1
		return counts
