"""Dependency-light UI model helpers for MFAU."""

from __future__ import annotations

import os as _ctx_os
import base64 as _ctx_b64
import binascii as _ctx_binascii
import json as _ctx_json
from typing import Any, Any as _Any, Dict as _Dict, List as _List, Optional as _Optional

from .capability_contract import get_capability_contract
from .mfa_runtime import MfauService


# ---------------------------------------------------------------------------
# Context resolution helpers (Flask)
# ---------------------------------------------------------------------------


def _clean_text(value: _Any) -> _Optional[str]:
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _object_value(source: _Any, name: str) -> _Any:
	if source is None:
		return None
	if isinstance(source, dict):
		return source.get(name)
	return getattr(source, name, None)


def _mapping_value(source: _Any, name: str) -> _Any:
	if source is None:
		return None
	getter = getattr(source, "get", None)
	return getter(name) if getter else None


def _resolve_current_user_id() -> str:
	"""Resolve user id from Flask context: g > session > request attrs > headers > query > env."""
	try:
		from flask import g as _g, session as _session, request as _req
		# 1. g.current_user
		cu = getattr(_g, "current_user", None)
		if isinstance(cu, dict) and cu.get("user_id"):
			return str(cu["user_id"])
		if isinstance(cu, str) and cu:
			return cu
		# 2. request.current_user attribute
		rcu = getattr(_req, "current_user", None)
		if isinstance(rcu, dict) and rcu.get("user_id"):
			return str(rcu["user_id"])
		# 3. session
		su = _session.get("user_id")
		if su:
			return str(su)
		# 4. Headers
		hu = _req.headers.get("X-APG-User-ID") or _req.headers.get("X-User-ID")
		if hu:
			return str(hu)
		# 5. Query params
		qu = _req.args.get("user_id")
		if qu:
			return str(qu)
	except Exception:
		pass
	# 6. Env fallback
	return _ctx_os.getenv("APG_DEFAULT_USER_ID", "anonymous")


def _resolve_current_tenant_id() -> str:
	"""Resolve tenant id from Flask context: g > session > request attrs > headers > query > env."""
	try:
		from flask import g as _g, session as _session, request as _req
		# 1. g.current_user
		cu = getattr(_g, "current_user", None)
		if isinstance(cu, dict) and cu.get("tenant_id"):
			return str(cu["tenant_id"])
		# 2. request.current_user attribute
		rcu = getattr(_req, "current_user", None)
		if isinstance(rcu, dict) and rcu.get("tenant_id"):
			return str(rcu["tenant_id"])
		# 3. session
		st = _session.get("tenant_id")
		if st:
			return str(st)
		# 4. Headers
		ht = _req.headers.get("X-APG-Tenant-ID") or _req.headers.get("X-Tenant-ID")
		if ht:
			return str(ht)
		# 5. Query params
		qt = _req.args.get("tenant_id") or _req.args.get("tenant")
		if qt:
			return str(qt)
	except Exception:
		pass
	# 6. Env fallback
	return _ctx_os.getenv("APG_DEFAULT_TENANT_ID", "default")


class MFAUserProfileView:
	"""Flask-AppBuilder view for MFA user profile management."""


def route_manifest(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"capability": "mfau",
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"api_prefix": contract["ui"]["api_prefix"],
	}


def dashboard_model(service: MfauService, tenant_id: str) -> dict[str, Any]:
	return {
		"component": "MFAUDashboard",
		"summary": service.dashboard_summary(tenant_id),
		"recent_audit_events": service.list_audit_events(tenant_id)[-10:],
		"mfa_agents": service.list_mfa_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"streaming": service.contract["streaming"],
		"theme_component": "factor_stack",
	}


def profile_registry_model(service: MfauService, tenant_id: str) -> dict[str, Any]:
	return {
		"component": "MFAProfileRegistry",
		"profiles": service.list_profiles(tenant_id),
		"columns": ["id", "status", "user_id", "policy_id", "primary_channel"],
		"theme_component": "profile_card",
	}


def method_registry_model(service: MfauService, tenant_id: str) -> dict[str, Any]:
	return {
		"component": "MFAMethods",
		"methods": service.list_methods(tenant_id),
		"method_types": service.configuration["methods"]["enabled"],
		"phishing_resistant": service.configuration["methods"]["phishing_resistant"],
		"theme_component": "method_card",
	}


def enrollment_wizard_model(service: MfauService, tenant_id: str) -> dict[str, Any]:
	return {
		"component": "MFAEnrollmentWizard",
		"steps": ["select_user", "choose_method", "verify_channel", "bind_device", "confirm"],
		"methods": service.configuration["methods"]["enabled"],
		"profiles": service.list_profiles(tenant_id),
		"theme_component": "enrollment_wizard",
	}


def challenge_console_model(service: MfauService, tenant_id: str) -> dict[str, Any]:
	return {
		"component": "MFAChallengeConsole",
		"challenges": service.list_challenges(tenant_id),
		"risk_thresholds": service.configuration["risk"],
		"theme_component": "challenge_panel",
	}


def risk_console_model(service: MfauService, tenant_id: str) -> dict[str, Any]:
	return {
		"component": "MFARiskConsole",
		"assessments": service.list_risk_assessments(tenant_id),
		"high_risk_threshold": service.configuration["risk"]["high_risk_threshold"],
		"critical_risk_threshold": service.configuration["risk"]["critical_risk_threshold"],
		"theme_component": "risk_meter",
	}


def device_trust_model(service: MfauService, tenant_id: str) -> dict[str, Any]:
	return {
		"component": "MFADeviceTrust",
		"devices": service.list_devices(tenant_id),
		"low_trust_threshold": service.configuration["risk"]["low_trust_device_threshold"],
		"theme_component": "device_trust",
	}


def recovery_center_model(service: MfauService, tenant_id: str) -> dict[str, Any]:
	return {
		"component": "MFARecoveryCenter",
		"recoveries": service.list_recoveries(tenant_id),
		"requires_verified_channel": service.configuration["recovery"]["verified_channel_required"],
		"theme_component": "recovery_timeline",
	}


def backup_code_model(service: MfauService, tenant_id: str) -> dict[str, Any]:
	return {
		"component": "BackupCodeManager",
		"code_sets": service.list_backup_code_sets(tenant_id),
		"default_count": service.configuration["backup_codes"]["default_count"],
		"theme_component": "backup_code_panel",
	}


def policy_studio_model(service: MfauService, tenant_id: str) -> dict[str, Any]:
	return {
		"component": "MFAPolicyStudio",
		"policies": service.list_policies(tenant_id),
		"settings": service.configuration["policies"],
		"theme_component": "policy_editor",
	}


def biometric_consent_model(service: MfauService, tenant_id: str) -> dict[str, Any]:
	return {
		"component": "MFABiometricConsent",
		"biometric_methods": [method for method in service.list_methods(tenant_id) if method["metadata"]["method_type"] == "biometric"],
		"requirements": service.configuration["biometrics"],
		"theme_component": "biometric_consent",
	}


def governance_model(service: MfauService, tenant_id: str) -> dict[str, Any]:
	return {
		"component": "MFAUGovernance",
		"configuration": service.configuration["governance"],
		"adapters": service.configuration["adapters"],
		"agents": service.contract["agents"],
		"streaming": service.contract["streaming"],
		"mfa_agents": service.list_mfa_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"audit_event_count": len(service.list_audit_events(tenant_id)),
	}


def mfa_agent_roster_model(service: MfauService, tenant_id: str) -> dict[str, Any]:
	agents = service.list_mfa_agents(tenant_id)
	return {
		"component": "MFASecurityAgentRoster",
		"agents": agents,
		"pending_review": [item for item in agents if item["status"] == "pending_review"],
		"supported_runtimes": service.contract["agents"]["supported_runtimes"],
		"supported_roles": service.contract["agents"]["supported_roles"],
		"privileged_roles": service.contract["agents"]["privileged_roles"],
		"theme_component": "mfa_agent_roster",
	}


def lifecycle_batch_model(service: MfauService, tenant_id: str) -> dict[str, Any]:
	batches = service.list_lifecycle_batches(tenant_id)
	return {
		"component": "MFAULifecycleBatchMonitor",
		"batches": batches,
		"denied": [item for item in batches if item["status"] == "denied"],
		"required_processor": service.contract["streaming"]["required_processor"],
		"required_operations": service.contract["streaming"]["required_operations"],
		"topics": service.contract["streaming"]["topics"],
		"theme_component": "bytewax_lifecycle_panel",
	}


def audit_timeline_model(service: MfauService, tenant_id: str) -> dict[str, Any]:
	return {
		"component": "MFAAuditTrail",
		"events": service.list_audit_events(tenant_id),
		"theme_component": "audit_timeline",
	}


def settings_model(service: MfauService, tenant_id: str) -> dict[str, Any]:
	return {
		"component": "MFAUSettings",
		"tenant_id": tenant_id,
		"configuration": service.configuration,
		"agents": service.contract["agents"],
		"streaming": service.contract["streaming"],
		"route_manifest": route_manifest(tenant_id),
	}
