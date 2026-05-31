"""Dependency-light UI model helpers for MFAU."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .mfa_runtime import MfauService


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
