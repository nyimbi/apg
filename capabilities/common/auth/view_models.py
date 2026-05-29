"""Dependency-light AUTH view models for package-composed UIs."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import AuthService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	contract = get_capability_contract(tenant_id)
	return [
		{
			"name": route["name"],
			"path": route["path"],
			"component": route["component"],
			"permission": route["permission"],
			"nav_group": route["nav_group"],
		}
		for route in contract["ui"]["routes"]
	]


def dashboard_model(
	service: AuthService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
	contract = get_capability_contract(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"identities": service.list_identities(tenant_id),
		"roles": service.list_roles(tenant_id),
		"role_approvals": service.list_role_assignment_approvals(tenant_id),
		"role_assignments": service.list_role_assignments(tenant_id),
		"sessions": service.list_sessions(tenant_id),
		"access_decisions": service.list_access_decisions(tenant_id),
		"privacy_queries": service.list_privacy_queries(tenant_id),
		"privacy_approvals": service.list_privacy_budget_approvals(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def role_workbench_model(
	service: AuthService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
	return {
		"roles": service.list_roles(tenant_id),
		"assignments": service.list_role_assignments(tenant_id),
		"approvals": service.list_role_assignment_approvals(tenant_id),
		"required_role_fields": ["id", "name", "permissions", "tier"],
		"required_assignment_fields": ["id", "user_id", "role_id", "assigned_by", "approval_id"],
		"role_tiers": ["standard", "privileged", "admin"],
	}


def approval_queue_model(
	service: AuthService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
	approvals = service.list_role_assignment_approvals(tenant_id)
	return {
		"approvals": approvals,
		"pending_approvals": [approval for approval in approvals if approval["status"] == "pending"],
		"decided_approvals": [approval for approval in approvals if approval["status"] != "pending"],
		"required_decision_fields": ["reviewer", "decision", "notes"],
		"guardrails": ["independent_reviewer", "reviewer_notes_required", "matching_user_and_role_required"],
	}


def session_center_model(
	service: AuthService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
	return {
		"sessions": service.list_sessions(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"required_controls": ["device_binding", "mfa_verified", "risk_level", "step_up_completed"],
	}


def access_decision_model(
	service: AuthService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
	return {
		"decisions": service.list_access_decisions(tenant_id),
		"assignments": service.list_role_assignments(tenant_id),
		"required_evidence": ["session_id", "permission", "mfa_verified", "risk_level"],
	}


def privacy_center_model(
	service: AuthService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
	approvals = service.list_privacy_budget_approvals(tenant_id)
	return {
		"queries": service.list_privacy_queries(tenant_id),
		"approvals": approvals,
		"pending_approvals": [approval for approval in approvals if approval["status"] == "pending"],
		"decided_approvals": [approval for approval in approvals if approval["status"] != "pending"],
		"identities": [
			{"id": identity["id"], "privacy_budget": identity["privacy_budget"]}
			for identity in service.list_identities(tenant_id)
		],
		"required_controls": ["epsilon_cost", "privacy_approval_id_when_budget_exhausted"],
		"required_decision_fields": ["reviewer", "decision", "notes"],
	}


def audit_model(
	service: AuthService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
	contract = get_capability_contract(tenant_id)
	return {
		"summary": service.dashboard_summary(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def _service_or_default(service: AuthService | None) -> AuthService:
	if service is not None:
		return service
	try:
		from .api_helpers import SERVICE

		return SERVICE
	except ImportError:  # pragma: no cover - standalone package loading path
		return AuthService()
