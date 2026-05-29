"""UI metadata helpers for the User Management capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import UsrmService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: UsrmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or UsrmService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def user_directory_model(
	service: UsrmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or UsrmService()
	return {
		"route": "/usrm/users",
		"tenant_id": tenant_id,
		"users": service.list_users(tenant_id),
		"statuses": ["active", "invited", "suspended", "review_required", "deprovisioned"],
	}


def profile_manager_model(
	service: UsrmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or UsrmService()
	return {
		"route": "/usrm/profiles",
		"tenant_id": tenant_id,
		"profiles": service.list_profiles(tenant_id),
		"privacy_sync_required": True,
	}


def lifecycle_queue_model(
	service: UsrmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or UsrmService()
	return {
		"route": "/usrm/lifecycle",
		"tenant_id": tenant_id,
		"invitations": service.list_invitations(tenant_id),
		"deprovisions": service.list_deprovisions(tenant_id),
		"bulk_actions": service.list_bulk_actions(tenant_id),
	}


def access_review_model(
	service: UsrmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or UsrmService()
	return {
		"route": "/usrm/access",
		"tenant_id": tenant_id,
		"role_assignments": service.list_role_assignments(tenant_id),
		"access_reviews": service.list_access_reviews(tenant_id),
		"mfa_required_for_privileged": True,
	}


def privacy_preferences_model(
	service: UsrmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or UsrmService()
	return {
		"route": "/usrm/privacy",
		"tenant_id": tenant_id,
		"profiles": service.list_profiles(tenant_id),
		"consent_notice_required": True,
	}


def deprovisioning_model(
	service: UsrmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or UsrmService()
	return {
		"route": "/usrm/deprovisioning",
		"tenant_id": tenant_id,
		"deprovisions": service.list_deprovisions(tenant_id),
		"access_revocation_required": True,
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"route": "/usrm/settings",
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"theme": contract["theme"],
	}
