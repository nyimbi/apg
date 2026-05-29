"""UI metadata and dashboard helpers for the Consent and Privacy Management capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import ConsService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: ConsService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ConsService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"purpose_registry": service.list_purposes(tenant_id),
		"privacy_notices": service.list_notices(tenant_id),
		"consent_ledger": service.list_consents(tenant_id),
		"preference_center": service.list_preferences(tenant_id),
		"request_queue": service.list_requests(tenant_id),
		"processing_decisions": service.list_processing_decisions(tenant_id),
		"audit_timeline": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def subject_privacy_model(service: ConsService, tenant_id: str, subject_id: str) -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"subject_id": subject_id,
		"consents": [item for item in service.list_consents(tenant_id) if item["subject_id"] == subject_id],
		"preferences": [item for item in service.list_preferences(tenant_id) if item["subject_id"] == subject_id],
		"requests": [item for item in service.list_requests(tenant_id) if item["subject_id"] == subject_id],
		"processing_decisions": [
			item for item in service.list_processing_decisions(tenant_id) if item["subject_id"] == subject_id
		],
	}
