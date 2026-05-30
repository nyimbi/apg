"""APG API service-mesh capability package."""

from __future__ import annotations

from .capability_contract import (
	GATEWAY_EVENT_STREAM,
	SUPPORTED_GATEWAY_AGENT_ROLES,
	SUPPORTED_GATEWAY_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
	streaming_manifest,
)
from .service import ASMService, CompositionGatewayService


__version__ = "2.1.0"
__capability_id__ = "composition_gateway"
__apg_dependencies__ = ["auth", "audl", "ntfy", "registry", "composition_access", "composition_events"]
__apg_optional_dependencies__ = ["i18n", "mchn", "secrets"]

CAPABILITY_INFO = {
	"capability_code": "ASM",
	"capability_name": "API Service Mesh",
	"category": "composition_orchestration",
	"subcategory": "service_mesh",
	"version": __version__,
	"description": "API service mesh, gateway routing, policy, traffic, certificate, and health lifecycle.",
	"multi_tenant": True,
	"audit_enabled": True,
}


def register_capability() -> dict[str, object]:
	"""Return package metadata used by APG capability discovery."""
	contract = get_capability_contract()
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"version": __version__,
		"provides": contract["provides"],
		"requires": contract["requires"],
		"ui": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}


__all__ = [
	"ASMService",
	"CAPABILITY_INFO",
	"CompositionGatewayService",
	"GATEWAY_EVENT_STREAM",
	"SUPPORTED_GATEWAY_AGENT_ROLES",
	"SUPPORTED_GATEWAY_AGENT_RUNTIMES",
	"evaluate_capability_rules",
	"event_stream_name",
	"get_capability_contract",
	"register_capability",
	"streaming_manifest",
]
