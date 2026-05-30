"""APG capability registry package."""

from __future__ import annotations

from .capability_contract import (
	REGISTRY_EVENT_STREAM,
	SUPPORTED_REGISTRY_AGENT_ROLES,
	SUPPORTED_REGISTRY_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
	streaming_manifest,
)
from .service import CRService, CompositionRegistryService, get_registry_service


CAPABILITY_ID = "composition_registry"
CAPABILITY_NAME = "Capability Registry"
CAPABILITY_VERSION = "2.1.0"


def register_capability() -> dict[str, object]:
	contract = get_capability_contract()
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": contract["provides"],
		"requires": contract["requires"],
		"ui": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}


__all__ = [
	"CAPABILITY_ID",
	"CAPABILITY_NAME",
	"CAPABILITY_VERSION",
	"CRService",
	"CompositionRegistryService",
	"REGISTRY_EVENT_STREAM",
	"SUPPORTED_REGISTRY_AGENT_ROLES",
	"SUPPORTED_REGISTRY_AGENT_RUNTIMES",
	"evaluate_capability_rules",
	"event_stream_name",
	"get_capability_contract",
	"get_registry_service",
	"register_capability",
	"streaming_manifest",
]
