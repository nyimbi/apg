"""APG composition access-control capability package."""

from __future__ import annotations

from .capability_contract import (
	ACCESS_EVENT_STREAM,
	SUPPORTED_ACCESS_AGENT_ROLES,
	SUPPORTED_ACCESS_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
	streaming_manifest,
)
from .models import (
	AccessAgentRecord,
	AccessAuditEventRecord,
	AccessDecisionRecord,
	AccessGrantRecord,
	AccessPolicyRecord,
	AccessProviderRecord,
	AccessResourceRecord,
	AccessSessionRecord,
)
from .service import CompositionAccessService


__version__ = "2.1.0"
__capability_id__ = "composition_access"
__apg_dependencies__ = ["auth", "audl", "ntfy", "conf", "registry"]
__apg_optional_dependencies__ = ["i18n", "mchn", "biom"]


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
	"ACCESS_EVENT_STREAM",
	"SUPPORTED_ACCESS_AGENT_ROLES",
	"SUPPORTED_ACCESS_AGENT_RUNTIMES",
	"AccessAgentRecord",
	"AccessAuditEventRecord",
	"AccessDecisionRecord",
	"AccessGrantRecord",
	"AccessPolicyRecord",
	"AccessProviderRecord",
	"AccessResourceRecord",
	"AccessSessionRecord",
	"CompositionAccessService",
	"evaluate_capability_rules",
	"event_stream_name",
	"get_capability_contract",
	"register_capability",
	"streaming_manifest",
]
