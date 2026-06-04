"""APG Monitoring and Observability capability.

Standalone package: ``pip install apg-common-moni``

Quick start::

	from apg_common_moni import get_capability_contract, evaluate_capability_rules

	contract = get_capability_contract(tenant_id="my_org")
	result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : moni
Provides      : observability_governance, metrics_lifecycle, monitoring_agent_composition, review_evidence
"""
from __future__ import annotations

__version__ = "1.0.0"
__package_name__ = "apg-common-moni"
__capability_id__ = "moni"

from .capability_contract import (  # noqa: E402
	get_capability_contract,
	evaluate_capability_rules,
)


def register_capability() -> dict:
	"""Return the full MONI capability registration for the APG registry."""
	from .capability_contract import (
		agent_manifest,
		streaming_manifest,
		ui_manifest,
		CapabilityTheme,
		default_rules,
	)
	contract = get_capability_contract("default")
	theme = CapabilityTheme()
	return {
		**contract,
		"ui_manifest": ui_manifest(),
		"ui_components": {
			route["name"]: route["path"]
			for route in ui_manifest()["routes"]
		},
		"dependencies": contract["requires"],
	}


__all__ = [
	"__version__",
	"__capability_id__",
	"get_capability_contract",
	"evaluate_capability_rules",
	"register_capability",
]
