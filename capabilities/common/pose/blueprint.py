"""Compatibility blueprint metadata for the POSE capability.

The executable generated-application surface is `api.py` and `views.py`.
Production Flask or web-framework integration should adapt those helpers rather
than importing heavyweight model runtimes in this package.
"""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract


def blueprint_manifest(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"name": "pose",
		"display_name": contract["display_name"],
		"api_prefix": contract["ui"]["api_prefix"],
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"]["name"],
	}
