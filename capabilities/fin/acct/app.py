"""Bank Account Management — capability entrypoint."""
from __future__ import annotations
from typing import Any


CAPABILITY_ID = "fin_acct"
DOMAIN = "fin"
VERSION = "1.0.0"


def semantic_model() -> dict[str, Any]:
	return {
		"format": "apg.semantic-model.v1",
		"capability_id": CAPABILITY_ID,
		"domain": DOMAIN,
		"version": VERSION,
	}


def component_manifest() -> dict[str, Any]:
	return {
		"format": "apg.component-manifest.v1",
		"capability_id": CAPABILITY_ID,
		"version": VERSION,
	}


def self_test() -> dict[str, Any]:
	return {"passed": True, "checks": {}, "routes": []}


def main() -> None:
	pass
