"""Executable capability contract for the NOT capability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from capabilities.capability_contract_factory import build_spec_capability_contract, evaluate_contract_rules


CAPABILITY_PATH = Path(__file__).resolve().parent


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	return build_spec_capability_contract(CAPABILITY_PATH, tenant_id, overrides)


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	return evaluate_contract_rules(get_capability_contract()["rule_engine"]["rules"], context)
