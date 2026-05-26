"""Registry tests for APG executable capability contracts."""

from __future__ import annotations

from capabilities.capability_contract_registry import (
	discover_contract_paths,
	evaluate_rules,
	get_contract,
	load_contract_registry,
	validate_contract_registry,
	validate_contract_shape,
)


def test_registry_discovers_and_validates_all_capability_contracts():
	paths = discover_contract_paths()
	registry = load_contract_registry()

	assert len(paths) >= 100
	assert len(registry) >= 100
	assert "nlpc" in registry
	assert "composition_events" in registry
	assert "fintech_gateway" in registry

	for record in registry.values():
		validate_contract_shape(record.contract, record.path)


def test_registry_returns_structured_validation_report():
	report = validate_contract_registry()

	assert report["valid"] is True
	assert report["contract_count"] >= 100
	assert report["error_count"] == 0
	assert report["errors"] == []
	assert {"nlpc", "composition_events", "fintech_gateway"} <= set(report["capabilities"])


def test_registry_returns_contract_by_capability_id():
	contract = get_contract("composition_events", tenant_id="tenant-events")

	assert contract["capability"] == "composition_events"
	assert contract["configuration"]["tenant_id"] == "tenant-events"
	assert contract["ui"]["requires_theme"] is True
	assert contract["theme"]["tokens"]["border.radius"] == "8px"


def test_registry_evaluates_capability_rules():
	result = evaluate_rules(
		"composition_events",
		{"tenant_context_present": False, "operation_type": "write", "policy_attached": False},
	)

	assert result["decision"] == "deny"
	assert "tenant_context_required" in result["matched_rules"]
	assert "operation_policy_required" in result["matched_rules"]
