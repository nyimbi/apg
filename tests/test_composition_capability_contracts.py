"""Generated-app capability contract coverage."""

from __future__ import annotations


def _exec_generated_module(source: str) -> dict[str, object]:
	namespace: dict[str, object] = {}
	exec(compile(source, "generated_capability_contracts.py", "exec"), namespace)
	return namespace


def test_generated_application_includes_executable_capability_contracts(engine, context):
	files = engine.generate_application_files(context)

	assert "capability_contracts.py" in files
	namespace = _exec_generated_module(files["capability_contracts.py"])

	validation = namespace["validate_capability_contracts"]()
	assert validation["errors"] == []

	contracts = namespace["list_capability_contracts"](tenant_id="tenant-alpha")
	assert contracts
	assert {"configuration", "configuration_schema", "rule_engine", "ui", "theme"}.issubset(
		next(iter(contracts.values()))
	)

	for capability_id, contract in contracts.items():
		assert contract["capability"] == capability_id
		assert contract["configuration"]["tenant_id"] == "tenant-alpha"
		assert contract["rule_engine"]["type"] == "deterministic"
		assert contract["rule_engine"]["rules"]
		assert contract["ui"]["requires_theme"] is True
		assert contract["ui"]["routes"]
		assert contract["theme"]["tokens"]


def test_generated_capability_contract_rules_are_executable(engine, context):
	files = engine.generate_application_files(context)
	namespace = _exec_generated_module(files["capability_contracts.py"])
	capability_id = next(iter(namespace["CAPABILITY_CONTRACTS"]))

	denied = namespace["evaluate_capability_rules"](
		capability_id,
		{"tenant_context_present": False},
	)
	assert denied["decision"] == "deny"
	assert "tenant_context_required" in denied["matched_rules"]

	review = namespace["evaluate_capability_rules"](
		capability_id,
		{
			"tenant_context_present": True,
			"operation_type": "read",
			"policy_attached": True,
			"risk_level": "high",
			"review_recorded": False,
		},
	)
	assert review["decision"] == "require_review"
	assert "high_risk_requires_review" in review["matched_rules"]


def test_generated_capability_registry_uses_actual_selected_metadata(engine, context):
	files = engine.generate_application_files(context)
	namespace = _exec_generated_module(files["capability_registry.py"])

	capability_ids = namespace["list_capabilities"]()
	assert capability_ids
	assert all("/" in capability_id for capability_id in capability_ids)

	auth_capabilities = namespace["get_capabilities_by_category"]("auth")
	assert auth_capabilities
	assert all(capability.category == "auth" for capability in auth_capabilities)
	assert all(capability.version for capability in auth_capabilities)
