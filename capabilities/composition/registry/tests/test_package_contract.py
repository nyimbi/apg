"""Capability registry package contract tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import types

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "tested_composition_registry"


def _load_module(name: str):
	if PACKAGE_NAME not in sys.modules:
		package = types.ModuleType(PACKAGE_NAME)
		package.__path__ = [str(PACKAGE_DIR)]
		sys.modules[PACKAGE_NAME] = package
	spec = importlib.util.spec_from_file_location(f"{PACKAGE_NAME}.{name}", PACKAGE_DIR / f"{name}.py")
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_streaming_routes_and_agents_are_valid():
	module = _load_module("capability_contract")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "composition_registry"
	assert "registry_agents" in contract["provides"]
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["streaming"]["processor"] == "bytewax"
	assert any(route["path"] == "/composition-registry/catalog" for route in contract["ui"]["routes"])
	assert any(route["path"] == "/composition-registry/agents" for route in contract["ui"]["routes"])


def test_rule_engine_blocks_missing_context_and_non_bytewax_import():
	module = _load_module("capability_contract")

	missing_context = module.evaluate_capability_rules({"tenant_context_present": False})
	bad_stream = module.evaluate_capability_rules({"tenant_context_present": True, "operation": "registry_import", "event_stream": "other"})

	assert missing_context["decision"] == "deny"
	assert "tenant_context_required" in missing_context["matched_rules"]
	assert bad_stream["decision"] == "deny"
	assert "registry_import_requires_bytewax" in bad_stream["matched_rules"]


def test_capability_dependency_composition_version_and_marketplace_lifecycle():
	service_module = _load_module("service")
	service = service_module.CompositionRegistryService()

	auth = service.register_capability("auth", "tenant-test", "Auth", "platform", "common", "1.0.0", ["authn"], "auth/contract.py")
	audit = service.register_capability("audl", "tenant-test", "Audit", "platform", "common", "1.0.0", ["audit"], "audl/contract.py")
	dependency = service.add_dependency("auth-audl", "tenant-test", "auth", "audl", "required", ">=1.0.0")
	composition = service.create_composition("secure-app", "tenant-test", "Secure App", "platform", ["auth", "audl"])
	published = service.publish_composition("tenant-test", composition["id"], "validated")
	version = service.release_version("auth-1", "tenant-test", "auth", "1.1.0", "compatible", reviewed_by="release")
	publication = service.publish_to_marketplace("auth-pub", "tenant-test", "auth", "README.md", "marketplace")

	assert auth["status"] == "registered"
	assert audit["status"] == "registered"
	assert dependency["status"] == "validated"
	assert composition["validation"]["valid"] is True
	assert published["status"] == "published"
	assert version["status"] == "released"
	assert publication["status"] == "prepared"
	assert service.dashboard_summary("tenant-test")["audit_event_count"] >= 7


def test_service_enforces_registry_guardrails():
	service_module = _load_module("service")
	service = service_module.CompositionRegistryService()

	try:
		service.register_capability("bad", "", "Bad", "owner", "common", "1.0.0", ["surface"], "contract.py")
	except PermissionError as exc:
		assert "tenant_context_required" in str(exc)
	else:
		raise AssertionError("expected tenant guardrail")

	try:
		service.register_capability("bad", "tenant-test", "Bad", "", "common", "1.0.0", ["surface"], "contract.py")
	except PermissionError as exc:
		assert "capability_requires_owner" in str(exc)
	else:
		raise AssertionError("expected owner guardrail")

	service.register_capability("auth", "tenant-test", "Auth", "owner", "common", "1.0.0", ["authn"], "auth/contract.py")
	try:
		service.release_version("auth-1", "tenant-test", "auth", "1.1.0", "")
	except PermissionError as exc:
		assert "version_release_requires_compatibility" in str(exc)
	else:
		raise AssertionError("expected compatibility guardrail")

	try:
		service.publish_to_marketplace("auth-pub", "tenant-test", "auth", "", "reviewer")
	except PermissionError as exc:
		assert "marketplace_publish_requires_documentation" in str(exc)
	else:
		raise AssertionError("expected documentation guardrail")


def test_agents_batch_api_views_and_app_are_executable():
	service_module = _load_module("service")
	api_module = _load_module("api")
	views_module = _load_module("views")
	app_module = _load_module("app")

	service = service_module.CompositionRegistryService()
	agent = service.register_registry_agent("tenant-test", "Dependency Review", "codex", "dependency_reviewer", "Review dependency graph.")
	agent_result = service.validate_agent_registry_action("tenant-test", agent["id"], "review_dependency", True, True)
	batch = service.validate_import_batch("tenant-test", 4)
	dashboard = views_module.dashboard_model(service, "tenant-test")
	agent_view = views_module.agent_workbench_model(service, "tenant-test")
	api_record = api_module.create_record({"id": "api-capability", "tenant_id": "tenant-api"})
	status = api_module.capability_status("tenant-api")
	self_test = app_module.self_test()
	model = app_module.semantic_model()

	assert agent_result["decision"] == "allow"
	assert batch["processor"] == "bytewax"
	assert dashboard["summary"]["registry_agent_count"] == 1
	assert agent_view["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert api_record["id"].startswith("registered_capability_")
	assert status["streaming"]["processor"] == "bytewax"
	assert self_test["passed"] is True
	assert model["capabilities"]["composition_registry"]["streaming"]["processor"] == "bytewax"
