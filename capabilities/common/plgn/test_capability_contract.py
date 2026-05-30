"""Regression coverage for the PLGN executable capability contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from capabilities.common.plgn import register_capability
from capabilities.common.plgn.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.plgn.service import PlgnService
from capabilities.common.plgn import views


PACKAGE_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_exposes_configuration_rules_ui_theme_and_streaming():
	contract = get_capability_contract("tenant-plgn", {"marketplace": {"tenant_install_policy_enabled": False}})

	assert contract["capability"] == "plgn"
	assert contract["configuration"]["tenant_id"] == "tenant-plgn"
	assert contract["configuration"]["marketplace"]["tenant_install_policy_enabled"] is False
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"marketplace",
		"plugins",
		"security",
		"plgn_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert contract["provides"] == [
		"plugin_registry",
		"extension_marketplace",
		"permission_review",
		"sandbox_policy",
		"plugin_release_lifecycle",
		"plgn_agents",
	]
	assert contract["requires"] == ["auth", "secu", "conf", "audl"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["configuration"]["plgn_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "marketplace", "plugins", "manifests", "permissions", "sandbox", "releases", "agents", "audit", "settings"}
	assert contract["theme"]["name"] == "plgn_extension_marketplace"


def test_rule_engine_enforces_plgn_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "register_plugin",
		"plugin_owner_assigned": False,
		"signature_verified": False,
		"manifest_schema_valid": False,
		"dependency_validation_passed": False,
		"supply_chain_scan_passed": False,
		"permissions_requested": True,
		"permission_review_recorded": False,
		"external_plugin": True,
		"external_review_recorded": False,
	})
	release_result = evaluate_capability_rules({"operation": "create_release", "signature_ref_present": False, "event_stream": "other-stream"})
	listing_result = evaluate_capability_rules({"operation": "publish_listing", "publisher_verified": False, "curated_listing": False})
	agent_result = evaluate_capability_rules({"plgn_agent_present": True, "agent_runtime_supported": False})
	batch_result = evaluate_capability_rules({"requested_operation": "batch_plugin_mutation", "event_stream": "other-stream"})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"plugin_requires_owner",
		"plugin_requires_signature",
		"plugin_requires_manifest_schema",
		"plugin_requires_dependency_validation",
		"plugin_requires_supply_chain_scan",
		"permissions_require_review",
		"external_plugin_requires_review",
	}
	assert set(release_result["matched_rules"]) == {"release_requires_signature_reference", "release_requires_bytewax_stream"}
	assert set(listing_result["matched_rules"]) == {"marketplace_requires_verified_publisher", "marketplace_requires_curated_listing"}
	assert agent_result["matched_rules"] == ["plgn_agent_runtime_supported"]
	assert batch_result["matched_rules"] == ["batch_plugin_mutation_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "plgn"
	assert "secu" in registration["dependencies"]
	assert registration["ui_components"]["marketplace"] == "/plgn/marketplace"
	assert registration["ui_components"]["agents"] == "/plgn/agents"
	assert registration["streaming"]["processor"] == "bytewax"
	assert "plgn:install" in registration["permissions"]


def test_service_runs_plugin_marketplace_release_and_enable_lifecycle():
	service = PlgnService()

	plugin = service.register_plugin(
		"plugin-risk",
		"tenant-plgn",
		"Risk scorer extension",
		"extension-owner",
		"1.2.0",
		"Datacraft",
		release_channel="stable",
		permissions=["identity", "network:external"],
		dependencies=["auth", "secu"],
		external_plugin=True,
		signature_verified=True,
		manifest_schema_valid=True,
		dependency_validation_passed=True,
		supply_chain_scan_passed=True,
		external_review_recorded=True,
		permission_review_recorded=True,
	)
	review = service.review_permissions(
		"perm-review",
		"tenant-plgn",
		"plugin-risk",
		"security-reviewer",
		approved_scopes=["identity", "network:external"],
		secret_access_allowed=True,
	)
	policy = service.attach_sandbox_policy(
		"sandbox-risk",
		"tenant-plgn",
		"plugin-risk",
		"restricted-tools",
		network_access="egress_allowlisted",
		filesystem_access="read_only",
		secret_access="deny",
		tool_allowlist=["score_customer"],
	)
	listing = service.publish_listing(
		"listing-risk",
		"tenant-plgn",
		"plugin-risk",
		"Risk scorer",
		publisher_verified=True,
		curated=True,
		install_policy="tenant_allowed",
	)
	release = service.create_release("release-risk", "tenant-plgn", "plugin-risk", "1.2.0", "stable", "sig-risk", event_stream="bytewax://plugins")
	installation = service.install_plugin("install-risk", "tenant-plgn", "plugin-risk", "tenant-admin")
	enabled = service.enable_plugin("install-risk", "tenant-plgn", "tenant-admin")
	agent = service.register_plgn_agent(
		tenant_id="tenant-plgn",
		name="Manifest reviewer",
		runtime="codex",
		role="manifest_reviewer",
		scope="review manifest, dependency, permission, sandbox, and release gates",
	)
	summary = service.dashboard_summary("tenant-plgn")
	dashboard = views.dashboard_model(service, "tenant-plgn")
	marketplace = views.marketplace_model(service, "tenant-plgn")
	releases = views.release_manager_model(service, "tenant-plgn")

	assert plugin["status"] == "registered"
	assert plugin["external_plugin"] is True
	assert review["approved_scopes"] == ["identity", "network:external"]
	assert policy["tool_allowlist"] == ["score_customer"]
	assert listing["status"] == "listed"
	assert release["status"] == "released"
	assert installation["status"] == "installed"
	assert enabled["status"] == "enabled"
	assert agent["runtime"] == "codex"
	assert agent["role"] == "manifest_reviewer"
	assert summary["enabled_plugin_count"] == 1
	assert summary["permission_review_count"] == 1
	assert summary["plgn_agent_count"] == 1
	assert service.validate_batch_plugin_mutation("bytewax")["decision"] == "allow"
	assert service.validate_batch_plugin_mutation("other-stream")["decision"] == "deny"
	assert dashboard["summary"]["plugin_count"] == 1
	assert dashboard["streaming"]["processor"] == "bytewax"
	assert marketplace["listings"][0]["id"] == "listing-risk"
	assert releases["releases"][0]["id"] == "release-risk"
	assert views.plgn_agent_model(service, "tenant-plgn")["plgn_agents"][0]["role"] == "manifest_reviewer"
	assert views.audit_trail_model(service, "tenant-plgn")["audit_events"]
	assert views.plugin_policy_model(service, "tenant-plgn")["streaming"]["processor"] == "bytewax"
	assert len(service.list_audit_events("tenant-plgn")) >= 8


def test_service_enforces_plugin_governance_guardrails():
	service = PlgnService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_plugin("plugin-no-tenant", "", "No tenant", "owner", "1.0.0", "publisher")

	with pytest.raises(PermissionError, match="plugin_owner_required"):
		service.register_plugin("plugin-no-owner", "tenant-plgn", "No owner", "", "1.0.0", "publisher")

	with pytest.raises(PermissionError, match="plugin_signature_required"):
		service.register_plugin("plugin-unsigned", "tenant-plgn", "Unsigned", "owner", "1.0.0", "publisher", signature_verified=False)

	with pytest.raises(PermissionError, match="permission_review_required"):
		service.register_plugin(
			"plugin-needs-review",
			"tenant-plgn",
			"Needs review",
			"owner",
			"1.0.0",
			"publisher",
			permissions=["identity"],
			permission_review_recorded=False,
		)

	with pytest.raises(PermissionError, match="external_plugin_review_required"):
		service.register_plugin(
			"plugin-external",
			"tenant-plgn",
			"External",
			"owner",
			"1.0.0",
			"publisher",
			external_plugin=True,
			external_review_recorded=False,
		)

	with pytest.raises(PermissionError, match="manifest_schema_required"):
		service.register_plugin("plugin-bad-manifest", "tenant-plgn", "Bad manifest", "owner", "1.0.0", "publisher", manifest_schema_valid=False)

	with pytest.raises(PermissionError, match="dependency_validation_required"):
		service.register_plugin("plugin-bad-deps", "tenant-plgn", "Bad dependencies", "owner", "1.0.0", "publisher", dependency_validation_passed=False)

	with pytest.raises(PermissionError, match="supply_chain_scan_required"):
		service.register_plugin("plugin-bad-scan", "tenant-plgn", "Bad scan", "owner", "1.0.0", "publisher", supply_chain_scan_passed=False)

	service.register_plugin(
		"plugin-risk",
		"tenant-plgn",
		"Risk scorer",
		"owner",
		"1.0.0",
		"publisher",
		permissions=["identity", "network:external"],
		permission_review_recorded=True,
	)

	with pytest.raises(PermissionError, match="all_requested_permissions_must_be_reviewed"):
		service.review_permissions("review-incomplete", "tenant-plgn", "plugin-risk", "reviewer", approved_scopes=["identity"])

	with pytest.raises(PermissionError, match="sensitive_permission_secret_policy_required"):
		service.review_permissions("review-sensitive", "tenant-plgn", "plugin-risk", "reviewer", approved_scopes=["identity", "network:external"])

	service.review_permissions(
		"review-ok",
		"tenant-plgn",
		"plugin-risk",
		"reviewer",
		approved_scopes=["identity", "network:external"],
		secret_access_allowed=True,
	)

	with pytest.raises(PermissionError, match="curated_listing_required"):
		service.publish_listing("listing-uncurated", "tenant-plgn", "plugin-risk", "Uncurated", curated=False)

	with pytest.raises(PermissionError, match="plugin_sandbox_required"):
		service.create_release("release-no-sandbox", "tenant-plgn", "plugin-risk", "1.0.0", "stable", "sig")

	service.attach_sandbox_policy("sandbox-risk", "tenant-plgn", "plugin-risk", "restricted")
	service.publish_listing("listing-risk", "tenant-plgn", "plugin-risk", "Risk scorer", install_policy="admin_only")

	with pytest.raises(PermissionError, match="release_signature_required"):
		service.create_release("release-no-signature", "tenant-plgn", "plugin-risk", "1.0.0", "stable", "")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.create_release("release-other-stream", "tenant-plgn", "plugin-risk", "1.0.0", "stable", "sig", event_stream="other-stream")

	service.create_release("release-risk", "tenant-plgn", "plugin-risk", "1.0.0", "stable", "sig")

	with pytest.raises(PermissionError, match="admin_install_required"):
		service.install_plugin("install-risk", "tenant-plgn", "plugin-risk", "analyst")

	with pytest.raises(PermissionError, match="plgn_agent_runtime_not_supported"):
		service.register_plgn_agent("tenant-plgn", "Unsupported", "unsupported", "manifest_reviewer", "review")
	with pytest.raises(PermissionError, match="plgn_agent_scope_required"):
		service.register_plgn_agent("tenant-plgn", "No scope", "codex", "manifest_reviewer", "")


def test_enable_requires_sandbox_policy():
	service = PlgnService()
	service.register_plugin("plugin-basic", "tenant-plgn", "Basic", "owner", "1.0.0", "publisher")
	service.publish_listing("listing-basic", "tenant-plgn", "plugin-basic", "Basic")
	service.install_plugin("install-basic", "tenant-plgn", "plugin-basic", "tenant-admin")

	with pytest.raises(PermissionError, match="plugin_sandbox_required"):
		service.enable_plugin("install-basic", "tenant-plgn", "tenant-admin")


def test_lifecycle_ids_are_tenant_scoped():
	service = PlgnService()

	for tenant_id, owner, title in (
		("tenant-a", "owner-a", "Plugin A"),
		("tenant-b", "owner-b", "Plugin B"),
	):
		service.register_plugin("shared-plugin", tenant_id, title, owner, "1.0.0", "publisher")
		service.publish_listing("shared-listing", tenant_id, "shared-plugin", title)
		service.register_plgn_agent(tenant_id, "Reviewer", "codex", "manifest_reviewer", "review tenant plugins", agent_id="shared-agent")

	assert service.list_plugins("tenant-a")[0]["owner"] == "owner-a"
	assert service.list_plugins("tenant-b")[0]["owner"] == "owner-b"
	assert service.list_marketplace_listings("tenant-a")[0]["title"] == "Plugin A"
	assert service.list_marketplace_listings("tenant-b")[0]["title"] == "Plugin B"
	assert service.list_plgn_agents("tenant-a")[0]["id"] == "shared-agent"
	assert service.list_plgn_agents("tenant-b")[0]["id"] == "shared-agent"


def test_generated_evidence_and_docs_are_current():
	app = _load_module("plgn_app_under_test", PACKAGE_DIR / "app.py")
	model = app.semantic_model()
	committed_model = json.loads((PACKAGE_DIR / "semantic_model.json").read_text(encoding="utf-8"))

	assert app.self_test()["passed"] is True
	assert model == committed_model
	assert model["capabilities"]["plgn"]["streaming"]["processor"] == "bytewax"
	assert model["capabilities"]["plgn"]["screens"]["agents"]["route"] == "/plgn/agents"
	for name in ("README.md", "SPECIFICATION.md", "PLAN.md"):
		assert (PACKAGE_DIR / name).exists()
