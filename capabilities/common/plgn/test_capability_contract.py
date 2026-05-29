"""Regression coverage for the PLGN executable capability contract."""

import pytest

from capabilities.common.plgn import register_capability
from capabilities.common.plgn.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.plgn.service import PlgnService
from capabilities.common.plgn import views


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-plgn", {"marketplace": {"tenant_install_policy_enabled": False}})

	assert contract["capability"] == "plgn"
	assert contract["configuration"]["tenant_id"] == "tenant-plgn"
	assert contract["configuration"]["marketplace"]["tenant_install_policy_enabled"] is False
	assert contract["configuration_schema"]["required"] == ["tenant_id", "marketplace", "plugins", "security", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "marketplace", "plugins", "manifests", "permissions", "sandbox", "releases", "settings"}
	assert contract["theme"]["name"] == "plgn_extension_marketplace"


def test_rule_engine_enforces_plgn_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "register_plugin", "plugin_owner_assigned": False, "signature_verified": False, "permissions_requested": True, "permission_review_recorded": False, "external_plugin": True, "external_review_recorded": False})
	enable_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "enable_plugin", "signature_verified": True, "sandbox_policy_attached": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "plugin_requires_owner", "plugin_requires_signature", "permissions_require_review", "external_plugin_requires_review"}
	assert enable_result["matched_rules"] == ["plugin_requires_sandbox"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "plgn"
	assert "secu" in registration["dependencies"]
	assert registration["ui_components"]["marketplace"] == "/plgn/marketplace"
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
	release = service.create_release("release-risk", "tenant-plgn", "plugin-risk", "1.2.0", "stable", "sig-risk")
	installation = service.install_plugin("install-risk", "tenant-plgn", "plugin-risk", "tenant-admin")
	enabled = service.enable_plugin("install-risk", "tenant-plgn", "tenant-admin")
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
	assert summary["enabled_plugin_count"] == 1
	assert summary["permission_review_count"] == 1
	assert dashboard["summary"]["plugin_count"] == 1
	assert marketplace["listings"][0]["id"] == "listing-risk"
	assert releases["releases"][0]["id"] == "release-risk"
	assert len(service.list_audit_events("tenant-plgn")) >= 7


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
	service.create_release("release-risk", "tenant-plgn", "plugin-risk", "1.0.0", "stable", "sig")

	with pytest.raises(PermissionError, match="admin_install_required"):
		service.install_plugin("install-risk", "tenant-plgn", "plugin-risk", "analyst")


def test_enable_requires_sandbox_policy():
	service = PlgnService()
	service.register_plugin("plugin-basic", "tenant-plgn", "Basic", "owner", "1.0.0", "publisher")
	service.publish_listing("listing-basic", "tenant-plgn", "plugin-basic", "Basic")
	service.install_plugin("install-basic", "tenant-plgn", "plugin-basic", "tenant-admin")

	with pytest.raises(PermissionError, match="plugin_sandbox_required"):
		service.enable_plugin("install-basic", "tenant-plgn", "tenant-admin")
