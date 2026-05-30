"""THEM package contract and deterministic runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.them import api, views
from capabilities.common.them.service import ThemService


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_is_valid():
	module = _load_module("package_contract_them", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "them"
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert "them_agents" in contract["provides"]


def test_app_entrypoint_is_publishable():
	module = _load_module("package_app_them", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "them" in model["capabilities"]


def test_theme_design_asset_preview_and_publish_lifecycle_executes():
	service = ThemService()

	theme = service.create_theme(
		tenant_id="tenant-a",
		name="Datacraft Operations",
		owner="design-lead",
		brand_name="Datacraft",
		guidelines_ref="brand://guidelines/datacraft",
	)
	tokens = service.update_tokens(
		tenant_id="tenant-a",
		theme_id=theme["id"],
		group="color",
		tokens={"color.primary": "#235789", "color.accent": "#F1A208"},
		updated_by="designer",
		contrast_validated=True,
	)
	asset = service.add_brand_asset(
		tenant_id="tenant-a",
		theme_id=theme["id"],
		asset_name="primary-logo",
		asset_type="logo",
		license_ref="license://logo/1",
		approved_by="brand-owner",
	)
	preview = service.create_preview(
		tenant_id="tenant-a",
		theme_id=theme["id"],
		surface="erp_shell",
		viewport="desktop",
		preview_ref="preview://theme/1",
		contrast_passed=True,
		created_by="designer",
	)
	publication = service.publish_theme(
		tenant_id="tenant-a",
		theme_id=theme["id"],
		published_by="release-manager",
		approval_ref="approval://theme/1",
		target_tenant_count=3,
	)
	summary = service.dashboard_summary("tenant-a")

	assert theme["status"] == "draft"
	assert tokens["version"] == 1
	assert asset["status"] == "approved"
	assert preview["contrast_passed"] is True
	assert publication["status"] == "published"
	assert summary["theme_count"] == 1
	assert summary["published_theme_count"] == 1
	assert summary["approved_asset_count"] == 1
	assert summary["streaming"]["processor"] == "bytewax"


def test_theme_guardrails_require_tenant_owner_license_preview_contrast_and_approval():
	service = ThemService()

	try:
		service.create_theme("", "No tenant", "owner", "Brand", "guidelines://brand")
	except PermissionError as exc:
		assert str(exc) == "tenant_context_required"
	else:
		raise AssertionError("missing tenant was accepted")

	try:
		service.create_theme("tenant-a", "No owner", "", "Brand", "guidelines://brand")
	except PermissionError as exc:
		assert str(exc) == "theme_owner_required"
	else:
		raise AssertionError("missing owner was accepted")

	theme = service.create_theme("tenant-a", "Brand theme", "owner", "Brand", "guidelines://brand")

	try:
		service.add_brand_asset("tenant-a", theme["id"], "logo", "logo", "", "brand-owner")
	except PermissionError as exc:
		assert str(exc) == "brand_asset_license_required"
	else:
		raise AssertionError("unlicensed brand asset was accepted")

	try:
		service.add_brand_asset("tenant-a", theme["id"], "logo", "logo", "license://logo", "")
	except PermissionError as exc:
		assert str(exc) == "brand_asset_approval_required"
	else:
		raise AssertionError("unapproved brand asset was accepted")

	try:
		service.publish_theme("tenant-a", theme["id"], "publisher", "approval://theme")
	except PermissionError as exc:
		assert str(exc) == "theme_preview_required"
	else:
		raise AssertionError("theme without preview was published")

	service.create_preview("tenant-a", theme["id"], "shell", "desktop", "preview://bad", False, "designer")

	try:
		service.publish_theme("tenant-a", theme["id"], "publisher", "approval://theme")
	except PermissionError as exc:
		assert str(exc) == "contrast_validation_required"
	else:
		raise AssertionError("theme without contrast validation was published")

	service.create_preview("tenant-a", theme["id"], "shell", "desktop", "preview://good", True, "designer")

	try:
		service.publish_theme("tenant-a", theme["id"], "publisher", "")
	except PermissionError as exc:
		assert str(exc) == "theme_publish_approval_required"
	else:
		raise AssertionError("theme without approval was published")

	try:
		service.publish_theme("tenant-a", theme["id"], "publisher", "approval://theme", event_stream="local")
	except PermissionError as exc:
		assert str(exc) == "bytewax_event_stream_required"
	else:
		raise AssertionError("theme publication without Bytewax was accepted")

	publication = service.publish_theme("tenant-a", theme["id"], "publisher", "approval://theme", target_tenant_count=8)
	assert publication["status"] == "review_required"
	assert publication["required_actions"] == ["review_theme_rollout"]


def test_theme_agents_and_batch_rollout_guardrails_execute():
	service = ThemService()

	agent = service.register_them_agent(
		tenant_id="tenant-a",
		name="Design reviewer",
		runtime="codex",
		role="design_token_reviewer",
		scope="review token, contrast, and rollout evidence",
	)
	privileged = service.validate_agent_theme_action(
		tenant_id="tenant-a",
		agent_id=agent["id"],
		action="publish_theme",
		privileged_scope=True,
	)
	approved = service.validate_agent_theme_action(
		tenant_id="tenant-a",
		agent_id=agent["id"],
		action="publish_theme",
		privileged_scope=True,
		human_approval_ref="approval://agent/theme",
	)
	batch_block = service.validate_batch_theme_rollout("tenant-a", 12, event_stream="local")

	assert agent["runtime"] == "codex"
	assert privileged["decision"] == "deny"
	assert privileged["matched_rules"] == ["privileged_agent_theme_action_requires_human_approval"]
	assert approved["decision"] == "allow"
	assert batch_block["decision"] == "deny"
	assert "batch_theme_rollout_requires_bytewax" in batch_block["matched_rules"]

	try:
		service.register_them_agent("tenant-a", "Unsupported", "unknown", "design_token_reviewer", "review")
	except PermissionError as exc:
		assert str(exc) == "them_agent_runtime_not_supported"
	else:
		raise AssertionError("unsupported theme agent runtime was accepted")


def test_api_and_view_models_expose_theme_system_surfaces():
	local_service = ThemService()
	api.SERVICE = local_service

	theme = api.create_theme({
		"tenant_id": "tenant-b",
		"name": "Operations",
		"owner": "design-lead",
		"brand_name": "Operations Brand",
		"guidelines_ref": "brand://guidelines/ops",
	})
	api.update_tokens({
		"tenant_id": "tenant-b",
		"theme_id": theme["id"],
		"group": "density",
		"tokens": {"density": "compact"},
		"updated_by": "designer",
		"contrast_validated": True,
	})
	api.add_brand_asset({
		"tenant_id": "tenant-b",
		"theme_id": theme["id"],
		"asset_name": "mark",
		"asset_type": "logo",
		"license_ref": "license://mark",
		"approved_by": "brand-owner",
	})
	api.create_preview({
		"tenant_id": "tenant-b",
		"theme_id": theme["id"],
		"surface": "dashboard",
		"viewport": "mobile",
		"preview_ref": "preview://ops/mobile",
		"contrast_passed": True,
		"created_by": "designer",
	})
	api.publish_theme({
		"tenant_id": "tenant-b",
		"theme_id": theme["id"],
		"published_by": "release-manager",
		"approval_ref": "approval://ops",
		"target_tenant_count": 1,
	})
	agent = api.register_them_agent({
		"tenant_id": "tenant-b",
		"name": "Brand reviewer",
		"runtime": "claude_code",
		"role": "brand_reviewer",
		"scope": "review brand assets",
	})

	status = api.capability_status("tenant-b")
	system = api.list_theme_system("tenant-b")
	dashboard = views.dashboard_model(local_service, "tenant-b")
	console = views.theme_console_model(local_service, "tenant-b")
	tokens = views.token_editor_model(local_service, "tenant-b")
	branding = views.brand_guidelines_model(local_service, "tenant-b")
	assets = views.brand_asset_manager_model(local_service, "tenant-b")
	preview = views.preview_model(local_service, "tenant-b")
	policies = views.policies_model(local_service, "tenant-b")
	agents = views.agent_workbench_model(local_service, "tenant-b")
	settings = views.settings_model("tenant-b")

	assert status["published_theme_count"] == 1
	assert status["them_agent_count"] == 1
	assert system["summary"]["approved_asset_count"] == 1
	assert system["them_agents"][0]["id"] == agent["id"]
	assert dashboard["summary"]["publication_count"] == 1
	assert console["route"] == "/them/themes"
	assert tokens["contrast_validation_required"] is True
	assert branding["guidelines_required"] is True
	assert assets["license_required"] is True
	assert preview["viewports"] == ["mobile", "tablet", "desktop"]
	assert policies["review_required"] == []
	assert agents["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert settings["theme"]["name"] == "them_brand_system"
	assert settings["streaming"]["processor"] == "bytewax"
