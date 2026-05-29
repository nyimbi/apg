"""Website Builder package runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.wsbl import views
from capabilities.common.wsbl.service import WsblService


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def _build_publishable_site(service: WsblService) -> tuple[dict, dict, dict, dict]:
	site = service.create_site(
		site_key="marketing",
		tenant_id="tenant-wsbl",
		name="Marketing Site",
		owner_id="owner-1",
		primary_domain="www.example.test",
		domain_validated=True,
	)
	component = service.create_component(
		component_key="hero",
		tenant_id="tenant-wsbl",
		name="Hero Banner",
		custom=True,
		reviewed=True,
		reviewed_by="reviewer-1",
		policy_id="component-policy",
	)
	page = service.create_page(site["id"], "home", "Home", tenant_id="tenant-wsbl")
	page = service.add_page_section(page["id"], component["id"], {"headline": "Welcome"}, actor_id="editor-1")
	return site, component, page, service.dashboard_summary("tenant-wsbl")


def test_package_contract_shape_and_entrypoint_are_publishable():
	contract_module = _load_module("wsbl_contract_runtime", PACKAGE_DIR / "capability_contract.py")
	app_module = _load_module("wsbl_app_runtime", PACKAGE_DIR / "app.py")

	contract = contract_module.get_capability_contract("tenant-test")
	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")

	self_test = app_module.self_test()
	manifest = app_module.component_manifest()
	model = app_module.semantic_model()

	assert contract["capability"] == "wsbl"
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert model["format"] == "apg.semantic-model.v1"
	assert "wsbl" in model["capabilities"]


def test_site_page_component_publish_lifecycle_executes():
	service = WsblService()
	site, component, page, summary = _build_publishable_site(service)
	request = service.create_publish_request(
		site["id"],
		requested_by="publisher-1",
		approval_recorded=True,
		accessibility_passed=True,
		consent_policy_attached=True,
	)
	published = service.publish_site(request["id"], actor_id="publisher-1")

	assert site["status"] == "ready"
	assert component["status"] == "approved"
	assert page["status"] == "review_ready"
	assert summary["site_count"] == 1
	assert request["status"] == "approved"
	assert published["site"]["status"] == "published"
	assert published["site"]["published_version"] == 1
	assert published["publish_request"]["status"] == "published"
	assert service.list_pages("tenant-wsbl")[0]["status"] == "published"


def test_create_site_requires_tenant_and_owner():
	service = WsblService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_site("missing-tenant", "", "Missing Tenant", "owner-1")
	with pytest.raises(PermissionError, match="site_owner_required"):
		service.create_site("missing-owner", "tenant-wsbl", "Missing Owner", "")


def test_unvalidated_domain_marks_site_pending_without_blocking_design():
	service = WsblService()
	site = service.create_site(
		site_key="campaign",
		tenant_id="tenant-wsbl",
		name="Campaign Site",
		owner_id="owner-1",
		primary_domain="campaign.example.test",
		domain_validated=False,
	)
	domain = service.list_domains("tenant-wsbl")[0]
	validated = service.validate_domain(domain["id"], actor_id="owner-1")
	updated_site = service.list_sites("tenant-wsbl")[0]

	assert site["status"] == "domain_pending"
	assert site["required_actions"] == ["validate_domain"]
	assert validated["validated"] is True
	assert updated_site["status"] == "ready"
	assert updated_site["required_actions"] == []


def test_custom_components_require_review_before_page_use():
	service = WsblService()
	site = service.create_site("docs", "tenant-wsbl", "Docs", "owner-1")
	component = service.create_component("custom-nav", "tenant-wsbl", "Custom Nav", custom=True)
	page = service.create_page(site["id"], "docs", "Docs", tenant_id="tenant-wsbl")

	with pytest.raises(PermissionError, match="component_review_required"):
		service.add_page_section(page["id"], component["id"], {"links": []}, actor_id="editor-1")

	reviewed = service.review_component(component["id"], reviewer_id="reviewer-1", policy_id="component-policy")
	updated_page = service.add_page_section(page["id"], component["id"], {"links": []}, actor_id="editor-1")

	assert reviewed["status"] == "approved"
	assert updated_page["sections"][0]["component_id"] == component["id"]


def test_publish_requires_approval_accessibility_and_consent_policy():
	service = WsblService()
	site, _, _, _ = _build_publishable_site(service)

	with pytest.raises(PermissionError, match="site_publish_approval_required"):
		service.create_publish_request(site["id"], "publisher-1", accessibility_passed=True, consent_policy_attached=True)
	with pytest.raises(PermissionError, match="accessibility_pass_required"):
		service.create_publish_request(site["id"], "publisher-1", approval_recorded=True, consent_policy_attached=True)

	review = service.create_publish_request(
		site["id"],
		"publisher-1",
		approval_recorded=True,
		accessibility_passed=True,
		consent_policy_attached=False,
	)

	assert review["status"] == "review_required"
	assert review["required_actions"] == ["attach_consent_policy"]
	with pytest.raises(PermissionError, match="publish_request_not_approved"):
		service.publish_site(review["id"], actor_id="publisher-1")


def test_api_helpers_expose_website_builder_lifecycle():
	from capabilities.common.wsbl import api

	api.SERVICE = WsblService()
	site = api.create_site({
		"site_key": "storefront",
		"tenant_id": "tenant-api",
		"name": "Storefront",
		"owner_id": "owner-api",
		"domain_validated": True,
	})
	component = api.create_component({
		"component_key": "product-list",
		"tenant_id": "tenant-api",
		"name": "Product List",
	})
	page = api.create_page({"site_id": site["id"], "tenant_id": "tenant-api", "slug": "products", "title": "Products"})
	page = api.add_page_section({"page_id": page["id"], "component_id": component["id"], "content": {"count": 3}})
	status = api.capability_status("tenant-api")
	listing = api.list_website_builder("tenant-api")

	assert page["status"] == "review_ready"
	assert status["site_count"] == 1
	assert status["page_count"] == 1
	assert listing["components"][0]["name"] == "Product List"


def test_view_models_match_routes_theme_and_builder_state():
	service = WsblService()
	site, component, page, _ = _build_publishable_site(service)
	request = service.create_publish_request(site["id"], "publisher-1", approval_recorded=True, accessibility_passed=True, consent_policy_attached=True)

	dashboard = views.dashboard_model(service, "tenant-wsbl")
	editor = views.page_editor_model(service, "tenant-wsbl", page["id"])
	components = views.component_library_model(service, "tenant-wsbl")
	publishing = views.publish_queue_model(service, "tenant-wsbl")
	analytics = views.analytics_model(service, "tenant-wsbl")
	settings = views.settings_model("tenant-wsbl")

	assert dashboard["summary"]["site_count"] == 1
	assert editor["selected_page"]["id"] == page["id"]
	assert editor["component_palette"][0]["id"] == component["id"]
	assert components["pending_review"] == []
	assert publishing["publish_requests"][0]["id"] == request["id"]
	assert analytics["signals"]["published_site_ratio"] == 0.0
	assert settings["theme"]["name"] == "wsbl_site_builder"
