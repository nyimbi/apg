"""Executable PIM capability package tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def _build_lifecycle(service):
	catalog = service.create_catalog("catalog-1", "tenant-test", "MAIN", "Main Catalog", "owner-1")
	product = service.create_product("product-1", "tenant-test", catalog["id"], "SKU-1", "Solar Charger", "physical", "owner-1")
	attribute = service.define_attribute("attr-1", "tenant-test", "description", "Description", "rich_text", "owner-1")
	value = service.set_attribute_value("value-1", "tenant-test", product["id"], attribute["id"], "Portable charger", "en")
	variant = service.create_variant("variant-1", "tenant-test", product["id"], "SKU-1-BLACK", {"color": "black"})
	content = service.enrich_content("content-1", "tenant-test", product["id"], "en", "Solar Charger", "Portable charger", True, "reviewer-1")
	asset = service.attach_asset("asset-1", "tenant-test", product["id"], "image", "https://example.test/asset.jpg", "licensed")
	compliance = service.record_compliance("comp-1", "tenant-test", product["id"], "rohs", "compliant", "doc-1")
	channel = service.create_channel_listing("listing-1", "tenant-test", product["id"], "web", "web-sku-1", "approver-1")
	publication = service.publish_product("pub-1", "tenant-test", product["id"], content["id"], channel["id"], "approver-2")
	quality = service.record_quality_issue("quality-1", "tenant-test", product["id"], "medium", "missing alt text")
	change = service.create_change_request("change-1", "tenant-test", product["id"], "update description", "requester-1")
	approved_change = service.approve_change(change["id"], "tenant-test", "approver-3")
	agent = service.register_pim_agent("tenant-test", "Catalog Reviewer", "codex", "catalog_reviewer", "review product data")
	return {"catalog": catalog, "product": product, "attribute": attribute, "value": value, "variant": variant, "content": content, "asset": asset, "compliance": compliance, "channel": channel, "publication": publication, "quality": quality, "change": approved_change, "agent": agent}


def test_contract_shape_streaming_routes_and_agents_are_valid():
	module = _load_module("contract_pim", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "pde_pim"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "pim_agents" in contract["provides"]
	assert "/pde/pim/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"


def test_rule_engine_blocks_missing_context_non_bytewax_and_review_gaps():
	module = _load_module("rules_pim", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "pim_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "enrich_content", "generated_content": True, "review_recorded": False})["matched_rules"] == ["generated_content_review_required"]
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "agent_action", "privileged_action": True, "human_approved": False})["decision"] == "require_review"


def test_service_executes_pim_lifecycle():
	service_module = _load_module("service_pim", PACKAGE_DIR / "service.py")
	service = service_module.ProductInformationLifecycleService()
	records = _build_lifecycle(service)
	summary = service.dashboard_summary("tenant-test")

	assert records["catalog"]["code"] == "MAIN"
	assert records["product"]["sku"] == "SKU-1"
	assert records["attribute"]["attribute_type"] == "rich_text"
	assert records["content"]["status"] == "approved"
	assert records["publication"]["status"] == "published"
	assert records["change"]["status"] == "approved"
	assert records["agent"]["role"] == "catalog_reviewer"
	assert summary["product_count"] == 1
	assert summary["audit_event_count"] == 14
	assert summary["streaming"]["processor"] == "bytewax"


def test_service_guardrails_reject_invalid_actions():
	service_module = _load_module("guardrail_service_pim", PACKAGE_DIR / "service.py")
	service = service_module.ProductInformationLifecycleService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_catalog("catalog", "", "MAIN", "Main", "owner")
	with pytest.raises(PermissionError, match="catalog_code_required"):
		service.create_catalog("catalog", "tenant-test", "", "Main", "owner")
	catalog = service.create_catalog("catalog", "tenant-test", "MAIN", "Main", "owner")
	with pytest.raises(PermissionError, match="product_type_not_supported"):
		service.create_product("product", "tenant-test", catalog["id"], "SKU", "Name", "unknown", "owner")
	product = service.create_product("product", "tenant-test", catalog["id"], "SKU", "Name", "physical", "owner")
	attribute = service.define_attribute("attribute", "tenant-test", "desc", "Description", "rich_text", "owner")
	with pytest.raises(PermissionError, match="attribute_locale_required"):
		service.set_attribute_value("value", "tenant-test", product["id"], attribute["id"], "Body")
	with pytest.raises(PermissionError, match="generated_content_review_required"):
		service.enrich_content("content", "tenant-test", product["id"], "en", "Title", "Body", True)
	with pytest.raises(PermissionError, match="compliance_review_required"):
		service.record_compliance("compliance", "tenant-test", product["id"], "CE", "review", "doc-1", "high")
	with pytest.raises(PermissionError, match="channel_not_supported"):
		service.create_channel_listing("listing", "tenant-test", product["id"], "unsupported", "web-sku", "approver")
	with pytest.raises(PermissionError, match="channel_approval_required"):
		service.create_channel_listing("listing", "tenant-test", product["id"], "web", "web-sku", "")
	with pytest.raises(PermissionError, match="approved_content_required"):
		service.publish_product("publication", "tenant-test", product["id"], "missing-content", "missing-channel", "approver")
	with pytest.raises(PermissionError, match="quality_owner_required"):
		service.record_quality_issue("quality", "tenant-test", product["id"], "high", "bad data")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, "queue")
	with pytest.raises(PermissionError, match="pim_agent_runtime_not_supported"):
		service.register_pim_agent("tenant-test", "Agent", "unsupported", "catalog_reviewer", "review")


def test_agents_batch_api_views_and_app_are_executable():
	api_module = _load_module("api_pim", PACKAGE_DIR / "api.py")
	views = _load_module("views_pim", PACKAGE_DIR / "views.py")
	app = _load_module("app_pim", PACKAGE_DIR / "app.py")

	catalog = api_module.create_catalog({"tenant_id": "tenant-api", "id": "catalog-api", "code": "API", "name": "API Catalog", "owner_id": "owner"})
	product = api_module.create_product({"tenant_id": "tenant-api", "id": "product-api", "catalog_id": catalog["id"], "sku": "SKU-API", "name": "API Product", "product_type": "digital", "owner_id": "owner"})
	agent = api_module.register_pim_agent({"tenant_id": "tenant-api", "name": "Quality Reviewer", "runtime": "claude_code", "role": "data_quality_reviewer"})
	model = views.product_model(api_module.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert product["id"] == "product-api"
	assert agent["role"] == "data_quality_reviewer"
	assert model["records"][0]["name"] == "API Product"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["pde_pim"]["screens"]["agents"]["route"] == "/pde/pim/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_pim", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["pde_pim"]["streaming"]["processor"] == "bytewax"
