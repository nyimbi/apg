"""SRCH package contract and deterministic runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.srch import api, views
from capabilities.common.srch.service import SrchService


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
	module = _load_module("materialized_contract_srch", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "srch"
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"]


def test_app_entrypoint_is_publishable():
	module = _load_module("materialized_app_srch", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "srch" in model["capabilities"]


def test_search_index_document_query_lifecycle_executes():
	service = SrchService()

	index = service.create_index(
		tenant_id="tenant-a",
		name="knowledge-base",
		owner="search-owner",
		content_type="article",
		classification="internal",
		source_lineage_ref="lineage://kb",
		embedding_index_ready=False,
	)
	ready_index = service.mark_embedding_index_ready("tenant-a", index["id"], "search-owner")
	documents = service.bulk_index_documents(
		tenant_id="tenant-a",
		index_id=index["id"],
		source_lineage_ref="lineage://kb/batch-1",
		documents=[
			{
				"document_id": "doc-1",
				"title": "APG search overview",
				"body": "APG provides keyword and semantic enterprise search.",
				"facets": {"module": "platform", "kind": "guide"},
			},
			{
				"document_id": "doc-2",
				"title": "APG indexing",
				"body": "Indexing requires source lineage and owner accountability.",
				"facets": {"module": "platform", "kind": "reference"},
			},
		],
	)
	response = service.query(
		tenant_id="tenant-a",
		query_text="semantic search",
		index_ids=[index["id"]],
		query_type="semantic",
		result_window=10,
		rbac_filter_applied=True,
	)
	summary = service.dashboard_summary("tenant-a")

	assert ready_index["embedding_index_ready"] is True
	assert len(documents) == 2
	assert response["query"]["status"] == "completed"
	assert response["query"]["result_count"] == 1
	assert response["results"][0]["document_id"] == "doc-1"
	assert response["facets"]["module"]["platform"] == 2
	assert summary["index_count"] == 1
	assert summary["document_count"] == 2
	assert summary["query_count"] == 1


def test_search_guardrails_require_tenant_owner_lineage_rbac_embeddings_and_review():
	service = SrchService()

	try:
		service.create_index("", "no-tenant", "owner")
	except PermissionError as exc:
		assert str(exc) == "tenant_context_required"
	else:
		raise AssertionError("missing tenant was accepted")

	try:
		service.create_index("tenant-a", "no-owner", "")
	except PermissionError as exc:
		assert str(exc) == "index_owner_required"
	else:
		raise AssertionError("missing owner was accepted")

	index = service.create_index(
		tenant_id="tenant-a",
		name="restricted-archive",
		owner="records-owner",
		classification="restricted",
		source_lineage_ref="lineage://restricted",
	)

	try:
		service.bulk_index_documents("tenant-a", index["id"], [{"document_id": "doc-1", "body": "restricted"}], None)
	except PermissionError as exc:
		assert str(exc) == "source_lineage_required"
	else:
		raise AssertionError("bulk indexing without lineage was accepted")

	service.index_document(
		tenant_id="tenant-a",
		index_id=index["id"],
		document_id="doc-1",
		title="Restricted record",
		body="Restricted semantic content",
		source_lineage_ref="lineage://restricted/doc-1",
	)

	try:
		service.query("tenant-a", "restricted", [index["id"]], rbac_filter_applied=False)
	except PermissionError as exc:
		assert str(exc) == "rbac_filter_required"
	else:
		raise AssertionError("restricted query without RBAC filter was accepted")

	try:
		service.query("tenant-a", "restricted", [index["id"]], query_type="semantic", rbac_filter_applied=True)
	except PermissionError as exc:
		assert str(exc) == "embedding_index_required"
	else:
		raise AssertionError("semantic query without embeddings was accepted")

	service.mark_embedding_index_ready("tenant-a", index["id"], "records-owner")
	reviewed = service.query(
		tenant_id="tenant-a",
		query_text="restricted",
		index_ids=[index["id"]],
		query_type="semantic",
		result_window=5001,
		rbac_filter_applied=True,
		review_recorded=False,
	)
	assert reviewed["query"]["status"] == "review_required"
	assert reviewed["query"]["required_actions"] == ["record_query_review"]


def test_api_and_view_models_expose_search_engine_surfaces():
	local_service = SrchService()
	api.SERVICE = local_service

	index = api.create_index({
		"tenant_id": "tenant-b",
		"name": "policies",
		"owner": "policy-owner",
		"classification": "internal",
		"source_lineage_ref": "lineage://policies",
		"embedding_index_ready": True,
	})
	api.index_document({
		"tenant_id": "tenant-b",
		"index_id": index["id"],
		"document_id": "policy-1",
		"title": "Travel policy",
		"body": "Travel policy defines approvals and reimbursements.",
		"facets": {"module": "hr", "kind": "policy"},
	})
	search_result = api.query({
		"tenant_id": "tenant-b",
		"query_text": "travel policy",
		"index_ids": [index["id"]],
		"query_type": "hybrid",
		"result_window": 25,
	})

	status = api.capability_status("tenant-b")
	search_engine = api.list_search_engine("tenant-b")
	dashboard = views.dashboard_model(local_service, "tenant-b")
	console = views.search_console_model(local_service, "tenant-b")
	indices = views.index_manager_model(local_service, "tenant-b")
	documents = views.document_indexer_model(local_service, "tenant-b")
	analytics = views.analytics_model(local_service, "tenant-b")
	governance = views.governance_model(local_service, "tenant-b")
	settings = views.settings_model("tenant-b")

	assert search_result["query"]["status"] == "completed"
	assert status["index_count"] == 1
	assert search_engine["summary"]["document_count"] == 1
	assert dashboard["summary"]["query_count"] == 1
	assert console["query_types"] == ["keyword", "semantic", "hybrid"]
	assert indices["classifications"] == ["public", "internal", "confidential", "restricted"]
	assert documents["lineage_required"] is True
	assert analytics["facets"]["kind"]["policy"] == 1
	assert governance["review_required_queries"] == []
	assert settings["theme"]["name"] == "srch_discovery_console"
