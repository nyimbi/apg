"""Regression coverage for the SRCH executable capability contract."""

from __future__ import annotations

import pytest

from capabilities.common.srch import api, register_capability, views
from capabilities.common.srch.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract,
)
from capabilities.common.srch.service import SrchService


def test_contract_exposes_full_lifecycle_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-search", {"query": {"max_result_window": 250}})

	assert contract["capability"] == "srch"
	assert contract["configuration"]["tenant_id"] == "tenant-search"
	assert contract["configuration"]["query"]["max_result_window"] == 250
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"indices",
		"documents",
		"indexing",
		"query",
		"ranking",
		"facets",
		"security",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"search",
		"indices",
		"documents",
		"bulk",
		"facets",
		"analytics",
		"ranking",
		"access",
		"governance",
		"audit",
		"settings",
	}
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "service.SrchService"
	assert next(route for route in contract["ui"]["routes"] if route["name"] == "audit")["permission"] == "srch:audit"
	assert contract["ui"]["api_prefix"] == "/srch/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert {"result_card", "facet_panel", "bulk_queue", "ranking_panel", "audit_timeline"} <= set(contract["theme"]["components"])


def test_rule_engine_enforces_search_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "query",
		"query_text_present": False,
		"index_ids_present": False,
		"query_type_present": False,
		"content_classification": "restricted",
		"rbac_filter_applied": False,
		"query_type": "semantic",
		"embedding_index_ready": False,
		"result_window": 5000,
		"result_window_review_check": True,
		"review_recorded": False,
	})
	bulk_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "bulk_index",
		"document_count": 0,
		"source_lineage_present": False,
	})
	batch_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "configure_batch_indexing",
		"event_stream": "kafka",
	})
	state_change_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"state_change_requested": True,
		"audit_event_recorded": False,
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"query_requires_text",
		"query_requires_index",
		"query_requires_type",
		"restricted_query_requires_rbac_filter",
		"semantic_query_requires_embeddings",
		"large_result_window_requires_review",
	}
	assert set(bulk_result["matched_rules"]) == {"bulk_index_requires_documents", "bulk_index_requires_lineage"}
	assert batch_result["matched_rules"] == ["batch_indexing_requires_bytewax"]
	assert state_change_result["matched_rules"] == ["search_state_change_requires_audit"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "srch"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "srch_discovery_console"
	assert registration["ui_components"]["indices"] == "/srch/indices"
	assert registration["ui_components"]["audit"] == "/srch/audit"
	assert "nlpc" in registration["dependencies"]
	assert "bulk_indexing" in registration["capabilities"]
	assert "query_analytics" in registration["capabilities"]
	assert "srch:audit" in registration["permissions"]


def test_search_index_document_query_and_view_lifecycle_executes():
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
		query_type="hybrid",
		result_window=10,
		rbac_filter_applied=True,
	)
	summary = service.dashboard_summary("tenant-a")
	dashboard = views.dashboard_model(service, "tenant-a")
	console = views.search_console_model(service, "tenant-a")
	indices = views.index_manager_model(service, "tenant-a")
	documents_view = views.document_indexer_model(service, "tenant-a")
	bulk = views.bulk_index_model(service, "tenant-a")
	facets = views.facet_explorer_model(service, "tenant-a")
	analytics = views.analytics_model(service, "tenant-a")
	ranking = views.ranking_model(service, "tenant-a")
	access = views.access_review_model(service, "tenant-a")
	governance = views.governance_model(service, "tenant-a")
	audit = views.audit_timeline_model(service, "tenant-a")

	assert ready_index["embedding_index_ready"] is True
	assert len(documents) == 2
	assert response["query"]["status"] == "completed"
	assert response["query"]["result_count"] == 1
	assert response["results"][0]["document_id"] == "doc-1"
	assert response["facets"]["module"]["platform"] == 2
	assert summary["index_count"] == 1
	assert summary["document_count"] == 2
	assert dashboard["summary"]["query_count"] == 1
	assert console["query_types"] == ["keyword", "semantic", "hybrid"]
	assert indices["classifications"] == ["public", "internal", "confidential", "restricted"]
	assert documents_view["lineage_required"] is True
	assert bulk["event_stream"] == "bytewax"
	assert facets["facets"]["kind"]["guide"] == 1
	assert analytics["facets"]["kind"]["reference"] == 1
	assert ranking["ranking"]["explain_ranking"] is True
	assert access["denied_queries"] == []
	assert governance["review_required_queries"] == []
	assert audit["audit_events"]


def test_search_guardrails_require_tenant_owner_lineage_rbac_embeddings_and_review():
	service = SrchService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_index("", "no-tenant", "owner", "document", "internal")

	with pytest.raises(PermissionError, match="index_owner_required"):
		service.create_index("tenant-a", "no-owner", "", "document", "internal")

	with pytest.raises(PermissionError, match="index_content_type_required"):
		service.create_index("tenant-a", "no-type", "owner", "", "internal")

	with pytest.raises(PermissionError, match="index_classification_required"):
		service.create_index("tenant-a", "no-classification", "owner", "document", "")

	with pytest.raises(PermissionError, match="restricted_index_lineage_required"):
		service.create_index("tenant-a", "restricted-no-lineage", "owner", "document", "restricted")

	index = service.create_index(
		tenant_id="tenant-a",
		name="restricted-archive",
		owner="records-owner",
		content_type="document",
		classification="restricted",
		source_lineage_ref="lineage://restricted",
	)

	with pytest.raises(PermissionError, match="source_lineage_required"):
		service.bulk_index_documents("tenant-a", index["id"], [{"document_id": "doc-1", "body": "restricted"}], None)

	with pytest.raises(PermissionError, match="bulk_documents_required"):
		service.bulk_index_documents("tenant-a", index["id"], [], "lineage://restricted/batch")

	with pytest.raises(PermissionError, match="document_title_required"):
		service.index_document(
			tenant_id="tenant-a",
			index_id=index["id"],
			document_id="doc-missing-title",
			title="",
			body="Restricted semantic content",
			source_lineage_ref="lineage://restricted/doc-missing-title",
		)

	service.index_document(
		tenant_id="tenant-a",
		index_id=index["id"],
		document_id="doc-1",
		title="Restricted record",
		body="Restricted semantic content",
		source_lineage_ref="lineage://restricted/doc-1",
	)

	with pytest.raises(PermissionError, match="rbac_filter_required"):
		service.query("tenant-a", "restricted", [index["id"]], query_type="keyword", rbac_filter_applied=False)

	with pytest.raises(PermissionError, match="embedding_index_required"):
		service.query("tenant-a", "restricted", [index["id"]], query_type="semantic", rbac_filter_applied=True)

	with pytest.raises(PermissionError, match="query_text_required"):
		service.query("tenant-a", "", [index["id"]], query_type="keyword")

	with pytest.raises(PermissionError, match="query_index_required"):
		service.query("tenant-a", "restricted", [], query_type="keyword")

	with pytest.raises(PermissionError, match="result_window_required"):
		service.query("tenant-a", "restricted", [index["id"]], query_type="keyword", result_window=0)

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


def test_search_review_evidence_unlocks_review_required_lifecycle_paths():
	service = SrchService()

	with pytest.raises(PermissionError, match="index_content_type_review_required"):
		service.create_index("tenant-a", "custom-content", "owner", "briefing", "internal")
	reviewed_type = service.create_index(
		tenant_id="tenant-a",
		name="reviewed-content",
		owner="owner",
		content_type="briefing",
		classification="internal",
		review_recorded=True,
	)
	assert reviewed_type["content_type"] == "briefing"

	with pytest.raises(PermissionError, match="restricted_index_lineage_required"):
		service.create_index("tenant-a", "unknown-classification", "owner", "document", "regulated", review_recorded=True)
	reviewed_classification = service.create_index(
		tenant_id="tenant-a",
		name="reviewed-classification",
		owner="owner",
		content_type="document",
		classification="regulated",
		source_lineage_ref="lineage://reviewed-classification",
		embedding_index_ready=True,
		review_recorded=True,
	)
	assert reviewed_classification["classification"] == "restricted"

	with pytest.raises(PermissionError, match="facet_key_review_required"):
		service.index_document(
			tenant_id="tenant-a",
			index_id=reviewed_type["id"],
			document_id="facet-review",
			title="Facet review",
			body="Custom facet review",
			facets={"custom": "reviewed"},
			source_lineage_ref="lineage://facet-review",
		)
	reviewed_document = service.index_document(
		tenant_id="tenant-a",
		index_id=reviewed_type["id"],
		document_id="facet-reviewed",
		title="Facet reviewed",
		body="Custom facet reviewed",
		facets={"custom": "reviewed"},
		source_lineage_ref="lineage://facet-reviewed",
		review_recorded=True,
	)
	assert reviewed_document["facets"]["custom"] == "reviewed"

	bulk_review = service.evaluate({
		"tenant_context_present": True,
		"operation": "bulk_index",
		"document_count": 10001,
		"source_lineage_present": True,
		"review_recorded": True,
	})
	assert bulk_review["decision"] == "allow"

	reviewed_query = service.query(
		tenant_id="tenant-a",
		query_text="custom facet",
		index_ids=[reviewed_type["id"]],
		query_type="expansion",
		rbac_filter_applied=True,
		review_recorded=True,
	)
	assert reviewed_query["query"]["status"] == "completed"
	assert reviewed_query["query"]["query_type"] == "expansion"


def test_api_helpers_expose_search_engine_surfaces():
	local_service = SrchService()
	api.SERVICE = local_service

	with pytest.raises(PermissionError, match="index_content_type_required"):
		api.create_index({
			"tenant_id": "tenant-b",
			"name": "missing-type",
			"owner": "policy-owner",
			"classification": "internal",
		})

	index = api.create_index({
		"tenant_id": "tenant-b",
		"name": "policies",
		"owner": "policy-owner",
		"content_type": "document",
		"classification": "internal",
		"source_lineage_ref": "lineage://policies",
		"embedding_index_ready": True,
	})
	reviewed_index = api.create_index({
		"tenant_id": "tenant-b",
		"name": "reviewed-type",
		"owner": "policy-owner",
		"content_type": "briefing",
		"classification": "internal",
		"review_recorded": True,
	})
	with pytest.raises(PermissionError, match="document_title_required"):
		api.index_document({
			"tenant_id": "tenant-b",
			"index_id": index["id"],
			"document_id": "missing-title",
			"body": "Travel policy defines approvals and reimbursements.",
		})

	api.index_document({
		"tenant_id": "tenant-b",
		"index_id": index["id"],
		"document_id": "policy-1",
		"title": "Travel policy",
		"body": "Travel policy defines approvals and reimbursements.",
		"facets": {"module": "hr", "kind": "policy"},
	})
	api.bulk_index_documents({
		"tenant_id": "tenant-b",
		"index_id": reviewed_index["id"],
		"source_lineage_ref": "lineage://reviewed-bulk",
		"review_recorded": True,
		"documents": [{
			"document_id": "reviewed-bulk-1",
			"title": "Reviewed bulk document",
			"body": "Reviewed bulk indexing path.",
		}],
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
	settings = views.settings_model("tenant-b")

	assert search_result["query"]["status"] == "completed"
	assert status["index_count"] == 2
	assert search_engine["summary"]["document_count"] == 2
	assert settings["theme"]["name"] == "srch_discovery_console"
