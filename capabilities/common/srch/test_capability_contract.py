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
		"agents",
		"streaming",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert len(contract["rule_engine"]["rules"]) >= 39
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
		"agents",
		"lifecycle",
		"audit",
		"settings",
	}
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "service.SrchService"
	assert contract["agents"]["first_class"] is True
	assert contract["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert contract["streaming"]["lifecycle_stream"] == "srch.lifecycle"
	assert next(route for route in contract["ui"]["routes"] if route["name"] == "audit")["permission"] == "srch:audit"
	assert contract["ui"]["api_prefix"] == "/srch/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert {
		"result_card",
		"facet_panel",
		"bulk_queue",
		"ranking_panel",
		"search_agent_roster",
		"bytewax_lifecycle_panel",
		"audit_timeline",
	} <= set(contract["theme"]["components"])


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
		"event_stream": "legacy_queue",
	})
	state_change_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"state_change_requested": True,
		"audit_event_recorded": False,
	})
	agent_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_search_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": True,
		"scope_present": True,
		"owner_present": True,
		"purpose_present": True,
		"contribution_disclosed": True,
		"privileged_role": False,
		"human_approval_required": False,
	})
	lifecycle_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "validate_srch_lifecycle_batch",
		"event_stream": "legacy_queue",
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
	assert agent_result["matched_rules"] == ["search_agent_runtime_supported"]
	assert agent_result["actions"][0]["reason"] == "unsupported_search_agent_runtime"
	assert lifecycle_result["matched_rules"] == ["bytewax_srch_stream_required"]
	assert lifecycle_result["actions"][0]["reason"] == "bytewax_lifecycle_stream_required"


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "srch"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["required_processor"] == "bytewax"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "srch_discovery_console"
	assert registration["ui_components"]["indices"] == "/srch/indices"
	assert registration["ui_components"]["agents"] == "/srch/agents"
	assert registration["ui_components"]["lifecycle"] == "/srch/lifecycle"
	assert registration["ui_components"]["audit"] == "/srch/audit"
	assert "nlpc" in registration["dependencies"]
	assert "aicr" in registration["dependencies"]
	assert "bulk_indexing" in registration["capabilities"]
	assert "query_analytics" in registration["capabilities"]
	assert "search_agent_composition" in registration["capabilities"]
	assert "lifecycle_batch_governance" in registration["capabilities"]
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
	agent = service.register_search_agent(
		agent_id="search-agent-1",
		tenant_id="tenant-a",
		name="Search Steward",
		runtime="codex",
		role="search_steward",
		scope="index document query review",
		owner="search-owner",
		purpose="govern search lifecycle changes",
	)
	batch = service.validate_srch_lifecycle_batch(
		tenant_id="tenant-a",
		event_stream="bytewax",
		mutation_count=2,
		operation="search_agent_batch",
		batch_id="srch-batch-001",
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
	agent_roster = views.search_agent_roster_model(service, "tenant-a")
	lifecycle = views.lifecycle_batch_model(service, "tenant-a")
	audit = views.audit_timeline_model(service, "tenant-a")

	assert ready_index["embedding_index_ready"] is True
	assert len(documents) == 2
	assert response["query"]["status"] == "completed"
	assert response["query"]["result_count"] == 1
	assert response["results"][0]["document_id"] == "doc-1"
	assert response["facets"]["module"]["platform"] == 2
	assert agent["runtime"] == "codex"
	assert agent["status"] == "active"
	assert batch["required_processor"] == "bytewax"
	assert batch["status"] == "accepted"
	assert summary["index_count"] == 1
	assert summary["document_count"] == 2
	assert summary["search_agent_count"] == 1
	assert summary["lifecycle_batch_count"] == 1
	assert dashboard["summary"]["query_count"] == 1
	assert dashboard["search_agents"][0]["id"] == "search-agent-1"
	assert dashboard["lifecycle_batches"][0]["id"] == "srch-batch-001"
	assert console["query_types"] == ["keyword", "semantic", "hybrid"]
	assert indices["classifications"] == ["public", "internal", "confidential", "restricted"]
	assert documents_view["lineage_required"] is True
	assert bulk["event_stream"] == "bytewax"
	assert facets["facets"]["kind"]["guide"] == 1
	assert analytics["facets"]["kind"]["reference"] == 1
	assert ranking["ranking"]["explain_ranking"] is True
	assert access["denied_queries"] == []
	assert governance["review_required_queries"] == []
	assert governance["agents"]["first_class"] is True
	assert agent_roster["agents"][0]["role"] == "search_steward"
	assert lifecycle["batches"][0]["operation"] == "search_agent_batch"
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


def test_service_enforces_search_agent_and_lifecycle_guardrails():
	service = SrchService()
	tenant_id = "tenant-agent"

	with pytest.raises(PermissionError, match="unsupported_search_agent_runtime"):
		service.register_search_agent(
			"agent-bad-runtime",
			tenant_id,
			"Bad Runtime",
			"unknown",
			"search_steward",
			"query review",
			"owner",
			"purpose",
		)

	with pytest.raises(PermissionError, match="search_agent_scope_required"):
		service.register_search_agent(
			"agent-no-scope",
			tenant_id,
			"No Scope",
			"codex",
			"search_steward",
			"",
			"owner",
			"purpose",
		)

	with pytest.raises(PermissionError, match="search_agent_contribution_disclosure_required"):
		service.register_search_agent(
			"agent-no-disclosure",
			tenant_id,
			"No Disclosure",
			"codex",
			"search_steward",
			"query review",
			"owner",
			"purpose",
			contribution_disclosed=False,
		)

	agent = service.register_search_agent(
		"agent-review",
		tenant_id,
		"Review Agent",
		"claude-code",
		"access policy reviewer",
		"restricted query policy review",
		"owner",
		"purpose",
	)

	assert agent["runtime"] == "claude_code"
	assert agent["role"] == "access_policy_reviewer"
	assert agent["status"] == "pending_review"
	assert service.dashboard_summary(tenant_id)["pending_agent_review_count"] == 1

	with pytest.raises(ValueError, match="srch_lifecycle_batch_empty"):
		service.validate_srch_lifecycle_batch(tenant_id, "bytewax", 0)
	with pytest.raises(ValueError, match="unsupported_srch_lifecycle_operation"):
		service.validate_srch_lifecycle_batch(tenant_id, "bytewax", 1, "unknown_batch")
	with pytest.raises(PermissionError, match="bytewax_lifecycle_stream_required"):
		service.validate_srch_lifecycle_batch(tenant_id, "legacy_queue", 1)

	assert service.list_lifecycle_batches(tenant_id)[0]["status"] == "denied"
	assert service.dashboard_summary(tenant_id)["denied_lifecycle_batch_count"] == 1


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
	agent = api.register_search_agent({
		"id": "api-agent",
		"tenant_id": "tenant-b",
		"name": "API Search Agent",
		"runtime": "opencode",
		"role": "search_steward",
		"scope": "policy search review",
		"owner": "policy-owner",
		"purpose": "govern policy search lifecycle changes",
	})
	batch = api.validate_srch_lifecycle_batch({
		"id": "api-batch",
		"tenant_id": "tenant-b",
		"event_stream": "bytewax",
		"mutation_count": 1,
		"operation": "search_agent_batch",
	})

	status = api.capability_status("tenant-b")
	search_engine = api.list_search_engine("tenant-b")
	settings = views.settings_model("tenant-b")

	assert search_result["query"]["status"] == "completed"
	assert agent["runtime"] == "opencode"
	assert batch["status"] == "accepted"
	assert api.list_search_agents("tenant-b")[0]["id"] == "api-agent"
	assert api.list_lifecycle_batches("tenant-b")[0]["id"] == "api-batch"
	assert status["index_count"] == 2
	assert status["search_agent_count"] == 1
	assert search_engine["summary"]["document_count"] == 2
	assert search_engine["summary"]["lifecycle_batch_count"] == 1
	assert settings["theme"]["name"] == "srch_discovery_console"
