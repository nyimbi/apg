"""Regression coverage for the KNGR executable capability contract."""

import pytest

from capabilities.common.kngr import api
from capabilities.common.kngr import register_capability
from capabilities.common.kngr.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.kngr.service import KngrService
from capabilities.common.kngr.views import (
	audit_timeline_model,
	context_explorer_model,
	curation_queue_model,
	dashboard_model,
	enrichment_console_model,
	entity_browser_model,
	governance_model,
	knowledge_agent_roster_model,
	lifecycle_batch_model,
	publication_model,
	reasoning_paths_model,
	relationship_browser_model,
	settings_model,
	source_manager_model,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-knowledge", {"reasoning": {"max_reasoning_depth": 3}})

	assert contract["capability"] == "kngr"
	assert contract["configuration"]["tenant_id"] == "tenant-knowledge"
	assert contract["configuration"]["reasoning"]["max_reasoning_depth"] == 3
	assert set(contract["configuration_schema"]["required"]) >= {"tenant_id", "sources", "entities", "relationships", "reasoning", "adapters", "ui", "theme"}
	assert set(contract["configuration_schema"]["required"]) >= {"agents", "streaming"}
	assert len(contract["rule_engine"]["rules"]) >= 45
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "sources", "entities", "relationships", "enrichment", "curation", "publication", "reasoning", "context", "governance", "agents", "lifecycle", "audit", "settings"}
	assert contract["provides"] == ["knowledge_graph", "semantic_context", "knowledge_agent_composition"]
	assert contract["requires"] == ["grph", "nlpc", "meta", "srch", "onto", "aicr", "conf"]
	assert contract["agents"]["first_class"] is True
	assert contract["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert "publication_reviewer" in contract["agents"]["privileged_roles"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert "knowledge_agent_batch" in contract["streaming"]["required_operations"]
	assert contract["ui"]["api_prefix"] == "/kngr/api/v1"
	assert contract["ui"]["view_module"] == "views.py"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "service.KngrService"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "semantic_graph" in contract["theme"]["components"]
	assert "knowledge_agent_roster" in contract["theme"]["components"]


def test_rule_engine_enforces_knowledge_graph_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "reason",
		"query_present": True,
		"entity_endpoints_present": True,
		"evidence_links_present": False,
		"reasoning_depth": 9,
		"review_recorded": False,
		"curation_recorded": False
	})
	publish_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "publish_graph",
		"curation_recorded": False
	})
	agent_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_knowledge_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"scope_present": False,
		"owner_present": False,
		"purpose_present": False,
		"contribution_disclosed": False,
		"privileged_role": True,
		"human_approval_required": False,
	})
	lifecycle_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "validate_kngr_lifecycle_batch",
		"event_stream": "kafka",
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"reasoning_requires_evidence",
		"deep_reasoning_requires_review"
	}
	assert publish_result["decision"] == "deny"
	assert publish_result["matched_rules"] == ["uncurated_public_graph_blocked"]
	assert agent_result["decision"] == "deny"
	assert {
		"knowledge_agent_runtime_supported",
		"knowledge_agent_role_supported",
		"knowledge_agent_requires_scope",
		"knowledge_agent_requires_owner",
		"knowledge_agent_requires_purpose",
		"knowledge_agent_requires_contribution_disclosure",
		"knowledge_agent_privileged_role_requires_human_approval",
	} <= set(agent_result["matched_rules"])
	assert lifecycle_result["decision"] == "deny"
	assert lifecycle_result["matched_rules"] == ["bytewax_kngr_stream_required"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "kngr"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "kngr_semantic_console"
	assert registration["ui_components"]["reasoning"] == "/kngr/reasoning"
	assert registration["ui_components"]["agents"] == "/kngr/agents"
	assert registration["ui_components"]["lifecycle"] == "/kngr/lifecycle"
	assert "grph" in registration["dependencies"]
	assert "aicr" in registration["dependencies"]
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["required_processor"] == "bytewax"
	assert registration["capabilities"]["graph_publication"]
	assert registration["capabilities"]["knowledge_agent_composition"]
	assert registration["capabilities"]["knowledge_lifecycle_batches"]
	assert registration["endpoints"]["audit"] == "/kngr/api/v1/audit"
	assert "kngr:reason" in registration["permissions"]
	assert "kngr:audit" in registration["permissions"]


def test_kngr_lifecycle_is_executable():
	service = KngrService()

	source = service.register_source(
		source_id="src-procurement",
		tenant_id="tenant-knowledge",
		name="Procurement source",
		source_uri="meta://procurement/events",
		owner="knowledge-steward",
		evidence_refs=["meta:source:procurement"],
		confidence_score=0.94,
		connector="meta",
	)
	request = service.resolve_entity(
		entity_id="entity-request",
		tenant_id="tenant-knowledge",
		canonical_label="Purchase request 1001",
		entity_type="purchase_request",
		source_id=source["id"],
		source_evidence_refs=["doc:pr-1001"],
		aliases=["PR-1001"],
		attributes={"amount": 9500},
		confidence_score=0.91,
	)
	supplier = service.resolve_entity(
		entity_id="entity-supplier",
		tenant_id="tenant-knowledge",
		canonical_label="Acme Supplies",
		entity_type="supplier",
		source_id=source["id"],
		source_evidence_refs=["doc:supplier-acme"],
		confidence_score=0.88,
	)
	relationship = service.link_relationship(
		relationship_id="rel-request-supplier",
		tenant_id="tenant-knowledge",
		subject_entity_id=request["id"],
		predicate="uses_supplier",
		object_entity_id=supplier["id"],
		source_id=source["id"],
		evidence_links=["doc:pr-1001"],
		confidence_score=0.89,
	)
	enrichment = service.enrich_entity(
		enrichment_id="enrich-request-risk",
		tenant_id="tenant-knowledge",
		entity_id=request["id"],
		semantic_labels=["procurement", "approval_required"],
		attributes={"risk_band": "medium"},
		evidence_links=["nlpc:pr-1001"],
		confidence_score=0.86,
	)
	reasoning = service.build_reasoning_path(
		path_id="path-request-supplier",
		tenant_id="tenant-knowledge",
		query="Why does this purchase request require supplier review?",
		start_entity_id=request["id"],
		end_entity_id=supplier["id"],
		relationship_ids=[relationship["id"]],
		evidence_links=["doc:pr-1001", "doc:supplier-acme"],
	)
	curation = service.curate_entity(
		curation_id="curate-request",
		tenant_id="tenant-knowledge",
		entity_id=request["id"],
		curator="knowledge-steward",
		decision="approved",
		evidence_links=["review:curation-1"],
	)
	service.curate_entity(
		curation_id="curate-supplier",
		tenant_id="tenant-knowledge",
		entity_id=supplier["id"],
		curator="knowledge-steward",
		decision="approved",
		evidence_links=["review:curation-2"],
	)
	publication = service.publish_graph(
		publication_id="pub-procurement",
		tenant_id="tenant-knowledge",
		name="Procurement knowledge graph",
		entity_ids=[request["id"], supplier["id"]],
		relationship_ids=[relationship["id"]],
		published_by="knowledge-steward",
		curation_recorded=True,
	)
	agent = service.register_knowledge_agent(
		agent_id="knowledge-agent-1",
		tenant_id="tenant-knowledge",
		name="Knowledge Steward",
		runtime="codex",
		role="knowledge_steward",
		scope="source entity relationship enrichment reasoning curation publication review",
		owner="knowledge-platform",
		purpose="govern curated procurement knowledge graph construction",
	)
	batch = service.validate_kngr_lifecycle_batch(
		tenant_id="tenant-knowledge",
		event_stream="bytewax",
		mutation_count=4,
		operation="knowledge_agent_batch",
		batch_id="kngr-batch-001",
	)

	assert request["curation_status"] == "draft"
	assert relationship["status"] == "active"
	assert enrichment["status"] == "active"
	assert reasoning["reasoning_depth"] == 1
	assert curation["decision"] == "approved"
	assert publication["status"] == "published"
	assert agent["runtime"] == "codex"
	assert agent["status"] == "active"
	assert batch["required_processor"] == "bytewax"
	assert service.dashboard_summary("tenant-knowledge")["entity_count"] == 2
	assert service.dashboard_summary("tenant-knowledge")["knowledge_agent_count"] == 1
	assert service.dashboard_summary("tenant-knowledge")["lifecycle_batch_count"] == 1
	assert service.context_neighborhood("tenant-knowledge", request["id"])["neighbor_count"] == 1
	assert dashboard_model(service, "tenant-knowledge")["summary"]["publication_count"] == 1
	assert entity_browser_model(service, "tenant-knowledge")["relationships"][0]["predicate"] == "uses_supplier"
	assert source_manager_model(service, "tenant-knowledge")["sources"][0]["id"] == "src-procurement"
	assert relationship_browser_model(service, "tenant-knowledge")["relationships"][0]["id"] == "rel-request-supplier"
	assert enrichment_console_model(service, "tenant-knowledge")["enrichments"][0]["id"] == "enrich-request-risk"
	assert curation_queue_model(service, "tenant-knowledge")["pending_entities"] == []
	assert reasoning_paths_model(service, "tenant-knowledge")["reasoning_paths"][0]["id"] == "path-request-supplier"
	assert context_explorer_model(service, "tenant-knowledge", request["id"])["neighborhood"]["relationship_count"] == 1
	assert governance_model(service, "tenant-knowledge")["publications"][0]["id"] == "pub-procurement"
	assert governance_model(service, "tenant-knowledge")["agents"]["first_class"] is True
	assert knowledge_agent_roster_model(service, "tenant-knowledge")["agents"][0]["id"] == "knowledge-agent-1"
	assert lifecycle_batch_model(service, "tenant-knowledge")["batches"][0]["id"] == "kngr-batch-001"
	assert publication_model(service, "tenant-knowledge")["publications"][0]["id"] == "pub-procurement"
	assert audit_timeline_model(service, "tenant-knowledge")["audit_events"]
	assert settings_model(service, "tenant-knowledge")["adapters"]["event_stream"] == "bytewax"
	assert settings_model(service, "tenant-knowledge")["streaming"]["required_processor"] == "bytewax"
	assert service.list_knowledge_graph("tenant-knowledge")["summary"]["publication_count"] == 1
	assert service.list_knowledge_graph("tenant-knowledge")["summary"]["knowledge_agent_count"] == 1
	assert service.list_knowledge_graph()["summary"]["entity_count"] == 2
	assert len(service.list_audit_events("tenant-knowledge")) >= 8


def test_kngr_service_enforces_policy_guardrails():
	service = KngrService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_source(
			source_id="src-missing-tenant",
			tenant_id="",
			name="Missing tenant",
			source_uri="meta://missing",
			owner="owner",
			evidence_refs=["evidence"],
			confidence_score=1.0,
		)

	with pytest.raises(PermissionError, match="source_owner_required"):
		service.register_source(
			source_id="src-no-owner",
			tenant_id="tenant-knowledge",
			name="No owner",
			source_uri="meta://source",
			owner="",
			evidence_refs=["evidence"],
			confidence_score=1.0,
		)

	with pytest.raises(PermissionError, match="low_confidence_source_review_required"):
		service.register_source(
			source_id="src-low-confidence",
			tenant_id="tenant-knowledge",
			name="Low confidence",
			source_uri="meta://low",
			owner="steward",
			evidence_refs=["meta:low"],
			confidence_score=0.4,
			review_recorded=False,
		)

	service.register_source(
		source_id="src-knowledge",
		tenant_id="tenant-knowledge",
		name="Knowledge source",
		source_uri="meta://knowledge",
		owner="steward",
		evidence_refs=["meta:source"],
		confidence_score=0.9,
	)

	with pytest.raises(PermissionError, match="source_evidence_required"):
		service.resolve_entity(
			entity_id="entity-no-evidence",
			tenant_id="tenant-knowledge",
			canonical_label="No evidence",
			entity_type="document",
			source_id="src-knowledge",
			source_evidence_refs=[],
		)

	entity_a = service.resolve_entity(
		entity_id="entity-a",
		tenant_id="tenant-knowledge",
		canonical_label="Entity A",
		entity_type="concept",
		source_id="src-knowledge",
		source_evidence_refs=["doc:a"],
	)
	entity_b = service.resolve_entity(
		entity_id="entity-b",
		tenant_id="tenant-knowledge",
		canonical_label="Entity B",
		entity_type="concept",
		source_id="src-knowledge",
		source_evidence_refs=["doc:b"],
	)

	with pytest.raises(PermissionError, match="low_confidence_entity_review_required"):
		service.resolve_entity(
			entity_id="entity-low-confidence",
			tenant_id="tenant-knowledge",
			canonical_label="Entity low",
			entity_type="concept",
			source_id="src-knowledge",
			source_evidence_refs=["doc:low"],
			confidence_score=0.4,
			review_recorded=False,
		)

	with pytest.raises(PermissionError, match="low_confidence_enrichment_review_required"):
		service.enrich_entity(
			enrichment_id="enrich-low",
			tenant_id="tenant-knowledge",
			entity_id=entity_a["id"],
			semantic_labels=["low_confidence"],
			attributes={},
			evidence_links=["nlpc:a"],
			confidence_score=0.4,
			review_recorded=False,
		)

	reviewed_relationship = service.link_relationship(
		relationship_id="rel-reviewed",
		tenant_id="tenant-knowledge",
		subject_entity_id=entity_a["id"],
		predicate="related_to",
		object_entity_id=entity_b["id"],
		source_id="src-knowledge",
		evidence_links=["doc:edge"],
		confidence_score=0.4,
		review_recorded=True,
	)
	assert reviewed_relationship["status"] == "accepted_with_review"

	with pytest.raises(PermissionError, match="reasoning_evidence_required"):
		service.build_reasoning_path(
			path_id="path-no-evidence",
			tenant_id="tenant-knowledge",
			query="Why?",
			start_entity_id=entity_a["id"],
			end_entity_id=entity_b["id"],
			relationship_ids=[reviewed_relationship["id"]],
			evidence_links=[],
		)

	deep_relationship_ids = []
	for index in range(6):
		relationship = service.link_relationship(
			relationship_id=f"rel-deep-{index}",
			tenant_id="tenant-knowledge",
			subject_entity_id=entity_a["id"],
			predicate=f"related_to_{index}",
			object_entity_id=entity_b["id"],
			source_id="src-knowledge",
			evidence_links=[f"doc:deep:{index}"],
			confidence_score=0.9,
		)
		deep_relationship_ids.append(relationship["id"])

	with pytest.raises(PermissionError, match="deep_reasoning_review_required"):
		service.build_reasoning_path(
			path_id="path-deep",
			tenant_id="tenant-knowledge",
			query="Deep path",
			start_entity_id=entity_a["id"],
			end_entity_id=entity_b["id"],
			relationship_ids=deep_relationship_ids,
			evidence_links=["doc:deep"],
			review_recorded=False,
		)

	with pytest.raises(PermissionError, match="curation_required"):
		service.publish_graph(
			publication_id="pub-no-curation",
			tenant_id="tenant-knowledge",
			name="No curation",
			entity_ids=[entity_a["id"]],
			relationship_ids=[],
			published_by="steward",
			curation_recorded=False,
		)

	with pytest.raises(KeyError, match="knowledge_entity_not_found"):
		service.context_neighborhood("another-tenant", entity_a["id"])


def test_knowledge_agent_and_lifecycle_guardrails_execute():
	service = KngrService()
	tenant_id = "tenant-agents"

	with pytest.raises(PermissionError, match="unsupported_knowledge_agent_runtime"):
		service.register_knowledge_agent(
			"agent-bad-runtime",
			tenant_id,
			"Bad Runtime",
			"unknown",
			"knowledge_steward",
			"entity review",
			"owner",
			"purpose",
		)

	with pytest.raises(PermissionError, match="knowledge_agent_scope_required"):
		service.register_knowledge_agent("agent-no-scope", tenant_id, "No Scope", "codex", "knowledge_steward", "", "owner", "purpose")

	with pytest.raises(PermissionError, match="knowledge_agent_contribution_disclosure_required"):
		service.register_knowledge_agent(
			"agent-no-disclosure",
			tenant_id,
			"No Disclosure",
			"codex",
			"knowledge_steward",
			"entity review",
			"owner",
			"purpose",
			contribution_disclosed=False,
		)

	agent = service.register_knowledge_agent(
		"agent-privileged",
		tenant_id,
		"Privileged Agent",
		"claude-code",
		"publication reviewer",
		"publication and reasoning decisions",
		"owner",
		"review privileged knowledge graph decisions",
		human_approval_required=False,
	)
	assert agent["runtime"] == "claude_code"
	assert agent["role"] == "publication_reviewer"
	assert agent["status"] == "pending_review"
	assert service.dashboard_summary(tenant_id)["pending_agent_review_count"] == 1

	with pytest.raises(ValueError, match="kngr_lifecycle_batch_empty"):
		service.validate_kngr_lifecycle_batch(tenant_id, "bytewax", 0)

	with pytest.raises(ValueError, match="unsupported_kngr_lifecycle_operation"):
		service.validate_kngr_lifecycle_batch(tenant_id, "bytewax", 1, "unknown_batch")

	with pytest.raises(PermissionError, match="bytewax_lifecycle_stream_required"):
		service.validate_kngr_lifecycle_batch(tenant_id, "kafka", 1)

	assert service.dashboard_summary(tenant_id)["denied_lifecycle_batch_count"] == 1


def test_api_helpers_expose_agent_and_lifecycle_surfaces():
	local_service = KngrService()
	api.SERVICE = local_service

	source = api.register_source({
		"id": "src-api",
		"tenant_id": "tenant-api",
		"name": "API source",
		"source_uri": "meta://api",
		"owner": "steward",
		"evidence_refs": ["meta:api"],
		"confidence_score": 0.9,
	})
	entity = api.resolve_entity({
		"id": "entity-api",
		"tenant_id": "tenant-api",
		"canonical_label": "API Entity",
		"entity_type": "concept",
		"source_id": source["id"],
		"source_evidence_refs": ["doc:api"],
	})
	agent = api.register_knowledge_agent({
		"id": "api-agent",
		"tenant_id": "tenant-api",
		"name": "API Knowledge Agent",
		"runtime": "opencode",
		"role": "enrichment_reviewer",
		"scope": "semantic enrichment review",
		"owner": "steward",
		"purpose": "review semantic enrichment quality",
	})
	batch = api.validate_kngr_lifecycle_batch({
		"id": "api-batch",
		"tenant_id": "tenant-api",
		"event_stream": "bytewax",
		"mutation_count": 2,
		"operation": "entity_batch",
	})
	status = api.capability_status("tenant-api")
	graph = api.list_knowledge_graph("tenant-api")

	assert entity["id"] == "entity-api"
	assert agent["id"] == "api-agent"
	assert batch["status"] == "accepted"
	assert api.list_knowledge_agents("tenant-api")[0]["id"] == "api-agent"
	assert api.list_lifecycle_batches("tenant-api")[0]["id"] == "api-batch"
	assert status["agent_count"] == 1
	assert status["lifecycle_batch_count"] == 1
	assert graph["summary"]["knowledge_agent_count"] == 1
