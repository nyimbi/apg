"""Regression coverage for the ONTO executable capability contract."""

import pytest

from capabilities.common.onto import register_capability
from capabilities.common.onto.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.onto.service import OntoService
from capabilities.common.onto.views import (
	dashboard_model,
	governance_model,
	mapping_workbench_model,
	ontology_registry_model,
	publication_queue_model,
	taxonomy_model,
	term_editor_model,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-onto", {"mapping": {"confidence_threshold": 0.9}})

	assert contract["capability"] == "onto"
	assert contract["configuration"]["tenant_id"] == "tenant-onto"
	assert contract["configuration"]["mapping"]["confidence_threshold"] == 0.9
	assert contract["configuration_schema"]["required"] == ["tenant_id", "ontology", "vocabulary", "mapping", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "ontologies", "terms", "mappings", "publication", "governance", "settings"}
	assert contract["ui"]["api_prefix"] == "/onto/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "taxonomy_tree" in contract["theme"]["components"]


def test_rule_engine_enforces_ontology_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "publish_ontology",
		"owner_assigned": False,
		"approval_recorded": False,
		"change_type": "breaking",
		"review_recorded": False,
		"mapping_confidence": 0.2,
		"duplicate_term_detected": True
	})
	term_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "create_term",
		"owner_assigned": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"publication_requires_approval",
		"breaking_change_requires_review",
		"low_confidence_mapping_requires_review",
		"duplicate_term_blocks_publication"
	}
	assert term_result["decision"] == "deny"
	assert term_result["matched_rules"] == ["term_requires_owner"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "onto"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "onto_vocabulary_workbench"
	assert registration["ui_components"]["mappings"] == "/onto/mappings"
	assert "kngr" in registration["dependencies"]
	assert "onto:publish" in registration["permissions"]


def test_onto_lifecycle_is_executable():
	service = OntoService()
	tenant_id = "tenant-onto"

	ontology = service.register_ontology(
		ontology_id="customer-ontology",
		tenant_id=tenant_id,
		name="Customer Ontology",
		owner="data-stewards",
		domain="crm",
		description="Customer vocabulary for CRM apps",
	)
	customer = service.create_term(
		term_id="term-customer",
		tenant_id=tenant_id,
		ontology_id=ontology["id"],
		label="Customer",
		owner="data-stewards",
		definition="A party that purchases products or services.",
		synonyms=["Client"],
	)
	account = service.create_term(
		term_id="term-account",
		tenant_id=tenant_id,
		ontology_id=ontology["id"],
		label="Account",
		owner="data-stewards",
		definition="A commercial relationship with a customer.",
	)
	service.add_synonym(tenant_id, customer["id"], "Buyer")
	edge = service.add_taxonomy_edge(
		edge_id="edge-customer-account",
		tenant_id=tenant_id,
		ontology_id=ontology["id"],
		parent_term_id=customer["id"],
		child_term_id=account["id"],
	)
	service.curate_term(
		review_id="review-customer",
		tenant_id=tenant_id,
		term_id=customer["id"],
		reviewer="chief-steward",
		status="curated",
	)
	service.curate_term(
		review_id="review-account",
		tenant_id=tenant_id,
		term_id=account["id"],
		reviewer="chief-steward",
		status="curated",
	)
	mapping = service.create_mapping(
		mapping_id="map-customer-meta",
		tenant_id=tenant_id,
		term_id=customer["id"],
		target_ref="meta:party.customer",
		mapping_type="exact",
		confidence=0.96,
	)
	reviewed_mapping = service.create_mapping(
		mapping_id="map-account-external",
		tenant_id=tenant_id,
		term_id=account["id"],
		target_ref="external:sales.account",
		mapping_type="close",
		confidence=0.72,
		review_recorded=True,
		review_ref="review:account-map",
	)
	publication = service.publish_ontology(
		publication_id="publish-customer-ontology",
		tenant_id=tenant_id,
		ontology_id=ontology["id"],
		approval_recorded=True,
		approval_ref="approval:onto-42",
	)

	assert edge["relationship_type"] == "broader_than"
	assert mapping["status"] == "active"
	assert reviewed_mapping["status"] == "reviewed"
	assert publication["status"] == "published"
	assert publication["term_count"] == 2
	assert service.list_terms(tenant_id)[0]["status"] == "published"

	summary = service.dashboard_summary(tenant_id)
	assert summary["ontology_count"] == 1
	assert summary["term_count"] == 2
	assert summary["mapping_count"] == 2
	assert summary["publication_count"] == 1

	assert dashboard_model(service, tenant_id)["summary"]["term_count"] == 2
	assert ontology_registry_model(service, tenant_id)["ontologies"][0]["id"] == "customer-ontology"
	assert term_editor_model(service, tenant_id)["reviews"]
	assert taxonomy_model(service, tenant_id)["edges"][0]["id"] == "edge-customer-account"
	assert mapping_workbench_model(service, tenant_id)["mappings"][0]["term_id"] == "term-account"
	assert publication_queue_model(service, tenant_id)["publications"][0]["approval_recorded"] is True
	assert governance_model(service, tenant_id)["audit_events"]


def test_onto_service_enforces_policy_guardrails():
	service = OntoService()
	tenant_id = "tenant-guardrails"

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_ontology(
			ontology_id="missing-tenant",
			tenant_id="",
			name="Missing Tenant",
			owner="steward",
			domain="test",
		)

	ontology = service.register_ontology(
		ontology_id="guardrail-ontology",
		tenant_id=tenant_id,
		name="Guardrail Ontology",
		owner="steward",
		domain="test",
	)
	with pytest.raises(PermissionError, match="term_owner_required"):
		service.create_term(
			term_id="ownerless",
			tenant_id=tenant_id,
			ontology_id=ontology["id"],
			label="Ownerless",
			owner="",
		)

	parent = service.create_term(
		term_id="parent",
		tenant_id=tenant_id,
		ontology_id=ontology["id"],
		label="Parent",
		owner="steward",
	)
	child = service.create_term(
		term_id="child",
		tenant_id=tenant_id,
		ontology_id=ontology["id"],
		label="Child",
		owner="steward",
	)
	service.add_taxonomy_edge(
		edge_id="parent-child",
		tenant_id=tenant_id,
		ontology_id=ontology["id"],
		parent_term_id=parent["id"],
		child_term_id=child["id"],
	)
	with pytest.raises(PermissionError, match="taxonomy_cycle_detected"):
		service.add_taxonomy_edge(
			edge_id="child-parent",
			tenant_id=tenant_id,
			ontology_id=ontology["id"],
			parent_term_id=child["id"],
			child_term_id=parent["id"],
		)

	with pytest.raises(PermissionError, match="breaking_change_review_required"):
		service.curate_term(
			review_id="breaking-no-review",
			tenant_id=tenant_id,
			term_id=parent["id"],
			reviewer="steward",
			change_type="breaking",
			review_recorded=False,
		)

	with pytest.raises(PermissionError, match="mapping_review_required"):
		service.create_mapping(
			mapping_id="low-confidence",
			tenant_id=tenant_id,
			term_id=parent["id"],
			target_ref="meta:low-confidence",
			confidence=0.5,
		)

	service.curate_term(
		review_id="review-parent",
		tenant_id=tenant_id,
		term_id=parent["id"],
		reviewer="steward",
		status="curated",
	)
	service.curate_term(
		review_id="review-child",
		tenant_id=tenant_id,
		term_id=child["id"],
		reviewer="steward",
		status="curated",
	)
	duplicate = service.create_term(
		term_id="duplicate-parent",
		tenant_id=tenant_id,
		ontology_id=ontology["id"],
		label="Parent",
		owner="steward",
	)
	service.curate_term(
		review_id="review-duplicate",
		tenant_id=tenant_id,
		term_id=duplicate["id"],
		reviewer="steward",
		status="curated",
	)
	with pytest.raises(PermissionError, match="publication_approval_required"):
		service.publish_ontology(
			publication_id="publish-no-approval",
			tenant_id=tenant_id,
			ontology_id=ontology["id"],
			approval_recorded=False,
		)
	with pytest.raises(PermissionError, match="duplicate_term_detected"):
		service.publish_ontology(
			publication_id="publish-duplicates",
			tenant_id=tenant_id,
			ontology_id=ontology["id"],
			approval_recorded=True,
		)

	with pytest.raises(LookupError, match="ontology_term_not_found"):
		service.create_mapping(
			mapping_id="cross-tenant",
			tenant_id="other-tenant",
			term_id=parent["id"],
			target_ref="meta:cross",
			confidence=1.0,
		)
