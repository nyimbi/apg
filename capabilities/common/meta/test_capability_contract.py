"""Regression coverage for the META executable capability contract."""

import pytest

from capabilities.common.meta import register_capability
from capabilities.common.meta.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)
from capabilities.common.meta.service import MetaService
from capabilities.common.meta.view_models import (
	adapter_health_model,
	asset_catalog_model,
	catalog_agent_roster_model,
	classification_review_model,
	dashboard_model,
	glossary_model,
	lifecycle_batch_model,
	settings_model
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-catalog",
		{"quality": {"minimum_certification_score": 92.0}}
	)

	assert contract["capability"] == "meta"
	assert contract["configuration"]["tenant_id"] == "tenant-catalog"
	assert contract["configuration"]["quality"]["minimum_certification_score"] == 92.0
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"catalog",
		"discovery",
		"classification",
		"lineage",
		"quality",
		"governance",
		"adapters",
		"agents",
		"streaming",
		"ui",
		"theme"
	]
	assert contract["provides"] == ["metadata_catalog_governance", "metadata_lifecycle", "catalog_agent_composition"]
	assert contract["requires"] == ["mdm", "auth", "audl"]
	assert contract["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert "classification_reviewer" in contract["agents"]["privileged_roles"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert contract["streaming"]["broker_core_dependency_allowed"] is False
	assert len(contract["rule_engine"]["rules"]) >= 25
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"catalog",
		"discovery",
		"lineage",
		"classification",
		"quality",
		"certification",
		"glossary",
		"impact",
		"search",
		"audit",
		"adapters",
		"agents",
		"lifecycle",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/meta/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "lineage_graph_viewer" in contract["theme"]["components"]
	assert "catalog_agent_roster" in contract["theme"]["components"]
	assert "bytewax_lifecycle_panel" in contract["theme"]["components"]
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"


def test_rule_engine_enforces_metadata_governance_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "publish_asset",
		"unsupported_asset_type": True,
		"business_key_present": False,
		"source_system_present": False,
		"asset_owner_assigned": False,
		"quality_assessment_present": False,
		"asset_sensitivity": "restricted",
		"classification_complete": False,
		"steward_assigned": False,
		"certification_requested": True,
		"lineage_available": False,
		"classification_confidence": 0.5,
		"steward_review_recorded": False,
		"asset_age_days": 120,
		"freshness_review_recorded": False,
		"term_owner_assigned": False,
		"source_and_target_registered": False,
		"connector_approved": False,
		"schedule_review_current": False,
		"impact_analysis_present": False,
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"published_asset_requires_owner",
		"publish_requires_quality_assessment",
		"restricted_asset_requires_classification",
		"sensitive_asset_requires_steward",
		"certified_asset_requires_lineage",
		"low_classification_confidence_requires_review",
		"stale_asset_requires_review"
	}


def test_rule_engine_enforces_catalog_agent_and_bytewax_guardrails():
	agent = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_catalog_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"agent_scope_present": False,
		"agent_owner_present": False,
		"agent_purpose_present": False,
		"contribution_disclosed": False,
		"privileged_agent_role": True,
		"human_approval_required": False,
	})
	batch = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "validate_meta_lifecycle_batch",
		"event_stream": "legacy_broker",
	})

	assert agent["decision"] == "deny"
	assert {
		"catalog_agent_runtime_supported",
		"catalog_agent_role_supported",
		"catalog_agent_requires_scope",
		"catalog_agent_requires_owner",
		"catalog_agent_requires_purpose",
		"catalog_agent_requires_contribution_disclosure",
		"catalog_agent_privileged_role_requires_human_approval",
	} <= set(agent["matched_rules"])
	assert batch["decision"] == "deny"
	assert "bytewax_meta_stream_required" in batch["matched_rules"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "meta_catalog_console"
	assert registration["ui_components"]["lineage"] == "/meta/lineage"
	assert registration["ui_components"]["agents"] == "/meta/agents"
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["required_processor"] == "bytewax"
	assert "mdm" in registration["dependencies"]


def test_meta_service_lifecycle_enforces_publication_and_classification_guardrails():
	service = MetaService()
	asset = service.register_asset(
		tenant_id="tenant-catalog",
		asset_id="asset-customers",
		asset_type="table",
		name="customers",
		business_key="warehouse.public.customers",
		source_system="warehouse",
		owner="data-owner",
		steward="data-steward",
		sensitivity="restricted",
	)
	low_confidence = service.classify_asset(
		tenant_id="tenant-catalog",
		asset_id=asset.asset_id,
		label="pii",
		confidence=0.52,
		classification_complete=False,
	)
	blocked_publish = service.publish_asset(
		tenant_id="tenant-catalog",
		asset_id=asset.asset_id,
	)
	blocked_publish_initial_status = blocked_publish.status
	blocked_publish_initial_rules = list(blocked_publish.matched_rules)
	low_confidence_initial_status = low_confidence.status
	low_confidence_initial_rules = list(low_confidence.matched_rules)
	review = service.review_classification(
		classification_id=low_confidence.classification_id,
		steward="data-steward",
		review_notes="Confirmed customer PII columns.",
	)
	quality = service.assess_quality(
		tenant_id="tenant-catalog",
		asset_id=asset.asset_id,
		score=91.0,
		dimensions={
			"completeness": 95.0,
			"freshness": 90.0,
			"accuracy": 91.0,
			"lineage": 88.0,
			"classification": 96.0,
			"usage": 87.0,
		},
		assessor="quality-engine",
	)
	classification = service.classify_asset(
		tenant_id="tenant-catalog",
		asset_id=asset.asset_id,
		label="pii",
		confidence=0.96,
		classification_complete=True,
		steward_review_recorded=True,
	)
	published = service.publish_asset(
		tenant_id="tenant-catalog",
		asset_id=asset.asset_id,
	)
	with pytest.raises(PermissionError, match="catalog_agent_human_approval_required"):
		service.register_catalog_agent(
			tenant_id="tenant-catalog",
			agent_id="agent-denied",
			name="Denied Classification Agent",
			runtime="codex",
			role="classification_reviewer",
			scope="restricted classifications",
			owner="metadata-office",
			purpose="review sensitive classifications",
		)
	assert any(event.event_type == "agent.registration_denied" for event in service.audit_events)
	agent = service.register_catalog_agent(
		tenant_id="tenant-catalog",
		agent_id="agent-certify",
		name="Certification Reviewer",
		runtime="Claude Code",
		role="certification reviewer",
		scope="metadata certification",
		owner="metadata-office",
		purpose="review certification evidence",
		human_approval_required=True,
	)
	batch = service.validate_meta_lifecycle_batch(
		tenant_id="tenant-catalog",
		event_stream="bytewax",
		mutation_count=6,
	)

	assert low_confidence_initial_status == "pending_review"
	assert low_confidence_initial_rules == ["low_classification_confidence_requires_review"]
	assert blocked_publish_initial_status == "draft"
	assert "publish_requires_quality_assessment" in blocked_publish_initial_rules
	assert review.status == "reviewed"
	assert quality.status == "accepted"
	assert classification.status == "accepted"
	assert published.status == "published"
	assert agent.runtime == "claude_code"
	assert agent.role == "certification_reviewer"
	assert batch.accepted is True
	assert batch.required_processor == "bytewax"
	assert service.dashboard_summary("tenant-catalog")["catalog_agent_count"] == 1


def test_meta_service_discovery_lineage_certification_and_retirement_guardrails():
	service = MetaService()
	source = service.register_asset(
		tenant_id="tenant-catalog",
		asset_id="pipeline-1",
		asset_type="pipeline",
		name="Customer sync",
		business_key="pipeline.customer.sync",
		source_system="orchestrator",
		owner="platform-owner",
		steward="platform-steward",
	)
	target = service.register_asset(
		tenant_id="tenant-catalog",
		asset_id="table-1",
		asset_type="table",
		name="Customer mart",
		business_key="warehouse.customer_mart",
		source_system="warehouse",
		owner="analytics-owner",
		steward="analytics-steward",
	)
	denied_discovery = service.schedule_discovery(
		tenant_id="tenant-catalog",
		connector_type="database",
		source_system="warehouse",
		schedule="0 2 * * *",
		connector_approved=False,
		schedule_review_current=True,
	)
	review_lineage = service.capture_lineage(
		tenant_id="tenant-catalog",
		source_asset_id=source.asset_id,
		target_asset_id=target.asset_id,
		lineage_type="transforms",
		depth=12,
		evidence="lineage-run-1",
	)
	service.assess_quality(
		tenant_id="tenant-catalog",
		asset_id=target.asset_id,
		score=93.0,
		dimensions={"completeness": 93.0},
		assessor="quality-engine",
	)
	blocked_certification = service.request_certification(
		tenant_id="tenant-catalog",
		asset_id=target.asset_id,
		requester="analytics-owner",
	)
	denied_retire = service.retire_asset(
		tenant_id="tenant-catalog",
		asset_id=target.asset_id,
		impact_analysis_present=False,
		actor="analytics-owner",
	)

	assert denied_discovery.status == "denied"
	assert denied_discovery.matched_rules == ["discovery_requires_approved_connector"]
	with pytest.raises(ValueError, match="cannot record results"):
		service.record_discovery_result(
			job_id=denied_discovery.job_id,
			discovered_asset_ids=[target.asset_id],
		)
	assert review_lineage.status == "pending_review"
	assert review_lineage.matched_rules == ["lineage_depth_requires_review"]
	assert blocked_certification.status == "denied"
	assert blocked_certification.matched_rules == ["certified_asset_requires_lineage"]
	assert denied_retire.status != "retired"
	assert denied_retire.matched_rules == ["retire_asset_requires_impact_analysis"]
	with pytest.raises(PermissionError, match="unsupported_catalog_agent_runtime"):
		service.register_catalog_agent(
			tenant_id="tenant-catalog",
			agent_id="bad-agent",
			name="Bad Agent",
			runtime="unsupported",
			role="metadata_steward_reviewer",
			scope="metadata review",
			owner="metadata-office",
			purpose="review catalog entries",
		)
	with pytest.raises(PermissionError, match="bytewax_meta_stream_required"):
		service.validate_meta_lifecycle_batch(
			tenant_id="tenant-catalog",
			event_stream="legacy_broker",
			mutation_count=1,
		)


def test_meta_service_glossary_and_view_models_are_composable():
	service = MetaService()
	asset = service.register_asset(
		tenant_id="tenant-catalog",
		asset_id="api-1",
		asset_type="api",
		name="Customer profile API",
		business_key="api.customer.profile",
		source_system="customer-platform",
		owner="api-owner",
		steward="api-steward",
	)
	denied_term = service.register_glossary_term(
		tenant_id="tenant-catalog",
		term="Customer",
		definition="A party buying goods or services.",
		owner=None,
		linked_asset_ids=[asset.asset_id],
	)
	approved_term = service.register_glossary_term(
		tenant_id="tenant-catalog",
		term="Customer profile",
		definition="The curated metadata definition for customer identity attributes.",
		owner="data-governance",
		linked_asset_ids=[asset.asset_id],
	)
	service.register_catalog_agent(
		tenant_id="tenant-catalog",
		agent_id="glossary-agent",
		name="Glossary Reviewer",
		runtime="opencode",
		role="glossary_reviewer",
		scope="business glossary",
		owner="metadata-office",
		purpose="review term ownership",
	)
	service.validate_meta_lifecycle_batch(
		tenant_id="tenant-catalog",
		event_stream="bytewax",
		mutation_count=4,
	)

	assert denied_term.status == "denied"
	assert denied_term.matched_rules == ["glossary_term_requires_owner"]
	assert approved_term.status == "active"
	assert dashboard_model(service, "tenant-catalog")["summary"]["asset_count"] == 1
	assert asset_catalog_model(service, "tenant-catalog")["columns"][0] == "asset_id"
	assert classification_review_model(service, "tenant-catalog")["review_actions"] == ["accept", "correct", "defer"]
	assert glossary_model(service, "tenant-catalog")["rows"][0]["term"] == "Customer"
	assert catalog_agent_roster_model(service, "tenant-catalog")["rows"][0]["role"] == "glossary_reviewer"
	assert lifecycle_batch_model(service, "tenant-catalog")["streaming"]["required_processor"] == "bytewax"
	assert adapter_health_model("tenant-catalog")["event_stream"] == "bytewax"
	assert settings_model("tenant-catalog")["configuration"]["tenant_id"] == "tenant-catalog"
