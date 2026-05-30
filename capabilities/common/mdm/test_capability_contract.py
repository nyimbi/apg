"""Regression coverage for the MDM executable capability contract."""

from capabilities.common.mdm import register_capability
from capabilities.common.mdm.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)
from capabilities.common.mdm.service import MdmService
from capabilities.common.mdm.view_models import (
	adapter_health_model,
	dashboard_model,
	duplicate_review_model,
	entity_workbench_model,
	publish_model,
	settings_model
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-master",
		{"quality": {"minimum_quality_score": 88.0}}
	)

	assert contract["capability"] == "mdm"
	assert contract["configuration"]["tenant_id"] == "tenant-master"
	assert contract["configuration"]["quality"]["minimum_quality_score"] == 88.0
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"entities",
		"quality",
		"matching",
		"survivorship",
		"governance",
		"integration",
		"adapters",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 15
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"entities",
		"golden_records",
		"quality",
		"duplicates",
		"stewardship",
		"lineage",
		"cross_references",
		"publish",
		"analytics",
		"audit",
		"adapters",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/mdm/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "golden_record_card" in contract["theme"]["components"]
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"


def test_rule_engine_enforces_master_data_guardrails():
	publish_result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "publish_entity",
		"data_owner_assigned": False,
		"latest_quality_assessment_present": False,
		"quality_score": 40.0,
		"duplicate_confidence": 82.0,
		"steward_review_recorded": False,
		"entity_classification": "restricted",
		"audit_evidence_present": False
	})

	merge_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "merge_golden_record",
		"survivorship_policy_present": False
	})

	assert publish_result["decision"] == "deny"
	assert set(publish_result["matched_rules"]) == {
		"tenant_context_required",
		"entity_publish_requires_data_owner",
		"publish_requires_latest_quality_assessment",
		"low_quality_blocks_publish",
		"duplicate_candidates_require_review",
		"restricted_entity_requires_audit_trail"
	}
	assert merge_result["decision"] == "deny"
	assert merge_result["matched_rules"] == ["golden_record_merge_requires_survivorship"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "mdm_golden_record_console"
	assert registration["ui_components"]["duplicates"] == "/mdm/duplicates"
	assert "mten" in registration["dependencies"]


def test_mdm_service_lifecycle_enforces_publish_and_duplicate_guardrails():
	service = MdmService()
	entity = service.register_entity(
		tenant_id="tenant-master",
		entity_id="cust-1",
		entity_type="customer",
		name="Acme Limited",
		business_key="ACME-001",
		source_system="crm",
		data_owner="steward-a",
		classification="internal",
	)
	candidate = service.register_entity(
		tenant_id="tenant-master",
		entity_id="cust-2",
		entity_type="customer",
		name="ACME Ltd",
		business_key="ACME-002",
		source_system="billing",
		data_owner="steward-b",
		classification="internal",
	)
	duplicate = service.create_duplicate_candidate(
		tenant_id="tenant-master",
		entity_id=entity.entity_id,
		candidate_entity_id=candidate.entity_id,
		confidence=82.0,
		reason="similar name and address",
	)
	denied_publish = service.publish_entity(
		tenant_id="tenant-master",
		entity_id=entity.entity_id,
		channel="bytewax.entity_stream",
	)
	quality = service.assess_quality(
		tenant_id="tenant-master",
		entity_id=entity.entity_id,
		overall_score=92.0,
		dimensions={
			"completeness": 96.0,
			"accuracy": 91.0,
			"consistency": 90.0,
			"validity": 94.0,
			"uniqueness": 88.0,
			"timeliness": 93.0,
		},
		assessor="quality-engine",
	)
	approved_publish = service.publish_entity(
		tenant_id="tenant-master",
		entity_id=entity.entity_id,
		channel="bytewax.entity_stream",
	)

	assert duplicate.status == "review_required"
	assert duplicate.matched_rules == ["duplicate_candidates_require_review"]
	assert denied_publish.status == "denied"
	assert "publish_requires_latest_quality_assessment" in denied_publish.matched_rules
	assert quality.status == "accepted"
	assert approved_publish.status == "published"
	assert service.dashboard_summary("tenant-master")["published_entity_count"] == 1


def test_mdm_service_blocks_restricted_entities_without_evidence():
	service = MdmService()
	record = service.register_entity(
		tenant_id="tenant-master",
		entity_id="emp-1",
		entity_type="employee",
		name="Employee One",
		business_key="EMP-001",
		source_system="hr",
		data_owner=None,
		classification="restricted",
		attributes={"national_id": "classified"},
	)

	assert record.status == "denied"
	assert set(record.matched_rules) >= {
		"restricted_entity_requires_data_owner",
		"restricted_entity_requires_audit_trail",
		"restricted_entity_requires_classification_evidence",
	}


def test_mdm_service_golden_record_and_cross_reference_guardrails():
	service = MdmService()
	entity = service.register_entity(
		tenant_id="tenant-master",
		entity_id="supplier-1",
		entity_type="supplier",
		name="Supply Co",
		business_key="SUP-001",
		source_system="erp",
		data_owner="steward-a",
	)
	golden = service.create_golden_record(
		tenant_id="tenant-master",
		entity_type="supplier",
		source_entity_ids=[entity.entity_id],
		survivorship_policy="most_trusted_source",
	)
	blocked_merge = service.merge_golden_record(
		tenant_id="tenant-master",
		golden_record_id=golden.golden_record_id,
		source_entity_ids=[entity.entity_id],
		survivorship_policy=None,
	)
	review_merge = service.merge_golden_record(
		tenant_id="tenant-master",
		golden_record_id=golden.golden_record_id,
		source_entity_ids=[entity.entity_id],
		survivorship_policy="most_trusted_source",
		conflict_present=True,
		independent_steward=None,
	)
	blocked_mapping = service.update_cross_reference(
		tenant_id="tenant-master",
		entity_id=entity.entity_id,
		source_system="erp",
		source_identifier="SUP-001",
		evidence_reference=None,
	)

	assert blocked_merge.status == "denied"
	assert blocked_merge.matched_rules == ["golden_record_merge_requires_survivorship"]
	assert review_merge.status == "pending_review"
	assert review_merge.matched_rules == ["conflicted_merge_requires_independent_steward"]
	assert blocked_mapping.status == "denied"
	assert blocked_mapping.matched_rules == ["cross_reference_requires_source_evidence"]


def test_view_models_and_settings_are_composable():
	service = MdmService()
	service.register_entity(
		tenant_id="tenant-master",
		entity_id="product-1",
		entity_type="product",
		name="Product One",
		business_key="SKU-001",
		source_system="pim",
		data_owner="steward-a",
	)

	assert dashboard_model(service, "tenant-master")["summary"]["entity_count"] == 1
	assert entity_workbench_model(service, "tenant-master")["columns"][0] == "entity_id"
	assert duplicate_review_model(service, "tenant-master")["review_actions"] == ["merge", "keep_separate", "defer"]
	assert publish_model(service, "tenant-master")["columns"][3] == "decision"
	assert adapter_health_model("tenant-master")["event_stream"] == "bytewax"
	assert settings_model("tenant-master")["configuration"]["tenant_id"] == "tenant-master"
