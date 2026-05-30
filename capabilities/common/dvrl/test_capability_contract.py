"""Regression coverage for the DVRL executable capability contract."""

from capabilities.common.dvrl import register_capability
from capabilities.common.dvrl.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)
from capabilities.common.dvrl.service import DVRLLifecycleService
from capabilities.common.dvrl.view_models import (
	adapter_health_model,
	dashboard_model,
	federation_map_model,
	query_workbench_model,
	settings_model,
	source_manager_model,
	virtual_table_catalog_model,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-federation", {"queries": {"default_timeout_seconds": 120}})

	assert contract["capability"] == "dvrl"
	assert contract["configuration"]["tenant_id"] == "tenant-federation"
	assert contract["configuration"]["queries"]["default_timeout_seconds"] == 120
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"sources",
		"schemas",
		"queries",
		"cache",
		"governance",
		"optimization",
		"adapters",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 20
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"query",
		"sources",
		"schemas",
		"virtual_tables",
		"federation",
		"policies",
		"cache",
		"metrics",
		"adapters",
		"audit",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/dvrl/api/v1"
	assert contract["ui"]["view_module"] == "view_models.py"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "federation_map" in contract["theme"]["components"]


def test_rule_engine_enforces_virtualization_guardrails():
	query_result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "execute_query",
		"data_classification": "restricted",
		"rbac_authorized": False,
		"parameterized": False,
		"write_query": True,
		"result_contains_sensitive_data": True,
		"cache_requested": True,
		"lineage_capture_enabled": False,
		"estimated_query_cost": 2500.0,
		"cost_review_recorded": False,
		"join_source_count": 4,
		"join_review_recorded": False,
		"requested_rows": 250000
	})
	source_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_source",
		"source_owner_assigned": False,
		"unsupported_source_type": True,
		"credentials_vaulted": False
	})

	assert query_result["decision"] == "deny"
	assert set(query_result["matched_rules"]) >= {
		"tenant_context_required",
		"query_requires_parameterization",
		"write_query_blocked",
		"restricted_query_requires_rbac",
		"sensitive_results_block_cache",
		"query_requires_lineage_capture",
		"high_cost_query_requires_review",
		"cross_source_join_requires_review",
		"query_result_limit_enforced"
	}
	assert source_result["decision"] == "deny"
	assert set(source_result["matched_rules"]) == {
		"source_registration_requires_owner",
		"source_type_must_be_supported",
		"source_registration_requires_credentials",
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "dvrl_federation_console"
	assert registration["ui_components"]["query"] == "/dvrl/query"
	assert "etlp" in registration["dependencies"]


def test_lifecycle_service_enforces_source_schema_query_cache_and_policy_guardrails():
	service = DVRLLifecycleService()
	tenant_id = "tenant-federation"

	denied_source = service.register_source(
		tenant_id=tenant_id,
		source_id="src-denied",
		name="Denied Source",
		source_type="database",
		owner="",
		credentials_vaulted=False,
		connection_encrypted=False,
	)
	assert denied_source.status == "denied"
	assert set(denied_source.matched_rules) >= {
		"source_registration_requires_owner",
		"source_registration_requires_credentials",
		"source_connection_requires_encryption",
	}

	source = service.register_source(
		tenant_id=tenant_id,
		source_id="src-orders",
		name="Orders Warehouse",
		source_type="warehouse",
		owner="data-platform",
		credentials_vaulted=True,
		connection_encrypted=True,
	)
	assert source.status == "registered"

	pending_activation = service.activate_source(
		tenant_id=tenant_id,
		source_id="src-orders",
		approver="risk",
		source_approval_recorded=False,
	)
	assert pending_activation.status == "pending_review"
	assert pending_activation.matched_rules == ["source_activation_requires_approval"]

	active_source = service.activate_source(
		tenant_id=tenant_id,
		source_id="src-orders",
		approver="risk",
		source_approval_recorded=True,
	)
	assert active_source.status == "active"

	schema = service.refresh_schema(
		tenant_id=tenant_id,
		schema_id="schema-orders",
		source_id="src-orders",
		name="orders",
		schema_age_days=45,
		schema_review_recorded=False,
		tables=["orders", "customers"],
	)
	assert schema.status == "pending_review"
	assert schema.matched_rules == ["schema_refresh_requires_review"]

	rejected_table = service.publish_virtual_table(
		tenant_id=tenant_id,
		table_id="vt-rejected",
		source_id="src-orders",
		name="Rejected Orders",
		owner=None,
		classification=None,
		classification_complete=False,
	)
	assert rejected_table.status == "denied"
	assert set(rejected_table.matched_rules) == {
		"virtual_table_requires_owner",
		"virtual_table_requires_classification",
	}

	table = service.publish_virtual_table(
		tenant_id=tenant_id,
		table_id="vt-orders",
		source_id="src-orders",
		name="Orders",
		owner="analytics",
		classification="internal",
		classification_complete=True,
	)
	assert table.status == "published"

	blocked_query = service.execute_query(
		tenant_id=tenant_id,
		query_id="qry-blocked",
		sql="DELETE FROM orders",
		actor="analyst",
		source_ids=["src-orders"],
		data_classification="restricted",
		rbac_authorized=False,
		parameterized=False,
		write_query=True,
		lineage_capture_enabled=False,
		estimated_query_cost=2500.0,
		cost_review_recorded=False,
		join_source_count=4,
		join_review_recorded=False,
		requested_rows=250000,
		result_contains_sensitive_data=True,
		cache_requested=True,
	)
	assert blocked_query.status == "denied"
	assert set(blocked_query.matched_rules) >= {
		"query_requires_parameterization",
		"write_query_blocked",
		"restricted_query_requires_rbac",
		"sensitive_results_block_cache",
		"query_requires_lineage_capture",
		"query_result_limit_enforced",
	}

	review_query = service.execute_query(
		tenant_id=tenant_id,
		query_id="qry-review",
		sql="SELECT * FROM orders WHERE id = :id",
		actor="analyst",
		source_ids=["src-orders"],
		data_classification="internal",
		rbac_authorized=True,
		parameterized=True,
		write_query=False,
		lineage_capture_enabled=True,
		estimated_query_cost=2500.0,
		cost_review_recorded=False,
		join_source_count=4,
		join_review_recorded=False,
		requested_rows=1000,
		result_contains_sensitive_data=False,
		cache_requested=False,
	)
	assert review_query.status == "pending_review"
	assert set(review_query.matched_rules) == {
		"high_cost_query_requires_review",
		"cross_source_join_requires_review",
	}

	allowed_query = service.execute_query(
		tenant_id=tenant_id,
		query_id="qry-allowed",
		sql="SELECT * FROM orders WHERE id = :id",
		actor="analyst",
		source_ids=["src-orders"],
		data_classification="internal",
		rbac_authorized=True,
		parameterized=True,
		write_query=False,
		lineage_capture_enabled=True,
		estimated_query_cost=50.0,
		cost_review_recorded=False,
		join_source_count=1,
		join_review_recorded=False,
		requested_rows=1000,
		result_contains_sensitive_data=False,
		cache_requested=True,
	)
	assert allowed_query.status == "planned"

	cache = service.cache_result(
		tenant_id=tenant_id,
		cache_id="cache-too-long",
		query_id="qry-allowed",
		ttl_seconds=7200,
	)
	assert cache.status == "denied"
	assert cache.matched_rules == ["cache_ttl_requires_limit"]

	policy = service.change_policy(
		tenant_id=tenant_id,
		policy_id="policy-cost",
		name="Cost threshold",
		actor="governance",
		policy_review_recorded=False,
	)
	assert policy.status == "pending_review"
	assert policy.matched_rules == ["policy_change_requires_review"]

	retired = service.retire_source(
		tenant_id=tenant_id,
		source_id="src-orders",
		actor="data-platform",
		impact_review_recorded=False,
	)
	assert retired.status == "denied"
	assert retired.matched_rules == ["source_retirement_requires_impact_review"]
	assert service.dashboard_summary(tenant_id)["audit_event_count"] >= 12


def test_generated_view_models_expose_composable_surfaces():
	service = DVRLLifecycleService()
	tenant_id = "tenant-ui"
	service.register_source(
		tenant_id=tenant_id,
		source_id="src-ui",
		name="UI Source",
		source_type="api",
		owner="ui",
		credentials_vaulted=True,
		connection_encrypted=True,
	)
	service.publish_virtual_table(
		tenant_id=tenant_id,
		table_id="vt-ui",
		source_id="src-ui",
		name="UI Table",
		owner="ui",
		classification="internal",
		classification_complete=True,
	)

	assert dashboard_model(service, tenant_id)["summary"]["source_count"] == 1
	assert source_manager_model(service, tenant_id)["columns"][0] == "source_id"
	assert virtual_table_catalog_model(service, tenant_id)["rows"][0]["table_id"] == "vt-ui"
	assert federation_map_model(service, tenant_id)["edges"] == [{"from": "src-ui", "to": "vt-ui", "kind": "publishes"}]
	assert query_workbench_model(service, tenant_id)["defaults"]["max_result_rows"] == 100000
	assert adapter_health_model(tenant_id)["event_stream"] == "bytewax"
	assert settings_model(tenant_id)["routes"]
