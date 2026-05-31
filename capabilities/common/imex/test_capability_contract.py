"""Regression coverage for the IMEX executable capability contract."""

import pytest

from capabilities.common.imex import ImexService, imex_capability, register_capability
from capabilities.common.imex.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.imex.view_models import (
	dashboard_model,
	job_designer_model,
	lifecycle_batch_model,
	transfer_agent_roster_model,
	transfer_monitor_model,
)


def test_contract_exposes_configuration_rules_ui_theme_and_adapters():
	contract = get_capability_contract("tenant-transfer", {"jobs": {"max_concurrent_jobs": 5}})

	assert contract["capability"] == "imex"
	assert contract["configuration"]["tenant_id"] == "tenant-transfer"
	assert contract["configuration"]["jobs"]["max_concurrent_jobs"] == 5
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"jobs",
		"formats",
		"validation",
		"security",
		"orchestration",
		"observability",
		"adapters",
		"agents",
		"streaming",
		"ui",
		"theme",
	]
	assert len(contract["rule_engine"]["rules"]) >= 38
	assert contract["provides"] == ["import_export", "bulk_transfer", "transfer_agent_composition"]
	assert contract["requires"] == ["etlp", "conn", "auth", "audl"]
	assert contract["agents"]["first_class"] is True
	assert "codex" in contract["agents"]["supported_runtimes"]
	assert "migration_reviewer" in contract["agents"]["supported_roles"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "imex_runtime.ImexService"
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"jobs",
		"designer",
		"mappings",
		"monitor",
		"validation",
		"imports",
		"exports",
		"approvals",
		"artifacts",
		"audit",
		"agents",
		"lifecycle",
		"settings",
	}
	assert contract["ui"]["api_prefix"] == "/imex/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "artifact_browser" in contract["theme"]["components"]
	assert "transfer_agent_roster" in contract["theme"]["components"]


def test_rule_engine_enforces_transfer_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "execute_job",
		"preview_validated": False,
		"environment": "production",
		"approval_recorded": False,
		"direction": "export",
		"data_classification": "sensitive",
		"export_encrypted": False,
		"record_count": 200000,
		"monitoring_enabled": False,
		"checkpointing_enabled": False,
		"quality_score": 0.5,
		"quality_review_recorded": False,
		"invalid_records_present": True,
		"quarantine_enabled": False,
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"preview_required_before_execute",
		"production_transfer_requires_approval",
		"sensitive_export_requires_encryption",
		"large_transfer_requires_monitoring",
		"checkpointing_required",
		"quality_review_required",
		"invalid_records_require_quarantine",
	}


def test_rule_engine_enforces_transfer_agent_and_bytewax_guardrails():
	agent_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_transfer_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"scope_present": False,
		"owner_present": False,
		"purpose_present": False,
		"contribution_disclosed": False,
		"privileged_role": True,
		"human_approval_required": False,
	})
	stream_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "validate_imex_lifecycle_batch",
		"event_stream": "kafka",
	})

	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) >= {
		"transfer_agent_runtime_supported",
		"transfer_agent_role_supported",
		"transfer_agent_requires_scope",
		"transfer_agent_requires_owner",
		"transfer_agent_requires_purpose",
		"transfer_agent_requires_contribution_disclosure",
		"transfer_agent_privileged_role_requires_human_approval",
	}
	assert stream_result["decision"] == "deny"
	assert stream_result["matched_rules"] == ["bytewax_imex_stream_required"]


def test_imex_runtime_executes_transfer_lifecycle():
	service = ImexService()
	service.register_endpoint("source-crm", "tenant-a", "CRM", "connection", "conn://crm", "data")
	service.register_endpoint("warehouse", "tenant-a", "Warehouse", "connection", "conn://warehouse", "data")
	service.create_mapping_profile("crm-map", "tenant-a", "CRM Map", "profiles/crm.json", "maps/crm.json", "quality/crm")
	job = service.create_job(
		"crm-import",
		"tenant-a",
		"CRM Import",
		"import",
		"source-crm",
		"warehouse",
		"csv",
		"data",
		"development",
		"crm-map",
		"sha256:abc",
	)
	service.validate_preview("tenant-a", "crm-import", quality_score=0.99)
	run = service.execute_job("tenant-a", "crm-import", "run-1", record_count=5000)
	with pytest.raises(PermissionError, match="transfer_run_not_completed"):
		service.publish_artifact("tenant-a", "early-artifact", "run-1", "s3://exports/early.csv", "sha256:early", "90d")
	completed = service.complete_run("tenant-a", "run-1", records_processed=5000, quality_score=0.99)
	artifact = service.publish_artifact("tenant-a", "artifact-1", "run-1", "s3://exports/crm.csv", "sha256:def", "90d")

	assert job["status"] == "draft"
	assert run["status"] == "running"
	assert completed["status"] == "completed"
	assert artifact["status"] == "published"
	assert service.dashboard_summary("tenant-a")["completed_run_count"] == 1


def test_imex_runtime_blocks_missing_evidence():
	service = ImexService()

	with pytest.raises(PermissionError, match="connector_binding_required"):
		service.register_endpoint("source", "tenant-a", "Source", "connection", "", "data")

	service.register_endpoint("source", "tenant-a", "Source", "connection", "conn://source", "data")
	service.register_endpoint("target", "tenant-a", "Target", "connection", "conn://target", "data")
	with pytest.raises(PermissionError, match="source_profile_required"):
		service.create_mapping_profile("map", "tenant-a", "Map", "", "maps/file.json", "quality/file")

	service.create_mapping_profile("map", "tenant-a", "Map", "profiles/file.json", "maps/file.json", "quality/file")
	with pytest.raises(PermissionError, match="unsupported_transfer_format"):
		service.create_job("bad", "tenant-a", "Bad", "import", "source", "target", "exe", "data", "development", "map", "sha256:abc")
	with pytest.raises(PermissionError, match="preview_validation_required"):
		service.create_job("job", "tenant-a", "Job", "import", "source", "target", "csv", "data", "development", "map", "sha256:abc")
		service.execute_job("tenant-a", "job", "run", record_count=10)


def test_imex_runtime_review_and_artifact_guardrails():
	service = ImexService()
	service.register_endpoint("source", "tenant-a", "Source", "connection", "conn://source", "data")
	service.register_endpoint("external", "tenant-a", "External", "connection", "conn://external", "data", external=True, approved=False)
	service.create_mapping_profile("map", "tenant-a", "Map", "profiles/file.json", "maps/file.json", "quality/file")
	job = service.create_job("export", "tenant-a", "Export", "export", "source", "external", "json", "data", "production", "map", "sha256:abc", data_classification="sensitive", destination_approved=False)
	assert job["status"] == "pending_review"
	assert any(review["review_type"] == "destination" for review in service.list_reviews("tenant-a"))

	service.validate_preview("tenant-a", "export", quality_score=0.99)
	with pytest.raises(PermissionError, match="production_approval_required"):
		service.execute_job("tenant-a", "export", "run", record_count=200000, export_encrypted=True)
	with pytest.raises(PermissionError, match="large_transfer_monitoring_required"):
		service.execute_job("tenant-a", "export", "run", record_count=200000, approval_recorded=True, monitoring_enabled=False)
	run = service.execute_job("tenant-a", "export", "run", record_count=200000, approval_recorded=True, monitoring_enabled=True)
	assert run["status"] == "running"
	service.complete_run("tenant-a", "run", records_processed=200000, quality_score=0.99)
	with pytest.raises(PermissionError, match="retention_policy_required"):
		service.publish_artifact("tenant-a", "artifact", "run", "s3://exports/file.json", "sha256:def", "")
	service.publish_artifact("tenant-a", "artifact", "run", "s3://exports/file.json", "sha256:def", "90d")
	with pytest.raises(PermissionError, match="idempotency_required"):
		service.replay_run("tenant-a", "run", "replay", "")
	service.replay_run("tenant-a", "run", "replay", "idem-001")
	with pytest.raises(PermissionError, match="purge_review_required"):
		service.purge_artifact("tenant-a", "artifact", "data", False)
	purged = service.purge_artifact("tenant-a", "artifact", "data", True)
	assert purged["status"] == "purged"


def test_imex_runtime_governs_agents_and_lifecycle_batches():
	service = ImexService()

	with pytest.raises(PermissionError, match="unsupported_transfer_agent_runtime"):
		service.register_transfer_agent(
			agent_id="unknown-agent",
			tenant_id="tenant-a",
			name="Unknown Agent",
			runtime="unsupported",
			role="migration_reviewer",
			scope="migration reviews",
			owner="platform",
			purpose="review transfer migrations",
		)

	pending = service.register_transfer_agent(
		agent_id="migration-agent",
		tenant_id="tenant-a",
		name="Migration Agent",
		runtime="Claude Code",
		role="migration reviewer",
		scope="migration reviews",
		owner="platform",
		purpose="review transfer migrations",
		human_approval_required=False,
	)
	active = service.register_transfer_agent(
		agent_id="data-steward",
		tenant_id="tenant-a",
		name="Data Steward",
		runtime="codex",
		role="data_steward",
		scope="transfer metadata stewardship",
		owner="integration-office",
		purpose="maintain transfer metadata",
	)

	with pytest.raises(PermissionError, match="bytewax_lifecycle_stream_required"):
		service.validate_imex_lifecycle_batch("tenant-a", "kafka", 2)
	batch = service.validate_imex_lifecycle_batch("tenant-a", "bytewax", 4)
	summary = service.dashboard_summary("tenant-a")

	assert pending["status"] == "pending_review"
	assert pending["runtime"] == "claude_code"
	assert active["status"] == "active"
	assert batch["status"] == "accepted"
	assert summary["transfer_agent_count"] == 2
	assert summary["lifecycle_batch_count"] == 2
	assert summary["denied_lifecycle_batch_count"] == 1


def test_registration_and_ui_models_are_composable():
	registration = register_capability()
	service = ImexService()
	service.register_endpoint("source", "tenant-a", "Source", "connection", "conn://source", "data")
	dashboard = dashboard_model(service, "tenant-a")
	designer = job_designer_model(service, "tenant-a")
	monitor = transfer_monitor_model(service, "tenant-a")
	service.register_transfer_agent(
		agent_id="data-steward",
		tenant_id="tenant-a",
		name="Data Steward",
		runtime="codex",
		role="data_steward",
		scope="transfer metadata stewardship",
		owner="integration-office",
		purpose="maintain transfer metadata",
	)
	service.validate_imex_lifecycle_batch("tenant-a", "bytewax", 1)
	agents = transfer_agent_roster_model(service, "tenant-a")
	batches = lifecycle_batch_model(service, "tenant-a")

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "imex_transfer_console"
	assert registration["ui_components"]["mappings"] == "/imex/mappings"
	assert "etlp" in registration["dependencies"]
	assert "moni" in registration["optional_dependencies"]
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["required_processor"] == "bytewax"
	assert dashboard["summary"]["endpoint_count"] == 1
	assert "create_job" in designer["actions"]
	assert monitor["runs"] == []
	assert agents["agents"][0]["runtime"] == "codex"
	assert batches["required_processor"] == "bytewax"
	assert callable(imex_capability.health_check)
