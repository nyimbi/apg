"""Regression coverage for the MLCM executable capability contract."""

import pytest

from capabilities.common.mlcm import register_capability
from capabilities.common.mlcm.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)
from capabilities.common.mlcm.service import MlcmService
from capabilities.common.mlcm.views import (
	audit_timeline_model,
	baseline_evidence_model,
	dashboard_model,
	deployment_board_model,
	drift_monitor_model,
	evaluation_console_model,
	governance_model,
	model_card_library_model,
	promotion_board_model,
	registry_model,
	rollback_console_model,
	version_manager_model,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-models", {"evaluation": {"minimum_eval_score": 0.9}})

	assert contract["capability"] == "mlcm"
	assert contract["configuration"]["tenant_id"] == "tenant-models"
	assert contract["configuration"]["evaluation"]["minimum_eval_score"] == 0.9
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"registry",
		"versions",
		"evaluation",
		"promotion",
		"deployment",
		"monitoring",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"registry",
		"versions",
		"model_cards",
		"evaluation",
		"baselines",
		"promotion",
		"deployments",
		"drift",
		"rollback",
		"governance",
		"audit",
		"settings"
	}
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "service.MlcmService"
	assert contract["ui"]["api_prefix"] == "/mlcm/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "promotion_gate_panel" in contract["theme"]["components"]


def test_rule_engine_enforces_model_lifecycle_guardrails():
	deployment_result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "deploy_model",
		"owner_assigned": False,
		"target_stage": "production",
		"approval_recorded": False,
		"model_card_present": False,
		"eval_score": 0.3,
		"promotion_requested": True,
		"drift_detected": True,
		"drift_review_recorded": False,
	})
	stream_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "configure_monitoring",
		"event_stream": "kafka",
	})

	assert deployment_result["decision"] == "deny"
	assert set(deployment_result["matched_rules"]) >= {
		"tenant_context_required",
		"production_promotion_requires_approval",
		"deployment_requires_model_card",
		"low_eval_score_blocks_promotion",
		"drifted_model_requires_review",
	}
	assert stream_result["decision"] == "deny"
	assert stream_result["matched_rules"] == ["bytewax_stream_required_for_monitoring"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "mlcm"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "mlcm_model_ops_console"
	assert registration["ui_components"]["deployments"] == "/mlcm/deployments"
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert "aicr" in registration["dependencies"]
	assert "mlcm:deploy" in registration["permissions"]


def test_mlcm_lifecycle_is_executable():
	service = MlcmService()
	tenant_id = "tenant-models"

	model = service.register_model(
		model_id="fraud-risk",
		tenant_id=tenant_id,
		name="Fraud Risk Model",
		owner="risk-ai",
		problem_type="classification",
		risk_level="high",
		tags=["fraud", "payments"],
	)
	version = service.create_version(
		version_id="fraud-risk-v1",
		tenant_id=tenant_id,
		model_id=model["id"],
		version="1.0.0",
		artifact_uri="s3://models/fraud-risk/1.0.0/model.pkl",
		model_card={
			"purpose": "score payment fraud risk",
			"owner": "risk-ai",
			"training_data": "payments-2026-q1",
			"limitations": "not calibrated for card-present transactions",
		},
		training_data_ref="dataset:payments-2026-q1",
		baseline_ref="baseline:fraud-risk-2026-q1",
	)
	evaluation = service.record_evaluation(
		evaluation_id="eval-fraud-v1",
		tenant_id=tenant_id,
		version_id=version["id"],
		score=0.91,
		baseline_ref="baseline:fraud-risk-2026-q1",
		metrics={"auc": 0.94, "precision": 0.87},
		evidence_refs=["report:eval-fraud-v1"],
		evaluator="ml-quality",
	)
	promotion = service.request_promotion(
		request_id="promote-fraud-v1-prod",
		tenant_id=tenant_id,
		version_id=version["id"],
		target_stage="production",
		requested_by="risk-ai",
		approval_recorded=True,
		approval_ref="approval:ml-gate-42",
	)
	target = service.create_target(
		target_id="risk-prod",
		tenant_id=tenant_id,
		name="Risk Production Endpoint",
		environment="production",
		serving_runtime="aicr-python",
		owner="risk-platform",
	)
	deployment = service.deploy_model(
		deployment_id="deploy-fraud-v1",
		tenant_id=tenant_id,
		version_id=version["id"],
		target_id=target["id"],
		replicas=3,
		canary_percent=10,
		approved_by="release-manager",
	)
	drift = service.record_drift(
		signal_id="drift-fraud-v1",
		tenant_id=tenant_id,
		version_id=version["id"],
		metric="psi",
		score=0.18,
		threshold=0.2,
	)

	assert model["owner"] == "risk-ai"
	assert version["stage"] == "dev"
	assert evaluation["status"] == "passed"
	assert promotion["status"] == "approved"
	assert service.list_versions(tenant_id)[0]["stage"] == "production"
	assert deployment["status"] == "serving"
	assert deployment["replicas"] == 3
	assert drift["status"] == "within_threshold"

	summary = service.dashboard_summary(tenant_id)
	assert summary["model_count"] == 1
	assert summary["production_version_count"] == 1
	assert summary["serving_count"] == 1
	assert summary["unresolved_drift_count"] == 0

	assert dashboard_model(service, tenant_id)["summary"]["deployment_count"] == 1
	assert registry_model(service, tenant_id)["models"][0]["id"] == "fraud-risk"
	assert version_manager_model(service, tenant_id)["promotions"][0]["status"] == "approved"
	assert evaluation_console_model(service, tenant_id)["evaluations"][0]["score"] == 0.91
	assert model_card_library_model(service, tenant_id)["model_cards"][0]["complete"] is True
	assert baseline_evidence_model(service, tenant_id)["baselines"][0]["baseline_ref"] == "baseline:fraud-risk-2026-q1"
	assert promotion_board_model(service, tenant_id)["promotions"][0]["status"] == "approved"
	assert deployment_board_model(service, tenant_id)["deployments"][0]["target_id"] == "risk-prod"
	assert drift_monitor_model(service, tenant_id)["unresolved"] == []
	assert audit_timeline_model(service, tenant_id)["audit_events"]
	assert governance_model(service, tenant_id)["audit_events"]


def test_mlcm_service_enforces_policy_guardrails():
	service = MlcmService()
	tenant_id = "tenant-guardrails"

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_model(
			model_id="missing-tenant",
			tenant_id="",
			name="Missing Tenant",
			owner="owner",
			problem_type="classification",
		)

	with pytest.raises(PermissionError, match="model_owner_required"):
		service.register_model(
			model_id="missing-owner",
			tenant_id=tenant_id,
			name="Missing Owner",
			owner="",
			problem_type="classification",
		)

	service.register_model(
		model_id="guardrail-model",
		tenant_id=tenant_id,
		name="Guardrail Model",
		owner="ml-team",
		problem_type="classification",
	)
	no_card = service.create_version(
		version_id="guardrail-v0",
		tenant_id=tenant_id,
		model_id="guardrail-model",
		version="0.1.0",
		artifact_uri="s3://models/guardrail/0.1.0/model.pkl",
	)
	service.create_target(
		target_id="guardrail-dev",
		tenant_id=tenant_id,
		name="Guardrail Dev",
		environment="dev",
		serving_runtime="aicr-python",
		owner="platform",
	)
	with pytest.raises(PermissionError, match="model_card_required"):
		service.deploy_model(
			deployment_id="deploy-no-card",
			tenant_id=tenant_id,
			version_id=no_card["id"],
			target_id="guardrail-dev",
		)

	version = service.create_version(
		version_id="guardrail-v1",
		tenant_id=tenant_id,
		model_id="guardrail-model",
		version="1.0.0",
		artifact_uri="s3://models/guardrail/1.0.0/model.pkl",
		model_card={
			"purpose": "classify support messages",
			"owner": "ml-team",
			"training_data": "tickets-2026",
			"limitations": "English-heavy data",
		},
	)
	service.record_evaluation(
		evaluation_id="eval-low",
		tenant_id=tenant_id,
		version_id=version["id"],
		score=0.61,
		baseline_ref="baseline:tickets",
	)
	with pytest.raises(PermissionError, match="evaluation_score_too_low"):
		service.request_promotion(
			request_id="promote-low",
			tenant_id=tenant_id,
			version_id=version["id"],
			target_stage="staging",
			requested_by="ml-team",
		)

	service.record_evaluation(
		evaluation_id="eval-pass",
		tenant_id=tenant_id,
		version_id=version["id"],
		score=0.88,
		baseline_ref="baseline:tickets",
	)
	with pytest.raises(PermissionError, match="promotion_approval_required"):
		service.request_promotion(
			request_id="promote-prod-no-approval",
			tenant_id=tenant_id,
			version_id=version["id"],
			target_stage="production",
			requested_by="ml-team",
		)

	service.request_promotion(
		request_id="promote-prod-approved",
		tenant_id=tenant_id,
		version_id=version["id"],
		target_stage="production",
		requested_by="ml-team",
		approval_recorded=True,
		approval_ref="approval:ok",
	)
	service.create_target(
		target_id="guardrail-prod",
		tenant_id=tenant_id,
		name="Guardrail Production",
		environment="production",
		serving_runtime="aicr-python",
		owner="platform",
	)
	service.record_drift(
		signal_id="drift-high",
		tenant_id=tenant_id,
		version_id=version["id"],
		metric="psi",
		score=0.34,
		threshold=0.2,
	)
	with pytest.raises(PermissionError, match="drift_review_required"):
		service.deploy_model(
			deployment_id="deploy-drifted",
			tenant_id=tenant_id,
			version_id=version["id"],
			target_id="guardrail-prod",
		)

	service.record_drift_review(
		signal_id="drift-high",
		tenant_id=tenant_id,
		review_ref="review:drift-accepted",
	)
	deployment = service.deploy_model(
		deployment_id="deploy-after-review",
		tenant_id=tenant_id,
		version_id=version["id"],
		target_id="guardrail-prod",
	)
	assert deployment["status"] == "serving"
	rollback_target = service.create_version(
		version_id="guardrail-v0.9",
		tenant_id=tenant_id,
		model_id="guardrail-model",
		version="0.9.0",
		artifact_uri="s3://models/guardrail/0.9.0/model.pkl",
		model_card={
			"purpose": "classify support messages",
			"owner": "ml-team",
			"training_data": "tickets-2025",
			"limitations": "legacy baseline",
		},
	)
	rollback = service.rollback_deployment(
		rollback_id="rollback-after-review",
		tenant_id=tenant_id,
		deployment_id=deployment["id"],
		to_version_id=rollback_target["id"],
		reason="drift mitigation",
		requested_by="ml-team",
	)
	assert rollback["status"] == "completed"
	assert rollback_console_model(service, tenant_id)["rollbacks"][0]["reason"] == "drift mitigation"

	with pytest.raises(PermissionError, match="model_retirement_review_required"):
		service.retire_model(
			retirement_id="retire-no-review",
			tenant_id=tenant_id,
			model_id="guardrail-model",
			impact_review_ref="",
		)

	retirement = service.retire_model(
		retirement_id="retire-guardrail",
		tenant_id=tenant_id,
		model_id="guardrail-model",
		impact_review_ref="impact:retire-guardrail",
		retired_by="ml-team",
	)
	assert retirement["status"] == "completed"
	assert service.dashboard_summary(tenant_id)["retired_model_count"] == 1

	with pytest.raises(LookupError, match="model_version_not_found"):
		service.record_evaluation(
			evaluation_id="cross-tenant",
			tenant_id="other-tenant",
			version_id=version["id"],
			score=0.9,
			baseline_ref="baseline:other",
		)
