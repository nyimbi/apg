"""SBOX package runtime and publish contract tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.sbox.service import SboxService


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_sbox_contract_shape_is_valid():
	module = _load_module("sbox_contract", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "sbox"
	assert contract["configuration"]["isolation"]["secret_redaction_required"] is True
	assert contract["configuration"]["datasets"]["dataset_lineage_required"] is True
	assert contract["ui"]["routes"]
	assert contract["theme"]["name"] == "sbox_safe_testing"


def test_sbox_app_entrypoint_is_publishable():
	module = _load_module("sbox_app", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "sbox" in model["capabilities"]
	assert "sandbox_requires_isolation_profile" in model["rules"]
	assert model["capabilities"]["sbox"]["theme"]["name"] == "sbox_safe_testing"


def test_sbox_lifecycle_executes_with_guardrails():
	service = SboxService()
	tenant_id = "tenant-sandbox"

	isolation = service.create_isolation_profile(tenant_id, "strict-local", approved_by="security")
	template = service.create_template(tenant_id, "plugin-test", "python", "qa-owner", tags=["Plugin", "CI"])
	dataset = service.register_dataset(tenant_id, "synthetic-orders", "synthetic", "qa-owner", "generated:orders:v1", 14)
	sandbox = service.create_sandbox(
		tenant_id,
		"orders-plugin-sandbox",
		template["id"],
		isolation["id"],
		"qa-owner",
		dataset_ids=[dataset["id"]],
	)
	run = service.start_run(tenant_id, sandbox["id"], "plugin", "qa-owner", tests_requested=3)
	completed = service.complete_run(tenant_id, run["id"], tests_passed=3, logs=["all checks passed"])
	summary = service.dashboard_summary(tenant_id)

	assert sandbox["state"] == "ready"
	assert sandbox["risk_score"] == 0
	assert completed["status"] == "passed"
	assert service.list_sandboxes(tenant_id)[0]["state"] == "completed"
	assert summary["sandbox_count"] == 1
	assert summary["passed_run_count"] == 1
	assert service.audit_events(tenant_id)


def test_sbox_policy_failures_are_enforced():
	service = SboxService()
	tenant_id = "tenant-guardrails"

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_template("", "missing-tenant", "python", "owner")

	with pytest.raises(PermissionError, match="sandbox_owner_required"):
		service.create_template(tenant_id, "missing-owner", "python", "")

	with pytest.raises(PermissionError, match="outbound_network_approval_required"):
		service.create_isolation_profile(tenant_id, "open-network", outbound_network_allowed=True)

	with pytest.raises(PermissionError, match="production_data_review_required"):
		service.register_dataset(tenant_id, "prod-copy", "production_sample", "data-owner", "warehouse:orders", 7)

	isolation = service.create_isolation_profile(tenant_id, "strict", approved_by="security")
	template = service.create_template(tenant_id, "long-lived", "python", "owner")
	with pytest.raises(PermissionError, match="long_lived_sandbox_review_required"):
		service.create_sandbox(tenant_id, "long-lived", template["id"], isolation["id"], "owner", ttl_hours=72)


def test_sbox_view_models_expose_composable_surfaces():
	from capabilities.common.sbox.views import (
		dashboard_model,
		dataset_manager_model,
		policy_center_model,
		run_monitor_model,
		sandbox_console_model,
		template_library_model,
	)

	service = SboxService()
	tenant_id = "tenant-view"
	isolation = service.create_isolation_profile(tenant_id, "strict", approved_by="security")
	template = service.create_template(tenant_id, "integration", "python", "owner")
	sandbox = service.create_sandbox(tenant_id, "integration-env", template["id"], isolation["id"], "owner")
	run = service.start_run(tenant_id, sandbox["id"], "integration", "owner", 1)
	service.complete_run(tenant_id, run["id"], 1)

	assert dashboard_model(service, tenant_id)["summary"]["sandbox_count"] == 1
	assert sandbox_console_model(service, tenant_id)["actions"] == ["create_sandbox", "expire_sandbox", "start_run"]
	assert template_library_model(service, tenant_id)["templates"]
	assert dataset_manager_model(service, tenant_id)["guardrails"]
	assert run_monitor_model(service, tenant_id)["runs"][0]["status"] == "passed"
	assert policy_center_model(service, tenant_id)["theme"]["name"] == "sbox_safe_testing"
