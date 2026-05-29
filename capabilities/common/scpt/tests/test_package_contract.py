"""SCPT package runtime and publish contract tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.scpt.service import ScptService


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_scpt_contract_shape_is_valid():
	module = _load_module("scpt_contract", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-script")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "scpt"
	assert contract["configuration"]["scripts"]["allowed_languages"] == ["python", "javascript", "apg"]
	assert contract["configuration"]["sandbox"]["sandbox_required"] is True
	assert contract["configuration"]["packages"]["dangerous_import_blocking"] is True
	assert contract["ui"]["routes"]
	assert contract["theme"]["name"] == "scpt_script_workbench"


def test_scpt_app_entrypoint_is_publishable():
	module = _load_module("scpt_app", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "scpt" in model["capabilities"]
	assert "dangerous_permission_requires_approval" in model["rules"]
	assert model["capabilities"]["scpt"]["theme"]["name"] == "scpt_script_workbench"


def test_scpt_lifecycle_executes_with_guardrails():
	service = ScptService()
	tenant_id = "tenant-scpt"

	policy = service.create_package_policy(tenant_id, "stdlib-only", "platform-owner", allowed_packages=["json"], network_policy_attached=False)
	sandbox = service.create_sandbox(tenant_id, "local-python", "platform-owner")
	script = service.create_script(
		tenant_id,
		"normalize-payload",
		"python",
		"result = input_payload",
		"automation-owner",
		package_policy_id=policy["id"],
		sandbox_id=sandbox["id"],
		tags=["Workflow", "Transform"],
	)
	published = service.publish_script(tenant_id, script["id"], "automation-owner")
	bound = service.bind_workflow(tenant_id, script["id"], "wflo:on_customer_created", "automation-owner")
	execution = service.execute_script(tenant_id, script["id"], sandbox["id"], "workflow-runner", {"customer_id": "C-1"})
	completed = service.complete_execution(tenant_id, execution["id"], output={"normalized": True}, runtime_seconds=0.01, memory_mb=32)
	summary = service.dashboard_summary(tenant_id)

	assert published["state"] == "published"
	assert bound["workflow_bindings"] == ["wflo:on_customer_created"]
	assert completed["status"] == "succeeded"
	assert service.list_scripts(tenant_id)[0]["tags"] == ["transform", "workflow"]
	assert summary["script_count"] == 1
	assert summary["succeeded_execution_count"] == 1
	assert service.audit_events(tenant_id)


def test_scpt_policy_failures_are_enforced():
	service = ScptService()
	tenant_id = "tenant-guardrails"

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_sandbox("", "missing-tenant", "owner")

	with pytest.raises(PermissionError, match="script_owner_required"):
		service.create_script(tenant_id, "missing-owner", "python", "result = 1", "")

	with pytest.raises(PermissionError, match="network_policy_required"):
		service.create_sandbox(tenant_id, "network", "owner", network_enabled=True)

	with pytest.raises(PermissionError, match="resource_review_required"):
		service.create_sandbox(tenant_id, "large", "owner", max_memory_mb=1024)

	policy = service.create_package_policy(tenant_id, "network-policy", "owner", network_policy_attached=True)
	sandbox = service.create_sandbox(tenant_id, "safe", "owner")
	with pytest.raises(PermissionError, match="dangerous_permission_approval_required"):
		service.create_script(tenant_id, "network-script", "python", "import requests\nresult = 1", "owner", package_policy_id=policy["id"], sandbox_id=sandbox["id"])

	with pytest.raises(ValueError, match="python_syntax_error"):
		service.create_script(tenant_id, "bad-python", "python", "def broken(:", "owner")

	script = service.create_script(tenant_id, "draft", "python", "result = 1", "owner", package_policy_id=policy["id"], sandbox_id=sandbox["id"])
	with pytest.raises(PermissionError, match="published_script_required"):
		service.execute_script(tenant_id, script["id"], sandbox["id"], "owner")


def test_scpt_view_models_expose_composable_surfaces():
	from capabilities.common.scpt.views import (
		approvals_model,
		dashboard_model,
		execution_console_model,
		package_policy_model,
		sandbox_monitor_model,
		script_registry_model,
		settings_model,
		workbench_model,
	)

	service = ScptService()
	tenant_id = "tenant-view"
	policy = service.create_package_policy(tenant_id, "stdlib", "owner", allowed_packages=["json"])
	sandbox = service.create_sandbox(tenant_id, "python", "owner")
	script = service.create_script(tenant_id, "hello", "python", "result = 1", "owner", package_policy_id=policy["id"], sandbox_id=sandbox["id"])
	service.publish_script(tenant_id, script["id"], "owner")
	execution = service.execute_script(tenant_id, script["id"], sandbox["id"], "owner")
	service.complete_execution(tenant_id, execution["id"])

	assert dashboard_model(service, tenant_id)["summary"]["script_count"] == 1
	assert workbench_model(service, tenant_id)["actions"] == ["create_script", "approve_script", "publish_script", "bind_workflow"]
	assert script_registry_model(service, tenant_id)["scripts"][0]["state"] == "published"
	assert execution_console_model(service, tenant_id)["executions"][0]["status"] == "succeeded"
	assert sandbox_monitor_model(service, tenant_id)["sandboxes"][0]["state"] == "ready"
	assert package_policy_model(service, tenant_id)["package_policies"][0]["name"] == "stdlib"
	assert approvals_model(service, tenant_id)["pending_scripts"] == []
	assert settings_model(service, tenant_id)["theme"]["name"] == "scpt_script_workbench"
