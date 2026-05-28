from fastapi.testclient import TestClient

from capabilities.composition.config.api import _api_runtime_state, create_app


AUTH_HEADERS = {
	"X-API-Key": "cc_test_key",
	"X-APG-User-ID": "config-user",
	"X-APG-Tenant-ID": "config-tenant",
}


def test_composition_config_api_imports_and_executes_runtime_resources():
	_api_runtime_state.clear()
	client = TestClient(create_app())

	workspace_response = client.post(
		"/workspaces",
		headers=AUTH_HEADERS,
		json={
			"name": "ERP Platform",
			"slug": "erp-platform",
			"settings": {"region": "africa-east"},
		},
	)
	assert workspace_response.status_code == 200
	workspace = workspace_response.json()
	assert workspace["tenant_id"] == "config-tenant"

	template_response = client.post(
		f"/templates?workspace_id={workspace['id']}",
		headers=AUTH_HEADERS,
		json={
			"name": "Approval Rules",
			"category": "workflow",
			"template_data": {"approval_limit": 1000},
			"is_public": True,
		},
	)
	assert template_response.status_code == 200
	assert template_response.json()["workspace_id"] == workspace["id"]

	config_response = client.post(
		f"/configurations?workspace_id={workspace['id']}",
		headers=AUTH_HEADERS,
		json={
			"name": "Purchase Approval",
			"key_path": "/erp/procurement/approval",
			"value": {"limit": 1000, "currency": "USD"},
			"tags": ["erp", "rules"],
		},
	)
	assert config_response.status_code == 200
	configuration = config_response.json()
	assert configuration["version"] == "1.0.0"
	assert configuration["status"] == "active"

	update_response = client.put(
		f"/configurations/{configuration['id']}?change_reason=raise-limit",
		headers=AUTH_HEADERS,
		json={"value": {"limit": 2500, "currency": "USD"}},
	)
	assert update_response.status_code == 200
	assert update_response.json()["version"] == "1.0.1"

	search_response = client.get("/configurations?query=approval", headers=AUTH_HEADERS)
	assert search_response.status_code == 200
	assert search_response.json()["total_count"] == 1

	deploy_response = client.post(
		f"/configurations/{configuration['id']}/deploy",
		headers=AUTH_HEADERS,
		json={"cloud_provider": "local", "environment_id": "test", "options": {"dry_run": False}},
	)
	assert deploy_response.status_code == 200
	assert deploy_response.json()["status"] == "deployed"

	deployments_response = client.get("/deployments?cloud_provider=local", headers=AUTH_HEADERS)
	assert deployments_response.status_code == 200
	assert deployments_response.json()["total_count"] == 1

	versions_response = client.get(f"/configurations/{configuration['id']}/versions", headers=AUTH_HEADERS)
	assert versions_response.status_code == 200
	assert versions_response.json()["total_count"] == 2

	restore_response = client.post(
		f"/configurations/{configuration['id']}/restore",
		headers=AUTH_HEADERS,
		json={"version": "1.0.0", "reason": "test rollback"},
	)
	assert restore_response.status_code == 200
	assert restore_response.json()["success"] is True

	usage_response = client.get("/analytics/usage", headers=AUTH_HEADERS)
	assert usage_response.status_code == 200
	assert usage_response.json()["total_configurations"] == 1

	audit_response = client.get("/security/audit-log", headers=AUTH_HEADERS)
	assert audit_response.status_code == 200
	assert audit_response.json()["total_count"] >= 5

	compliance_response = client.get("/security/compliance-report?framework=SOC2", headers=AUTH_HEADERS)
	assert compliance_response.status_code == 200
	assert compliance_response.json()["framework"] == "SOC2"
	assert compliance_response.json()["compliance_score"] > 0


def test_composition_config_api_has_no_static_runtime_placeholders():
	api_source = "capabilities/composition/config/api.py"
	models_source = "capabilities/composition/config/models.py"
	combined = open(api_source, encoding="utf-8").read() + open(models_source, encoding="utf-8").read()

	for marker in (
		"pass\n",
		"This would query",
		"would be implemented",
		"placeholder",
		"user:pass@localhost",
		"redis://localhost:6379",
		"regex=",
	):
		assert marker not in combined
