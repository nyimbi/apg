import pytest

from capabilities.common.conf.ai_engine import ConfigurationIntelligenceEngine
from capabilities.common.conf.models import (
	CloudProvider,
	CMResource,
	ConfigurationDSL,
	ResourceState,
	ResourceType,
)


def _resource(last_known_config: dict | None = None) -> CMResource:
	return CMResource(
		tenant_id="tenant-1234",
		name="orders-db",
		resource_type=ResourceType.DATABASE,
		cloud_provider=CloudProvider.AWS,
		configuration=ConfigurationDSL(
			kind="database",
			metadata={"name": "orders-db"},
			spec={
				"resources": {"cpu": "2", "memory": "4Gi"},
				"security": {"encryption_at_rest": True},
				"monitoring": {"enabled": True},
				"backup": {"enabled": True},
			},
		),
		state=ResourceState.DEPLOYED,
		last_known_config=last_known_config,
		tags={"owner": "finance"},
	)


@pytest.mark.asyncio
async def test_basic_ai_engine_generates_configuration_from_natural_language() -> None:
	engine = ConfigurationIntelligenceEngine(tenant_id="tenant-1234")
	await engine.initialize()

	intent = await engine.parse_natural_language_intent(
		"Create a production postgres database called ledger-db with backup and high availability",
		{"environment": "production"},
	)
	configuration = await engine.generate_configuration_from_intent(intent)

	assert intent["intent"] == "create"
	assert intent["resource_type"] == "database"
	assert intent["requirements"]["replicas"] == 2
	assert configuration["kind"] == "database"
	assert configuration["metadata"]["name"] == "ledger-db"
	assert configuration["metadata"]["tenant_id"] == "tenant-1234"
	assert configuration["spec"]["backup"]["enabled"] is True
	assert configuration["spec"]["security"]["encryption_at_rest"] is True

	metrics = await engine.get_metrics()
	assert metrics["natural_language_requests"] == 1
	assert metrics["configurations_generated"] == 1
	assert metrics["predictions_made"] == 1


@pytest.mark.asyncio
async def test_basic_ai_engine_detects_drift_and_builds_remediation() -> None:
	engine = ConfigurationIntelligenceEngine(tenant_id="tenant-1234")
	await engine.initialize()
	resource = _resource(last_known_config={
		"resources": {"cpu": "1", "memory": "4Gi"},
		"security": {"encryption_at_rest": False},
		"monitoring": {"enabled": True},
		"backup": {"enabled": False},
	})

	drift = await engine.detect_configuration_drift(resource)
	remediation = await engine.generate_remediation_plan(drift)

	assert drift["has_drift"] is True
	assert {item["path"] for item in drift["details"]["differences"]} >= {
		"resources.cpu",
		"security.encryption_at_rest",
		"backup.enabled",
	}
	assert remediation["automated"] is True
	assert remediation["priority"] == "high"
	assert {action["target"] for action in remediation["actions"]} >= {
		"resources.cpu",
		"security.encryption_at_rest",
		"backup.enabled",
	}


@pytest.mark.asyncio
async def test_basic_ai_engine_optimizes_and_evaluates_policy_compliance() -> None:
	engine = ConfigurationIntelligenceEngine(tenant_id="tenant-1234")
	await engine.initialize()

	optimized = await engine.optimize_configuration({
		"kind": "web_application",
		"spec": {"resources": {"cpu": "1"}},
	})
	assert optimized["spec"]["resources"]["memory"] == "2Gi"
	assert optimized["spec"]["monitoring"]["enabled"] is True
	assert optimized["spec"]["security"]["encryption_in_transit"] is True

	resource = _resource()
	compliance = await engine.evaluate_policy_compliance(
		{"rules": [{"type": "require_encryption"}, {"type": "require_tag", "tag": "cost_center"}]},
		resource,
	)
	remediation = await engine.generate_compliance_remediation({}, resource, compliance)

	assert compliance["compliant"] is False
	assert compliance["violations"] == ["Missing required tag: cost_center"]
	assert remediation == [{"type": "compliance_fix", "target": "tags.cost_center", "value": "required"}]
