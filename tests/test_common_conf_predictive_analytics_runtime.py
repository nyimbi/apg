import pytest

from capabilities.common.conf.models import (
	CloudProvider,
	CMResource,
	ConfigurationDSL,
	ResourceState,
	ResourceType,
)
from capabilities.common.conf.predictive_analytics import PredictiveConfigAnalytics


def _resource(
	name: str,
	spec: dict,
	state: ResourceState = ResourceState.DEPLOYED,
	resource_type: ResourceType = ResourceType.DATABASE,
	estimated_cost: float = 0.0,
	performance_metrics: dict | None = None,
) -> CMResource:
	return CMResource(
		tenant_id="tenant-1234",
		name=name,
		resource_type=resource_type,
		cloud_provider=CloudProvider.AWS,
		configuration=ConfigurationDSL(
			kind=resource_type.value,
			metadata={"name": name},
			spec=spec,
		),
		state=state,
		estimated_cost_monthly=estimated_cost,
		performance_metrics=performance_metrics or {},
	)


@pytest.mark.asyncio
async def test_predictive_analytics_identifies_high_risk_configuration() -> None:
	analytics = PredictiveConfigAnalytics(tenant_id="tenant-1234")
	await analytics.initialize()
	resource = _resource(
		"customer-db",
		{
			"resources": {"cpu": "2", "memory": "8Gi"},
			"security": {"encryption_at_rest": False},
			"monitoring": False,
			"backup": False,
		},
		state=ResourceState.DRIFTED,
		estimated_cost=2400.0,
		performance_metrics={"cpu_usage": 91, "memory_usage": 88, "error_rate": 7},
	)
	resource.validation_errors.append("storage class is not approved")
	resource.policy_violations.append("database must use customer-managed keys")

	analysis = await analytics.analyze_configuration_risks(resource)

	assert analysis["resource_id"] == resource.id
	assert analysis["resource_name"] == "customer-db"
	assert analysis["tenant_id"] == "tenant-1234"
	assert analysis["risk_score"] >= 0.9
	assert analysis["risk_level"] == "critical"
	assert analysis["autonomous_remediation_available"] is True
	assert {issue["type"] for issue in analysis["predicted_issues"]} >= {
		"configuration_drift",
		"policy_non_compliance",
		"unencrypted_resource",
		"capacity_saturation",
		"error_rate_spike",
	}
	assert {recommendation["category"] for recommendation in analysis["recommendations"]} >= {
		"cost",
		"governance",
		"security",
	}

	cached = await analytics.get_resource_insights(resource)
	assert cached == analysis

	metrics = await analytics.get_metrics()
	assert metrics["predictions_made"] == 1
	assert metrics["resources_analyzed"] == 1
	assert metrics["risks_prevented"] >= 5
	assert metrics["cost_optimizations_suggested"] == 1
	assert metrics["accuracy_rate"] == 0.82


@pytest.mark.asyncio
async def test_predictive_analytics_aggregates_system_risk() -> None:
	analytics = PredictiveConfigAnalytics(tenant_id="tenant-1234")
	await analytics.initialize()
	healthy_api = _resource(
		"orders-api",
		{
			"resources": {"cpu": "1", "memory": "2Gi"},
			"security": {"encryption_at_rest": True},
			"monitoring": {"enabled": True},
			"backup": {"retention_days": 7},
			"replicas": 3,
		},
		resource_type=ResourceType.KUBERNETES_DEPLOYMENT,
		performance_metrics={"cpu_usage": 45, "memory_usage": 50, "error_rate": 0.1},
	)
	risky_storage = _resource(
		"archive-store",
		{
			"resources": {"storage": "10Ti"},
			"security": {"encryption_at_rest": False},
			"monitoring": False,
			"backup": False,
		},
		state=ResourceState.FAILED,
		resource_type=ResourceType.STORAGE,
		estimated_cost=1600.0,
	)

	system = await analytics.get_system_insights({
		healthy_api.id: healthy_api,
		risky_storage.id: risky_storage,
	})

	assert system["resource_count"] == 2
	assert system["high_risk_resources"] == 1
	assert system["overall_health"] in {"watch", "degraded", "critical"}
	assert system["predicted_incidents"]
	assert system["optimization_opportunities"]
	assert system["cost_savings_potential"] == 240.0
