"""Real-time gateway AI processor should execute without external Redis."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from capabilities.composition.gateway.real_time_ai_processor import (
	ActionExecutor,
	InMemoryAsyncMeshStore,
	PolicyDeploymentEngine,
	RealTimeFailurePredictor,
)


class FakeDBSession:
	def __init__(self):
		self.commit_count = 0

	async def commit(self) -> None:
		self.commit_count += 1


@pytest.mark.asyncio
async def test_policy_deployment_records_active_route_rules_without_redis() -> None:
	db_session = FakeDBSession()
	engine = PolicyDeploymentEngine(db_session)
	policy = SimpleNamespace(
		compiled_rules={
			"route_rules": [
				{"target_service": "checkout-api", "weight": 80},
				{"service": "payment-api", "weight": 20},
			]
		},
		deployment_status="pending",
		deployed_at=None,
	)

	result = await engine.deploy_policy(policy)

	assert isinstance(engine.redis_client, InMemoryAsyncMeshStore)
	assert result["status"] == "success"
	assert result["deployed_rules"] == 2
	assert policy.deployment_status == "deployed"
	assert db_session.commit_count == 1

	checkout_keys = await engine.redis_client.keys("active_route_rule:checkout-api:*")
	payment_keys = await engine.redis_client.keys("active_route_rule:payment-api:*")
	assert len(checkout_keys) == 1
	assert len(payment_keys) == 1

	checkout_state = json.loads(await engine.redis_client.get(checkout_keys[0]))
	assert checkout_state["target_service"] == "checkout-api"
	assert checkout_state["rule_config"] == {"target_service": "checkout-api", "weight": 80}
	assert checkout_state["status"] == "active"


@pytest.mark.asyncio
async def test_action_executor_mutates_runtime_mesh_state_without_redis() -> None:
	executor = ActionExecutor(FakeDBSession())

	first = await executor.execute_action({
		"action": "scale_service",
		"target_service": "checkout-api",
		"parameters": {"replicas": 3},
	})
	second = await executor.execute_action({
		"action": "scale_service",
		"target_service": "checkout-api",
		"parameters": {"replicas": 5},
	})
	limits = await executor.execute_action({
		"action": "increase_resource_limits",
		"target_service": "checkout-api",
		"parameters": {"cpu": "2", "memory": "4Gi"},
	})

	assert first["status"] == "executed"
	assert first["result"]["previous_replicas"] == 1
	assert second["result"]["previous_replicas"] == 3
	assert second["result"]["new_replicas"] == 5
	assert limits["result"]["new_limits"] == {"cpu": "2", "memory": "4Gi"}

	service_state = json.loads(await executor.redis_client.get("service_runtime:checkout-api"))
	limit_state = json.loads(await executor.redis_client.get("resource_limits:checkout-api"))
	action_results = await executor.redis_client.keys("action_result:*")

	assert service_state["replicas"] == 5
	assert service_state["previous_replicas"] == 3
	assert limit_state["limits"] == {"cpu": "2", "memory": "4Gi"}
	assert len(action_results) == 3


@pytest.mark.asyncio
async def test_failure_predictor_generates_and_stores_preventive_actions_without_redis() -> None:
	predictor = RealTimeFailurePredictor(FakeDBSession())

	predictions = await predictor.predict_failures({
		"checkout-api": {"error_rate": 0.2, "latency_ms": 900, "cpu_percent": 85},
		"catalog-api": {"error_rate": 0.0, "latency_ms": 80, "cpu_percent": 20},
	})
	actions = await predictor.generate_preventive_actions(predictions)

	assert len(predictions) == 1
	assert predictions[0].service_name == "checkout-api"
	assert predictions[0].failure_probability >= 0.5
	assert {action["action"] for action in actions} == {"scale_service", "update_circuit_breaker"}

	stored_predictions = await predictor.redis_client.keys("prediction:*")
	assert len(stored_predictions) == 1
