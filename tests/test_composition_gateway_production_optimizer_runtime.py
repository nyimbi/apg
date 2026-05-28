import pytest

from capabilities.composition.gateway.production_optimizer import ProductionOptimizer


@pytest.mark.asyncio
async def test_production_optimizer_runs_snapshot_cycle_without_external_services() -> None:
	optimizer = ProductionOptimizer()

	metrics = {
		"services": {
			"user-api": {
				"avg_response_time": 150.0,
				"request_rate": 1000.0,
				"error_rate": 2.5,
				"cpu_usage": 75.0,
				"memory_usage": 60.0,
			},
			"payment-api": {
				"avg_response_time": 300.0,
				"request_rate": 500.0,
				"error_rate": 5.0,
				"cpu_usage": 85.0,
				"memory_usage": 80.0,
			},
		},
		"load_balancers": {
			"user-lb": {
				"algorithm": "round_robin",
				"active_connections": 150,
				"backend_health": {"healthy": 3, "unhealthy": 0},
			},
		},
	}

	result = await optimizer.run_optimization_cycle(metrics)

	assert result["optimizations_applied"] >= 1
	assert result["optimized_pools"]["payment-api"]["recommended_config"]["pool_size"] >= 5
	assert result["cache_recommendations"]["user-api"]["recommended_config"]["strategy"] == "adaptive"
	assert result["load_balancer_recommendations"]["user-lb"]["recommended_config"]["algorithm"] == "least_connections"
	assert result["performance_improvement"]["latency_reduction_ms"] > 0


@pytest.mark.asyncio
async def test_production_optimizer_exposes_specific_optimization_methods() -> None:
	optimizer = ProductionOptimizer()
	metrics = {
		"services": {
			"analytics-api": {
				"avg_response_time": 260.0,
				"request_rate": 800.0,
				"error_rate": 1.5,
				"cpu_usage": 70.0,
			}
		}
	}

	pools = await optimizer.optimize_connection_pools(metrics)
	cache = await optimizer.optimize_caching_strategy(metrics)

	assert "optimized_pools" in pools
	assert "cache_recommendations" in cache
	assert pools["optimizations_applied"] == 1
	assert cache["optimizations_applied"] == 1
