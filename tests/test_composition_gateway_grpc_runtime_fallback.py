from types import SimpleNamespace

import pytest

from capabilities.composition.gateway.grpc_protocol_support import (
	GRPCServiceMeshProxy,
	GRPCServiceStatus,
)


class _CircuitBreakerManager:
	def get_circuit_breaker(self, name: str):
		return SimpleNamespace(state=SimpleNamespace(value="closed"))


@pytest.mark.asyncio
async def test_grpc_proxy_registers_services_without_grpc_runtime_dependencies() -> None:
	proxy = GRPCServiceMeshProxy(
		db_session=None,
		cert_manager=None,
		circuit_breaker_manager=_CircuitBreakerManager(),
		listen_port=50052,
	)

	await proxy.register_service(
		"InventoryService",
		[
			{
				"endpoint": "inventory.internal",
				"port": 50051,
				"weight": 2.0,
				"metadata": {"package": "apg.inventory"},
			}
		],
	)

	metrics = await proxy.get_metrics()

	assert proxy.registered_services["InventoryService"]["endpoint_count"] == 1
	assert metrics["registered_services"] == 1
	assert metrics["total_endpoints"] == 1
	assert metrics["services"]["InventoryService"]["healthy_endpoints"] == 1
	assert metrics["services"]["InventoryService"]["health_percentage"] == 100


@pytest.mark.asyncio
async def test_grpc_load_balancer_returns_registered_endpoint() -> None:
	proxy = GRPCServiceMeshProxy(
		db_session=None,
		cert_manager=None,
		circuit_breaker_manager=_CircuitBreakerManager(),
	)

	await proxy.register_service(
		"BillingService",
		[{"endpoint": "billing.internal", "port": 50051}],
	)

	endpoint = await proxy.load_balancer.get_endpoint("BillingService")

	assert endpoint["endpoint"] == "billing.internal"
	assert endpoint["port"] == 50051
	assert endpoint["health_status"] == GRPCServiceStatus.SERVING.value
