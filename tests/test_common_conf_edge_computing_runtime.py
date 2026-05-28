import pytest

from capabilities.common.conf.edge_computing_integration import (
	EdgeComputeCapability,
	EdgeComputingManager,
	EdgeConnectivity,
	EdgeDeploymentStrategy,
	EdgeDeviceType,
)


def _device(name: str, latitude: float, longitude: float, memory_gb: int = 4) -> dict:
	return {
		"name": name,
		"device_type": EdgeDeviceType.MANUFACTURING,
		"location": {
			"latitude": latitude,
			"longitude": longitude,
			"timezone": "Africa/Nairobi",
		},
		"hardware_specs": {
			"cpu_cores": 2,
			"memory_gb": memory_gb,
			"storage_gb": 16,
		},
		"connectivity": [EdgeConnectivity.WIFI, EdgeConnectivity.CELLULAR_4G],
		"compute_capability": EdgeComputeCapability.SMALL,
	}


@pytest.mark.asyncio
async def test_edge_registration_cluster_and_configuration_are_executable() -> None:
	manager = EdgeComputingManager("tenant-1234")
	first_device_id = await manager.register_edge_device(_device("factory-sensor-001", -1.286389, 36.817223))
	second_device_id = await manager.register_edge_device(_device("factory-sensor-002", -1.29, 36.82, memory_gb=8))

	assert manager.devices[first_device_id].health_status == "healthy"
	assert manager.devices[first_device_id].metadata["monitoring"]["enabled"] is True
	assert second_device_id in manager.devices[first_device_id].metadata["cluster_discovery"]["nearby_devices"]

	cluster_id = await manager.create_edge_cluster({
		"name": "nairobi-factory",
		"description": "Nairobi factory edge devices",
		"devices": [first_device_id, second_device_id],
		"geographic_region": {"name": "Nairobi"},
		"cluster_type": "manufacturing",
	})
	cluster = manager.clusters[cluster_id]

	assert cluster.health_score == 100.0
	assert cluster.geographic_region["centroid"]["latitude"] < 0
	assert cluster.failover_configuration["enabled"] is True
	assert cluster.failover_configuration["preferred_connectivity"] in {"wifi", "cellular_4g"}

	config_id = await manager.create_edge_configuration({
		"name": "predictive-maintenance-agent",
		"target_devices": [first_device_id],
		"target_clusters": [cluster_id],
		"configuration_spec": {
			"resources": {
				"cpu_cores": 8,
				"memory_mb": 32768,
				"storage_mb": 100000,
			},
			"networking": {},
			"storage": {"cache_mb": 100000},
		},
		"deployment_strategy": EdgeDeploymentStrategy.CANARY,
	})
	config = manager.configurations[config_id]

	assert config.configuration_spec["resources"]["cpu_cores"] == 2
	assert config.configuration_spec["resources"]["memory_mb"] == 4096
	assert config.configuration_spec["resources"]["optimized_for_edge"] is True
	assert config.configuration_spec["networking"]["delta_sync"] is True
	assert config.configuration_spec["networking"]["preferred_connectivity"] in {"wifi", "cellular_4g"}
	assert config.configuration_spec["storage"]["cache_mb"] <= 3276
	assert config.resource_constraints["max_memory_mb"] == 1024
	assert config.security_policies["device_authentication"] is True


@pytest.mark.asyncio
async def test_edge_canary_deployment_expands_cluster_targets_and_tracks_health() -> None:
	manager = EdgeComputingManager("tenant-1234")
	first_device_id = await manager.register_edge_device(_device("factory-sensor-101", -1.286389, 36.817223))
	second_device_id = await manager.register_edge_device(_device("factory-sensor-102", -1.29, 36.82))
	cluster_id = await manager.create_edge_cluster({
		"name": "nairobi-factory",
		"devices": [first_device_id, second_device_id],
		"geographic_region": {"name": "Nairobi"},
		"cluster_type": "manufacturing",
	})
	config_id = await manager.create_edge_configuration({
		"name": "quality-agent",
		"target_devices": [first_device_id],
		"target_clusters": [cluster_id],
		"configuration_spec": {"resources": {"cpu_cores": 1}, "networking": {}, "storage": {}},
		"deployment_strategy": EdgeDeploymentStrategy.CANARY,
	})

	deployment_id = await manager.deploy_edge_configuration(config_id)
	deployment = manager.deployments[deployment_id]

	assert deployment.status == "completed"
	assert deployment.progress_percentage == 100.0
	assert deployment.target_devices == [first_device_id, second_device_id]
	assert set(deployment.successful_devices) == {first_device_id, second_device_id}
	assert deployment.failed_devices == []
	assert deployment.health_checks[-1]["healthy"] is True
	assert manager.devices[first_device_id].current_config_version == config_id
	assert manager.devices[second_device_id].configuration_state == "configured"
