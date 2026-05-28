import pytest

from capabilities.common.conf.models import CloudProvider, ResourceType
from capabilities.common.conf.universal_abstraction import (
	UniversalResource,
	UniversalResourceLayer,
)


@pytest.mark.asyncio
async def test_universal_abstraction_translates_storage_across_providers() -> None:
	layer = UniversalResourceLayer(tenant_id="tenant-1234")
	await layer.initialize()
	resource = UniversalResource(
		name="customer-archive",
		resource_type=ResourceType.STORAGE,
		storage_specs={
			"bucket_name": "customer-archive",
			"account_name": "customerarchive",
			"versioning": True,
		},
		security_specs={"https_only": True, "uniform_access": True},
		tags={"environment": "test"},
	)

	translations = {
		provider: await adapter.translate_resource(resource)
		for provider, adapter in layer.providers.items()
	}

	assert translations[CloudProvider.AWS]["Type"] == "AWS::S3::Bucket"
	assert translations[CloudProvider.AZURE]["resources"][0]["type"] == "Microsoft.Storage/storageAccounts"
	assert translations[CloudProvider.GCP]["resources"][0]["type"] == "storage.v1.bucket"

	validations = [
		await adapter.validate_resource(resource)
		for adapter in layer.providers.values()
	]
	assert all(result.valid for result in validations)


@pytest.mark.asyncio
async def test_universal_abstraction_translates_databases_across_providers() -> None:
	layer = UniversalResourceLayer(tenant_id="tenant-1234")
	await layer.initialize()
	resource = UniversalResource(
		name="orders-db",
		resource_type=ResourceType.DATABASE,
		compute_specs={
			"db_instance_class": "db.t3.micro",
			"engine": "postgres",
			"sku_name": "Basic",
			"database_version": "POSTGRES_15",
			"tier": "db-f1-micro",
		},
		storage_specs={
			"allocated_storage": 20,
			"max_size_bytes": 2147483648,
			"disk_size_gb": 20,
		},
		security_specs={"encryption_enabled": True},
	)

	translations = {
		provider: await adapter.translate_resource(resource)
		for provider, adapter in layer.providers.items()
	}

	assert translations[CloudProvider.AWS]["Type"] == "AWS::RDS::DBInstance"
	assert translations[CloudProvider.AZURE]["resources"][0]["type"] == "Microsoft.Sql/servers/databases"
	assert translations[CloudProvider.GCP]["resources"][0]["type"] == "sqladmin.v1beta4.instance"

	validations = [
		await adapter.validate_resource(resource)
		for adapter in layer.providers.values()
	]
	assert all(result.valid for result in validations)


@pytest.mark.asyncio
async def test_universal_abstraction_translates_containers_across_providers() -> None:
	layer = UniversalResourceLayer(tenant_id="tenant-1234")
	await layer.initialize()
	resource = UniversalResource(
		name="orders-worker",
		resource_type=ResourceType.CONTAINER,
		compute_specs={
			"image": "registry.example.com/orders-worker:1.0.0",
			"cpu": 1,
			"memory": "512Mi",
			"memory_gb": 1.5,
		},
		network_specs={"network_mode": "awsvpc"},
		tags={"service": "orders"},
	)

	translations = {
		provider: await adapter.translate_resource(resource)
		for provider, adapter in layer.providers.items()
	}

	assert translations[CloudProvider.AWS]["Type"] == "AWS::ECS::TaskDefinition"
	assert translations[CloudProvider.AZURE]["resources"][0]["type"] == "Microsoft.ContainerInstance/containerGroups"
	assert translations[CloudProvider.GCP]["resources"][0]["type"] == "run.googleapis.com/v1.namespaces.services"

	validations = [
		await adapter.validate_resource(resource)
		for adapter in layer.providers.values()
	]
	assert all(result.valid for result in validations)
