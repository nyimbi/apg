"""KEYM multi-cloud federation adapter runtime tests."""

from __future__ import annotations

import pytest

from capabilities.common.keym.cloud_federation import (
	CloudKeyFederationManager,
	CloudProvider,
	FederationPolicy,
	SyncStatus,
)
from capabilities.common.keym.models import KeyAlgorithm, KeyUsage, create_key_spec_async


def _all_provider_config() -> dict[str, dict[str, object]]:
	return {
		provider.value: {
			"enabled": True,
			"region": "test-region",
			"api_key": f"{provider.value}-credential",
		}
		for provider in CloudProvider
	}


@pytest.mark.asyncio
async def test_cloud_federation_initializes_all_provider_adapters():
	manager = CloudKeyFederationManager(_all_provider_config())

	await manager.initialize_cloud_providers()

	assert set(manager.cloud_clients) == set(CloudProvider)
	for provider, client in manager.cloud_clients.items():
		assert client["provider"] == provider
		assert client["status"] == "connected"
		assert client["adapter_mode"] == "external_provider_boundary"
		assert client["client"]["type"] == "apg_key_provider_adapter"
		assert client["client"]["provider"] == provider.value
		assert client["credentials_configured"] is True
		assert client["primary_region"] == "test-region"


@pytest.mark.asyncio
async def test_cloud_federation_lifecycle_across_provider_adapters():
	manager = CloudKeyFederationManager(_all_provider_config())
	await manager.initialize_cloud_providers()
	key_spec = await create_key_spec_async(
		tenant_id="tenant-keym",
		algorithm=KeyAlgorithm.AES_256,
		usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
		name="Federated Ledger Key",
		created_by="security@datacraft.co.ke",
	)
	policy = FederationPolicy(
		primary_provider=CloudProvider.AWS,
		backup_providers=[
			CloudProvider.AZURE,
			CloudProvider.GCP,
			CloudProvider.IBM_CLOUD,
			CloudProvider.ORACLE_CLOUD,
			CloudProvider.ALIBABA_CLOUD,
			CloudProvider.DIGITAL_OCEAN,
			CloudProvider.VULTR,
		],
		replication_regions={provider: ["test-region"] for provider in CloudProvider},
	)

	references = await manager.create_federated_key(key_spec, policy)
	sync_results = await manager.sync_federated_key(key_spec.id)
	rotation_results = await manager.rotate_federated_key(key_spec.id)
	failover_ref = await manager.failover_to_backup(key_spec.id, CloudProvider.AWS)
	migrated = await manager.migrate_key_between_providers(
		key_spec.id,
		CloudProvider.AZURE,
		CloudProvider.VULTR,
	)
	status = await manager.get_federation_status(key_spec.id)

	assert {ref.provider for ref in references} == set(CloudProvider)
	assert set(sync_results) == set(CloudProvider) - {CloudProvider.AWS}
	assert all(result == SyncStatus.IN_SYNC for result in sync_results.values())
	assert set(rotation_results) == set(CloudProvider)
	assert all(rotation_results.values())
	assert failover_ref is not None
	assert failover_ref.provider == CloudProvider.AZURE
	assert manager.federation_policies[key_spec.id].primary_provider == CloudProvider.AZURE
	assert migrated is True
	assert status["overall_status"] == "healthy"
	assert status["providers"][CloudProvider.AWS.value]["key_id"].startswith("aws-kms-")
	assert status["providers"][CloudProvider.VULTR.value]["sync_status"] == SyncStatus.IN_SYNC.value
