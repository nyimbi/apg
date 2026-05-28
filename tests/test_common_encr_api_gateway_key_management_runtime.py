import base64

import pytest

from capabilities.common.encr.api_gateway import APIRequest, EnterpriseAPIGateway
from capabilities.common.encr.models import PostQuantumAlgorithm


def _request(path: str, method: str = "GET", body: dict | None = None) -> APIRequest:
	return APIRequest(
		tenant_id="tenantkey123",
		endpoint_path=path,
		method=method,
		body=body,
		client_ip="127.0.0.1",
		user_agent="pytest",
	)


@pytest.mark.asyncio
async def test_api_gateway_generates_and_lists_real_tenant_keys():
	gateway = EnterpriseAPIGateway()

	generated = await gateway._handle_key_management_request(
		_request(
			"/v1/keys/generate",
			"POST",
			{
				"algorithm": PostQuantumAlgorithm.CRYSTALS_KYBER_512.value,
				"security_level": "level_3",
				"key_metadata": {"purpose": "api-test"},
			},
		),
		None,
	)
	listed = await gateway._handle_key_management_request(_request("/v1/keys"), None)

	assert generated["key_id"] in gateway.encryption_service.post_quantum_crypto.keypairs
	assert base64.b64decode(generated["public_key"])
	assert listed["total_count"] == 1
	assert listed["keys"][0]["key_id"] == generated["key_id"]
	assert listed["keys"][0]["usage_context"] == {"purpose": "api-test"}
	assert len(listed["keys"][0]["public_key_fingerprint"]) == 64


@pytest.mark.asyncio
async def test_api_gateway_lists_empty_key_inventory_without_fabricating_keys():
	gateway = EnterpriseAPIGateway()

	listed = await gateway._handle_key_management_request(_request("/v1/keys"), None)

	assert listed["keys"] == []
	assert listed["total_count"] == 0
