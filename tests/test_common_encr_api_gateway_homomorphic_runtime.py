import pytest

from capabilities.common.encr.api_gateway import (
	APIRequest,
	EnterpriseAPIGateway,
	ValidationError,
)


def _request(path: str, body: dict, tenant_id: str = "tenantapi123") -> APIRequest:
	return APIRequest(
		tenant_id=tenant_id,
		endpoint_path=path,
		method="POST",
		body=body,
		client_ip="127.0.0.1",
		user_agent="pytest",
	)


@pytest.mark.asyncio
async def test_api_gateway_homomorphic_encrypt_and_add_are_executable():
	gateway = EnterpriseAPIGateway()

	first = await gateway._handle_homomorphic_request(
		_request("/v1/homomorphic/encrypt", {"data": [10], "scheme": "ckks"}),
		None,
	)
	second = await gateway._handle_homomorphic_request(
		_request("/v1/homomorphic/encrypt", {"data": [32], "scheme": "ckks"}),
		None,
	)
	result = await gateway._handle_homomorphic_request(
		_request(
			"/v1/homomorphic/add",
			{
				"ciphertext1_id": first["ciphertext_id"],
				"ciphertext2_id": second["ciphertext_id"],
			},
		),
		None,
	)

	assert first["ciphertext_id"] in gateway.homomorphic_ciphertexts
	assert second["ciphertext_id"] in gateway.homomorphic_ciphertexts
	assert result["result_ciphertext_id"] in gateway.homomorphic_ciphertexts
	assert result["operation_count"] == 1
	assert result["noise_growth"] > 0
	assert len(result["result_payload_hash"]) == 64
	assert b'"result":42.0' in gateway.homomorphic_ciphertexts[result["result_ciphertext_id"]].ciphertext_data


@pytest.mark.asyncio
async def test_api_gateway_homomorphic_encrypt_rejects_non_numeric_payloads():
	gateway = EnterpriseAPIGateway()

	with pytest.raises(ValidationError, match="numeric"):
		await gateway._handle_homomorphic_request(
			_request("/v1/homomorphic/encrypt", {"data": ["not-a-number"], "scheme": "ckks"}),
			None,
		)
