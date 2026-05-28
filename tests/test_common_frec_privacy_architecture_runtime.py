import base64

import pytest

from capabilities.common.frec.privacy_architecture import PrivacyArchitectureEngine


async def _grant_default_consent(engine: PrivacyArchitectureEngine, user_id: str) -> None:
	result = await engine.manage_user_consent(
		user_id,
		{
			"consent_given": True,
			"allowed_purposes": ["identity_verification"],
			"allowed_data_categories": ["facial_biometric"],
			"legal_basis": "consent",
		},
	)
	assert result["success"] is True


@pytest.mark.asyncio
async def test_homomorphic_privacy_processing_returns_executable_encrypted_result():
	engine = PrivacyArchitectureEngine("tenant_runtime")
	await _grant_default_consent(engine, "user_runtime")

	result = await engine.process_with_privacy(
		b"facial-biometric-sample",
		{
			"user_id": "user_runtime",
			"privacy_level": "maximum",
			"processing_mode": "homomorphic",
			"processing_purpose": "identity_verification",
			"data_categories": ["facial_biometric"],
			"retention_policy": "short_term",
		},
	)

	assert result["success"] is True
	encrypted_result = base64.b64decode(result["encrypted_result"])
	assert len(encrypted_result) == 32
	assert any(encrypted_result)
	assert "encrypted_domain_computation" in result["privacy_metadata"]["privacy_techniques_applied"]
	assert result["privacy_metadata"]["data_minimization"] is True


@pytest.mark.asyncio
async def test_on_device_template_generation_returns_protected_bytes():
	engine = PrivacyArchitectureEngine("tenant_runtime")
	await _grant_default_consent(engine, "user_runtime")

	result = await engine.process_with_privacy(
		b"on-device-biometric-sample",
		{
			"user_id": "user_runtime",
			"privacy_level": "enhanced",
			"processing_mode": "on_device",
			"processing_purpose": "identity_verification",
			"data_categories": ["facial_biometric"],
			"retention_policy": "session_only",
		},
	)
	template = engine._generate_privacy_preserving_template(b"on-device-biometric-sample")

	assert result["success"] is True
	assert result["local_template_size"] == len(template)
	assert len(template) > 0
	assert "local_template_generation" in result["privacy_metadata"]["privacy_techniques_applied"]
