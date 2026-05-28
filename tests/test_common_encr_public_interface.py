"""Public ENCR interface regressions."""

from __future__ import annotations

import pytest

from capabilities.common.encr import APGEncryptionInterface, encryption_interface


@pytest.mark.asyncio
async def test_public_quantum_safe_interface_round_trips_bytes():
	"""The exported ENCR interface should be executable, not a NotImplemented facade."""
	interface = APGEncryptionInterface()
	tenant_id = "tenant123"
	plaintext = b"general-ledger journal payload"

	encrypted = await interface.encrypt_quantum_safe(plaintext, tenant_id)
	decrypted = await interface.decrypt_quantum_safe(encrypted, tenant_id)

	assert encrypted.startswith(b"APG_ENCR:")
	assert encrypted != plaintext
	assert decrypted == plaintext


@pytest.mark.asyncio
async def test_public_zero_knowledge_interface_returns_proof_artifact():
	tenant_id = "tenant123"
	result = await encryption_interface.encrypt_zero_knowledge(
		b"payroll-record",
		{"tenant_id": tenant_id, "user_id": "controller"},
	)

	assert result["encrypted_data"].startswith(b"APG_ENCR:")
	assert result["session_id"]
	assert len(result["access_proof"]) == 64
	assert result["privacy_guarantee_level"] >= 0.99


@pytest.mark.asyncio
async def test_public_encrypted_computation_emits_decryptable_result():
	interface = APGEncryptionInterface()
	tenant_id = "tenant123"
	values = [
		await interface.encrypt_quantum_safe(b"10", tenant_id),
		await interface.encrypt_quantum_safe(b"32", tenant_id),
	]

	encrypted_result = await interface.compute_on_encrypted_data(
		values,
		"add",
		tenant_id=tenant_id,
	)
	result = interface._open_envelope(encrypted_result, tenant_id, expected_mode="homomorphic-result")

	assert result["plaintext"] == b"42.0"
	assert result["metadata"]["operation"] == "add"


@pytest.mark.asyncio
async def test_public_autonomous_key_lifecycle_returns_action_plan():
	interface = APGEncryptionInterface()

	decision = await interface.autonomous_key_lifecycle(
		"key_123",
		{
			"tenant_id": "tenant123",
			"key_age_days": 120,
			"usage_count": 1500,
			"threat_level": "critical",
		},
	)

	assert decision["key_id"] == "key_123"
	assert decision["tenant_id"] == "tenant123"
	assert decision["actions"] == ["rotate", "backup", "upgrade_quantum_safe"]
	assert decision["confidence"] > 0
