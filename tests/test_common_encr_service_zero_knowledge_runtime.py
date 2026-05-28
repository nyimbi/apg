import pytest

from capabilities.common.encr.service import (
	ProofVerificationError,
	ThresholdCryptographyError,
	ZeroKnowledgeEncryptionEngine,
)


@pytest.mark.asyncio
async def test_zero_knowledge_threshold_encryption_round_trips_with_required_shares():
	engine = ZeroKnowledgeEncryptionEngine()
	await engine.initialize()
	tenant_id = "tenantzk123"
	client_key = await engine.derive_client_key("biometric-fingerprint-template", tenant_id)
	server_key = await engine.generate_server_key_share(tenant_id, "operation-001")
	plaintext = b"payroll bank account payload"

	encrypted_data, threshold_shares = await engine.threshold_encrypt(
		plaintext,
		client_key,
		server_key,
		threshold=2,
	)
	decrypted = await engine.threshold_decrypt(encrypted_data, threshold_shares)

	assert encrypted_data.startswith(b"APG_ZK:")
	assert encrypted_data != plaintext
	assert len(threshold_shares) == 2
	assert all(share.startswith(b"APG_ZK_SHARE:") for share in threshold_shares)
	assert decrypted == plaintext


@pytest.mark.asyncio
async def test_zero_knowledge_threshold_decryption_rejects_tampered_share():
	engine = ZeroKnowledgeEncryptionEngine()
	await engine.initialize()
	client_key = await engine.derive_client_key("biometric-context", "tenantzk456")
	server_key = await engine.generate_server_key_share("tenantzk456", "operation-002")
	encrypted_data, threshold_shares = await engine.threshold_encrypt(
		b"restricted field",
		client_key,
		server_key,
		threshold=2,
	)
	tampered_share = bytearray(threshold_shares[0])
	tampered_share[-1] = ord(b"A") if tampered_share[-1] != ord(b"A") else ord(b"B")
	threshold_shares[0] = bytes(tampered_share)

	with pytest.raises(ThresholdCryptographyError):
		await engine.threshold_decrypt(encrypted_data, threshold_shares)


@pytest.mark.asyncio
async def test_zero_knowledge_access_proof_verifies_and_rejects_wrong_tenant():
	engine = ZeroKnowledgeEncryptionEngine()
	await engine.initialize()
	proof = await engine.generate_access_proof(
		{"tenant_id": "tenantzk789", "user_id": "controller", "session_id": "session-1"},
		b"encrypted-envelope",
		{"purpose": "payroll_review"},
	)

	assert await engine.verify_access_proof(proof, {"tenant_id": "tenantzk789"}) is True

	with pytest.raises(ProofVerificationError, match="tenant mismatch"):
		await engine.verify_access_proof(proof, {"tenant_id": "other-tenant"})
