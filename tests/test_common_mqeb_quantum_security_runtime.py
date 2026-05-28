import pytest

from capabilities.common.mqeb.models import MQMessage
from capabilities.common.mqeb.quantum_security import (
	QuantumAlgorithm,
	QuantumKeyManager,
	QuantumSecurityEngine,
)


@pytest.mark.asyncio
async def test_kyber_simulation_round_trips_public_encrypt_private_decrypt():
	ops = QuantumKeyManager()._kyber_operations
	public_key, private_key = await ops.generate_keypair()

	plaintext = b"Test message for Kyber-compatible APG envelope"
	ciphertext = await ops.encrypt(public_key, plaintext)

	assert ciphertext != plaintext
	assert ciphertext.startswith(b"APG-MQEB-KYBER-SIM-V1:")
	assert await ops.decrypt(private_key, ciphertext) == plaintext


@pytest.mark.asyncio
async def test_quantum_security_engine_encrypts_and_decrypts_message_payload():
	engine = QuantumSecurityEngine(mqeb_service=None)
	message = MQMessage(
		topic="financial.payment.authorized",
		payload=b"sensitive ledger event",
		tenant_id="tenant_runtime",
		source_application="payment_service",
	)

	assert await engine.encrypt_message(message, {"authenticated": True})
	assert message.encrypted is True
	assert message.encryption_key_id is not None
	assert message.headers["quantum_algorithm"] == QuantumAlgorithm.CRYSTALS_KYBER_512.value

	decrypted = await engine.decrypt_message(message, {"authenticated": True})

	assert decrypted == b"sensitive ledger event"
