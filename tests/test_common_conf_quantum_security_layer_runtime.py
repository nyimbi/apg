import json

import pytest

from capabilities.common.conf.quantum_security_layer import QuantumSecurityManager


@pytest.mark.asyncio
async def test_quantum_secure_configuration_round_trips_with_authenticated_encryption():
	manager = QuantumSecurityManager("tenant-conf-runtime")
	config = {
		"database": {"host": "db.internal", "password": "secret-value"},
		"features": {"billing": True, "inventory": False},
	}

	secure_config_id = await manager.quantum_encrypt_configuration("config-main", config)
	secure_config = manager.secure_configs[secure_config_id]

	assert secure_config.encrypted_data != json.dumps(config, sort_keys=True).encode("utf-8")
	assert secure_config.initialization_vector is not None
	assert len(secure_config.initialization_vector) == 12
	assert secure_config.authentication_tag is not None
	assert len(secure_config.authentication_tag) == 16
	assert secure_config.metadata["signature_key_id"] in manager.keys

	decrypted = await manager.quantum_decrypt_configuration(secure_config_id)

	assert decrypted == config
	assert any(operation.operation_type == "encryption" and operation.success for operation in manager.operation_history)
	assert any(operation.operation_type == "decryption" and operation.success for operation in manager.operation_history)


@pytest.mark.asyncio
async def test_quantum_secure_configuration_rejects_tampered_ciphertext():
	manager = QuantumSecurityManager("tenant-conf-runtime-tamper")
	secure_config_id = await manager.quantum_encrypt_configuration(
		"config-sensitive",
		{"api_key": "do-not-accept-tampering"},
	)
	secure_config = manager.secure_configs[secure_config_id]
	tampered = bytearray(secure_config.encrypted_data)
	tampered[0] ^= 0x01
	secure_config.encrypted_data = bytes(tampered)

	with pytest.raises(ValueError, match="signature|Authentication tag"):
		await manager.quantum_decrypt_configuration(secure_config_id)
