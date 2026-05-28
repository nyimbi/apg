import base64

import pytest

from capabilities.common.encr.mobile_apps import AndroidKeystoreConfig, AndroidNativeIntegration


@pytest.mark.asyncio
async def test_android_keystore_encrypt_decrypt_round_trips_plaintext():
	android = AndroidNativeIntegration("tenant_runtime")
	await android.initialize_android_keystore(AndroidKeystoreConfig(key_alias_prefix="apg_test"))
	key = await android.generate_keystore_key("payment_token")

	plaintext = b"mobile secret payload"
	encrypted = await android.encrypt_with_keystore(key["key_alias"], plaintext)
	decrypted = await android.decrypt_with_keystore(
		key["key_alias"],
		encrypted["ciphertext"],
		encrypted["iv"],
	)

	assert encrypted["ciphertext"] != base64.b64encode(plaintext).decode()
	assert base64.b64decode(decrypted["plaintext"]) == plaintext


@pytest.mark.asyncio
async def test_android_keystore_rejects_tampered_ciphertext():
	android = AndroidNativeIntegration("tenant_runtime")
	await android.initialize_android_keystore(AndroidKeystoreConfig(key_alias_prefix="apg_test"))
	key = await android.generate_keystore_key("payment_token")

	encrypted = await android.encrypt_with_keystore(key["key_alias"], b"mobile secret payload")
	tampered = bytearray(base64.b64decode(encrypted["ciphertext"]))
	tampered[-1] ^= 1

	result = await android.decrypt_with_keystore(
		key["key_alias"],
		base64.b64encode(tampered).decode(),
		encrypted["iv"],
	)

	assert result == {"error": "Ciphertext authentication failed", "key_alias": key["key_alias"]}
