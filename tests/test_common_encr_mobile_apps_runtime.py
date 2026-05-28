import base64

import pytest

from capabilities.common.encr.mobile_apps import (
	AndroidKeystoreConfig,
	AndroidNativeIntegration,
	MobileAppManager,
	MobilePlatform,
)


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


@pytest.mark.asyncio
async def test_mobile_app_manager_android_hardware_encrypt_decrypt_round_trip():
	manager = MobileAppManager("tenant_runtime")
	device = await manager.register_device(
		MobilePlatform.ANDROID,
		{"model": "Pixel Runtime", "os_version": "15", "has_strongbox": True},
	)
	app = await manager.install_app(device.id, {"version": "1.0.0"})

	encrypted = await manager.perform_encryption_operation(app.id, "encrypt", b"manager mobile payload")
	decrypted = await manager.perform_encryption_operation(
		app.id,
		"decrypt",
		encrypted["result"]["encrypted_data"].encode("utf-8"),
	)

	assert encrypted["success"] is True
	assert encrypted["security"]["secure_element_used"] is True
	assert base64.b64decode(decrypted["result"]["decrypted_data"]) == b"manager mobile payload"


@pytest.mark.asyncio
async def test_mobile_app_manager_ios_software_encrypt_decrypt_round_trip():
	manager = MobileAppManager("tenant_runtime")
	device = await manager.register_device(
		MobilePlatform.IOS,
		{"model": "iPhone Runtime", "os_version": "17", "has_secure_enclave": False},
	)
	app = await manager.install_app(device.id, {"version": "1.0.0"})

	encrypted = await manager.perform_encryption_operation(app.id, "encrypt", b"ios software payload")
	decrypted = await manager.perform_encryption_operation(
		app.id,
		"decrypt",
		encrypted["result"]["encrypted_data"].encode("utf-8"),
	)

	assert encrypted["success"] is True
	assert encrypted["security"]["secure_element_used"] is False
	assert base64.b64decode(decrypted["result"]["decrypted_data"]) == b"ios software payload"
