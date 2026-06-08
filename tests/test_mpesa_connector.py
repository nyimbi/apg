"""Tests for MPESA Daraja 2.0 connector.

Tests validate the connector interface, API request construction, OAuth token
caching, and error handling — without requiring a live MPESA account.
pytest-httpserver is used for HTTP mocking per project conventions.
"""
from __future__ import annotations

import base64
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


TENANT = "tenant-ke"
USER = "admin"

SAMPLE_CONFIG = {
	"name": "MPESA Test",
	"tenant_id": TENANT,
	"user_id": USER,
	"consumer_key": "test-consumer-key",
	"consumer_secret": "test-consumer-secret",
	"shortcode": "174379",
	"passkey": "bfb279f9aa9bdbcf158e97dd71a467cd2e0c893059b10f78e6b72ada1ed2c919",
	"environment": "sandbox",
	"initiator_name": "testapi",
	"initiator_password": "Safaricom987!",
	"callback_url_base": "https://myapp.example.com",
}

TOKEN_RESPONSE = {
	"access_token": "test-access-token-abc123",
	"expires_in": "3599",
}

STK_PUSH_RESPONSE = {
	"MerchantRequestID": "29115-34620561-1",
	"CheckoutRequestID": "ws_CO_191220191020363925",
	"ResponseCode": "0",
	"ResponseDescription": "Success. Request accepted for processing",
	"CustomerMessage": "Success. Request accepted for processing",
}

B2C_RESPONSE = {
	"ConversationID": "AG_20191219_00005797af5d7d75f652",
	"OriginatorConversationID": "16740-34861180-1",
	"ResponseCode": "0",
	"ResponseDescription": "Accept the service request successfully.",
}


def make_connector():
	from capabilities.composition.orchestration.connectors.africa.mpesa_connector import (
		MPESAConnector,
		MPESAConfiguration,
	)
	config = MPESAConfiguration(**SAMPLE_CONFIG)
	return MPESAConnector(config)


# ── MPESAConfiguration ────────────────────────────────────────────────────────

def test_mpesa_configuration_validates_environment():
	from capabilities.composition.orchestration.connectors.africa.mpesa_connector import MPESAConfiguration
	from pydantic import ValidationError

	with pytest.raises(ValidationError):
		MPESAConfiguration(**{**SAMPLE_CONFIG, "environment": "invalid_env"})


def test_mpesa_configuration_valid_production():
	from capabilities.composition.orchestration.connectors.africa.mpesa_connector import MPESAConfiguration
	cfg = MPESAConfiguration(**{**SAMPLE_CONFIG, "environment": "production"})
	assert cfg.environment == "production"


# ── OAuth token management ────────────────────────────────────────────────────

async def test_refresh_token_sets_token_and_expiry():
	connector = make_connector()

	mock_resp = MagicMock()
	mock_resp.json.return_value = TOKEN_RESPONSE
	mock_resp.raise_for_status = MagicMock()

	mock_client = AsyncMock()
	mock_client.get = AsyncMock(return_value=mock_resp)
	connector._client = mock_client

	await connector._refresh_token()

	assert connector._token == "test-access-token-abc123"
	assert connector._token_expires_at > time.time()


async def test_refresh_token_skipped_when_still_valid():
	connector = make_connector()
	connector._token = "existing-token"
	connector._token_expires_at = time.time() + 3600  # Not expired

	mock_client = AsyncMock()
	connector._client = mock_client

	await connector._refresh_token()

	# Should NOT have called the OAuth endpoint
	mock_client.get.assert_not_called()


async def test_refresh_token_includes_basic_auth():
	connector = make_connector()

	mock_resp = MagicMock()
	mock_resp.json.return_value = TOKEN_RESPONSE
	mock_resp.raise_for_status = MagicMock()

	mock_client = AsyncMock()
	mock_client.get = AsyncMock(return_value=mock_resp)
	connector._client = mock_client

	await connector._refresh_token()

	call_kwargs = mock_client.get.call_args
	auth_header = call_kwargs[1]["headers"]["Authorization"]
	assert auth_header.startswith("Basic ")

	# Verify the encoded credentials are correct
	encoded = auth_header[len("Basic "):]
	decoded = base64.b64decode(encoded).decode()
	assert decoded == "test-consumer-key:test-consumer-secret"


# ── STK Push ──────────────────────────────────────────────────────────────────

async def test_stk_push_posts_to_correct_endpoint():
	connector = make_connector()
	connector._token = "test-token"
	connector._token_expires_at = time.time() + 3600

	mock_resp = MagicMock()
	mock_resp.json.return_value = STK_PUSH_RESPONSE
	mock_resp.raise_for_status = MagicMock()

	mock_client = AsyncMock()
	mock_client.post = AsyncMock(return_value=mock_resp)
	connector._client = mock_client

	result = await connector.stk_push(
		amount=100,
		phone="254712345678",
		account_reference="ORDER-001",
		transaction_desc="Test payment",
	)

	assert result["ResponseCode"] == "0"
	assert result["CheckoutRequestID"] == "ws_CO_191220191020363925"

	call_args = mock_client.post.call_args
	assert "/mpesa/stkpush/v1/processrequest" in str(call_args[0][0])

	payload = call_args[1]["json"]
	assert payload["Amount"] == "100"
	assert payload["PhoneNumber"] == "254712345678"
	assert payload["BusinessShortCode"] == "174379"
	assert "Password" in payload
	assert "Timestamp" in payload


async def test_stk_push_truncates_long_reference():
	"""AccountReference is limited to 12 chars per MPESA spec."""
	connector = make_connector()
	connector._token = "test-token"
	connector._token_expires_at = time.time() + 3600

	mock_resp = MagicMock()
	mock_resp.json.return_value = STK_PUSH_RESPONSE
	mock_resp.raise_for_status = MagicMock()

	mock_client = AsyncMock()
	mock_client.post = AsyncMock(return_value=mock_resp)
	connector._client = mock_client

	await connector.stk_push(100, "254712345678", "A" * 20, "desc")

	payload = mock_client.post.call_args[1]["json"]
	assert len(payload["AccountReference"]) <= 12


# ── B2C ───────────────────────────────────────────────────────────────────────

async def test_b2c_payment_posts_correct_command_id():
	connector = make_connector()
	connector._token = "test-token"
	connector._token_expires_at = time.time() + 3600

	mock_resp = MagicMock()
	mock_resp.json.return_value = B2C_RESPONSE
	mock_resp.raise_for_status = MagicMock()

	mock_client = AsyncMock()
	mock_client.post = AsyncMock(return_value=mock_resp)
	connector._client = mock_client

	result = await connector.b2c_payment(
		amount=500,
		phone="254712345678",
		command_id="SalaryPayment",
		remarks="Monthly salary",
	)

	assert result["ResponseCode"] == "0"
	payload = mock_client.post.call_args[1]["json"]
	assert payload["CommandID"] == "SalaryPayment"
	assert payload["Amount"] == "500"
	assert payload["PartyB"] == "254712345678"


# ── execute_request routing ───────────────────────────────────────────────────

async def test_execute_operation_raises_for_unknown_operation():
	connector = make_connector()
	connector._token = "test-token"
	connector._token_expires_at = time.time() + 3600
	connector._client = AsyncMock()

	with pytest.raises(ValueError, match="Unknown MPESA operation"):
		await connector._execute_operation("nonexistent_op", {})


# ── Connector registry ────────────────────────────────────────────────────────

def test_connector_registry_lists_mpesa():
	from capabilities.composition.orchestration.connectors.connector_registry import ConnectorRegistry
	registry = ConnectorRegistry()
	available = {c["id"]: c for c in registry.list_available()}
	assert "mpesa" in available
	assert available["mpesa"]["category"] == "payment"
	assert "KE" in available["mpesa"]["regions"]


def test_connector_registry_mpesa_is_installed():
	from capabilities.composition.orchestration.connectors.connector_registry import ConnectorRegistry
	registry = ConnectorRegistry()
	installed = registry.list_installed()
	assert "mpesa" in installed


def test_connector_registry_unknown_raises():
	from capabilities.composition.orchestration.connectors.connector_registry import ConnectorRegistry
	registry = ConnectorRegistry()
	with pytest.raises(KeyError, match="Unknown connector"):
		registry.get("nonexistent", tenant_id=TENANT)


def test_connector_registry_planned_raises_import_error():
	from capabilities.composition.orchestration.connectors.connector_registry import ConnectorRegistry
	registry = ConnectorRegistry()
	with pytest.raises(ImportError, match="planned but not yet implemented"):
		registry.get("stripe", tenant_id=TENANT)


def test_connector_registry_get_metadata():
	from capabilities.composition.orchestration.connectors.connector_registry import ConnectorRegistry
	registry = ConnectorRegistry()
	meta = registry.get_metadata("mpesa")
	assert meta is not None
	assert meta["display_name"] == "MPESA (Safaricom Daraja 2.0)"
	assert "required_env" in meta


# ── Signature verification ────────────────────────────────────────────────────

def test_verify_callback_signature_valid():
	import hmac as _hmac
	import hashlib
	connector = make_connector()
	secret = "my-webhook-secret"
	payload = b'{"Body": {"stkCallback": {"ResultCode": 0}}}'
	sig = _hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
	assert connector.verify_callback_signature(payload, sig, secret) is True


def test_verify_callback_signature_invalid():
	connector = make_connector()
	payload = b'{"tampered": true}'
	assert connector.verify_callback_signature(payload, "wrong-signature", "secret") is False
