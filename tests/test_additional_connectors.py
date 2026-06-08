"""Tests for Equity Bank, KCB, and Stripe connectors."""
from __future__ import annotations
import json
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── Equity Bank ───────────────────────────────────────────────────────────────

def make_equity():
    from capabilities.composition.orchestration.connectors.africa.equity_connector import (
        EquityBankConnector, EquityBankConfiguration,
    )
    config = EquityBankConfiguration(
        name="Equity Test", tenant_id="tenant-eq", user_id="admin",
        client_id="test-client", client_secret="test-secret",
    )
    return EquityBankConnector(config)


async def test_equity_account_inquiry_calls_correct_endpoint():
    conn = make_equity()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"accountId": "123", "balance": 50000}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    conn._client = mock_client
    conn._token = "test-token"
    conn._token_expires_at = 9999999999.0

    result = await conn.account_inquiry("ACC-001")
    assert result["accountId"] == "123"
    call_args = mock_client.get.call_args[0][0]
    assert "ACC-001" in call_args


async def test_equity_internal_transfer_posts_payload():
    conn = make_equity()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"transactionId": "TXN-001", "status": "success"}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    conn._client = mock_client
    conn._token = "test-token"
    conn._token_expires_at = 9999999999.0

    result = await conn.internal_transfer("ACC-001", "ACC-002", 1000.0, "KES", "Test transfer")
    payload = mock_client.post.call_args[1]["json"]
    assert payload["source"]["accountId"] == "ACC-001"
    assert payload["transfer"]["amount"] == "1000.0"


async def test_equity_unknown_operation_raises():
    conn = make_equity()
    conn._client = AsyncMock()
    conn._token = "test-token"
    conn._token_expires_at = 9999999999.0

    with pytest.raises(ValueError, match="Unknown Equity Bank operation"):
        await conn._execute_operation("nonexistent", {})


def test_equity_in_registry():
    from capabilities.composition.orchestration.connectors.connector_registry import ConnectorRegistry
    r = ConnectorRegistry()
    assert "equity_bank" in r.list_installed()
    meta = r.get_metadata("equity_bank")
    assert "KE" in meta["regions"]


# ── KCB Bank ──────────────────────────────────────────────────────────────────

def make_kcb():
    from capabilities.composition.orchestration.connectors.africa.kcb_connector import (
        KCBConnector, KCBConfiguration,
    )
    config = KCBConfiguration(
        name="KCB Test", tenant_id="tenant-kcb", user_id="admin",
        consumer_key="test-key", consumer_secret="test-secret",
    )
    return KCBConnector(config)


async def test_kcb_account_inquiry():
    conn = make_kcb()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"accountNumber": "1234567890", "availableBalance": 25000}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    conn._client = mock_client
    conn._token = "test-token"
    conn._token_expires_at = 9999999999.0

    result = await conn.account_inquiry("1234567890")
    assert result["accountNumber"] == "1234567890"


async def test_kcb_bulk_payroll_posts_records():
    conn = make_kcb()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"batchId": "BATCH-001", "processed": 2}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    conn._client = mock_client
    conn._token = "test-token"
    conn._token_expires_at = 9999999999.0

    records = [
        {"employee_id": "E001", "account_number": "ACC-001", "amount": 50000},
        {"employee_id": "E002", "account_number": "ACC-002", "amount": 60000},
    ]
    result = await conn.bulk_payroll("SRC-ACC", records)
    payload = mock_client.post.call_args[1]["json"]
    assert payload["sourceAccount"] == "SRC-ACC"
    assert len(payload["transactions"]) == 2


def test_kcb_in_registry():
    from capabilities.composition.orchestration.connectors.connector_registry import ConnectorRegistry
    r = ConnectorRegistry()
    assert "kcb" in r.list_installed()


# ── Stripe ────────────────────────────────────────────────────────────────────

def make_stripe():
    from capabilities.composition.orchestration.connectors.stripe_connector import (
        StripeConnector, StripeConfiguration,
    )
    config = StripeConfiguration(
        name="Stripe Test", tenant_id="tenant-stripe", user_id="admin",
        secret_key="sk_test_abc123",
    )
    return StripeConnector(config)


async def test_stripe_create_payment_intent_correct_payload():
    conn = make_stripe()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"id": "pi_test_001", "status": "requires_payment_method"}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    conn._client = mock_client

    result = await conn.create_payment_intent(1000, "kes")
    payload = mock_client.post.call_args[1]["data"]
    assert payload["amount"] == "1000"
    assert payload["currency"] == "kes"
    assert result["id"] == "pi_test_001"


async def test_stripe_create_refund_posts_payment_intent_id():
    conn = make_stripe()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"id": "re_test_001", "status": "succeeded"}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    conn._client = mock_client

    result = await conn.create_refund("pi_test_001", amount=500)
    payload = mock_client.post.call_args[1]["data"]
    assert payload["payment_intent"] == "pi_test_001"
    assert payload["amount"] == "500"


async def test_stripe_retrieve_balance():
    conn = make_stripe()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "available": [{"currency": "kes", "amount": 100000}]
    }
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    conn._client = mock_client

    result = await conn.retrieve_balance()
    assert result["available"][0]["currency"] == "kes"


def test_stripe_webhook_signature_verification():
    import hashlib, hmac, time
    conn = make_stripe()
    secret = "whsec_test_secret"
    payload = b'{"type": "payment_intent.succeeded"}'
    timestamp = str(int(time.time()))
    signed_payload = f"{timestamp}.{payload.decode()}".encode()
    sig = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()
    header = f"t={timestamp},v1={sig}"

    assert conn.verify_webhook_signature(payload, header, secret) is True
    assert conn.verify_webhook_signature(payload, f"t={timestamp},v1=wrong", secret) is False


def test_stripe_in_registry():
    from capabilities.composition.orchestration.connectors.connector_registry import ConnectorRegistry
    r = ConnectorRegistry()
    assert "stripe" in r.list_installed()
    meta = r.get_metadata("stripe")
    assert meta["category"] == "payment"
