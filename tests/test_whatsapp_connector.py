"""Tests for WhatsApp Business Cloud API connector."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from capabilities.composition.orchestration.connectors.whatsapp_connector import (
    WhatsAppConnector, WhatsAppConfiguration,
)

TENANT = "tenant-wa"
USER = "admin"

CONFIG = WhatsAppConfiguration(
    name="WhatsApp Test",
    tenant_id=TENANT,
    user_id=USER,
    access_token="test-token",
    phone_number_id="12345678",
)


def make_connector():
    return WhatsAppConnector(CONFIG)


def mock_response(data: dict, status: int = 200):
    resp = MagicMock()
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    resp.status_code = status
    return resp


# ── send_text ────────────────────────────────────────────────────────────────

async def test_send_text_posts_correct_payload():
    connector = make_connector()
    mock_resp = mock_response({"messages": [{"id": "msg-001"}]})
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    connector._client = mock_client

    result = await connector.send_text("254712345678", "Hello from APG!")

    mock_client.post.assert_called_once()
    payload = mock_client.post.call_args[1]["json"]
    assert payload["messaging_product"] == "whatsapp"
    assert payload["type"] == "text"
    assert payload["to"] == "254712345678"
    assert payload["text"]["body"] == "Hello from APG!"


async def test_send_text_strips_leading_plus():
    connector = make_connector()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response({}))
    connector._client = mock_client

    await connector.send_text("+254712345678", "Test")
    payload = mock_client.post.call_args[1]["json"]
    assert not payload["to"].startswith("+")


async def test_send_text_truncates_to_4096_chars():
    connector = make_connector()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response({}))
    connector._client = mock_client

    await connector.send_text("254712345678", "x" * 5000)
    payload = mock_client.post.call_args[1]["json"]
    assert len(payload["text"]["body"]) <= 4096


# ── send_template ────────────────────────────────────────────────────────────

async def test_send_template_posts_correct_payload():
    connector = make_connector()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response({"messages": [{"id": "msg-002"}]}))
    connector._client = mock_client

    result = await connector.send_template(
        "254712345678",
        template_name="payment_confirmation",
        language_code="sw",
        components=[{"type": "body", "parameters": [{"type": "text", "text": "1000"}]}],
    )

    payload = mock_client.post.call_args[1]["json"]
    assert payload["type"] == "template"
    assert payload["template"]["name"] == "payment_confirmation"
    assert payload["template"]["language"]["code"] == "sw"
    assert len(payload["template"]["components"]) == 1


# ── send_interactive_buttons ─────────────────────────────────────────────────

async def test_send_interactive_buttons_limits_to_3():
    connector = make_connector()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response({}))
    connector._client = mock_client

    buttons = [
        {"id": "btn1", "title": "Option 1"},
        {"id": "btn2", "title": "Option 2"},
        {"id": "btn3", "title": "Option 3"},
        {"id": "btn4", "title": "Option 4"},  # Should be trimmed
    ]
    await connector.send_interactive_buttons("254712345678", "Choose:", buttons)
    payload = mock_client.post.call_args[1]["json"]
    assert len(payload["interactive"]["action"]["buttons"]) == 3


async def test_send_interactive_buttons_payload_structure():
    connector = make_connector()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response({}))
    connector._client = mock_client

    await connector.send_interactive_buttons(
        "254712345678", "Confirm payment?",
        [{"id": "yes", "title": "Yes"}, {"id": "no", "title": "No"}],
        header_text="Payment Alert",
    )
    payload = mock_client.post.call_args[1]["json"]
    assert payload["type"] == "interactive"
    assert payload["interactive"]["type"] == "button"
    assert payload["interactive"]["header"]["text"] == "Payment Alert"


# ── mark_read ────────────────────────────────────────────────────────────────

async def test_mark_read_posts_correct_payload():
    connector = make_connector()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response({"success": True}))
    connector._client = mock_client

    await connector.mark_read("msg-incoming-001")
    payload = mock_client.post.call_args[1]["json"]
    assert payload["status"] == "read"
    assert payload["message_id"] == "msg-incoming-001"
    assert payload["messaging_product"] == "whatsapp"


# ── Webhook verification ──────────────────────────────────────────────────────

def test_verify_webhook_returns_challenge_on_valid():
    connector = make_connector()
    result = connector.verify_webhook(
        mode="subscribe", token="my-secret", challenge="abc123", verify_token="my-secret"
    )
    assert result == "abc123"


def test_verify_webhook_returns_none_on_invalid_token():
    connector = make_connector()
    result = connector.verify_webhook(
        mode="subscribe", token="wrong-token", challenge="abc123", verify_token="my-secret"
    )
    assert result is None


def test_verify_webhook_returns_none_on_wrong_mode():
    connector = make_connector()
    result = connector.verify_webhook(
        mode="unsubscribe", token="my-secret", challenge="abc123", verify_token="my-secret"
    )
    assert result is None


# ── Registry ─────────────────────────────────────────────────────────────────

def test_connector_registry_lists_whatsapp_installed():
    from capabilities.composition.orchestration.connectors.connector_registry import ConnectorRegistry
    r = ConnectorRegistry()
    assert "whatsapp" in r.list_installed()


def test_connector_registry_whatsapp_metadata():
    from capabilities.composition.orchestration.connectors.connector_registry import ConnectorRegistry
    r = ConnectorRegistry()
    meta = r.get_metadata("whatsapp")
    assert meta is not None
    assert meta["category"] == "messaging"
    assert "WHATSAPP_ACCESS_TOKEN" in meta["required_env"]
