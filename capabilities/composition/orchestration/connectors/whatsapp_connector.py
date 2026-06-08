"""WhatsApp Business Cloud API Connector.

WhatsApp Business is the dominant messaging platform across Africa, South Asia,
and Latin America — used by 2B+ users. The Cloud API (Meta Graph API) enables
APG capabilities (ckm_not, crm_adv) to send transactional messages, templates,
and interactive components.

Reference: https://developers.facebook.com/docs/whatsapp/cloud-api

Authentication: Bearer token (permanent system user access token from Meta)
Base URL: https://graph.facebook.com/v21.0
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from pydantic import Field

from .base_connector import BaseConnector, ConnectorConfiguration

_log = logging.getLogger(__name__)
_GRAPH_BASE = "https://graph.facebook.com"
_API_VERSION = "v21.0"


class WhatsAppConfiguration(ConnectorConfiguration):
	access_token: str = Field(..., description="Meta system user access token")
	phone_number_id: str = Field(..., description="WhatsApp Business phone number ID")
	business_account_id: str = Field(default="", description="WhatsApp Business Account ID")
	api_version: str = Field(default=_API_VERSION, description="Meta Graph API version")


class WhatsAppConnector(BaseConnector):
	"""WhatsApp Business Cloud API connector.

	Supports:
	  - Send text messages
	  - Send template messages (pre-approved business messages)
	  - Send interactive messages (buttons, lists)
	  - Send media (images, documents, audio, video)
	  - Mark messages as read
	  - Upload media files
	  - Webhook verification
	"""

	def __init__(self, config: WhatsAppConfiguration) -> None:
		super().__init__(config)
		self._config: WhatsAppConfiguration = config
		self._base_url = f"{_GRAPH_BASE}/{config.api_version}/{config.phone_number_id}"
		self._client: httpx.AsyncClient | None = None

	async def _connect(self) -> None:
		self._client = httpx.AsyncClient(
			base_url=self._base_url,
			timeout=self._config.timeout_seconds,
			headers={
				"Authorization": f"Bearer {self._config.access_token}",
				"Content-Type": "application/json",
			},
		)

	async def _disconnect(self) -> None:
		if self._client:
			await self._client.aclose()
			self._client = None

	async def _execute_operation(self, operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
		handlers = {
			"send_text": self._send_text,
			"send_template": self._send_template,
			"send_interactive_buttons": self._send_interactive_buttons,
			"send_image": self._send_media,
			"send_document": self._send_media,
			"mark_read": self._mark_read,
		}
		handler = handlers.get(operation)
		if handler is None:
			raise ValueError(f"Unknown WhatsApp operation: {operation!r}")
		return await handler(**parameters)

	async def _health_check(self) -> bool:
		try:
			resp = await self._client.get(
				f"/{_API_VERSION}/{self._config.phone_number_id}",
				params={"access_token": self._config.access_token},
			)
			return resp.status_code == 200
		except Exception:
			return False

	# ── Public operations ──────────────────────────────────────────────

	async def send_text(self, to: str, text: str, preview_url: bool = False) -> dict[str, Any]:
		"""Send a plain text message.

		Args:
			to: Phone number in E.164 format (e.g. "254712345678")
			text: Message body (max 4096 chars)
			preview_url: Show URL preview if text contains a link
		"""
		return await self._execute_operation("send_text", {
			"to": to, "text": text, "preview_url": preview_url,
		})

	async def send_template(
		self,
		to: str,
		template_name: str,
		language_code: str = "en",
		components: list[dict[str, Any]] | None = None,
	) -> dict[str, Any]:
		"""Send a pre-approved business template message.

		Templates must be approved by Meta before use. Use for transactional
		messages like payment confirmations, appointment reminders, OTPs.

		Args:
			to: Recipient phone number in E.164 format
			template_name: Name of the approved template
			language_code: Language code (en, sw, fr, etc.)
			components: List of template component objects (header, body, buttons)
		"""
		return await self._execute_operation("send_template", {
			"to": to, "template_name": template_name,
			"language_code": language_code, "components": components or [],
		})

	async def send_interactive_buttons(
		self,
		to: str,
		body_text: str,
		buttons: list[dict[str, str]],
		header_text: str = "",
		footer_text: str = "",
	) -> dict[str, Any]:
		"""Send an interactive message with reply buttons (max 3 buttons).

		Args:
			buttons: List of {"id": "...", "title": "..."} dicts (max 3)
		"""
		return await self._execute_operation("send_interactive_buttons", {
			"to": to, "body_text": body_text, "buttons": buttons[:3],
			"header_text": header_text, "footer_text": footer_text,
		})

	async def send_image(
		self, to: str, image_url: str, caption: str = ""
	) -> dict[str, Any]:
		"""Send an image by URL."""
		return await self._execute_operation("send_image", {
			"to": to, "media_type": "image", "url": image_url, "caption": caption,
		})

	async def send_document(
		self, to: str, document_url: str, filename: str = "", caption: str = ""
	) -> dict[str, Any]:
		"""Send a document (PDF, DOCX, etc.) by URL."""
		return await self._execute_operation("send_document", {
			"to": to, "media_type": "document", "url": document_url,
			"filename": filename, "caption": caption,
		})

	async def mark_read(self, message_id: str) -> dict[str, Any]:
		"""Mark an incoming message as read (shows blue ticks to sender)."""
		return await self._execute_operation("mark_read", {"message_id": message_id})

	def verify_webhook(
		self, mode: str, token: str, challenge: str, verify_token: str
	) -> str | None:
		"""Verify a WhatsApp webhook subscription request.

		Returns the challenge string if verification passes, None if it fails.
		"""
		if mode == "subscribe" and token == verify_token:
			return challenge
		return None

	# ── Private implementation ─────────────────────────────────────────

	def _messages_url(self) -> str:
		return "/messages"

	async def _send_text(self, to: str, text: str, preview_url: bool) -> dict[str, Any]:
		payload = {
			"messaging_product": "whatsapp",
			"recipient_type": "individual",
			"to": to.lstrip("+"),
			"type": "text",
			"text": {"body": text[:4096], "preview_url": preview_url},
		}
		resp = await self._client.post(self._messages_url(), json=payload)
		resp.raise_for_status()
		return resp.json()

	async def _send_template(
		self, to: str, template_name: str, language_code: str, components: list[dict[str, Any]]
	) -> dict[str, Any]:
		payload = {
			"messaging_product": "whatsapp",
			"to": to.lstrip("+"),
			"type": "template",
			"template": {
				"name": template_name,
				"language": {"code": language_code},
				"components": components,
			},
		}
		resp = await self._client.post(self._messages_url(), json=payload)
		resp.raise_for_status()
		return resp.json()

	async def _send_interactive_buttons(
		self, to: str, body_text: str, buttons: list[dict[str, str]],
		header_text: str, footer_text: str,
	) -> dict[str, Any]:
		interactive: dict[str, Any] = {
			"type": "button",
			"body": {"text": body_text},
			"action": {
				"buttons": [
					{"type": "reply", "reply": {"id": b["id"], "title": b["title"][:20]}}
					for b in buttons
				]
			},
		}
		if header_text:
			interactive["header"] = {"type": "text", "text": header_text}
		if footer_text:
			interactive["footer"] = {"text": footer_text}

		payload = {
			"messaging_product": "whatsapp",
			"to": to.lstrip("+"),
			"type": "interactive",
			"interactive": interactive,
		}
		resp = await self._client.post(self._messages_url(), json=payload)
		resp.raise_for_status()
		return resp.json()

	async def _send_media(
		self, to: str, media_type: str, url: str,
		caption: str = "", filename: str = "",
	) -> dict[str, Any]:
		media_obj: dict[str, Any] = {"link": url}
		if caption:
			media_obj["caption"] = caption[:1024]
		if filename and media_type == "document":
			media_obj["filename"] = filename

		payload = {
			"messaging_product": "whatsapp",
			"to": to.lstrip("+"),
			"type": media_type,
			media_type: media_obj,
		}
		resp = await self._client.post(self._messages_url(), json=payload)
		resp.raise_for_status()
		return resp.json()

	async def _mark_read(self, message_id: str) -> dict[str, Any]:
		payload = {
			"messaging_product": "whatsapp",
			"status": "read",
			"message_id": message_id,
		}
		resp = await self._client.post(self._messages_url(), json=payload)
		resp.raise_for_status()
		return resp.json()


def whatsapp_connector_from_env(tenant_id: str, user_id: str = "system") -> WhatsAppConnector:
	"""Construct WhatsAppConnector from environment variables."""
	config = WhatsAppConfiguration(
		name="WhatsApp Business",
		tenant_id=tenant_id,
		user_id=user_id,
		access_token=os.environ["WHATSAPP_ACCESS_TOKEN"],
		phone_number_id=os.environ["WHATSAPP_PHONE_NUMBER_ID"],
		business_account_id=os.environ.get("WHATSAPP_BUSINESS_ACCOUNT_ID", ""),
		api_version=os.environ.get("WHATSAPP_API_VERSION", _API_VERSION),
	)
	return WhatsAppConnector(config)
