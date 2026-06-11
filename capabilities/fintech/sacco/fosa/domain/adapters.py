"""Domain adapters for SACCO FOSA — event emission, M-PESA, card issuance."""
from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)


def get_audit_adapter(capability_id: str = "fintech_sacco_fosa"):
	"""Return a NATS event adapter when available, else None."""
	nats_url = os.environ.get("NATS_URL")
	if nats_url:
		try:
			from capabilities.common.nats.nats_adapter import NATSEventAdapter
			return NATSEventAdapter(capability_id)
		except Exception as exc:
			_log.debug("NATS unavailable: %s", exc)
	return None


class MpesaAdapter:
	"""Stub adapter for Safaricom M-PESA Daraja API.

	Production implementation connects to:
	  - C2B confirmation URL (webhook consumer)
	  - B2C payment initiation (REST client to Safaricom)

	Environment variables:
	  MPESA_CONSUMER_KEY, MPESA_CONSUMER_SECRET,
	  MPESA_SHORTCODE, MPESA_PASSKEY, MPESA_ENV (sandbox|production)
	"""

	def __init__(self) -> None:
		self.consumer_key = os.environ.get("MPESA_CONSUMER_KEY", "")
		self.consumer_secret = os.environ.get("MPESA_CONSUMER_SECRET", "")
		self.shortcode = os.environ.get("MPESA_SHORTCODE", "")
		self.env = os.environ.get("MPESA_ENV", "sandbox")
		self._enabled = bool(self.consumer_key and self.consumer_secret)

	@property
	def enabled(self) -> bool:
		return self._enabled

	async def initiate_b2c(
		self,
		phone_number: str,
		amount: str,
		reference: str,
		remarks: str = "FOSA withdrawal",
	) -> dict:
		"""Initiate B2C payment to member phone. Returns Safaricom response dict."""
		if not self._enabled:
			_log.info("M-PESA adapter not configured — returning mock B2C response")
			return {
				"ConversationID": f"mock-{reference}",
				"OriginatorConversationID": reference,
				"ResponseCode": "0",
				"ResponseDescription": "Accept the service request successfully.",
			}
		# Production: call Safaricom Daraja B2C endpoint
		raise NotImplementedError("Wire up Safaricom Daraja B2C integration")

	async def validate_c2b(self, mpesa_reference: str, amount: str, phone_number: str) -> bool:
		"""Validate incoming C2B payment (called by Daraja validation URL)."""
		# Production: verify against expected transactions
		return True

	async def confirm_c2b(self, payload: dict) -> dict:
		"""Process confirmed C2B payment (called by Daraja confirmation URL)."""
		return {
			"mpesa_reference": payload.get("TransID"),
			"amount": payload.get("TransAmount"),
			"phone_number": payload.get("MSISDN"),
			"transaction_time": payload.get("TransTime"),
		}


class CardIssuanceAdapter:
	"""Stub adapter for card bureau integration (Visa/Mastercard/Prepaid).

	Production connects to card bureau API (e.g. Sanlam, DPO, NIC).
	"""

	def __init__(self) -> None:
		self.bureau_url = os.environ.get("CARD_BUREAU_URL", "")
		self.api_key = os.environ.get("CARD_BUREAU_API_KEY", "")
		self._enabled = bool(self.bureau_url and self.api_key)

	async def request_card(
		self,
		card_type: str,
		card_name: str,
		account_number: str,
		member_id: str,
	) -> dict:
		"""Submit card issuance request to bureau. Returns tracking reference."""
		if not self._enabled:
			return {
				"bureau_reference": f"MOCK-CARD-{member_id[:8].upper()}",
				"status": "submitted",
				"estimated_delivery_days": 7,
			}
		raise NotImplementedError("Wire up card bureau API")

	async def get_card_status(self, bureau_reference: str) -> dict:
		"""Check card production status."""
		return {"bureau_reference": bureau_reference, "status": "in_production"}


# Module singletons (lazy-initialized)
_mpesa: MpesaAdapter | None = None
_card_bureau: CardIssuanceAdapter | None = None


def get_mpesa_adapter() -> MpesaAdapter:
	global _mpesa
	if _mpesa is None:
		_mpesa = MpesaAdapter()
	return _mpesa


def get_card_adapter() -> CardIssuanceAdapter:
	global _card_bureau
	if _card_bureau is None:
		_card_bureau = CardIssuanceAdapter()
	return _card_bureau
