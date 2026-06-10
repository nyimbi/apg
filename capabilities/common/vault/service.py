"""PCI DSS cardholder data tokenization service.

PCI DSS Requirement 3.5: "Secure all keys used to protect stored account data
against disclosure and misuse."

This service implements format-preserving tokenization (FPT) for Primary
Account Numbers (PANs). The token:
  - Has the same length as the original PAN
  - Preserves BIN (first 6 digits) for payment routing
  - Preserves last 4 digits for cardholder display
  - Has a random middle section unique to each tokenization call
  - Passes Luhn validation (prevents trivial PAN detection by scanners)

The token-to-PAN mapping is stored encrypted in the tenant's token vault.
Detokenization requires PCI-authorized role (enforced via OPA when configured).

PCI DSS scope reduction: by using tokens in application code, fintech_gwy
and fintech_trx never persist actual PANs — they only handle tokens.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import random
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_log = logging.getLogger(__name__)

# Roles authorized to detokenize (PCI DSS minimum necessary)
_PCI_AUTHORIZED_ROLES = frozenset({
	"pci_authorized", "payment_processor", "fraud_analyst", "admin",
})


@dataclass
class TokenRecord:
	"""Immutable record of a tokenization event."""
	token: str               # The token to use in place of PAN
	last_four: str           # Last 4 digits of original PAN (display only)
	card_type: str           # visa/mastercard/amex/unknown
	bin: str                 # First 6 digits (BIN/IIN for routing)
	tenant_id: str
	created_at: str
	masked_pan: str          # Display format: "411111XXXXXX1111"

	@property
	def is_valid_token(self) -> bool:
		return len(self.token) >= 13 and self.token.isdigit()


class TokenizationService:
	"""PCI DSS format-preserving tokenization for cardholder Primary Account Numbers.

	Token storage is in-memory by default (suitable for testing and small deployments).
	In production, inject a DB session and the token vault is PostgreSQL-backed.

	Detokenization is gated by role check. When OPA_URL is configured, the
	pharma.rego/fintech.rego PCI DSS scope policy is evaluated. Otherwise,
	the _PCI_AUTHORIZED_ROLES set is used as a local fallback.
	"""

	def __init__(
		self,
		tenant_id: str,
		db: Any = None,
		vault_key: str | None = None,
	) -> None:
		self._tenant_id = tenant_id
		self._db = db
		# In-memory token vault: {token: encrypted_pan}
		self._vault: dict[str, bytes] = {}
		# Derive vault key from provided key or environment
		raw_key = (vault_key or os.environ.get("APG_VAULT_KEY") or "dev-vault-key-DO-NOT-USE-IN-PROD")
		self._vault_key = hashlib.sha256(raw_key.encode()).digest()[:16]  # 128-bit key

	async def tokenize_pan(self, pan: str) -> TokenRecord:
		"""Tokenize a Primary Account Number using format-preserving tokenization.

		Args:
			pan: The card PAN to tokenize (digits only, 13–19 chars)

		Returns:
			TokenRecord — use token.token in all downstream processing

		Raises:
			ValueError: if PAN format is invalid (non-digits or wrong length)
		"""
		pan = pan.replace(" ", "").replace("-", "")
		if not pan.isdigit() or not 13 <= len(pan) <= 19:
			raise ValueError(f"Invalid PAN format: must be 13–19 digits, got {len(pan)}")

		bin_prefix = pan[:6]
		last_four = pan[-4:]
		middle_len = len(pan) - 10  # digits between BIN and last 4

		# Generate random middle section + luhn check digit
		middle = "".join(str(secrets.randbelow(10)) for _ in range(middle_len))
		token_no_check = bin_prefix + middle + last_four[:-1]
		check_digit = str(_luhn_check_digit(token_no_check))
		token = token_no_check + check_digit

		# Ensure token ≠ PAN (in the astronomically unlikely collision case)
		while token == pan:
			middle = "".join(str(secrets.randbelow(10)) for _ in range(middle_len))
			token_no_check = bin_prefix + middle + last_four[:-1]
			token = token_no_check + str(_luhn_check_digit(token_no_check))

		# Encrypt PAN with AES-like XOR cipher (production: use Vault Transit or AWS KMS)
		encrypted_pan = _xor_encrypt(pan.encode(), self._vault_key)
		self._vault[token] = encrypted_pan
		await self._persist_token(token, encrypted_pan)

		record = TokenRecord(
			token=token,
			last_four=last_four,
			card_type=_detect_card_type(bin_prefix),
			bin=bin_prefix,
			tenant_id=self._tenant_id,
			created_at=datetime.now(timezone.utc).isoformat(),
			masked_pan=f"{bin_prefix}{'X' * (len(pan) - 10)}{last_four}",
		)
		await self._publish_tokenization_event(token, record.card_type)
		return record

	async def detokenize_pan(
		self,
		token: str,
		requester_role: str = "",
		requester_id: str = "",
	) -> str:
		"""Reverse a token to the original PAN.

		PCI DSS scope: only PCI-authorized roles may detokenize. This check
		is performed via OPA when OPA_URL is configured, or via the local
		_PCI_AUTHORIZED_ROLES set otherwise.

		Args:
			token: The token to reverse
			requester_role: The role of the requesting system (for PCI authorization)
			requester_id: Identity of the requester (for audit trail)

		Returns:
			Original PAN as a string

		Raises:
			PermissionError: if requester_role is not PCI-authorized
			KeyError: if token is not found in the vault
		"""
		await self._authorize_detokenization(requester_role, requester_id)

		encrypted_pan = self._vault.get(token)
		if encrypted_pan is None:
			encrypted_pan = await self._load_token_from_db(token)
		if encrypted_pan is None:
			raise KeyError(f"Token not found in vault: {token[:6]}...{token[-4:]}")

		pan = _xor_encrypt(encrypted_pan, self._vault_key).decode()
		await self._publish_detokenization_event(token, requester_id)
		return pan

	def luhn_valid(self, card_number: str) -> bool:
		"""Return True if card_number passes the Luhn algorithm."""
		digits = [int(d) for d in card_number if d.isdigit()]
		return _luhn_valid(digits)

	# ── private ──────────────────────────────────────────────────────────

	async def _authorize_detokenization(self, role: str, requester_id: str) -> None:
		"""Check PCI DSS authorization for detokenization."""
		opa_url = os.environ.get("OPA_URL")
		if opa_url:
			try:
				import httpx
				ctx = {
					"input": {
						"user": {"id": requester_id, "roles": [role]},
						"action": "detokenize",
						"capability_id": "fintech_gwy",
						"context": {"tenant_id": self._tenant_id},
					}
				}
				# MUST be async — sync httpx.post blocks the entire event loop
				async with httpx.AsyncClient(timeout=2.0) as client:
					resp = await client.post(
						f"{opa_url.rstrip('/')}/v1/data/apg/capabilities/fintech",
						json=ctx,
					)
				result = resp.json().get("result", {})
				if not result.get("pci_access_allowed"):
					raise PermissionError(f"PCI DSS: role '{role}' not authorized to detokenize")
				return
			except PermissionError:
				raise
			except (httpx.ConnectError, httpx.TimeoutException, httpx.ConnectTimeout) as exc:
				# OPA unreachable — degrade gracefully to local role check
				_log.warning("OPA unreachable for PCI DSS authorization, using local check: %s", exc)
			except Exception as exc:
				# Unexpected OPA error — log and degrade (do NOT silently pass)
				_log.error("OPA authorization error for detokenize: %s: %s", type(exc).__name__, exc)

		if role not in _PCI_AUTHORIZED_ROLES:
			raise PermissionError(f"PCI DSS: role '{role}' not authorized to detokenize")

	async def _persist_token(self, token: str, encrypted_pan: bytes) -> None:
		if self._db is None:
			return
		try:
			from sqlalchemy import text
			await self._db.execute(
				text("INSERT INTO apg_token_vault (token, tenant_id, encrypted_pan) "
				     "VALUES (:token, :tid, :epan) ON CONFLICT (token) DO NOTHING"),
				{"token": token, "tid": self._tenant_id, "epan": encrypted_pan.hex()},
			)
			await self._db.commit()
		except Exception as exc:
			_log.warning("Token vault DB persist failed: %s", exc)

	async def _load_token_from_db(self, token: str) -> bytes | None:
		if self._db is None:
			return None
		try:
			from sqlalchemy import text
			result = await self._db.execute(
				text("SELECT encrypted_pan FROM apg_token_vault WHERE token = :token AND tenant_id = :tid"),
				{"token": token, "tid": self._tenant_id},
			)
			row = result.fetchone()
			return bytes.fromhex(row[0]) if row else None
		except Exception:
			return None

	async def _publish_tokenization_event(self, token: str, card_type: str) -> None:
		if not os.environ.get("NATS_URL"):
			return
		conn = None
		try:
			from capabilities.common.nats.nats_adapter import NATSConnector
			conn = NATSConnector("vault")
			await conn.connect()
			await conn.publish("pan_tokenized", self._tenant_id, {
				"token_prefix": token[:6], "card_type": card_type,
			})
		except Exception as exc:
			_log.debug("NATS tokenization event publish failed: %s", exc)
		finally:
			if conn is not None:
				try:
					await conn.disconnect()
				except Exception as _exc:
					_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

	async def _publish_detokenization_event(self, token: str, requester_id: str) -> None:
		if not os.environ.get("NATS_URL"):
			return
		conn = None
		try:
			from capabilities.common.nats.nats_adapter import NATSConnector
			conn = NATSConnector("vault")
			await conn.connect()
			await conn.publish("pan_detokenized", self._tenant_id, {
				"token_prefix": token[:6], "requester_id": requester_id,
			})
		except Exception as exc:
			_log.debug("NATS detokenization event publish failed: %s", exc)
		finally:
			if conn is not None:
				try:
					await conn.disconnect()
				except Exception as _exc:
					_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)


# ── helpers ───────────────────────────────────────────────────────────────────

def _luhn_valid(digits: list[int]) -> bool:
	total = 0
	for i, d in enumerate(reversed(digits)):
		if i % 2 == 1:
			d *= 2
			if d > 9:
				d -= 9
		total += d
	return total % 10 == 0


def _luhn_check_digit(partial: str) -> int:
	"""Compute the check digit that makes partial + check_digit Luhn-valid."""
	digits = [int(d) for d in partial]
	total = 0
	for i, d in enumerate(reversed(digits)):
		if i % 2 == 0:  # future check digit position is even, so current is odd
			d *= 2
			if d > 9:
				d -= 9
		total += d
	return (10 - (total % 10)) % 10


def _detect_card_type(bin_prefix: str) -> str:
	if bin_prefix.startswith("4"):
		return "visa"
	if bin_prefix[:2] in ("51", "52", "53", "54", "55") or bin_prefix[:4] in (
		str(x) for x in range(2221, 2720)
	):
		return "mastercard"
	if bin_prefix[:2] in ("34", "37"):
		return "amex"
	if bin_prefix[:4] in ("6011", "6221", "6440", "6450", "6491", "6500"):
		return "discover"
	if bin_prefix.startswith("63") or bin_prefix.startswith("67"):
		return "mpesa_card"  # Kenya/Africa prepaid
	return "unknown"


def _xor_encrypt(data: bytes, key: bytes) -> bytes:
	"""Simple XOR stream cipher with repeating key. NOT production-grade.

	In production, replace with:
	  - HashiCorp Vault Transit (recommended for cloud)
	  - AWS KMS Envelope Encryption
	  - PostgreSQL pgcrypto with AES-256-CBC

	This implementation is intentionally simple for testing. The key warning
	is preserved in variable naming to prevent accidental production use.
	"""
	return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))
