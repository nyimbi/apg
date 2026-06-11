"""NIBSS Nigeria Interbank Settlement System Connector.

Implements APG's BaseConnector ABC for NIBSS — the Nigeria Interbank
Settlement System. Covers two clearing rails:

  NIP  (NIBSS Instant Payment)  — real-time 24/7 instant transfers, settled
       individually within seconds, max ₦50 million per transaction.

  NEFT (NIBSS Electronic Funds Transfer) — batch transfers, settled in
       three daily windows (8am, 12pm, 4pm WAT).

Authentication: OAuth2 client_credentials token + HMAC-SHA256 request
signing. Every mutating request includes an X-NIBSS-Signature header
computed over (timestamp + request body) with the institution's HMAC key.

Reference: https://nibss-plc.com.ng/developer (restricted to licensed FIs)

Configuration via environment variables or NIBSSConfiguration:
    NIBSS_INSTITUTION_CODE   CBN-assigned 3-digit institution code
    NIBSS_CLIENT_ID          OAuth2 client ID
    NIBSS_CLIENT_SECRET      OAuth2 client secret
    NIBSS_HMAC_KEY           HMAC-SHA256 signing key (hex)
    NIBSS_ENV                "test" | "production" (default: test)
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import httpx
from pydantic import Field

from ..base_connector import BaseConnector, ConnectorConfiguration
from capabilities.common.reliability.circuit_breaker import get_circuit_breaker  # noqa: F401 — used by BaseConnector

_log = logging.getLogger(__name__)

_TEST_BASE = "https://testapi.nibss-plc.com.ng"
_PRODUCTION_BASE = "https://api.nibss-plc.com.ng"

# NIBSS API paths
_PATH_TOKEN = "/auth/token"
_PATH_NIP_TRANSFER = "/nip/transfer"
_PATH_NIP_NAME_ENQUIRY = "/nip/nameenquiry"
_PATH_NIP_STATUS = "/nip/transfer/{session_id}/status"
_PATH_NEFT_TRANSFER = "/neft/transfer"
_PATH_NEFT_BATCH_STATUS = "/neft/batch/{batch_id}/status"
_PATH_BANK_CODES = "/nip/bankcodes"
_PATH_TRANSACTION_HISTORY = "/transactions/history"
_PATH_VALIDATE_ACCOUNT = "/nip/validate"
_PATH_HEALTH = "/health"


class NIBSSConfiguration(ConnectorConfiguration):
    """Configuration for the NIBSS NIP + NEFT connector."""

    institution_code: str = Field(..., description="CBN 3-digit institution code")
    client_id: str = Field(..., description="OAuth2 client ID issued by NIBSS")
    client_secret: str = Field(..., description="OAuth2 client secret issued by NIBSS")
    hmac_key: str = Field(..., description="Hex-encoded HMAC-SHA256 signing key")
    environment: str = Field(default="test", pattern="^(test|production)$")


class NIBSSConnector(BaseConnector):
    """NIBSS Nigeria Interbank Settlement connector (NIP + NEFT).

    Supports:
      - nip_transfer           — real-time interbank transfer via NIP
      - nip_name_enquiry       — resolve account name before transfer
      - get_nip_status         — check NIP session status
      - neft_transfer          — batch interbank transfer via NEFT
      - get_neft_batch_status  — check NEFT batch processing status
      - get_bank_codes         — retrieve list of NIP-enabled banks
      - get_transaction_history — paginated transaction history
      - validate_account       — validate account number at a bank
    """

    def __init__(self, config: NIBSSConfiguration) -> None:
        super().__init__(config)
        self._config: NIBSSConfiguration = config
        self._base_url = (
            _TEST_BASE if config.environment == "test" else _PRODUCTION_BASE
        )
        self._token: str = ""
        self._token_expires_at: float = 0.0
        self._client: httpx.AsyncClient | None = None

    # ── BaseConnector abstract methods ─────────────────────────────────────

    async def _connect(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=30.0,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Institution-Code": self._config.institution_code,
            },
        )
        await self._refresh_token()
        _log.info(
            "NIBSS connector connected (%s, institution=%s)",
            self._config.environment,
            self._config.institution_code,
        )

    async def _disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        self._token = ""
        self._token_expires_at = 0.0

    async def _execute_operation(
        self, operation: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Route named operation to the appropriate NIBSS API call."""
        handlers: dict[str, Any] = {
            "nip_transfer": self._nip_transfer,
            "nip_name_enquiry": self._nip_name_enquiry,
            "get_nip_status": self._get_nip_status,
            "neft_transfer": self._neft_transfer,
            "get_neft_batch_status": self._get_neft_batch_status,
            "get_bank_codes": self._get_bank_codes,
            "get_transaction_history": self._get_transaction_history,
            "validate_account": self._validate_account,
        }
        handler = handlers.get(operation)
        if handler is None:
            raise ValueError(
                f"Unknown NIBSS operation: {operation!r}. Valid: {list(handlers)}"
            )
        return await handler(**parameters)

    async def _health_check(self) -> bool:
        try:
            await self._refresh_token()
            return bool(self._token)
        except Exception:
            return False

    # ── Public operation methods ───────────────────────────────────────────

    async def nip_transfer(
        self,
        amount: Decimal,
        beneficiary_account_number: str,
        beneficiary_bank_code: str,
        beneficiary_name: str,
        narration: str,
        transaction_reference: str,
    ) -> dict[str, Any]:
        """Initiate a real-time NIP interbank transfer.

        Args:
            amount:                    Transfer amount in NGN
            beneficiary_account_number: 10-digit NUBAN account number
            beneficiary_bank_code:     3-digit CBN bank code of receiving bank
            beneficiary_name:          Validated beneficiary name (from name_enquiry)
            narration:                 Payment narration (max 100 chars)
            transaction_reference:     Unique reference assigned by originating institution

        Returns:
            dict with session_id, status, message
        """
        return await self._execute_operation(
            "nip_transfer",
            {
                "amount": amount,
                "beneficiary_account_number": beneficiary_account_number,
                "beneficiary_bank_code": beneficiary_bank_code,
                "beneficiary_name": beneficiary_name,
                "narration": narration,
                "transaction_reference": transaction_reference,
            },
        )

    async def nip_name_enquiry(
        self, beneficiary_account_number: str, bank_code: str
    ) -> dict[str, Any]:
        """Resolve account name and KYC tier before initiating a NIP transfer.

        Returns:
            dict with account_name, kyc_level, bank_name
        """
        return await self._execute_operation(
            "nip_name_enquiry",
            {
                "beneficiary_account_number": beneficiary_account_number,
                "bank_code": bank_code,
            },
        )

    async def get_nip_status(self, session_id: str) -> dict[str, Any]:
        """Check the final status of a NIP session.

        Returns:
            dict with status, message, completed_at
        """
        return await self._execute_operation(
            "get_nip_status", {"session_id": session_id}
        )

    async def neft_transfer(
        self,
        amount: Decimal,
        beneficiary_account: str,
        beneficiary_bank_code: str,
        narration: str,
        reference: str,
    ) -> dict[str, Any]:
        """Submit a batch NEFT interbank transfer.

        Args:
            amount:                Transfer amount in NGN
            beneficiary_account:   10-digit NUBAN account number
            beneficiary_bank_code: 3-digit CBN bank code
            narration:             Payment narration (max 100 chars)
            reference:             Unique reference for this credit entry

        Returns:
            dict with batch_id, status
        """
        return await self._execute_operation(
            "neft_transfer",
            {
                "amount": amount,
                "beneficiary_account": beneficiary_account,
                "beneficiary_bank_code": beneficiary_bank_code,
                "narration": narration,
                "reference": reference,
            },
        )

    async def get_neft_batch_status(self, batch_id: str) -> dict[str, Any]:
        """Check the processing status of a NEFT batch.

        Returns:
            dict with status, count (items), amount (total)
        """
        return await self._execute_operation(
            "get_neft_batch_status", {"batch_id": batch_id}
        )

    async def get_bank_codes(self) -> list[dict[str, Any]]:
        """Retrieve the current list of NIP-enabled banks.

        Returns:
            List of dicts with bank_code, bank_name, nip_enabled
        """
        return await self._execute_operation("get_bank_codes", {})

    async def get_transaction_history(
        self,
        from_date: str,
        to_date: str,
        page: int = 1,
    ) -> dict[str, Any]:
        """Retrieve paginated transaction history.

        Args:
            from_date: Start date "YYYY-MM-DD"
            to_date:   End date "YYYY-MM-DD"
            page:      Page number (1-based)

        Returns:
            Paginated response with items, total, page, page_size
        """
        return await self._execute_operation(
            "get_transaction_history",
            {"from_date": from_date, "to_date": to_date, "page": page},
        )

    async def validate_account(
        self, account_number: str, bank_code: str
    ) -> dict[str, Any]:
        """Validate that an account number exists at a given bank.

        Returns:
            dict with valid (bool), account_name
        """
        return await self._execute_operation(
            "validate_account",
            {"account_number": account_number, "bank_code": bank_code},
        )

    # ── Private implementation ─────────────────────────────────────────────

    async def _refresh_token(self) -> None:
        """Fetch or reuse an OAuth2 client_credentials token."""
        if time.time() < self._token_expires_at - 60:
            return  # still valid (60s buffer)

        client = self._client or httpx.AsyncClient(
            base_url=self._base_url, timeout=10.0
        )
        resp = await client.post(
            _PATH_TOKEN,
            data={
                "grant_type": "client_credentials",
                "client_id": self._config.client_id,
                "client_secret": self._config.client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        expires_in = int(data.get("expires_in", 3600))
        self._token_expires_at = time.time() + expires_in
        _log.debug("NIBSS OAuth token refreshed (expires_in=%ss)", expires_in)

    def _auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def _sign_request(self, body_bytes: bytes) -> dict[str, str]:
        """Compute HMAC-SHA256 signature over timestamp+body."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        message = timestamp.encode() + body_bytes
        sig = hmac.new(
            bytes.fromhex(self._config.hmac_key),
            message,
            hashlib.sha256,
        ).hexdigest()
        return {
            "X-NIBSS-Timestamp": timestamp,
            "X-NIBSS-Signature": sig,
        }

    def _signed_headers(self, body_bytes: bytes) -> dict[str, str]:
        headers = self._auth_header()
        headers.update(self._sign_request(body_bytes))
        return headers

    async def _nip_transfer(
        self,
        amount: Decimal,
        beneficiary_account_number: str,
        beneficiary_bank_code: str,
        beneficiary_name: str,
        narration: str,
        transaction_reference: str,
    ) -> dict[str, Any]:
        await self._refresh_token()
        import json as _json

        payload = {
            "institutionCode": self._config.institution_code,
            "amount": str(amount),
            "currency": "NGN",
            "beneficiaryAccountNumber": beneficiary_account_number,
            "beneficiaryBankCode": beneficiary_bank_code,
            "beneficiaryName": beneficiary_name[:100],
            "narration": narration[:100],
            "transactionReference": transaction_reference,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        body_bytes = _json.dumps(payload).encode()
        resp = await self._client.post(
            _PATH_NIP_TRANSFER,
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                **self._signed_headers(body_bytes),
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "session_id": data.get("sessionId", ""),
            "status": data.get("status", ""),
            "message": data.get("message", ""),
        }

    async def _nip_name_enquiry(
        self, beneficiary_account_number: str, bank_code: str
    ) -> dict[str, Any]:
        await self._refresh_token()
        import json as _json

        payload = {
            "accountNumber": beneficiary_account_number,
            "bankCode": bank_code,
            "institutionCode": self._config.institution_code,
        }
        body_bytes = _json.dumps(payload).encode()
        resp = await self._client.post(
            _PATH_NIP_NAME_ENQUIRY,
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                **self._signed_headers(body_bytes),
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "account_name": data.get("accountName", ""),
            "kyc_level": data.get("kycLevel", ""),
            "bank_name": data.get("bankName", ""),
        }

    async def _get_nip_status(self, session_id: str) -> dict[str, Any]:
        await self._refresh_token()
        path = _PATH_NIP_STATUS.format(session_id=session_id)
        resp = await self._client.get(
            path,
            headers=self._auth_header(),
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "status": data.get("status", ""),
            "message": data.get("message", ""),
            "completed_at": data.get("completedAt", ""),
        }

    async def _neft_transfer(
        self,
        amount: Decimal,
        beneficiary_account: str,
        beneficiary_bank_code: str,
        narration: str,
        reference: str,
    ) -> dict[str, Any]:
        await self._refresh_token()
        import json as _json

        payload = {
            "institutionCode": self._config.institution_code,
            "amount": str(amount),
            "currency": "NGN",
            "beneficiaryAccount": beneficiary_account,
            "beneficiaryBankCode": beneficiary_bank_code,
            "narration": narration[:100],
            "reference": reference,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        body_bytes = _json.dumps(payload).encode()
        resp = await self._client.post(
            _PATH_NEFT_TRANSFER,
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                **self._signed_headers(body_bytes),
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "batch_id": data.get("batchId", ""),
            "status": data.get("status", ""),
        }

    async def _get_neft_batch_status(self, batch_id: str) -> dict[str, Any]:
        await self._refresh_token()
        path = _PATH_NEFT_BATCH_STATUS.format(batch_id=batch_id)
        resp = await self._client.get(
            path,
            headers=self._auth_header(),
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "status": data.get("status", ""),
            "count": data.get("count", 0),
            "amount": Decimal(str(data.get("amount", 0))),
        }

    async def _get_bank_codes(self) -> list[dict[str, Any]]:
        await self._refresh_token()
        resp = await self._client.get(
            _PATH_BANK_CODES,
            headers=self._auth_header(),
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "bank_code": b.get("bankCode", ""),
                "bank_name": b.get("bankName", ""),
                "nip_enabled": b.get("nipEnabled", False),
            }
            for b in data.get("banks", [])
        ]

    async def _get_transaction_history(
        self, from_date: str, to_date: str, page: int
    ) -> dict[str, Any]:
        await self._refresh_token()
        resp = await self._client.get(
            _PATH_TRANSACTION_HISTORY,
            params={"fromDate": from_date, "toDate": to_date, "page": page, "pageSize": 50},
            headers=self._auth_header(),
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()

    async def _validate_account(
        self, account_number: str, bank_code: str
    ) -> dict[str, Any]:
        await self._refresh_token()
        import json as _json

        payload = {
            "accountNumber": account_number,
            "bankCode": bank_code,
            "institutionCode": self._config.institution_code,
        }
        body_bytes = _json.dumps(payload).encode()
        resp = await self._client.post(
            _PATH_VALIDATE_ACCOUNT,
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                **self._signed_headers(body_bytes),
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "valid": data.get("valid", False),
            "account_name": data.get("accountName", ""),
        }


def nibss_connector_from_env(
    tenant_id: str, user_id: str = "system"
) -> NIBSSConnector:
    """Construct NIBSSConnector from environment variables."""
    config = NIBSSConfiguration(
        name="NIBSS Nigeria",
        tenant_id=tenant_id,
        user_id=user_id,
        institution_code=os.environ["NIBSS_INSTITUTION_CODE"],
        client_id=os.environ["NIBSS_CLIENT_ID"],
        client_secret=os.environ["NIBSS_CLIENT_SECRET"],
        hmac_key=os.environ["NIBSS_HMAC_KEY"],
        environment=os.environ.get("NIBSS_ENV", "test"),
    )
    return NIBSSConnector(config)
