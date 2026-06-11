"""BCEAO STAR-UEMOA West Africa Interbank Connector.

Implements APG's BaseConnector ABC for the STAR-UEMOA system operated by
the Banque Centrale des États de l'Afrique de l'Ouest (BCEAO). STAR-UEMOA
is the real-time gross settlement and batch interbank transfer system for
the eight-member West African Economic and Monetary Union (WAEMU/UEMOA):

    Côte d'Ivoire · Sénégal · Mali · Burkina Faso
    Bénin · Niger · Togo · Guinée-Bissau

Currency: XOF (CFA Franc BCEAO) — pegged 1:655.957 to EUR.

Authentication: institutional certificate (presented as HTTP header
X-Institution-Cert-Serial) plus a static API key issued by BCEAO.
All requests carry X-Participant-Code (BIC) and X-API-Key headers.

Reference: https://star-uemoa.bceao.int (restricted to BCEAO participants)

Configuration via environment variables or BCEAOConfiguration:
    BCEAO_PARTICIPANT_CODE    Institution BIC / participant code
    BCEAO_API_KEY             BCEAO-issued API key
    BCEAO_INSTITUTION_BIC     SWIFT BIC of the institution
    BCEAO_COUNTRY_CODE        ISO-3166 alpha-2 country code (e.g. "CI", "SN")
    BCEAO_ENV                 "test" | "production" (default: test)
"""
from __future__ import annotations

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

_TEST_BASE = "https://star-uemoa-test.bceao.int"
_PRODUCTION_BASE = "https://star-uemoa.bceao.int"

# STAR-UEMOA API paths
_PATH_TRANSFER = "/api/v1/transfers"
_PATH_TRANSFER_STATUS = "/api/v1/transfers/{operation_id}/status"
_PATH_EXCHANGE_RATE = "/api/v1/exchange-rates"
_PATH_SETTLEMENT_REPORT = "/api/v1/reports/settlement"
_PATH_VALIDATE_RIB = "/api/v1/rib/validate"
_PATH_CORRESPONDENT_BANKS = "/api/v1/banks/correspondents"
_PATH_DAILY_LIMITS = "/api/v1/participant/limits"
_PATH_HEALTH = "/api/v1/health"

# UEMOA member country codes
UEMOA_COUNTRY_CODES = frozenset({"CI", "SN", "ML", "BF", "BJ", "NE", "TG", "GW"})


class BCEAOConfiguration(ConnectorConfiguration):
    """Configuration for the BCEAO STAR-UEMOA connector."""

    participant_code: str = Field(..., description="Institution BIC or BCEAO participant code")
    api_key: str = Field(..., description="BCEAO-issued API key for the institution")
    institution_bic: str = Field(..., description="SWIFT BIC of the institution (e.g. ECOICIAB)")
    country_code: str = Field(..., description="ISO-3166 alpha-2 country code within UEMOA")
    environment: str = Field(default="test", pattern="^(test|production)$")


class BCEAOConnector(BaseConnector):
    """BCEAO STAR-UEMOA West African interbank connector.

    Supports:
      - send_transfer                  — XOF interbank credit transfer
      - get_transfer_status            — check settlement status
      - get_exchange_rate              — cross-currency rate from BCEAO
      - get_settlement_report          — daily settled/failed transaction report
      - validate_rib                   — validate a BCEAO RIB number
      - get_correspondent_banks        — list correspondent banks by country
      - get_daily_limits               — query per-transaction and aggregate limits
    """

    def __init__(self, config: BCEAOConfiguration) -> None:
        super().__init__(config)
        self._config: BCEAOConfiguration = config
        self._base_url = (
            _TEST_BASE if config.environment == "test" else _PRODUCTION_BASE
        )
        self._client: httpx.AsyncClient | None = None

    # ── BaseConnector abstract methods ─────────────────────────────────────

    async def _connect(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=30.0,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-API-Key": self._config.api_key,
                "X-Participant-Code": self._config.participant_code,
                "X-Institution-BIC": self._config.institution_bic,
                "X-Country-Code": self._config.country_code,
            },
        )
        _log.info(
            "BCEAO STAR-UEMOA connector connected (%s, participant=%s, country=%s)",
            self._config.environment,
            self._config.participant_code,
            self._config.country_code,
        )

    async def _disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _execute_operation(
        self, operation: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Route named operation to the appropriate STAR-UEMOA API call."""
        handlers: dict[str, Any] = {
            "send_transfer": self._send_transfer,
            "get_transfer_status": self._get_transfer_status,
            "get_exchange_rate": self._get_exchange_rate,
            "get_settlement_report": self._get_settlement_report,
            "validate_rib": self._validate_rib,
            "get_correspondent_banks": self._get_correspondent_banks,
            "get_daily_limits": self._get_daily_limits,
        }
        handler = handlers.get(operation)
        if handler is None:
            raise ValueError(
                f"Unknown BCEAO operation: {operation!r}. Valid: {list(handlers)}"
            )
        return await handler(**parameters)

    async def _health_check(self) -> bool:
        try:
            resp = await self._client.get(_PATH_HEALTH, timeout=10.0)
            return resp.status_code < 500
        except Exception:
            return False

    # ── Public operation methods ───────────────────────────────────────────

    async def send_transfer(
        self,
        amount: Decimal,
        beneficiary_rib: str,
        beneficiary_bank_code: str,
        beneficiary_name: str,
        motif: str,
        reference: str,
    ) -> dict[str, Any]:
        """Submit an XOF interbank credit transfer via STAR-UEMOA.

        Args:
            amount:                Transfer amount in XOF (CFA Franc)
            beneficiary_rib:       Beneficiary RIB (Relevé d'Identité Bancaire)
            beneficiary_bank_code: BCEAO bank code of the receiving institution
            beneficiary_name:      Full name of the beneficiary
            motif:                 Payment motif/narration (max 140 chars)
            reference:             Unique reference assigned by the originating institution

        Returns:
            dict with operation_id, status
        """
        return await self._execute_operation(
            "send_transfer",
            {
                "amount": amount,
                "beneficiary_rib": beneficiary_rib,
                "beneficiary_bank_code": beneficiary_bank_code,
                "beneficiary_name": beneficiary_name,
                "motif": motif,
                "reference": reference,
            },
        )

    async def get_transfer_status(self, operation_id: str) -> dict[str, Any]:
        """Check settlement status of a STAR-UEMOA transfer.

        Returns:
            dict with status, execution_date, reason_code
        """
        return await self._execute_operation(
            "get_transfer_status", {"operation_id": operation_id}
        )

    async def get_exchange_rate(
        self, from_currency: str, to_currency: str
    ) -> dict[str, Any]:
        """Retrieve the current BCEAO exchange rate between two currencies.

        Args:
            from_currency: ISO-4217 source currency code (e.g. "XOF")
            to_currency:   ISO-4217 target currency code (e.g. "EUR", "USD", "GHS")

        Returns:
            dict with rate (Decimal), timestamp
        """
        return await self._execute_operation(
            "get_exchange_rate",
            {"from_currency": from_currency, "to_currency": to_currency},
        )

    async def get_settlement_report(self, report_date: str) -> dict[str, Any]:
        """Fetch the daily settlement report for a given date.

        Args:
            report_date: Date in "YYYY-MM-DD" format

        Returns:
            dict with settled (list), failed (list), total_amount (Decimal)
        """
        return await self._execute_operation(
            "get_settlement_report", {"report_date": report_date}
        )

    async def validate_rib(
        self, rib_number: str, country_code: str
    ) -> dict[str, Any]:
        """Validate a BCEAO RIB (Relevé d'Identité Bancaire) number.

        Args:
            rib_number:   RIB string (format varies by UEMOA country)
            country_code: ISO-3166 alpha-2 country code of the issuing bank

        Returns:
            dict with valid (bool), bank_name, branch
        """
        return await self._execute_operation(
            "validate_rib",
            {"rib_number": rib_number, "country_code": country_code},
        )

    async def get_correspondent_banks(
        self, country_code: str
    ) -> list[dict[str, Any]]:
        """List BCEAO correspondent banks for a UEMOA member country.

        Args:
            country_code: ISO-3166 alpha-2 country code (must be in UEMOA)

        Returns:
            List of dicts with bic, name, country
        """
        if country_code not in UEMOA_COUNTRY_CODES:
            raise ValueError(
                f"Country {country_code!r} is not a UEMOA member. "
                f"Valid codes: {sorted(UEMOA_COUNTRY_CODES)}"
            )
        return await self._execute_operation(
            "get_correspondent_banks", {"country_code": country_code}
        )

    async def get_daily_limits(self) -> dict[str, Any]:
        """Query the per-transaction and daily aggregate XOF limits.

        Returns:
            dict with single_transaction_limit (Decimal), daily_aggregate_limit (Decimal)
        """
        return await self._execute_operation("get_daily_limits", {})

    # ── Private implementation ─────────────────────────────────────────────

    def _request_headers(self) -> dict[str, str]:
        """Return base request headers including timestamp for traceability."""
        return {
            "X-Request-Time": datetime.now(timezone.utc).isoformat(),
            "X-API-Key": self._config.api_key,
            "X-Participant-Code": self._config.participant_code,
            "X-Institution-BIC": self._config.institution_bic,
            "X-Country-Code": self._config.country_code,
        }

    async def _send_transfer(
        self,
        amount: Decimal,
        beneficiary_rib: str,
        beneficiary_bank_code: str,
        beneficiary_name: str,
        motif: str,
        reference: str,
    ) -> dict[str, Any]:
        payload = {
            "participantCode": self._config.participant_code,
            "institutionBIC": self._config.institution_bic,
            "countryCode": self._config.country_code,
            "amount": str(amount),
            "currency": "XOF",
            "beneficiaryRIB": beneficiary_rib,
            "beneficiaryBankCode": beneficiary_bank_code,
            "beneficiaryName": beneficiary_name[:140],
            "motif": motif[:140],
            "reference": reference,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        resp = await self._client.post(
            _PATH_TRANSFER,
            json=payload,
            headers=self._request_headers(),
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "operation_id": data.get("operationId", ""),
            "status": data.get("status", ""),
        }

    async def _get_transfer_status(self, operation_id: str) -> dict[str, Any]:
        path = _PATH_TRANSFER_STATUS.format(operation_id=operation_id)
        resp = await self._client.get(
            path,
            headers=self._request_headers(),
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "status": data.get("status", ""),
            "execution_date": data.get("executionDate", ""),
            "reason_code": data.get("reasonCode", ""),
        }

    async def _get_exchange_rate(
        self, from_currency: str, to_currency: str
    ) -> dict[str, Any]:
        resp = await self._client.get(
            _PATH_EXCHANGE_RATE,
            params={"from": from_currency, "to": to_currency},
            headers=self._request_headers(),
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "rate": Decimal(str(data.get("rate", 0))),
            "timestamp": data.get("timestamp", ""),
        }

    async def _get_settlement_report(self, report_date: str) -> dict[str, Any]:
        resp = await self._client.get(
            _PATH_SETTLEMENT_REPORT,
            params={"date": report_date, "participantCode": self._config.participant_code},
            headers=self._request_headers(),
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "settled": data.get("settled", []),
            "failed": data.get("failed", []),
            "total_amount": Decimal(str(data.get("totalAmount", 0))),
        }

    async def _validate_rib(
        self, rib_number: str, country_code: str
    ) -> dict[str, Any]:
        resp = await self._client.post(
            _PATH_VALIDATE_RIB,
            json={"rib": rib_number, "countryCode": country_code},
            headers=self._request_headers(),
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "valid": data.get("valid", False),
            "bank_name": data.get("bankName", ""),
            "branch": data.get("branch", ""),
        }

    async def _get_correspondent_banks(
        self, country_code: str
    ) -> list[dict[str, Any]]:
        resp = await self._client.get(
            _PATH_CORRESPONDENT_BANKS,
            params={"countryCode": country_code},
            headers=self._request_headers(),
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "bic": b.get("bic", ""),
                "name": b.get("name", ""),
                "country": b.get("country", ""),
            }
            for b in data.get("banks", [])
        ]

    async def _get_daily_limits(self) -> dict[str, Any]:
        resp = await self._client.get(
            _PATH_DAILY_LIMITS,
            params={"participantCode": self._config.participant_code},
            headers=self._request_headers(),
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "single_transaction_limit": Decimal(
                str(data.get("singleTransactionLimit", 0))
            ),
            "daily_aggregate_limit": Decimal(
                str(data.get("dailyAggregateLimit", 0))
            ),
        }


def bceao_connector_from_env(
    tenant_id: str, user_id: str = "system"
) -> BCEAOConnector:
    """Construct BCEAOConnector from environment variables."""
    config = BCEAOConfiguration(
        name="BCEAO STAR-UEMOA",
        tenant_id=tenant_id,
        user_id=user_id,
        participant_code=os.environ["BCEAO_PARTICIPANT_CODE"],
        api_key=os.environ["BCEAO_API_KEY"],
        institution_bic=os.environ["BCEAO_INSTITUTION_BIC"],
        country_code=os.environ["BCEAO_COUNTRY_CODE"],
        environment=os.environ.get("BCEAO_ENV", "test"),
    )
    return BCEAOConnector(config)
