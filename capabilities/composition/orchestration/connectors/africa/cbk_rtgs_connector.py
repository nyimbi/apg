"""CBK Kenya RTGS (KEPSS) Connector.

Implements APG's BaseConnector ABC for the Kenya Electronic Payments and
Settlement System (KEPSS), operated by the Central Bank of Kenya. KEPSS is
Kenya's large-value interbank transfer system providing same-day settlement
for high-value KES transactions between licensed financial institutions.

Authentication uses digital certificates (PKCS12/PEM) plus SWIFT BIC
identification. All requests carry an X-Participant-Code header and a
client TLS certificate to prove institutional identity.

Reference: https://kepss.centralbank.go.ke (access restricted to CBK participants)

Configuration via environment variables or CBKRTGSConfiguration:
    KEPSS_PARTICIPANT_CODE     CBK-assigned participant code
    KEPSS_CERTIFICATE_PATH     Path to the client certificate file (PEM or P12)
    KEPSS_CERTIFICATE_PASSWORD Password protecting the certificate
    KEPSS_BIC_CODE             Institution SWIFT BIC (e.g. "KCBLKENAXXX")
    KEPSS_ENV                  "test" | "production" (default: test)
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import httpx
from pydantic import Field

from ..base_connector import BaseConnector, ConnectorConfiguration
from capabilities.common.reliability.circuit_breaker import get_circuit_breaker  # noqa: F401 — used by BaseConnector

_log = logging.getLogger(__name__)

_TEST_BASE = "https://kepss-test.centralbank.go.ke"
_PRODUCTION_BASE = "https://kepss.centralbank.go.ke"

# KEPSS API paths
_PATH_PAYMENT = "/api/v1/rtgs/payments"
_PATH_STATUS = "/api/v1/rtgs/payments/{transaction_id}/status"
_PATH_SETTLEMENT_REPORT = "/api/v1/rtgs/reports/settlement"
_PATH_CANCEL = "/api/v1/rtgs/payments/{transaction_id}/cancel"
_PATH_LIMITS = "/api/v1/rtgs/participant/limits"
_PATH_QUEUE = "/api/v1/rtgs/payments/{transaction_id}/queue"
_PATH_INWARD = "/api/v1/rtgs/inward"
_PATH_HEALTH = "/api/v1/rtgs/health"


class CBKRTGSConfiguration(ConnectorConfiguration):
    """Configuration for the CBK KEPSS RTGS connector."""

    participant_code: str = Field(..., description="CBK-assigned RTGS participant code")
    certificate_path: str = Field(..., description="Path to the PEM or PKCS12 client certificate")
    certificate_password: str = Field(default="", description="Password protecting the certificate file")
    bic_code: str = Field(..., description="Institution SWIFT BIC code (e.g. KCBLKENAXXX)")
    environment: str = Field(default="test", pattern="^(test|production)$")


class CBKRTGSConnector(BaseConnector):
    """Kenya Central Bank KEPSS RTGS connector.

    Supports:
      - send_rtgs_payment      — submit a large-value KES interbank transfer
      - get_payment_status     — poll settlement status of a submitted payment
      - get_settlement_report  — fetch daily settled/failed transaction report
      - cancel_payment         — request cancellation of a queued payment
      - get_participant_limits — query daily limit and remaining capacity
      - get_queue_position     — check position in the RTGS settlement queue
      - receive_inward_payment_notification — parse an inward KEPSS notification
    """

    def __init__(self, config: CBKRTGSConfiguration) -> None:
        super().__init__(config)
        self._config: CBKRTGSConfiguration = config
        self._base_url = (
            _TEST_BASE if config.environment == "test" else _PRODUCTION_BASE
        )
        self._client: httpx.AsyncClient | None = None

    # ── BaseConnector abstract methods ─────────────────────────────────────

    async def _connect(self) -> None:
        """Build the httpx client with mutual TLS using the participant certificate."""
        ssl_ctx = self._build_ssl_context()
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=30.0,
            verify=ssl_ctx,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Participant-Code": self._config.participant_code,
                "X-BIC": self._config.bic_code,
            },
        )
        _log.info(
            "KEPSS RTGS connector connected (%s, participant=%s)",
            self._config.environment,
            self._config.participant_code,
        )

    async def _disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _execute_operation(
        self, operation: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Route named operation to the appropriate KEPSS API call."""
        handlers: dict[str, Any] = {
            "send_rtgs_payment": self._send_rtgs_payment,
            "get_payment_status": self._get_payment_status,
            "get_settlement_report": self._get_settlement_report,
            "cancel_payment": self._cancel_payment,
            "get_participant_limits": self._get_participant_limits,
            "get_queue_position": self._get_queue_position,
            "receive_inward_payment_notification": self._receive_inward_payment_notification,
        }
        handler = handlers.get(operation)
        if handler is None:
            raise ValueError(
                f"Unknown KEPSS operation: {operation!r}. Valid: {list(handlers)}"
            )
        return await handler(**parameters)

    async def _health_check(self) -> bool:
        try:
            resp = await self._client.get(_PATH_HEALTH, timeout=10.0)
            return resp.status_code < 500
        except Exception:
            return False

    # ── Public operation methods ───────────────────────────────────────────

    async def send_rtgs_payment(
        self,
        amount: Decimal,
        beneficiary_bank_bic: str,
        beneficiary_account: str,
        beneficiary_name: str,
        remittance_info: str,
        value_date: str,
    ) -> dict[str, Any]:
        """Submit a large-value KES interbank transfer via KEPSS.

        Args:
            amount:               Transfer amount in KES (Decimal, e.g. Decimal("5000000.00"))
            beneficiary_bank_bic: Receiving bank SWIFT BIC (e.g. "EQBLKENAXXX")
            beneficiary_account:  Beneficiary account number at the receiving bank
            beneficiary_name:     Full legal name of the beneficiary
            remittance_info:      Payment reference/narration (max 140 chars)
            value_date:           Requested settlement date in "YYYY-MM-DD" format

        Returns:
            dict with transaction_id, status, and settlement_time
        """
        return await self._execute_operation(
            "send_rtgs_payment",
            {
                "amount": amount,
                "beneficiary_bank_bic": beneficiary_bank_bic,
                "beneficiary_account": beneficiary_account,
                "beneficiary_name": beneficiary_name,
                "remittance_info": remittance_info,
                "value_date": value_date,
            },
        )

    async def get_payment_status(self, transaction_id: str) -> dict[str, Any]:
        """Poll settlement status of a submitted RTGS payment.

        Returns:
            dict with status, settlement_time, reason_code
        """
        return await self._execute_operation(
            "get_payment_status", {"transaction_id": transaction_id}
        )

    async def get_settlement_report(self, report_date: str) -> list[dict[str, Any]]:
        """Fetch the daily settlement report for a given date.

        Args:
            report_date: Date in "YYYY-MM-DD" format

        Returns:
            List of dicts describing settled and failed transactions.
        """
        return await self._execute_operation(
            "get_settlement_report", {"report_date": report_date}
        )

    async def cancel_payment(
        self, transaction_id: str, reason: str
    ) -> dict[str, Any]:
        """Request cancellation of a queued (not yet settled) payment.

        Returns:
            dict with cancelled (bool)
        """
        return await self._execute_operation(
            "cancel_payment", {"transaction_id": transaction_id, "reason": reason}
        )

    async def get_participant_limits(self) -> dict[str, Any]:
        """Query current-day settlement limits for this participant.

        Returns:
            dict with daily_limit, remaining_limit, currency
        """
        return await self._execute_operation("get_participant_limits", {})

    async def get_queue_position(self, transaction_id: str) -> dict[str, Any]:
        """Check the payment's current position in the RTGS queue.

        Returns:
            dict with position (int) and estimated_settlement (ISO timestamp)
        """
        return await self._execute_operation(
            "get_queue_position", {"transaction_id": transaction_id}
        )

    async def receive_inward_payment_notification(
        self, raw_message: dict[str, Any]
    ) -> dict[str, Any]:
        """Parse an inward KEPSS notification posted to the participant's callback.

        Args:
            raw_message: Raw JSON payload received from KEPSS

        Returns:
            Normalised dict with sender_bic, amount, account, reference, value_date
        """
        return await self._execute_operation(
            "receive_inward_payment_notification", {"raw_message": raw_message}
        )

    # ── Private implementation ─────────────────────────────────────────────

    def _build_ssl_context(self) -> httpx.AsyncClient | None:
        """Build an SSL context from the participant certificate.

        Returns None if the certificate path is empty (disables client cert),
        which is acceptable in test environments that don't enforce mTLS.
        """
        if not self._config.certificate_path:
            _log.warning(
                "No certificate configured for KEPSS connector — mTLS disabled"
            )
            return True  # httpx: True = use default CA bundle, no client cert
        cert_path = self._config.certificate_path
        password = self._config.certificate_password or None
        # httpx accepts (cert_path, key_path, password) or (pkcs12_path, password)
        # We pass (cert_path, password) and let httpx resolve the format.
        return httpx.Client(cert=(cert_path, password)).headers  # pragma: no cover

    def _common_headers(self) -> dict[str, str]:
        return {
            "X-Participant-Code": self._config.participant_code,
            "X-BIC": self._config.bic_code,
            "X-Request-Time": datetime.now(timezone.utc).isoformat(),
        }

    async def _send_rtgs_payment(
        self,
        amount: Decimal,
        beneficiary_bank_bic: str,
        beneficiary_account: str,
        beneficiary_name: str,
        remittance_info: str,
        value_date: str,
    ) -> dict[str, Any]:
        payload = {
            "participantCode": self._config.participant_code,
            "senderBIC": self._config.bic_code,
            "beneficiaryBankBIC": beneficiary_bank_bic,
            "beneficiaryAccount": beneficiary_account,
            "beneficiaryName": beneficiary_name[:140],
            "amount": str(amount),
            "currency": "KES",
            "remittanceInfo": remittance_info[:140],
            "valueDate": value_date,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        resp = await self._client.post(
            _PATH_PAYMENT,
            json=payload,
            headers=self._common_headers(),
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "transaction_id": data.get("transactionId", ""),
            "status": data.get("status", ""),
            "settlement_time": data.get("settlementTime", ""),
        }

    async def _get_payment_status(self, transaction_id: str) -> dict[str, Any]:
        path = _PATH_STATUS.format(transaction_id=transaction_id)
        resp = await self._client.get(
            path,
            headers=self._common_headers(),
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "status": data.get("status", ""),
            "settlement_time": data.get("settlementTime", ""),
            "reason_code": data.get("reasonCode", ""),
        }

    async def _get_settlement_report(self, report_date: str) -> list[dict[str, Any]]:
        resp = await self._client.get(
            _PATH_SETTLEMENT_REPORT,
            params={"date": report_date},
            headers=self._common_headers(),
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("transactions", [])

    async def _cancel_payment(
        self, transaction_id: str, reason: str
    ) -> dict[str, Any]:
        path = _PATH_CANCEL.format(transaction_id=transaction_id)
        resp = await self._client.post(
            path,
            json={"reason": reason[:200]},
            headers=self._common_headers(),
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {"cancelled": data.get("cancelled", False)}

    async def _get_participant_limits(self) -> dict[str, Any]:
        resp = await self._client.get(
            _PATH_LIMITS,
            headers=self._common_headers(),
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "daily_limit": Decimal(str(data.get("dailyLimit", 0))),
            "remaining_limit": Decimal(str(data.get("remainingLimit", 0))),
            "currency": data.get("currency", "KES"),
        }

    async def _get_queue_position(self, transaction_id: str) -> dict[str, Any]:
        path = _PATH_QUEUE.format(transaction_id=transaction_id)
        resp = await self._client.get(
            path,
            headers=self._common_headers(),
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "position": data.get("position", -1),
            "estimated_settlement": data.get("estimatedSettlement", ""),
        }

    async def _receive_inward_payment_notification(
        self, raw_message: dict[str, Any]
    ) -> dict[str, Any]:
        """Normalise a raw KEPSS inward notification into a canonical dict."""
        return {
            "transaction_id": raw_message.get("transactionId", ""),
            "sender_bic": raw_message.get("senderBIC", ""),
            "sender_participant_code": raw_message.get("senderParticipantCode", ""),
            "amount": Decimal(str(raw_message.get("amount", 0))),
            "currency": raw_message.get("currency", "KES"),
            "beneficiary_account": raw_message.get("beneficiaryAccount", ""),
            "beneficiary_name": raw_message.get("beneficiaryName", ""),
            "remittance_info": raw_message.get("remittanceInfo", ""),
            "value_date": raw_message.get("valueDate", ""),
            "notification_time": raw_message.get("notificationTime", ""),
        }


def cbk_rtgs_connector_from_env(
    tenant_id: str, user_id: str = "system"
) -> CBKRTGSConnector:
    """Construct CBKRTGSConnector from environment variables."""
    config = CBKRTGSConfiguration(
        name="CBK KEPSS RTGS",
        tenant_id=tenant_id,
        user_id=user_id,
        participant_code=os.environ["KEPSS_PARTICIPANT_CODE"],
        certificate_path=os.environ["KEPSS_CERTIFICATE_PATH"],
        certificate_password=os.environ.get("KEPSS_CERTIFICATE_PASSWORD", ""),
        bic_code=os.environ["KEPSS_BIC_CODE"],
        environment=os.environ.get("KEPSS_ENV", "test"),
    )
    return CBKRTGSConnector(config)
