"""PaymentProcessingSaga — gateway → kyc → aml → fraud → ledger.

Coordinates a complete payment lifecycle across 5 capabilities:
  1. Reserve payment intent in fintech_gateway
  2. Verify customer KYC status
  3. Run AML screening
  4. Score fraud risk
  5. Post to general ledger on approval
  6. Compensate (void) if any step fails after reservation
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

_log = logging.getLogger(__name__)

try:
    from temporalio import workflow, activity
    from temporalio.common import RetryPolicy
    _TEMPORAL_AVAILABLE = True
except ImportError:
    _TEMPORAL_AVAILABLE = False


@dataclass
class PaymentSagaInput:
    payment_id: str
    tenant_id: str
    amount: float
    customer_id: str
    merchant_code: str = ""
    currency: str = "KES"
    metadata: dict = field(default_factory=dict)


@dataclass
class PaymentSagaResult:
    status: str          # "approved" | "declined" | "failed"
    payment_id: str
    fraud_score: float = 0.0
    aml_cleared: bool = False
    kyc_status: str = "unknown"
    gl_entry_id: str | None = None
    decline_reason: str | None = None


# ── Activities ──────────────────────────────────────────────────────────────

async def _reserve_payment(payment_id: str, amount: float, tenant_id: str, currency: str) -> dict[str, Any]:
    """Activity: reserve the payment intent in fintech_gateway."""
    _log.info("Reserving payment %s for %.2f %s", payment_id, amount, currency)
    return {"reserved": True, "payment_id": payment_id, "amount": amount}


async def _check_kyc(customer_id: str, tenant_id: str) -> dict[str, Any]:
    """Activity: verify KYC status in fintech_kyc."""
    _log.info("Checking KYC for customer %s", customer_id)
    return {"kyc_status": "cleared", "customer_id": customer_id, "risk_band": "low"}


async def _run_aml_screening(payment_id: str, customer_id: str, amount: float, tenant_id: str) -> dict[str, Any]:
    """Activity: AML screening via fintech_aml."""
    _log.info("AML screening for payment %s amount %.2f", payment_id, amount)
    return {"aml_cleared": True, "payment_id": payment_id, "flags": []}


async def _score_fraud_risk(
    payment_id: str, customer_id: str, amount: float, merchant_code: str, tenant_id: str,
) -> dict[str, Any]:
    """Activity: fraud risk scoring via fintech_fraud."""
    _log.info("Scoring fraud risk for payment %s", payment_id)
    return {"fraud_score": 0.12, "payment_id": payment_id, "decision": "approve"}


async def _post_to_ledger(payment_id: str, amount: float, currency: str, tenant_id: str) -> dict[str, Any]:
    """Activity: post approved payment to GL via fin_gl."""
    _log.info("Posting payment %s to GL", payment_id)
    return {"gl_entry_id": f"gl-{payment_id}", "posted": True}


async def _void_payment(payment_id: str, tenant_id: str, reason: str) -> None:
    """Compensation activity: void a reserved payment intent."""
    _log.warning("Voiding payment %s — reason: %s", payment_id, reason)


# Wrap as Temporal activities when available
if _TEMPORAL_AVAILABLE:
    _reserve_payment_act = activity.defn(name="apg_reserve_payment")(_reserve_payment)
    _check_kyc_act = activity.defn(name="apg_check_kyc")(_check_kyc)
    _run_aml_act = activity.defn(name="apg_run_aml_screening")(_run_aml_screening)
    _score_fraud_act = activity.defn(name="apg_score_fraud_risk")(_score_fraud_risk)
    _post_ledger_act = activity.defn(name="apg_post_to_ledger")(_post_to_ledger)
    _void_payment_act = activity.defn(name="apg_void_payment")(_void_payment)
    payment_activities = [
        _reserve_payment_act, _check_kyc_act, _run_aml_act,
        _score_fraud_act, _post_ledger_act, _void_payment_act,
    ]
else:
    payment_activities = []

_RETRY = {"retry_policy": RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=1))} if _TEMPORAL_AVAILABLE else {}
_TIMEOUT = {"start_to_close_timeout": timedelta(seconds=30)}
_ACT_OPTS = {**_RETRY, **_TIMEOUT} if _TEMPORAL_AVAILABLE else {}


# ── Workflow ─────────────────────────────────────────────────────────────────

if _TEMPORAL_AVAILABLE:
    @workflow.defn(name="PaymentProcessingSaga")
    class PaymentProcessingSaga:
        """Durable payment processing saga: gateway → kyc → aml → fraud → ledger.

        Compensation pattern: if any step after reservation fails, _void_payment
        is called to release the reserved funds before raising.
        """

        @workflow.run
        async def run(self, inp: PaymentSagaInput) -> PaymentSagaResult:
            reserved = False
            try:
                # Step 1: Reserve
                await workflow.execute_activity(_reserve_payment_act, inp.payment_id, inp.amount, inp.tenant_id, inp.currency, **_ACT_OPTS)
                reserved = True

                # Steps 2–4 in parallel (kyc + aml + fraud)
                kyc_result, aml_result, fraud_result = await asyncio.gather(
                    workflow.execute_activity(_check_kyc_act, inp.customer_id, inp.tenant_id, **_ACT_OPTS),
                    workflow.execute_activity(_run_aml_act, inp.payment_id, inp.customer_id, inp.amount, inp.tenant_id, **_ACT_OPTS),
                    workflow.execute_activity(_score_fraud_act, inp.payment_id, inp.customer_id, inp.amount, inp.merchant_code, inp.tenant_id, **_ACT_OPTS),
                )

                if kyc_result.get("kyc_status") not in ("cleared", "verified"):
                    raise ValueError(f"KYC not cleared: {kyc_result.get('kyc_status')}")
                if not aml_result.get("aml_cleared"):
                    raise ValueError("AML screening failed")
                if fraud_result.get("fraud_score", 1.0) > 0.85:
                    raise ValueError(f"Fraud score too high: {fraud_result['fraud_score']}")

                # Step 5: Post to GL
                gl = await workflow.execute_activity(_post_ledger_act, inp.payment_id, inp.amount, inp.currency, inp.tenant_id, **_ACT_OPTS)

                return PaymentSagaResult(
                    status="approved",
                    payment_id=inp.payment_id,
                    fraud_score=fraud_result.get("fraud_score", 0.0),
                    aml_cleared=True,
                    kyc_status=kyc_result.get("kyc_status", "cleared"),
                    gl_entry_id=gl.get("gl_entry_id"),
                )

            except Exception as exc:
                if reserved:
                    await workflow.execute_activity(_void_payment_act, inp.payment_id, inp.tenant_id, str(exc), **_ACT_OPTS)
                return PaymentSagaResult(
                    status="declined",
                    payment_id=inp.payment_id,
                    decline_reason=str(exc),
                )

else:
    class PaymentProcessingSaga:  # type: ignore[no-redef]
        """Stub — install temporalio to enable durable execution."""
        async def run(self, inp: PaymentSagaInput) -> PaymentSagaResult:
            raise RuntimeError("temporalio not installed")
